"""Manage synchronous reads, hydration, and identity mapping."""

import logging
import os
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Self, TypeVar, cast, overload

from sgalchemy._core._compiler import FindCall, SummarizeCall, compile_statement
from sgalchemy._core._engine import Engine
from sgalchemy._core._engine_common import (
    AUTH_RECOVERY_NOTE,
    AUTH_UPLOAD_NOTE,
    CallProgress,
    add_auth_note,
    add_note_once,
)
from sgalchemy._core._options import LazyLoad, SelectinLoad, normalize_relationship_loaders
from sgalchemy._core._registry import (
    REGISTRY_SITE_MISMATCH_MESSAGE,
    Entity,
    EntityState,
    RegistryBinding,
    entity_state,
)
from sgalchemy._core._result import Result
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import Select
from sgalchemy._errors import CommitError, CompilationError, SchemaUnavailableError, SgalchemyError
from sgalchemy._orm._loaders import LazyContext, execute_selectins
from sgalchemy._orm._session_common import (
    annotate_read_error,
    projection_result,
    refresh_call,
    validate_query_payload,
)
from sgalchemy._orm._state import SessionState, validate_point_records
from sgalchemy._orm._uow import (
    SOLE_BATCH_STAGED_NOTE,
    CommitOutcome,
    CommitPlan,
    CommitStep,
    ConfirmedFieldRead,
    ConfirmedFieldReadGroup,
    RecoveryAdapter,
    UnitOfWork,
    apply_batch_indeterminate,
    apply_batch_results,
    apply_confirmed_field_readback,
    apply_revive_result,
    apply_upload_indeterminate,
    apply_upload_result,
    batch_failure_note,
    commit_steps,
    confirmed_field_read_call,
    confirmed_field_read_codes,
    discard_local_state,
    group_upload_field_reads,
    mark_indeterminate_entities,
    new_session_unit_of_work,
    plan_commit,
    preflight_uploads,
    schema_drift_error,
    serialize_chunk,
    stage_add,
    stage_delete,
    stage_revive,
    staged_intent_snapshot,
    validate_batch_size,
)
from sgalchemy._shotgun import QueryResult, WireRecord, WireRecords
from sgalchemy.schema._model import EntitySchema

_ENGINE_LOG = logging.getLogger('sgalchemy.engine')
_LOGGER = logging.getLogger('sgalchemy.session')
_EntityT = TypeVar('_EntityT', bound=Entity)
_T = TypeVar('_T')


class Session:
    """Manage entity identity and staged writes with explicit commits.

    Queries never autoflush. Ordinary field access performs no provider I/O
    unless a synchronous lazy loader was explicitly selected.

    Context management returns the session and closes local state on exit
    without committing or suppressing exceptions. Close restores local edits,
    discards staged work, and detaches owned entities without undoing provider
    writes or disposing the engine. The cleared session remains reusable.

    Only the methods listed in the session guide are supported. Other
    non-underscore methods, such as `attach_for_staging`, `hydrate_record`,
    `hydration_scope`, `register_identity`, and `unit_of_work`, are
    package-internal orchestration hooks and may change without notice.
    """

    def __init__(self, engine: Engine, *, max_implicit_queries: int | None = None) -> None:
        """Bind a new empty identity map to one engine.

        Args:
            engine: Engine bound to the session.
            max_implicit_queries: Maximum lazy-load queries, or `None` for no
                limit.
        """
        self._active_step: CommitStep | None = None
        self._engine = engine
        self._implicit_queries = 0
        self._max_implicit_queries = max_implicit_queries
        self._state = SessionState(
            self,
            engine.registry,
            engine.prepared_binding,
            engine.mark_schema_stale,
        )
        self._unit_of_work = new_session_unit_of_work()

    def __enter__(self) -> Self:
        """Return this session for context-manager use."""
        return self

    def _annotated(self, context: str, operation: Callable[[], _T]) -> _T:
        """Run provider work while safely preserving and annotating faults."""
        try:
            return operation()
        except Exception as err:
            annotate_read_error(err, context)
            raise

    def _call_commit_step(
        self,
        step: CommitStep,
        progress: CallProgress,
        method: str,
        *args: object,
    ) -> Any:  # noqa: ANN401 - Commit-step kind selects unrelated provider arguments and results.
        """Select the active step and observe its resolved provider entry."""
        self._active_step = step
        return self._engine.call_with_progress(method, args, {}, progress)

    def _execute_upload_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
        progress: CallProgress,
    ) -> ConfirmedFieldRead:
        """Execute and reconcile one staged image upload."""
        upload = step.require_upload()
        state = entity_state(upload.obj)
        if state.id is None:
            raise RuntimeError('upload entity has no server id')
        self._call_commit_step(
            step,
            progress,
            step.kind,
            state.entity_type,
            state.id,
            os.fsdecode(upload.path),
        )
        return apply_upload_result(upload, outcome)

    def _identity_values(self) -> list[Entity]:
        """Return a snapshot of identity-mapped objects."""
        return self._state.identity_values()

    def _require_owned_class(self, cls: type[Entity]) -> str:
        """Return a class type after enforcing this engine's registry ownership."""
        return self._state.require_owned_class(cls)

    def _require_owned_object(self, obj: Entity) -> EntityState:
        """Return state after enforcing registry and Session ownership."""
        return self._state.require_owned_object(obj)

    def _route_batch_failure(
        self,
        err: Exception,
        step: CommitStep,
        outcome: CommitOutcome,
        sole_batch: bool,
        *,
        invoked: bool,
    ) -> CommitError | None:
        """Return a wrapped fault, or `None` for raw sole-batch passthrough."""
        if sole_batch:
            add_note_once(err, SOLE_BATCH_STAGED_NOTE)
            if note := batch_failure_note(step, outcome, invoked=invoked):
                add_note_once(err, note)
            add_note_once(err, f'sgalchemy: failed during {step.description}')
            add_auth_note(err, err, AUTH_RECOVERY_NOTE)
            return None
        return _commit_error_with_batch_status(err, step, outcome, invoked=invoked)

    def add(self, obj: Entity) -> None:
        """Stage one pending entity for creation.

        Args:
            obj: Schema-bound pending entity from `Entity.new()`.

        Raises:
            SgalchemyError: If `obj` is foreign, persistent, indeterminate, or
                owned elsewhere.
        """
        stage_add(self, obj)

    def assert_identity_available(self, obj: Entity) -> None:
        """Reject another object mapped under the same persistent identity.

        Args:
            obj: Persistent object whose identity must remain unique.

        Raises:
            SgalchemyError: If a different object already owns the identity.
        """
        self._state.assert_identity_available(obj)

    def assert_same_registry(self, value: Entity | type[Entity]) -> None:
        """Reject a class or object owned by another registry or open session.

        Args:
            value: Entity class or instance to validate before local use.

        Raises:
            SgalchemyError: If `value` belongs to another registry or open
                session.
        """
        self._state.assert_same_registry(value)

    def attach_for_staging(self, obj: Entity) -> None:
        """Attach one detached entity for local staging.

        Args:
            obj: Entity to attach after registry validation.

        Raises:
            SgalchemyError: If another open session owns `obj`.
        """
        self._state.attach(obj)

    def close(self) -> None:
        """Revert local edits, discard staged work, and detach owned entities.

        Closing performs no provider I/O and cannot undo provider writes.
        """
        discard_local_state(
            RecoveryAdapter(
                unit_of_work=self._unit_of_work,
                detach_canceled=self._state.detach_canceled,
                discard_identities=self._state.discard_identities,
                identity_values=self._state.identity_values,
                owns_entity=self._state.owns_entity,
            )
        )
        self._state.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close local state without suppressing exceptions."""
        self.close()

    def commit(self, batch_size: int | None = None) -> None:
        """Execute staged work with bounded pre-write and readback recovery.

        A commit is atomic only when its complete write plan uses one provider
        `batch()` call. Planning may prepare schema or force one pre-write
        refresh. Confirmed-write and thumbnail reconciliation may issue bounded
        read-only calls.

        Uploads that fail before resolved provider entry retain their staged
        paths and trust state. Only invoked uploads need ambiguous-write
        retirement and recovery; earlier confirmed uploads still read back.

        Args:
            batch_size: Optional positive maximum requests per batch call. A
                supplied limit must fit every indivisible operation group. An
                ordinary same-owner update/delete group requires at least two
                requests.

        Raises:
            CommitError: If execution has multi-step outcome evidence or a
                successful batch response cannot be reconciled safely. Its four
                partitions distinguish committed, staged, indeterminate, and
                withdrawn plan items.
            RetiredEntityWriteError: If a retired entity has staged writes
                without a matching revival.
            SgalchemyError: If local staging, ownership, or schema validation
                prevents a write plan.
            Exception: Original provider or schema-preparation exception; a
                sole-batch provider failure remains unwrapped.
            ValueError: If `batch_size` is not a positive integer or `None`, or
                is smaller than an indivisible operation group.
        """
        commit_session(self, batch_size, wrap_ordinary_failure=False)

    def delete(self, obj: Entity) -> None:
        """Stage retirement or cancel an inverse local transition.

        Args:
            obj: Pending or persistent entity to transition.

        Raises:
            SgalchemyError: If `obj` is foreign, unstaged, indeterminate, or
                owned elsewhere.
        """
        stage_delete(self, obj)

    def detach_canceled(self, obj: Entity) -> None:
        """Detach a canceled object absent from this identity map.

        Args:
            obj: Entity removed from local staging.
        """
        self._state.detach_canceled(obj)

    @property
    def engine(self) -> Engine:
        """Return the engine bound to this session.

        Returns:
            Bound engine.
        """
        return self._engine

    @overload
    def execute_call(
        self,
        call: FindCall,
        binding: RegistryBinding | None,
    ) -> WireRecords: ...

    @overload
    def execute_call(
        self,
        call: SummarizeCall,
        binding: RegistryBinding | None,
    ) -> WireRecord: ...

    @overload
    def execute_call(
        self,
        call: FindCall | SummarizeCall,
        binding: RegistryBinding | None,
    ) -> QueryResult: ...

    def execute_call(
        self,
        call: FindCall | SummarizeCall,
        binding: RegistryBinding | None,
    ) -> QueryResult:
        """Execute compiled provider work with Session fault context.

        Args:
            call: Compiled find or summarize call.
            binding: Captured operation generation associated with the call and
                later hydration.

        Returns:
            Provider payload in its documented outer shape.

        Raises:
            SgalchemyError: If the outer result shape contradicts the call.
            Exception: Original provider exception, annotated and unwrapped.
        """
        del binding
        context = f'raised while executing {call}'
        payload: QueryResult = self._annotated(context, lambda: self._engine.execute_call(call))
        return validate_query_payload(call, payload)

    def _read_confirmed_field_group(self, group: ConfirmedFieldReadGroup) -> None:
        """Execute and apply one bounded current-value field read."""
        call = confirmed_field_read_call(group)
        if call is None:
            return
        records = self.execute_call(call, group.binding)
        apply_confirmed_field_readback(self, group, records)

    def _recover_confirmed_fields(
        self,
        step: CommitStep,
        groups: tuple[ConfirmedFieldReadGroup, ...],
    ) -> None:
        """Read trustworthy fields omitted from one confirmed batch step."""
        for group in groups:
            field_codes, missing_field_codes = confirmed_field_read_codes(group)
            _LOGGER.warning(
                'provider-response contract violation during %s: '
                'entity_type=%s retired_only=%s missing_fields=%s',
                step.description,
                group.entity_type,
                group.retired_only,
                missing_field_codes,
            )
            if field_codes:
                self._read_confirmed_field_group(group)

    def _execute_batch_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
        sole_batch: bool,
        progress: CallProgress,
    ) -> None:
        """Execute and reconcile one transactional provider batch."""
        try:
            requests = serialize_chunk(step.operations)
            raw_results = self._call_commit_step(step, progress, 'batch', requests)
        except CommitError:
            raise
        except Exception as err:
            routed = self._route_batch_failure(
                err, step, outcome, sole_batch, invoked=progress.invoked.is_set()
            )
            if routed is None:
                raise
            raise routed from err
        try:
            groups = apply_batch_results(
                self,
                step.operations,
                raw_results,
                outcome,
                self._state.identity_map.__setitem__,
            )
        except Exception as err:
            if not outcome.contains_committed(step.operations):
                apply_batch_indeterminate(self, step.operations, outcome)
            commit_error = outcome.commit_error(err, step)
            add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
            raise commit_error from err
        try:
            self._recover_confirmed_fields(step, groups)
        except Exception:
            _LOGGER.exception('confirmed-write readback failed during %s', step.description)

    def _recover_upload_fields(
        self,
        reads: list[ConfirmedFieldRead],
        binding: RegistryBinding | None,
    ) -> None:
        """Attempt each entity type's bounded upload current-value read."""
        for group in group_upload_field_reads(reads, binding):
            try:
                self._read_confirmed_field_group(group)
            except Exception:
                _LOGGER.exception(
                    'confirmed-upload readback failed for entity_type=%s',
                    group.entity_type,
                )

    def expire(self, obj: Entity) -> None:
        """Mark one session-owned entity expired.

        Args:
            obj: Entity to expire.

        Raises:
            SgalchemyError: If this session does not own `obj`.
        """
        self._state.expire(obj)

    def expire_all(self) -> None:
        """Mark every session-owned entity expired."""
        self._state.expire_all()

    def hydrate_record(
        self,
        entity_type_name: str,
        record: Mapping[str, Any],
        binding: RegistryBinding | None,
        *,
        retired: bool | None = None,
    ) -> Entity:
        """Hydrate one wire record through a captured binding.

        Args:
            entity_type_name: Canonical expected entity type.
            record: Provider record containing canonical identity.
            binding: Complete operation binding, or captured absence.
            retired: Root retirement observation, or `None` to preserve
                lifecycle state. Staged deletes and revivals take precedence.
                Linked records do not inherit this observation.

        Returns:
            Existing or newly allocated entity.

        Raises:
            SgalchemyError: If identity is absent, malformed, or contradictory.
        """
        return self._state.hydrate_record(entity_type_name, record, binding, retired=retired)

    def hydration_scope(self) -> AbstractContextManager[None]:
        """Share staged identity lookup for synchronous response hydration only."""
        return self._state.hydration_scope()

    def load_binding(self, context: str, *, force: bool = False) -> RegistryBinding:
        """Load one binding under a public operation's provider context.

        Args:
            context: Session fault annotation for schema preparation failures.
            force: Ignore prepared schema and disk cache.

        Returns:
            Complete Registry binding captured after preparation.

        Raises:
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        return self._annotated(context, lambda: self._engine.load_binding(force=force))

    def _compile_scalars_call(
        self,
        stmt: Select[Entity],
        binding: RegistryBinding,
        context: str,
    ) -> tuple[FindCall, RegistryBinding]:
        """Compile once, then recover one joined-load schema drift before I/O."""
        try:
            call = compile_statement(stmt, binding.schemas)
        except SchemaDrift:
            current = self._engine.prepared_binding()
            if current is not None and current is not binding:
                retry_binding = current
            else:
                _ENGINE_LOG.info('joinedload schema drift: refreshing schema once')
                retry_binding = self.load_binding(context, force=True)
            try:
                call = compile_statement(stmt, retry_binding.schemas)
            except SchemaDrift as error:
                raise CompilationError(
                    f'{error.entity_type}.{error.field_code} target '
                    f'{error.target_type!r} remains absent after refreshing schema once; '
                    'the joinedload cannot be compiled'
                ) from None
            binding = retry_binding
        if not isinstance(call, FindCall):
            raise TypeError('summary statements use session.execute(statement)')
        return call, binding

    def _plan_commit(self) -> CommitPlan:
        """Build one complete plan with bounded pre-write schema recovery."""
        binding = self._engine.prepared_binding()
        refreshed = False
        while True:
            try:
                return plan_commit(self, binding, self._state.identity_map)
            except SchemaDrift as error:
                current = self._engine.prepared_binding()
                if current is not binding:
                    binding = current
                    continue
                if binding is None:
                    registry_binding = self._engine.registry.binding_snapshot()
                    if registry_binding is not None and registry_binding.key != self._engine.site:
                        raise SgalchemyError(REGISTRY_SITE_MISMATCH_MESSAGE) from None
                    binding = self.load_binding('raised while preparing commit')
                    continue
                if not refreshed:
                    _ENGINE_LOG.info('commit schema drift: refreshing schema once')
                    binding = self.load_binding('raised while preparing commit', force=True)
                    refreshed = True
                    continue
                raise schema_drift_error(error, refreshed=refreshed) from None

    def _require_type_schema(
        self,
        context: str,
        type_name: str,
    ) -> tuple[EntitySchema, RegistryBinding]:
        """Return the type's schema and binding, forcing at most one refresh."""
        binding = self.load_binding(context)
        schema = binding.schemas.get(type_name)
        if schema is None:
            binding = self.load_binding(context, force=True)
            schema = binding.schemas.get(type_name)
        if schema is None:
            raise SchemaUnavailableError(
                f'{type_name} schema remains absent after one forced schema refresh; '
                'make the entity type visible on the site, then retry'
            )
        return schema, binding

    def execute(self, stmt: Select[None]) -> Result | dict[str, Any]:
        """Execute a projection or summary statement.

        Args:
            stmt: Projection or summary statement.

        Returns:
            Projection result or raw summary payload.

        Raises:
            TypeError: If `stmt` selects entities.
            CompilationError: If the statement cannot compile.
            SgalchemyError: If the provider returns the wrong outer shape.
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        if stmt.entity is not None:
            raise TypeError('entity statements use session.scalars(statement)')
        binding = self.load_binding(f'raised while executing statement {stmt}')
        call = compile_statement(stmt, binding.schemas)
        if isinstance(call, SummarizeCall):
            return self.execute_call(call, binding)
        records = self.execute_call(call, binding)
        return projection_result(call, records)

    def get(self, cls: type[_EntityT], entity_id: int) -> _EntityT | None:
        """Return one entity by identity, loading it on a cache miss.

        Args:
            cls: Reflected entity class.
            entity_id: Positive server identifier.

        Returns:
            Identity-mapped entity or `None` when absent.

        Raises:
            SchemaUnavailableError: If the entity type remains absent after one
                forced schema refresh.
            SgalchemyError: If identity or registry ownership is invalid.
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        if type(entity_id) is not int or entity_id <= 0:
            raise SgalchemyError('entity id must be a positive integer id')
        type_name = self._require_owned_class(cls)
        existing = self._state.entity(type_name, entity_id)
        if existing is not None:
            self._require_owned_object(existing)
            return cast('_EntityT', existing)
        context = f'raised while executing find_one({type_name!r}, id={entity_id})'
        schema, binding = self._require_type_schema(context, type_name)
        record = self._annotated(
            context,
            lambda: self._engine.call(
                'find_one', type_name, [['id', 'is', entity_id]], schema.readable_field_codes()
            ),
        )
        validated = validate_point_records(
            [] if record is None else [record],
            type_name,
            entity_id,
            allow_absent=True,
        )
        if validated is None:
            return None
        return cast('_EntityT', self.hydrate_record(type_name, validated, binding))

    def merge_dict(self, raw: Mapping[str, Any]) -> Entity:
        """Adopt one raw entity dictionary without hidden I/O.

        Args:
            raw: Entity dictionary containing canonical type and id.

        Returns:
            Identity-mapped entity.

        Raises:
            SgalchemyError: If root identity is not canonical.
        """
        return self._state.merge_dict(raw)

    def refresh(self, obj: Entity) -> None:
        """Reload readable fields without overwriting local dirty values.

        Only fields returned by the provider and actually refreshed have their
        trust markers cleared. Dirty or omitted fields retain their markers.
        Retired entities are refreshed with `retired_only=True`. Refresh
        preserves lifecycle state and any staged delete or revival.

        Args:
            obj: Session-owned persistent entity.

        Raises:
            SchemaUnavailableError: If the entity type remains absent after one
                forced schema refresh.
            SgalchemyError: If ownership or provider result is invalid.
            Exception: Original provider exception, annotated and unwrapped.
        """
        state = self._require_owned_object(obj)
        if state.id is None:
            raise SgalchemyError('cannot refresh an entity without a server identity')
        context = f'raised while refreshing {state.entity_type}(id={state.id})'
        schema, binding = self._require_type_schema(context, state.entity_type)
        call = refresh_call(state, schema)
        records = self.execute_call(call, binding)
        record = validate_point_records(
            records,
            state.entity_type,
            state.id,
            allow_absent=False,
        )
        self.hydrate_record(state.entity_type, record, binding)
        state.expired = False

    def register_identity(self, obj: Entity) -> None:
        """Register one persistent object after collision-safe validation.

        Args:
            obj: Persistent object to register.
        """
        self._state.register_identity(obj)

    def reserve_implicit_calls(self, count: int) -> None:
        """Reserve a complete lazy-load call batch before provider I/O.

        Args:
            count: Number of per-target-type calls the lazy load requires.

        Raises:
            SgalchemyError: If the reservation would exceed the configured
                budget.
        """
        limit = self._max_implicit_queries
        if limit is not None and self._implicit_queries + count > limit:
            noun = 'query' if count == 1 else 'queries'
            remaining = max(limit - self._implicit_queries, 0)
            raise SgalchemyError(
                f'max_implicit_queries={limit} budget exceeded: lazy load needs '
                f'{count} implicit {noun}, but {remaining} remain'
            )
        self._implicit_queries += count

    def revive(self, obj: Entity) -> None:
        """Stage revival or cancel an inverse local transition.

        Args:
            obj: Persistent entity to transition.

        Raises:
            SgalchemyError: If `obj` is pending, already active without a
                staged delete, foreign, indeterminate, or owned elsewhere.
        """
        stage_revive(self, obj)

    def rollback(self) -> None:
        """Restore dirty snapshots and discard staged local work without I/O.

        Rollback changes only local state and cannot undo provider writes.
        """
        discard_local_state(
            RecoveryAdapter(
                unit_of_work=self._unit_of_work,
                detach_canceled=self._state.detach_canceled,
                discard_identities=self._state.discard_identities,
                identity_values=self._state.identity_values,
                owns_entity=self._state.owns_entity,
            )
        )

    def scalars(self, stmt: Select[_EntityT]) -> list[_EntityT]:
        """Execute one entity statement and its requested relationship loads.

        A `joinedload()` may refresh target schema once and recompile before
        its single primary query. `selectinload()` performs bounded observed
        target reads. Opted-in `lazyload()` defers bounded sync I/O to later
        attribute access.

        The primary query observes root entities as active or retired from
        `retired_only`, unless that exact object has a staged delete or
        revival.
        Linked entities do not inherit the root retirement mode. Dirty values
        and their existing field-trust rules are preserved.

        Args:
            stmt: Entity-select statement.

        Returns:
            Identity-mapped entities in provider order.

        Raises:
            TypeError: If `stmt` is a projection or summary.
            CompilationError: If the statement or loader options cannot
                compile.
            SgalchemyError: If registry ownership, returned identity data, or
                relationship target availability is invalid.
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        if stmt.entity is None:
            raise TypeError('projection and summary statements use session.execute(statement)')
        type_name = self._require_owned_class(stmt.entity)
        relationship_options = normalize_relationship_loaders(stmt.loader_options)
        context = f'raised while executing entity statement {stmt}'
        binding = self.load_binding(context)
        call, binding = self._compile_scalars_call(stmt, binding, context)
        if call.entity_type != type_name:
            raise SgalchemyError('compiled entity type contradicts the owned statement class')
        records = self.execute_call(call, binding)
        with self.hydration_scope():
            objects = [
                self.hydrate_record(
                    call.entity_type,
                    record,
                    binding,
                    retired=bool(call.extra.get('retired_only', False)),
                )
                for record in records
            ]
        selectin_options: list[SelectinLoad] = []
        for option in relationship_options:
            if isinstance(option, LazyLoad):
                context = LazyContext(self, objects, option)
                for obj in objects:
                    entity_state(obj).lazy[option.field.filter_field] = context
            elif isinstance(option, SelectinLoad):
                selectin_options.append(option)
        execute_selectins(self, objects, selectin_options, binding)
        return cast('list[_EntityT]', objects)

    @property
    def unit_of_work(self) -> UnitOfWork:
        """Return this session's identity-based staging state.

        Returns:
            Package-internal staging state exposing immutable category
            snapshots.
        """
        return self._unit_of_work

    def _execute_revive_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
        progress: CallProgress,
    ) -> None:
        """Execute and reconcile one staged entity revival."""
        obj = step.entities[0]
        state = entity_state(obj)
        if state.id is None:
            raise RuntimeError('revive entity has no server id')
        try:
            self._call_commit_step(step, progress, 'revive', state.entity_type, state.id)
        except CommitError:
            raise
        except Exception as err:
            commit_error = outcome.commit_error(err, step)
            add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
            raise commit_error from err
        try:
            self.register_identity(obj)
            apply_revive_result(self.unit_of_work, obj, outcome)
        except CommitError:
            raise
        except Exception as err:
            commit_error = outcome.commit_error(err, step)
            add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
            raise commit_error from err

    def _execute_commit_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
        sole_batch: bool,
        progress: CallProgress,
    ) -> ConfirmedFieldRead | None:
        """Execute and reconcile one already-planned serial provider step."""
        if step.kind == 'revive':
            self._execute_revive_step(step, outcome, progress)
            return None
        if step.kind == 'batch':
            self._execute_batch_step(step, outcome, sole_batch, progress)
            return None
        return self._execute_upload_step(step, outcome, progress)


def _commit_error_with_batch_status(
    err: Exception,
    step: CommitStep,
    outcome: CommitOutcome,
    *,
    invoked: bool,
) -> CommitError:
    """Build an ordinary failure with batch invocation and auth recovery notes."""
    commit_error = outcome.commit_error(err, step)
    if note := batch_failure_note(step, outcome, invoked=invoked):
        add_note_once(commit_error, note)
    add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
    return commit_error


def _retire_sync_upload_attempt(
    session: 'Session',
    step: CommitStep,
    outcome: CommitOutcome,
    progress: CallProgress,
) -> ConfirmedFieldRead | None:
    """Retire one submitted upload locally, or ignore a non-upload step."""
    if not progress.invoked.is_set() or not step.is_upload:
        return None
    upload = step.require_upload()
    return apply_upload_indeterminate(session, upload, outcome)


def _execute_sync_commit_plan(
    session: 'Session',
    plan: CommitPlan,
    batch_size: int | None,
    outcome: CommitOutcome,
    *,
    wrap_ordinary_failure: bool,
) -> None:
    """Execute one finalized synchronous commit plan."""
    execute_step = session._execute_commit_step  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    recover_upload_fields = session._recover_upload_fields  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    steps = commit_steps(plan, batch_size)
    sole_batch = len(steps) == 1 and steps[0].kind == 'batch'
    pending_upload_reads: list[ConfirmedFieldRead] = []
    for step in steps:
        if pending_upload_reads and not step.is_upload:
            recover_upload_fields(pending_upload_reads, plan.binding)
            pending_upload_reads.clear()
        progress = CallProgress()
        try:
            read = execute_step(step, outcome, sole_batch, progress)
        except CommitError:
            session._active_step = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            recover_upload_fields(
                pending_upload_reads if not progress.invoked.is_set() else [], plan.binding
            )
            _retire_sync_upload_attempt(session, step, outcome, progress)
            raise
        except Exception as err:
            session._active_step = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            ambiguous = _retire_sync_upload_attempt(session, step, outcome, progress)
            if ambiguous is not None:
                recover_upload_fields([*pending_upload_reads, ambiguous], plan.binding)
                commit_error = outcome.commit_error(err, step)
                add_auth_note(err, commit_error, AUTH_UPLOAD_NOTE)
                raise commit_error from err
            recover_upload_fields(pending_upload_reads, plan.binding)
            if step.is_upload or wrap_ordinary_failure:
                commit_error = _commit_error_with_batch_status(
                    err, step, outcome, invoked=progress.invoked.is_set()
                )
                raise commit_error from err
            raise
        except BaseException:
            submitted_step = session._active_step if progress.invoked.is_set() else None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            session._active_step = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            confirmed_batch = submitted_step is not None and (
                submitted_step.kind == 'batch'
                and outcome.contains_committed(submitted_step.operations)
            )
            if submitted_step is not None and submitted_step.is_upload:
                _retire_sync_upload_attempt(session, submitted_step, outcome, progress)
            else:
                items = () if submitted_step is None or confirmed_batch else submitted_step.items
                entities = (
                    () if submitted_step is None or confirmed_batch else submitted_step.entities
                )
                outcome.mark_indeterminate(items)
                mark_indeterminate_entities(entities)
            raise
        session._active_step = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if read is not None:
            pending_upload_reads.append(read)
    recover_upload_fields(pending_upload_reads, plan.binding)


def commit_session(
    session: 'Session',
    batch_size: int | None = None,
    *,
    wrap_ordinary_failure: bool,
) -> None:
    """Execute one normalized synchronous commit.

    Factory commit-on-success mode wraps ordinary validation, planning, and
    execution failures before its caller closes the Session and discards the
    local retry queue. Initial failure-evidence capture precedes this wrapping
    boundary.

    Args:
        session: Session whose staged work is committed.
        batch_size: Optional positive maximum requests per batch call. A
            supplied limit must fit every indivisible operation group.
        wrap_ordinary_failure: Wrap ordinary validation, planning, and
            execution failures in `CommitError`.

    Raises:
        CommitError: If an ordinary failure occurs within the wrapping boundary
            while wrapping is enabled, or the ordinary commit path reports
            partial outcome evidence.
        Exception: Original ordinary failure before wrapping begins or when
            wrapping is disabled.
        ValueError: When wrapping is disabled, if `batch_size` is neither a
            positive integer nor `None`, or is smaller than an indivisible
            operation group.
    """
    outcome = CommitOutcome(
        committed=[],
        staged=(
            staged_intent_snapshot(
                session.unit_of_work,
                session._identity_values(),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            )
            if wrap_ordinary_failure
            else []
        ),
        indeterminate=[],
        withdrawn=[],
    )
    try:
        validate_batch_size(batch_size)
        plan_commit_call = session._plan_commit  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        plan = plan_commit_call()
        plan = preflight_uploads(plan)
        outcome = CommitOutcome.from_plan(plan)
        _execute_sync_commit_plan(
            session,
            plan,
            batch_size,
            outcome,
            wrap_ordinary_failure=wrap_ordinary_failure,
        )
    except CommitError:
        raise
    except Exception as err:
        if not wrap_ordinary_failure:
            raise
        commit_error = outcome.commit_error(err, None)
        add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
        raise commit_error from err
