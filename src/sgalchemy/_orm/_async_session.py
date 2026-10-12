"""Manage asynchronous reads and task-owned entity identity."""

# pyright: reportPrivateUsage=false

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Iterable, Mapping
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self, TypeVar, cast, overload

from sgalchemy._core._async_engine import AsyncEngine
from sgalchemy._core._compiler import (
    FindCall,
    SummarizeCall,
    compile_statement,
    validate_relationship_field,
)
from sgalchemy._core._engine_common import (
    AUTH_RECOVERY_NOTE,
    AUTH_UPLOAD_NOTE,
    CallProgress,
    add_auth_note,
    add_note_once,
)
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._options import LazyLoad, SelectinLoad, normalize_relationship_loaders
from sgalchemy._core._registry import (
    REGISTRY_SITE_MISMATCH_MESSAGE,
    Entity,
    RegistryBinding,
    entity_state,
)
from sgalchemy._core._result import Result
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import Select
from sgalchemy._errors import CommitError, CompilationError, SchemaUnavailableError, SgalchemyError
from sgalchemy._orm._async_loaders import execute_selectins, load_relationship
from sgalchemy._orm._session_common import (
    annotate_read_error,
    projection_result,
    refresh_call,
    validate_query_payload,
)
from sgalchemy._orm._state import SessionState, validate_point_records
from sgalchemy._orm._uow import (
    SOLE_BATCH_STAGED_NOTE,
    BatchReconciliationError,
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
    count_unsubmitted_work,
    discard_local_state,
    group_upload_field_reads,
    mark_indeterminate_entities,
    new_session_unit_of_work,
    plan_commit,
    preflight_uploads,
    retire_interrupted_step,
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

_AUTOMATIC_CLEANUP_WARNING = (
    'automatic task cleanup detached entities, reverted local edits, '
    'and discarded uncommitted work'
)
_ENGINE_LOG = logging.getLogger('sgalchemy.engine')
_LOGGER = logging.getLogger('sgalchemy.session')
_EntityT = TypeVar('_EntityT', bound=Entity)


class AsyncSession:
    """Manage entity identity and staged writes with explicit awaited I/O.

    Use one session per concurrent task. Operations require a running task and
    reject another task while an operation is active. Queries never autoflush,
    and attribute access cannot use synchronous lazy loading.

    Async context management returns the session and closes local state on exit
    without committing or suppressing exceptions. Close restores local edits,
    discards staged work, and detaches entities without undoing provider
    writes. Normal close permits reuse. If cancellation or another control-flow
    interruption quarantines a submitted write, closing leaves the session
    unusable; replace it after reconciling the server outcome.

    Only the methods listed in the session guide are supported. Other
    non-underscore methods, such as `attach_for_staging`, `hydrate_record`,
    `hydration_scope`, `register_identity`, and `unit_of_work`, are
    package-internal orchestration hooks and may change without notice.
    """

    def _assert_task_access(self) -> asyncio.Task[object]:
        """Reject work from another task while this session is active."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None:
            raise SgalchemyError('AsyncSession operations require a running asyncio task')
        owner = self._owner
        if owner is not None and owner is not task:
            raise SgalchemyError(
                'AsyncSession cannot be used by a concurrent task; create one session per task'
            )
        return task

    def _assert_usable(self) -> None:
        """Reject use after an indeterminate cancellation or its closing discard."""
        if self._discarded_after_cancel:
            raise SgalchemyError(
                'this AsyncSession was closed after an indeterminate write; '
                'create a new AsyncSession'
            )
        if self._quarantined_write is not None:
            raise SgalchemyError(
                f'AsyncSession has an indeterminate {self._quarantined_write}; '
                'reconcile the server outcome, close this session, and create '
                'a new AsyncSession'
            )

    def _assert_session_access(self) -> None:
        """Reject unusable-session access or another active task."""
        self._assert_task_access()
        self._assert_usable()

    def __init__(self, engine: AsyncEngine) -> None:
        """Bind a new empty identity map to one async engine.

        Args:
            engine: Async engine bound to the session.
        """
        self._active_step: CommitStep | None = None
        self._automatic_cleanup_details: tuple[str, bool] = ('', False)
        self._automatic_cleanup_done = False
        self._automatic_cleanup_pending = False
        self._cleanup_anchors: list[Entity] = []
        self._depth = 0
        self._discarded_after_cancel = False
        self._engine = engine
        self._owner: asyncio.Task[object] | None = None
        self._quarantined_write: str | None = None
        self._state = SessionState(
            self,
            engine.registry,
            engine.prepared_binding,
            engine.mark_schema_stale,
            self._assert_session_access,
        )
        self._unit_of_work = new_session_unit_of_work()

    async def __aenter__(self) -> Self:
        """Return this session for async context-manager use."""
        self._assert_session_access()
        return self

    def _identity_values(self) -> list[Entity]:
        """Return a snapshot of identity-mapped objects."""
        self._assert_session_access()
        return self._state.identity_values()

    @asynccontextmanager
    async def _operation_guard(
        self,
        *,
        allow_quarantined: bool = False,
    ) -> AsyncGenerator[None]:
        """Reject overlapping use while permitting same-task helper nesting."""
        task = self._assert_task_access()
        if not allow_quarantined:
            self._assert_usable()
        self._owner = task
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                if self._automatic_cleanup_pending:
                    task_name, cancelled = self._automatic_cleanup_details
                    automatic_cleanup(
                        self,
                        task_name=task_name,
                        cancelled=cancelled,
                    )

    def _quarantine_submitted_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
    ) -> None:
        """Remove one submitted step from retry and quarantine its exact entities."""
        for obj in retire_interrupted_step(self, step):
            if all(candidate is not obj for candidate in self._cleanup_anchors):
                self._cleanup_anchors.append(obj)
        outcome.mark_indeterminate(step.items)
        mark_indeterminate_entities(step.entities)

    def _record_commit_interruption(
        self,
        err: BaseException,
        step: CommitStep | None,
        outcome: CommitOutcome,
        progress: CallProgress | None,
    ) -> None:
        """Quarantine an invoked call or dispatched work that can outlive its waiter."""
        if progress is None or not (progress.invoked.is_set() or progress.dispatched.is_set()):
            return
        confirmed_batch = (
            step is not None
            and step.kind == 'batch'
            and outcome.contains_committed(step.operations)
        )
        if step is None or confirmed_batch:
            return
        self._quarantine_submitted_step(step, outcome)
        self._quarantined_write = step.description
        add_note_once(
            err,
            f'sgalchemy: async commit interrupted during {step.description}; '
            'server outcome is indeterminate, this session will not retry it, '
            'and the session must be closed',
        )

    def _route_commit_exception(
        self,
        err: Exception,
        step: CommitStep | None,
        outcome: CommitOutcome,
        steps: tuple[CommitStep, ...],
        *,
        invoked: bool,
    ) -> Exception | None:
        """Return a wrapped fault, or `None` for raw sole-batch passthrough."""
        if isinstance(err, BatchReconciliationError):
            commit_error = outcome.commit_error(err, step)
            add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
            return commit_error
        if (
            step is not None
            and step.kind == 'batch'
            and outcome.contains_committed(step.operations)
        ):
            commit_error = outcome.commit_error(err, step)
            add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
            return commit_error
        if len(steps) == 1 and steps[0].kind == 'batch':
            add_note_once(err, SOLE_BATCH_STAGED_NOTE)
            if note := batch_failure_note(step, outcome, invoked=invoked):
                add_note_once(err, note)
            add_note_once(err, f'sgalchemy: failed during {steps[0].description}')
            add_auth_note(err, err, AUTH_RECOVERY_NOTE)
            return None
        commit_error = outcome.commit_error(err, step)
        if note := batch_failure_note(step, outcome, invoked=invoked):
            add_note_once(commit_error, note)
        add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
        return commit_error

    def add(self, obj: Entity) -> None:
        """Stage one pending entity for creation.

        Args:
            obj: Schema-bound pending entity from `Entity.new()`.

        Raises:
            SgalchemyError: If `obj` is foreign, persistent, indeterminate, or
                owned elsewhere.
        """
        self._assert_session_access()
        stage_add(self, obj)

    def assert_identity_available(self, obj: Entity) -> None:
        """Reject another object mapped under the same persistent identity.

        Args:
            obj: Persistent object whose identity must remain unique.

        Raises:
            SgalchemyError: If a different object already owns the identity.
        """
        self._assert_session_access()
        self._state.assert_identity_available(obj)

    def assert_same_registry(self, value: Entity | type[Entity]) -> None:
        """Reject a class or object owned by another registry or open session.

        Args:
            value: Entity class or instance to validate before local use.

        Raises:
            SgalchemyError: If `value` belongs to another registry or open
                session.
        """
        self._assert_session_access()
        self._state.assert_same_registry(value)

    def attach_for_staging(self, obj: Entity) -> None:
        """Attach one detached entity for local staging.

        Args:
            obj: Entity to attach after access and registry checks.

        Raises:
            SgalchemyError: If another open session owns `obj`.
        """
        self._assert_session_access()
        self._state.attach(obj)

    async def close(self) -> None:
        """Revert local edits, detach entities, and retire quarantine locally.

        Closing performs no provider I/O and cannot undo provider writes. A
        session closed after an indeterminate submitted write stays unusable.
        """
        async with self._operation_guard(allow_quarantined=True):
            _release_local_state(self, _automatic_recovery_adapter(self))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close local state without suppressing exceptions."""
        del exc_type, exc_value, traceback
        await self.close()

    async def commit(self, batch_size: int | None = None) -> None:
        """Execute staged work with bounded pre-write and readback recovery.

        A commit is atomic only when its complete write plan uses one provider
        `batch()` call. Planning may prepare schema or force one pre-write
        refresh. Confirmed-write and thumbnail reconciliation may issue bounded
        read-only calls. Uninvoked uploads retain their staged paths and trust
        state after ordinary failure. A `BaseException` after provider entry,
        or after threaded dispatch that can outlive its waiter, quarantines the
        session and affected entities.

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
        await commit_async_session(self, batch_size, wrap_ordinary_failure=False)

    def delete(self, obj: Entity) -> None:
        """Stage retirement or cancel an inverse local transition.

        Args:
            obj: Pending or persistent entity to transition.

        Raises:
            SgalchemyError: If `obj` is foreign, unstaged, indeterminate, or
                owned elsewhere.
        """
        self._assert_session_access()
        stage_delete(self, obj)

    def detach_canceled(self, obj: Entity) -> None:
        """Detach a canceled object absent from this identity map.

        Args:
            obj: Entity removed from local staging.
        """
        self._assert_session_access()
        self._state.detach_canceled(obj)

    @property
    def engine(self) -> AsyncEngine:
        """Return the async engine bound to this session.

        Returns:
            Bound async engine.
        """
        return self._engine

    def expire(self, obj: Entity) -> None:
        """Mark one session-owned entity expired without I/O.

        Args:
            obj: Entity to expire.

        Raises:
            SgalchemyError: If this session does not own `obj`.
        """
        self._assert_session_access()
        self._state.expire(obj)

    def expire_all(self) -> None:
        """Mark every session-owned entity expired without I/O."""
        self._assert_session_access()
        self._state.expire_all()

    @overload
    def finish_execute_call_result(
        self,
        call: FindCall,
        payload_or_error: QueryResult | BaseException,
        *,
        raised: bool,
    ) -> WireRecords: ...

    @overload
    def finish_execute_call_result(
        self,
        call: SummarizeCall,
        payload_or_error: QueryResult | BaseException,
        *,
        raised: bool,
    ) -> WireRecord: ...

    @overload
    def finish_execute_call_result(
        self,
        call: FindCall | SummarizeCall,
        payload_or_error: QueryResult | BaseException,
        *,
        raised: bool,
    ) -> QueryResult: ...

    def finish_execute_call_result(
        self,
        call: FindCall | SummarizeCall,
        payload_or_error: QueryResult | BaseException,
        *,
        raised: bool,
    ) -> QueryResult:
        """Validate or raise one completed compiled-call result in the owner task.

        Args:
            call: Compiled provider call whose result completed.
            payload_or_error: Raw provider payload or captured call failure.
            raised: Whether `payload_or_error` was raised by provider
                execution.

        Returns:
            Provider payload in its documented outer shape.

        Raises:
            BaseException: Captured call failure, preserving its identity.
            SgalchemyError: If the outer result shape contradicts the call.
        """
        self._assert_session_access()
        if raised:
            error = cast('BaseException', payload_or_error)
            if isinstance(error, Exception):
                context = f'raised while executing {call}'
                annotate_read_error(error, context)
            raise error
        return validate_query_payload(call, payload_or_error)

    @overload
    async def execute_call(
        self,
        call: FindCall,
        binding: RegistryBinding | None,
    ) -> WireRecords: ...

    @overload
    async def execute_call(
        self,
        call: SummarizeCall,
        binding: RegistryBinding | None,
    ) -> WireRecord: ...

    @overload
    async def execute_call(
        self,
        call: FindCall | SummarizeCall,
        binding: RegistryBinding | None,
    ) -> QueryResult: ...

    async def execute_call(
        self,
        call: FindCall | SummarizeCall,
        binding: RegistryBinding | None,
    ) -> QueryResult:
        """Execute compiled provider work through a required captured binding.

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
        async with self._operation_guard():
            del binding
            try:
                payload: QueryResult = await self._engine.execute_call(call)
            except Exception as err:
                self.finish_execute_call_result(call, err, raised=True)
                raise
            return self.finish_execute_call_result(call, payload, raised=False)

    async def _read_confirmed_field_group(self, group: ConfirmedFieldReadGroup) -> None:
        """Execute and apply one bounded current-value field read."""
        call = confirmed_field_read_call(group)
        if call is None:
            return
        records = await self.execute_call(call, group.binding)
        apply_confirmed_field_readback(self, group, records)

    async def _recover_confirmed_fields(
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
                await self._read_confirmed_field_group(group)

    async def _recover_upload_fields(
        self,
        reads: list[ConfirmedFieldRead],
        binding: RegistryBinding | None,
    ) -> None:
        """Attempt each entity type's bounded upload current-value read."""
        for group in group_upload_field_reads(reads, binding):
            try:
                await self._read_confirmed_field_group(group)
            except Exception:
                _LOGGER.exception(
                    'confirmed-upload readback failed for entity_type=%s',
                    group.entity_type,
                )

    def hydrate_record(
        self,
        entity_type_name: str,
        record: Mapping[str, Any],
        binding: RegistryBinding | None,
        *,
        retired: bool | None = None,
    ) -> Entity:
        """Hydrate one wire record through a required captured binding.

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
            SgalchemyError: If task access or identity is invalid.
        """
        self._assert_session_access()
        return self._state.hydrate_record(entity_type_name, record, binding, retired=retired)

    def hydration_scope(self) -> AbstractContextManager[None]:
        """Share staged identity lookup for synchronous response hydration only."""
        self._assert_session_access()
        return self._state.hydration_scope()

    async def load_binding(self, context: str, *, force: bool = False) -> RegistryBinding:
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
        async with self._operation_guard():
            try:
                return await self._engine.load_binding(force=force)
            except Exception as err:
                annotate_read_error(err, context)
                raise

    async def _compile_scalars_call(
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
                retry_binding = await self.load_binding(context, force=True)
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
            raise TypeError('summary statements use await session.execute(statement)')
        return call, binding

    async def _plan_commit(self) -> CommitPlan:
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
                    binding = await self.load_binding('raised while preparing commit')
                    continue
                if not refreshed:
                    _ENGINE_LOG.info('commit schema drift: refreshing schema once')
                    binding = await self.load_binding(
                        'raised while preparing commit',
                        force=True,
                    )
                    refreshed = True
                    continue
                raise schema_drift_error(error, refreshed=refreshed) from None

    async def _require_type_schema(
        self,
        context: str,
        type_name: str,
    ) -> tuple[EntitySchema, RegistryBinding]:
        """Return the type's schema and binding, forcing at most one refresh."""
        binding = await self.load_binding(context)
        schema = binding.schemas.get(type_name)
        if schema is None:
            binding = await self.load_binding(context, force=True)
            schema = binding.schemas.get(type_name)
        if schema is None:
            raise SchemaUnavailableError(
                f'{type_name} schema remains absent after one forced schema refresh; '
                'make the entity type visible on the site, then retry'
            )
        return schema, binding

    async def execute(self, stmt: Select[None]) -> Result | dict[str, Any]:
        """Execute one projection or summary statement.

        Args:
            stmt: Projection or summary statement.

        Returns:
            Projection result or raw summary payload.

        Raises:
            TypeError: If `stmt` selects entities.
            CompilationError: If the statement cannot compile.
            SgalchemyError: If the session is unusable or the provider returns
                the wrong outer shape.
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        async with self._operation_guard():
            if stmt.entity is not None:
                raise TypeError('entity statements use await session.scalars(statement)')
            binding = await self.load_binding(f'raised while executing statement {stmt}')
            call = compile_statement(stmt, binding.schemas)
            if isinstance(call, SummarizeCall):
                return await self.execute_call(call, binding)
            records = await self.execute_call(call, binding)
            return projection_result(call, records)

    async def get(self, cls: type[_EntityT], entity_id: int) -> _EntityT | None:
        """Return one entity by identity, loading it on a cache miss.

        Args:
            cls: Reflected entity class.
            entity_id: Positive server identifier.

        Returns:
            Identity-mapped entity or `None` when absent.

        Raises:
            SchemaUnavailableError: If the entity type remains absent after one
                forced schema refresh.
            SgalchemyError: If session access, identity, or registry ownership
                is invalid.
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        async with self._operation_guard():
            if type(entity_id) is not int or entity_id <= 0:
                raise SgalchemyError('entity id must be a positive integer id')
            type_name = self._state.require_owned_class(cls)
            existing = self._state.entity(type_name, entity_id)
            if existing is not None:
                self._state.require_owned_object(existing)
                return cast('_EntityT', existing)
            context = f'raised while executing find_one({type_name!r}, id={entity_id})'
            schema, binding = await self._require_type_schema(context, type_name)
            try:
                record = await self._engine.call(
                    'find_one',
                    type_name,
                    [['id', 'is', entity_id]],
                    schema.readable_field_codes(),
                )
            except Exception as err:
                annotate_read_error(err, context)
                raise
            validated = validate_point_records(
                [] if record is None else [record],
                type_name,
                entity_id,
                allow_absent=True,
            )
            if validated is None:
                return None
            return cast(
                '_EntityT',
                self._state.hydrate_record(
                    type_name,
                    validated,
                    binding,
                ),
            )

    async def load(
        self,
        objects: Entity | Iterable[Entity],
        field: FieldExpr,
    ) -> None:
        """Explicitly load exactly one relationship across owned entities.

        Validate all parents before I/O and reject pending parents. Fetch the
        field only for persistent parents where it is absent; present `None` or
        `[]` values count as loaded. Then preflight observed target types,
        optionally refresh schema once, and perform bounded target reads.
        Dirty relationship values remain protected. An empty iterable is a
        no-op.

        Args:
            objects: One entity or an iterable consumed once to load.
            field: Direct relationship field rooted at every object's type.

        Raises:
            SgalchemyError: If ownership, field root, parent identity, loaded
                relationship values, or a required parent response is invalid.
            CompilationError: If the field is not a readable direct
                relationship.
            RelationshipTargetUnavailableError: If observed target types remain
                absent after one forced schema refresh.
            Exception: Original provider exception, annotated and unwrapped.
        """
        async with self._operation_guard():
            materialized = [objects] if isinstance(objects, Entity) else list(objects)
            if not materialized:
                return
            states = [self._state.require_owned_object(obj) for obj in materialized]
            for state in states:
                object_type = state.entity_type
                if object_type != field.root_type:
                    raise SgalchemyError(
                        f'cannot load {field.root_type}.{field.filter_field} for '
                        f'{object_type}: field root {field.root_type!r} does not match '
                        f'object type {object_type!r}'
                    )
            if any(state.id is None for state in states):
                raise SgalchemyError(
                    'cannot load a relationship for an entity without a server identity'
                )
            context = f'raised while loading {field.root_type}.{field.filter_field}'
            binding = await self.load_binding(context)
            field_code = validate_relationship_field(
                field,
                'load',
                states[0].entity_type,
                binding.schemas,
            )
            missing_ids: dict[bool, list[int]] = {}
            for retired in (False, True):
                ids = sorted(
                    {
                        cast('int', state.id)
                        for state in states
                        if field_code not in state.loaded and state.deleted is retired
                    }
                )
                if ids:
                    missing_ids[retired] = ids
            await load_relationship(self, materialized, field, missing_ids, binding)

    def merge_dict(self, raw: Mapping[str, Any]) -> Entity:
        """Adopt one raw entity dictionary without I/O.

        Args:
            raw: Entity dictionary containing canonical type and id.

        Returns:
            Identity-mapped entity.

        Raises:
            SgalchemyError: If task access or root identity is invalid.
        """
        self._assert_session_access()
        return self._state.merge_dict(raw)

    async def refresh(self, obj: Entity) -> None:
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
        async with self._operation_guard():
            state = self._state.require_owned_object(obj)
            if state.id is None:
                raise SgalchemyError('cannot refresh an entity without a server identity')
            context = f'raised while refreshing {state.entity_type}(id={state.id})'
            schema, binding = await self._require_type_schema(context, state.entity_type)
            call = refresh_call(state, schema)
            records = await self.execute_call(call, binding)
            record = validate_point_records(
                records,
                state.entity_type,
                state.id,
                allow_absent=False,
            )
            self._state.hydrate_record(state.entity_type, record, binding)
            state.expired = False

    def register_identity(self, obj: Entity) -> None:
        """Register one persistent object after collision-safe validation.

        Args:
            obj: Persistent object to register.
        """
        self._assert_session_access()
        self._state.register_identity(obj)

    def revive(self, obj: Entity) -> None:
        """Stage revival or cancel an inverse local transition.

        Args:
            obj: Persistent entity to transition.

        Raises:
            SgalchemyError: If `obj` is pending, already active without a
                staged delete, foreign, indeterminate, or owned elsewhere.
        """
        self._assert_session_access()
        stage_revive(self, obj)

    def rollback(self) -> None:
        """Restore dirty snapshots and discard staged local work without I/O.

        Rollback changes only local state and cannot undo provider writes.
        """
        self._assert_session_access()
        discard_local_state(_automatic_recovery_adapter(self))

    async def scalars(self, stmt: Select[_EntityT]) -> list[_EntityT]:
        """Execute one entity statement and its awaited relationship loads.

        A `joinedload()` may refresh target schema once and recompile before
        its single primary query. `selectinload()` performs bounded observed
        target reads. Async sessions reject `lazyload()` because attribute
        access cannot await I/O.

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
            SgalchemyError: If session access, registry ownership, returned
                identity data, or relationship target availability is invalid.
            Exception: Original schema-cache or provider exception, annotated
                and unwrapped.
        """
        async with self._operation_guard():
            if stmt.entity is None:
                raise TypeError(
                    'projection and summary statements use await session.execute(statement)'
                )
            type_name = self._state.require_owned_class(stmt.entity)
            if any(isinstance(option, LazyLoad) for option in stmt.loader_options):
                raise SgalchemyError(
                    'AsyncSession cannot await lazyload() attribute access; use '
                    'selectinload() or await session.load(...)'
                )
            relationship_options = normalize_relationship_loaders(stmt.loader_options)
            context = f'raised while executing entity statement {stmt}'
            binding = await self.load_binding(context)
            call, binding = await self._compile_scalars_call(stmt, binding, context)
            if call.entity_type != type_name:
                raise SgalchemyError('compiled entity type contradicts the owned statement class')
            records = await self.execute_call(call, binding)
            with self.hydration_scope():
                objects = [
                    self._state.hydrate_record(
                        call.entity_type,
                        record,
                        binding,
                        retired=bool(call.extra.get('retired_only', False)),
                    )
                    for record in records
                ]
            await execute_selectins(
                self,
                objects,
                [
                    option.field
                    for option in relationship_options
                    if isinstance(option, SelectinLoad)
                ],
                binding,
            )
            return cast('list[_EntityT]', objects)

    @property
    def unit_of_work(self) -> UnitOfWork:
        """Return this session's identity-based staging state.

        Returns:
            Package-internal staging state exposing immutable category
            snapshots.
        """
        self._assert_session_access()
        return self._unit_of_work

    async def _execute_commit_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
        progress: CallProgress,
    ) -> ConfirmedFieldRead | None:
        """Execute and reconcile one already-planned serial provider step."""
        if step.kind == 'revive':
            obj = step.entities[0]
            state = entity_state(obj)
            if state.id is None:
                raise RuntimeError('revive entity has no server id')
            self._active_step = step
            await self._engine.call_with_progress(
                'revive', (state.entity_type, state.id), {}, progress
            )
            self.register_identity(obj)
            apply_revive_result(self.unit_of_work, obj, outcome)
            return None
        if step.kind == 'batch':
            self._active_step = step
            requests = serialize_chunk(step.operations)
            results = await self._engine.call_with_progress('batch', (requests,), {}, progress)
            try:
                groups = apply_batch_results(
                    self,
                    step.operations,
                    results,
                    outcome,
                    self._state.identity_map.__setitem__,
                )
            except Exception as err:
                if not outcome.contains_committed(step.operations):
                    apply_batch_indeterminate(self, step.operations, outcome)
                commit_error = outcome.commit_error(err, step)
                add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
                raise commit_error from err
            self._active_step = None
            try:
                await self._recover_confirmed_fields(step, groups)
            except Exception:
                _LOGGER.exception('confirmed-write readback failed during %s', step.description)
            return None
        upload = step.require_upload()
        state = entity_state(upload.obj)
        if state.id is None:
            raise RuntimeError('upload entity has no server id')
        self._active_step = step
        await self._engine.call_with_progress(
            step.kind,
            (state.entity_type, state.id, os.fsdecode(upload.path)),
            {},
            progress,
        )
        return apply_upload_result(upload, outcome)


@dataclass(frozen=True)
class _AsyncCommitFailure:
    """Capture one ordinary async commit failure and its exact outcome state."""

    _err: Exception
    _step: CommitStep | None
    _outcome: CommitOutcome
    _steps: tuple[CommitStep, ...]
    _pending_upload_reads: list[ConfirmedFieldRead]
    _binding: RegistryBinding | None
    _wrap_ordinary_failure: bool
    _invoked: bool


def _automatic_recovery_adapter(session: 'AsyncSession') -> RecoveryAdapter:
    """Build guard-free local cleanup capabilities for one AsyncSession."""
    state = session._state
    unit_of_work = session._unit_of_work
    return RecoveryAdapter(
        unit_of_work=unit_of_work,
        detach_canceled=state.detach_canceled,
        discard_identities=state.discard_identities,
        identity_values=state.identity_values,
        owns_entity=state.owns_entity,
        cleanup_anchors=session._cleanup_anchors,
    )


def _release_local_state(session: AsyncSession, recovery: RecoveryAdapter) -> None:
    """Discard local work and retire any cancellation quarantine."""
    was_quarantined = session._quarantined_write is not None
    discard_local_state(recovery)
    session._state.close()
    session._active_step = None
    session._quarantined_write = None
    if was_quarantined:
        session._discarded_after_cancel = True


async def _route_async_commit_failure(
    session: 'AsyncSession',
    failure: _AsyncCommitFailure,
) -> Exception | None:
    """Return the failure that the normalized async commit should raise."""
    recover_upload_fields = session._recover_upload_fields
    step = failure._step
    if failure._invoked and step is not None and step.is_upload:
        try:
            upload = step.require_upload()
        except RuntimeError as error:
            return error
        ambiguous = apply_upload_indeterminate(session, upload, failure._outcome)
        await recover_upload_fields(
            [*failure._pending_upload_reads, ambiguous],
            failure._binding,
        )
        commit_error = failure._outcome.commit_error(failure._err, step)
        add_auth_note(failure._err, commit_error, AUTH_UPLOAD_NOTE)
        return commit_error
    await recover_upload_fields(failure._pending_upload_reads, failure._binding)
    route_exception = session._route_commit_exception
    routed = route_exception(
        failure._err, step, failure._outcome, failure._steps, invoked=failure._invoked
    )
    if routed is None and failure._wrap_ordinary_failure:
        commit_error = failure._outcome.commit_error(failure._err, step)
        if note := batch_failure_note(step, failure._outcome, invoked=failure._invoked):
            add_note_once(commit_error, note)
        add_auth_note(failure._err, commit_error, AUTH_RECOVERY_NOTE)
        return commit_error
    return routed


async def _execute_async_commit_plan(
    session: 'AsyncSession',
    plan: CommitPlan,
    batch_size: int | None,
    outcome: CommitOutcome,
    *,
    wrap_ordinary_failure: bool,
) -> None:
    """Execute one finalized asynchronous commit plan."""
    execute_step = session._execute_commit_step
    recover_upload_fields = session._recover_upload_fields
    record_interruption = session._record_commit_interruption
    steps = commit_steps(plan, batch_size)
    pending_upload_reads: list[ConfirmedFieldRead] = []
    progress = None
    try:
        for step in steps:
            if pending_upload_reads and not step.is_upload:
                await recover_upload_fields(pending_upload_reads, plan.binding)
                pending_upload_reads.clear()
            progress = CallProgress()
            read = await execute_step(step, outcome, progress)
            session._active_step = None
            if read is not None:
                pending_upload_reads.append(read)
        await recover_upload_fields(pending_upload_reads, plan.binding)
    except CommitError:
        step = session._active_step
        session._active_step = None
        await recover_upload_fields(
            pending_upload_reads if progress is None or not progress.invoked.is_set() else [],
            plan.binding,
        )
        if (
            progress is not None
            and progress.invoked.is_set()
            and step is not None
            and step.is_upload
        ):
            apply_upload_indeterminate(session, step.require_upload(), outcome)
        raise
    except Exception as err:
        step = session._active_step
        session._active_step = None
        routed = await _route_async_commit_failure(
            session,
            _AsyncCommitFailure(
                _err=err,
                _step=step,
                _outcome=outcome,
                _steps=steps,
                _pending_upload_reads=pending_upload_reads,
                _binding=plan.binding,
                _wrap_ordinary_failure=wrap_ordinary_failure,
                _invoked=progress is not None and progress.invoked.is_set(),
            ),
        )
        if routed is None:
            raise
        raise routed from err
    except BaseException as err:
        step = session._active_step
        session._active_step = None
        record_interruption(err, step, outcome, progress)
        raise


def automatic_cleanup(
    session: 'AsyncSession',
    *,
    task_name: str,
    cancelled: bool,
) -> None:
    """Release one completed task's AsyncSession state without provider I/O.

    Cleanup restores dirty snapshots, detaches entities, preserves quarantine,
    and warns before discarding unsubmitted work. Active borrowing defers the
    destructive step until the operation guard reaches depth zero. Failures are
    logged and never restore the task registry entry.

    Args:
        session: Scoped session owned by the completed task.
        task_name: Completed task name for non-secret diagnostics.
        cancelled: Whether the completed task was cancelled.
    """
    if session._automatic_cleanup_done:
        return
    if session._depth != 0:
        if not session._automatic_cleanup_pending:
            _LOGGER.warning(
                'automatic task cleanup deferred for cross-task borrowing: task=%r cancelled=%s',
                task_name,
                cancelled,
            )
            session._automatic_cleanup_details = (task_name, cancelled)
            session._automatic_cleanup_pending = True
        return

    session._automatic_cleanup_done = True
    session._automatic_cleanup_pending = False
    session._automatic_cleanup_details = ('', False)
    try:
        recovery = _automatic_recovery_adapter(session)
        dirty, pending, deletes, revives = count_unsubmitted_work(recovery)
        if dirty or pending or deletes or revives:
            _LOGGER.warning(
                '%s: task=%r cancelled=%s dirty=%d pending=%d deletes=%d revives=%d',
                _AUTOMATIC_CLEANUP_WARNING,
                task_name,
                cancelled,
                dirty,
                pending,
                deletes,
                revives,
            )
        _release_local_state(session, recovery)
    except BaseException:
        _LOGGER.exception(
            'automatic task cleanup failed after releasing task scope: task=%r cancelled=%s',
            task_name,
            cancelled,
        )


async def commit_async_session(
    session: 'AsyncSession',
    batch_size: int | None = None,
    *,
    wrap_ordinary_failure: bool,
) -> None:
    """Execute one normalized asynchronous commit.

    Factory commit-on-success mode wraps ordinary validation, planning, and
    execution failures before its caller closes the AsyncSession and discards
    the local retry queue. Session access and initial failure-evidence capture
    precede this wrapping boundary.

    Args:
        session: AsyncSession whose staged work is committed.
        batch_size: Optional positive maximum requests per batch call. A
            supplied limit must fit every indivisible operation group.
        wrap_ordinary_failure: Wrap ordinary validation, planning, and
            execution failures in `CommitError`.

    Raises:
        SgalchemyError: If session access is rejected before wrapping begins.
        CommitError: If an ordinary failure occurs within the wrapping boundary
            while wrapping is enabled, or the ordinary commit path reports
            partial outcome evidence.
        Exception: Original ordinary failure before wrapping begins or when
            wrapping is disabled.
        ValueError: When wrapping is disabled, if `batch_size` is neither a
            positive integer nor `None`, or is smaller than an indivisible
            operation group.
    """
    operation_guard = session._operation_guard
    async with operation_guard():
        outcome = CommitOutcome(
            committed=[],
            staged=(
                staged_intent_snapshot(session.unit_of_work, session._identity_values())
                if wrap_ordinary_failure
                else []
            ),
            indeterminate=[],
            withdrawn=[],
        )
        try:
            validate_batch_size(batch_size)
            plan_commit_call = session._plan_commit
            plan = await plan_commit_call()
            plan = await asyncio.to_thread(preflight_uploads, plan)
            outcome = CommitOutcome.from_plan(plan)
            await _execute_async_commit_plan(
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
