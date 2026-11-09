"""Manage asynchronous reads and task-owned entity identity."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Iterable, Mapping
from contextlib import asynccontextmanager
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
from sgalchemy._core._engine import Engine
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
    reject_outdated,
)
from sgalchemy._core._result import Result
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import Select
from sgalchemy._errors import CommitError, CompilationError, SchemaUnavailableError, SgalchemyError
from sgalchemy._orm._async_loaders import execute_selectins, load_relationship
from sgalchemy._orm._session_common import (
    annotate_read_error,
    commit_error_with_batch_status,
    projection_result,
    refresh_call,
    route_batch_failure,
    validate_query_payload,
)
from sgalchemy._orm._state import SessionState, validate_point_records
from sgalchemy._orm._uow import (
    CommitOutcome,
    CommitPlan,
    CommitStep,
    ConfirmedFieldRead,
    ConfirmedFieldReadGroup,
    RecoveryAdapter,
    UnitOfWork,
    accepted_batch_pending,
    accepted_batch_reported,
    add_accepted_batch_notes,
    apply_batch_indeterminate,
    apply_batch_results,
    apply_confirmed_field_readback,
    apply_revive_result,
    apply_upload_indeterminate,
    apply_upload_result,
    batch_accepted,
    build_commit_error,
    commit_steps,
    confirmed_field_read_call,
    confirmed_field_read_codes,
    count_unsubmitted_work,
    discard_local_state,
    finish_accepted_batch_for_cleanup,
    group_upload_field_reads,
    mark_indeterminate_entities,
    new_session_unit_of_work,
    plan_commit,
    preflight_uploads,
    prepare_local_cleanup,
    release_accepted_batch,
    retire_interrupted_step,
    retry_accepted_batch,
    schema_drift_error,
    serialize_chunk,
    stage_add,
    stage_delete,
    stage_revive,
    staged_intent_snapshot,
    unfinished_cleanup_note,
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
    `register_identity`, and `unit_of_work`, are package-internal orchestration
    hooks and may change without notice.

    If an operation raises `SgalchemyError` saying SG saved a batch but
    sgalchemy could not finish updating the session's objects, call
    `rollback()` or `close()`, then query the affected records again.
    """

    def _assert_not_quarantined(self) -> None:
        """Reject use after an indeterminate cancellation or its closing discard."""
        if self.discarded_after_cancel:
            raise SgalchemyError(
                'this AsyncSession was closed after an indeterminate write; '
                'create a new AsyncSession'
            )
        if self.quarantined_write is not None:
            raise SgalchemyError(
                f'AsyncSession has an indeterminate {self.quarantined_write}; '
                'reconcile the server outcome, close this session, and create '
                'a new AsyncSession'
            )

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
        """Reject quarantine and finish pending accepted bookkeeping."""
        self._assert_not_quarantined()
        if accepted_batch_pending(self.raw_unit_of_work):
            retry_accepted_batch(_automatic_recovery_adapter(self))

    def _assert_session_access(self) -> None:
        """Reject unusable-session access or another active task."""
        self._assert_task_access()
        self._assert_usable()

    def __init__(self, engine: AsyncEngine) -> None:
        """Bind a new empty identity map to one async engine.

        Args:
            engine: Async engine bound to the session.

        Raises:
            TypeError: If `engine` is an Engine.
        """
        if isinstance(engine, Engine):
            raise TypeError(
                'AsyncSession() requires an AsyncEngine, not an Engine; use Session(engine)'
            )
        self.active_step: CommitStep | None = None
        self.automatic_cleanup_details: tuple[str, bool] = ('', False)
        self.automatic_cleanup_done = False
        self.automatic_cleanup_pending = False
        self.cleanup_anchors: list[Entity] = []
        self.depth = 0
        self.discarded_after_cancel = False
        self._engine = engine
        self._owner: asyncio.Task[object] | None = None
        self.quarantined_write: str | None = None
        self.state = SessionState(
            self,
            engine.registry,
            engine.prepared_binding,
            engine.mark_schema_stale,
            self._assert_session_access,
        )
        self.raw_unit_of_work = new_session_unit_of_work()

    async def __aenter__(self) -> Self:
        """Return this session for async context-manager use."""
        self._assert_session_access()
        return self

    def _quarantine_submitted_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
    ) -> None:
        """Remove one submitted step from retry and quarantine its exact entities."""
        for obj in retire_interrupted_step(self, step):
            if all(candidate is not obj for candidate in self.cleanup_anchors):
                self.cleanup_anchors.append(obj)
        outcome.mark_indeterminate(step.items)
        mark_indeterminate_entities(step.entities)

    def add(self, obj: Entity) -> None:
        """Stage a pending create or attach a detached persistent entity.

        Later field assignments on attached entities are sent by `commit()`.
        Adding a persistent entity already owned by this session is a no-op.

        Args:
            obj: Schema-bound pending or persistent entity.

        Raises:
            SgalchemyError: If `obj` belongs to a different registry, is
                outdated or indeterminate, belongs to another open session, or
                collides with a different object loaded or staged for deletion
                or revival.
        """
        reject_outdated(obj)
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
        self.state.assert_identity_available(obj)

    def assert_same_registry(self, value: Entity | type[Entity]) -> None:
        """Reject a class or object owned by another registry or open session.

        Args:
            value: Entity class or instance to validate before local use.

        Raises:
            SgalchemyError: If `value` belongs to another registry or open
                session.
        """
        self._assert_session_access()
        self.state.assert_same_registry(value)

    def attach_for_staging(self, obj: Entity) -> None:
        """Attach one detached entity for local staging.

        Args:
            obj: Entity to attach after access and registry checks.

        Raises:
            SgalchemyError: If another open session owns `obj`.
        """
        self._assert_session_access()
        self.state.attach(obj)

    def delete(self, obj: Entity) -> None:
        """Stage retirement or cancel an inverse local transition.

        Args:
            obj: Pending or persistent entity to transition.

        Raises:
            SgalchemyError: If `obj` is foreign, unstaged, indeterminate, or
                owned elsewhere.
        """
        reject_outdated(obj)
        self._assert_session_access()
        stage_delete(self, obj)

    def detach_canceled(self, obj: Entity) -> None:
        """Detach a canceled object absent from this identity map.

        Args:
            obj: Entity removed from local staging.
        """
        self._assert_session_access()
        self.state.detach_canceled(obj)

    def expire(self, obj: Entity) -> None:
        """Mark one session-owned entity expired without I/O.

        Args:
            obj: Entity to expire.

        Raises:
            SgalchemyError: If this session does not own `obj`.
        """
        reject_outdated(obj)
        self._assert_session_access()
        self.state.expire(obj)

    def expire_all(self) -> None:
        """Mark every session-owned entity expired without I/O."""
        self._assert_session_access()
        self.state.expire_all()

    def identity_values(self) -> list[Entity]:
        """Return a snapshot of identity-mapped objects."""
        self._assert_session_access()
        return self.state.identity_values()

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
        return self.state.merge_dict(raw)

    @asynccontextmanager
    async def operation_guard(
        self,
        *,
        allow_quarantined: bool = False,
    ) -> AsyncGenerator[None]:
        """Reject overlapping use while permitting same-task helper nesting."""
        task = self._assert_task_access()
        if not allow_quarantined:
            self._assert_usable()
        self._owner = task
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if self.depth == 0:
                self._owner = None
                if self.automatic_cleanup_pending:
                    task_name, cancelled = self.automatic_cleanup_details
                    automatic_cleanup(
                        self,
                        task_name=task_name,
                        cancelled=cancelled,
                    )

    async def close(self) -> None:
        """Revert local edits, detach entities, and retire quarantine locally.

        Closing performs no provider I/O and cannot undo provider writes. A
        session closed after an indeterminate submitted write stays unusable.

        An object that shows as `Type(id=N, outdated)` could not be updated
        after SG saved its batch; any use raises `SgalchemyError`, so discard
        it and query its record again.
        """
        async with self.operation_guard(allow_quarantined=True):
            recovery = _automatic_recovery_adapter(self)
            prepare_local_cleanup(recovery, 'close()')
            _release_local_state(self, recovery)

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

    @property
    def engine(self) -> AsyncEngine:
        """Return the async engine bound to this session.

        Returns:
            Bound async engine.
        """
        return self._engine

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
    ) -> WireRecords: ...

    @overload
    async def execute_call(
        self,
        call: SummarizeCall,
    ) -> WireRecord: ...

    @overload
    async def execute_call(
        self,
        call: FindCall | SummarizeCall,
    ) -> QueryResult: ...

    async def execute_call(
        self,
        call: FindCall | SummarizeCall,
    ) -> QueryResult:
        """Execute compiled provider work with Session fault context.

        Args:
            call: Compiled find or summarize call.

        Returns:
            Provider payload in its documented outer shape.

        Raises:
            SgalchemyError: If the outer result shape contradicts the call.
            Exception: Original provider exception, annotated and unwrapped.
        """
        async with self.operation_guard():
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
        records = await self.execute_call(call)
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
        return self.state.hydrate_record(entity_type_name, record, binding, retired=retired)

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
        async with self.operation_guard():
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
        async with self.operation_guard():
            if stmt.entity is not None:
                raise TypeError('entity statements use await session.scalars(statement)')
            binding = await self.load_binding(f'raised while executing statement {stmt}')
            call = compile_statement(stmt, binding.schemas)
            if isinstance(call, SummarizeCall):
                return await self.execute_call(call)
            records = await self.execute_call(call)
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
        async with self.operation_guard():
            if type(entity_id) is not int or entity_id <= 0:
                raise SgalchemyError('entity id must be a positive integer id')
            type_name = self.state.require_owned_class(cls)
            existing = self.state.entity(type_name, entity_id)
            if existing is not None:
                self.state.require_owned_object(existing)
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
                self.state.hydrate_record(
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
        materialized = [objects] if isinstance(objects, Entity) else list(objects)
        for obj in materialized:
            reject_outdated(obj)
        async with self.operation_guard():
            if not materialized:
                return
            states = [self.state.require_owned_object(obj) for obj in materialized]
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

    async def plan_commit(self) -> CommitPlan:
        """Build one complete plan with bounded pre-write schema recovery.

        Schema preparation may perform provider and cache I/O. No provider
        writes are sent.

        Returns:
            Ordered commit plan built against one captured schema binding, or
            an empty plan with no binding when no work is staged.

        Raises:
            RetiredEntityWriteError: If a retired entity has staged writes
                without revival intent.
            SgalchemyError: If local validation fails or schema remains
                incompatible after bounded recovery.
            Exception: Original schema-preparation or provider failure,
                annotated and unwrapped.
        """
        binding = self._engine.prepared_binding()
        refreshed = False
        while True:
            try:
                return plan_commit(self, binding, self.state.identity_map)
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

    def record_commit_interruption(
        self,
        err: BaseException,
        step: CommitStep | None,
        outcome: CommitOutcome,
        progress: CallProgress | None,
    ) -> None:
        """Quarantine an invoked call or dispatched work that can outlive its waiter."""
        if progress is None or not (progress.invoked.is_set() or progress.dispatched.is_set()):
            return
        confirmed_batch = step is not None and batch_accepted(self.raw_unit_of_work, step, outcome)
        if step is None or confirmed_batch:
            return
        self._quarantine_submitted_step(step, outcome)
        self.quarantined_write = step.description
        add_note_once(
            err,
            f'sgalchemy: async commit interrupted during {step.description}; '
            'server outcome is indeterminate, this session will not retry it, '
            'and the session must be closed',
        )

    async def recover_upload_fields(
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
        reject_outdated(obj)
        async with self.operation_guard():
            state = self.state.require_owned_object(obj)
            if state.id is None:
                raise SgalchemyError('cannot refresh an entity without a server identity')
            context = f'raised while refreshing {state.entity_type}(id={state.id})'
            schema, binding = await self._require_type_schema(context, state.entity_type)
            call = refresh_call(state, schema)
            records = await self.execute_call(call)
            record = validate_point_records(
                records,
                state.entity_type,
                state.id,
                allow_absent=False,
            )
            self.state.hydrate_record(state.entity_type, record, binding)
            state.expired = False

    def register_identity(self, obj: Entity) -> None:
        """Register one persistent object after collision-safe validation.

        Args:
            obj: Persistent object to register.
        """
        self._assert_session_access()
        self.state.register_identity(obj)

    def revive(self, obj: Entity) -> None:
        """Stage revival or cancel an inverse local transition.

        Args:
            obj: Persistent entity to transition.

        Raises:
            SgalchemyError: If `obj` is pending, already active without a
                staged delete, foreign, indeterminate, or owned elsewhere.
        """
        reject_outdated(obj)
        self._assert_session_access()
        stage_revive(self, obj)

    def rollback(self) -> None:
        """Restore dirty snapshots and discard staged local work without I/O.

        Rollback changes only local state and cannot undo provider writes.

        An object that shows as `Type(id=N, outdated)` could not be updated
        after SG saved its batch; any use raises `SgalchemyError`, so discard
        it and query its record again.
        """
        self._assert_task_access()
        self._assert_not_quarantined()
        recovery = _automatic_recovery_adapter(self)
        prepare_local_cleanup(recovery, 'rollback()')
        discard_local_state(recovery)

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
        async with self.operation_guard():
            if stmt.entity is None:
                raise TypeError(
                    'projection and summary statements use await session.execute(statement)'
                )
            type_name = self.state.require_owned_class(stmt.entity)
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
            records = await self.execute_call(call)
            self._assert_session_access()
            objects = [
                self.state.hydrate_record(
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
            Package-internal staging state for this session.
        """
        self._assert_session_access()
        return self.raw_unit_of_work

    async def execute_commit_step(
        self,
        step: CommitStep,
        outcome: CommitOutcome,
        progress: CallProgress,
    ) -> ConfirmedFieldRead | None:
        """Execute and reconcile one already-planned serial provider step.

        Args:
            step: Validated revive, batch, or thumbnail-upload step.
            outcome: Commit partitions updated by successful reconciliation.
            progress: Fresh events recording dispatch and provider entry.

        Returns:
            Current-value read descriptor after an upload, or `None` after a
            batch or revive. Batch echo omissions are read back here.

        Raises:
            CommitError: If a successful batch cannot be reconciled safely.
            asyncio.CancelledError: If the awaiting task is cancelled.
            Exception: Other provider or local reconciliation failures.
        """
        if step.kind == 'revive':
            obj = step.entities[0]
            state = entity_state(obj)
            if state.id is None:
                raise RuntimeError('revive entity has no server id')
            self.active_step = step
            await self._engine.call_with_progress(
                'revive', (state.entity_type, state.id), {}, progress
            )
            self.register_identity(obj)
            apply_revive_result(self.unit_of_work, obj, outcome)
            return None
        if step.kind == 'batch':
            self.active_step = step
            requests = serialize_chunk(step.operations)
            results = await self._engine.call_with_progress('batch', (requests,), {}, progress)
            try:
                self._assert_session_access()
                groups = apply_batch_results(
                    self, _automatic_recovery_adapter(self), step, results, outcome
                )
            except Exception as err:
                if not batch_accepted(self.raw_unit_of_work, step, outcome):
                    apply_batch_indeterminate(self, step.operations, outcome)
                commit_error = build_commit_error(err, step, outcome, self.raw_unit_of_work)
                add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
                raise commit_error from err
            try:
                await self._recover_confirmed_fields(step, groups)
            except Exception:
                _LOGGER.exception('confirmed-write readback failed during %s', step.description)
            return None
        upload = step.require_upload()
        state = entity_state(upload.obj)
        if state.id is None:
            raise RuntimeError('upload entity has no server id')
        self.active_step = step
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

    err: Exception
    step: CommitStep | None
    outcome: CommitOutcome
    steps: tuple[CommitStep, ...]
    pending_upload_reads: list[ConfirmedFieldRead]
    binding: RegistryBinding | None
    wrap_ordinary_failure: bool
    invoked: bool


def _automatic_recovery_adapter(session: 'AsyncSession') -> RecoveryAdapter:
    """Build guard-free local cleanup capabilities for one AsyncSession."""
    state = session.state
    unit_of_work = session.raw_unit_of_work
    return RecoveryAdapter(
        unit_of_work=unit_of_work,
        detach_canceled=state.detach_canceled,
        discard_identities=state.discard_identities,
        identity_values=state.identity_values,
        install_identity=state.identity_map.__setitem__,
        owns_entity=state.owns_entity,
        cleanup_anchors=session.cleanup_anchors,
    )


def _release_local_state(session: AsyncSession, recovery: RecoveryAdapter) -> None:
    """Discard local work and retire any cancellation quarantine."""
    was_quarantined = session.quarantined_write is not None
    discard_local_state(recovery)
    session.state.close()
    session.active_step = None
    session.quarantined_write = None
    if was_quarantined:
        session.discarded_after_cancel = True


async def _route_async_commit_failure(
    session: 'AsyncSession',
    failure: _AsyncCommitFailure,
) -> Exception | None:
    """Return the failure that the normalized async commit should raise."""
    step = failure.step
    if failure.invoked and step is not None and step.is_upload:
        try:
            upload = step.require_upload()
        except RuntimeError as error:
            return error
        ambiguous = apply_upload_indeterminate(session, upload, failure.outcome)
        await session.recover_upload_fields(
            [*failure.pending_upload_reads, ambiguous],
            failure.binding,
        )
        commit_error = build_commit_error(
            failure.err, step, failure.outcome, session.raw_unit_of_work
        )
        add_auth_note(failure.err, commit_error, AUTH_UPLOAD_NOTE)
        return commit_error
    await session.recover_upload_fields(failure.pending_upload_reads, failure.binding)
    steps = failure.steps
    routed = route_batch_failure(
        failure.err,
        failure.outcome,
        session.raw_unit_of_work,
        sole_batch_step=steps[0] if len(steps) == 1 and steps[0].kind == 'batch' else None,
        invoked=failure.invoked,
        raw_passthrough=not failure.wrap_ordinary_failure,
    )
    if routed is None and failure.wrap_ordinary_failure:
        return commit_error_with_batch_status(
            failure.err,
            failure.outcome.current_step,
            failure.outcome,
            session.raw_unit_of_work,
            invoked=failure.invoked,
        )
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
    steps = commit_steps(plan, batch_size)
    pending_upload_reads: list[ConfirmedFieldRead] = []
    progress = None
    try:
        for step in steps:
            if pending_upload_reads and not step.is_upload:
                await session.recover_upload_fields(pending_upload_reads, plan.binding)
                pending_upload_reads.clear()
            progress = CallProgress()
            outcome.current_step = step
            read = await session.execute_commit_step(step, outcome, progress)
            session.active_step = None
            outcome.current_step = None
            if read is not None:
                pending_upload_reads.append(read)
        await session.recover_upload_fields(pending_upload_reads, plan.binding)
    except CommitError:
        step = session.active_step
        session.active_step = None
        await session.recover_upload_fields(
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
        step = session.active_step
        session.active_step = None
        routed = await _route_async_commit_failure(
            session,
            _AsyncCommitFailure(
                err=err,
                step=step,
                outcome=outcome,
                steps=steps,
                pending_upload_reads=pending_upload_reads,
                binding=plan.binding,
                wrap_ordinary_failure=wrap_ordinary_failure,
                invoked=progress is not None and progress.invoked.is_set(),
            ),
        )
        if routed is None:
            raise
        raise routed from err
    except BaseException as err:
        step = session.active_step
        session.active_step = None
        session.record_commit_interruption(err, step, outcome, progress)
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
    if session.automatic_cleanup_done:
        return
    if session.depth != 0:
        if not session.automatic_cleanup_pending:
            _LOGGER.warning(
                'automatic task cleanup deferred for cross-task borrowing: task=%r cancelled=%s',
                task_name,
                cancelled,
            )
            session.automatic_cleanup_details = (task_name, cancelled)
            session.automatic_cleanup_pending = True
        return

    session.automatic_cleanup_done = True
    session.automatic_cleanup_pending = False
    session.automatic_cleanup_details = ('', False)
    try:
        recovery = _automatic_recovery_adapter(session)
        release = finish_accepted_batch_for_cleanup(recovery, 'automatic task cleanup')
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
        if release:
            release_accepted_batch(recovery)
        _release_local_state(session, recovery)
    except BaseException:
        if accepted_batch_pending(session.raw_unit_of_work):
            _LOGGER.exception(
                'automatic task cleanup failed after releasing task scope: '
                'task=%r cancelled=%s; %s',
                task_name,
                cancelled,
                unfinished_cleanup_note('automatic task cleanup'),
            )
        else:
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
    async with session.operation_guard():
        outcome = CommitOutcome(
            committed=[],
            staged=(
                staged_intent_snapshot(session.unit_of_work, session.identity_values())
                if wrap_ordinary_failure
                else []
            ),
            indeterminate=[],
            withdrawn=[],
        )
        try:
            validate_batch_size(batch_size)
            plan = await session.plan_commit()
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
            step = outcome.current_step
            accepted = accepted_batch_reported(session.raw_unit_of_work, step, outcome)
            if not wrap_ordinary_failure and not accepted:
                raise
            commit_error = build_commit_error(
                err, step if accepted else None, outcome, session.raw_unit_of_work
            )
            add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
            raise commit_error from err
        except BaseException as err:
            step = outcome.current_step
            if accepted_batch_reported(session.raw_unit_of_work, step, outcome):
                add_accepted_batch_notes(err, session.raw_unit_of_work, step)
            raise
