"""Identity-based staging and pure unit-of-work planning."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self, TypeAlias, cast

from sgalchemy._core._compiler import FindCall
from sgalchemy._core._registry import (
    Entity,
    EntityState,
    RegistryBinding,
    container_children,
    entity_state,
    initialize_entity,
)
from sgalchemy._core._registry import entity_type as reflected_entity_type
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._errors import CommitError, RetiredEntityWriteError, SgalchemyError
from sgalchemy._shotgun import (
    THUMBNAIL_COUPLED_CLEAR_FIELD,
    THUMBNAIL_PRIMARY_FIELD,
    THUMBNAIL_UPLOAD_METHODS,
    BatchItem,
    WireRecord,
    WireRecords,
    WireTraversalContainer,
)

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from sgalchemy.schema._model import EntitySchema

_BATCH_OUTCOME_UNKNOWN_NOTE = (
    'sgalchemy: the failed batch call started but returned no usable result; '
    'its server outcome is unknown'
)
_BATCH_UNSENT_NOTE = 'sgalchemy: the failed batch was not sent to the provider'
SOLE_BATCH_STAGED_NOTE = (
    'sgalchemy: single-batch commit; all operations remain staged in this session'
)

_CYCLIC_CONTAINER_NOTE = (
    'container value contains a reference cycle; cyclic containers cannot be '
    'represented as wire data, so break the cycle before staging or serializing'
)

_LOGGER = logging.getLogger('sgalchemy.session')
_PlanItem: TypeAlias = 'Entity | _Op | UploadPlanItem'
_StagingCategory: TypeAlias = Literal['new', 'deleted', 'revived']
_UploadPath: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]


class BatchReconciliationError(SgalchemyError):
    """Batch response cannot map safely; reconcile through a fresh session."""


@dataclass(frozen=True)
class ConfirmedFieldRead:
    """Describe exact entity fields needing a bounded current-value read.

    Omitted-field diagnostics may include fields resolved deterministically.
    Ambiguous uploads retain uncertain write provenance after recovery.
    """

    _obj: Entity
    _field_codes: tuple[str, ...]
    _missing_field_codes: tuple[str, ...]
    _unverified: bool = False


@dataclass(frozen=True)
class ConfirmedFieldReadGroup:
    """Group compatible entity fields for one bounded current-value read.

    Attributes:
        entity_type: Canonical type shared by every read.
        retired_only: Whether to query retired entities.
        binding: Schema snapshot used to hydrate the returned values.
    """

    entity_type: str
    retired_only: bool
    binding: RegistryBinding | None
    _reads: tuple[ConfirmedFieldRead, ...]


@dataclass
class _Op:
    """One create, update, or delete operation in a commit plan."""

    request_type: Literal['create', 'update', 'delete']
    entity_type: str
    entity_id: int | None
    data: dict[str, Any]
    obj: Entity
    binding: RegistryBinding | None = None


class UploadPlanItem(NamedTuple):
    """Describe one deferred thumbnail upload in a commit plan.

    Attributes:
        obj: Exact entity receiving the upload.
        field_code: Provider image field to update.
        path: Local image path submitted to the provider.
    """

    obj: Entity
    field_code: str
    path: _UploadPath


@dataclass
class CommitPlan:
    """Carry ordered local work for the commit executor.

    Revivals precede dependency-ordered batch waves and thumbnail uploads.
    Final retirements follow uploads and their readback.

    Attributes:
        binding: Captured schema generation, or `None` for an empty plan.
    """

    _revives: list[Entity]
    _waves: list[list[_Op]]
    _uploads: list[UploadPlanItem]
    binding: RegistryBinding | None
    _final_retirements: list[_Op] = field(default_factory=list[_Op])


@dataclass(frozen=True)
class CommitStep:
    """Describe one serial provider call and its exact affected work.

    Attributes:
        kind: Provider operation category.
        description: Human-readable phase context for diagnostics.
        entities: Exact entity objects affected by this call.
        items: Exact plan objects tracked in outcome partitions.
        operations: Batch operations when `kind` is `batch`.
    """

    kind: Literal[
        'revive',
        'batch',
        'upload_thumbnail',
        'upload_filmstrip_thumbnail',
    ]
    description: str
    entities: tuple[Entity, ...]
    items: tuple[_PlanItem, ...]
    operations: tuple[_Op, ...] = ()
    _upload: UploadPlanItem | None = None

    @property
    def is_upload(self) -> bool:
        """Return whether this step submits a thumbnail upload."""
        return self.kind in {'upload_thumbnail', 'upload_filmstrip_thumbnail'}

    def require_upload(self) -> UploadPlanItem:
        """Return this upload step's descriptor.

        Raises:
            RuntimeError: If the step has no upload descriptor.
        """
        if self._upload is None:
            raise RuntimeError('upload commit step requires upload data')
        return self._upload


@dataclass
class CommitOutcome:
    """Four disjoint failure-time partitions for one planned commit.

    Only `staged` items remain eligible for automatic retry. `committed` and
    `indeterminate` items must not be retried; `withdrawn` items require new
    explicit user intent.
    """

    committed: list[_PlanItem]
    staged: list[_PlanItem]
    indeterminate: list[_PlanItem]
    withdrawn: list[_PlanItem]

    def _remove_staged(self, items: tuple[_PlanItem, ...]) -> None:
        """Remove exact plan objects from the staged partition in linear time."""
        item_ids = {id(item) for item in items}
        self.staged[:] = [candidate for candidate in self.staged if id(candidate) not in item_ids]

    def _confirm(self, items: tuple[_PlanItem, ...]) -> None:
        """Move exact plan items from staged to server-confirmed."""
        self._remove_staged(items)
        self.committed.extend(items)

    def _staged_items_for(
        self,
        entities: tuple[Entity, ...],
    ) -> tuple[_PlanItem, ...]:
        """Return staged plan items owned by any exact affected entity."""
        return tuple(
            candidate
            for candidate in self.staged
            if any(_plan_item_owner(candidate) is entity for entity in entities)
        )

    def _withdraw(self, items: tuple[_PlanItem, ...]) -> None:
        """Remove exact never-submitted follow-ups from automatic retry."""
        self._remove_staged(items)
        self.withdrawn.extend(items)

    def commit_error(self, err: Exception, step: CommitStep | None) -> CommitError:
        """Build the public error with partition-specific safe-action notes.

        Args:
            err: Backend exception that ended the commit.
            step: Step active when `err` was raised, if submission had begun.

        Returns:
            Public error containing snapshots of all four partitions.
        """
        error = CommitError(
            str(err),
            committed=list(self.committed),
            staged=list(self.staged),
            indeterminate=list(self.indeterminate),
            withdrawn=list(self.withdrawn),
        )
        if self.staged:
            error.add_note(
                'sgalchemy: if this session remains open and usable, a later commit '
                'may retry only the staged partition'
            )
        if self.indeterminate:
            if any(isinstance(item, _Op) for item in self.indeterminate):
                error.add_note(
                    'sgalchemy: discard affected objects and reconcile server state '
                    'through a fresh query in a new session'
                )
            else:
                fields = [
                    (entity_state(item.obj), item.field_code)
                    for item in self.indeterminate
                    if isinstance(item, UploadPlanItem)
                ]
                has_unverified = any(code in state.unverified_writes for state, code in fields)
                has_refresh_required = any(
                    code in state.refresh_required for state, code in fields
                )
                if has_unverified:
                    error.add_note(
                        'sgalchemy: each affected thumbnail field has a readable provider '
                        'value marked unverified; refresh it to accept the current server '
                        'state, or explicitly reassign it as new intent'
                    )
                if has_refresh_required or not has_unverified:
                    error.add_note(
                        'sgalchemy: refresh each affected thumbnail field before reading '
                        'it, or explicitly reassign it as new intent'
                    )
        if self.withdrawn:
            error.add_note(
                'sgalchemy: explicitly restage withdrawn work only after reconciling the '
                'earlier submitted outcome'
            )
        if step is not None:
            error.add_note(f'sgalchemy: failed during {step.description}')
        return error

    def contains_committed(self, items: tuple[_PlanItem, ...]) -> bool:
        """Return whether every exact item is in the committed partition.

        Args:
            items: Exact plan objects whose confirmation is required.

        Returns:
            Whether every item is committed by object identity.
        """
        return all(any(candidate is item for candidate in self.committed) for item in items)

    @classmethod
    def from_plan(cls, plan: CommitPlan) -> Self:
        """Start with every planned item staged in deterministic execution order.

        Args:
            plan: Pure commit plan whose items have not been submitted.

        Returns:
            New outcome with all plan items in `staged`.
        """
        items: list[_PlanItem] = [*plan._revives]
        items.extend(op for wave in plan._waves for op in wave)
        items.extend(plan._uploads)
        items.extend(plan._final_retirements)
        return cls(committed=[], staged=items, indeterminate=[], withdrawn=[])

    def mark_indeterminate(self, items: tuple[_PlanItem, ...]) -> None:
        """Move exact unsafe items from staged to indeterminate.

        Args:
            items: Exact plan objects unsafe to retry automatically.
        """
        self._remove_staged(items)
        self.indeterminate.extend(items)


def _apply_deterministic_thumbnail_clear(operation: _Op) -> None:
    """Apply authoritative local values for one confirmed thumbnail clear."""
    if (
        THUMBNAIL_PRIMARY_FIELD not in operation.data
        or operation.data[THUMBNAIL_PRIMARY_FIELD] is not None
    ):
        return
    state = entity_state(operation.obj)
    state.accept_field_value(THUMBNAIL_PRIMARY_FIELD, None)
    if THUMBNAIL_COUPLED_CLEAR_FIELD in state.dirty:
        return
    state.accept_unstaged_value(THUMBNAIL_COUPLED_CLEAR_FIELD, None)


def _batch_result_structure_error(operation: _Op, result: object) -> str | None:
    """Return a structural mismatch message for one positioned batch result."""
    if operation.request_type == 'delete':
        return (
            None
            if type(result) is bool
            else 'batch delete result must be a boolean matching its request'
        )
    if not isinstance(result, dict):
        return f'batch {operation.request_type} result must be a dictionary matching its request'
    record = cast('WireRecord', result)
    wire_type = record.get('type')
    entity_id = record.get('id')
    if type(wire_type) is not str or wire_type != operation.entity_type:
        error = (
            f'batch result type {wire_type!r} contradicts '
            f'{operation.request_type} request type {operation.entity_type!r}'
        )
    elif type(entity_id) is not int or entity_id <= 0:
        error = f'batch {operation.request_type} result requires a positive integer id'
    elif operation.request_type == 'update' and entity_id != operation.entity_id:
        error = (
            f'batch update result id {entity_id!r} contradicts request id {operation.entity_id!r}'
        )
    else:
        error = None
    return error


def _clear_plan_item_fields(item: _PlanItem) -> None:
    """Clear retry provenance for fields owned by one unsafe plan item."""
    if isinstance(item, _Op):
        obj = item.obj
        field_codes = item.data
    elif isinstance(item, UploadPlanItem):
        obj = item.obj
        field_codes = (item.field_code,)
    else:
        return
    state = entity_state(obj)
    for field_code in field_codes:
        state.discard_field_provenance(field_code)


def _contains_identity(items: list[Entity], obj: Entity) -> bool:
    """Return whether a staging list contains the exact object."""
    return any(item is obj for item in items)


def _append_entity_references(
    value: object,
    references: list[Entity],
    seen_containers: set[int],
) -> None:
    """Append unique nested entity values while tolerating container cycles."""
    if isinstance(value, Entity):
        if not _contains_identity(references, value):
            references.append(value)
        return
    children = container_children(value)
    if children is None:
        return
    container = cast('WireTraversalContainer', value)
    container_id = id(container)
    if container_id in seen_containers:
        return
    seen_containers.add(container_id)
    for item in children:
        _append_entity_references(item, references, seen_containers)


def _degrade_submitted_echo(operation: _Op) -> None:
    """Remove untrusted submitted values after one contained hydration fault."""
    state = entity_state(operation.obj)
    for field_code in operation.data:
        if field_code == THUMBNAIL_PRIMARY_FIELD and operation.data[field_code] is None:
            continue
        state.mark_field_unknown(field_code)


def _discard_missing_echo_fields(operation: _Op, record: WireRecord) -> None:
    """Mark only nondeterministic submitted fields omitted from one echo."""
    state = entity_state(operation.obj)
    for field_code in operation.data.keys() - record.keys():
        if field_code == THUMBNAIL_PRIMARY_FIELD and operation.data[field_code] is None:
            continue
        state.mark_field_unknown(field_code)


def _entity_references(value: object) -> list[Entity]:
    """Return unique entity values nested in supported containers."""
    references: list[Entity] = []
    _append_entity_references(value, references, set())
    return references


def _group_confirmed_field_reads(
    reads: list[tuple[str, bool, RegistryBinding | None, ConfirmedFieldRead]],
) -> tuple[ConfirmedFieldReadGroup, ...]:
    """Group one batch step's readbacks by type and retirement mode."""
    grouped: dict[
        tuple[str, bool],
        tuple[RegistryBinding | None, list[ConfirmedFieldRead]],
    ] = {}
    for entity_type, retired_only, binding, read in reads:
        key = (entity_type, retired_only)
        if key not in grouped:
            grouped[key] = (binding, [])
        grouped[key][1].append(read)
    return tuple(
        ConfirmedFieldReadGroup(
            entity_type=entity_type,
            retired_only=retired_only,
            binding=binding,
            _reads=tuple(group_reads),
        )
        for (entity_type, retired_only), (binding, group_reads) in grouped.items()
    )


def _indivisible_operation_groups(operations: list[_Op]) -> list[list[_Op]]:
    """Group adjacent same-owner updates and deletes without equality checks."""
    groups: list[list[_Op]] = []
    index = 0
    while index < len(operations):
        operation = operations[index]
        if (
            operation.request_type == 'update'
            and index + 1 < len(operations)
            and operations[index + 1].request_type == 'delete'
            and operations[index + 1].obj is operation.obj
        ):
            groups.append([operation, operations[index + 1]])
            index += 2
            continue
        groups.append([operation])
        index += 1
    return groups


def _new_entity_id_set() -> set[int]:
    """Return one independently mutable exact-object ID set."""
    return set()


def _new_entity_list() -> list[Entity]:
    """Return one independently mutable entity staging list."""
    return []


def _normalize_upload_path(path: _UploadPath) -> str:
    """Return one normalized readable regular-file path."""
    normalized = os.fspath(path)
    if normalized in {'', b''}:
        raise ValueError('not a nonempty regular file')
    nonblocking = getattr(os, 'O_NONBLOCK', 0)
    if not nonblocking and not Path(os.fsdecode(normalized)).is_file():
        raise ValueError('not a nonempty regular file')
    descriptor = os.open(normalized, os.O_RDONLY | nonblocking)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size == 0:
            raise ValueError('not a nonempty regular file')
    finally:
        os.close(descriptor)
    return os.fsdecode(normalized)


def _operation_dependencies(
    groups: list[list[_Op]],
) -> tuple[list[set[int]], list[set[int]]]:
    """Build dependency and dependent indexes for pending-create references."""
    pending_indices = {
        id(operation.obj): index
        for index, group in enumerate(groups)
        for operation in group
        if operation.request_type == 'create'
    }
    dependencies: list[set[int]] = [set() for _group in groups]
    dependents: list[set[int]] = [set() for _group in groups]
    for group_index, group in enumerate(groups):
        for operation in group:
            for value in operation.data.values():
                for reference in _entity_references(value):
                    if entity_state(reference).id is not None:
                        continue
                    dependency_index = pending_indices.get(id(reference))
                    if dependency_index is None:
                        continue
                    dependencies[group_index].add(dependency_index)
                    dependents[dependency_index].add(group_index)
    return dependencies, dependents


def _operation_fields(
    schema: EntitySchema,
    obj: Entity,
) -> tuple[dict[str, Any], list[UploadPlanItem]]:
    """Partition preflighted dirty fields into batch data and deferred uploads."""
    state = entity_state(obj)
    data: dict[str, Any] = {}
    uploads: list[UploadPlanItem] = []
    for field_code, value in state.dirty.items():
        field_schema = schema.fields[field_code]
        if field_code in THUMBNAIL_UPLOAD_METHODS:
            if field_code == THUMBNAIL_PRIMARY_FIELD and value is None:
                data[field_code] = None
            else:
                uploads.append(UploadPlanItem(obj, field_code, value))
        elif field_schema.data_type == 'image':
            raise SgalchemyError(
                f'unsupported image upload field code {state.entity_type}.{field_code}'
            )
        else:
            data[field_code] = value
    return data, uploads


def _operation_waves(operations: list[_Op]) -> list[list[_Op]]:
    """Topologically partition operations around pending-create dependencies."""
    groups = _indivisible_operation_groups(operations)
    dependencies, dependents = _operation_dependencies(groups)
    ready = [index for index, values in enumerate(dependencies) if not values]
    processed: set[int] = set()
    waves: list[list[_Op]] = []
    while ready:
        waves.append([operation for index in ready for operation in groups[index]])
        processed.update(ready)
        next_ready: list[int] = []
        for completed_index in ready:
            for dependent_index in dependents[completed_index]:
                dependencies[dependent_index].discard(completed_index)
                if not dependencies[dependent_index] and dependent_index not in processed:
                    next_ready.append(dependent_index)
        ready = sorted(set(next_ready))
    if len(processed) != len(groups):
        raise SgalchemyError('pending create dependency cycle prevents commit planning')
    return waves


def _persistent_write_operations(
    write_candidates: list[Entity],
    deleted: Collection[Entity],
    schemas: Mapping[str, EntitySchema],
    binding: RegistryBinding,
) -> tuple[list[_Op], list[UploadPlanItem], list[Entity]]:
    """Plan persistent updates, uploads, and ordinary retirement owner groups."""
    operations: list[_Op] = []
    uploads: list[UploadPlanItem] = []
    normal_retirement_owners: list[Entity] = []
    deleted_ids = {id(obj) for obj in deleted}
    for obj in write_candidates:
        state = entity_state(obj)
        if not state.dirty:
            continue
        data, entity_uploads = _operation_fields(schemas[state.entity_type], obj)
        uploads.extend(entity_uploads)
        if data:
            operations.append(_Op('update', state.entity_type, state.id, data, obj, binding))
        if id(obj) in deleted_ids and not entity_uploads:
            operations.append(_Op('delete', state.entity_type, state.id, {}, obj, binding))
            normal_retirement_owners.append(obj)
    return operations, uploads, normal_retirement_owners


def _plan_item_owner(item: _PlanItem) -> Entity:
    """Return the entity that owns one commit plan item."""
    if isinstance(item, Entity):
        return item
    if isinstance(item, _Op):
        return item.obj
    return item.obj


def _remaining_retirement_operations(
    deleted: Collection[Entity],
    uploads: list[UploadPlanItem],
    normal_retirement_owners: list[Entity],
    binding: RegistryBinding,
) -> tuple[list[_Op], list[_Op]]:
    """Partition clean retirements from upload-dependent final retirements."""
    normal: list[_Op] = []
    final: list[_Op] = []
    normal_retirement_owner_ids = {id(obj) for obj in normal_retirement_owners}
    upload_owner_ids = {id(upload.obj) for upload in uploads}
    for obj in deleted:
        state = entity_state(obj)
        retirement = _Op('delete', state.entity_type, state.id, {}, obj, binding)
        if id(obj) in upload_owner_ids:
            final.append(retirement)
        elif id(obj) not in normal_retirement_owner_ids:
            normal.append(retirement)
    return normal, final


def _remove_identity(items: list[Entity], obj: Entity) -> bool:
    """Remove the exact object and return whether it was present."""
    original_length = len(items)
    items[:] = [item for item in items if item is not obj]
    return len(items) != original_length


def _require_dirty_type(state: EntityState, field_code: str) -> str:
    """Return required local write provenance or report local corruption."""
    assigned_type = state.dirty_types.get(field_code)
    if assigned_type is None:
        raise SgalchemyError(
            f'staged write classification is missing for '
            f'{state.entity_type}.{field_code}; call session.rollback() and reassign '
            'the field before commit()'
        )
    return assigned_type


def _require_entity_schema(
    schemas: Mapping[str, EntitySchema],
    obj: Entity,
) -> EntitySchema:
    """Return the current entity schema or report recoverable drift."""
    state = entity_state(obj)
    schema = schemas.get(state.entity_type)
    if schema is None:
        raise SchemaDrift(state.entity_type, '*', 'entity type')
    return schema


def _restore_withdrawn_upload(item: _PlanItem) -> bool:
    """Restore one never-submitted upload field without retaining its path."""
    if not isinstance(item, UploadPlanItem):
        return False
    state = entity_state(item.obj)
    state.restore_field_from_snapshot(item.field_code)
    return True


def _retire_batch_operation_provenance(
    operation: _Op,
) -> None:
    """Retire one confirmed operation's submitted field provenance."""
    state = entity_state(operation.obj)
    for field_code in operation.data:
        state.retire_confirmed_write(field_code)


def _unique_entities(items: Iterable[Entity]) -> list[Entity]:
    """Return the first occurrence of each exact entity object."""
    seen: set[int] = set()
    unique: list[Entity] = []
    for item in items:
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item)
    return unique


def _dirty_participants(
    write_candidates: list[Entity],
    pending: Collection[Entity],
) -> list[Entity]:
    """Return every unique dirty object participating in this commit."""
    candidates = _unique_entities([*write_candidates, *pending])
    return [obj for obj in candidates if entity_state(obj).dirty]


def _validate_dirty_fields(schema: EntitySchema, obj: Entity) -> None:
    """Validate every dirty field against its current write classification."""
    state = entity_state(obj)
    for field_code in state.dirty:
        assigned_type = _require_dirty_type(state, field_code)
        field_schema = schema.fields.get(field_code)
        if field_schema is None:
            raise SchemaDrift(state.entity_type, field_code, 'present field')
        if not field_schema.api_readable:
            raise SchemaDrift(state.entity_type, field_code, 'API-writable field')
        if assigned_type != field_schema.data_type:
            raise SchemaDrift(
                state.entity_type,
                field_code,
                f'assigned type {assigned_type!r}',
            )
        if not field_schema.editable and field_code not in THUMBNAIL_UPLOAD_METHODS:
            raise SchemaDrift(state.entity_type, field_code, 'editable field')


def _validate_retired_write_intent(
    dirty_participants: list[Entity],
    revived: Collection[Entity],
) -> None:
    """Reject dirty retired objects without exact-object revival intent."""
    revived_ids = {id(obj) for obj in revived}
    for obj in dirty_participants:
        state = entity_state(obj)
        if state.deleted and state.dirty and id(obj) not in revived_ids:
            raise RetiredEntityWriteError(
                f'{state.entity_type}(id={state.id}) is retired and has staged writes; '
                'call session.revive(obj) to write them or session.rollback() to discard them'
            )


def _validate_schema_participants(
    schemas: Mapping[str, EntitySchema],
    participants: list[Entity],
) -> None:
    """Validate participating objects against one captured schema binding."""
    for obj in participants:
        schema = _require_entity_schema(schemas, obj)
        if entity_state(obj).dirty:
            _validate_dirty_fields(schema, obj)


def _validated_confirmed_field_records(
    group: ConfirmedFieldReadGroup,
    records: WireRecords,
) -> tuple[tuple[ConfirmedFieldRead, WireRecord], ...]:
    """Match only complete canonical readback rows without mutating state."""
    matched: list[tuple[ConfirmedFieldRead, WireRecord]] = []
    for read in group._reads:
        state = entity_state(read._obj)
        candidates = [
            record
            for record in records
            if type(record.get('type')) is str
            and record.get('type') == group.entity_type
            and type(record.get('id')) is int
            and record.get('id') == state.id
        ]
        if len(candidates) != 1:
            continue
        record = candidates[0]
        if any(field_code not in record for field_code in read._field_codes):
            continue
        matched.append((read, record))
    return tuple(matched)


def _wire_value_recursive(value: object, active: set[int]) -> object:
    """Serialize one value while rejecting active container cycles."""
    if isinstance(value, Entity):
        state = entity_state(value)
        link: WireRecord = {'type': state.entity_type, 'id': state.id}
        return link
    children = container_children(value)
    if children is None:
        return value
    container = cast('WireTraversalContainer', value)
    container_id = id(container)
    if container_id in active:
        raise SgalchemyError(_CYCLIC_CONTAINER_NOTE)
    active.add(container_id)
    try:
        if isinstance(container, Mapping):
            return {key: _wire_value_recursive(item, active) for key, item in container.items()}
        if isinstance(container, list):
            return [_wire_value_recursive(item, active) for item in children]
        return tuple(_wire_value_recursive(item, active) for item in children)
    finally:
        active.discard(container_id)


def _wire_value(value: object) -> object:
    """Recursively convert entity objects into canonical relationship links."""
    return _wire_value_recursive(value, set())


def apply_upload_result(
    upload: UploadPlanItem,
    outcome: CommitOutcome,
) -> ConfirmedFieldRead:
    """Retire one upload's local path before confirming its outcome.

    Args:
        upload: Exact entity, field code, and path submitted to the provider.
        outcome: Commit partitions to update after local reconciliation.

    Returns:
        Exact current-value read for the confirmed field.
    """
    state = entity_state(upload.obj)
    state.retire_upload_field(upload.field_code)
    outcome._confirm((upload,))
    return ConfirmedFieldRead(
        _obj=upload.obj,
        _field_codes=(upload.field_code,),
        _missing_field_codes=(upload.field_code,),
    )


def assert_entity_identity_consistent(
    obj: Entity,
    mapped_keys: Collection[tuple[str, int]],
    *,
    require_persistent: bool = False,
) -> None:
    """Reject reflected-type, server-ID, and exact-object key contradictions.

    Args:
        obj: Entity whose current identity must agree with its class.
        mapped_keys: Every known identity-map key for this exact object.
        require_persistent: Whether a pending identity is forbidden.

    Raises:
        SgalchemyError: If the identity or any mapped key is inconsistent.
    """
    state = entity_state(obj)
    reflected_type = reflected_entity_type(type(obj))
    if state.entity_type != reflected_type:
        raise SgalchemyError(
            f'entity state type {state.entity_type!r} contradicts reflected class '
            f'entity type {reflected_type!r}'
        )
    if state.id is None and not require_persistent:
        if mapped_keys:
            raise SgalchemyError(
                'pending entity cannot appear under a persistent identity map key'
            )
        return
    if type(state.id) is not int or state.id <= 0:
        raise SgalchemyError('staged persistent object requires a positive integer id')
    canonical_key = (state.entity_type, state.id)
    if any(key != canonical_key for key in mapped_keys):
        raise SgalchemyError(
            f'identity map key for the exact object must equal canonical '
            f'{canonical_key!r}; found {mapped_keys!r}'
        )


@dataclass
class UnitOfWork:
    """Expose immutable snapshots of one session's staging categories.

    Use session staging methods to maintain ownership, exact-object uniqueness,
    and mutually exclusive categories. Only this module mutates the underlying
    collections and their membership indexes.

    Attributes:
        new: Owned active pending entities staged for creation.
        deleted: Owned active persistent entities staged for retirement.
        revived: Owned retired persistent entities staged for revival.
    """

    _new: list[Entity] = field(default_factory=_new_entity_list, init=False, repr=False)
    _deleted: list[Entity] = field(default_factory=_new_entity_list, init=False, repr=False)
    _revived: list[Entity] = field(default_factory=_new_entity_list, init=False, repr=False)

    @property
    def deleted(self) -> tuple[Entity, ...]:
        """Return the ordered snapshot of staged retirements.

        Returns:
            Exact objects staged for deletion at the time of access.
        """
        return tuple(self._deleted)

    @property
    def new(self) -> tuple[Entity, ...]:
        """Return the ordered snapshot of staged creates.

        Returns:
            Exact objects staged for creation at the time of access.
        """
        return tuple(self._new)

    @property
    def revived(self) -> tuple[Entity, ...]:
        """Return the ordered snapshot of staged revivals.

        Returns:
            Exact objects staged for revival at the time of access.
        """
        return tuple(self._revived)


@dataclass
class _SessionUnitOfWork(UnitOfWork):
    """Session-only staging mutations maintaining exact-object membership."""

    _new_ids: set[int] = field(default_factory=_new_entity_id_set, init=False, repr=False)
    _deleted_ids: set[int] = field(default_factory=_new_entity_id_set, init=False, repr=False)
    _revived_ids: set[int] = field(default_factory=_new_entity_id_set, init=False, repr=False)

    def _staged_category(
        self,
        category: _StagingCategory,
    ) -> tuple[list[Entity], set[int]]:
        """Return the authoritative list and derived IDs for one category."""
        if category == 'new':
            return self._new, self._new_ids
        if category == 'deleted':
            return self._deleted, self._deleted_ids
        return self._revived, self._revived_ids

    def _append_staged(self, category: _StagingCategory, obj: Entity) -> bool:
        """Append an exact object if absent from one staging category."""
        staged, staged_ids = self._staged_category(category)
        obj_id = id(obj)
        if obj_id in staged_ids:
            return False
        staged.append(obj)
        staged_ids.add(obj_id)
        return True

    def _clear_staged(self, category: _StagingCategory) -> None:
        """Clear one authoritative staging list and its derived IDs."""
        staged, staged_ids = self._staged_category(category)
        staged.clear()
        staged_ids.clear()

    def _contains_staged(self, category: _StagingCategory, obj: Entity) -> bool:
        """Return whether one category contains the exact staged object."""
        _staged, staged_ids = self._staged_category(category)
        return id(obj) in staged_ids

    def _discard_staged(self, category: _StagingCategory, obj: Entity) -> bool:
        """Discard an exact object from one staging category."""
        staged, staged_ids = self._staged_category(category)
        obj_id = id(obj)
        if obj_id not in staged_ids:
            return False
        removed = _remove_identity(staged, obj)
        staged_ids.discard(obj_id)
        return removed

    def _discard_staged_many(
        self, category: _StagingCategory, objects: Collection[Entity]
    ) -> None:
        """Compact one staging list in place for exact confirmed objects."""
        if not objects:
            return
        staged, staged_ids = self._staged_category(category)
        removed_ids = {id(obj) for obj in objects}
        if staged_ids.isdisjoint(removed_ids):
            return
        staged[:] = [obj for obj in staged if id(obj) not in removed_ids]
        staged_ids.difference_update(removed_ids)


@dataclass(frozen=True)
class RecoveryAdapter:
    """Provide local cleanup capabilities without session access guards.

    Keep the adapter local to rollback or close. Its callbacks must support
    no-provider-I/O cleanup even when ordinary session access is forbidden.

    Attributes:
        unit_of_work: Staging categories to clear.
        detach_canceled: Detach canceled entities absent from the identity map.
        discard_identities: Remove exact objects' provisional identities in one
            pass.
        identity_values: Return the current identity-map object snapshot.
        owns_entity: Check whether cleanup may mutate an exact object.
        cleanup_anchors: Owned interrupted entities no longer discoverable
            through normal identity or staging state.
    """

    unit_of_work: UnitOfWork
    detach_canceled: Callable[[Entity], None]
    discard_identities: Callable[[Iterable[Entity]], None]
    identity_values: Callable[[], list[Entity]]
    owns_entity: Callable[[Entity], bool]
    cleanup_anchors: list[Entity] = field(default_factory=_new_entity_list)


class _UnitOfWorkSession(Protocol):
    """Minimal session contract consumed by shared staging and planning."""

    def assert_identity_available(self, obj: Entity) -> None:
        """Reject another object mapped under the same persistent identity.

        Args:
            obj: Persistent staged object whose identity must remain unique.

        Raises:
            SgalchemyError: If the session maps a different object with the
                same identity.
        """
        ...

    def assert_same_registry(self, value: Entity | type[Entity]) -> None:
        """Reject a class or object owned by another registry or open session.

        Args:
            value: Entity class or instance to validate.

        Raises:
            SgalchemyError: If `value` belongs to another registry or open
                session.
        """
        ...

    def attach_for_staging(self, obj: Entity) -> None:
        """Attach a detached object or reject another open session's object.

        Args:
            obj: Entity that requires new local staging.

        Raises:
            SgalchemyError: If another open session owns `obj`.
        """
        ...

    def detach_canceled(self, obj: Entity) -> None:
        """Detach a canceled object when it is absent from the identity map.

        Args:
            obj: Entity whose staged transition was canceled.
        """
        ...

    def hydrate_record(
        self,
        entity_type_name: str,
        record: Mapping[str, Any],
        binding: RegistryBinding | None,
    ) -> Entity:
        """Hydrate one wire record through the commit's captured binding.

        Args:
            entity_type_name: Canonical expected entity type.
            record: Provider record containing canonical identity.
            binding: Complete commit binding, or captured absence.

        Returns:
            Existing or newly allocated entity.
        """
        ...

    def hydration_scope(self) -> AbstractContextManager[None]:
        """Share staged identity lookup until this synchronous phase ends."""
        ...

    def register_identity(self, obj: Entity) -> None:
        """Register one persistent entity after collision-safe validation.

        Args:
            obj: Persistent entity to register.
        """
        ...

    @property
    def unit_of_work(self) -> UnitOfWork:
        """Return this session's identity-based staging lists.

        Returns:
            Mutable unit-of-work staging state owned by the session.
        """
        ...


class _PlanningSession(_UnitOfWorkSession, Protocol):
    """Session state consumed by the pure commit planner."""


def _apply_confirmed_field_read(
    session: _UnitOfWorkSession,
    read: ConfirmedFieldRead,
    record: dict[str, Any],
    binding: RegistryBinding | None,
) -> None:
    """Hydrate one complete current-value readback into its exact entity."""
    state = entity_state(read._obj)
    if state.id is None:
        return
    filtered = {'type': state.entity_type, 'id': state.id}
    filtered.update((field_code, record[field_code]) for field_code in read._field_codes)
    try:
        session.hydrate_record(state.entity_type, filtered, binding)
    except Exception:
        for field_code in read._field_codes:
            state.mark_field_unknown(field_code)
        raise
    if read._unverified:
        state.mark_writes_unverified(read._field_codes)


def _create_identity_error(
    session: _UnitOfWorkSession,
    operation: _Op,
    entity_id: int,
    returned_identities: set[tuple[str, int]],
) -> str | None:
    """Return a prevalidated create-identity collision message, if any."""
    key = (operation.entity_type, entity_id)
    if key in returned_identities:
        return f'batch create results contain duplicate identity {key!r}'
    returned_identities.add(key)
    probe = Entity.__new__(Entity)
    initialize_entity(
        probe,
        EntityState(entity_type=operation.entity_type, id=entity_id),
    )
    try:
        session.assert_identity_available(probe)
    except SgalchemyError as err:
        return str(err)
    return None


def _has_staged_work(session: _PlanningSession, identity_values: list[Entity]) -> bool:
    """Return whether commit planning has any effective local work."""
    return (
        any(entity_state(obj).dirty for obj in identity_values)
        or bool(session.unit_of_work._new)
        or bool(session.unit_of_work._deleted)
        or bool(session.unit_of_work._revived)
    )


def _hydrate_batch_echoes(
    session: _UnitOfWorkSession,
    chunk: tuple[_Op, ...],
    results: list[BatchItem],
) -> tuple[ConfirmedFieldReadGroup, ...]:
    """Contain per-operation echo faults and describe trustworthy omissions."""
    reads: list[tuple[str, bool, RegistryBinding | None, ConfirmedFieldRead]] = []
    for operation, result in zip(chunk, results, strict=True):
        if operation.request_type == 'delete':
            continue
        record = cast('WireRecord', result)
        hydration_record = record
        deterministic_clear = (
            THUMBNAIL_PRIMARY_FIELD in operation.data
            and operation.data[THUMBNAIL_PRIMARY_FIELD] is None
        )
        if deterministic_clear:
            hydration_record = {
                key: value for key, value in record.items() if key not in THUMBNAIL_UPLOAD_METHODS
            }
        try:
            session.hydrate_record(operation.entity_type, hydration_record, operation.binding)
        except Exception:
            _LOGGER.exception(
                'confirmed-write echo hydration failed for %s(id=%s)',
                operation.entity_type,
                entity_state(operation.obj).id,
            )
            _degrade_submitted_echo(operation)
            continue
        missing = tuple(sorted(operation.data.keys() - record.keys()))
        if not missing:
            continue
        recoverable = tuple(
            field_code
            for field_code in missing
            if not (deterministic_clear and field_code == THUMBNAIL_PRIMARY_FIELD)
        )
        reads.append(
            (
                operation.entity_type,
                entity_state(operation.obj).deleted,
                operation.binding,
                ConfirmedFieldRead(operation.obj, recoverable, missing),
            )
        )
    return _group_confirmed_field_reads(reads)


def _session_unit_of_work(unit_of_work: UnitOfWork) -> _SessionUnitOfWork:
    """Return session-only indexed staging state or report an invariant fault."""
    if not isinstance(unit_of_work, _SessionUnitOfWork):
        raise TypeError('session requires indexed unit-of-work state')
    return unit_of_work


def _retire_batch_chunk_provenance(
    session: _UnitOfWorkSession,
    chunk: tuple[_Op, ...],
) -> None:
    """Retire confirmed fields and compact each affected staging category once."""
    removals: dict[_StagingCategory, list[Entity]] = {'new': [], 'deleted': []}
    for operation in chunk:
        if operation.request_type == 'create':
            removals['new'].append(operation.obj)
        elif operation.request_type == 'delete':
            removals['deleted'].append(operation.obj)
        _retire_batch_operation_provenance(operation)
    unit_of_work = _session_unit_of_work(session.unit_of_work)
    for category, objects in removals.items():
        if objects:
            unit_of_work._discard_staged_many(category, objects)


def _establish_batch_outcome(
    session: _UnitOfWorkSession,
    chunk: tuple[_Op, ...],
    results: list[BatchItem],
    outcome: CommitOutcome,
    install_identity: Callable[[tuple[str, int], Entity], None],
) -> None:
    """Establish one prevalidated transactional chunk without fallible hydration."""
    outcome._confirm(chunk)
    _retire_batch_chunk_provenance(session, chunk)
    for operation in chunk:
        _apply_deterministic_thumbnail_clear(operation)
    for operation, result in zip(chunk, results, strict=True):
        if operation.request_type == 'delete':
            continue
        record = cast('WireRecord', result)
        _discard_missing_echo_fields(operation, record)
    for operation, result in zip(chunk, results, strict=True):
        if operation.request_type != 'create':
            continue
        record = cast('WireRecord', result)
        entity_id = cast('int', record['id'])
        state = entity_state(operation.obj)
        state.id = entity_id
        state.loaded['id'] = entity_id
    for operation in chunk:
        if operation.request_type == 'delete':
            entity_state(operation.obj).deleted = True
    for operation in chunk:
        state = entity_state(operation.obj)
        entity_id = cast('int', state.id)
        install_identity((state.entity_type, entity_id), operation.obj)


def _validate_batch_results(
    session: _UnitOfWorkSession,
    operations: tuple[_Op, ...],
    raw_results: object,
) -> list[BatchItem]:
    """Validate one complete successful batch payload before local mutation."""
    if not isinstance(raw_results, list):
        raise BatchReconciliationError(
            'batch result count must match submitted operations; discard affected '
            'objects and reconcile server state through a fresh query in a new session'
        )
    results = cast('list[BatchItem]', raw_results)
    if len(results) != len(operations):
        raise BatchReconciliationError(
            'batch result count must match submitted operations; discard affected '
            'objects and reconcile server state through a fresh query in a new session'
        )
    returned_create_identities: set[tuple[str, int]] = set()
    for operation, result in zip(operations, results, strict=True):
        structure_error = _batch_result_structure_error(operation, result)
        if structure_error is not None:
            raise BatchReconciliationError(
                f'{structure_error}; discard affected objects and reconcile server state '
                'through a fresh query in a new session'
            )
        if operation.request_type != 'create':
            continue
        record = cast('WireRecord', result)
        identity_error = _create_identity_error(
            session,
            operation,
            cast('int', record['id']),
            returned_create_identities,
        )
        if identity_error is not None:
            raise BatchReconciliationError(
                f'{identity_error}; discard affected objects and reconcile server state '
                'through a fresh query in a new session'
            )
    return results


def _validate_entity_graph_recursive(
    session: _UnitOfWorkSession,
    value: object,
    active: set[int],
) -> None:
    """Walk one assigned value, rejecting foreign entities and container cycles."""
    if isinstance(value, Entity):
        session.assert_same_registry(value)
        state = entity_state(value)
        if state.indeterminate:
            raise SgalchemyError(
                f'relationship references {state.entity_type} with an indeterminate '
                'server outcome; discard it and reconcile through a fresh query'
            )
        if state.id is None and not _session_unit_of_work(session.unit_of_work)._contains_staged(
            'new', value
        ):
            raise SgalchemyError('pending relationship entity is not staged in this session')
        return
    children = container_children(value)
    if children is None:
        return
    container = cast('WireTraversalContainer', value)
    container_id = id(container)
    if container_id in active:
        raise SgalchemyError(_CYCLIC_CONTAINER_NOTE)
    active.add(container_id)
    try:
        for item in children:
            _validate_entity_graph_recursive(session, item, active)
    finally:
        active.discard(container_id)


def _validate_entity_graph(
    session: _UnitOfWorkSession,
    value: object,
) -> None:
    """Recursively reject nested entities outside this unit of work."""
    _validate_entity_graph_recursive(session, value, set())


def _validate_staged_object(
    session: _PlanningSession,
    obj: Entity,
) -> None:
    """Reject staged objects whose registry, ownership, or access is invalid."""
    session.assert_same_registry(obj)


def _validate_staging_state(
    session: _PlanningSession,
    identity_by_key: Mapping[tuple[str, int], Entity],
) -> None:
    """Validate staged object access and distinct-object identity collisions."""
    staged_categories: tuple[tuple[_StagingCategory, list[Entity]], ...] = (
        ('new', session.unit_of_work._new),
        ('deleted', session.unit_of_work._deleted),
        ('revived', session.unit_of_work._revived),
    )
    staged_entries: list[tuple[Entity, tuple[str, int] | None]] = []
    staged_by_identity: dict[tuple[str, int], Entity] = {}
    for category, staged in staged_categories:
        for obj in staged:
            identity_key = None
            if category != 'new':
                state = entity_state(obj)
                if type(state.id) is int and state.id > 0:
                    identity_key = (state.entity_type, state.id)
                    staged_by_identity[identity_key] = obj
            staged_entries.append((obj, identity_key))

    for obj, identity_key in staged_entries:
        _validate_staged_object(session, obj)
        if identity_key is None:
            continue
        state = entity_state(obj)
        owners = (
            staged_by_identity[identity_key],
            identity_by_key.get(identity_key, obj),
        )
        if any(owner is not obj for owner in owners):
            raise SgalchemyError(
                f'session already tracks a different {state.entity_type}(id={state.id}) '
                'object; use session.get(type(obj), obj.id) to retrieve it, or close '
                'this session before staging the detached copy'
            )


def _validate_local_state(
    session: _PlanningSession,
    identity_by_key: Mapping[tuple[str, int], Entity],
    dirty_participants: list[Entity],
) -> None:
    """Validate staging state, relationship graphs, and identity uniqueness."""
    _validate_staging_state(session, identity_by_key)
    for obj in dirty_participants:
        state = entity_state(obj)
        if state.indeterminate:
            raise SgalchemyError(
                f'{state.entity_type} with an indeterminate server outcome cannot own '
                'staged writes; discard it and reconcile through a fresh query'
            )
        for field_code, value in state.dirty.items():
            _require_dirty_type(state, field_code)
            _validate_entity_graph(session, value)
    pending_operations = [
        _Op(
            'create',
            entity_state(obj).entity_type,
            None,
            dict(entity_state(obj).dirty),
            obj,
        )
        for obj in session.unit_of_work._new
    ]
    _operation_waves(pending_operations)


def apply_batch_indeterminate(
    session: _UnitOfWorkSession,
    operations: tuple[_Op, ...],
    outcome: CommitOutcome,
) -> None:
    """Quarantine one unreconcilable submitted batch and its follow-ups.

    Submitted operations become indeterminate. Definitely unsent same-owner
    work becomes withdrawn, and later plain commits retry neither partition.

    Args:
        session: Session whose local staging produced the batch.
        operations: Exact operations whose provider call returned successfully.
        outcome: Commit partitions to update after reconciliation failed.
    """
    affected_entities = tuple(operation.obj for operation in operations)
    unsafe_items = outcome._staged_items_for(affected_entities)
    withdrawn_items = tuple(
        item for item in unsafe_items if all(item is not operation for operation in operations)
    )
    unit_of_work = _session_unit_of_work(session.unit_of_work)
    for operation in operations:
        obj = operation.obj
        if operation.request_type != 'create':
            session.register_identity(obj)
        unit_of_work._discard_staged('new', obj)
        unit_of_work._discard_staged('deleted', obj)
        state = entity_state(obj)
        state.indeterminate = True
        if operation.request_type == 'create':
            session.detach_canceled(obj)
    for item in unsafe_items:
        _clear_plan_item_fields(item)
    outcome.mark_indeterminate(operations)
    outcome._withdraw(withdrawn_items)


def apply_batch_results(
    session: _UnitOfWorkSession,
    chunk: tuple[_Op, ...],
    results: object,
    outcome: CommitOutcome,
    install_identity: Callable[[tuple[str, int], Entity], None],
) -> tuple[ConfirmedFieldReadGroup, ...]:
    """Validate a complete batch, then establish its outcome before hydration.

    Structural validation covers every result before local mutation. The
    complete trusted chunk becomes committed and loses retry provenance before
    fallible echo hydration begins. Per-operation hydration failures are
    contained by the caller without reclassifying the confirmed write.

    Args:
        session: Session whose local state produced the batch.
        chunk: Exact submitted operations.
        results: Successful provider payload to reconcile.
        outcome: Commit partitions to update after reconciliation.
        install_identity: Non-validating identity-map insertion callback.

    Returns:
        Grouped current-value reads for trustworthy echo omissions.

    Raises:
        BatchReconciliationError: If results cannot map safely to requests.
    """
    with session.hydration_scope():
        validated = _validate_batch_results(session, chunk, results)
    _establish_batch_outcome(session, chunk, validated, outcome, install_identity)
    with session.hydration_scope():
        return _hydrate_batch_echoes(session, chunk, validated)


def apply_confirmed_field_readback(
    session: _UnitOfWorkSession,
    group: ConfirmedFieldReadGroup,
    records: WireRecords,
) -> None:
    """Apply complete canonical rows from one confirmed-field recovery call.

    Args:
        session: Session whose exact confirmed entities receive read values.
        group: Entity fields requested by one bounded current-value read.
        records: Provider rows returned for the exact group identities.
    """
    with session.hydration_scope():
        for read, record in _validated_confirmed_field_records(group, records):
            _apply_confirmed_field_read(session, read, record, group.binding)


def apply_revive_result(
    unit_of_work: UnitOfWork,
    obj: Entity,
    outcome: CommitOutcome,
) -> None:
    """Reconcile one successful revive against local staging.

    Args:
        unit_of_work: Session staging lists that produced the revive.
        obj: Entity confirmed revived by the provider.
        outcome: Commit partitions to update after local reconciliation.
    """
    _session_unit_of_work(unit_of_work)._discard_staged('revived', obj)
    entity_state(obj).deleted = False
    outcome._confirm((obj,))


def apply_upload_indeterminate(
    session: _UnitOfWorkSession,
    upload: UploadPlanItem,
    outcome: CommitOutcome,
) -> ConfirmedFieldRead:
    """Remove one ambiguous upload path and unsent follow-ups from retry.

    The attempted upload becomes indeterminate, same-owner unsent follow-ups
    become withdrawn, and the field requires one bounded current-value read.

    Args:
        session: Session whose local state produced the upload.
        upload: Exact entity, field code, and path whose provider call raised.
        outcome: Commit partitions to update after the ambiguous attempt.

    Returns:
        Exact current-value read for the ambiguous field.
    """
    session.register_identity(upload.obj)
    state = entity_state(upload.obj)
    state.retire_upload_field(upload.field_code)
    outcome.mark_indeterminate((upload,))
    follow_ups = tuple(
        item
        for item in outcome._staged_items_for((upload.obj,))
        if not isinstance(item, _Op) or item.request_type == 'delete'
    )
    for item in follow_ups:
        if not _restore_withdrawn_upload(item):
            _clear_plan_item_fields(item)
    _session_unit_of_work(session.unit_of_work)._discard_staged('deleted', upload.obj)
    outcome._withdraw(follow_ups)
    return ConfirmedFieldRead(
        _obj=upload.obj,
        _field_codes=(upload.field_code,),
        _missing_field_codes=(upload.field_code,),
        _unverified=True,
    )


def batch_failure_note(
    step: CommitStep | None,
    outcome: CommitOutcome,
    *,
    invoked: bool,
) -> str | None:
    """Describe provider invocation for an unconfirmed failed batch.

    Args:
        step: Failed commit step, or `None` when no step was active.
        outcome: Failure-time partitions with unconfirmed batch work still
            staged.
        invoked: Whether the provider call started, even if no result returned.

    Returns:
        The batch status note, or `None` for a missing, non-batch, or confirmed
        step.
    """
    if step is None or step.kind != 'batch' or outcome.contains_committed(step.operations):
        return None
    return _BATCH_OUTCOME_UNKNOWN_NOTE if invoked else _BATCH_UNSENT_NOTE


def confirmed_field_read_codes(
    group: ConfirmedFieldReadGroup,
) -> tuple[list[str], list[str]]:
    """Return sorted unique requested and missing codes for one read group.

    Args:
        group: Compatible confirmed entity reads.

    Returns:
        Requested field codes and missing-field diagnostics.
    """
    field_codes = sorted({field_code for read in group._reads for field_code in read._field_codes})
    missing_field_codes = sorted(
        {field_code for read in group._reads for field_code in read._missing_field_codes}
    )
    return field_codes, missing_field_codes


def confirmed_field_read_call(group: ConfirmedFieldReadGroup) -> FindCall | None:
    """Build one bounded provider call for a confirmed field read group.

    Args:
        group: Compatible confirmed entity reads.

    Returns:
        Provider call for the requested fields, or `None` when none are
        requested.
    """
    field_codes, _missing_field_codes = confirmed_field_read_codes(group)
    if not field_codes:
        return None
    entity_ids = sorted({cast('int', entity_state(read._obj).id) for read in group._reads})
    return FindCall(
        entity_type=group.entity_type,
        filters=[['id', 'in', entity_ids]],
        fields=field_codes,
        extra={'retired_only': True} if group.retired_only else {},
    )


def count_unsubmitted_work(recovery: RecoveryAdapter) -> tuple[int, int, int, int]:
    """Count owned local work remaining after submitted provenance retirement.

    Args:
        recovery: Guard-free access to one session's owned local state.

    Returns:
        Counts for dirty owned entities, pending creates, staged deletes, and
        staged revives.
    """
    unit_of_work = recovery.unit_of_work
    identity_values = _unique_entities([*recovery.identity_values(), *recovery.cleanup_anchors])
    dirty = sum(
        bool(entity_state(obj).dirty) for obj in identity_values if recovery.owns_entity(obj)
    )
    pending = sum(1 for obj in _unique_entities(unit_of_work._new) if recovery.owns_entity(obj))
    deletes = sum(
        1 for obj in _unique_entities(unit_of_work._deleted) if recovery.owns_entity(obj)
    )
    revives = sum(
        1 for obj in _unique_entities(unit_of_work._revived) if recovery.owns_entity(obj)
    )
    return dirty, pending, deletes, revives


def discard_local_state(recovery: RecoveryAdapter) -> None:
    """Restore dirty snapshots, clear staging, and detach canceled entities.

    This guard-free local operation performs no provider I/O and cannot undo
    provider writes. It preserves each field's prior trust provenance.

    Args:
        recovery: Stack-local capability for the owning rollback or close.
    """
    unit_of_work = recovery.unit_of_work
    pending = list(unit_of_work._new)
    staged = [*pending, *unit_of_work._deleted, *unit_of_work._revived]
    cleanup_anchors = list(recovery.cleanup_anchors)
    owned_pending = [obj for obj in pending if recovery.owns_entity(obj)]
    owned_staged = [obj for obj in _unique_entities(staged) if recovery.owns_entity(obj)]
    owned_cleanup = [obj for obj in _unique_entities(cleanup_anchors) if recovery.owns_entity(obj)]
    owned_objects = _unique_entities(
        [
            *[obj for obj in recovery.identity_values() if recovery.owns_entity(obj)],
            *owned_staged,
            *owned_cleanup,
        ]
    )
    for obj in owned_objects:
        entity_state(obj).restore_staged_fields()
    indexed_unit_of_work = _session_unit_of_work(unit_of_work)
    indexed_unit_of_work._clear_staged('new')
    indexed_unit_of_work._clear_staged('deleted')
    indexed_unit_of_work._clear_staged('revived')
    recovery.cleanup_anchors.clear()
    recovery.discard_identities(owned_pending)
    for obj in _unique_entities([*owned_staged, *owned_cleanup]):
        recovery.detach_canceled(obj)


def group_upload_field_reads(
    reads: list[ConfirmedFieldRead],
    binding: RegistryBinding | None,
) -> tuple[ConfirmedFieldReadGroup, ...]:
    """Group upload current-value reads by canonical entity type.

    Args:
        reads: Upload field descriptors awaiting one current-value read.
        binding: Registry binding captured for the complete commit.

    Returns:
        One bounded read group per entity type in upload order.
    """
    return _group_confirmed_field_reads(
        [
            (
                entity_state(read._obj).entity_type,
                False,
                binding,
                read,
            )
            for read in reads
        ]
    )


def mark_indeterminate_entities(entities: tuple[Entity, ...]) -> None:
    """Permanently reject entities touched by an ambiguous submitted step.

    Args:
        entities: Entities whose server state can no longer be known locally.
    """
    for obj in entities:
        entity_state(obj).indeterminate = True


def new_session_unit_of_work() -> UnitOfWork:
    """Return private indexed staging state with read-only category snapshots.

    Returns:
        New package-internal staging state with immutable category snapshots.
    """
    return _SessionUnitOfWork()


def plan_commit(
    session: _PlanningSession,
    binding: RegistryBinding | None,
    identity_map: Mapping[tuple[str, int], Entity],
) -> CommitPlan:
    """Build an ordered commit plan from every exact local write candidate.

    Candidates include identity-mapped objects and detached objects staged for
    deletion or revival. Local validation precedes schema recovery signals.
    Deletes with uploads move to final retirement batches; ordinary same-owner
    updates precede deletes in their transactional batch.

    Args:
        session: Session owning identity state and staged transitions.
        binding: Exact prepared schema generation, or `None` before
            preparation.
        identity_map: State-owned identity mapping to snapshot for this
            attempt.

    Returns:
        Revives, dependency-ordered batch waves, uploads, and final
        retirements.
        No effective work produces an empty plan without requiring a binding.

    Raises:
        RetiredEntityWriteError: If a dirty retired entity lacks revival
            intent.
        SchemaDrift: If schema preparation or a bounded refresh is required
            before the caller can rebuild the plan.
        SgalchemyError: If local ownership, identity collisions, write
            provenance, relationship references, or dependency graphs are
            invalid.
    """
    identity_items = list(identity_map.items())
    identity_by_key = dict(identity_items)
    identity_values = [candidate for _key, candidate in identity_items]
    write_candidates = _unique_entities(
        [
            *identity_values,
            *session.unit_of_work._revived,
            *session.unit_of_work._deleted,
        ]
    )
    dirty_participants = _dirty_participants(write_candidates, session.unit_of_work._new)
    _validate_retired_write_intent(dirty_participants, session.unit_of_work._revived)
    _validate_local_state(
        session,
        identity_by_key,
        dirty_participants,
    )
    if not _has_staged_work(session, identity_values):
        return CommitPlan(_revives=[], _waves=[], _uploads=[], binding=None)
    if binding is None:
        raise SchemaDrift('commit', '*', 'prepared schema')
    schemas = binding.schemas
    schema_participants = _unique_entities(
        [
            *session.unit_of_work._new,
            *session.unit_of_work._deleted,
            *session.unit_of_work._revived,
            *dirty_participants,
        ]
    )
    _validate_schema_participants(schemas, schema_participants)

    operations, uploads, normal_retirement_owners = _persistent_write_operations(
        write_candidates,
        session.unit_of_work._deleted,
        schemas,
        binding,
    )
    for obj in session.unit_of_work._new:
        state = entity_state(obj)
        data, entity_uploads = _operation_fields(schemas[state.entity_type], obj)
        uploads.extend(entity_uploads)
        operations.append(_Op('create', state.entity_type, None, data, obj, binding))
    normal_retirements, final_retirements = _remaining_retirement_operations(
        session.unit_of_work._deleted,
        uploads,
        normal_retirement_owners,
        binding,
    )
    operations.extend(normal_retirements)
    return CommitPlan(
        _revives=list(session.unit_of_work._revived),
        _waves=_operation_waves(operations),
        _uploads=uploads,
        binding=binding,
        _final_retirements=final_retirements,
    )


def preflight_uploads(plan: CommitPlan) -> CommitPlan:
    """Validate and normalize all thumbnail paths before any provider write.

    Args:
        plan: Final commit plan whose uploads have not been submitted.

    Returns:
        A plan whose upload tuples contain normalized filesystem paths.

    Raises:
        SgalchemyError: If any upload path is unusable before provider I/O.
    """
    uploads: list[UploadPlanItem] = []
    for upload in plan._uploads:
        state = entity_state(upload.obj)
        try:
            normalized = _normalize_upload_path(upload.path)
        except (TypeError, ValueError, OSError) as err:
            raise SgalchemyError(
                f'{state.entity_type}.{upload.field_code} thumbnail path must name a nonempty '
                'readable regular file; assign a local str or os.PathLike path'
            ) from err
        uploads.append(UploadPlanItem(upload.obj, upload.field_code, normalized))
    return replace(plan, _uploads=uploads)


def retire_interrupted_step(
    session: _UnitOfWorkSession,
    step: CommitStep,
) -> tuple[Entity, ...]:
    """Retire only the exact submitted step from local retry provenance.

    Args:
        session: Session whose provider call was interrupted.
        step: Exact provider step submitted before the interruption.

    Returns:
        Submitted entities no longer discoverable through ordinary identity or
        staging state and requiring cleanup-only ownership anchors.
    """
    if step.kind == 'batch':
        cleanup_anchors: list[Entity] = []
        _retire_batch_chunk_provenance(session, step.operations)
        for operation in step.operations:
            if operation.request_type == 'create':
                session.detach_canceled(operation.obj)
            elif operation.request_type == 'delete':
                cleanup_anchors.append(operation.obj)
        return tuple(cleanup_anchors)
    if step.kind == 'revive':
        unit_of_work = _session_unit_of_work(session.unit_of_work)
        cleanup_anchors = [
            obj for obj in step.entities if unit_of_work._discard_staged('revived', obj)
        ]
        return tuple(cleanup_anchors)
    upload = step._upload
    if upload is None:
        return ()
    state = entity_state(upload.obj)
    state.retire_upload_field(upload.field_code)
    return ()


def schema_drift_error(error: SchemaDrift, *, refreshed: bool) -> SgalchemyError:
    """Translate exhausted planning drift into one actionable public error.

    Args:
        error: Recoverable schema fact that remained invalid after bounded
            recovery.
        refreshed: Whether recovery forced a provider refresh instead of cold
            preparation.

    Returns:
        Public pre-write error teaching the caller to discard stale intent.
    """
    if error.field_code == '*':
        fact = f'no longer contains entity type {error.entity_type!r}'
    elif error.target_type == 'present field':
        fact = f'no longer contains {error.entity_type}.{error.field_code}'
    elif error.target_type == 'API-writable field':
        fact = f'marks {error.entity_type}.{error.field_code} as no longer API-writable'
    elif error.target_type == 'editable field':
        fact = f'marks {error.entity_type}.{error.field_code} as noneditable'
    else:
        fact = (
            f'classifies {error.entity_type}.{error.field_code} differently from its '
            f'{error.target_type}'
        )
    recovery = 'refreshing' if refreshed else 'preparing'
    return SgalchemyError(
        f'prepared schema {fact} after {recovery} schema once; call session.rollback() '
        'and recreate staged work from the refreshed schema'
    )


def serialize_chunk(chunk: tuple[_Op, ...]) -> WireRecords:
    """Serialize one operation chunk for a provider batch call.

    Args:
        chunk: Exact dependency-ready operations for one batch call.

    Returns:
        Provider request dictionaries in operation order.
    """
    requests: WireRecords = []
    for operation in chunk:
        request: WireRecord = {
            'request_type': operation.request_type,
            'entity_type': operation.entity_type,
        }
        if operation.request_type == 'create':
            request['data'] = _wire_value(operation.data)
        elif operation.request_type == 'update':
            request['entity_id'] = operation.entity_id
            request['data'] = _wire_value(operation.data)
        else:
            request['entity_id'] = operation.entity_id
        requests.append(request)
    return requests


def stage_add(session: _UnitOfWorkSession, obj: Entity) -> None:
    """Attach and idempotently stage one pending create.

    Args:
        session: Session that will own and stage `obj`.
        obj: Pending entity to stage.

    Raises:
        SgalchemyError: If `obj` is persistent or cannot attach to `session`.
    """
    session.assert_same_registry(obj)
    state = entity_state(obj)
    if state.id is not None:
        raise SgalchemyError('session.add() requires a pending Entity.new() object')
    session.attach_for_staging(obj)
    _session_unit_of_work(session.unit_of_work)._append_staged('new', obj)


def stage_delete(session: _UnitOfWorkSession, obj: Entity) -> None:
    """Stage retirement or cancel an uncommitted create or revive.

    Args:
        session: Session that owns the requested transition.
        obj: Entity to retire or remove from inverse staging.

    Raises:
        SgalchemyError: If `obj` cannot be used by `session` or is pending and
            unstaged.
    """
    session.assert_same_registry(obj)
    state = entity_state(obj)
    unit_of_work = _session_unit_of_work(session.unit_of_work)
    if state.id is None:
        if not unit_of_work._discard_staged('new', obj):
            raise SgalchemyError('pending entity is not staged in this session')
        session.detach_canceled(obj)
        return
    if state.deleted:
        if unit_of_work._discard_staged('revived', obj):
            session.detach_canceled(obj)
        return
    if unit_of_work._contains_staged('deleted', obj):
        return
    session.attach_for_staging(obj)
    unit_of_work._append_staged('deleted', obj)


def stage_revive(session: _UnitOfWorkSession, obj: Entity) -> None:
    """Stage revival or cancel an uncommitted delete.

    Args:
        session: Session that owns the requested transition.
        obj: Entity to revive or remove from inverse staging.

    Raises:
        SgalchemyError: If `obj` cannot be used, is pending, or is active
            without a staged delete to cancel.
    """
    session.assert_same_registry(obj)
    state = entity_state(obj)
    unit_of_work = _session_unit_of_work(session.unit_of_work)
    if state.id is None:
        raise SgalchemyError('pending entity cannot be revived')
    if state.deleted:
        if unit_of_work._contains_staged('revived', obj):
            return
        session.attach_for_staging(obj)
        unit_of_work._append_staged('revived', obj)
        return
    if unit_of_work._discard_staged('deleted', obj):
        session.detach_canceled(obj)
        return
    raise SgalchemyError('active entity cannot be revived unless its delete is staged')


def staged_intent_snapshot(
    unit_of_work: UnitOfWork,
    identity_values: list[Entity],
) -> list[_PlanItem]:
    """Capture schema-free staged entities in deterministic local order.

    Args:
        unit_of_work: Current identity-based staging lists.
        identity_values: Snapshot of identity-mapped entities.

    Returns:
        Unique staged intent in revive, dirty, new, then delete order.
    """
    candidates = [*unit_of_work._revived]
    candidates.extend(obj for obj in identity_values if entity_state(obj).dirty)
    candidates.extend(unit_of_work._new)
    candidates.extend(unit_of_work._deleted)
    return cast('list[_PlanItem]', _unique_entities(candidates))


def validate_batch_size(batch_size: int | None) -> None:
    """Reject an invalid explicit provider batch limit.

    Args:
        batch_size: Explicit provider batch limit or `None`.

    Raises:
        ValueError: If the value is not a positive integer or `None`.
    """
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError('batch_size must be a positive integer or None')


def _chunk_wave(
    wave: list[_Op],
    batch_size: int | None,
) -> tuple[tuple[_Op, ...], ...]:
    """Split one dependency wave only when an explicit positive size is given."""
    validate_batch_size(batch_size)
    if not wave:
        return ()
    if batch_size is None:
        return (tuple(wave),)
    groups = _indivisible_operation_groups(wave)
    chunks: list[tuple[_Op, ...]] = []
    current: list[_Op] = []
    for group in groups:
        if len(group) > batch_size:
            raise ValueError(
                f'batch_size={batch_size} cannot split an update/delete owner group; '
                f'the required minimum is {len(group)}'
            )
        if current and len(current) + len(group) > batch_size:
            chunks.append(tuple(current))
            current = []
        current.extend(group)
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def commit_steps(plan: CommitPlan, batch_size: int | None) -> tuple[CommitStep, ...]:
    """Materialize revives, batches, uploads, and final retirements in order.

    Args:
        plan: Pure dependency-ordered commit plan.
        batch_size: Optional explicit maximum operations per batch call.

    Returns:
        Immutable serial provider-call steps.

    Raises:
        SgalchemyError: If an upload field has no supported provider step.
        ValueError: If `batch_size` is invalid or cannot preserve an
            indivisible operation group.
    """
    steps: list[CommitStep] = []
    for obj in plan._revives:
        state = entity_state(obj)
        steps.append(
            CommitStep(
                kind='revive',
                description=f'revive {state.entity_type}(id={state.id})',
                entities=(obj,),
                items=(obj,),
            )
        )
    for wave_index, wave in enumerate(plan._waves, start=1):
        for chunk_index, chunk in enumerate(_chunk_wave(wave, batch_size), start=1):
            size_context = '' if batch_size is None else f' (batch_size={batch_size})'
            steps.append(
                CommitStep(
                    kind='batch',
                    description=f'batch wave {wave_index} chunk {chunk_index}{size_context}',
                    entities=tuple(operation.obj for operation in chunk),
                    items=tuple(chunk),
                    operations=tuple(chunk),
                )
            )
    for upload in plan._uploads:
        state = entity_state(upload.obj)
        kind = THUMBNAIL_UPLOAD_METHODS.get(upload.field_code)
        if kind is None:
            raise SgalchemyError(
                f'unsupported image upload field code {state.entity_type}.{upload.field_code}'
            )
        steps.append(
            CommitStep(
                kind=kind,
                description=f'{kind} for {state.entity_type}(id={state.id}).{upload.field_code}',
                entities=(upload.obj,),
                items=(upload,),
                _upload=upload,
            )
        )
    for chunk_index, chunk in enumerate(
        _chunk_wave(plan._final_retirements, batch_size),
        start=1,
    ):
        size_context = '' if batch_size is None else f' (batch_size={batch_size})'
        steps.append(
            CommitStep(
                kind='batch',
                description=f'final retirement batch {chunk_index}{size_context}',
                entities=tuple(operation.obj for operation in chunk),
                items=tuple(chunk),
                operations=tuple(chunk),
            )
        )
    return tuple(steps)
