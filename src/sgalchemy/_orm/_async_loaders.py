"""Execute awaited relationship loads."""

import asyncio
from typing import TYPE_CHECKING

from sgalchemy._core._compiler import FindCall
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._registry import Entity, RegistryBinding, entity_state
from sgalchemy._errors import SgalchemyError
from sgalchemy._orm._loaders import (
    collect_selectin_refs,
    missing_selectin_targets,
    relationship_error,
    union_selectin_refs,
)

if TYPE_CHECKING:
    from sgalchemy._orm._async_session import AsyncSession


async def _execute_selectin_refs(
    session: 'AsyncSession',
    references: dict[str, set[int]],
    binding: RegistryBinding,
) -> None:
    """Execute already-preflighted target finds for one relationship."""
    calls: list[FindCall] = []
    for target_type in sorted(references):
        ids = references[target_type]
        schema = binding.schemas[target_type]
        calls.append(
            FindCall(
                entity_type=target_type,
                filters=[['id', 'in', sorted(ids)]],
                fields=schema.readable_field_codes(),
            )
        )
    tasks = [asyncio.create_task(session.engine.execute_call(call)) for call in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for call, task, result in zip(calls, tasks, results, strict=True):
        if task.cancelled():
            raised = True
            completed = result
        else:
            error = task.exception()
            raised = error is not None
            completed = error if error is not None else result
        records = session.finish_execute_call_result(call, completed, raised=raised)
        with session.hydration_scope():
            for record in records:
                session.hydrate_record(call.entity_type, record, binding)


def _require_complete_parent_response(
    objects: list[Entity],
    field: FieldExpr,
    missing_parent_ids: list[int],
) -> None:
    """Reject a parent fetch that omits any exact requested relationship."""
    missing_ids = set(missing_parent_ids)
    field_code = field.filter_field
    if any(
        entity_state(obj).id in missing_ids and field_code not in entity_state(obj).loaded
        for obj in objects
    ):
        raise SgalchemyError(
            f'{field.root_type}.{field_code} load received an incomplete provider response; '
            'retry after confirming the provider returns every requested parent row with '
            'this relationship field'
        )


async def _selectin_binding(
    session: 'AsyncSession',
    batches: list[tuple[FieldExpr, dict[str, set[int]]]],
    binding: RegistryBinding,
) -> RegistryBinding:
    """Preflight all observed targets and force schema at most once."""
    missing = missing_selectin_targets(
        (references for _field, references in batches),
        binding,
    )
    if not missing:
        return binding
    first_missing_field = next(
        field for field, references in batches if missing_selectin_targets([references], binding)
    )
    refreshed = await session.load_binding(
        f'raised while loading {first_missing_field.root_type}.{first_missing_field.filter_field}',
        force=True,
    )
    for field, references in batches:
        remaining = missing_selectin_targets([references], refreshed)
        if remaining:
            raise relationship_error(field, remaining, refreshed)
    return refreshed


async def execute_selectins(
    session: 'AsyncSession',
    objects: list[Entity],
    fields: list[FieldExpr],
    binding: RegistryBinding,
) -> None:
    """Preflight one target union, then run bounded async target reads.

    References shared by relationship fields are coalesced into one read per
    observed target type. Missing target schema triggers at most one forced
    refresh before secondary queries.

    Args:
        session: Async session owning the result objects and identity map.
        objects: Originating entity result set.
        fields: Relationship fields in loader order.
        binding: Complete operation binding captured before target loading.

    Raises:
        RelationshipTargetUnavailableError: If observed targets remain absent
            after the bounded refresh attempt.
        SgalchemyError: If a loaded relationship has a non-entity link value.
        Exception: Original provider exception, annotated and unwrapped.
    """
    batches = [(field, collect_selectin_refs(objects, field.filter_field)) for field in fields]
    active = [(field, references) for field, references in batches if references]
    if not active:
        return
    binding = await _selectin_binding(session, active, binding)
    references = union_selectin_refs(references for _field, references in active)
    await _execute_selectin_refs(session, references, binding)


async def load_relationship(
    session: 'AsyncSession',
    objects: list[Entity],
    field: FieldExpr,
    missing_parent_ids: dict[bool, list[int]],
    binding: RegistryBinding,
) -> None:
    """Fetch missing parent relationship values, then load observed targets.

    Present `None` and `[]` values need no parent fetch. Dirty values remain
    protected during hydration. Target loading preflights the observed-type
    union and may force one schema refresh before secondary queries.

    Args:
        session: Async session owning the result objects and identity map.
        objects: Originating entity result set.
        field: Validated direct relationship field to load.
        missing_parent_ids: Missing parent IDs grouped by retirement state.
        binding: Complete operation binding captured before validation.

    Raises:
        SgalchemyError: If the provider omits a required parent relationship
            value or a loaded relationship contains a non-entity link value.
        RelationshipTargetUnavailableError: If observed targets remain absent
            after the bounded refresh attempt.
        Exception: Original provider exception, annotated and unwrapped.
    """
    for retired, parent_ids in missing_parent_ids.items():
        call = FindCall(
            entity_type=field.root_type,
            filters=[['id', 'in', parent_ids]],
            fields=[field.filter_field],
            extra={'retired_only': True} if retired else {},
        )
        records = await session.execute_call(call, binding)
        with session.hydration_scope():
            for record in records:
                session.hydrate_record(field.root_type, record, binding)
        _require_complete_parent_response(objects, field, parent_ids)
    await execute_selectins(
        session,
        objects,
        [field],
        binding,
    )
