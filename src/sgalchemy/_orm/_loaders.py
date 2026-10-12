"""Execute synchronous relationship loads."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from sgalchemy._core._compiler import FindCall
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._options import LazyLoad, SelectinLoad
from sgalchemy._core._registry import Entity, RegistryBinding, entity_state
from sgalchemy._errors import RelationshipTargetUnavailableError, SgalchemyError

if TYPE_CHECKING:
    from sgalchemy._orm._session import Session

_LOGGER = logging.getLogger('sgalchemy.session')


@dataclass
class LazyContext:
    """Coordinate one deferred relationship load across its originating rows.

    Resolution reserves the whole per-target-type query count before I/O and
    clears only lazy entries still owned by this context after success.

    Failed schema snapshots are retained so a newer binding permits a fresh
    recovery attempt.
    """

    _session: 'Session'
    _objects: list[Entity]
    _opt: LazyLoad
    _failed_bindings: list[RegistryBinding] = field(default_factory=list[RegistryBinding])

    def _retire(self, objects: list[Entity], field_code: str) -> None:
        """Clear this context's exact owned lazy entries."""
        for obj in objects:
            state = entity_state(obj)
            if state.lazy.get(field_code) is self:
                state.lazy.pop(field_code)

    def resolve(self, field_code: str) -> None:
        """Reserve the query budget and resolve this context's owned links.

        Reserved calls remain charged if later schema or provider work fails.
        Successful resolution clears owned lazy entries; failure leaves them
        available for a later access subject to the remaining budget and
        failed-binding memory.

        Args:
            field_code: Alias-resolved relationship field being accessed.

        Raises:
            SgalchemyError: If the complete load exceeds the implicit-query
                budget or a relationship contains a non-entity link value.
            RelationshipTargetUnavailableError: If observed targets remain
                absent after the bounded refresh attempt.
            Exception: Original provider exception, annotated and unwrapped.
        """
        owned_objects = [
            obj for obj in self._objects if entity_state(obj).lazy.get(field_code) is self
        ]
        references = collect_selectin_refs(owned_objects, field_code)
        try:
            self._session.reserve_implicit_calls(len(references))
        except SgalchemyError as error:
            raise SgalchemyError(
                f'{error}; triggering attribute {self._opt.field.root_type}.{field_code}'
            ) from None
        _LOGGER.info(
            'lazy load: %s.%s calls=%s',
            self._opt.field.root_type,
            field_code,
            len(references),
        )
        if not references:
            self._retire(owned_objects, field_code)
            return
        binding = self._session.load_binding(
            f'raised while lazy-loading {self._opt.field.root_type}.{field_code}'
        )
        _execute_selectin(
            self._session,
            owned_objects,
            self._opt,
            binding,
            failed_bindings=self._failed_bindings,
        )
        self._retire(owned_objects, field_code)


def _execute_selectin_refs(
    session: 'Session',
    binding: RegistryBinding,
    references: dict[str, set[int]],
    *,
    on_target_complete: Callable[[str], None] | None = None,
    preserve_target_order: bool = False,
) -> None:
    """Execute already-preflighted target finds for one relationship option."""
    target_types = references if preserve_target_order else sorted(references)
    for target_type in target_types:
        ids = references[target_type]
        schema = binding.schemas[target_type]
        call = FindCall(
            entity_type=target_type,
            filters=[['id', 'in', sorted(ids)]],
            fields=schema.readable_field_codes(),
        )
        records = session.execute_call(call, binding)
        with session.hydration_scope():
            for record in records:
                session.hydrate_record(target_type, record, binding)
        if on_target_complete is not None:
            on_target_complete(target_type)


def collect_selectin_refs(
    objects: list[Entity],
    field_code: str,
) -> dict[str, set[int]]:
    """Collect canonical linked identities by target entity type.

    Args:
        objects: Originating entity result set.
        field_code: Alias-resolved relationship field.

    Returns:
        Linked server IDs grouped by canonical target entity type.

    Raises:
        SgalchemyError: If a relationship value is not `None`, an entity, or a
            list of entities.
    """
    references: dict[str, set[int]] = {}
    for obj in objects:
        state = entity_state(obj)
        value = state.loaded.get(field_code)
        if value is None:
            links: list[Any] = []
        elif isinstance(value, list):
            links = cast('list[Any]', value)
        else:
            links = [value]
        for link in links:
            if not isinstance(link, Entity):
                raise SgalchemyError(
                    f'{state.entity_type}.{field_code} relationship value must be None, '
                    'an Entity, or a list containing only Entity objects'
                )
            linked_state = entity_state(link)
            if linked_state.id is not None:
                references.setdefault(linked_state.entity_type, set()).add(linked_state.id)
    return references


def missing_selectin_targets(
    reference_batches: Iterable[dict[str, set[int]]],
    binding: RegistryBinding | None,
) -> list[str]:
    """Return sorted observed target types absent from one binding.

    Args:
        reference_batches: Observed target identities grouped by relationship.
        binding: Complete operation binding, or captured absence.

    Returns:
        Sorted unique canonical target types absent from `binding`.
    """
    return sorted(
        {
            target_type
            for references in reference_batches
            for target_type in references
            if binding is None or target_type not in binding.schemas
        }
    )


def relationship_error(
    field: FieldExpr,
    missing: list[str],
    binding: RegistryBinding | None,
) -> RelationshipTargetUnavailableError:
    """Build one teaching error from observed targets and reflected context.

    Args:
        field: Relationship field whose observed targets are unavailable.
        missing: Sorted canonical target types absent after recovery.
        binding: Last complete binding checked by the loader.

    Returns:
        Public relationship-target error with reflected context.
    """
    field_code = field.filter_field
    owner = field.root_type
    owner_schema = None if binding is None else binding.schemas.get(owner)
    field_schema = None if owner_schema is None else owner_schema.fields.get(field_code)
    valid_types = [] if field_schema is None else sorted(field_schema.valid_types)
    return RelationshipTargetUnavailableError(
        f'{owner}.{field_code} observed target types {missing} remain absent after one '
        f'forced schema refresh; reflected valid_types={valid_types}; make those target '
        'schemas visible, then retry the explicit relationship load'
    )


def _selectin_binding(
    session: 'Session',
    opt: LazyLoad,
    references: dict[str, set[int]],
    binding: RegistryBinding,
    failed_bindings: list[RegistryBinding],
) -> RegistryBinding:
    """Preflight every observed target and force schema at most once."""
    missing = missing_selectin_targets([references], binding)
    if not missing:
        return binding
    if any(failed is binding for failed in failed_bindings):
        raise relationship_error(opt.field, missing, binding)
    context = f'raised while loading {opt.field.root_type}.{opt.field.filter_field}'
    refreshed = session.load_binding(context, force=True)
    remaining = missing_selectin_targets([references], refreshed)
    if remaining:
        if not any(failed is refreshed for failed in failed_bindings):
            failed_bindings.append(refreshed)
        raise relationship_error(opt.field, remaining, refreshed)
    return refreshed


def _execute_selectin(
    session: 'Session',
    objects: list[Entity],
    opt: LazyLoad,
    binding: RegistryBinding,
    *,
    failed_bindings: list[RegistryBinding],
) -> None:
    """Load one relationship across a result set, once per observed type."""
    references = collect_selectin_refs(objects, opt.field.filter_field)
    if references:
        binding = _selectin_binding(session, opt, references, binding, failed_bindings)
    _execute_selectin_refs(session, binding, references)


def union_selectin_refs(
    reference_batches: Iterable[dict[str, set[int]]],
) -> dict[str, set[int]]:
    """Merge target identities while retaining first-seen target order.

    Args:
        reference_batches: Target identities grouped by relationship.

    Returns:
        New target identity sets in first-seen target order.
    """
    merged: dict[str, set[int]] = {}
    for references in reference_batches:
        for target_type, ids in references.items():
            merged.setdefault(target_type, set()).update(ids)
    return merged


def execute_selectins(
    session: 'Session',
    objects: list[Entity],
    options: list[SelectinLoad],
    binding: RegistryBinding,
) -> None:
    """Preflight one target union, then run bounded select-in target reads.

    Missing target schema triggers at most one forced refresh before any
    secondary query.

    Args:
        session: Session owning the result objects and identity map.
        objects: Originating entity result set.
        options: Relationship options in caller order.
        binding: Complete primary-query binding.

    Raises:
        RelationshipTargetUnavailableError: If observed targets remain absent
            after one forced schema refresh.
        Exception: Original provider exception, annotated and unwrapped.
    """
    batches = [
        (option, collect_selectin_refs(objects, option.field.filter_field)) for option in options
    ]
    missing = missing_selectin_targets(
        (references for _option, references in batches),
        binding,
    )
    if missing:
        first_missing = next(
            option
            for option, references in batches
            if missing_selectin_targets([references], binding)
        )
        context = (
            f'raised while loading {first_missing.field.root_type}.'
            f'{first_missing.field.filter_field}'
        )
        refreshed = session.load_binding(context, force=True)
        for option, references in batches:
            remaining = missing_selectin_targets([references], refreshed)
            if remaining:
                raise relationship_error(option.field, remaining, refreshed)
        binding = refreshed
    merged_references = union_selectin_refs(references for _option, references in batches)
    pending_targets = [set(references) for _option, references in batches]
    retired_options: set[int] = set()

    def retire_completed(target_type: str | None) -> None:
        for index, (option, _references) in enumerate(batches):
            if index in retired_options:
                continue
            if target_type is not None:
                pending_targets[index].discard(target_type)
            if pending_targets[index]:
                continue
            field_code = option.field.filter_field
            for obj in objects:
                entity_state(obj).lazy.pop(field_code, None)
            retired_options.add(index)

    retire_completed(None)
    _execute_selectin_refs(
        session,
        binding,
        merged_references,
        on_target_complete=retire_completed,
        preserve_target_order=len(batches) > 1,
    )
