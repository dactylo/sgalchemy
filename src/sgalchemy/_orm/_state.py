"""Own session identity and hydrate provider records without I/O."""

import copy
import datetime
import logging
from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal, cast, overload

from sgalchemy._core._registry import (
    Entity,
    EntityState,
    Registry,
    RegistryBinding,
    assert_entity_access,
    entity_registry,
    entity_state,
    entity_type,
    initialize_entity,
)
from sgalchemy._errors import SgalchemyError
from sgalchemy._orm._uow import UnitOfWork, assert_entity_identity_consistent
from sgalchemy._shotgun import WireContainer
from sgalchemy.schema._model import EntitySchema, FieldSchema

_LOGGER = logging.getLogger('sgalchemy.session')


class SessionState:
    """Own one session's identity map and pure record-to-object transitions."""

    def _apply_record(
        self,
        obj: Entity,
        record: Mapping[str, Any],
        binding: RegistryBinding | None,
    ) -> None:
        """Apply flat fields, then fold deep fields into linked objects."""
        state = entity_state(obj)
        schema = None if binding is None else binding.schemas.get(state.entity_type)
        deep: list[tuple[str, Any]] = []
        for raw_key, value in record.items():
            if '.' in raw_key:
                link_field = raw_key.partition('.')[0]
                if link_field not in state.dirty:
                    deep.append((raw_key, value))
                continue
            if raw_key in {'type', 'id'} or raw_key in state.dirty:
                continue
            state.accept_field_value(raw_key, self._coerce(schema, raw_key, value, binding))
        for raw_key, value in sorted(deep, key=lambda item: item[0].count('.')):
            self._fold_deep_field(obj, raw_key, value, binding)

    def _assert_identity_consistent(self, obj: Entity) -> None:
        """Reject reflected-type, server-ID, or identity-map contradictions."""
        mapped_keys = [key for key, candidate in self._identity.items() if candidate is obj]
        assert_entity_identity_consistent(obj, mapped_keys)

    def _contains(self, obj: Entity) -> bool:
        """Return whether the identity map contains the exact object."""
        state = entity_state(obj)
        if type(state.id) is not int or state.id <= 0:
            return False
        return self._identity.get((state.entity_type, state.id)) is obj

    def _guard_entity_access(self, state: EntityState) -> None:
        """Enforce configured task access or this state's current ownership."""
        if self._session_access_guard is not None:
            self._session_access_guard()
        elif state.session is not self._owner:
            raise SgalchemyError('entity object is no longer owned by its staging session')

    def __init__(
        self,
        owner: object,
        registry: Registry,
        schema_provider: Callable[[], RegistryBinding | None],
        mark_schema_stale: Callable[[], None],
        entity_access_guard: Callable[[], None] | None = None,
    ) -> None:
        """Bind pure state decisions to one session owner and registry.

        Args:
            owner: Session that owns attached entities.
            registry: Registry that owns reflected entity classes.
            schema_provider: Callable returning the current prepared binding.
            mark_schema_stale: Callback invalidating stale prepared schema.
            entity_access_guard: Optional session access check for entity
                reads.
        """
        self._entity_access_guard = self._guard_entity_access
        self._identity: dict[tuple[str, int], Entity] = {}
        self._hydration_index: dict[tuple[str, int], list[Entity]] | None = None
        self._mark_schema_stale = mark_schema_stale
        self._owner = owner
        self._registry = registry
        self._schema_provider = schema_provider
        self._session_access_guard = entity_access_guard

    def _install_access_guard(self, obj: Entity) -> None:
        """Install this state owner's shared entity access guard."""
        entity_state(obj).access_guard = self._entity_access_guard

    def _shape_matches(self, field_schema: FieldSchema, value: object) -> bool:
        """Return whether a wire value has the cached field type's outer shape."""
        if value is None:
            matches = True
        elif field_schema.data_type == 'entity':
            matches = _is_canonical_link(value)
        elif field_schema.data_type == 'multi_entity':
            matches = isinstance(value, list) and all(
                _is_canonical_link(item) for item in cast('list[Any]', value)
            )
        elif field_schema.data_type in {'tag_list', 'addressing'}:
            matches = isinstance(value, list)
        elif field_schema.data_type in {'serializable', 'url'}:
            matches = isinstance(value, dict)
        elif field_schema.data_type == 'date':
            matches = isinstance(value, str) or (
                isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)
            )
        elif field_schema.data_type == 'date_time':
            matches = isinstance(value, (str, datetime.datetime))
        else:
            matches = not isinstance(value, (dict, list))
        return matches

    def _staged_identity_candidates(self) -> Collection[Entity]:
        """Return persistent staged identities when the owner has a unit of work."""
        unit_of_work = getattr(self._owner, 'unit_of_work', None)
        if not isinstance(unit_of_work, UnitOfWork):
            return ()
        return (*unit_of_work.deleted, *unit_of_work.revived)

    def _staged_for_identity(self, key: tuple[str, int]) -> Collection[Entity]:
        """Select every staged owner for one identity from this operation's index."""
        if self._hydration_index is not None:
            return self._hydration_index.get(key, ())
        return tuple(
            candidate
            for candidate in self._staged_identity_candidates()
            if (entity_state(candidate).entity_type, entity_state(candidate).id) == key
        )

    def _warn_shape(
        self,
        entity_type_name: str,
        field_code: str,
        data_type: str,
        value: object,
    ) -> None:
        """Report a schema contradiction and mark prepared schema stale."""
        _LOGGER.warning(
            'schema shape mismatch: %s.%s type=%s value=%r',
            entity_type_name,
            field_code,
            data_type,
            value,
        )
        self._mark_schema_stale()

    def _coerce(
        self,
        schema: EntitySchema | None,
        field_code: str,
        value: object,
        binding: RegistryBinding | None,
    ) -> object:
        """Coerce documented wire shapes while preserving contradictions."""
        field_schema = None if schema is None else schema.fields.get(field_code)
        if schema is None or field_schema is None:
            result = value
        elif not self._shape_matches(field_schema, value):
            self._warn_shape(schema.type_name, field_code, field_schema.data_type, value)
            result = value
        elif value is None:
            result = None
        elif field_schema.data_type == 'entity':
            link = cast('dict[str, Any]', value)
            result = self.hydrate_record(cast('str', link['type']), link, binding)
        elif field_schema.data_type == 'multi_entity':
            links = cast('list[dict[str, Any]]', value)
            result = [
                self.hydrate_record(cast('str', item['type']), item, binding) for item in links
            ]
        elif isinstance(value, str) and field_schema.data_type in {'date', 'date_time'}:
            try:
                if field_schema.data_type == 'date':
                    result = datetime.date.fromisoformat(value)
                else:
                    result = datetime.datetime.fromisoformat(value)
            except ValueError:
                self._warn_shape(schema.type_name, field_code, field_schema.data_type, value)
                result = value
        else:
            result = value
        if type(result) in (dict, list, tuple):
            return _copy_container_graph(cast('WireContainer', result))
        return result

    def _fold_deep_field(
        self,
        root: Entity,
        raw_key: str,
        value: object,
        binding: RegistryBinding | None,
    ) -> None:
        """Validate and fold one rendered deep field, preserving contradictions."""
        tokens = raw_key.split('.')
        current = root
        index = 0
        while index + 2 < len(tokens):
            field_code = tokens[index]
            expected_type = tokens[index + 1]
            state = entity_state(current)
            if field_code in state.dirty:
                return
            linked = state.loaded.get(field_code)
            schema = None if binding is None else binding.schemas.get(state.entity_type)
            field_schema = None if schema is None else schema.fields.get(field_code)
            if (
                field_code in state.loaded
                and linked is None
                and value is None
                and field_schema is not None
                and field_schema.data_type == 'entity'
                and expected_type in field_schema.valid_types
            ):
                entity_state(root).loaded.pop(raw_key, None)
                return
            if not isinstance(linked, Entity) or entity_state(linked).entity_type != expected_type:
                entity_state(root).loaded[raw_key] = value
                self._warn_shape(
                    entity_state(root).entity_type,
                    raw_key,
                    'deep_field',
                    value,
                )
                return
            current = linked
            index += 2
        if index != len(tokens) - 1:
            entity_state(root).loaded[raw_key] = value
            self._warn_shape(entity_state(root).entity_type, raw_key, 'deep_field', value)
            return
        if tokens[index] in entity_state(current).dirty:
            return
        self._apply_record(current, {tokens[index]: value}, binding)
        entity_state(root).loaded.pop(raw_key, None)

    def assert_identity_available(self, obj: Entity) -> None:
        """Reject another object mapped or staged under the same identity.

        Args:
            obj: Persistent object whose identity must remain unique.

        Raises:
            SgalchemyError: If a different object already owns the identity.
        """
        state = entity_state(obj)
        if state.id is None:
            return
        key = (state.entity_type, state.id)
        mapped = self._identity.get(key)
        mapped_collision = mapped is not None and mapped is not obj
        staged_collision = any(
            candidate is not obj for candidate in self._staged_for_identity(key)
        )
        if mapped_collision or staged_collision:
            raise SgalchemyError(
                f'session already tracks a different {state.entity_type}(id={state.id}) '
                'object; use session.get(type(obj), obj.id) to retrieve it, or close this '
                'session before staging the detached copy'
            )

    def _attach_hydrated(self, obj: Entity, key: tuple[str, int]) -> None:
        """Attach one canonical hydrated identity without scanning the identity map."""
        cls = type(obj)
        if entity_registry(cls) is not self._registry:
            raise SgalchemyError(
                f'{cls.__name__} instance belongs to a different registry than this '
                'session engine; build it from engine.entities / '
                'engine.registry.entity_class(...), or use a Session on its own engine'
            )
        state = entity_state(obj)
        if state.session is not self._owner:
            raise SgalchemyError('identity map contains an entity owned by another session')
        assert_entity_access(obj)
        assert_entity_identity_consistent(obj, [key], require_persistent=True)
        self.assert_identity_available(obj)
        self._install_access_guard(obj)

    def assert_same_registry(self, value: Entity | type[Entity]) -> None:
        """Reject a class or object owned by another registry or open session.

        Args:
            value: Entity class or instance to validate before local use.

        Raises:
            SgalchemyError: If `value` belongs to another registry or open
                session.
        """
        cls = type(value) if isinstance(value, Entity) else value
        if entity_registry(cls) is not self._registry:
            raise SgalchemyError(
                f'{cls.__name__} instance belongs to a different registry than this '
                'session engine; build it from engine.entities / '
                'engine.registry.entity_class(...), or use a Session on its own engine'
            )
        if isinstance(value, Entity):
            assert_entity_access(value)
            owner = entity_state(value).session
            if owner is not None and owner is not self._owner:
                raise SgalchemyError(
                    f'{cls.__name__} entity belongs to another open session; close that '
                    'session before staging it here'
                )

    def attach(self, obj: Entity) -> None:
        """Attach one compatible entity to this state owner.

        Args:
            obj: Entity to attach after registry and identity validation.

        Raises:
            SgalchemyError: If ownership, registry, or identity is
                incompatible.
        """
        self.assert_same_registry(obj)
        state = entity_state(obj)
        key = (state.entity_type, state.id) if type(state.id) is int and state.id > 0 else None
        if key is not None and self._identity.get(key) is obj:
            assert_entity_identity_consistent(obj, [key])
        else:
            assert_entity_identity_consistent(obj, [])
            self.assert_identity_available(obj)
        state.session = self._owner
        self._install_access_guard(obj)

    def detach_canceled(self, obj: Entity) -> None:
        """Detach a canceled object absent from this identity map.

        Args:
            obj: Entity removed from local staging.
        """
        if self._contains(obj):
            return
        state = entity_state(obj)
        state.lazy.clear()
        state.session = None
        state.access_guard = None

    def discard_identities(self, objs: Iterable[Entity]) -> None:
        """Remove every identity-map entry for the exact objects in one pass.

        Args:
            objs: Entities whose provisional identities must be removed.
        """
        object_ids = {id(obj) for obj in objs}
        if not object_ids:
            return
        keys = [key for key, candidate in self._identity.items() if id(candidate) in object_ids]
        for key in keys:
            self._identity.pop(key)

    def entity(self, entity_type_name: str, entity_id: int) -> Entity | None:
        """Return one identity-map entry without schema capture.

        Args:
            entity_type_name: Canonical entity type.
            entity_id: Positive server identifier.

        Returns:
            Existing entity, or `None`.
        """
        return self._identity.get((entity_type_name, entity_id))

    @contextmanager
    def hydration_scope(self) -> Generator[None, None, None]:
        """Reuse staged identities during one synchronous response hydration.

        Nested scopes share the outer index. Exit before provider I/O or
        staging changes; the next operation rebuilds from the authoritative raw
        lists.
        """
        if self._hydration_index is not None:
            yield
            return
        index: dict[tuple[str, int], list[Entity]] = {}
        for candidate in self._staged_identity_candidates():
            state = entity_state(candidate)
            if state.id is not None:
                index.setdefault((state.entity_type, state.id), []).append(candidate)
        self._hydration_index = index
        try:
            yield
        finally:
            self._hydration_index = None

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
            SgalchemyError: If identity is absent, malformed, or contradictory.
        """
        with self.hydration_scope():
            wire_type = record.get('type', entity_type_name)
            entity_id = record.get('id')
            if not _is_canonical_entity_type(entity_type_name):
                raise SgalchemyError("entity identity requires a non-empty string 'type'")
            if wire_type != entity_type_name:
                raise SgalchemyError(
                    f'wire entity type {wire_type!r} contradicts expected {entity_type_name!r}'
                )
            if type(entity_id) is not int or entity_id <= 0:
                raise SgalchemyError('entity identity requires a positive integer id')
            alias_binding = binding if binding is not None else self._registry.binding_snapshot()
            if (
                alias_binding is not None
                and entity_type_name not in alias_binding.schemas
                and entity_type_name in alias_binding.entity_aliases
            ):
                canonical = alias_binding.entity_aliases[entity_type_name]
                raise SgalchemyError(
                    f'wire entity type {entity_type_name!r} is a display alias for '
                    f'{canonical!r}; canonical wire links must use the canonical type'
                )
            key = (entity_type_name, entity_id)
            obj = self._identity.get(key)
            if obj is None:
                class_binding = binding
                if (
                    binding is not None
                    and entity_type_name not in binding.schemas
                    and entity_type_name not in binding.entity_aliases
                ):
                    class_binding = None
                cls = self._registry.entity_class_for(entity_type_name, binding=class_binding)
                obj = cls.__new__(cls)
                initialize_entity(
                    obj,
                    EntityState(
                        entity_type=entity_type_name,
                        id=entity_id,
                        loaded={'id': entity_id},
                        session=self._owner,
                    ),
                )
                self._attach_hydrated(obj, key)
                self._identity[key] = obj
            else:
                self._attach_hydrated(obj, key)
            state = entity_state(obj)
            if retired is not None and not any(
                candidate is obj for candidate in self._staged_for_identity(key)
            ):
                state.deleted = retired
            previously_loaded = frozenset(key for key in state.loaded if '.' not in key)
            state.loaded['id'] = entity_id
            self._apply_record(obj, record, binding)
            direct_fields = {key for key in record if key != 'type' and '.' not in key}
            if previously_loaded <= direct_fields:
                state.expired = False
            return obj

    def _hydrate(self, entity_type_name: str, record: Mapping[str, Any]) -> Entity:
        """Capture prepared binding and hydrate one wire record."""
        binding = self._schema_provider()
        return self.hydrate_record(entity_type_name, record, binding)

    @property
    def identity_map(self) -> dict[tuple[str, int], Entity]:
        """Return the state-owned mutable identity map.

        Returns:
            Shared storage used by state decisions and compatibility views.
        """
        return self._identity

    def identity_values(self) -> list[Entity]:
        """Return a snapshot of identity-mapped objects.

        Returns:
            Objects currently held by this state identity map.
        """
        return list(self._identity.values())

    def merge_dict(self, raw: Mapping[str, Any]) -> Entity:
        """Adopt one raw entity dictionary without hidden I/O.

        Args:
            raw: Entity dictionary containing canonical type and id.

        Returns:
            Identity-mapped entity.

        Raises:
            SgalchemyError: If root identity is not canonical.
        """
        type_name = raw.get('type')
        if not isinstance(type_name, str) or not type_name:
            raise SgalchemyError("entity identity requires a non-empty string 'type'")
        return self._hydrate(type_name, raw)

    def owns_entity(self, obj: Entity) -> bool:
        """Return whether this state owner owns `obj`.

        Args:
            obj: Candidate entity.

        Returns:
            Whether this owner may mutate `obj`.
        """
        return entity_state(obj).session is self._owner

    def close(self) -> None:
        """Detach owned entities and clear all identity references."""
        for obj in self._identity.values():
            if not self.owns_entity(obj):
                continue
            state = entity_state(obj)
            state.lazy.clear()
            state.session = None
            state.access_guard = None
        self._identity.clear()

    def register_identity(self, obj: Entity) -> None:
        """Register one persistent object after collision-safe validation.

        Args:
            obj: Persistent object to register.

        Raises:
            RuntimeError: If `obj` has no server id.
            SgalchemyError: If identity is inconsistent or unavailable.
        """
        self._assert_identity_consistent(obj)
        self.assert_identity_available(obj)
        state = entity_state(obj)
        if state.id is None:
            raise RuntimeError('persistent identity registration requires a server id')
        self._identity[(state.entity_type, state.id)] = obj

    def require_owned_class(self, cls: type[Entity]) -> str:
        """Return a class type after enforcing this registry's ownership.

        Args:
            cls: Reflected entity class.

        Returns:
            Canonical entity type.

        Raises:
            SgalchemyError: If `cls` belongs to another registry.
        """
        self.assert_same_registry(cls)
        return entity_type(cls)

    def require_owned_object(self, obj: Entity) -> EntityState:
        """Return state after enforcing registry and state ownership.

        Args:
            obj: Entity required to belong to this owner.

        Returns:
            Owned entity state.

        Raises:
            SgalchemyError: If `obj` belongs to another registry or owner.
        """
        self.assert_same_registry(obj)
        state = entity_state(obj)
        if state.session is not self._owner:
            raise SgalchemyError('entity object must be owned by this session')
        return state

    def expire(self, obj: Entity) -> None:
        """Mark one state-owned entity expired.

        Args:
            obj: Entity to expire.

        Raises:
            SgalchemyError: If this state owner does not own `obj`.
        """
        self.require_owned_object(obj).expired = True

    def expire_all(self) -> None:
        """Mark every state-owned entity expired."""
        objects = list(self._identity.values())
        for obj in objects:
            self.require_owned_object(obj)
        for obj in objects:
            entity_state(obj).expired = True


def _copy_container_graph(value: WireContainer) -> WireContainer:
    """Detach exact wire-container topology while retaining opaque leaves."""
    container_types = (dict, list, tuple)
    memo: dict[int, Any] = {}
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) not in container_types:
            memo[id(item)] = item
            continue
        container = cast('dict[Any, Any] | list[Any] | tuple[Any, ...]', item)
        item_id = id(container)
        if item_id in seen:
            continue
        seen.add(item_id)
        if isinstance(container, dict):
            for key, child in container.items():
                pending.extend((key, child))
        else:
            pending.extend(container)
    return copy.deepcopy(value, memo)


def _is_canonical_entity_type(value: object) -> bool:
    """Return whether a value is one canonical wire entity type."""
    return isinstance(value, str) and bool(value)


def _is_canonical_link(value: object) -> bool:
    """Return whether a value is one valid server entity-link dictionary."""
    if not isinstance(value, dict):
        return False
    link = cast('dict[str, object]', value)
    type_name = link.get('type')
    entity_id = link.get('id')
    return (
        isinstance(type_name, str) and bool(type_name) and type(entity_id) is int and entity_id > 0
    )


@overload
def validate_point_records(
    records: Sequence[object],
    entity_type_name: str,
    entity_id: int,
    *,
    allow_absent: Literal[False],
) -> dict[str, Any]: ...


@overload
def validate_point_records(
    records: Sequence[object],
    entity_type_name: str,
    entity_id: int,
    *,
    allow_absent: Literal[True],
) -> dict[str, Any] | None: ...


def validate_point_records(
    records: Sequence[object],
    entity_type_name: str,
    entity_id: int,
    *,
    allow_absent: bool,
) -> dict[str, Any] | None:
    """Return one record only when cardinality and identity are exact.

    Args:
        records: Point-read records before hydration.
        entity_type_name: Canonical requested entity type.
        entity_id: Positive requested server identifier.
        allow_absent: Whether zero records should return `None`.

    Returns:
        The sole exact record, or `None` for an allowed absent result.

    Raises:
        SgalchemyError: If cardinality, record shape, type, or ID contradicts
            the requested identity.
    """
    if not records:
        if allow_absent:
            return None
        raise SgalchemyError(f'{entity_type_name}(id={entity_id}) no longer exists on the server')
    if len(records) != 1:
        raise SgalchemyError(
            f'point read for {entity_type_name}(id={entity_id}) requires exactly one '
            f'record; provider returned {len(records)}'
        )
    record = records[0]
    if not isinstance(record, dict):
        raise SgalchemyError('point-read payload must contain a dictionary record')
    typed = cast('dict[str, Any]', record)
    wire_type = typed.get('type')
    wire_id = typed.get('id')
    if wire_type != entity_type_name or type(wire_id) is not int or wire_id != entity_id:
        raise SgalchemyError(
            f'point read requested {entity_type_name}(id={entity_id}) but provider '
            f'returned {wire_type}(id={wire_id!r})'
        )
    return typed
