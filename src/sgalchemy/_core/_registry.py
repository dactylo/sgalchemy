"""Manage entity classes, instrumentation, and registry bindings."""

from __future__ import annotations

import os
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol, Self, cast

from sgalchemy._core import _entity
from sgalchemy._core._expressions import FieldExpr, Filter, FilterGroup, RawFilters
from sgalchemy._errors import (
    RefreshRequiredError,
    SchemaUnavailableError,
    SgalchemyError,
    StaleObjectError,
    UnfetchedFieldError,
)
from sgalchemy._shotgun import (
    THUMBNAIL_COUPLED_CLEAR_FIELD,
    THUMBNAIL_UPLOAD_METHODS,
    WireRecord,
    WireTraversalContainer,
)
from sgalchemy.schema._aliases import entity_aliases, field_aliases
from sgalchemy.schema._model import EntitySchema, FieldSchema

REGISTRY_SITE_MISMATCH_MESSAGE: str = (
    'registry is already bound to a different site (one site per Registry); '
    'pass registry=Registry() or bind_default=False to the engine factory for a second site'
)

_DEFAULT_REGISTRIES: list[Registry] = []
_DEFAULT_REGISTRY_LOCK = Lock()
_DEFAULT_SESSION_REGISTRIES: list[Registry] = []
_DEFAULT_TRANSITION: dict[str, object] = {'reserved': False}
_ENTITY_FINDERS: list[
    Callable[
        [type[Entity], tuple[Filter | FilterGroup | RawFilters, ...], dict[str, object]],
        list[Entity],
    ]
] = []
# Keep public callback annotations resolvable through `typing.get_type_hints()`.
_RUNTIME_ANNOTATION_TYPES = (Callable, EntitySchema, FieldSchema)


class _EntityMeta(type):
    """Metaclass giving entity classes `FieldExpr` class attributes."""

    def __getattr__(cls, name: str) -> FieldExpr:
        """Return the field expression for `name` (schema-bound when available)."""
        if name.startswith('_'):
            raise AttributeError(name)
        if cls is Entity:
            raise AttributeError(name)
        entity_cls = cast('type[object]', cls)
        registry = entity_registry(entity_cls)
        type_name = entity_type(entity_cls)
        binding = None if registry is None else registry.binding_snapshot()
        if binding is None:
            return FieldExpr(type_name, (name,), None)
        schema = binding.schemas.get(type_name)
        if schema is None:
            return FieldExpr(type_name, (name,), None)
        code = (
            name if name in schema.fields else binding.field_aliases.get(type_name, {}).get(name)
        )
        if code is None:
            raise AttributeError(f'{type_name} has no field or field alias {name!r}')
        field_schema = schema.fields[code]
        if not field_schema.api_readable:
            raise AttributeError(
                f'{type_name}.{code} is a UI-only query/summary field; it returns no '
                'usable data through the API'
            )
        return FieldExpr(type_name, (code,), field_schema, schemas=binding.schemas)


@dataclass(frozen=True)
class InspectionInfo:
    """Capture identity, lifecycle, and field trust without I/O.

    `inspect()` copies the outer field mappings and trust sets. Nested field
    values remain shared; this is not a deep copy.

    Attributes:
        entity_type: Canonical entity type.
        id: Server identity, or `None` for a pending entity.
        loaded: Current locally loaded values, including staged assignments.
        dirty: Values staged for writing.
        expired: Whether ordinary field access requires refresh.
        deleted: Locally observed retired state.
        refresh_required: Fields lacking a trustworthy readable value.
        unverified_writes: Fields with a readable current provider value whose
            write provenance remains uncertain.
    """

    entity_type: str
    id: int | None
    loaded: dict[str, Any]
    dirty: dict[str, Any]
    expired: bool
    deleted: bool
    refresh_required: frozenset[str] = frozenset()
    unverified_writes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RegistryBinding:
    """Capture one atomically published site schema and its aliases.

    Registry-produced bindings detach and protect their mapping layers.
    Retaining a binding preserves that operation's schema generation across
    later Registry refreshes.

    Attributes:
        key: Site identity used to prevent cross-site rebinding.
        schemas: Canonical entity names mapped to parsed schemas.
        entity_aliases: Display-name identifiers mapped to canonical types.
        field_aliases: Per-entity alias identifiers mapped to field codes.
        alias_reports: Diagnostics for skipped alias collisions.
    """

    key: str
    schemas: Mapping[str, EntitySchema]
    entity_aliases: Mapping[str, str]
    field_aliases: Mapping[str, Mapping[str, str]]
    alias_reports: tuple[str, ...]


class _LazyResolver(Protocol):
    """Resolve a deferred field value through its owning session."""

    def resolve(self, field_code: str) -> None:
        """Resolve the deferred field identified by its provider code."""


def _new_field_set() -> set[str]:
    """Return one independently mutable field-code set."""
    return set()


def _new_lazy_dict() -> dict[str, _LazyResolver]:
    """Return one independently mutable deferred-loader mapping."""
    return {}


def _new_state_dict() -> dict[str, Any]:
    """Return one independently mutable field-state mapping."""
    return {}


def _new_type_dict() -> dict[str, str]:
    """Return one independently mutable dirty-field type mapping."""
    return {}


@dataclass
class EntityState:
    """Hold mutable identity, field values, and write provenance for an entity.

    Session and unit-of-work code maintain this state together. The first
    assignment to a field captures its prior value and trust markers for local
    rollback.

    Attributes:
        entity_type: Canonical entity type.
        id: Positive server identity, or `None` while pending.
        loaded: Locally available values, including staged assignments.
        dirty: Field values staged for writing.
        clean: Loaded values captured before each field's first staged write.
        lazy: Deferred relationship-loading contexts keyed by field code.
        session: Current owner, or `None` while detached.
        access_guard: Optional owner access check invoked before entity use.
        expired: Whether ordinary field access requires refresh.
        deleted: Locally observed retired state.
        indeterminate: Whether an ambiguous entity-level outcome forbids use.
        dirty_types: Reflected data types captured when fields were assigned.
        refresh_required: Fields without trustworthy readable values.
        unverified_writes: Readable fields with uncertain write provenance.
        clean_refresh_required: Pre-assignment refresh-required markers.
        clean_unverified_writes: Pre-assignment unverified-write markers.
    """

    entity_type: str
    id: int | None
    loaded: dict[str, Any] = dc_field(default_factory=_new_state_dict)
    dirty: dict[str, Any] = dc_field(default_factory=_new_state_dict)
    clean: dict[str, Any] = dc_field(default_factory=_new_state_dict)
    lazy: dict[str, _LazyResolver] = dc_field(default_factory=_new_lazy_dict)
    session: object | None = None
    access_guard: Callable[[EntityState], None] | None = None
    expired: bool = False
    deleted: bool = False
    indeterminate: bool = False
    dirty_types: dict[str, str] = dc_field(default_factory=_new_type_dict)
    refresh_required: set[str] = dc_field(default_factory=_new_field_set)
    unverified_writes: set[str] = dc_field(default_factory=_new_field_set)
    clean_refresh_required: set[str] = dc_field(default_factory=_new_field_set)
    clean_unverified_writes: set[str] = dc_field(default_factory=_new_field_set)

    def _capture_assignment_snapshot(self, field_code: str) -> None:
        """Capture one field's clean value and trust markers before its first write."""
        if field_code in self.dirty:
            return
        if field_code in self.loaded:
            self.clean[field_code] = self.loaded[field_code]
        if field_code in self.refresh_required:
            self.clean_refresh_required.add(field_code)
        if field_code in self.unverified_writes:
            self.clean_unverified_writes.add(field_code)

    def accept_field_value(self, field_code: str, value: object) -> None:
        """Store a readable field value and replace its trust markers.

        Args:
            field_code: Provider field code whose value was established.
            value: Readable value to retain locally.
        """
        self.loaded[field_code] = value
        self.lazy.pop(field_code, None)
        self.refresh_required.discard(field_code)
        self.unverified_writes.discard(field_code)

    def accept_unstaged_value(self, field_code: str, value: object) -> None:
        """Store an unstaged value and discard stale first-write trust markers.

        Args:
            field_code: Provider field code whose unstaged value was
                established.
            value: Readable value to retain locally.
        """
        self.accept_field_value(field_code, value)
        self.clean_refresh_required.discard(field_code)
        self.clean_unverified_writes.discard(field_code)

    def discard_field_provenance(self, field_code: str) -> None:
        """Discard one field's clean snapshot and staged write.

        Args:
            field_code: Provider field code whose write provenance is retired.
        """
        self.clean.pop(field_code, None)
        self.clean_refresh_required.discard(field_code)
        self.clean_unverified_writes.discard(field_code)
        self.dirty.pop(field_code, None)
        self.dirty_types.pop(field_code, None)

    def _retire_field_for_refresh(self, field_code: str) -> None:
        """Retire one field's value and write provenance pending a refresh."""
        self.discard_field_provenance(field_code)
        self.loaded.pop(field_code, None)
        self.refresh_required.add(field_code)

    def mark_field_unknown(self, field_code: str) -> None:
        """Forget an unreadable value while retaining its write provenance.

        Args:
            field_code: Provider field code whose value is unknown.
        """
        self.loaded.pop(field_code, None)
        self.refresh_required.add(field_code)

    def mark_writes_unverified(self, field_codes: Iterable[str]) -> None:
        """Mark unstaged fields whose confirmed reads leave write provenance uncertain.

        Args:
            field_codes: Provider field codes from a confirmed readback.
        """
        self.unverified_writes.update(code for code in field_codes if code not in self.dirty)

    def restore_field_from_snapshot(self, field_code: str) -> None:
        """Restore one field's clean value and trust markers, then retire its write.

        Args:
            field_code: Provider field code whose first-write snapshot is
                restored.
        """
        if field_code in self.clean:
            self.loaded[field_code] = self.clean[field_code]
        else:
            self.loaded.pop(field_code, None)
        self.refresh_required.discard(field_code)
        if field_code in self.clean_refresh_required:
            self.refresh_required.add(field_code)
        self.unverified_writes.discard(field_code)
        if field_code in self.clean_unverified_writes:
            self.unverified_writes.add(field_code)
        self.discard_field_provenance(field_code)

    def restore_staged_fields(self) -> None:
        """Restore staged fields and discard orphaned first-write provenance."""
        field_codes = tuple(
            dict.fromkeys(
                (
                    *self.dirty,
                    *self.clean,
                    *self.dirty_types,
                    *self.clean_refresh_required,
                    *self.clean_unverified_writes,
                )
            )
        )
        for field_code in field_codes:
            if field_code in self.dirty:
                self.restore_field_from_snapshot(field_code)
            else:
                self.discard_field_provenance(field_code)

    def retire_confirmed_write(self, field_code: str) -> None:
        """Discard a confirmed write's staged provenance and uncertainty.

        Args:
            field_code: Provider field code confirmed by the completed write.
        """
        self.discard_field_provenance(field_code)
        self.unverified_writes.discard(field_code)

    def retire_upload_field(self, field_code: str) -> None:
        """Retire an upload and forget its uncertain write marker.

        Args:
            field_code: Provider image field to reload.
        """
        self._retire_field_for_refresh(field_code)
        self.unverified_writes.discard(field_code)


class Entity(metaclass=_EntityMeta):
    """Represent a reflected or opaque canonical entity with local ORM state.

    Obtain classes from a Registry or `engine.entities`; create pending
    instances with `new()` or retrieve persistent instances through a session.
    Public field assignment stages local intent without provider I/O.

    Reads preserve identity and loaded values, reject expired or untrustworthy
    fields, and raise when required schema or projection data is unavailable.
    Only an explicitly configured synchronous lazy loader may perform field
    access I/O. Opaque entities retain identity and loaded wire values but
    cannot be created or mutated until their schema becomes available.

    Private attributes bypass reflected-field handling. Representations remain
    I/O-free across lifecycle states.
    """

    _sg_state: EntityState

    def _sg_assign_image(self, code: str, value: object, state: EntityState) -> bool:
        """Validate image intent and return whether generic staging remains."""
        if code not in THUMBNAIL_UPLOAD_METHODS:
            raise SgalchemyError(
                f'{state.entity_type}.{code} is an unsupported image assignment; '
                'local paths are accepted only for image and filmstrip_image'
            )
        if value is not None:
            if not isinstance(value, (str, os.PathLike)):
                raise SgalchemyError(
                    f'{state.entity_type}.{code} accepts only a local str or os.PathLike path'
                )
            return True
        if code == THUMBNAIL_COUPLED_CLEAR_FIELD:
            raise SgalchemyError(
                f'{state.entity_type}.filmstrip_image=None is ignored by the provider; '
                'use image=None to clear both thumbnails'
            )
        if THUMBNAIL_COUPLED_CLEAR_FIELD in state.dirty:
            state.restore_field_from_snapshot(THUMBNAIL_COUPLED_CLEAR_FIELD)
        if state.id is not None:
            return True
        state.restore_field_from_snapshot(code)
        state.refresh_required.discard(code)
        state.unverified_writes.discard(code)
        state.loaded[code] = None
        return False

    @classmethod
    def _sg_binding(cls) -> RegistryBinding | None:
        """Capture this type's complete registry binding once."""
        registry = entity_registry(cls)
        return None if registry is None else registry.binding_snapshot()

    def _sg_prepare_assignment(
        self,
        field_schema: FieldSchema,
        code: str,
        value: object,
        state: EntityState,
    ) -> bool:
        """Validate field-specific assignment and return whether generic staging remains."""
        if code in THUMBNAIL_UPLOAD_METHODS or field_schema.data_type == 'image':
            return self._sg_assign_image(code, value, state)
        if not field_schema.editable:
            raise SgalchemyError(
                f'{state.entity_type}.{code} is reflected as noneditable; cached editability '
                'may be stale, so call engine.load_schema(force=True) and assign again; '
                'provider permissions remain authoritative'
            )
        return True

    @staticmethod
    def _sg_render_name_part(name_field: str | None, state: EntityState) -> str:
        """Render an optional loaded name segment without letting value repr escape."""
        if (
            name_field is None
            or name_field not in state.loaded
            or name_field in state.refresh_required
            or state.expired
        ):
            return ''
        rendered: str | None = None
        with suppress(Exception):
            rendered = repr(state.loaded[name_field])
        if rendered is None:
            return ''
        return f', {name_field}={rendered}'

    def __repr__(self) -> str:
        """Return the concise entity representation without performing I/O."""
        state = self.__dict__.get('_sg_state')
        if state is None:
            return f'{type(self).__name__}(uninitialized)'
        if state.indeterminate:
            return f'{state.entity_type}(indeterminate)'
        binding = self._sg_binding()
        schema = None if binding is None else binding.schemas.get(state.entity_type)
        name_field = None if schema is None else schema.name_field
        name_part = self._sg_render_name_part(name_field, state)
        if state.id is None:
            return f'{state.entity_type}(unsaved{name_part})'
        return f'{state.entity_type}(id={state.id}{name_part})'

    def _sg_resolve(self, name: str, binding: RegistryBinding | None) -> str:
        """Resolve a field alias through one captured binding."""
        if binding is None:
            return name
        return binding.field_aliases.get(self._sg_state.entity_type, {}).get(name, name)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - Reflected fields have schema-selected runtime value types.
        """Read a loaded field or resolve its explicitly configured lazy load.

        Args:
            name: Reflected field code or supported field alias.

        Returns:
            Current loaded value after any configured synchronous lazy load.

        Raises:
            AttributeError: If the name is private, unknown, or API-unreadable.
            StaleObjectError: If the entity is expired and requires refresh.
            RefreshRequiredError: If the field has no trustworthy current
                value.
            SchemaUnavailableError: If the value is not loaded and the entity
                schema is unavailable.
            UnfetchedFieldError: If a known readable field was not loaded.
            SgalchemyError: If entity ownership or outcome state forbids
                access, or the configured relationship load cannot proceed.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        assert_entity_access(self)
        state = self._sg_state
        binding = self._sg_binding()
        code = self._sg_resolve(name, binding)
        if state.expired:
            raise StaleObjectError(
                f'{state.entity_type}(id={state.id}) is expired; call '
                f'session.refresh(obj) before reading {code!r}'
            )
        context = state.lazy.get(code)
        if context is not None:
            context.resolve(code)
        if code in state.refresh_required:
            raise RefreshRequiredError(
                f'{state.entity_type}(id={state.id}) field {code!r} has no trustworthy '
                'current value; call session.refresh(obj) or await session.refresh(obj) '
                'before reading it'
            )
        if code in state.loaded:
            return state.loaded[code]
        schema = None if binding is None else binding.schemas.get(state.entity_type)
        if schema is None:
            raise SchemaUnavailableError(
                f'{state.entity_type}(id={state.id}) schema is unavailable; call '
                'engine.load_schema(force=True) before reading fields'
            )
        if code in schema.fields:
            if not schema.fields[code].api_readable:
                raise AttributeError(
                    f'{state.entity_type}.{code} is a UI-only query/summary field; it '
                    'returns no usable data through the API'
                )
            raise UnfetchedFieldError(
                f'{state.entity_type}.{code} was not loaded by this query; add '
                f'load_only({state.entity_type}.{code}) to load it, or '
                'joinedload()/selectinload() if it is a relationship'
            )
        raise AttributeError(name)

    def _sg_assign(self, name: str, value: object, binding: RegistryBinding | None) -> None:
        """Validate and stage one field through one captured binding."""
        state = self._sg_state
        schema = None if binding is None else binding.schemas.get(state.entity_type)
        if schema is None:
            raise SchemaUnavailableError(
                f'{state.entity_type} schema is unavailable; call '
                'engine.load_schema(force=True) before field assignment'
            )
        code = self._sg_resolve(name, binding)
        if code == 'id':
            raise AttributeError(
                f'{state.entity_type}.id is assigned by the server (hydration and create '
                'results set it); to reference an existing entity use session.get'
            )
        field_schema = schema.fields.get(code)
        if field_schema is None:
            raise AttributeError(
                f'{state.entity_type} has no field or field alias {name!r}; nothing was staged'
            )
        if not field_schema.api_readable:
            raise AttributeError(
                f'{state.entity_type}.{code} is a UI-only query/summary field '
                'and cannot be written'
            )
        if not self._sg_prepare_assignment(field_schema, code, value, state):
            return
        state._capture_assignment_snapshot(code)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        state.refresh_required.discard(code)
        state.unverified_writes.discard(code)
        state.dirty[code] = value
        state.dirty_types[code] = field_schema.data_type
        state.loaded[code] = value

    def __setattr__(self, name: str, value: object) -> None:
        """Stage a reflected field assignment without provider I/O.

        Ordinary field writes resolve aliases, capture first-write rollback
        state, and replace loaded and dirty values. Private attributes use
        ordinary attribute storage. Setting `image=None` on a pending entity
        cancels pending thumbnail upload intent without staging a clear.

        Args:
            name: Field code, supported alias, or private attribute name.
            value: New local intent. Thumbnail fields accept local string or
                PathLike paths. For persistent entities, `image=None` stages a
                coupled thumbnail clear. `filmstrip_image=None` is rejected.

        Raises:
            AttributeError: If a public field is unknown, API-unreadable, or
                the server-assigned identity.
            SchemaUnavailableError: If the entity schema is unavailable.
            SgalchemyError: If access is unsafe, the field is noneditable, or
                the thumbnail assignment is unsupported.
        """
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        assert_entity_access(self)
        binding = self._sg_binding()
        self._sg_assign(name, value, binding)

    def as_dict(self) -> dict[str, Any]:
        """Return trustworthy loaded values in recursive wire shape.

        Returns:
            Type and id plus loaded fields. Refresh-required fields are
            omitted;
            loaded unverified fields are included.

        Raises:
            SgalchemyError: If this entity or an included linked entity is
                unsafe to access, or an active container reference cycle is
                found.
        """
        assert_entity_access(self)
        binding = self._sg_binding()

        def wire_value(value: object, active: set[int]) -> object:
            if isinstance(value, Entity):
                assert_entity_access(value)
                linked_state = entity_state(value)
                link: WireRecord = {'type': linked_state.entity_type, 'id': linked_state.id}
                linked_schema = (
                    None if binding is None else binding.schemas.get(linked_state.entity_type)
                )
                name_field = None if linked_schema is None else linked_schema.name_field
                if (
                    name_field is not None
                    and name_field in linked_state.loaded
                    and name_field not in linked_state.refresh_required
                ):
                    link['name'] = linked_state.loaded[name_field]
                elif 'name' in linked_state.loaded and 'name' not in linked_state.refresh_required:
                    link['name'] = linked_state.loaded['name']
                return link
            children = container_children(value)
            if children is None:
                return value
            container = cast('WireTraversalContainer', value)
            container_id = id(container)
            if container_id in active:
                raise SgalchemyError(
                    'container value contains a reference cycle; cyclic containers '
                    'cannot be serialized by as_dict()'
                )
            active.add(container_id)
            try:
                if isinstance(container, Mapping):
                    return {key: wire_value(item, active) for key, item in container.items()}
                if isinstance(container, list):
                    return [wire_value(item, active) for item in children]
                return tuple(wire_value(item, active) for item in children)
            finally:
                active.discard(container_id)

        state = self._sg_state
        result: dict[str, Any] = {'type': state.entity_type, 'id': state.id}
        result.update(
            (key, wire_value(value, set()))
            for key, value in state.loaded.items()
            if key != 'id' and key not in state.refresh_required
        )
        return result

    @classmethod
    def find(
        cls,
        *conditions: Filter | FilterGroup | RawFilters,
        **eq_filters: object,
    ) -> list[Self]:
        """Find entities through the installed synchronous default session.

        Args:
            *conditions: Filter expressions joined with AND.
            **eq_filters: Field or alias names matched for equality.

        Returns:
            Hydrated entities from the default session.

        Raises:
            SgalchemyError: If no default session is configured, or the ORM
                finder has not been installed.
        """
        if not _ENTITY_FINDERS:
            raise SgalchemyError(
                'no default session is configured; call sgalchemy.connect(...) first'
            )
        return cast('list[Self]', _ENTITY_FINDERS[0](cls, conditions, eq_filters))

    @classmethod
    def new(cls, **fields: object) -> Self:
        """Construct one schema-validated pending entity without I/O.

        Args:
            **fields: Bound-schema field codes or aliases to stage.

        Returns:
            New pending entity with validated dirty state.

        Raises:
            SgalchemyError: If schema, editability, or field-value validation
                fails.
            AttributeError: If a field is unknown, UI-only, or server-assigned.
        """
        binding = cls._sg_binding()
        if binding is None or entity_type(cls) not in binding.schemas:
            raise SchemaUnavailableError(
                f'{entity_type(cls)} schema is unavailable; call '
                'engine.load_schema(force=True) before Entity.new()'
            )
        obj = cls.__new__(cls)
        initialize_entity(obj, EntityState(entity_type=entity_type(cls), id=None))
        for name, value in fields.items():
            cls._sg_assign(obj, name, value, binding)
        return obj


def container_children(value: object) -> Iterable[object] | None:
    """Return children for a supported recursive container, or `None`.

    Args:
        value: Candidate Mapping, list, tuple, or leaf value.

    Returns:
        Mapping values, the original list or tuple, or `None` for a leaf.
    """
    if isinstance(value, Mapping):
        return cast('Mapping[object, object]', value).values()
    if isinstance(value, list):
        return cast('list[object]', value)
    if isinstance(value, tuple):
        return cast('tuple[object, ...]', value)
    return None


def entity_reserved_attribute_names() -> frozenset[str]:
    """Return names unavailable to reflected field aliases.

    Returns:
        Attributes already resolved by `Entity`, `_EntityMeta`, `object`, or
        `type` before dynamic field lookup.
    """
    return frozenset(dir(Entity)) | frozenset(dir(_EntityMeta))


# Registry
# --------


class Registry:
    """Own stable entity classes and atomically published schema bindings.

    An unbound Registry can create symbolic classes by name. Binding supplies
    field metadata and display aliases without replacing existing canonical
    classes. A Registry accepts only one site identity; later binds refresh
    that same site's schema atomically.

    Retained binding snapshots preserve an operation's schema generation.
    Registry operations perform no provider or schema-cache I/O.
    """

    def __init__(self) -> None:
        """Create an unbound registry."""
        self._bind_lock = Lock()
        self._binding: RegistryBinding | None = None
        self._class_lock = Lock()
        self._classes: dict[str, type[Entity]] = {}

    def _build(self, canonical: str) -> type[Entity]:
        """Create, cache, and return one canonical entity class."""
        cls = cast('type[Entity]', _EntityMeta(canonical, (Entity,), {}))
        _entity.register_entity_class(cls, canonical, self)
        self._classes[canonical] = cls
        return cls

    def _make_binding(
        self,
        schemas: Mapping[str, EntitySchema],
        key: str,
    ) -> RegistryBinding:
        """Build one complete immutable candidate before publication."""
        detached = {
            type_name: replace(
                entity,
                fields=MappingProxyType(dict(entity.fields)),
            )
            for type_name, entity in schemas.items()
        }
        entity_alias_map, reports = entity_aliases(detached)
        field_alias_map: dict[str, dict[str, str]] = {}
        reserved_names = entity_reserved_attribute_names()
        for type_name, entity in detached.items():
            aliases, field_reports = field_aliases(entity, reserved_names=reserved_names)
            field_alias_map[type_name] = aliases
            reports.extend(field_reports)
        return RegistryBinding(
            key=key,
            schemas=MappingProxyType(detached),
            entity_aliases=MappingProxyType(dict(entity_alias_map)),
            field_aliases=MappingProxyType(
                {
                    name: MappingProxyType(dict(aliases))
                    for name, aliases in field_alias_map.items()
                }
            ),
            alias_reports=tuple(reports),
        )

    @property
    def alias_reports(self) -> list[str]:
        """Return reports from one complete binding snapshot.

        Returns:
            Copy of alias collision reports.
        """
        binding = self._binding
        return [] if binding is None else list(binding.alias_reports)

    def bind(self, schemas: Mapping[str, EntitySchema], *, key: str) -> None:
        """Bind one site's schemas or atomically refresh the same site.

        Args:
            schemas: Canonical entity schemas.
            key: Site identity and schema cache key.

        Raises:
            SgalchemyError: If already bound to a different site.
        """
        candidate = self._make_binding(schemas, key)
        with self._bind_lock:
            current = self._binding
            if current is not None and key != current.key:
                raise SgalchemyError(REGISTRY_SITE_MISMATCH_MESSAGE)
            self._binding = candidate

    def binding_snapshot(self) -> RegistryBinding | None:
        """Return the current complete immutable binding snapshot.

        Returns:
            Current binding, or `None` while unbound.
        """
        return self._binding

    @property
    def bound_key(self) -> str | None:
        """Return the bound site key.

        Returns:
            Site key, or `None` while unbound.
        """
        binding = self._binding
        return None if binding is None else binding.key

    def entity_class_for(
        self,
        name: str,
        *,
        binding: RegistryBinding | None,
    ) -> type[Entity]:
        """Return one cached class resolved through a captured binding.

        Args:
            name: Canonical type name or an alias known to `binding`.
            binding: Complete operation snapshot, or `None` for symbolic
                cache-only resolution.

        Returns:
            Stable instrumented entity class.

        Raises:
            AttributeError: If `binding` does not know `name`.
        """
        if binding is None:
            canonical = name
        else:
            canonical = name if name in binding.schemas else binding.entity_aliases.get(name)
            if canonical is None:
                raise AttributeError(f'unknown entity type or alias {name!r}')
        with self._class_lock:
            return self._classes.get(canonical) or self._build(canonical)

    def entity_class(self, name: str) -> type[Entity]:
        """Return one cached class resolved through the current binding.

        Args:
            name: Canonical type name, or a display alias when bound.

        Returns:
            Stable instrumented entity class.

        Raises:
            AttributeError: If a bound registry does not know the name.
        """
        binding = self._binding
        return self.entity_class_for(name, binding=binding)

    @property
    def is_bound(self) -> bool:
        """Return whether a site schema is bound.

        Returns:
            `True` after the first successful bind.
        """
        return self._binding is not None

    def known_names(self) -> set[str]:
        """Return names known in one registry state snapshot.

        Returns:
            Detached canonical and alias names when bound, or cached symbolic
            names when unbound.
        """
        with self._class_lock:
            binding = self._binding
            class_names = set(self._classes)
        if binding is None:
            return class_names
        return set(binding.schemas) | set(binding.entity_aliases)

    def schema_snapshot(self) -> Mapping[str, EntitySchema] | None:
        """Return schemas from the current complete binding snapshot.

        Returns:
            Immutable schema mapping, or `None` while unbound.
        """
        binding = self._binding
        return None if binding is None else binding.schemas


def _default_registry_locked() -> Registry:
    """Return the lazy default Registry; the module lock must already be held."""
    if not _DEFAULT_REGISTRIES:
        _DEFAULT_REGISTRIES.append(Registry())
    return _DEFAULT_REGISTRIES[0]


@contextmanager
def _entity_classes_locked(
    registry: Registry,
) -> Generator[dict[str, type[Entity]], None, None]:
    """Yield one Registry's classes while its class lock is held."""
    with registry._class_lock:  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        yield registry._classes  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001


def can_replace_default_registry(registry: Registry) -> bool:
    """Return whether a reserved default Registry has no materialized classes.

    Args:
        registry: Reserved process default Registry.

    Returns:
        Whether pre-freeze pointer replacement remains permitted.
    """
    with _DEFAULT_REGISTRY_LOCK:
        if _default_registry_locked() is not registry:
            return False
        with _entity_classes_locked(registry) as classes:
            return not classes


def commit_default_registry_transition(
    registry: Registry,
    *,
    default_bound: bool,
    publish: Callable[[], None],
) -> None:
    """Commit one reserved Registry and scoped-session transition.

    Args:
        registry: Registry owned by the prepared default Session.
        default_bound: Whether the connection was paired to process defaults.
        publish: Infallible scoped-session publication callback.

    Raises:
        SgalchemyError: If no transition is reserved or a replacement froze.
    """
    with _DEFAULT_REGISTRY_LOCK:
        current = _default_registry_locked()
        if not _DEFAULT_TRANSITION['reserved']:
            raise SgalchemyError('no default registry transition is reserved')
        if default_bound and registry is not current:
            with _entity_classes_locked(current) as classes:
                if classes:
                    raise SgalchemyError(
                        'the default registry already owns entity classes and cannot be '
                        'replaced; use engine.entities, registry=Registry(), or a separate '
                        'process for another site'
                    )
                _DEFAULT_REGISTRIES[0] = registry
        _DEFAULT_SESSION_REGISTRIES[:] = [registry]
        publish()
        _DEFAULT_TRANSITION['reserved'] = False
        _DEFAULT_TRANSITION.pop('owner', None)


def default_registry() -> Registry:
    """Return the process-global default entity registry.

    Returns:
        Lazily constructed default registry. Existing engines and Sessions keep
        their originally paired registry.
    """
    with _DEFAULT_REGISTRY_LOCK:
        return _default_registry_locked()


def entity_registry(cls: type[object]) -> Registry | None:
    """Return the registry that owns one entity class.

    Args:
        cls: Reflected or symbolic entity class.

    Returns:
        Owning registry, or `None` for a compiler-only target.
    """
    return cast('Registry | None', _entity.entity_registry(cls))


def entity_state(obj: Entity) -> EntityState:
    """Return one entity's ORM state for sibling runtime modules.

    Args:
        obj: Instrumented entity instance.

    Returns:
        Mutable ORM state owned by the instance.
    """
    return obj._sg_state  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001


def assert_entity_access(obj: Entity) -> None:
    """Reject indeterminate or session-conflicting entity access.

    Args:
        obj: Entity whose direct state is about to be read or mutated.

    Raises:
        SgalchemyError: If the entity has an indeterminate server outcome.
    """
    state = entity_state(obj)
    if state.indeterminate:
        raise SgalchemyError(
            f'{state.entity_type} has an indeterminate server outcome; discard this object '
            'and reconcile through a fresh query in a new session'
        )
    if state.access_guard is not None:
        state.access_guard(state)


def entity_type(cls: type[object]) -> str:
    """Return one entity class's registered API type.

    Args:
        cls: Reflected, symbolic, or compiler-only entity class.

    Returns:
        Canonical or symbolic API entity type.
    """
    return _entity.entity_type(cls)


def initialize_entity(obj: Entity, state: EntityState) -> None:
    """Install initial ORM state during hydration or pending construction.

    Args:
        obj: Newly allocated entity instance.
        state: Initial session and field state.
    """
    object.__setattr__(obj, '_sg_state', state)


def inspect(obj: Entity) -> InspectionInfo:
    """Return an I/O-free snapshot of one entity's user-facing state.

    Args:
        obj: Entity whose state to inspect.

    Returns:
        Entity identity, loaded and dirty fields, lifecycle flags, and copies
        of field-level trust markers.

    Raises:
        SgalchemyError: If direct access to the entity is not currently safe.
    """
    assert_entity_access(obj)
    state = entity_state(obj)
    return InspectionInfo(
        entity_type=state.entity_type,
        id=state.id,
        loaded=dict(state.loaded),
        dirty=dict(state.dirty),
        expired=state.expired,
        deleted=state.deleted,
        refresh_required=frozenset(state.refresh_required),
        unverified_writes=frozenset(state.unverified_writes),
    )


def install_entity_finder(
    callback: Callable[
        [type[Entity], tuple[Filter | FilterGroup | RawFilters, ...], dict[str, object]],
        list[Entity],
    ],
) -> None:
    """Install the ORM-owned implementation used by `Entity.find()`.

    Args:
        callback: Finder that resolves the current default session.
    """
    _ENTITY_FINDERS[:] = [callback]


def release_default_registry_transition(owner: object) -> None:
    """Release a failed default Registry transition reservation.

    Args:
        owner: Private reservation identity that may release the transition.
    """
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_TRANSITION.get('owner') is owner:
            _DEFAULT_TRANSITION['reserved'] = False
            _DEFAULT_TRANSITION.pop('owner', None)


def reserve_default_registry_transition(owner: object) -> Registry:
    """Reserve one connection transition and return the current default.

    Args:
        owner: Private identity used to restrict rollback to this reservation.

    Returns:
        Current process default Registry.

    Raises:
        SgalchemyError: If another transition is already reserved.
    """
    with _DEFAULT_REGISTRY_LOCK:
        if _DEFAULT_TRANSITION['reserved']:
            raise SgalchemyError(
                'a connection is establishing the default registry; '
                'retry after connect() completes'
            )
        registry = _default_registry_locked()
        _DEFAULT_TRANSITION['owner'] = owner
        _DEFAULT_TRANSITION['reserved'] = True
        return registry


def resolve_default_entity_class(
    name: str,
    cache: Callable[[type[Entity]], None],
) -> type[Entity]:
    """Resolve and optionally cache one default-Registry entity class.

    Args:
        name: Canonical entity type or bound display alias.
        cache: Canonical-class cache publisher.

    Returns:
        Entity class from one complete default Registry generation.

    Raises:
        AttributeError: If the bound default Registry does not know `name`.
    """
    with _DEFAULT_REGISTRY_LOCK:
        registry = _default_registry_locked()
        was_bound = registry.is_bound
        cls = registry.entity_class(name)
        if was_bound and cls.__name__ == name:
            cache(cls)
        return cls


def set_default_registry(registry: Registry) -> None:
    """Replace the process-global default without retargeting existing owners.

    Args:
        registry: New process default.

    Raises:
        SgalchemyError: If a connection transition is reserved, an installed
            default Session owns another registry, or the current default owns
            entity classes.
    """
    # Lock order is always module default lock, then the current registry's class lock.
    with _DEFAULT_REGISTRY_LOCK:
        current = _default_registry_locked()
        if _DEFAULT_TRANSITION['reserved']:
            raise SgalchemyError(
                'a connection is establishing the default registry; '
                'retry after connect() completes'
            )
        if _DEFAULT_SESSION_REGISTRIES and registry is not _DEFAULT_SESSION_REGISTRIES[0]:
            raise SgalchemyError(
                'a default session is installed; use sgalchemy.connect(force=True) '
                'or engine.entities'
            )
        with _entity_classes_locked(current) as classes:
            if classes:
                raise SgalchemyError(
                    'the default registry already owns entity classes and cannot be replaced; '
                    'use sgalchemy.connect(force=True) or engine.entities for another site'
                )
            _DEFAULT_REGISTRIES[0] = registry
