import copy
import json
import pickle
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema, reset_default_registry_state

import sgalchemy
from sgalchemy import _errors, _shotgun
from sgalchemy._core import _expressions as ex
from sgalchemy._core import _registry as registry_module
from sgalchemy._core._registry import (
    Entity,
    EntityState,
    Registry,
    RegistryBinding,
    assert_entity_access,
    default_registry,
    entity_registry,
    entity_state,
    entity_type,
    initialize_entity,
    inspect,
    set_default_registry,
)
from sgalchemy._errors import SgalchemyError, StaleObjectError, UnfetchedFieldError
from sgalchemy._orm._session import Session
from sgalchemy.schema import _model
from sgalchemy.schema._model import EntitySchema


class _LazyContext:
    def resolve(self, field_code: str) -> None:
        pass


def _assert_field_provenance_containers(
    state: EntityState,
    expected: EntityState,
) -> None:
    assert state.loaded == expected.loaded
    assert state.dirty == expected.dirty
    assert state.clean == expected.clean
    assert state.lazy == expected.lazy
    assert state.dirty_types == expected.dirty_types
    assert state.refresh_required == expected.refresh_required
    assert state.unverified_writes == expected.unverified_writes
    assert state.clean_refresh_required == expected.clean_refresh_required
    assert state.clean_unverified_writes == expected.clean_unverified_writes


def _connect(fake_sg: _FakeShotgun, tmp_path: Path) -> Session:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    engine.load_schema()
    return Session(engine)


def _field_provenance_state(
    has_loaded_clean: bool,
    prior_refresh_required: bool,
    prior_unverified_write: bool,
) -> tuple[EntityState, str, str, Path, _LazyContext]:
    field_code = 'image'
    clean_value = 'https://example.test/old.jpg'
    dirty_value = Path('/renders/new.jpg')
    lazy_context = _LazyContext()
    loaded = {'code': 'stable'}
    if has_loaded_clean:
        loaded[field_code] = clean_value
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded=loaded,
        lazy={field_code: lazy_context},
        refresh_required={field_code} if prior_refresh_required else set(),
        unverified_writes={field_code} if prior_unverified_write else set(),
    )
    return state, field_code, clean_value, dirty_value, lazy_context


def _serializable_registry(fake_sg: _FakeShotgun) -> Registry:
    cache = fetch_schema(fake_sg)
    cache.fields['Shot']['description']['data_type']['value'] = 'serializable'
    registry = Registry()
    registry.bind(_model.build_schema(cache), key=fake_sg.base_url)
    return registry


def _stage_field_provenance(
    state: EntityState,
    field_code: str,
    dirty_value: Path,
) -> None:
    if field_code in state.loaded:
        state.clean[field_code] = state.loaded[field_code]
    if field_code in state.refresh_required:
        state.clean_refresh_required.add(field_code)
    if field_code in state.unverified_writes:
        state.clean_unverified_writes.add(field_code)
    state.loaded[field_code] = dirty_value
    state.dirty[field_code] = dirty_value
    state.dirty_types[field_code] = 'image'


@pytest.fixture
def bound(fake_sg: _FakeShotgun) -> Registry:
    registry = Registry()
    registry.bind(_model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url)
    return registry


def test_as_dict_allows_shared_acyclic_containers(fake_sg: _FakeShotgun) -> None:
    registry = _serializable_registry(fake_sg)
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    shared = {'value': 1}
    payload = {'left': shared, 'right': shared, 'nested': [shared]}
    initialize_entity(
        shot,
        EntityState(entity_type='Shot', id=12, loaded={'description': payload}),
    )

    assert shot.as_dict()['description'] == payload


def test_as_dict_converts_entities_nested_in_mappings_and_tuples(
    fake_sg: _FakeShotgun,
) -> None:
    registry = _serializable_registry(fake_sg)
    sequence_cls = registry.entity_class('Sequence')
    sequence = sequence_cls.__new__(sequence_cls)
    initialize_entity(sequence, EntityState(entity_type='Sequence', id=7, loaded={'code': 'SQ01'}))
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(
            entity_type='Shot',
            id=12,
            loaded={
                'description': {
                    'link': sequence,
                    'nested': ({'again': [sequence]}, 'keep'),
                },
            },
        ),
    )
    link = {'type': 'Sequence', 'id': 7, 'name': 'SQ01'}

    assert shot.as_dict() == {
        'type': 'Shot',
        'id': 12,
        'description': {
            'link': link,
            'nested': ({'again': [link]}, 'keep'),
        },
    }


def test_as_dict_json_serializes_nested_containers(fake_sg: _FakeShotgun) -> None:
    registry = _serializable_registry(fake_sg)
    sequence_cls = registry.entity_class('Sequence')
    sequence = sequence_cls.__new__(sequence_cls)
    initialize_entity(sequence, EntityState(entity_type='Sequence', id=7, loaded={'code': 'SQ01'}))
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(
            entity_type='Shot',
            id=12,
            loaded={'description': {'nested': [sequence]}},
        ),
    )
    encoded = json.loads(json.dumps(shot.as_dict()))

    assert encoded['description'] == {'nested': [{'type': 'Sequence', 'id': 7, 'name': 'SQ01'}]}


def test_as_dict_rejects_cyclic_containers_with_domain_error(fake_sg: _FakeShotgun) -> None:
    registry = _serializable_registry(fake_sg)
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    cyclic: dict[str, Any] = {}
    cyclic['self'] = cyclic
    initialize_entity(shot, EntityState(entity_type='Shot', id=12, loaded={'description': cyclic}))

    with pytest.raises(SgalchemyError) as exc_info:
        shot.as_dict()

    assert str(exc_info.value) == (
        'container value contains a reference cycle; cyclic containers '
        'cannot be serialized by as_dict()'
    )


def test_as_dict_uses_one_binding_snapshot(
    fake_sg: _FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    sequence_cls = registry.entity_class('Sequence')
    sequence = sequence_cls.__new__(sequence_cls)
    initialize_entity(sequence, EntityState(entity_type='Sequence', id=7, loaded={'code': 'SQ01'}))
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot, EntityState(entity_type='Shot', id=10, loaded={'sg_sequence': sequence})
    )
    fields = dict(schemas['Sequence'].fields)
    fields.pop('code')
    changed = {**schemas, 'Sequence': replace(schemas['Sequence'], fields=fields, name_field=None)}
    entered = Event()
    release = Event()
    original = registry_module.assert_entity_access

    def blocking_access(obj: Entity) -> None:
        original(obj)
        if obj is sequence:
            entered.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(registry_module, 'assert_entity_access', blocking_access)
    with ThreadPoolExecutor(max_workers=1) as pool:
        serializing = pool.submit(shot.as_dict)
        assert entered.wait(timeout=2)
        registry.bind(changed, key=fake_sg.base_url)
        release.set()
        assert serializing.result(timeout=2)['sg_sequence']['name'] == 'SQ01'


def test_base_entity_missing_class_field_uses_attribute_protocol() -> None:
    sentinel = object()

    assert hasattr(Entity, 'code') is False
    assert getattr(Entity, 'code', sentinel) is sentinel

    shot = Registry().entity_class('Shot')
    expression = shot.code
    assert expression.root_type == 'Shot'
    assert expression.path == ('code',)
    assert object.__getattribute__(expression, '_schema') is None
    assert object.__getattribute__(expression, '_schemas') is None


def test_binding_detaches_and_freezes_caller_schema(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    original_shot = schemas['Shot']
    backing = dict(schemas)
    registry.bind(MappingProxyType(backing), key=fake_sg.base_url)
    backing.clear()
    assert isinstance(original_shot.fields, dict)
    original_shot.fields.pop('description')
    shot = registry.entity_class('Shot')
    binding = object.__getattribute__(registry, '_binding')
    assert binding is not None
    expression = shot.description
    assert expression.root_type == 'Shot'
    assert expression.path == ('description',)
    assert (
        object.__getattribute__(expression, '_schema')
        is binding.schemas['Shot'].fields['description']
    )
    assert object.__getattribute__(expression, '_schemas') is binding.schemas
    with pytest.raises(TypeError):
        binding.schemas['Shot'].fields['description'] = binding.schemas['Shot'].fields['id']


def test_binding_skips_aliases_shadowed_by_entity_api(fake_sg: _FakeShotgun) -> None:
    cache = fetch_schema(fake_sg)
    for code in ('sg_mro', 'sg_new'):
        cache.fields['Shot'][code] = {
            'data_type': {'value': 'text'},
            'name': {'value': code},
            'properties': {},
        }
    registry = Registry()
    registry.bind(_model.build_schema(cache), key=fake_sg.base_url)
    binding = registry.binding_snapshot()
    assert binding is not None
    shot_cls = registry.entity_class('Shot')

    assert 'new' not in binding.field_aliases['Shot']
    assert 'mro' not in binding.field_aliases['Shot']
    assert callable(shot_cls.new)
    assert callable(shot_cls.mro)
    assert shot_cls.sg_new.filter_field == 'sg_new'
    assert shot_cls.sg_mro.filter_field == 'sg_mro'
    assert all(name in ' '.join(registry.alias_reports) for name in ('new', 'mro'))


def test_bound_class_expressions_aliases_and_identity(bound: Registry) -> None:
    shot = bound.entity_class('Shot')
    binding = object.__getattribute__(bound, '_binding')
    assert binding is not None
    assert (shot.status_list == 'ip') == ex.Filter('sg_status_list', 'is', 'ip')
    expression = shot.sg_sequence.code
    assert expression.root_type == 'Shot'
    assert expression.path == ('sg_sequence', 'Sequence', 'code')
    assert (
        object.__getattribute__(expression, '_schema')
        is binding.schemas['Sequence'].fields['code']
    )
    assert object.__getattribute__(expression, '_schemas') is binding.schemas
    with pytest.raises(AttributeError):
        _ = shot.not_a_field
    with pytest.raises(AttributeError, match='UI-only'):
        _ = shot.sg_qc_summary
    assert bound.entity_class('Vendor') is bound.entity_class('CustomEntity03')
    with pytest.raises(AttributeError, match='unknown entity'):
        bound.entity_class('NoSuchType')


def test_concurrent_canonical_and_alias_materialization_share_one_class(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    start = Barrier(2)
    entered_build = Event()
    release_build = Event()
    second_attempted = Event()
    real_lock = Lock()
    counter_lock = Lock()
    attempts = 0
    tracking_lock = MagicMock()
    original_build = object.__getattribute__(registry, '_build')

    def enter_lock() -> None:
        nonlocal attempts
        with counter_lock:
            attempts += 1
            if attempts == 2:
                second_attempted.set()
        real_lock.acquire()

    def exit_lock(*_args: object) -> None:
        real_lock.release()

    def blocking_build(canonical: str) -> type[Entity]:
        entered_build.set()
        assert release_build.wait(timeout=2)
        return original_build(canonical)

    tracking_lock.__enter__.side_effect = enter_lock
    tracking_lock.__exit__.side_effect = exit_lock
    monkeypatch.setattr(registry, '_class_lock', tracking_lock)
    monkeypatch.setattr(registry, '_build', blocking_build)

    def lookup(name: str) -> type[Entity]:
        start.wait(timeout=2)
        return registry.entity_class(name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        canonical = pool.submit(lookup, 'CustomEntity03')
        alias = pool.submit(lookup, 'Vendor')
        assert entered_build.wait(timeout=2)
        observed_second_attempt = second_attempted.wait(timeout=2)
        release_build.set()
        canonical_cls = canonical.result(timeout=2)
        alias_cls = alias.result(timeout=2)
    assert observed_second_attempt
    assert canonical_cls is alias_cls


def test_concurrent_first_bind_rejects_a_different_site(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    entered = Event()
    release = Event()
    original = registry_module.field_aliases

    def blocking_field_aliases(
        entity: EntitySchema,
        *,
        reserved_names: Collection[str] = (),
    ) -> tuple[dict[str, str], list[str]]:
        entered.set()
        assert release.wait(timeout=2)
        return original(entity, reserved_names=reserved_names)

    monkeypatch.setattr(registry_module, 'field_aliases', blocking_field_aliases)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(registry.bind, schemas, key=fake_sg.base_url)
        assert entered.wait(timeout=2)
        second = pool.submit(registry.bind, schemas, key='https://other.example.com')
        release.set()
        successes = 0
        errors: list[SgalchemyError] = []
        for future in (first, second):
            try:
                future.result(timeout=2)
                successes += 1
            except SgalchemyError as error:
                errors.append(error)
    assert successes == 1
    assert len(errors) == 1
    assert 'different site' in str(errors[0])


def test_container_children_classifies_only_supported_containers() -> None:
    mapping: _shotgun.WireTraversalContainer = MappingProxyType({'first': 1, 'second': 2})
    list_value = [1, 2]
    tuple_value = (1, 2)

    mapping_children = registry_module.container_children(mapping)

    assert mapping_children is not None
    assert list(mapping_children) == [1, 2]
    assert registry_module.container_children(list_value) is list_value
    assert registry_module.container_children(tuple_value) is tuple_value
    assert registry_module.container_children({1, 2}) is None
    assert registry_module.container_children(frozenset({1, 2})) is None
    assert registry_module.container_children('leaf') is None


def test_default_registry_concurrent_creation_has_one_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [])
    start = Barrier(8)

    def lookup() -> Registry:
        start.wait(timeout=2)
        return default_registry()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lookup) for _index in range(8)]
        registries = [future.result(timeout=2) for future in futures]
    assert all(registry is registries[0] for registry in registries)


def test_default_registry_replacement_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_default_registry_state(monkeypatch)
    first = default_registry()
    replacement = Registry()
    set_default_registry(replacement)
    assert default_registry() is replacement
    replacement.entity_class('Shot')
    with pytest.raises(SgalchemyError, match='cannot be replaced'):
        set_default_registry(first)


def test_default_registry_transition_rechecks_materialization_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    replacement = Registry()
    owner = object()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    monkeypatch.setattr(registry_module, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': False},
        raising=False,
    )
    registry_module.reserve_default_registry_transition(owner)
    current.entity_class('Shot')

    with pytest.raises(SgalchemyError, match='cannot be replaced'):
        registry_module.commit_default_registry_transition(
            replacement,
            default_bound=True,
            publish=lambda: pytest.fail('publication must not run'),
        )

    assert default_registry() is current
    assert vars(registry_module)['_DEFAULT_SESSION_REGISTRIES'] == []
    with pytest.raises(SgalchemyError, match='connection is establishing'):
        set_default_registry(replacement)
    registry_module.release_default_registry_transition(owner)


def test_default_registry_transition_release_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    replacement = Registry()
    owner = object()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    monkeypatch.setattr(registry_module, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': False},
        raising=False,
    )

    assert registry_module.reserve_default_registry_transition(owner) is current
    registry_module.release_default_registry_transition(owner)
    set_default_registry(replacement)

    assert default_registry() is replacement
    assert registry_module.reserve_default_registry_transition(owner) is replacement
    registry_module.release_default_registry_transition(owner)


def test_default_registry_transition_release_requires_matching_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    replacement = Registry()
    owner = object()
    competitor = object()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    monkeypatch.setattr(registry_module, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': False},
        raising=False,
    )

    assert registry_module.reserve_default_registry_transition(owner) is current
    with pytest.raises(SgalchemyError, match='connection is establishing'):
        registry_module.reserve_default_registry_transition(competitor)
    registry_module.release_default_registry_transition(competitor)

    with pytest.raises(SgalchemyError, match='connection is establishing'):
        set_default_registry(replacement)
    registry_module.release_default_registry_transition(owner)
    set_default_registry(replacement)
    assert default_registry() is replacement


def test_default_registry_transition_reserves_setter_and_records_session_pairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    owner = object()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    monkeypatch.setattr(registry_module, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': False},
        raising=False,
    )
    reserved = registry_module.reserve_default_registry_transition(owner)

    assert reserved is current
    with pytest.raises(SgalchemyError, match='connection is establishing'):
        set_default_registry(Registry())

    published: list[None] = []
    registry_module.commit_default_registry_transition(
        current,
        default_bound=True,
        publish=lambda: published.append(None),
    )

    assert published == [None]
    with pytest.raises(SgalchemyError, match=r'connect\(force=True\)'):
        set_default_registry(Registry())
    set_default_registry(current)


def test_default_replacement_blocks_lookup_until_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    replacement = Registry()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    entered_lock = Event()
    release_lock = Event()
    real_lock = Lock()
    tracking_lock = MagicMock()

    def enter_lock() -> None:
        real_lock.acquire()
        entered_lock.set()
        assert release_lock.wait(timeout=2)

    def exit_lock(*_args: object) -> None:
        real_lock.release()

    tracking_lock.__enter__.side_effect = enter_lock
    tracking_lock.__exit__.side_effect = exit_lock
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRY_LOCK', tracking_lock)
    lookup_started = Event()

    def lookup() -> Registry:
        lookup_started.set()
        return default_registry()

    with ThreadPoolExecutor(max_workers=2) as pool:
        replacing = pool.submit(set_default_registry, replacement)
        assert entered_lock.wait(timeout=2)
        looking_up = pool.submit(lookup)
        assert lookup_started.wait(timeout=2)
        release_lock.set()
        replacing.result(timeout=2)
        assert looking_up.result(timeout=2) is replacement


def test_default_replacement_waits_for_materialization_and_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    entered_build = Event()
    release_build = Event()
    original_build = object.__getattribute__(current, '_build')

    def blocking_build(canonical: str) -> type[Entity]:
        entered_build.set()
        assert release_build.wait(timeout=2)
        return original_build(canonical)

    monkeypatch.setattr(current, '_build', blocking_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        materializing = pool.submit(current.entity_class, 'Shot')
        assert entered_build.wait(timeout=2)
        replacing = pool.submit(set_default_registry, Registry())
        release_build.set()
        materializing.result(timeout=2)
        with pytest.raises(SgalchemyError, match='cannot be replaced'):
            replacing.result(timeout=2)
    assert default_registry() is current


def test_default_replacement_wins_before_old_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    replacement = Registry()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    setter_has_class_lock = Event()
    release_setter = Event()
    materializer_attempted = Event()
    real_lock = Lock()
    counter_lock = Lock()
    attempts = 0
    tracking_lock = MagicMock()

    def enter_lock() -> None:
        nonlocal attempts
        with counter_lock:
            attempts += 1
            attempt = attempts
            if attempt == 2:
                materializer_attempted.set()
        real_lock.acquire()
        if attempt == 1:
            setter_has_class_lock.set()
            assert release_setter.wait(timeout=2)

    def exit_lock(*_args: object) -> None:
        real_lock.release()

    tracking_lock.__enter__.side_effect = enter_lock
    tracking_lock.__exit__.side_effect = exit_lock
    monkeypatch.setattr(current, '_class_lock', tracking_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replacing = pool.submit(set_default_registry, replacement)
        assert setter_has_class_lock.wait(timeout=2)
        materializing = pool.submit(current.entity_class, 'Shot')
        assert materializer_attempted.wait(timeout=2)
        release_setter.set()
        replacing.result(timeout=2)
        materialized = materializing.result(timeout=2)
    assert entity_registry(materialized) is current
    assert default_registry() is replacement


def test_editable_nonconventional_image_assignment_remains_unsupported(
    fake_sg: _FakeShotgun,
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    fields = dict(schemas['Shot'].fields)
    fields['description'] = replace(fields['description'], data_type='image', editable=True)
    registry.bind(
        {**schemas, 'Shot': replace(schemas['Shot'], fields=fields)},
        key=fake_sg.base_url,
    )
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(shot, EntityState(entity_type='Shot', id=12))

    with pytest.raises(SgalchemyError, match=r'only.*image.*filmstrip_image'):
        shot.description = '/renders/thumbnail.jpg'
    assert entity_state(shot).dirty == {}


def test_empty_default_registry_can_replace_initializes_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Registry()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [])

    assert registry_module.can_replace_default_registry(candidate) is False
    assert len(vars(registry_module)['_DEFAULT_REGISTRIES']) == 1


def test_empty_default_registry_reservation_initializes_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [])
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': False},
        raising=False,
    )

    owner = object()
    reserved = registry_module.reserve_default_registry_transition(owner)

    assert vars(registry_module)['_DEFAULT_REGISTRIES'] == [reserved]
    registry_module.release_default_registry_transition(owner)


def test_empty_default_registry_transition_commit_replaces_lazy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = Registry()
    published: list[None] = []
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [])
    monkeypatch.setattr(registry_module, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': True},
        raising=False,
    )

    registry_module.commit_default_registry_transition(
        replacement,
        default_bound=True,
        publish=lambda: published.append(None),
    )

    assert default_registry() is replacement
    assert vars(registry_module)['_DEFAULT_SESSION_REGISTRIES'] == [replacement]
    assert published == [None]


def test_entity_state_accept_unstaged_value_drops_stale_first_write_markers() -> None:
    state = EntityState(
        entity_type='Shot',
        id=12,
        refresh_required={'filmstrip_image'},
        unverified_writes={'filmstrip_image'},
        clean_refresh_required={'filmstrip_image'},
        clean_unverified_writes={'filmstrip_image'},
    )

    state.accept_unstaged_value('filmstrip_image', None)

    assert state.loaded == {'filmstrip_image': None}
    assert state.refresh_required == set()
    assert state.unverified_writes == set()
    assert state.clean_refresh_required == set()
    assert state.clean_unverified_writes == set()


@pytest.mark.parametrize('has_loaded_clean', [False, True])
@pytest.mark.parametrize('prior_refresh_required', [False, True])
@pytest.mark.parametrize('prior_unverified_write', [False, True])
def test_entity_state_capture_assignment_snapshot_transition_matrix(
    has_loaded_clean: bool,
    prior_refresh_required: bool,
    prior_unverified_write: bool,
) -> None:
    state, field_code, clean_value, dirty_value, lazy_context = _field_provenance_state(
        has_loaded_clean,
        prior_refresh_required,
        prior_unverified_write,
    )

    state._capture_assignment_snapshot(field_code)
    state.refresh_required.discard(field_code)
    state.unverified_writes.discard(field_code)
    state.dirty[field_code] = dirty_value
    state.dirty_types[field_code] = 'image'
    state.loaded[field_code] = dirty_value
    state._capture_assignment_snapshot(field_code)

    expected = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'stable', field_code: dirty_value},
        dirty={field_code: dirty_value},
        clean={field_code: clean_value} if has_loaded_clean else {},
        lazy={field_code: lazy_context},
        dirty_types={field_code: 'image'},
        clean_refresh_required={field_code} if prior_refresh_required else set(),
        clean_unverified_writes={field_code} if prior_unverified_write else set(),
    )
    _assert_field_provenance_containers(state, expected)


def test_entity_state_defaults_are_independent() -> None:
    state = EntityState(entity_type='Shot', id=None)
    assert (
        state.loaded,
        state.dirty,
        state.clean,
        state.lazy,
        state.dirty_types,
        state.refresh_required,
        state.unverified_writes,
        state.clean_refresh_required,
        state.clean_unverified_writes,
    ) == (
        {},
        {},
        {},
        {},
        {},
        set(),
        set(),
        set(),
        set(),
    )
    other = EntityState(entity_type='Shot', id=None)
    state.refresh_required.add('code')
    state.unverified_writes.add('description')
    state.clean_refresh_required.add('sg_status_list')
    state.clean_unverified_writes.add('sg_cut_duration')
    assert other.refresh_required == set()
    assert other.unverified_writes == set()
    assert other.clean_refresh_required == set()
    assert other.clean_unverified_writes == set()


@pytest.mark.parametrize('has_loaded_clean', [False, True])
@pytest.mark.parametrize('prior_refresh_required', [False, True])
@pytest.mark.parametrize('prior_unverified_write', [False, True])
def test_entity_state_discard_field_provenance_transition_matrix(
    has_loaded_clean: bool,
    prior_refresh_required: bool,
    prior_unverified_write: bool,
) -> None:
    state, field_code, _clean_value, dirty_value, lazy_context = _field_provenance_state(
        has_loaded_clean,
        prior_refresh_required,
        prior_unverified_write,
    )
    _stage_field_provenance(
        state,
        field_code,
        dirty_value,
    )

    state.discard_field_provenance(field_code)

    expected = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'stable', field_code: dirty_value},
        lazy={field_code: lazy_context},
        refresh_required={field_code} if prior_refresh_required else set(),
        unverified_writes={field_code} if prior_unverified_write else set(),
    )
    _assert_field_provenance_containers(state, expected)


def test_entity_state_mark_field_unknown_preserves_write_provenance() -> None:
    context = _LazyContext()
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'old'},
        lazy={'code': context},
        dirty={'code': 'new'},
        unverified_writes={'code'},
    )

    state.mark_field_unknown('code')

    assert state.loaded == {}
    assert state.lazy == {'code': context}
    assert state.dirty == {'code': 'new'}
    assert state.refresh_required == {'code'}
    assert state.unverified_writes == {'code'}


def test_entity_state_mark_writes_unverified_skips_restaged_fields() -> None:
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'a', 'description': 'new'},
        dirty={'description': 'new'},
    )

    state.mark_writes_unverified(('code', 'description'))

    assert state.unverified_writes == {'code'}


@pytest.mark.parametrize('has_loaded_clean', [False, True])
@pytest.mark.parametrize('prior_refresh_required', [False, True])
@pytest.mark.parametrize('prior_unverified_write', [False, True])
def test_entity_state_restore_field_from_snapshot_transition_matrix(
    has_loaded_clean: bool,
    prior_refresh_required: bool,
    prior_unverified_write: bool,
) -> None:
    state, field_code, clean_value, dirty_value, lazy_context = _field_provenance_state(
        has_loaded_clean,
        prior_refresh_required,
        prior_unverified_write,
    )
    _stage_field_provenance(
        state,
        field_code,
        dirty_value,
    )

    state.restore_field_from_snapshot(field_code)

    expected_loaded = {'code': 'stable'}
    if has_loaded_clean:
        expected_loaded[field_code] = clean_value
    expected = EntityState(
        entity_type='Shot',
        id=12,
        loaded=expected_loaded,
        lazy={field_code: lazy_context},
        refresh_required={field_code} if prior_refresh_required else set(),
        unverified_writes={field_code} if prior_unverified_write else set(),
    )
    _assert_field_provenance_containers(state, expected)


def test_entity_state_restore_staged_fields_restores_writes_and_drops_strays() -> None:
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'new', 'description': 'kept'},
        dirty={'code': 'new'},
        clean={'code': 'old'},
        dirty_types={'code': 'text', 'sg_status_list': 'status_list'},
        clean_unverified_writes={'description'},
    )

    state.restore_staged_fields()

    assert state.loaded == {'code': 'old', 'description': 'kept'}
    assert state.dirty == {}
    assert state.clean == {}
    assert state.dirty_types == {}
    assert state.clean_unverified_writes == set()


def test_entity_state_retire_confirmed_write_clears_provenance_and_uncertainty() -> None:
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'new'},
        dirty={'code': 'new'},
        clean={'code': 'old'},
        dirty_types={'code': 'text'},
        unverified_writes={'code'},
        clean_unverified_writes={'code'},
    )

    state.retire_confirmed_write('code')

    assert state.loaded == {'code': 'new'}
    assert state.dirty == {}
    assert state.clean == {}
    assert state.dirty_types == {}
    assert state.unverified_writes == set()
    assert state.clean_unverified_writes == set()


@pytest.mark.parametrize('has_loaded_clean', [False, True])
@pytest.mark.parametrize('prior_refresh_required', [False, True])
@pytest.mark.parametrize('prior_unverified_write', [False, True])
def test_entity_state_retire_field_for_refresh_transition_matrix(
    has_loaded_clean: bool,
    prior_refresh_required: bool,
    prior_unverified_write: bool,
) -> None:
    state, field_code, _clean_value, dirty_value, lazy_context = _field_provenance_state(
        has_loaded_clean,
        prior_refresh_required,
        prior_unverified_write,
    )
    _stage_field_provenance(
        state,
        field_code,
        dirty_value,
    )

    state._retire_field_for_refresh(field_code)

    expected = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'stable'},
        lazy={field_code: lazy_context},
        refresh_required={field_code},
        unverified_writes={field_code} if prior_unverified_write else set(),
    )
    _assert_field_provenance_containers(state, expected)


@pytest.mark.parametrize('was_unverified', [False, True])
def test_entity_state_accept_field_value_clears_stale_read_state(*, was_unverified: bool) -> None:
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'code': 'old'},
        lazy={'code': _LazyContext()},
        refresh_required={'code'},
        unverified_writes={'code'} if was_unverified else set(),
    )

    state.accept_field_value('code', 'new')

    assert state.loaded == {'code': 'new'}
    assert state.lazy == {}
    assert state.refresh_required == set()
    assert state.unverified_writes == set()


def test_entity_state_retire_upload_field_clears_uncertain_write() -> None:
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'image': 'old'},
        dirty={'image': 'new'},
        clean={'image': 'old'},
        dirty_types={'image': 'image'},
        unverified_writes={'image'},
    )

    state.retire_upload_field('image')

    assert state.loaded == {}
    assert state.dirty == {}
    assert state.clean == {}
    assert state.dirty_types == {}
    assert state.refresh_required == {'image'}
    assert state.unverified_writes == set()


def test_expired_access_guard_and_indeterminate_contracts(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    guarded = shot_cls.__new__(shot_cls)
    calls: list[EntityState] = []
    initialize_entity(
        guarded,
        EntityState(
            entity_type='Shot',
            id=1,
            loaded={'code': 'AB'},
            access_guard=calls.append,
        ),
    )
    assert guarded.code == 'AB'
    guarded.description = 'new'
    assert guarded.as_dict()['description'] == 'new'
    assert calls == [entity_state(guarded)] * 3
    entity_state(guarded).expired = True
    with pytest.raises(StaleObjectError, match=r'session\.refresh\(obj\)'):
        _ = guarded.code
    entity_state(guarded).indeterminate = True
    assert repr(guarded) == 'Shot(indeterminate)'
    with pytest.raises(SgalchemyError, match='indeterminate server outcome'):
        assert_entity_access(guarded)


def test_inspect_enforces_access_guard_and_indeterminate_state() -> None:
    shot_cls = Registry().entity_class('Shot')
    guarded = shot_cls.__new__(shot_cls)
    calls: list[None] = []
    initialize_entity(
        guarded,
        EntityState(
            entity_type='Shot',
            id=1,
            loaded={'id': 1},
            access_guard=lambda _state: calls.append(None),
        ),
    )

    assert sgalchemy.inspect(guarded).id == 1
    assert calls == [None]

    entity_state(guarded).indeterminate = True
    with pytest.raises(SgalchemyError, match='indeterminate server outcome'):
        sgalchemy.inspect(guarded)
    assert calls == [None]


def test_inspect_returns_guarded_detached_state_snapshot(
    fake_sg: _FakeShotgun, tmp_path: Path
) -> None:
    session = _connect(fake_sg, tmp_path)
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A'}]]
    shot_cls = session.engine.registry.entity_class('Shot')
    shot = session.scalars(sgalchemy.select(shot_cls))[0]
    shot.code = 'B'
    state = entity_state(shot)
    state.deleted = True
    state.refresh_required.add('description')
    state.unverified_writes.add('sg_status_list')
    session.expire(shot)

    info = sgalchemy.inspect(shot)

    assert (info.entity_type, info.id) == ('Shot', 1)
    assert info.loaded == {'id': 1, 'code': 'B'}
    assert info.dirty == {'code': 'B'}
    assert info.expired
    assert info.deleted
    assert info.refresh_required == frozenset({'description'})
    assert info.unverified_writes == frozenset({'sg_status_list'})
    info.loaded.clear()
    info.dirty.clear()
    assert state.loaded == {'id': 1, 'code': 'B'}
    assert state.dirty == {'code': 'B'}
    assert state.refresh_required == {'description'}
    assert state.unverified_writes == {'sg_status_list'}


def test_instance_state_repr_and_assignment_errors(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    obj = shot_cls.__new__(shot_cls)
    initialize_entity(obj, EntityState(entity_type='Shot', id=12, loaded={'id': 12, 'code': 'AB'}))
    assert obj.code == 'AB'
    assert repr(obj) == "Shot(id=12, code='AB')"
    with pytest.raises(UnfetchedFieldError, match=r'load_only.*joinedload.*selectinload'):
        _ = obj.description
    obj.description = 'new'
    assert entity_state(obj).dirty == {'description': 'new'}
    assert entity_state(obj).dirty_types == {'description': 'text'}
    with pytest.raises(AttributeError, match='nothing was staged'):
        obj.descriptoin = 'oops'
    with pytest.raises(AttributeError, match='assigned by the server'):
        obj.id = 99
    with pytest.raises(AttributeError, match='UI-only'):
        obj.sg_qc_summary = 'oops'


def test_instance_write_captures_binding_after_access_guard(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    fields = dict(schemas['Shot'].fields)
    fields.pop('description')
    changed = {**schemas, 'Shot': replace(schemas['Shot'], fields=fields)}
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)

    def refresh(_state: EntityState) -> None:
        registry.bind(changed, key=fake_sg.base_url)

    initialize_entity(
        shot,
        EntityState(entity_type='Shot', id=10, access_guard=refresh),
    )
    with pytest.raises(AttributeError, match='nothing was staged'):
        shot.description = 'new'
    assert entity_state(shot).dirty == {}
    assert entity_state(shot).loaded == {}


def test_instance_write_updates_dirty_type_from_current_binding(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(shot, EntityState(entity_type='Shot', id=10))
    shot.description = 'ordinary text'
    assert entity_state(shot).dirty_types == {'description': 'text'}

    fields = dict(schemas['Shot'].fields)
    fields['description'] = replace(fields['description'], data_type='image')
    registry.bind(
        {**schemas, 'Shot': replace(schemas['Shot'], fields=fields)},
        key=fake_sg.base_url,
    )
    with pytest.raises(SgalchemyError, match=r'only.*image.*filmstrip_image'):
        shot.description = '/renders/thumbnail.jpg'
    assert entity_state(shot).dirty_types == {'description': 'text'}


def test_instance_write_uses_one_binding_snapshot(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(shot, EntityState(entity_type='Shot', id=10))
    fields = dict(schemas['Shot'].fields)
    fields.pop('sg_status_list')
    changed = {**schemas, 'Shot': replace(schemas['Shot'], fields=fields)}
    entered = Event()
    release = Event()
    original = type.__getattribute__(registry_module.Entity, '_sg_resolve')

    def blocking_resolve(
        obj: Entity,
        name: str,
        binding: RegistryBinding | None,
    ) -> str:
        entered.set()
        assert release.wait(timeout=2)
        return original(obj, name, binding)

    monkeypatch.setattr(registry_module.Entity, '_sg_resolve', blocking_resolve)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(setattr, shot, 'status_list', 'ip')
        assert entered.wait(timeout=2)
        registry.bind(changed, key=fake_sg.base_url)
        release.set()
        writing.result(timeout=2)
    assert entity_state(shot).dirty == {'sg_status_list': 'ip'}
    assert entity_state(shot).dirty_types == {'sg_status_list': 'status_list'}


def test_internal_contracts_and_recursive_link_serialization(bound: Registry) -> None:
    sequence_cls = bound.entity_class('Sequence')
    sequence = sequence_cls.__new__(sequence_cls)
    initialize_entity(sequence, EntityState(entity_type='Sequence', id=7, loaded={'code': 'SQ01'}))
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(
            entity_type='Shot',
            id=12,
            loaded={'sg_sequence': sequence, 'assets': [[sequence, 'keep'], 3]},
        ),
    )
    assert entity_registry(shot_cls) is bound
    assert shot.as_dict() == {
        'type': 'Shot',
        'id': 12,
        'sg_sequence': {'type': 'Sequence', 'id': 7, 'name': 'SQ01'},
        'assets': [[{'type': 'Sequence', 'id': 7, 'name': 'SQ01'}, 'keep'], 3],
    }


def test_known_names_returns_detached_snapshots(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    registry.entity_class('Shot')
    unbound_names = registry.known_names()
    assert unbound_names == {'Shot'}
    unbound_names.add('Injected')
    assert registry.known_names() == {'Shot'}

    registry.bind(_model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url)
    bound_names = registry.known_names()
    assert {'Shot', 'CustomEntity03', 'Vendor'} <= bound_names
    bound_names.clear()
    assert 'Shot' in registry.known_names()


def test_new_rejects_server_assigned_id(bound: Registry) -> None:
    with pytest.raises(AttributeError, match='assigned by the server'):
        bound.entity_class('Shot').new(id=41, code='invalid')


def test_new_requires_bound_schema() -> None:
    symbolic = Registry().entity_class('Shot')
    with pytest.raises(SgalchemyError, match='load_schema'):
        symbolic.new(code='not-validated')


@pytest.mark.parametrize('field_code', ['image', 'filmstrip_image'])
def test_noneditable_conventional_thumbnail_paths_remain_supported(
    bound: Registry,
    tmp_path: Path,
    field_code: str,
) -> None:
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(
            entity_type='Shot',
            id=12,
            loaded={'id': 12, 'image': 'old', 'filmstrip_image': 'old-strip'},
        ),
    )
    path = tmp_path / 'thumb.jpg'
    path.write_bytes(b'image')

    setattr(shot, field_code, path)

    state = entity_state(shot)
    assert state.dirty == {field_code: path}


def test_noneditable_filmstrip_clear_keeps_teaching_error(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(shot, EntityState(entity_type='Shot', id=12))

    with pytest.raises(SgalchemyError, match=r'image=None'):
        shot.filmstrip_image = None
    assert entity_state(shot).dirty == {}


def test_noneditable_id_assignment_keeps_specific_error(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(shot, EntityState(entity_type='Shot', id=12))

    with pytest.raises(AttributeError, match='assigned by the server'):
        shot.id = 13
    assert entity_state(shot).dirty == {}


def test_noneditable_image_clear_remains_supported(bound: Registry, tmp_path: Path) -> None:
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(
            entity_type='Shot',
            id=12,
            loaded={'id': 12, 'image': 'old', 'filmstrip_image': 'old-strip'},
        ),
    )
    path = tmp_path / 'strip.jpg'
    path.write_bytes(b'image')

    shot.filmstrip_image = path
    shot.image = None

    state = entity_state(shot)
    assert state.loaded['image'] is None
    assert state.loaded['filmstrip_image'] == 'old-strip'
    assert state.dirty == {'image': None}


def test_noneditable_ordinary_assignment_preserves_state(fake_sg: _FakeShotgun) -> None:
    cache = fetch_schema(fake_sg)
    cache.fields['Shot']['description']['editable'] = {'value': False}
    registry = Registry()
    registry.bind(_model.build_schema(cache), key=fake_sg.base_url)
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    state = EntityState(
        entity_type='Shot',
        id=12,
        loaded={'description': 'old'},
        refresh_required={'description'},
        unverified_writes={'description'},
    )
    initialize_entity(shot, state)
    before = copy.deepcopy(state)

    with pytest.raises(SgalchemyError) as excinfo:
        shot.description = 'new'

    message = str(excinfo.value)
    assert 'Shot.description' in message
    assert 'engine.load_schema(force=True)' in message
    assert 'stale' in message
    assert 'provider permissions remain authoritative' in message
    assert not isinstance(excinfo.value, AttributeError)
    assert state == before


def test_opaque_entity_access_mutation_and_protocol_probes(bound: Registry) -> None:
    opaque_cls = bound.entity_class_for('FutureVendor', binding=None)
    opaque = opaque_cls.__new__(opaque_cls)
    initialize_entity(
        opaque,
        EntityState(
            entity_type='FutureVendor',
            id=9,
            loaded={'id': 9, 'name': 'Future Co'},
        ),
    )

    assert repr(opaque) == 'FutureVendor(id=9)'
    assert opaque.name == 'Future Co'
    assert opaque.as_dict() == {'type': 'FutureVendor', 'id': 9, 'name': 'Future Co'}
    assert inspect(opaque).entity_type == 'FutureVendor'
    with pytest.raises(_errors.SchemaUnavailableError, match='FutureVendor'):
        _ = opaque.code
    with pytest.raises(_errors.SchemaUnavailableError):
        getattr(opaque, 'code', None)
    with pytest.raises(_errors.SchemaUnavailableError):
        hasattr(opaque, 'code')
    with pytest.raises(AttributeError):
        _ = opaque._private_probe
    with pytest.raises(AttributeError):
        _ = opaque.__setstate__
    with pytest.raises(_errors.SchemaUnavailableError, match='FutureVendor'):
        opaque.code = 'blocked'
    with pytest.raises(_errors.SchemaUnavailableError, match='FutureVendor'):
        opaque_cls.new(code='blocked')
    assert entity_state(opaque).dirty == {}
    assert copy.copy(opaque) is not opaque
    assert copy.deepcopy(opaque) is not opaque
    try:
        pickle.dumps(opaque)
    except _errors.SchemaUnavailableError:
        pytest.fail('schema-unavailable field access escaped pickle protocol probing')
    except pickle.PicklingError:
        pass


def test_private_session_repair_is_gated_by_outgoing_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = Registry()
    private = Registry()
    owner = object()
    monkeypatch.setattr(registry_module, '_DEFAULT_REGISTRIES', [current])
    monkeypatch.setattr(registry_module, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(
        registry_module,
        '_DEFAULT_TRANSITION',
        {'reserved': False},
        raising=False,
    )
    registry_module.reserve_default_registry_transition(owner)
    registry_module.commit_default_registry_transition(
        private,
        default_bound=False,
        publish=lambda: None,
    )
    current.entity_class('Shot')

    with pytest.raises(SgalchemyError, match='cannot be replaced'):
        set_default_registry(private)

    assert default_registry() is current


def test_rebind_publishes_one_atomic_snapshot(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    old_binding = object.__getattribute__(registry, '_binding')
    entered = Event()
    release = Event()
    original = registry_module.field_aliases

    def blocking_field_aliases(
        entity: EntitySchema,
        *,
        reserved_names: Collection[str] = (),
    ) -> tuple[dict[str, str], list[str]]:
        entered.set()
        assert release.wait(timeout=2)
        return original(entity, reserved_names=reserved_names)

    monkeypatch.setattr(registry_module, 'field_aliases', blocking_field_aliases)
    with ThreadPoolExecutor(max_workers=1) as pool:
        rebinding = pool.submit(registry.bind, dict(schemas), key=fake_sg.base_url)
        assert entered.wait(timeout=2)
        assert object.__getattribute__(registry, '_binding') is old_binding
        expression = registry.entity_class('Shot').status_list
        assert expression.root_type == 'Shot'
        assert expression.path == ('sg_status_list',)
        assert (
            object.__getattribute__(expression, '_schema')
            is old_binding.schemas['Shot'].fields['sg_status_list']
        )
        assert object.__getattribute__(expression, '_schemas') is old_binding.schemas
        release.set()
        rebinding.result(timeout=2)
    assert object.__getattribute__(registry, '_binding') is not old_binding


def test_rebind_same_site_refreshes_and_foreign_site_raises(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key=fake_sg.base_url)
    registry.bind(dict(schemas), key=fake_sg.base_url)
    binding = registry.binding_snapshot()
    with pytest.raises(SgalchemyError) as raised:
        registry.bind(schemas, key='https://other.example.com')
    assert str(raised.value) == (
        'registry is already bound to a different site (one site per Registry); '
        'pass registry=Registry() or bind_default=False to the engine factory for a second site'
    )
    assert registry.binding_snapshot() is binding


def test_refresh_required_does_not_escape_copy_protocol_probes(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(entity_type='Shot', id=12, loaded={'id': 12}),
    )
    entity_state(shot).refresh_required.add('code')

    assert copy.copy(shot) is not shot
    assert copy.deepcopy(shot) is not shot
    try:
        pickle.dumps(shot)
    except _errors.RefreshRequiredError:
        pytest.fail('refresh-required field access escaped pickle protocol probing')
    except pickle.PicklingError:
        pass


def test_refresh_required_is_not_attribute_absence(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(
        shot,
        EntityState(entity_type='Shot', id=12, loaded={'id': 12, 'code': 'old'}),
    )
    state = entity_state(shot)
    state.loaded.pop('code')
    state.refresh_required.add('code')

    with pytest.raises(_errors.RefreshRequiredError, match=r'Shot\(id=12\).*code.*refresh'):
        _ = shot.code
    with pytest.raises(_errors.RefreshRequiredError):
        getattr(shot, 'code', None)
    with pytest.raises(_errors.RefreshRequiredError):
        hasattr(shot, 'code')
    assert repr(shot) == 'Shot(id=12)'
    assert shot.as_dict() == {'type': 'Shot', 'id': 12}
    info = inspect(shot)
    assert info.refresh_required == frozenset({'code'})
    assert info.unverified_writes == frozenset()


def test_registry_binding_snapshot_distinguishes_current_from_captured_absence(
    fake_sg: _FakeShotgun,
) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))

    assert registry.binding_snapshot() is None
    assert registry.schema_snapshot() is None
    assert registry.bound_key is None
    symbolic = registry.entity_class_for('Vendor', binding=None)
    assert symbolic is registry.entity_class_for('Vendor', binding=None)
    assert entity_type(symbolic) == 'Vendor'

    registry.bind(schemas, key=fake_sg.base_url)
    binding = registry.binding_snapshot()
    assert binding is not None
    assert registry.bound_key == fake_sg.base_url
    assert registry.schema_snapshot() is binding.schemas
    assert registry.entity_class('Vendor') is registry.entity_class('CustomEntity03')
    assert registry.entity_class_for('Vendor', binding=None) is symbolic
    assert registry.entity_class_for('Vendor', binding=binding) is registry.entity_class(
        'CustomEntity03'
    )


def test_repr_omits_name_when_value_repr_raises(bound: Registry) -> None:
    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError('broken repr')

    shot_cls = bound.entity_class('Shot')
    saved = shot_cls.__new__(shot_cls)
    initialize_entity(saved, EntityState(entity_type='Shot', id=3, loaded={'code': BrokenRepr()}))
    assert repr(saved) == 'Shot(id=3)'
    unsaved = shot_cls.__new__(shot_cls)
    initialize_entity(
        unsaved,
        EntityState(entity_type='Shot', id=None, loaded={'code': BrokenRepr()}),
    )
    assert repr(unsaved) == 'Shot(unsaved)'


def test_repr_unsaved_unloaded_and_uninitialized(bound: Registry) -> None:
    shot_cls = bound.entity_class('Shot')
    unsaved = shot_cls.__new__(shot_cls)
    initialize_entity(unsaved, EntityState(entity_type='Shot', id=None, loaded={'code': 'X'}))
    assert repr(unsaved) == "Shot(unsaved, code='X')"
    unloaded = shot_cls.__new__(shot_cls)
    initialize_entity(unloaded, EntityState(entity_type='Shot', id=3))
    assert repr(unloaded) == 'Shot(id=3)'
    assert repr(shot_cls.__new__(shot_cls)) == 'Shot(uninitialized)'


def test_stale_prebind_alias_class_stays_symbolic(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    vendor = registry.entity_class('Vendor')
    registry.bind(_model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url)
    expr = vendor.code
    assert (
        object.__getattribute__(expr, '_entity_type'),
        object.__getattribute__(expr, '_schema'),
    ) == ('Vendor', None)


def test_unbound_classes_are_symbolic_and_stable() -> None:
    registry = Registry()
    shot = registry.entity_class('Shot')
    assert shot is registry.entity_class('Shot')
    expression = shot.code
    assert expression.root_type == 'Shot'
    assert expression.path == ('code',)
    assert object.__getattribute__(expression, '_schema') is None
    assert object.__getattribute__(expression, '_schemas') is None
    vendor = registry.entity_class('Vendor')
    assert entity_type(vendor) == 'Vendor'
    assert vendor is registry.entity_class('Vendor')


def test_unbound_instance_write_is_rejected_before_staging() -> None:
    shot_cls = Registry().entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    initialize_entity(shot, EntityState(entity_type='Shot', id=None))
    with pytest.raises(SgalchemyError, match='load_schema'):
        shot.code = 'not-validated'
    assert entity_state(shot).dirty == {}
