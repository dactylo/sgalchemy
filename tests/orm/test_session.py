from __future__ import annotations

import copy
import datetime
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Never, ParamSpec, TypeVar, cast

import pytest
from _support import (
    AUTHENTICATION_FAULT,
    SCHEMA_ABSENT_MESSAGE,
    BrokenNoteError,
    YieldCountingEntities,
    _preserve_signature,
    capture_hydration_bindings,
)

from sgalchemy import inspect
from sgalchemy._core._compiler import FindCall
from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._expressions import func
from sgalchemy._core._options import joinedload, lazyload, selectinload
from sgalchemy._core._registry import (
    Entity,
    EntityState,
    Registry,
    RegistryBinding,
    entity_state,
    initialize_entity,
)
from sgalchemy._core._result import Result
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import select
from sgalchemy._errors import (
    CommitError,
    CompilationError,
    SchemaUnavailableError,
    SgalchemyError,
    StaleObjectError,
)
from sgalchemy._orm import Session
from sgalchemy._orm import _session as session_module

if TYPE_CHECKING:
    from pathlib import Path

    from _fakes import _FakeShotgun
    from _support import _SyncCallback

    from sgalchemy._orm._uow import CommitPlan

_P = ParamSpec('_P')
_R = TypeVar('_R')


class _NoScanIdentity(dict[tuple[str, int], Entity]):
    def items(self) -> Never:
        raise AssertionError('normal hydration scanned identity items')

    def values(self) -> Never:
        raise AssertionError('normal hydration scanned identity values')


class _OpaqueDict(dict[str, Any]):
    def __deepcopy__(self, _memo: dict[int, Any]) -> Never:
        raise AssertionError('opaque dictionary subclass was copied')


class _OpaqueMapping(Mapping[str, Any]):
    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __deepcopy__(self, _memo: dict[int, Any]) -> Never:
        raise AssertionError('opaque mapping was copied')

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _engine_with_symbolic_vendor(
    fake_sg: _FakeShotgun, tmp_path: Path
) -> tuple[Engine, type[Entity]]:
    registry = Registry()
    vendor_cls = registry.entity_class('FutureVendor')
    engine = create_engine(
        connection=fake_sg, cache_root=tmp_path, registry=registry, bind_default=False
    )
    engine.load_schema()
    return engine, vendor_cls


def test_auth_upload_failure_notes_unknown_outcome_on_commit_error(
    engine: Engine, fake_sg: _FakeShotgun, tmp_path: Path
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 9, 'image': 'old-image'})
    image_path = tmp_path / 'auth-fail.jpg'
    image_path.write_bytes(b'image')
    shot.image = image_path
    fault = AUTHENTICATION_FAULT('session expired')
    fake_sg._errors.setdefault('upload_thumbnail', []).append(fault)

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
    assert excinfo.value.__cause__ is fault
    assert 'upload outcome' in joined
    assert 'session.refresh' in joined
    assert 'remain staged' not in joined


def test_bare_link_hydration_does_not_clear_link_expiration(engine: Engine) -> None:
    engine.load_schema()
    session = Session(engine)
    sequence = session.merge_dict({'type': 'Sequence', 'id': 7, 'code': 'old-code'})
    session.expire(sequence)

    session.merge_dict({'type': 'Shot', 'id': 5, 'sg_sequence': {'type': 'Sequence', 'id': 7}})

    assert entity_state(sequence).expired
    with pytest.raises(StaleObjectError):
        _ = sequence.code


def test_bound_hydration_never_treats_display_alias_as_opaque_canonical_wire(
    engine: Engine,
) -> None:
    engine.load_schema()
    session = Session(engine)

    with pytest.raises(SgalchemyError, match=r'Vendor.*alias.*CustomEntity03'):
        session.merge_dict({'type': 'Vendor', 'id': 9})

    assert session._identity_values() == []


def test_broken_add_note_does_not_mask_provider_error(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    session = Session(engine)
    error = BrokenNoteError('provider unavailable')
    fake_sg._errors['find'] = [error]
    with pytest.raises(BrokenNoteError) as excinfo:
        session.execute_call(FindCall('Shot', [], ['id']), None)
    assert excinfo.value is error


@pytest.mark.parametrize('operation', ['execute', 'get', 'refresh', 'scalars'])
def test_cold_schema_fault_is_annotated_by_every_provider_operation(
    engine: Engine, fake_sg: _FakeShotgun, operation: str
) -> None:
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.merge_dict({'type': 'Shot', 'id': 1})
    error = RuntimeError('schema unavailable')
    fake_sg._errors['schema_read'] = [error]
    calls = {
        'execute': lambda: session.execute(select(shot_cls.code)),
        'get': lambda: session.get(shot_cls, 2),
        'refresh': lambda: session.refresh(shot),
        'scalars': lambda: session.scalars(select(shot_cls)),
    }
    with pytest.raises(RuntimeError) as excinfo:
        calls[operation]()
    assert excinfo.value is error
    assert len(error.__notes__) == 1
    assert error.__notes__[0].startswith('sgalchemy:')


def test_cold_stale_disk_schema_forces_one_refresh_before_commit(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    stale_sg = type(fake_sg)()
    stale_schema = {name: fields for name, fields in schema_fixture.items() if name != 'Shot'}
    monkeypatch.setattr(stale_sg, 'schema_read', lambda: copy.deepcopy(stale_schema))
    warmer = create_engine(
        connection=stale_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    warmer.load_schema()
    engine = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 33})
    session.delete(shot)
    original_load = engine.load_binding
    force_values: list[bool] = []

    @_preserve_signature(original_load)
    def tracking_load(*, force: bool = False) -> RegistryBinding:
        force_values.append(force)
        return original_load(force=force)

    monkeypatch.setattr(engine, 'load_binding', tracking_load)

    session.commit()

    assert force_values == [False, True]
    names = [name for name, _, _ in fake_sg._calls]
    assert names.count('schema_read') == 1
    assert names.count('schema_entity_read') == 1
    assert names.count('batch') == 1
    assert entity_state(shot).deleted


def test_commit_schema_recovery_forces_once_when_schema_remains_invalid(
    caplog: pytest.LogCaptureFixture,
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    engine.load_schema()
    complete = engine.prepared_binding()
    assert complete is not None
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 32})
    session.delete(shot)
    engine.registry.bind(
        {name: schema for name, schema in complete.schemas.items() if name != 'Shot'},
        key=complete.key,
    )
    stale_raw = {name: schema for name, schema in schema_fixture.items() if name != 'Shot'}

    def stale_schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return copy.deepcopy(stale_raw)

    monkeypatch.setattr(fake_sg, 'schema_read', stale_schema_read)

    with (
        caplog.at_level('INFO', logger='sgalchemy.engine'),
        pytest.raises(SgalchemyError, match=r'entity type.*refreshing schema once'),
    ):
        session.commit()

    names = [name for name, _, _ in fake_sg._calls]
    assert names.count('schema_read') == 2
    assert names.count('batch') == 0
    assert caplog.messages.count('commit schema drift: refreshing schema once') == 1


def test_commit_schema_recovery_reuses_concurrently_published_binding_without_io(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine.load_schema()
    complete = engine.prepared_binding()
    assert complete is not None
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 31})
    session.delete(shot)
    engine.registry.bind(
        {name: schema for name, schema in complete.schemas.items() if name != 'Shot'},
        key=complete.key,
    )
    original_plan_commit: _SyncCallback[
        [Session, RegistryBinding | None, Mapping[tuple[str, int], Entity]],
        CommitPlan,
    ] = session_module.plan_commit
    publications = 0

    def publish_on_drift(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return call(*args, **kwargs)
            except SchemaDrift:
                nonlocal publications
                publications += 1
                engine.registry.bind(dict(complete.schemas), key=complete.key)
                raise

        return wrapper

    publish_during_failed_plan = publish_on_drift(original_plan_commit)

    monkeypatch.setattr(session_module, 'plan_commit', publish_during_failed_plan)

    session.commit()

    names = [name for name, _, _ in fake_sg._calls]
    assert entity_state(shot).deleted
    assert publications == 1
    assert names.count('schema_read') == 1
    assert names.count('batch') == 1


def test_context_close_detaches_all_entities(engine: Engine) -> None:
    with Session(engine) as session:
        shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'A'})
        sequence = session.merge_dict({'type': 'Sequence', 'id': 2, 'code': 'S'})
        assert session.engine is engine
    assert entity_state(shot).session is None
    assert entity_state(sequence).session is None


@pytest.mark.parametrize('path', ['sg_sequence.Asset.code', 'sg_sequence.Sequence.code'])
def test_deep_field_mismatch_is_preserved_and_marks_schema_stale(
    engine: Engine,
    fake_sg: _FakeShotgun,
    caplog: pytest.LogCaptureFixture,
    path: str,
) -> None:
    engine.load_schema()
    session = Session(engine)
    raw: dict[str, Any] = {'type': 'Shot', 'id': 1, path: 'wire-value'}
    if path.startswith('sg_sequence.Asset'):
        raw['sg_sequence'] = {'type': 'Sequence', 'id': 7}
    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        shot = session.merge_dict(raw)
    assert entity_state(shot).loaded[path] == 'wire-value'
    assert 'schema shape mismatch' in caplog.text
    engine.load_schema()
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2


def test_deep_field_recovery_removes_preserved_dotted_value(engine: Engine) -> None:
    engine.load_schema()
    session = Session(engine)
    path = 'sg_sequence.Sequence.code'
    shot = session.merge_dict({'type': 'Shot', 'id': 1, path: 'stale'})
    assert entity_state(shot).loaded[path] == 'stale'
    session.merge_dict(
        {
            'type': 'Shot',
            'id': 1,
            'sg_sequence': {'type': 'Sequence', 'id': 7},
            path: 'current',
        }
    )
    assert shot.sg_sequence.code == 'current'
    assert path not in entity_state(shot).loaded
    assert path not in shot.as_dict()


def test_dirty_relationship_shields_link_and_descendants(engine: Engine) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 1,
            'sg_sequence': {'type': 'Sequence', 'id': 7, 'code': 'old'},
            'sg_sequence.Sequence.code': 'old',
        }
    )
    replacement = session.merge_dict({'type': 'Sequence', 'id': 8, 'code': 'local'})
    shot.sg_sequence = replacement
    session.merge_dict(
        {
            'type': 'Shot',
            'id': 1,
            'sg_sequence': {'type': 'Sequence', 'id': 7},
            'sg_sequence.Sequence.code': 'server',
        }
    )
    assert shot.sg_sequence is replacement
    assert replacement.code == 'local'


def test_execute_call_accepts_captured_binding_witness(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    call = FindCall('Shot', [], ['id'])
    fake_sg._find_results = [[{'id': 1}]]
    binding = engine.load_binding()

    assert session.execute_call(call, binding) == [{'id': 1}]


def test_execute_call_rejects_wrong_provider_shapes(
    engine: Engine, fake_sg: _FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.load_schema()
    session = Session(engine)
    call = FindCall('Shot', [], ['id'])
    binding = engine.load_binding()

    def bad_find(
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, bool]:
        del entity_type, filters, fields, kwargs
        return {'bad': True}

    monkeypatch.setattr(fake_sg, 'find', bad_find)
    with pytest.raises(SgalchemyError, match='find payload'):
        session.execute_call(call, binding)


def test_execute_projection_summary_and_entity_gate(engine: Engine, fake_sg: _FakeShotgun) -> None:
    engine.load_schema()
    shot_cls = engine.registry.entity_class('Shot')
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A'}]]
    session = Session(engine)
    projected = session.execute(select(shot_cls.code))
    assert isinstance(projected, Result)
    assert projected[0].code == 'A'
    calls_before = len(fake_sg._calls)
    with pytest.raises(TypeError, match='scalars'):
        session.execute(cast('Any', select(shot_cls)))
    assert len(fake_sg._calls) == calls_before


def test_expire_rejects_foreign_registry_state_without_mutation(engine: Engine) -> None:
    session = Session(engine)
    foreign_cls = Registry().entity_class('Shot')
    foreign = foreign_cls.__new__(foreign_cls)
    initialize_entity(foreign, EntityState('Shot', 1, session=session))
    with pytest.raises(SgalchemyError, match='registry'):
        session.expire(foreign)
    assert not entity_state(foreign).expired


def test_get_miss_none_hit_and_fault_context(engine: Engine, fake_sg: _FakeShotgun) -> None:
    session = Session(engine)
    engine.load_schema()
    shot_cls = engine.registry.entity_class('Shot')
    fake_sg._find_results = [[], [{'type': 'Shot', 'id': 5, 'code': 'X'}]]
    assert session.get(shot_cls, 4) is None
    shot = session.get(shot_cls, 5)
    calls_before = len(fake_sg._calls)
    assert session.get(shot_cls, 5) is shot
    assert len(fake_sg._calls) == calls_before
    fake_sg._errors['find_one'] = [RuntimeError('provider broke')]
    with pytest.raises(RuntimeError) as excinfo:
        session.get(shot_cls, 6)
    assert excinfo.value.__notes__ == ["sgalchemy: raised while executing find_one('Shot', id=6)"]


def test_get_rejects_foreign_registry_before_identity_hit_or_io(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    session = Session(engine)
    session.merge_dict({'type': 'Shot', 'id': 5, 'code': 'local'})
    foreign_cls = Registry().entity_class('Shot')
    with pytest.raises(SgalchemyError, match='registry'):
        session.get(foreign_cls, 5)
    assert fake_sg._calls == []


def test_get_rejects_mismatched_identity_before_registration(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot_cls = engine.entities.Shot
    fake_sg._find_results = [[{'type': 'Shot', 'id': 2, 'code': 'wrong'}]]

    with pytest.raises(SgalchemyError, match=r'requested Shot\(id=1\).*Shot\(id=2\)'):
        session.get(shot_cls, 1)

    assert session._identity_values() == []


def test_get_type_absent_after_refresh_raises_teaching_error(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, vendor_cls = _engine_with_symbolic_vendor(fake_sg, tmp_path)
    session = Session(engine)
    reads_before = [name for name, _args, _kwargs in fake_sg._calls].count('schema_read')

    with pytest.raises(SchemaUnavailableError) as excinfo:
        session.get(vendor_cls, 5)

    assert str(excinfo.value) == SCHEMA_ABSENT_MESSAGE
    calls = [name for name, _args, _kwargs in fake_sg._calls]
    assert calls.count('schema_read') == reads_before + 1
    assert calls.count('find_one') == 0


def test_get_unknown_type_forces_one_refresh_then_succeeds(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    engine, vendor_cls = _engine_with_symbolic_vendor(fake_sg, tmp_path)
    extended = copy.deepcopy(schema_fixture)
    extended['FutureVendor'] = {
        'id': {'data_type': {'value': 'number'}, 'name': {'value': 'Id'}, 'properties': {}},
        'code': {'data_type': {'value': 'text'}, 'name': {'value': 'Code'}, 'properties': {}},
    }
    schema_reads: list[bool] = []

    def extended_read() -> dict[str, Any]:
        schema_reads.append(True)
        return copy.deepcopy(extended)

    monkeypatch.setattr(fake_sg, 'schema_read', extended_read)
    fake_sg._find_results = [[{'type': 'FutureVendor', 'id': 5, 'code': 'V5'}]]
    session = Session(engine)
    hydration_bindings = capture_hydration_bindings(session, monkeypatch)

    found = session.get(vendor_cls, 5)

    assert found is not None
    assert found.id == 5
    assert len(hydration_bindings) == 1
    assert hydration_bindings[0] is engine.prepared_binding()
    assert len(schema_reads) == 1
    assert [name for name, _args, _kwargs in fake_sg._calls].count('find_one') == 1


def test_hydration_alias_clone_preserves_cycle(
    engine: Engine,
    fake_sg: _FakeShotgun,
    schema_fixture: dict[str, Any],
) -> None:
    schema_fixture['Shot']['description']['data_type']['value'] = 'serializable'
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema(force=True)
    session = Session(engine)
    loop: dict[str, Any] = {}
    loop['self'] = loop

    shot = session.merge_dict({'type': 'Shot', 'id': 42, 'description': loop})
    loaded = cast('dict[str, Any]', entity_state(shot).loaded['description'])

    assert loaded is not loop
    assert loaded['self'] is loaded


def test_hydration_alias_detaches_raw_pre_schema_container(engine: Engine) -> None:
    session = Session(engine)
    shared = ['2026-07-25']
    payload = {'left': shared, 'right': shared}

    shot = session.merge_dict({'type': 'Shot', 'id': 43, 'description': payload})
    loaded = cast('dict[str, list[str]]', entity_state(shot).loaded['description'])
    shared.append('2026-07-26')

    assert loaded == {'left': ['2026-07-25'], 'right': ['2026-07-25']}
    assert type(loaded) is dict
    assert type(loaded['left']) is list
    assert type(loaded['left'][0]) is str
    assert loaded['left'] is loaded['right']


def test_hydration_detaches_mutable_container_graph(
    engine: Engine,
    fake_sg: _FakeShotgun,
    schema_fixture: dict[str, Any],
) -> None:
    schema_fixture['Shot']['description']['data_type']['value'] = 'serializable'
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema(force=True)
    session = Session(engine)
    linked = session.merge_dict({'type': 'Asset', 'id': 9, 'code': 'asset'})
    shared = {'values': [1]}
    payload = {'left': shared, 'right': shared, 'linked': linked}
    shot = session.merge_dict({'type': 'Shot', 'id': 41, 'description': payload})
    loaded = entity_state(shot).loaded['description']

    shared['values'].append(2)

    assert loaded == {'left': {'values': [1]}, 'right': {'values': [1]}, 'linked': linked}
    assert loaded['left'] is loaded['right']
    assert loaded['linked'] is linked


def test_hydration_detaches_shared_mutable_descendant_in_exact_tuple(engine: Engine) -> None:
    session = Session(engine)
    shared = [1]
    payload = (shared, shared)

    shot = session.merge_dict({'type': 'Shot', 'id': 46, 'description': payload})
    loaded = cast('tuple[list[int], list[int]]', entity_state(shot).loaded['description'])
    shared.append(2)

    assert type(loaded) is tuple
    assert loaded is not payload
    assert loaded == ([1], [1])
    assert loaded[0] is loaded[1]
    assert loaded[0] is not shared


def test_hydration_fast_path_avoids_full_identity_scans(engine: Engine) -> None:
    engine.load_schema()
    session = Session(engine)
    first = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'first'})
    identity = _NoScanIdentity(session._state.identity_map)
    session._state._identity = identity

    second = session.merge_dict({'type': 'Shot', 'id': 2, 'code': 'second'})
    reused = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'updated'})

    assert second.code == 'second'
    assert reused is first
    assert first.code == 'updated'


def test_hydration_scope_reuses_live_identity_and_clears_after_link_failure(
    engine: Engine,
) -> None:
    engine.load_schema()
    session = Session(engine)
    staged = session.merge_dict({'type': 'Shot', 'id': 99})
    session.delete(staged)
    queue = YieldCountingEntities(session.unit_of_work._deleted)
    session.unit_of_work._deleted = queue

    def hydrate_response() -> None:
        with session.hydration_scope():
            first = session.merge_dict({'type': 'Project', 'id': 1})
            with session.hydration_scope():
                shot = session.merge_dict(
                    {'type': 'Shot', 'id': 1, 'project': {'type': 'Project', 'id': 1}}
                )
            assert shot.project is first
            session.merge_dict({'type': 'Shot', 'id': 2, 'project': {'type': 'Vendor', 'id': 1}})

    with pytest.raises(SgalchemyError, match='display alias'):
        hydrate_response()
    assert queue._visits == 1
    assert session._state._hydration_index is None

    # A later operation observes canceled staging, including lifecycle precedence.
    session.revive(staged)
    binding = engine.prepared_binding()
    assert (
        session.hydrate_record('Shot', {'type': 'Shot', 'id': 99}, binding, retired=True) is staged
    )
    assert inspect(staged).deleted is True
    session.revive(staged)
    session.hydrate_record('Shot', {'type': 'Shot', 'id': 99}, binding, retired=False)
    assert inspect(staged).deleted is True


def test_hydration_treats_builtin_container_subclass_as_opaque_leaf(engine: Engine) -> None:
    session = Session(engine)
    opaque = _OpaqueDict({'nested': [1]})
    payload = {'opaque': opaque}

    shot = session.merge_dict({'type': 'Shot', 'id': 45, 'description': payload})
    loaded = cast('dict[str, Any]', entity_state(shot).loaded['description'])

    assert loaded is not payload
    assert loaded['opaque'] is opaque


def test_hydration_treats_non_wire_mapping_as_opaque_leaf(engine: Engine) -> None:
    session = Session(engine)
    opaque = _OpaqueMapping({'nested': [1]})
    proxy = MappingProxyType({'nested': [2]})
    sibling = [3]
    payload = {'opaque': opaque, 'proxy': proxy, 'sibling': sibling}

    shot = session.merge_dict({'type': 'Shot', 'id': 44, 'description': payload})
    loaded = cast('dict[str, Any]', entity_state(shot).loaded['description'])
    sibling.append(4)

    assert loaded is not payload
    assert loaded['opaque'] is opaque
    assert loaded['proxy'] is proxy
    assert loaded['sibling'] == [3]
    assert loaded['sibling'] is not sibling


def test_identity_hit_skips_binding_and_lazy_resolve_captures_fresh_binding(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    registry = Registry()
    engine = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'first',
        registry=registry,
        bind_default=False,
    )
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 8}]}],
        [{'type': 'Asset', 'id': 8, 'code': '2026-07-25'}],
    ]
    session = Session(engine)
    shot_cls = registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    refreshed_schema = copy.deepcopy(schema_fixture)
    refreshed_schema['Asset']['code']['data_type']['value'] = 'date'
    refresher_sg = type(fake_sg)()
    monkeypatch.setattr(
        refresher_sg,
        'schema_read',
        lambda: copy.deepcopy(refreshed_schema),
    )
    refresher = create_engine(
        connection=refresher_sg,
        cache_root=tmp_path / 'refresher',
        registry=registry,
        bind_default=False,
    )
    refresher.load_schema()
    load_binding = engine.load_binding
    binding_reads = 0

    def track_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nonlocal binding_reads
            binding_reads += 1
            return call(*args, **kwargs)

        return wrapper

    tracking_load_binding = track_call(load_binding)

    monkeypatch.setattr(engine, 'load_binding', tracking_load_binding)

    assert session.get(shot_cls, 1) is shot
    assert binding_reads == 0
    assert shot.assets[0].code == datetime.date(2026, 7, 25)
    assert binding_reads == 1


@pytest.mark.parametrize(
    'operation',
    ['expire', 'expire_all', 'get', 'merge_dict', 'refresh', 'scalars'],
)
def test_indeterminate_entity_rejected_by_public_session_object_operations(
    engine: Engine,
    fake_sg: _FakeShotgun,
    operation: str,
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 5, 'code': 'original'})
    entity_state(shot).indeterminate = True
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 5, 'code': 'server'}],
    ]
    shot_cls = engine.registry.entity_class('Shot')
    calls = {
        'expire': lambda: session.expire(shot),
        'expire_all': session.expire_all,
        'get': lambda: session.get(shot_cls, 5),
        'merge_dict': lambda: session.merge_dict({'type': 'Shot', 'id': 5, 'code': 'server'}),
        'refresh': lambda: session.refresh(shot),
        'scalars': lambda: session.scalars(select(shot_cls)),
    }
    provider_calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='indeterminate server outcome'):
        calls[operation]()

    assert entity_state(shot).loaded['code'] == 'original'
    assert not entity_state(shot).expired
    if operation in {'expire', 'expire_all', 'get', 'merge_dict', 'refresh'}:
        assert len(fake_sg._calls) == provider_calls_before


def test_invalid_date_is_preserved_and_valid_datetime_is_coerced(
    engine: Engine, fake_sg: _FakeShotgun, caplog: pytest.LogCaptureFixture
) -> None:
    schema = fake_sg.schema_read()
    schema['Shot']['created_at'] = {
        'data_type': {'value': 'date_time'},
        'name': {'value': 'Created At'},
        'properties': {},
    }
    fake_sg.schema_read = lambda: schema
    engine.load_schema(force=True)
    session = Session(engine)
    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        invalid = session.merge_dict({'type': 'Shot', 'id': 1, 'sg_due_date': 'not-a-date'})
    valid = session.merge_dict(
        {'type': 'Shot', 'id': 2, 'created_at': '2026-07-22T12:34:56+00:00'}
    )
    assert invalid.sg_due_date == 'not-a-date'
    assert valid.created_at == datetime.datetime.fromisoformat('2026-07-22T12:34:56+00:00')
    assert 'schema shape mismatch' in caplog.text


def test_joinedload_null_link_with_non_null_projection_remains_schema_contradiction(
    engine: Engine,
    fake_sg: _FakeShotgun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine.load_schema()
    row = {
        'type': 'Shot',
        'id': 1,
        'code': 'null-link',
        'sg_sequence': None,
        'sg_sequence.Sequence.code': 'unexpected',
    }
    fake_sg._find_results = [[row]]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        shot = session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence)))[0]

    path = 'sg_sequence.Sequence.code'
    assert entity_state(shot).loaded[path] == 'unexpected'
    assert shot.as_dict()[path] == 'unexpected'
    assert 'schema shape mismatch' in caplog.text
    engine.load_schema()
    assert [name for name, _args, _kwargs in fake_sg._calls].count('schema_read') == 2


def test_joinedload_projections_do_not_clear_link_expiration(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    engine.load_schema()
    session = Session(engine)
    sequence = session.merge_dict(
        {'type': 'Sequence', 'id': 7, 'code': 'old-code', 'episode': None}
    )
    session.expire(sequence)
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 5,
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.id': 7,
                'sg_sequence.Sequence.code': 'fresh-code',
                'sg_sequence.Sequence.episode': None,
            }
        ]
    ]
    shot_cls = engine.registry.entity_class('Shot')

    shot = session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence)))[0]

    assert shot.sg_sequence is sequence
    assert entity_state(sequence).expired
    with pytest.raises(StaleObjectError):
        _ = sequence.code


def test_merge_without_schema_has_no_io_and_validates_identity(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    session = Session(engine)
    ambiguous = {'payload': {'type': 'Unknown', 'id': 7}, 'items': [{'type': 'X', 'id': 8}]}
    obj = session.merge_dict({'type': 'Shot', 'id': 11, **ambiguous})
    assert obj.as_dict() == {'type': 'Shot', 'id': 11, **ambiguous}
    assert fake_sg._calls == []
    for raw, message in [
        ({'type': '', 'id': 1}, "non-empty string 'type'"),
        ({'type': 'Shot', 'id': True}, 'positive integer id'),
        ({'type': 'Shot', 'id': 0}, 'positive integer id'),
    ]:
        with pytest.raises(SgalchemyError, match=message):
            session.merge_dict(raw)


def test_no_work_commit_does_not_load_schema(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(engine)

    def unexpected_load(*, force: bool = False) -> Never:
        del force
        raise AssertionError('no-work commit must not load schema')

    monkeypatch.setattr(engine, 'load_binding', unexpected_load)

    session.commit()


def test_non_auth_sole_batch_failure_keeps_staged_note_without_auth_guidance(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    session.add(shot_cls.new(code='other-fail'))
    fake_sg._errors.setdefault('batch', []).append(RuntimeError('boom'))

    with pytest.raises(RuntimeError) as excinfo:
        session.commit()

    joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
    assert 'remain staged' in joined
    assert 'no automatic retry' not in joined


def test_old_session_uses_shared_registry_refresh_for_compile_and_hydration(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    registry = Registry()
    first = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'first',
        registry=registry,
        bind_default=False,
    )
    first.load_schema()
    refreshed_schema = copy.deepcopy(schema_fixture)
    refreshed_schema['Shot']['description']['data_type']['value'] = 'date'
    refresher_sg = type(fake_sg)()
    monkeypatch.setattr(
        refresher_sg,
        'schema_read',
        lambda: copy.deepcopy(refreshed_schema),
    )
    refresher = create_engine(
        connection=refresher_sg,
        cache_root=tmp_path / 'refresher',
        registry=registry,
        bind_default=False,
    )
    refresher.load_schema()
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'description': '2026-07-25'}],
    ]
    session = Session(first)
    shot_cls = registry.entity_class('Shot')

    shot = session.scalars(select(shot_cls))[0]

    assert shot.description == datetime.date(2026, 7, 25)


def test_partial_hydration_does_not_clear_whole_object_expiration(engine: Engine) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict(
        {'type': 'Shot', 'id': 5, 'code': 'old-code', 'description': 'old-description'}
    )
    session.expire(shot)

    session.merge_dict({'type': 'Shot', 'id': 5, 'code': 'new-code'})

    assert entity_state(shot).expired
    with pytest.raises(StaleObjectError):
        _ = shot.description

    session.merge_dict(
        {'type': 'Shot', 'id': 5, 'code': 'fresh-code', 'description': 'fresh-description'}
    )
    assert not entity_state(shot).expired
    assert (shot.code, shot.description) == ('fresh-code', 'fresh-description')


def test_read_auth_failure_carries_read_recovery_note(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    fault = AUTHENTICATION_FAULT('session expired')
    fake_sg._errors.setdefault('find', []).append(fault)

    with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
        session.scalars(select(shot_cls))

    assert excinfo.value is fault
    joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
    assert 'rerun the read' in joined
    assert 'remain staged' not in joined


def test_refresh_expire_all_and_missing(engine: Engine, fake_sg: _FakeShotgun) -> None:
    session = Session(engine)
    first = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'A'})
    second = session.merge_dict({'type': 'Shot', 'id': 2, 'code': 'B'})
    session.expire_all()
    with pytest.raises(StaleObjectError):
        _ = first.code
    assert entity_state(second).expired
    first_state = entity_state(first)
    first_state.loaded.pop('code')
    first_state.refresh_required.update({'code', 'description'})
    first_state.unverified_writes.update({'sg_status_list', 'sg_cut_duration'})
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'code': 'new',
                'sg_status_list': 'ip',
            }
        ],
        [],
    ]
    session.refresh(first)
    assert first.code == 'new'
    assert first.sg_status_list == 'ip'
    assert first_state.refresh_required == {'description'}
    assert first_state.unverified_writes == {'sg_cut_duration'}
    calls_after_refresh = list(fake_sg._calls)
    info = inspect(first)
    assert repr(first) == "Shot(id=1, code='new')"
    assert first.as_dict()['code'] == 'new'
    assert info.refresh_required == frozenset({'description'})
    assert info.unverified_writes == frozenset({'sg_cut_duration'})
    assert fake_sg._calls == calls_after_refresh
    with pytest.raises(SgalchemyError, match='no longer exists'):
        session.refresh(second)


def test_refresh_opaque_object_succeeds_after_one_forced_refresh(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    engine, _vendor_cls = _engine_with_symbolic_vendor(fake_sg, tmp_path)
    extended = copy.deepcopy(schema_fixture)
    extended['FutureVendor'] = {
        'id': {'data_type': {'value': 'number'}, 'name': {'value': 'Id'}, 'properties': {}},
        'code': {'data_type': {'value': 'text'}, 'name': {'value': 'Code'}, 'properties': {}},
    }
    schema_reads: list[bool] = []

    def extended_read() -> dict[str, Any]:
        schema_reads.append(True)
        return copy.deepcopy(extended)

    monkeypatch.setattr(fake_sg, 'schema_read', extended_read)
    session = Session(engine)
    opaque = session.merge_dict({'type': 'FutureVendor', 'id': 9})
    hydration_bindings = capture_hydration_bindings(session, monkeypatch)
    fake_sg._find_results = [[{'type': 'FutureVendor', 'id': 9, 'code': 'V9'}]]

    session.refresh(opaque)

    assert opaque.code == 'V9'
    assert len(hydration_bindings) == 1
    assert hydration_bindings[0] is engine.prepared_binding()
    assert len(schema_reads) == 1


def test_refresh_opaque_object_teaches_after_one_forced_refresh(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, _vendor_cls = _engine_with_symbolic_vendor(fake_sg, tmp_path)
    session = Session(engine)
    opaque = session.merge_dict({'type': 'FutureVendor', 'id': 9})
    reads_before = [name for name, _args, _kwargs in fake_sg._calls].count('schema_read')

    with pytest.raises(SchemaUnavailableError) as excinfo:
        session.refresh(opaque)

    assert str(excinfo.value) == SCHEMA_ABSENT_MESSAGE
    calls = [name for name, _args, _kwargs in fake_sg._calls]
    assert calls.count('schema_read') == reads_before + 1
    assert calls.count('find') == 0


def test_refresh_preserves_dirty_values(engine: Engine, fake_sg: _FakeShotgun) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
    shot.code = 'local'
    state = entity_state(shot)
    state.refresh_required.add('code')
    state.unverified_writes.add('code')
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'server'}]]
    session.refresh(shot)
    assert state.loaded['code'] == 'local'
    assert state.refresh_required == {'code'}
    assert state.unverified_writes == {'code'}
    assert not state.expired


def test_refresh_rejects_foreign_registry_state_without_io(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    session = Session(engine)
    foreign_cls = Registry().entity_class('Shot')
    foreign = foreign_cls.__new__(foreign_cls)
    initialize_entity(foreign, EntityState('Shot', 1, session=session))
    with pytest.raises(SgalchemyError, match='registry'):
        session.refresh(foreign)
    assert fake_sg._calls == []


@pytest.mark.parametrize(
    ('rows', 'message'),
    [
        ([{'type': 'Shot', 'id': 2, 'code': 'wrong'}], r'requested Shot\(id=1\)'),
        (
            [
                {'type': 'Shot', 'id': 1, 'code': 'first'},
                {'type': 'Shot', 'id': 1, 'code': 'second'},
            ],
            'exactly one.*2',
        ),
    ],
)
def test_refresh_rejects_mismatched_or_multiple_rows_without_mutation(
    engine: Engine,
    fake_sg: _FakeShotgun,
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'before'})
    session.expire(shot)
    state = entity_state(shot)
    loaded_before = dict(state.loaded)
    fake_sg._find_results = [rows]

    with pytest.raises(SgalchemyError, match=message):
        session.refresh(shot)

    assert state.loaded == loaded_before
    assert state.expired is True
    assert session._identity_values() == [shot]


def test_refresh_retired_entity_requests_retired_only(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    retired = session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'old-retired'})
    active = session.merge_dict({'type': 'Shot', 'id': 42, 'code': 'old-active'})
    session.delete(retired)
    session.commit()
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 41, 'code': 'retired'}],
        [{'type': 'Shot', 'id': 42, 'code': 'active'}],
    ]

    session.refresh(retired)

    refresh_find = [call for call in fake_sg._calls if call[0] == 'find'][-1]
    assert refresh_find[2]['retired_only'] is True
    assert retired.code == 'retired'

    session.refresh(active)

    refresh_find = [call for call in fake_sg._calls if call[0] == 'find'][-1]
    assert 'retired_only' not in refresh_find[2]
    assert active.code == 'active'


def test_rejected_hydration_does_not_publish_identity_phantom(engine: Engine) -> None:
    source = Session(engine)
    detached = source.merge_dict({'type': 'Shot', 'id': 41, 'code': 'detached'})
    source.close()
    session = Session(engine)
    session.delete(detached)

    with pytest.raises(SgalchemyError, match='different Shot'):
        session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'server'})

    assert session._identity_values() == []
    session.rollback()
    assert session._identity_values() == []


def test_repeated_joinedload_with_null_link_does_not_stale_schema(
    engine: Engine,
    fake_sg: _FakeShotgun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine.load_schema()
    row = {
        'type': 'Shot',
        'id': 1,
        'code': 'null-link',
        'sg_sequence': None,
        'sg_sequence.Sequence.id': None,
        'sg_sequence.Sequence.code': None,
        'sg_sequence.Sequence.episode': None,
    }
    fake_sg._find_results = [[dict(row)], [dict(row)], [dict(row)]]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    statement = select(shot_cls).options(joinedload(shot_cls.sg_sequence))

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        shots = [session.scalars(statement)[0] for _ in range(3)]

    assert [name for name, _args, _kwargs in fake_sg._calls].count('schema_read') == 1
    assert 'schema shape mismatch' not in caplog.text
    assert all('.' not in key for key in entity_state(shots[-1]).loaded)
    assert all('.' not in key for key in shots[-1].as_dict())


def test_retired_query_marks_root_deleted(engine: Engine, fake_sg: _FakeShotgun) -> None:
    engine.load_schema()
    session = Session(engine)
    fake_sg._find_results = [[{'type': 'Shot', 'id': 41, 'code': 'retired'}]]
    shot = session.scalars(select(engine.entities.Shot).execution_options(retired_only=True))[0]
    assert inspect(shot).deleted is True
    session.revive(shot)
    assert len(session.unit_of_work.revived) == 1
    assert session.unit_of_work.revived[0] is shot


def test_scalar_rehydration_clears_root_and_link_expiry(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 1,
            'sg_sequence': {'type': 'Sequence', 'id': 7, 'code': 'old'},
        }
    )
    sequence = entity_state(shot).loaded['sg_sequence']
    session.expire(shot)
    session.expire(sequence)
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7, 'code': 'new'},
            }
        ]
    ]
    shot_cls = engine.registry.entity_class('Shot')
    assert session.scalars(select(shot_cls))[0] is shot
    assert not entity_state(shot).expired
    assert not entity_state(sequence).expired
    assert sequence.code == 'new'


def test_scalars_identity_date_links_and_dirty_shield(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'code': 'A',
                'sg_due_date': '2026-07-01',
                'sg_sequence': {'type': 'Sequence', 'id': 7, 'name': 'SEQ'},
                'sg_sequence.Sequence.episode': {'type': 'Episode', 'id': 9},
                'sg_sequence.Sequence.episode.Episode.code': 'EP01',
            }
        ],
        [{'type': 'Shot', 'id': 1, 'code': 'server'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls))[0]
    assert shot.id == 1
    assert shot.sg_due_date == datetime.date(2026, 7, 1)
    assert shot.sg_sequence.name == 'SEQ'
    assert shot.sg_sequence.episode.code == 'EP01'
    shot.code = 'local'
    assert session.scalars(select(shot_cls))[0] is shot
    assert shot.code == 'local'


def test_scalars_rejects_foreign_registry_before_io(engine: Engine, fake_sg: _FakeShotgun) -> None:
    foreign_cls = Registry().entity_class('Shot')
    session = Session(engine)
    with pytest.raises(SgalchemyError, match='registry'):
        session.scalars(select(foreign_cls))
    assert fake_sg._calls == []


@pytest.mark.parametrize('retired', [False, True])
@pytest.mark.parametrize('child_baseline', [None, False, True])
def test_scalars_retirement_observation_is_root_only(
    engine: Engine,
    fake_sg: _FakeShotgun,
    retired: bool,
    child_baseline: bool | None,
) -> None:
    engine.load_schema()
    session = Session(engine)
    child = None
    if child_baseline is not None:
        child = session.merge_dict({'type': 'Sequence', 'id': 51, 'code': 'before'})
        entity_state(child).deleted = child_baseline
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 41,
                'sg_sequence': {'type': 'Sequence', 'id': 51},
                'sg_sequence.Sequence.code': 'linked',
            }
        ]
    ]

    shot = session.scalars(select(engine.entities.Shot).execution_options(retired_only=retired))[0]

    assert inspect(shot).deleted is retired
    assert isinstance(shot.sg_sequence, Entity)
    assert inspect(shot.sg_sequence).deleted is bool(child_baseline)
    assert shot.sg_sequence.code == 'linked'
    if child is not None:
        assert shot.sg_sequence is child


@pytest.mark.parametrize('retired', [False, True])
@pytest.mark.parametrize(
    ('baseline', 'transition'),
    [(None, None), (False, None), (True, None), (False, 'delete'), (True, 'revive')],
)
def test_scalars_retirement_observation_preserves_staged_intent(
    engine: Engine,
    fake_sg: _FakeShotgun,
    retired: bool,
    baseline: bool | None,
    transition: str | None,
) -> None:
    engine.load_schema()
    session = Session(engine)
    original = None
    if baseline is not None:
        original = session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'before'})
        state = entity_state(original)
        state.deleted = baseline
        original.code = 'local'
        if transition is not None:
            getattr(session, transition)(original)
        state.refresh_required.update({'code', 'description', 'sg_status_list'})
        state.unverified_writes.update({'code', 'description', 'sg_status_list'})
    deleted_before = tuple(session.unit_of_work.deleted)
    revived_before = tuple(session.unit_of_work.revived)
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 41, 'code': 'server', 'description': 'refreshed'}]
    ]

    shot = session.scalars(select(engine.entities.Shot).execution_options(retired_only=retired))[0]

    assert inspect(shot).deleted is (baseline if transition is not None else retired)
    assert tuple(map(id, session.unit_of_work.deleted)) == tuple(map(id, deleted_before))
    assert tuple(map(id, session.unit_of_work.revived)) == tuple(map(id, revived_before))
    if original is not None:
        assert shot is original
        state = entity_state(shot)
        assert state.dirty == {'code': 'local'}
        assert state.loaded['code'] == 'local'
        assert state.refresh_required == {'code', 'sg_status_list'}
        assert state.unverified_writes == {'code', 'sg_status_list'}
    else:
        assert shot.code == 'server'
    assert shot.description == 'refreshed'
    session.merge_dict({'type': 'Shot', 'id': 41, 'description': 'merged'})
    assert inspect(shot).deleted is (baseline if transition is not None else retired)


def test_schema_recovery_never_refreshes_after_first_submitted_write(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    registry = Registry()
    engine = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'first',
        registry=registry,
        bind_default=False,
    )
    engine.load_schema()
    session = Session(engine)
    shot_cls = registry.entity_class('Shot')
    shot = shot_cls.new(code='planned')
    session.add(shot)
    batch_entered = Event()
    release_batch = Event()
    original_batch = fake_sg.batch

    def blocking_batch(requests: list[dict[str, Any]]) -> list[Any]:
        batch_entered.set()
        assert release_batch.wait(timeout=10)
        results = original_batch(requests)
        assert isinstance(results[0], dict)
        results[0]['description'] = '2026-07-25'
        return results

    monkeypatch.setattr(fake_sg, 'batch', blocking_batch)
    refreshed_schema = copy.deepcopy(schema_fixture)
    refreshed_schema['Shot']['description']['data_type']['value'] = 'date'
    refresher_sg = type(fake_sg)()
    monkeypatch.setattr(
        refresher_sg,
        'schema_read',
        lambda: copy.deepcopy(refreshed_schema),
    )
    refresher = create_engine(
        connection=refresher_sg,
        cache_root=tmp_path / 'refresher',
        registry=registry,
        bind_default=False,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        committing = pool.submit(session.commit)
        assert batch_entered.wait(timeout=2)
        refresher.load_schema()
        release_batch.set()
        committing.result(timeout=2)

    assert shot.description == '2026-07-25'
    refreshed = session.merge_dict({'type': 'Shot', 'id': 2, 'description': '2026-07-25'})
    assert refreshed.description == datetime.date(2026, 7, 25)
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 1


def test_selectin_secondary_auth_failure_keeps_primaries_and_read_note(
    engine: Engine, fake_sg: _FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'code': 'sq010',
                'assets': [{'type': 'Asset', 'id': 200}],
            }
        ],
    ]
    original_find = fake_sg.find
    find_calls: list[int] = []
    fault = AUTHENTICATION_FAULT('session expired')

    def track_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            find_calls.append(1)
            if len(find_calls) >= 2:
                raise fault
            return call(*args, **kwargs)

        return wrapper

    flaky_find = track_call(original_find)

    monkeypatch.setattr(fake_sg, 'find', flaky_find)
    statement = select(shot_cls).options(selectinload(shot_cls.assets))

    with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
        session.scalars(statement)

    assert excinfo.value is fault
    joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
    assert 'rerun the read' in joined
    assert list(session._identity_values())


def test_session_operation_keeps_entry_binding_through_midflight_refresh(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    initial_schema = copy.deepcopy(schema_fixture)
    initial_schema['Shot']['description']['data_type']['value'] = 'date'
    monkeypatch.setattr(fake_sg, 'schema_read', lambda: copy.deepcopy(initial_schema))
    registry = Registry()
    engine = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'first',
        registry=registry,
        bind_default=False,
    )
    engine.load_schema()
    session = Session(engine)
    shot_cls = registry.entity_class('Shot')
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'description': '2026-07-25'}],
        [{'type': 'Shot', 'id': 1, 'description': '2026-07-25'}],
    ]
    find_entered = Event()
    release_find = Event()
    original_find = fake_sg.find

    def block_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            find_entered.set()
            assert release_find.wait(timeout=10)
            return call(*args, **kwargs)

        return wrapper

    blocking_find = block_call(original_find)

    monkeypatch.setattr(fake_sg, 'find', blocking_find)
    refresher_sg = type(fake_sg)()
    refresher = create_engine(
        connection=refresher_sg,
        cache_root=tmp_path / 'refresher',
        registry=registry,
        bind_default=False,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        operation = pool.submit(session.scalars, select(shot_cls))
        assert find_entered.wait(timeout=2)
        refresher.load_schema()
        release_find.set()
        shot = operation.result(timeout=2)[0]

    assert shot.description == datetime.date(2026, 7, 25)
    monkeypatch.setattr(fake_sg, 'find', original_find)
    assert session.scalars(select(shot_cls))[0].description == '2026-07-25'


@pytest.mark.parametrize(
    ('field_code', 'value'),
    [
        ('code', ['not', 'scalar']),
        ('sg_sequence', 'not-a-link'),
        ('sg_sequence', {'type': 'Sequence', 'id': True}),
        ('assets', 'not-a-list'),
        ('assets', [{'type': 'Asset', 'id': 0}]),
    ],
)
def test_shape_mismatch_preserves_value_and_marks_stale(
    engine: Engine,
    fake_sg: _FakeShotgun,
    field_code: str,
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine.load_schema()
    session = Session(engine)
    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        obj = session.merge_dict({'type': 'Shot', 'id': 10, field_code: value})
    assert entity_state(obj).loaded[field_code] == value
    assert 'schema shape mismatch' in caplog.text
    engine.load_schema()
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2


@pytest.mark.parametrize('size', [100, 200, 400])
def test_response_hydration_bounds_staged_item_visits(
    engine: Engine, fake_sg: _FakeShotgun, size: int
) -> None:
    engine.load_schema()
    session = Session(engine)
    deleted = [session.merge_dict({'type': 'Shot', 'id': 1000 + i}) for i in range(size)]
    revived = [session.merge_dict({'type': 'Shot', 'id': 2000 + i}) for i in range(size)]
    for obj in deleted:
        session.delete(obj)
    for obj in revived:
        entity_state(obj).deleted = True
        session.revive(obj)
    deleted_queue = YieldCountingEntities(session.unit_of_work._deleted)
    revived_queue = YieldCountingEntities(session.unit_of_work._revived)
    session.unit_of_work._deleted = deleted_queue
    session.unit_of_work._revived = revived_queue
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': i + 1, 'project': {'type': 'Project', 'id': i + 1}}
            for i in range(size)
        ]
    ]

    objects = session.scalars(select(engine.entities.Shot))

    assert len(objects) == size
    assert len(session._identity_values()) == 4 * size
    assert objects[0].project.id == 1
    assert deleted_queue._visits + revived_queue._visits <= 6 * size


@pytest.mark.parametrize('collision_first', [False, True])
def test_hydration_scope_preserves_all_raw_list_collisions(
    engine: Engine, collision_first: bool
) -> None:
    engine.load_schema()
    session = Session(engine)
    owned = session.merge_dict({'type': 'Shot', 'id': 1})
    detached_session = Session(engine)
    collision = detached_session.merge_dict({'type': 'Shot', 'id': 1})
    detached_session.close()
    candidates = [collision, owned] if collision_first else [owned, collision]
    session.unit_of_work._deleted.extend(candidates)

    with pytest.raises(SgalchemyError, match='different Shot'):
        session.merge_dict({'type': 'Shot', 'id': 1})

    assert session._state._hydration_index is None
    session.unit_of_work._deleted[:] = [owned]
    assert session.merge_dict({'type': 'Shot', 'id': 1}) is owned


def test_shared_access_guard_preserves_sync_ownership(engine: Engine) -> None:
    session = Session(engine)
    first = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'first'})
    second = session.merge_dict({'type': 'Shot', 'id': 2, 'code': 'second'})
    first_state = entity_state(first)
    second_state = entity_state(second)
    first_state.session = None

    with pytest.raises(SgalchemyError, match='no longer owned'):
        _ = first.code

    assert first_state.access_guard is second_state.access_guard


def test_sole_batch_auth_failure_notes_state_local_facts_only(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    session.add(shot_cls.new(code='auth-fail'))
    fault = AUTHENTICATION_FAULT('session expired')
    fake_sg._errors.setdefault('batch', []).append(fault)

    with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
        session.commit()

    assert excinfo.value is fault
    joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
    assert 'its server outcome is unknown' in joined
    assert 'remain staged' in joined
    assert 'no automatic retry' in joined
    assert 'nothing committed' not in joined
    assert 'did not execute' not in joined
    assert sum('no automatic retry' in note for note in excinfo.value.__notes__) == 1


def test_stale_schema_fault_is_annotated(engine: Engine, fake_sg: _FakeShotgun) -> None:
    engine.load_schema()
    engine.mark_schema_stale()
    error = RuntimeError('schema refresh unavailable')
    fake_sg._errors['schema_read'] = [error]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    with pytest.raises(RuntimeError) as excinfo:
        session.scalars(select(shot_cls))
    assert excinfo.value is error
    assert error.__notes__[0].startswith('sgalchemy:')


def test_state_hydration_rejects_cross_identity_corruption(engine: Engine) -> None:
    session = Session(engine)
    with pytest.raises(SgalchemyError, match='contradicts expected'):
        session._state._hydrate('Shot', {'type': 'Asset', 'id': 1})
    with pytest.raises(SgalchemyError, match='positive integer id'):
        session._state._hydrate('Shot', {'type': 'Shot', 'id': False})


def test_summary_returns_raw_payload(engine: Engine) -> None:
    engine.load_schema()
    shot_cls = engine.registry.entity_class('Shot')
    session = Session(engine)
    payload = session.execute(select(func.count(shot_cls.id)))
    assert payload == {'summaries': {}, 'groups': []}


def test_unknown_wire_link_becomes_one_stable_opaque_entity(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot = session.merge_dict(
        {'type': 'Shot', 'id': 1, 'sg_vendor': {'type': 'FutureVendor', 'id': 9}}
    )
    opaque = cast('Entity', shot.sg_vendor)
    opaque_cls = type(opaque)

    assert opaque.as_dict() == {'type': 'FutureVendor', 'id': 9}
    assert session.merge_dict({'type': 'FutureVendor', 'id': 9}) is opaque
    with pytest.raises(SchemaUnavailableError, match='FutureVendor'):
        _ = opaque.code
    with pytest.raises(SchemaUnavailableError):
        hasattr(opaque, 'code')
    with pytest.raises(AttributeError):
        _ = opaque._private_probe
    calls_before_query = list(fake_sg._calls)
    with pytest.raises(CompilationError, match=r'FutureVendor.*schema.*refresh'):
        session.scalars(select(opaque_cls))
    assert fake_sg._calls == calls_before_query

    binding = engine.prepared_binding()
    assert binding is not None
    future_schema = replace(
        binding.schemas['CustomEntity03'],
        type_name='FutureVendor',
        display_name='Future Vendor',
    )
    engine.registry.bind(
        {**binding.schemas, 'FutureVendor': future_schema},
        key=binding.key,
    )

    assert engine.registry.entity_class('FutureVendor') is opaque_cls
    assert session.merge_dict({'type': 'FutureVendor', 'id': 9, 'code': 'FV-9'}) is opaque
    assert opaque.code == 'FV-9'
    shot.sg_vendor = opaque
    assert shot.sg_vendor is opaque


def test_unprepared_engine_hydration_keeps_captured_absence_with_shared_binding(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    registry = Registry()
    prepared = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'prepared',
        registry=registry,
        bind_default=False,
    )
    prepared.load_schema()
    unprepared_sg = type(fake_sg)()
    unprepared = create_engine(
        connection=unprepared_sg,
        cache_root=tmp_path / 'unprepared',
        registry=registry,
        bind_default=False,
    )
    session = Session(unprepared)

    alias = {'type': 'Vendor', 'id': 7, 'code': 'raw'}
    operations = (
        lambda: session.merge_dict(alias),
        lambda: session.hydrate_record('Vendor', alias, None),
    )
    for operation in operations:
        with pytest.raises(SgalchemyError, match=r'Vendor.*alias.*CustomEntity03'):
            operation()
        assert session._identity_values() == []

    future = session.merge_dict(
        {
            'type': 'FutureVendor',
            'id': 7,
            'code': 'raw',
            'payload': {'type': 'Shot', 'id': 1},
        }
    )

    assert type(future).__name__ == 'FutureVendor'
    assert future.as_dict()['payload'] == {'type': 'Shot', 'id': 1}
    assert unprepared_sg._calls == []

    unprepared.load_schema()
    prepared_shot = session.merge_dict({'type': 'Shot', 'id': 8, 'sg_due_date': '2026-07-25'})
    assert prepared_shot.sg_due_date == datetime.date(2026, 7, 25)


def test_unrelated_sgalchemy_note_does_not_suppress_current_context(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    error = RuntimeError('provider unavailable')
    error.add_note('sgalchemy: prior unrelated context')
    fake_sg._errors['find'] = [error]
    session = Session(engine)
    call = FindCall('Shot', [], ['id'])
    binding = engine.load_binding()
    with pytest.raises(RuntimeError) as excinfo:
        session.execute_call(call, binding)
    assert excinfo.value is error
    assert error.__notes__[0] == 'sgalchemy: prior unrelated context'
    assert error.__notes__.count(f'sgalchemy: raised while executing {call}') == 1
