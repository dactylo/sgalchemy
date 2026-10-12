import asyncio
import builtins
import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema

import sgalchemy.testing as testing_module
from sgalchemy._core._engine import Engine
from sgalchemy._core._registry import Registry, default_registry, set_default_registry
from sgalchemy._core._statement import select
from sgalchemy._errors import SgalchemyError
from sgalchemy._orm._session import Session
from sgalchemy.schema import _cache
from sgalchemy.testing import _EngineOverOptions, engine_over, mockgun_engine


@dataclass
class _BackendCase:
    _engine: Engine
    _fake: _FakeShotgun | None


def _minimal_schema_cache() -> _cache.SchemaCache:
    return _cache.SchemaCache(
        fields={
            'Shot': {
                'code': {
                    'data_type': {'value': 'text'},
                    'name': {'value': 'Shot Code'},
                    'properties': {},
                }
            }
        },
        entities={'Shot': {'name': {'value': 'Shot'}}},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture(params=('fake', 'mockgun'))
def backend_case(
    request: pytest.FixtureRequest,
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> _BackendCase:
    if request.param == 'fake':
        engine = engine_over(
            fake_sg,
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )
        return _BackendCase(engine, fake_sg)
    pytest.importorskip('shotgun_api3')
    engine = mockgun_engine(fetch_schema(fake_sg), cache_root=tmp_path)
    return _BackendCase(engine, None)


def test_async_mockgun_replaces_same_root_schema(tmp_path: Path) -> None:
    mockgun = pytest.importorskip('shotgun_api3.lib.mockgun')
    if not hasattr(mockgun, 'AsyncShotgun'):
        pytest.skip('selected provider lacks mockgun.AsyncShotgun')

    first_cache = _minimal_schema_cache()
    second_cache = _minimal_schema_cache()
    first_cache.fields['Shot']['code']['name']['value'] = 'First Code'
    second_cache.fields['Shot']['code']['name']['value'] = 'Second Code'
    first_fields = copy.deepcopy(first_cache.fields)
    first_entities = copy.deepcopy(first_cache.entities)
    second_fields = copy.deepcopy(second_cache.fields)
    second_entities = copy.deepcopy(second_cache.entities)

    async def scenario() -> None:
        first = testing_module.async_mockgun_connection(first_cache, tmp_path)
        assert (await first.schema_read())['Shot']['code']['name']['value'] == 'First Code'
        second = testing_module.async_mockgun_connection(second_cache, tmp_path)
        assert (await second.schema_read())['Shot']['code']['name']['value'] == 'Second Code'

    asyncio.run(scenario())
    assert first_cache.fields == first_fields
    assert first_cache.entities == first_entities
    assert second_cache.fields == second_fields
    assert second_cache.entities == second_entities


def test_async_mockgun_reports_missing_async_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = ModuleType('mockgun_without_async')

    def import_mockgun(_name: str) -> ModuleType:
        return module

    monkeypatch.setattr(testing_module, 'import_module', import_mockgun)
    with pytest.raises(SgalchemyError, match=r'lacks mockgun\.AsyncShotgun') as excinfo:
        testing_module.async_mockgun_connection(_minimal_schema_cache(), tmp_path)
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_async_mockgun_reports_provider_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_import(_name: str) -> ModuleType:
        raise ImportError('provider unavailable')

    monkeypatch.setattr(testing_module, 'import_module', fail_import)
    with pytest.raises(SgalchemyError, match=r"install the 'sgalchemy\[async\]' extra") as excinfo:
        testing_module.async_mockgun_connection(_minimal_schema_cache(), tmp_path)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_backend_create_query_update_contract(backend_case: _BackendCase) -> None:
    engine = backend_case._engine
    engine.load_schema()
    shot_cls = engine.entities.Shot
    session = Session(engine)
    pending = shot_cls.new(code='contract-A')
    session.add(pending)
    session.commit()
    if backend_case._fake is not None:
        backend_case._fake._find_results = [
            [{'type': 'Shot', 'id': pending.id, 'code': 'contract-A'}]
        ]
    found = session.scalars(select(shot_cls).where(shot_cls.id == pending.id))[0]
    found.code = 'contract-B'
    session.commit()
    if backend_case._fake is not None:
        batches = [call for call in backend_case._fake._calls if call[0] == 'batch']
        assert len(batches) == 2
        assert batches[-1][1][0] == [
            {
                'request_type': 'update',
                'entity_type': 'Shot',
                'entity_id': pending.id,
                'data': {'code': 'contract-B'},
            }
        ]
    else:
        session.close()
        fresh = Session(engine)
        persisted = fresh.scalars(select(shot_cls).where(shot_cls.id == pending.id))[0]
        assert persisted.code == 'contract-B'
        fresh.close()


def test_engine_over_honors_explicit_default_binding(fake_sg: _FakeShotgun) -> None:
    process_registry = default_registry()

    engine = engine_over(fake_sg, registry=None, bind_default=True)

    assert engine.registry is process_registry


def test_engine_over_honors_explicit_default_binding_without_registry(
    fake_sg: _FakeShotgun,
) -> None:
    process_registry = default_registry()

    engine = engine_over(fake_sg, bind_default=True)

    assert engine.registry is process_registry


def test_engine_over_honors_explicit_none_registry_with_isolation_default(
    fake_sg: _FakeShotgun,
) -> None:
    process_registry = default_registry()

    engine = engine_over(fake_sg, registry=None)

    assert engine.registry is not process_registry


def test_engine_over_honors_explicit_registry(fake_sg: _FakeShotgun) -> None:
    registry = Registry()

    engine = engine_over(fake_sg, registry=registry)
    default_bound = engine_over(fake_sg, registry=registry, bind_default=True)

    assert engine.registry is registry
    assert default_bound.registry is registry


def test_engine_over_isolates_sites_and_process_default(tmp_path: Path) -> None:
    process_registry = default_registry()
    first_connection: Any = _FakeShotgun()
    first_connection.base_url = 'https://first.example.com'
    second_connection: Any = _FakeShotgun()
    second_connection.base_url = 'https://second.example.com'
    try:
        first = engine_over(first_connection, cache_root=tmp_path / 'first')
        second = engine_over(second_connection, cache_root=tmp_path / 'second')

        first.load_schema()
        second.load_schema()
        first_shot = first.entities.Shot
        second_shot = second.entities.Shot

        assert first.registry is not second.registry
        assert first_shot is not second_shot
        replacement = Registry()
        set_default_registry(replacement)
        assert default_registry() is replacement
    finally:
        set_default_registry(process_registry)


def test_engine_over_option_keys() -> None:
    assert _EngineOverOptions.__required_keys__ == frozenset()
    assert _EngineOverOptions.__optional_keys__ == frozenset(
        {'policy', 'echo', 'registry', 'bind_default', 'serialize'}
    )


def test_mockgun_engine_accepts_module_without_legacy_schema_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shotgun_api3.lib import mockgun  # noqa: PLC0415 - Test provider capability shape.

    provider_module = ModuleType('mockgun_without_legacy_schema_cache')
    monkeypatch.setattr(provider_module, 'Shotgun', vars(mockgun)['Shotgun'], raising=False)
    provider_package = ModuleType('shotgun_api3.lib_without_legacy_schema_cache')
    monkeypatch.setattr(provider_package, 'mockgun', provider_module, raising=False)
    original_import = builtins.__import__

    def import_without_legacy_schema_cache(
        name: str,
        global_vars: dict[str, Any] | None = None,
        local_vars: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == 'shotgun_api3.lib' and fromlist == ('mockgun',):
            return provider_package
        return original_import(name, global_vars, local_vars, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', import_without_legacy_schema_cache)

    engine = mockgun_engine(_minimal_schema_cache(), tmp_path)

    assert 'Shot' in engine.schema


def test_mockgun_engine_does_not_mutate_caller_schema(tmp_path: Path) -> None:
    pytest.importorskip('shotgun_api3')
    schema_cache = _minimal_schema_cache()
    original_fields = copy.deepcopy(schema_cache.fields)
    original_entities = copy.deepcopy(schema_cache.entities)

    mockgun_engine(schema_cache, tmp_path)

    assert schema_cache.fields == original_fields
    assert schema_cache.entities == original_entities


def test_mockgun_engine_replaces_stale_same_root_schema(
    fake_sg: _FakeShotgun, tmp_path: Path
) -> None:
    pytest.importorskip('shotgun_api3')
    first_cache = fetch_schema(fake_sg)
    first = mockgun_engine(first_cache, tmp_path)
    first.load_schema()
    fields = copy.deepcopy(first_cache.fields)
    entities = copy.deepcopy(first_cache.entities)
    del fields['Shot']
    del entities['Shot']
    second_cache = _cache.SchemaCache(
        fields=fields,
        entities=entities,
        fetched_at=first_cache.fetched_at,
        server_info=first_cache.server_info,
    )

    second = mockgun_engine(second_cache, tmp_path)
    second.load_schema()

    assert 'Shot' not in second.schema


def test_mockgun_engine_reports_missing_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        global_vars: dict[str, Any] | None = None,
        local_vars: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == 'shotgun_api3.lib':
            raise ImportError('blocked optional provider')
        return original_import(name, global_vars, local_vars, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', blocked_import)

    with pytest.raises(SgalchemyError, match=r'sgalchemy\[official\]') as excinfo:
        mockgun_engine(_minimal_schema_cache(), tmp_path)

    assert isinstance(excinfo.value.__cause__, ImportError)


def test_mockgun_engine_writes_exact_adapted_pickles(tmp_path: Path) -> None:
    pytest.importorskip('shotgun_api3')

    engine = mockgun_engine(_minimal_schema_cache(), tmp_path)

    pickle_paths = sorted(path.name for path in tmp_path.glob('*.pickle'))
    assert pickle_paths == [
        'mockgun_schema.pickle',
        'mockgun_schema_entity.pickle',
    ]
    fields = engine.connection.schema_read()
    entities = engine.connection.schema_entity_read()
    assert fields == {
        'EventLogEntry': {
            'description': {
                'data_type': {'value': 'text'},
                'name': {'value': 'Description'},
                'properties': {'default_value': {'value': None}},
            },
            'event_type': {
                'data_type': {'value': 'text'},
                'name': {'value': 'Event Type'},
                'properties': {'default_value': {'value': None}},
            },
            'id': {
                'data_type': {'value': 'number'},
                'name': {'value': 'Id'},
                'properties': {'default_value': {'value': None}},
            },
        },
        'Shot': {
            'code': {
                'data_type': {'value': 'text'},
                'name': {'value': 'Shot Code'},
                'properties': {'default_value': {'value': None}},
            }
        },
    }
    assert entities == {
        'EventLogEntry': {'name': {'value': 'Event Log Entry'}},
        'Shot': {'name': {'value': 'Shot'}},
    }
