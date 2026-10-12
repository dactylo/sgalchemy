import asyncio
import copy
import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from _fakes import _FakeAsyncShotgun, _FakeShotgun, _FakeShotgunFactory
from _support import (
    AUTHENTICATION_FAULT,
    STATIC_FORMS,
    isolated_async_engine,
    reset_default_registry_state,
)

from sgalchemy._core import _registry as registry_module
from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._engine import create_engine
from sgalchemy._core._registry import Registry, entity_state
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema._cache import SchemaCache, load_cache, store_cache
from sgalchemy.schema._cachedir import cache_file

_NON_SCHEMA_FAULT_CASES: list[tuple[str, Exception, tuple[Any, ...]]] = [
    ('find', RuntimeError('requested record does not exist'), ('Shot', [], ['id'])),
    ('upload_thumbnail', OSError('file does not exist'), ('Shot', 1, 'missing.jpg')),
]


def test_async_creator_is_public_awaited_and_lazy(tmp_path: Path) -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []

        async def make() -> _FakeAsyncShotgun:
            client = _FakeAsyncShotgun()
            created.append(client)
            return client

        engine = create_async_engine(
            'https://fake.example.com',
            async_creator=cast('Any', make),
            cache_root=tmp_path,
        )
        assert created == []
        await engine.call('schema_read')
        assert len(created) == 1

    asyncio.run(scenario())


def test_async_creator_rejects_sync_client_before_publication() -> None:
    async def scenario() -> None:
        created: list[_FakeShotgun] = []

        async def make() -> _FakeShotgun:
            client = _FakeShotgun()
            created.append(client)
            return client

        engine = create_async_engine(
            'https://fake.example.com',
            async_creator=cast('Any', make),
        )

        for _ in range(2):
            with pytest.raises(TypeError, match='required asynchronous client surface'):
                await engine.call('schema_read')

        assert len(created) == 2
        assert all(client._closed for client in created)
        assert all(client._calls == [] for client in created)
        await engine.dispose()

    asyncio.run(scenario())


def test_async_creator_rejects_sync_result_and_awaits_cleanup() -> None:
    async def scenario() -> None:
        client = _FakeAsyncShotgun()

        def make() -> _FakeAsyncShotgun:
            return client

        engine = create_async_engine('https://fake.example.com', async_creator=cast('Any', make))

        with pytest.raises(TypeError, match='async_creator returned a non-awaitable'):
            await engine.call('schema_read')

        assert client._closed is True
        assert [name for name, _, _ in client._calls].count('close') == 1

    asyncio.run(scenario())


def test_async_default_engine_rejects_different_site_binding(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reset_default_registry_state(monkeypatch)
    first = create_async_engine(
        connection=fake_async_sg,
        cache_root=tmp_path / 'first',
    )
    other = type(fake_async_sg)()
    other.base_url = 'https://other.example.com'
    second = create_async_engine(
        connection=other,
        cache_root=tmp_path / 'second',
    )

    async def scenario() -> None:
        await first.load_schema()
        binding = first.registry.binding_snapshot()
        with pytest.raises(SgalchemyError) as raised:
            await second.load_schema()
        assert str(raised.value) == (
            'registry is already bound to a different site (one site per Registry); '
            'pass registry=Registry() or bind_default=False to the engine factory '
            'for a second site'
        )
        assert first.registry.binding_snapshot() is binding

    asyncio.run(scenario())
    assert second._prepared_schema() is None


def test_async_engine_call_attaches_auth_note_for_direct_users(
    fake_async_sg: _FakeAsyncShotgun,
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(connection=fake_async_sg)
        fault = AUTHENTICATION_FAULT('rejected')
        fake_async_sg._errors.setdefault('find', []).append(fault)

        with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
            await engine.call('find', 'Shot', [])

        assert excinfo.value is fault
        notes = getattr(excinfo.value, '__notes__', [])
        assert any('no automatic retry' in note for note in notes)

    asyncio.run(scenario())


def test_async_engine_call_attaches_auth_note_when_async_creator_rejects() -> None:
    async def scenario() -> None:
        fault = AUTHENTICATION_FAULT('creator rejected')

        async def rejected_creator() -> _FakeAsyncShotgun:
            raise fault

        engine = create_async_engine(
            'https://fake.example.com', async_creator=cast('Any', rejected_creator)
        )

        with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
            await engine.call('find', 'Shot', [])

        assert excinfo.value is fault
        assert any(
            'no automatic retry' in note for note in getattr(excinfo.value, '__notes__', [])
        )

    asyncio.run(scenario())


def test_async_engine_call_does_not_mark_schema_stale_for_async_creator_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fault = RuntimeError('unknown field during creator setup')
        marked = False

        async def rejected_creator() -> _FakeAsyncShotgun:
            raise fault

        def mark_schema_stale() -> None:
            nonlocal marked
            marked = True

        engine = create_async_engine(
            'https://fake.example.com', async_creator=cast('Any', rejected_creator)
        )
        monkeypatch.setattr(engine, 'mark_schema_stale', mark_schema_stale)

        with pytest.raises(RuntimeError) as excinfo:
            await engine.call('find', 'Shot', [])

        assert excinfo.value is fault
        assert not marked

    asyncio.run(scenario())


def test_async_engine_call_marks_schema_stale_for_provider_invocation(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fault = RuntimeError('unknown field')
        marked = False

        def mark_schema_stale() -> None:
            nonlocal marked
            marked = True

        engine = create_async_engine(connection=fake_async_sg)
        monkeypatch.setattr(engine, 'mark_schema_stale', mark_schema_stale)
        fake_async_sg._errors.setdefault('find', []).append(fault)

        with pytest.raises(RuntimeError) as excinfo:
            await engine.call('find', 'Shot', [])

        assert excinfo.value is fault
        assert marked

    asyncio.run(scenario())


def test_async_engine_dispose_resets_and_engine_remains_usable(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(connection=fake_async_sg, cache_root=tmp_path)
        async with engine:
            await engine.call('schema_read')
            await engine.dispose()
            await engine.call('schema_read')
        assert fake_async_sg._closed is False

    asyncio.run(scenario())


def test_async_engine_load_schema_classifies_async_creator_auth_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        fault = AUTHENTICATION_FAULT('creator rejected')

        async def rejected_creator() -> _FakeAsyncShotgun:
            raise fault

        engine = create_async_engine(
            'https://fake.example.com',
            async_creator=cast('Any', rejected_creator),
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )

        with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
            await engine.load_schema()

        assert excinfo.value is fault
        assert any('no automatic retry' in note for note in getattr(fault, '__notes__', []))
        assert all('acquisition' not in name for name in vars(fault))
        assert fault.__context__ is None

    asyncio.run(scenario())


def test_async_engine_registry_is_read_only(fake_async_sg: _FakeAsyncShotgun) -> None:
    engine = create_async_engine(
        connection=fake_async_sg,
        registry=Registry(),
        bind_default=False,
    )

    with pytest.raises(AttributeError):
        object.__setattr__(engine, 'registry', Registry())


def test_async_engine_rejects_cross_loop_reuse(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    engine = create_async_engine(
        connection=fake_async_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    asyncio.run(engine.load_schema())
    with pytest.raises(RuntimeError, match='event loop'):
        asyncio.run(engine.load_schema(force=True))


@pytest.mark.parametrize(
    ('args', 'kwargs', 'message'),
    [
        (
            (),
            {},
            (
                'pass exactly one of a supplied connection, url with credentials, '
                'or url with a creator'
            ),
        ),
        (
            ('ftp://x.example.com',),
            {'script_name': 's', 'api_key': 'k'},
            'url must be a complete http or https site URL',
        ),
        (
            ('https://x.example.com',),
            {'script_name': 's'},
            'script credentials require both script_name and api_key',
        ),
        (
            ('https://x.example.com',),
            {'script_name': 's', 'api_key': 'k', 'connection': object()},
            'a supplied connection cannot be combined with url, credentials, or creators',
        ),
        (
            (),
            {'strategy': 'native', 'sync_connection': object()},
            'native strategy does not accept sync_connection',
        ),
        (
            (),
            {'connection': object(), 'script_name': 'ignored'},
            'a supplied connection cannot be combined with url, credentials, or creators',
        ),
        (
            (),
            {'strategy': 'threaded', 'connection': object()},
            'threaded strategy does not accept connection or async_creator',
        ),
        (
            (),
            {'strategy': 'threaded'},
            (
                'pass exactly one of a supplied connection, url with credentials, '
                'or url with a creator'
            ),
        ),
        (
            (),
            {'strategy': 'threaded', 'sync_connection': object(), 'api_key': 'ignored'},
            'a supplied connection cannot be combined with url, credentials, or creators',
        ),
        (
            (),
            {'strategy': 'invalid'},
            "strategy must be 'native' or 'threaded'",
        ),
    ],
)
def test_async_engine_rejects_invalid_factory_combinations(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_async_engine(*args, **kwargs)


def test_async_engine_shared_creator_failure_never_marks_schema_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fault = RuntimeError('unknown field during creator setup')
        entered = asyncio.Event()
        release = asyncio.Event()
        marked = 0

        async def rejected_creator() -> _FakeAsyncShotgun:
            entered.set()
            await release.wait()
            raise fault

        def mark_schema_stale() -> None:
            nonlocal marked
            marked += 1

        async def capture_failure(engine: AsyncEngine) -> Exception:
            try:
                await engine.call('find', 'Shot', [])
            except RuntimeError as err:
                return err
            raise AssertionError('provider call unexpectedly succeeded')

        engine = create_async_engine(
            'https://fake.example.com', async_creator=cast('Any', rejected_creator)
        )
        monkeypatch.setattr(engine, 'mark_schema_stale', mark_schema_stale)
        first = asyncio.create_task(capture_failure(engine))
        await entered.wait()
        second = asyncio.create_task(capture_failure(engine))
        await asyncio.sleep(0)
        release.set()

        failures = await asyncio.gather(first, second)
        assert failures[0] is fault
        assert failures[1] is fault
        assert marked == 0

    asyncio.run(scenario())


def test_async_engine_valid_live_schema_publishes_and_updates_cache(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    registry = Registry()

    async def scenario() -> None:
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path,
            registry=registry,
            bind_default=False,
        )

        schemas = await engine.load_schema(force=True)

        assert 'Shot' in schemas
        assert registry.binding_snapshot() is not None

    asyncio.run(scenario())
    cached = load_cache(cache_file(fake_async_sg.base_url, root=tmp_path))
    assert cached is not None
    assert 'Shot' in cached.fields


def test_async_factory_accepts_http_for_caller_owned_sources(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
) -> None:
    fake_async_sg.base_url = 'http://caller.example.com'
    fake_sg.base_url = 'http://caller.example.com'
    supplied_native = create_async_engine(connection=fake_async_sg)
    supplied_threaded = create_async_engine(strategy='threaded', sync_connection=fake_sg)
    created_async = _FakeAsyncShotgun()
    created_async.base_url = 'http://caller.example.com'
    created_sync = _FakeShotgun()
    created_sync.base_url = 'http://caller.example.com'
    async_creator_calls = 0
    creator_calls = 0

    async def async_creator() -> _FakeAsyncShotgun:
        nonlocal async_creator_calls
        async_creator_calls += 1
        return created_async

    def creator() -> _FakeShotgun:
        nonlocal creator_calls
        creator_calls += 1
        return created_sync

    owned_native = create_async_engine(
        'http://caller.example.com', async_creator=cast('Any', async_creator)
    )
    owned_threaded = create_async_engine(
        'http://caller.example.com',
        strategy='threaded',
        creator=cast('Any', creator),
    )

    assert supplied_native.site == 'http://caller.example.com'
    assert supplied_threaded.site == 'http://caller.example.com'
    assert async_creator_calls == 0
    assert creator_calls == 0

    async def scenario() -> None:
        await owned_native.call('schema_read')
        await owned_threaded.call('schema_read')
        await owned_native.dispose()
        await owned_threaded.dispose()
        await supplied_native.dispose()
        await supplied_threaded.dispose()

    asyncio.run(scenario())
    assert async_creator_calls == 1
    assert creator_calls == 1
    assert fake_async_sg._closed is False
    assert fake_sg._closed is False


def test_async_factory_canonicalizes_supplied_connection_sites(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
) -> None:
    fake_async_sg.base_url = 'HTTPS://FAKE.EXAMPLE.COM/'
    fake_sg.base_url = 'HTTPS://FAKE.EXAMPLE.COM/'

    native = create_async_engine(connection=fake_async_sg)
    threaded = create_async_engine(strategy='threaded', sync_connection=fake_sg)

    assert native.site == 'https://fake.example.com'
    assert threaded.site == 'https://fake.example.com'


def test_async_find_preserves_compiler_call_shape(fake_async_sg: _FakeAsyncShotgun) -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            connection=fake_async_sg,
            registry=Registry(),
            bind_default=False,
        )
        order = [{'field_name': 'id', 'direction': 'asc'}]
        call = FindCall(
            entity_type='Shot',
            filters=[['code', 'is', 'AB']],
            fields=['id', 'code'],
            order=order,
            limit=5,
            page=2,
            extra={'retired_only': True},
        )

        await engine.execute_call(call)

        assert fake_async_sg._calls[-1] == (
            'find',
            ('Shot', [['code', 'is', 'AB']], ['id', 'code']),
            {
                'order': order,
                'limit': 5,
                'page': 2,
                'retired_only': True,
            },
        )

    asyncio.run(scenario())


def test_async_schema_is_explicit_and_cached(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        process_defaults = list(registry_module._DEFAULT_REGISTRIES)
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        other = isolated_async_engine(fake_async_sg, tmp_path)
        assert engine.registry is not other.registry
        assert process_defaults == registry_module._DEFAULT_REGISTRIES
        with pytest.raises(SgalchemyError, match='load_schema'):
            _ = engine.schema
        await engine.load_schema()
        assert 'Shot' in engine.schema
        binding = await engine.load_binding()
        assert binding is engine.prepared_binding()
        assert binding.schemas is engine.schema
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('credentials', STATIC_FORMS)
@pytest.mark.parametrize('strategy', ['threaded', 'native'])
def test_async_static_credentials_require_https(
    credentials: dict[str, str],
    strategy: Literal['native', 'threaded'],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module
    provider_imported = False

    def import_module(name: str, package: str | None = None) -> ModuleType:
        nonlocal provider_imported
        if name == 'shotgun_api3':
            provider_imported = True
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', import_module)

    with pytest.raises(
        ValueError, match='static credentials require an https site URL'
    ) as excinfo:
        create_async_engine(
            'http://plaintext.example.com',
            strategy=strategy,
            script_name=credentials.get('script_name'),
            api_key=credentials.get('api_key'),
            session_token=credentials.get('session_token'),
        )

    message = str(excinfo.value)
    assert provider_imported is False
    for sensitive in ('plaintext.example.com', 'script', 'secret-key', 'secret-token'):
        assert sensitive not in message

    secure = create_async_engine(
        'https://secure.example.com',
        strategy=strategy,
        script_name=credentials.get('script_name'),
        api_key=credentials.get('api_key'),
        session_token=credentials.get('session_token'),
    )
    assert secure.site == 'https://secure.example.com'
    assert provider_imported is False


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        (
            {'strategy': 'native', 'session_token': 'tok', 'max_workers': 2},
            'native strategy does not accept max_workers',
        ),
        (
            {'strategy': 'native', 'session_token': 'tok', 'creator': _FakeShotgun},
            'native strategy does not accept sync_connection or creator',
        ),
        (
            {
                'strategy': 'threaded',
                'session_token': 'tok',
                'async_creator': _FakeShotgun,
            },
            'threaded strategy does not accept connection or async_creator',
        ),
        (
            {'strategy': 'threaded', 'session_token': 'tok', 'max_workers': 0},
            'max_workers must be at least 1',
        ),
        ({'session_token': ''}, 'session_token must not be empty'),
        (
            {'session_token': 'tok', 'script_name': 'a', 'api_key': 'b'},
            'pass either script_name with api_key or session_token, not both',
        ),
    ],
)
def test_async_factory_rejects_inconsistent_options(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_async_engine('https://x.example.com', **kwargs)


@pytest.mark.parametrize(
    ('method', 'error', 'args'),
    _NON_SCHEMA_FAULT_CASES,
)
def test_async_non_schema_fault_does_not_mark_schema_stale(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    method: str,
    error: Exception,
    args: tuple[Any, ...],
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        fake_async_sg._errors[method] = [error]
        with pytest.raises(type(error), match='does not exist'):
            await engine.call(method, *args)
        await engine.load_schema()
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'malformed_shape',
    [
        pytest.param('field_metadata', id='field-metadata'),
        pytest.param('entity_metadata', id='entity-metadata'),
        pytest.param('valid_types_value', id='valid-types-value'),
    ],
)
def test_async_engine_rejects_malformed_live_schema_without_publication_or_cache_update(
    fake_async_sg: _FakeAsyncShotgun,
    malformed_shape: str,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    path = cache_file(fake_async_sg.base_url, root=tmp_path)
    existing = SchemaCache(
        fields=copy.deepcopy(schema_fixture),
        entities={'Shot': {'name': {'value': 'Shot'}}},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    store_cache(path, existing)
    original_bytes = path.read_bytes()
    live_fields = copy.deepcopy(schema_fixture)
    if malformed_shape == 'field_metadata':
        live_fields['Shot']['code'] = ['schema-secret']
    elif malformed_shape == 'valid_types_value':
        live_fields['Shot']['project']['properties']['valid_types']['value'] = 'schema-secret'

    async def schema_read() -> dict[str, Any]:
        await fake_async_sg._record('schema_read')
        return live_fields

    monkeypatch.setattr(fake_async_sg, 'schema_read', schema_read)
    if malformed_shape == 'entity_metadata':

        async def schema_entity_read() -> dict[str, Any]:
            await fake_async_sg._record('schema_entity_read')
            return {'Shot': ['schema-secret']}

        monkeypatch.setattr(fake_async_sg, 'schema_entity_read', schema_entity_read)
    registry = Registry()

    async def scenario() -> None:
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path,
            registry=registry,
            bind_default=False,
        )
        with pytest.raises(
            SgalchemyError,
            match='provider returned malformed schema; cache was not updated',
        ) as raised:
            await engine.load_schema(force=True)
        assert registry.binding_snapshot() is None
        assert engine._prepared_schema() is None
        assert path.read_bytes() == original_bytes
        assert 'schema-secret' not in str(raised.value)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('info_value', 'expected'),
    [
        pytest.param({'version': [8, 64, 0]}, {'version': [8, 64, 0]}, id='dictionary'),
        pytest.param(['8.64.0'], None, id='non-dictionary'),
    ],
)
def test_async_schema_refresh_preserves_only_dictionary_server_info(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    info_value: object,
    expected: dict[str, Any] | None,
) -> None:
    async def info() -> object:
        await fake_async_sg._record('info')
        return info_value

    monkeypatch.setattr(fake_async_sg, 'info', info, raising=False)

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()

    asyncio.run(scenario())
    cached = load_cache(cache_file(fake_async_sg.base_url, root=tmp_path))
    assert cached is not None
    assert cached.server_info == expected


def test_async_summary_preserves_grouping_presence(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        summarize = AsyncMock(wraps=fake_async_sg.summarize)
        monkeypatch.setattr(fake_async_sg, 'summarize', summarize)
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        ungrouped = SummarizeCall(
            entity_type='Shot',
            filters=[],
            summary_fields=[{'field': 'id', 'type': 'count'}],
        )
        await engine.execute_call(ungrouped)
        assert 'grouping' not in summarize.call_args.kwargs

        grouping = [{'field': 'sg_status_list', 'type': 'exact', 'direction': 'asc'}]
        grouped = SummarizeCall(
            entity_type='Shot',
            filters=[],
            summary_fields=[{'field': 'id', 'type': 'count'}],
            grouping=grouping,
        )
        await engine.execute_call(grouped)
        assert summarize.call_args.kwargs['grouping'] == grouping

    asyncio.run(scenario())


def test_native_credentials_are_lazy_and_missing_extra_is_taught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, 'shotgun_api3', raising=False)
    real_import_module = importlib.import_module
    imported = False

    def import_module(name: str, package: str | None = None) -> ModuleType:
        nonlocal imported
        if name == 'shotgun_api3':
            imported = True
            return ModuleType('shotgun_api3')
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', import_module)
    engine = create_async_engine(
        'https://x.example.com',
        script_name='s',
        api_key='k',
        registry=Registry(),
        bind_default=False,
    )
    assert imported is False
    assert engine._prepared_schema() is None
    with pytest.raises(SgalchemyError, match='load_schema'):
        _ = engine.entities

    async def scenario() -> None:
        with pytest.raises(ImportError, match=r'sg-python-api\[async\]'):
            await engine.call('schema_read')

    asyncio.run(scenario())
    assert imported is True


def test_native_static_session_token_builds_async_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        recorded: dict[str, Any] = {}

        class AsyncProvider(_FakeAsyncShotgun):
            def __init__(self, base_url: str, **kwargs: object) -> None:
                super().__init__()
                recorded['base_url'] = base_url
                recorded.update(kwargs)
                self.base_url = base_url

        provider = ModuleType('shotgun_api3')
        provider.__dict__['AsyncShotgun'] = AsyncProvider
        monkeypatch.setitem(sys.modules, 'shotgun_api3', provider)
        engine = create_async_engine(
            'https://x.example.com',
            session_token=str(123),
            cache_root=tmp_path,
        )
        await engine.call('schema_read')
        assert recorded == {
            'base_url': 'https://x.example.com',
            'session_token': '123',
        }

    asyncio.run(scenario())


def test_owned_native_connection_is_lazy_and_closed_by_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class RecordingAsyncShotgun(_FakeAsyncShotgun):
        def __init__(
            self,
            base_url: str,
            *,
            script_name: str | None,
            api_key: str | None,
        ) -> None:
            super().__init__()
            self.base_url = base_url
            self._credentials = script_name, api_key
            instances.append(self)

    provider = ModuleType('shotgun_api3')
    provider.__dict__['AsyncShotgun'] = RecordingAsyncShotgun
    monkeypatch.setitem(sys.modules, 'shotgun_api3', provider)
    engine = create_async_engine(
        'https://x.example.com',
        script_name='s',
        api_key='k',
        registry=Registry(),
        bind_default=False,
    )
    assert instances == []

    async def scenario() -> None:
        async with engine as entered:
            assert entered is engine
            await engine.call('schema_read')

    asyncio.run(scenario())
    assert len(instances) == 1
    assert instances[0]._credentials == ('s', 'k')
    assert instances[0]._closed is True


def test_stale_mark_during_awaited_load_is_not_lost(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        entered, release = fake_async_sg._block('schema_read')
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        loading = asyncio.create_task(engine.load_schema())
        await entered.wait()
        engine.mark_schema_stale()
        release.set()
        await loading
        await engine.load_schema()
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2

    asyncio.run(scenario())


def test_supplied_sync_connection_defaults_to_one_worker_and_opts_into_more(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine = create_async_engine(strategy='threaded', sync_connection=fake_sg, cache_root=tmp_path)
    assert cast('Any', engine)._backend._max_workers == 1
    wide = create_async_engine(
        strategy='threaded',
        sync_connection=fake_sg,
        max_workers=3,
        cache_root=tmp_path,
    )
    assert cast('Any', wide)._backend._max_workers == 3


def test_sync_and_async_default_engines_share_live_schema_binding(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    reset_default_registry_state(monkeypatch)
    sync_engine = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'sync',
    )
    async_engine = create_async_engine(
        connection=fake_async_sg,
        cache_root=tmp_path / 'async',
    )
    assert sync_engine.registry is async_engine.registry
    sync_engine.load_schema()
    shot_cls = sync_engine.entities.Shot

    async_schema = copy.deepcopy(schema_fixture)
    async_schema['Shot']['description']['data_type']['value'] = 'image'

    async def schema_read() -> dict[str, Any]:
        await fake_async_sg._record('schema_read')
        return copy.deepcopy(async_schema)

    monkeypatch.setattr(fake_async_sg, 'schema_read', schema_read)

    async def scenario() -> None:
        refreshed = await async_engine.load_schema(force=True)
        assert refreshed is sync_engine.registry.schema_snapshot()
        assert sync_engine.schema is refreshed
        assert sync_engine._prepared_schema() is refreshed
        assert async_engine.schema is refreshed
        assert async_engine._prepared_schema() is refreshed
        assert async_engine.entities.Shot is shot_cls
        assert refreshed['Shot'].fields['description'].data_type == 'image'

    asyncio.run(scenario())

    sync_schema = copy.deepcopy(schema_fixture)
    sync_schema['Shot']['description']['data_type']['value'] = 'date'
    monkeypatch.setattr(fake_sg, 'schema_read', lambda: copy.deepcopy(sync_schema))
    refreshed = sync_engine.load_schema(force=True)
    assert async_engine.schema is refreshed
    assert async_engine._prepared_schema() is refreshed
    assert async_engine.entities.Shot is shot_cls
    assert entity_state(shot_cls.new(description='2026-07-25')).dirty_types == {
        'description': 'date'
    }


def test_threaded_creator_is_public_and_lazy(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = _FakeShotgunFactory()
        engine = create_async_engine(
            'https://fake.example.com',
            strategy='threaded',
            creator=cast('Any', factory),
            cache_root=tmp_path,
        )
        assert factory._instances == []
        await engine.call('schema_read')
        assert len(factory._instances) == 1

    asyncio.run(scenario())


def test_threaded_factory_uses_supplied_sync_connection(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            sync_connection=fake_sg,
            strategy='threaded',
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )

        await engine.load_schema()
        assert 'Shot' in engine.schema
        await engine.dispose()

    asyncio.run(scenario())
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 1


def test_threaded_owned_client_defaults_to_four_workers(tmp_path: Path) -> None:
    engine = create_async_engine(
        'https://fake.example.com',
        strategy='threaded',
        creator=cast('Any', _FakeShotgunFactory()),
        cache_root=tmp_path,
    )

    assert cast('Any', engine)._backend._max_workers == 4
