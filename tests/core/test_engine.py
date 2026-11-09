import copy
import gc
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import ModuleType
from typing import Any, Never, cast
from unittest.mock import MagicMock, Mock

import pytest

from sgalchemy._core import _engine
from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._engine import Engine, _ThreadSlot, create_engine
from sgalchemy._core._engine_common import AUTH_RECOVERY_NOTE, CallProgress
from sgalchemy._core._registry import Registry, default_registry, entity_type
from sgalchemy._errors import SgalchemyError
from sgalchemy._shotgun import AsyncClosableShotgunLike, ClosableShotgunLike
from sgalchemy.schema._cache import SchemaCache, load_cache, store_cache
from sgalchemy.schema._cachedir import cache_file
from sgalchemy.schema._policy import TTLPolicy
from tests._fakes import FakeAsyncShotgun, FakeShotgun, FakeShotgunFactory
from tests._support import AUTHENTICATION_FAULT, STATIC_FORMS, break_provider_import

_NOT_CACHED = (
    '; using the fetched schema without caching it. Set SGALCHEMY_SCHEMA_CACHE or pass '
    'cache_root= to use a writable directory'
)
_PROVIDER_AUTH_ERROR: type[Exception] = type('ProviderAuthError', (AUTHENTICATION_FAULT,), {})
_SCHEMA_FAULT_CASES: list[tuple[str, Exception, tuple[Any, ...], bool]] = [
    ('find', RuntimeError('unknown field'), ('Shot', [], ['id']), True),
    ('find', RuntimeError('entity type Ghost is invalid'), ('Shot', [], ['id']), True),
    ('summarize', RuntimeError('field Shot.nope does not exist'), ('Shot', [], []), True),
    ('find', RuntimeError('requested record does not exist'), ('Shot', [], ['id']), False),
    ('find', OSError('field Shot.nope does not exist'), ('Shot', [], ['id']), False),
    ('upload_thumbnail', RuntimeError('unknown field'), ('Shot', 1, 'x.jpg'), False),
]


class _CountingShotgun(FakeShotgun):
    def __init__(self) -> None:
        super().__init__()
        self._close_calls = 0
        self._close_lock = Lock()

    def close(self) -> None:
        with self._close_lock:
            self._close_calls += 1
            close_call = self._close_calls
        if close_call == 1:
            raise RuntimeError('close failed')
        self.closed = True


class _InfoShotgun(FakeShotgun):
    def __init__(self, server_info: object) -> None:
        super().__init__()
        self._server_info = server_info
        self.info_calls = 0

    def info(self) -> object:
        self.info_calls += 1
        return self._server_info


class _SequencedLock:
    def __init__(
        self,
        republish_waiting: Event,
        release_republish: Event,
        retry_scanned: Event,
    ) -> None:
        self._entries: dict[int, int] = {}
        self._lock = Lock()
        self._release_republish = release_republish
        self._republish_waiting = republish_waiting
        self._retry_scanned = retry_scanned

    def __enter__(self) -> None:
        thread_id = get_ident()
        entry = self._entries.get(thread_id, 0) + 1
        self._entries[thread_id] = entry
        if threading.current_thread().name == 'failing-dispose' and entry == 2:
            self._republish_waiting.set()
            assert self._release_republish.wait(timeout=5)
        self._lock.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self._lock.release()
        if threading.current_thread().name == 'retry-dispose':
            thread_id = get_ident()
            if self._entries[thread_id] == 1:
                self._retry_scanned.set()


def _blocked_root(tmp_path: Path) -> Path:
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('')
    return blocker


def test_auth_fault_while_loading_schema_carries_one_recovery_note(tmp_path: Path) -> None:
    def creator() -> ClosableShotgunLike:
        raise AUTHENTICATION_FAULT('expired')

    engine = create_engine(
        'https://fake.example.com', creator=creator, cache_root=tmp_path, bind_default=False
    )
    with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
        engine.load_schema()
    assert getattr(excinfo.value, '__notes__', []).count(AUTH_RECOVERY_NOTE) == 1


def test_cache_serialization_errors_still_propagate(tmp_path: Path) -> None:
    connection = _InfoShotgun({'opaque': object()})
    with pytest.raises(TypeError, match='is not JSON serializable'):
        create_engine(connection=connection, cache_root=tmp_path, bind_default=False).load_schema()


def test_call_progress_excludes_echo_failure(
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(connection=fake_sg, echo=True, bind_default=False)
    observation = CallProgress()
    fault = RuntimeError('logging failed')

    def fail_logging(*_args: object, **_kwargs: object) -> Never:
        raise fault

    monkeypatch.setattr(_engine._LOGGER, 'info', fail_logging)
    with pytest.raises(RuntimeError) as excinfo:
        engine.call_with_progress('upload_thumbnail', ('Shot', 90, 'image.jpg'), {}, observation)
    assert excinfo.value is fault
    assert not observation.invoked.is_set()
    assert fake_sg.calls == []


def test_call_progress_marks_only_resolved_provider_entry(
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(connection=fake_sg, bind_default=False)
    progress = CallProgress()
    with pytest.raises(AttributeError):
        engine.call_with_progress('missing_method', (), {}, progress)
    assert not progress.invoked.is_set()
    assert not progress.dispatched.is_set()

    def provider(*, progress: str) -> str:
        assert observation.invoked.is_set()
        return progress

    observation = CallProgress()
    monkeypatch.setattr(fake_sg, 'schema_read', provider)
    assert (
        engine.call_with_progress('schema_read', (), {'progress': 'wire'}, observation) == 'wire'
    )
    assert not observation.dispatched.is_set()
    assert not progress.invoked.is_set()


def test_concurrent_close_retry_does_not_retain_closed_slot() -> None:
    republish_waiting = Event()
    release_republish = Event()
    retry_scanned = Event()
    client = _CountingShotgun()

    def creator() -> ClosableShotgunLike:
        return client

    engine = create_engine('https://fake.example.com', creator=creator)
    _ = engine.connection
    vars(engine)['_disposal_lock'] = _SequencedLock(
        republish_waiting, release_republish, retry_scanned
    )
    failures: list[RuntimeError] = []

    def failing_dispose() -> None:
        try:
            engine.dispose()
        except RuntimeError as err:
            failures.append(err)

    failing = threading.Thread(target=failing_dispose, name='failing-dispose')
    retrying = threading.Thread(target=engine.dispose, name='retry-dispose')
    failing.start()
    assert republish_waiting.wait(timeout=5)
    retrying.start()
    assert retry_scanned.wait(timeout=5)
    release_republish.set()
    failing.join(timeout=5)
    retrying.join(timeout=5)

    assert not failing.is_alive()
    assert not retrying.is_alive()
    assert len(failures) == 1
    assert str(failures[0]) == 'close failed'
    assert client.closed is True
    assert client._close_calls == 2
    engine.dispose()
    assert vars(engine)['_retired_slots'] == set()
    engine.dispose()
    assert client._close_calls == 2


def test_connection_property_attaches_auth_note_when_creator_rejects() -> None:
    fault = AUTHENTICATION_FAULT('creator rejected')

    def rejected_creator() -> ClosableShotgunLike:
        raise fault

    engine = create_engine('https://fake.example.com', creator=rejected_creator)

    with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
        _ = engine.connection

    assert excinfo.value is fault
    assert getattr(excinfo.value, '__notes__', []).count(AUTH_RECOVERY_NOTE) == 1


def test_connection_property_returns_fresh_client_after_dispose() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)
    first = engine.connection
    engine.dispose()
    second = engine.connection
    assert second is not first
    assert factory.instances[0].closed is True
    assert factory.instances[1].closed is False


def test_construction_and_properties_are_lazy(
    fake_sg: FakeShotgun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, 'shotgun_api3', raising=False)
    created = create_engine(
        'https://x.example.com', script_name='s', api_key='k', cache_root=tmp_path
    )
    assert 'shotgun_api3' not in sys.modules
    assert created._prepared_schema() is None
    with pytest.raises(SgalchemyError, match='load_schema'):
        _ = created.schema
    assert 'shotgun_api3' not in sys.modules
    assert fake_sg.calls == []


def test_create_engine_accepts_creator_with_url() -> None:
    fake = FakeShotgun()
    engine = create_engine('https://fake.example.com', creator=lambda: fake)
    assert engine.connection is fake


def test_create_engine_accepts_http_for_caller_owned_sources(
    fake_sg: FakeShotgun,
) -> None:
    fake_sg.base_url = 'http://caller.example.com'
    supplied = create_engine(connection=fake_sg)
    created = FakeShotgun()
    created.base_url = 'http://caller.example.com'
    creator_calls = 0

    def creator() -> ClosableShotgunLike:
        nonlocal creator_calls
        creator_calls += 1
        return created

    owned = create_engine('http://caller.example.com', creator=creator)

    assert supplied.connection is fake_sg
    assert creator_calls == 0
    assert owned.connection is created
    assert creator_calls == 1
    owned.dispose()


def test_create_engine_accepts_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}
    value = 'tok'

    class Provider:
        def __init__(self, base_url: str, **kwargs: object) -> None:
            recorded['base_url'] = base_url
            recorded.update(kwargs)
            self.base_url = base_url

        def schema_read(self) -> dict[str, Any]:
            return {}

    provider = ModuleType('shotgun_api3')
    provider.__dict__['Shotgun'] = Provider
    monkeypatch.setitem(sys.modules, 'shotgun_api3', provider)
    engine = create_engine('https://x.example.com', session_token=value)
    _ = engine.connection
    assert recorded == {'base_url': 'https://x.example.com', 'session_token': 'tok'}


def test_create_engine_rejects_an_async_connection(fake_async_sg: FakeAsyncShotgun) -> None:
    with pytest.raises(TypeError) as excinfo:
        create_engine(connection=fake_async_sg)  # pyright: ignore[reportArgumentType]
    assert str(excinfo.value) == (
        'connection= received an async client; pass it to create_async_engine(connection=...) '
        'instead'
    )


def test_create_engine_rejects_connection_with_creator() -> None:
    connection: Any = FakeShotgun()
    creator: Any = FakeShotgun
    with pytest.raises(ValueError, match='combined'):
        create_engine(connection=connection, creator=creator)


def test_create_engine_rejects_creator_without_url_or_with_credentials() -> None:
    creator: Any = FakeShotgun
    value = 'tok'
    with pytest.raises(ValueError, match='exactly one'):
        create_engine(creator=creator)
    with pytest.raises(ValueError, match='exactly one'):
        create_engine('https://x.example.com', creator=creator, session_token=value)


def test_create_engine_rejects_userinfo_url_without_echoing_it() -> None:
    value = 'tok'
    with pytest.raises(ValueError, match='userinfo') as excinfo:
        create_engine('https://user:sec@x.example.com', session_token=value)
    assert 'sec' not in str(excinfo.value)


@pytest.mark.parametrize('credentials', STATIC_FORMS)
def test_create_engine_static_credentials_require_https(
    credentials: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, 'shotgun_api3', raising=False)

    with pytest.raises(
        ValueError, match='static credentials require an https site URL'
    ) as excinfo:
        create_engine(
            'http://plaintext.example.com',
            script_name=credentials.get('script_name'),
            api_key=credentials.get('api_key'),
            session_token=credentials.get('session_token'),
        )

    message = str(excinfo.value)
    assert 'shotgun_api3' not in sys.modules
    for sensitive in ('plaintext.example.com', 'script', 'secret-key', 'secret-token'):
        assert sensitive not in message

    secure = create_engine(
        'https://secure.example.com',
        script_name=credentials.get('script_name'),
        api_key=credentials.get('api_key'),
        session_token=credentials.get('session_token'),
    )
    assert secure.site == 'https://secure.example.com'
    assert 'shotgun_api3' not in sys.modules


@pytest.mark.parametrize(
    'kwargs',
    [
        {'session_token': 'tok', 'script_name': 'a', 'api_key': 'b'},
        {'session_token': ''},
    ],
)
def test_create_engine_rejects_invalid_url_credentials(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r'script|session_token|credential'):
        create_engine('https://x.example.com', **kwargs)


def test_creator_rejects_async_client_before_publication(
    recwarn: pytest.WarningsRecorder,
) -> None:
    created: list[FakeAsyncShotgun] = []

    def make() -> AsyncClosableShotgunLike:
        client = FakeAsyncShotgun()
        created.append(client)
        return client

    engine = create_engine('https://fake.example.com', creator=cast('Any', make))

    for _ in range(2):
        with pytest.raises(TypeError, match='required synchronous client surface'):
            _ = engine.connection

    gc.collect()
    assert len(created) == 2
    assert all(client.closed is False for client in created)
    assert all(client.calls == [] for client in created)
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    engine.dispose()


def test_credential_connections_are_owned_and_reused_per_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = Barrier(2)
    instances: list[Any] = []

    class RecordingShotgun:
        def __init__(self, base_url: str, *, script_name: str | None, api_key: str | None) -> None:
            self.base_url = base_url
            self.credentials = script_name, api_key
            self.thread_ids: list[int] = []
            instances.append(self)

        def schema_entity_read(self) -> dict[str, Any]:
            self.thread_ids.append(get_ident())
            return {}

        def schema_read(self) -> dict[str, Any]:
            self.thread_ids.append(get_ident())
            barrier.wait(timeout=2)
            return {}

    provider = ModuleType('shotgun_api3')
    provider.__dict__['Shotgun'] = RecordingShotgun
    monkeypatch.setitem(sys.modules, 'shotgun_api3', provider)
    created = create_engine('https://x.example.com', script_name='s', api_key='k')

    def call_twice() -> None:
        created.call('schema_read')
        created.call('schema_entity_read')

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(call_twice) for _ in range(2)]
        for future in futures:
            future.result(timeout=2)
    assert len(instances) == 2
    assert all(instance.credentials == ('s', 'k') for instance in instances)
    assert all(len(instance.thread_ids) == 2 for instance in instances)
    assert all(len(set(instance.thread_ids)) == 1 for instance in instances)


def test_dead_thread_slot_finalizer_closes_owned_client() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)

    def worker() -> None:
        engine.call('schema_read')

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    client = factory.instances[0]
    del thread
    gc.collect()
    assert client.closed is True


def test_dead_thread_slot_finalizer_failure_is_retained_for_dispose_retry() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)

    def worker() -> None:
        engine.call('schema_read')
        factory.instances[0].close_errors.append(RuntimeError('finalizer close failed'))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    client = factory.instances[0]
    del thread
    gc.collect()
    assert client.close_errors == []
    assert client.closed is False
    engine.dispose()
    assert client.closed is True


def test_dispose_close_failure_propagates_and_retry_closes_the_rest() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)
    engine.call('schema_read')
    failing = factory.instances[0]
    failing.close_errors.append(RuntimeError('close failed'))
    with pytest.raises(RuntimeError, match='close failed'):
        engine.dispose()
    engine.dispose()
    assert failing.closed is True


def test_dispose_closes_idle_owned_clients_and_engine_stays_usable() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)
    engine.call('schema_read')
    first = factory.instances[0]
    engine.dispose()
    assert first.closed is True
    engine.call('schema_read')
    assert len(factory.instances) == 2
    assert factory.instances[1].closed is False


def test_dispose_defers_close_of_active_client_without_altering_outcome() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)
    entered = threading.Event()
    release = threading.Event()
    result: list[Any] = []

    def worker() -> None:
        engine.call('schema_read')
        worker_client = factory.instances[0]
        worker_client.block_next('find', entered=entered, release=release)
        result.append(engine.call('find', 'Shot', []))

    thread = threading.Thread(target=worker)
    try:
        thread.start()
        assert entered.wait(timeout=5)
        client = factory.instances[0]
        engine.dispose()
        assert client.closed is False
        client.close_errors.append(RuntimeError('deferred close failed'))
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(timeout=5)
        engine.dispose()
    assert not thread.is_alive()
    assert result == [[]]
    assert client.closed is True


def test_dispose_is_idempotent_and_never_closes_supplied_connections(
    fake_sg: FakeShotgun,
) -> None:
    engine = create_engine(connection=fake_sg)
    engine.dispose()
    engine.dispose()
    assert fake_sg.closed is False


def test_engine_call_attaches_auth_note_for_direct_users(fake_sg: FakeShotgun) -> None:
    engine = create_engine(connection=fake_sg)
    fault = _PROVIDER_AUTH_ERROR('rejected')
    fake_sg.errors.setdefault('find', []).append(fault)

    with pytest.raises(_PROVIDER_AUTH_ERROR) as excinfo:
        engine.call('find', 'Shot', [])

    assert excinfo.value is fault
    notes = getattr(excinfo.value, '__notes__', [])
    assert any('no automatic retry' in note for note in notes)


def test_engine_call_attaches_auth_note_when_creator_rejects() -> None:
    fault = AUTHENTICATION_FAULT('creator rejected')

    def rejected_creator() -> ClosableShotgunLike:
        raise fault

    engine = create_engine('https://fake.example.com', creator=rejected_creator)

    with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
        engine.call('find', 'Shot', [])

    assert excinfo.value is fault
    assert getattr(excinfo.value, '__notes__', []).count(AUTH_RECOVERY_NOTE) == 1


def test_engine_call_does_not_mark_schema_stale_when_creator_setup_looks_schema_like(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fault = RuntimeError('unknown field during creator setup')
    marked = False

    def rejected_creator() -> ClosableShotgunLike:
        raise fault

    def mark_schema_stale() -> None:
        nonlocal marked
        marked = True

    engine = create_engine('https://fake.example.com', creator=rejected_creator)
    monkeypatch.setattr(engine, 'mark_schema_stale', mark_schema_stale)

    with pytest.raises(RuntimeError) as excinfo:
        engine.call('find', 'Shot', [])

    assert excinfo.value is fault
    assert not marked


def test_engine_registry_is_read_only(fake_sg: FakeShotgun) -> None:
    created = create_engine(
        connection=fake_sg,
        registry=Registry(),
        bind_default=False,
    )

    with pytest.raises(AttributeError):
        object.__setattr__(created, 'registry', Registry())


@pytest.mark.parametrize(
    ('args', 'kwargs'),
    [
        ((), {}),
        (('https://x.example.com',), {'connection': object()}),
        (('https://x.example.com',), {}),
        (('https://x.example.com',), {'script_name': 's'}),
        (('https://x.example.com',), {'api_key': 'k'}),
        ((), {'connection': object(), 'script_name': 's'}),
        ((), {'connection': object(), 'api_key': 'k'}),
    ],
)
def test_engine_rejects_invalid_connection_and_credential_combinations(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match=r'.+'):
        create_engine(*args, **kwargs)


@pytest.mark.parametrize(
    'malformed_shape',
    [
        pytest.param('field_metadata', id='field-metadata'),
        pytest.param('entity_metadata', id='entity-metadata'),
        pytest.param('valid_types_value', id='valid-types-value'),
    ],
)
def test_engine_rejects_malformed_live_schema_without_publication_or_cache_update(
    fake_sg: FakeShotgun,
    malformed_shape: str,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    path = cache_file(fake_sg.base_url, root=tmp_path)
    existing = SchemaCache(
        fields=copy.deepcopy(schema_fixture),
        entities=fake_sg.schema_entity_read(),
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    store_cache(path, existing)
    original_bytes = path.read_bytes()
    live_fields = copy.deepcopy(schema_fixture)
    if malformed_shape == 'field_metadata':
        live_fields['Shot']['code'] = ['schema-secret']
    elif malformed_shape == 'valid_types_value':
        live_fields['Shot']['project']['properties']['valid_types']['value'] = 'schema-secret'
    monkeypatch.setattr(fake_sg, 'schema_read', lambda: live_fields)
    if malformed_shape == 'entity_metadata':
        monkeypatch.setattr(fake_sg, 'schema_entity_read', lambda: {'Shot': ['schema-secret']})
    registry = Registry()
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=registry,
        bind_default=False,
    )

    with pytest.raises(
        SgalchemyError,
        match='provider returned malformed schema; cache was not updated',
    ) as raised:
        created.load_schema(force=True)
    assert registry.binding_snapshot() is None
    assert created._prepared_schema() is None
    assert path.read_bytes() == original_bytes
    assert 'schema-secret' not in str(raised.value)


def test_engine_schema_reads_through_shared_registry_refresh(
    fake_sg: FakeShotgun,
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
    replacement_schema = copy.deepcopy(schema_fixture)
    replacement_schema['Shot']['description']['data_type']['value'] = 'image'
    second_sg = type(fake_sg)()
    monkeypatch.setattr(
        second_sg,
        'schema_read',
        lambda: copy.deepcopy(replacement_schema),
    )
    second = create_engine(
        connection=second_sg,
        cache_root=tmp_path / 'second',
        registry=registry,
        bind_default=False,
    )

    refreshed = second.load_schema()

    assert refreshed is registry.schema_snapshot()
    assert first.schema is refreshed
    assert first._prepared_schema() is refreshed
    assert first.load_schema() is refreshed
    assert first.schema['Shot'].fields['description'].data_type == 'image'


def test_engine_schema_refresh_normalizes_non_dictionary_server_info(
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def info() -> list[str]:
        fake_sg.record('info')
        return ['8.64.0']

    monkeypatch.setattr(fake_sg, 'info', info, raising=False)
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )

    created.load_schema()

    cached = load_cache(cache_file(fake_sg.base_url, root=tmp_path))
    assert cached is not None
    assert cached.server_info is None
    assert [name for name, _, _ in fake_sg.calls] == [
        'info',
        'schema_read',
        'schema_entity_read',
    ]


@pytest.mark.parametrize(
    'info_value',
    [
        pytest.param(None, id='absent'),
        pytest.param({'version': [8, 64, 0]}, id='non-callable'),
    ],
)
def test_engine_schema_refresh_normalizes_unavailable_server_info(
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    info_value: object | None,
) -> None:
    if info_value is not None:
        monkeypatch.setattr(fake_sg, 'info', info_value, raising=False)
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )

    created.load_schema()

    cached = load_cache(cache_file(fake_sg.base_url, root=tmp_path))
    assert cached is not None
    assert cached.server_info is None
    assert [name for name, _, _ in fake_sg.calls] == [
        'schema_read',
        'schema_entity_read',
    ]


def test_engine_schema_refresh_records_callable_server_info(
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server_info = {'version': [8, 64, 0]}

    def info() -> dict[str, Any]:
        fake_sg.record('info')
        return server_info

    monkeypatch.setattr(fake_sg, 'info', info, raising=False)
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )

    created.load_schema()

    cached = load_cache(cache_file(fake_sg.base_url, root=tmp_path))
    assert cached is not None
    assert cached.server_info == server_info
    assert [name for name, _, _ in fake_sg.calls] == [
        'info',
        'schema_read',
        'schema_entity_read',
    ]


def test_engine_valid_live_schema_publishes_and_updates_cache(
    fake_sg: FakeShotgun,
    tmp_path: Path,
) -> None:
    registry = Registry()
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=registry,
        bind_default=False,
    )

    schemas = created.load_schema(force=True)

    assert 'Shot' in schemas
    assert registry.binding_snapshot() is not None
    cached = load_cache(cache_file(fake_sg.base_url, root=tmp_path))
    assert cached is not None
    assert 'Shot' in cached.fields


def test_execute_call_passes_exact_find_kwargs_and_echoes(
    fake_sg: FakeShotgun, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
        echo=True,
    )
    compiled = FindCall(
        entity_type='Shot',
        filters=[['code', 'is', 'AB']],
        fields=['id', 'code'],
        order=[{'field_name': 'id', 'direction': 'asc'}],
        limit=5,
        page=2,
        extra={'retired_only': True},
    )
    with caplog.at_level('INFO', logger='sgalchemy.engine'):
        created.execute_call(compiled)
    assert fake_sg.calls[-1] == (
        'find',
        ('Shot', [['code', 'is', 'AB']], ['id', 'code']),
        {
            'order': [{'field_name': 'id', 'direction': 'asc'}],
            'limit': 5,
            'page': 2,
            'retired_only': True,
        },
    )
    assert 'find' in caplog.records[-1].message


def test_execute_call_preserves_summary_grouping_presence(
    engine: Engine, fake_sg: FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    summarize = Mock(wraps=fake_sg.summarize)
    monkeypatch.setattr(fake_sg, 'summarize', summarize)
    engine.execute_call(
        SummarizeCall(
            entity_type='Shot',
            filters=[],
            summary_fields=[{'field': 'id', 'type': 'count'}],
            extra={'include_archived_projects': False},
        )
    )
    assert 'grouping' not in summarize.call_args.kwargs
    grouping = [{'field': 'sg_status_list', 'type': 'exact', 'direction': 'asc'}]
    result = engine.execute_call(
        SummarizeCall(
            entity_type='Shot',
            filters=[],
            summary_fields=[{'field': 'id', 'type': 'count'}],
            grouping=grouping,
        )
    )
    assert summarize.call_args.kwargs['grouping'] == grouping
    assert result == {'summaries': {}, 'groups': []}


def test_explicit_serialize_is_rejected_without_supplied_connection() -> None:
    value = 'tok'
    with pytest.raises(ValueError, match='serialize'):
        create_engine('https://x.example.com', session_token=value, serialize=True)


def test_failed_close_is_retained_across_paths_until_disposal_retry() -> None:
    factory = FakeShotgunFactory()
    engine = create_engine('https://fake.example.com', creator=factory)
    engine.call('schema_read')
    failing = factory.instances[0]
    failing.close_errors.append(RuntimeError('close failed'))
    failing.close_errors.append(RuntimeError('close failed again'))
    with pytest.raises(RuntimeError, match='close failed'):
        engine.dispose()
    engine.call('schema_read')  # discards the stale slot quietly; failure retained
    assert failing.closed is False
    engine.dispose()
    assert failing.closed is True
    assert len(factory.instances) == 2


def test_failed_registry_bind_does_not_publish_schema(
    fake_sg: FakeShotgun, tmp_path: Path
) -> None:
    registry = Registry()
    first = create_engine(
        connection=fake_sg, cache_root=tmp_path, registry=registry, bind_default=False
    )
    first.load_schema()
    other: Any = MagicMock()
    other.base_url = 'https://other.example.com'
    other.schema_read = fake_sg.schema_read
    other.schema_entity_read = fake_sg.schema_entity_read
    second = create_engine(
        connection=other, cache_root=tmp_path, registry=registry, bind_default=False
    )
    binding = registry.binding_snapshot()
    with pytest.raises(SgalchemyError) as raised:
        second.load_schema()
    assert str(raised.value) == (
        'registry is already bound to a different site (one site per Registry); '
        'pass registry=Registry() or bind_default=False to the engine factory for a second site'
    )
    assert second._prepared_schema() is None
    assert registry.binding_snapshot() is binding


def test_fresh_disk_cache_uses_policy_without_backend(
    fake_sg: FakeShotgun, tmp_path: Path
) -> None:
    cached = SchemaCache(
        fields=fake_sg.schema_read(),
        entities=fake_sg.schema_entity_read(),
        fetched_at=datetime.now(UTC),
    )
    fake_sg.calls.clear()
    store_cache(cache_file(fake_sg.base_url, root=tmp_path), cached)
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        policy=TTLPolicy(hours=1),
        registry=Registry(),
        bind_default=False,
    )
    assert 'Shot' in created.load_schema()
    assert fake_sg.calls == []


def test_load_schema_probes_info_through_call_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = _InfoShotgun({'version': [8, 0, 0]})
    engine = create_engine(
        connection.base_url, creator=lambda: connection, cache_root=tmp_path, bind_default=False
    )
    acquired: list[_ThreadSlot] = []
    released: list[_ThreadSlot] = []
    acquire, release = Engine._acquire, Engine._release

    def counting_acquire(self: Engine) -> _ThreadSlot:
        slot = acquire(self)
        acquired.append(slot)
        return slot

    def counting_release(self: Engine, slot: _ThreadSlot) -> None:
        released.append(slot)
        release(self, slot)

    monkeypatch.setattr(Engine, '_acquire', counting_acquire)
    monkeypatch.setattr(Engine, '_release', counting_release)
    monkeypatch.setattr(Engine, 'connection', property(lambda _self: pytest.fail('raw read')))
    engine.load_schema()
    assert connection.info_calls == 1
    assert len(acquired) == len(released) == 4
    assert acquired == released
    assert not connection.closed
    engine.dispose()
    assert connection.closed


def test_load_schema_with_source_reports_fetch_reuse_and_write_failure(
    fake_sg: FakeShotgun, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    load_schema_with_source = _engine.load_schema_with_source

    def fresh(root: Path) -> Engine:
        return create_engine(connection=fake_sg, cache_root=root, bind_default=False)

    prepared = fresh(tmp_path)
    first = load_schema_with_source(prepared, force=False)
    assert (first.fetched, first.store_error) == (True, None)
    assert first.path == cache_file('https://fake.example.com', root=tmp_path)
    reused = load_schema_with_source(fresh(tmp_path), force=False)
    assert reused.fetched is False
    assert reused.cache.fetched_at == first.cache.fetched_at
    with pytest.raises(RuntimeError, match='no schema load happened'):
        load_schema_with_source(prepared, force=False)
    forced = load_schema_with_source(prepared, force=True)
    assert forced.fetched is True
    with caplog.at_level(logging.WARNING, logger='sgalchemy.engine'):
        failed = load_schema_with_source(fresh(_blocked_root(tmp_path)), force=False)
    assert failed.fetched is True
    assert isinstance(failed.store_error, OSError)
    assert [r for r in caplog.records if r.name == 'sgalchemy.engine'] == []


def test_missing_provider_dependency_keeps_its_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    break_provider_import(monkeypatch, 'provider_dependency')
    engine = create_engine(
        'https://fake.example.com',
        session_token=tmp_path.name,
        cache_root=tmp_path,
        bind_default=False,
    )
    with pytest.raises(ModuleNotFoundError) as excinfo:
        engine.load_schema()
    assert excinfo.value.name == 'provider_dependency'
    assert str(excinfo.value) == "No module named 'provider_dependency'"
    assert excinfo.value.__cause__ is None


def test_missing_provider_names_the_extras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, 'shotgun_api3', None)
    engine = create_engine(
        'https://fake.example.com',
        session_token=tmp_path.name,
        cache_root=tmp_path,
        bind_default=False,
    )
    with pytest.raises(ModuleNotFoundError) as excinfo:
        engine.load_schema()
    assert excinfo.value.name == 'shotgun_api3'
    assert str(excinfo.value) == (
        'no SG API provider is installed; install sgalchemy with the official, api, or async '
        'extra, or pass connection= or creator='
    )
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
    assert excinfo.value.__cause__.name == 'shotgun_api3'


def test_registry_selection(fake_sg: FakeShotgun) -> None:
    explicit = Registry()
    assert (
        create_engine(connection=fake_sg, registry=explicit, bind_default=False).registry
        is explicit
    )
    assert create_engine(connection=fake_sg).registry is default_registry()
    private = create_engine(connection=fake_sg, bind_default=False).registry
    assert private is not default_registry()


@pytest.mark.parametrize(
    'case',
    _SCHEMA_FAULT_CASES,
)
def test_schema_fault_classification(
    engine: Engine,
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, Exception, tuple[Any, ...], bool],
) -> None:
    method, error, args, stale = case
    marked = False

    def mark() -> None:
        nonlocal marked
        marked = True

    monkeypatch.setattr(engine, 'mark_schema_stale', mark)
    fake_sg.errors[method] = [error]
    with pytest.raises(type(error), match=r'exist|field|entity'):
        engine.call(method, *args)
    assert marked is stale


def test_schema_fault_marks_stale_after_serialization_unlock(
    fake_sg: FakeShotgun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = False
    marked = False
    tracking_lock = MagicMock()

    def enter_lock() -> None:
        nonlocal owned
        owned = True

    def exit_lock(*_args: object) -> None:
        nonlocal owned
        owned = False

    def mark_stale() -> None:
        nonlocal marked
        assert not owned
        marked = True

    tracking_lock.__enter__.side_effect = enter_lock
    tracking_lock.__exit__.side_effect = exit_lock
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
        serialize=True,
    )
    created._call_lock = tracking_lock
    monkeypatch.setattr(created, 'mark_schema_stale', mark_stale)
    fake_sg.find_error = RuntimeError('API read() Shot.made_up does not exist')
    with pytest.raises(RuntimeError, match='made_up'):
        created.call('find', 'Shot', [], ['id'])
    assert marked


def test_schema_load_cache_force_policy_and_stale_lifecycle(
    engine: Engine, fake_sg: FakeShotgun
) -> None:
    first = engine.load_schema()
    assert first is engine.registry.schema_snapshot()
    assert engine._prepared_schema() is engine.registry.schema_snapshot()
    assert engine.schema is engine.registry.schema_snapshot()
    assert engine.prepared_binding() is engine.registry.binding_snapshot()
    assert engine.load_binding() is engine.registry.binding_snapshot()
    assert entity_type(engine.entities.Shot) == 'Shot'
    engine.load_schema()
    assert [name for name, _, _ in fake_sg.calls].count('schema_read') == 1
    engine.load_schema(force=True)
    engine.mark_schema_stale()
    engine.load_schema()
    assert [name for name, _, _ in fake_sg.calls].count('schema_read') == 3


def test_serialize_defaults_to_locked_for_supplied_connection(fake_sg: FakeShotgun) -> None:
    engine = create_engine(connection=fake_sg)
    assert engine._call_lock is not None
    relaxed = create_engine(connection=fake_sg, serialize=False)
    assert relaxed._call_lock is None


def test_stale_mark_is_not_lost_during_schema_load(
    engine: Engine, fake_sg: FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = Event()
    release = Event()
    original = fake_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=2)
        return original()

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        loading = pool.submit(engine.load_schema)
        assert entered.wait(timeout=2)
        marking = pool.submit(engine.mark_schema_stale)
        release.set()
        loading.result(timeout=2)
        marking.result(timeout=2)
    engine.load_schema()
    assert [name for name, _, _ in fake_sg.calls].count('schema_read') == 2


def test_string_cache_root_is_accepted(fake_sg: FakeShotgun, tmp_path: Path) -> None:
    engine = create_engine(connection=fake_sg, cache_root=str(tmp_path), bind_default=False)
    engine.load_schema()
    assert cache_file(engine.site, root=tmp_path).exists()


@pytest.mark.parametrize(('serialize', 'expected'), [(False, 2), (True, 1)])
def test_supplied_connection_serialization(
    fake_sg: FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    serialize: bool,
    expected: int,
) -> None:
    barrier = Barrier(2) if not serialize else None
    first_entered = Event()
    release = Event()
    lock = Lock()
    active = 0
    max_active = 0

    def blocking_schema_read() -> dict[str, Any]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        first_entered.set()
        if barrier is not None:
            barrier.wait(timeout=2)
        else:
            assert release.wait(timeout=2)
        with lock:
            active -= 1
        return {}

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    created = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
        serialize=serialize,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(created.call, 'schema_read')
        assert first_entered.wait(timeout=2)
        second = pool.submit(created.call, 'schema_read')
        if serialize:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert max_active == expected


@pytest.mark.parametrize('invalid', [False, True])
def test_call_progress_excludes_acquisition_and_validation_failure(
    fake_sg: FakeShotgun,
    invalid: bool,
) -> None:
    def creator() -> ClosableShotgunLike:
        if not invalid:
            raise RuntimeError('creation failed')
        fake_sg.base_url = 'https://different.example.com'
        return fake_sg

    engine = create_engine('https://fake.example.com', creator=creator, bind_default=False)
    progress = CallProgress()
    with pytest.raises((ValueError, RuntimeError)):
        engine.call_with_progress('schema_read', (), {}, progress)
    assert not progress.invoked.is_set()
    assert not progress.dispatched.is_set()


def test_unwritable_cache_warns_and_publishes_the_fetched_schema(
    fake_sg: FakeShotgun, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _blocked_root(tmp_path)
    engine = create_engine(connection=fake_sg, cache_root=root, bind_default=False)
    with caplog.at_level(logging.WARNING, logger='sgalchemy.engine'):
        schemas = engine.load_schema()
    assert 'Shot' in schemas
    assert engine.schema == schemas
    path = cache_file(engine.site, root=root)
    [message] = [r.getMessage() for r in caplog.records if r.name == 'sgalchemy.engine']
    assert message.startswith(f'could not write the schema cache {path}: ')
    assert message.endswith(_NOT_CACHED)
