import asyncio
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from sgalchemy._core._async_engine import create_async_engine
from sgalchemy._core._construction import (
    _normalize_site,
    checked_async_client,
    checked_sync_client,
    close_quietly,
    ensure_awaitable,
    validate_site,
    validate_static_credentials,
)
from sgalchemy._core._engine import create_engine
from sgalchemy._shotgun import (
    REQUIRED_SHOTGUN_METHODS,
)
from tests._fakes import FakeAsyncShotgun, FakeShotgun


def test_checked_async_client_rejects_malformed_site_safely() -> None:
    fake = FakeAsyncShotgun()
    fake.base_url = 'http://[::1'

    with pytest.raises(
        ValueError, match='async_creator returned a client for a different site'
    ) as excinfo:
        checked_async_client(fake, 'https://fake.example.com')

    assert (fake.closed, str(excinfo.value)) == (
        False,
        'async_creator returned a client for a different site',
    )


@pytest.mark.parametrize('method_name', REQUIRED_SHOTGUN_METHODS)
def test_checked_async_client_rejects_missing_call_surface(method_name: str) -> None:
    client = FakeAsyncShotgun()
    setattr(client, method_name, None)

    with pytest.raises(TypeError, match='required asynchronous client surface'):
        checked_async_client(client, 'https://fake.example.com')

    assert client.closed is False


@pytest.mark.parametrize('method_name', REQUIRED_SHOTGUN_METHODS)
def test_checked_async_client_rejects_sync_call_surface(method_name: str) -> None:
    client = FakeAsyncShotgun()
    setattr(client, method_name, getattr(FakeShotgun(), method_name))

    with pytest.raises(TypeError, match='required asynchronous client surface'):
        checked_async_client(client, 'https://fake.example.com')

    assert client.closed is False


def test_checked_async_client_rejects_sync_close_surface() -> None:
    base: Any = FakeAsyncShotgun

    class SyncCloseFake(base):
        def close(self) -> None:
            self.closed = True

    client = SyncCloseFake()

    with pytest.raises(TypeError, match='required asynchronous client surface'):
        checked_async_client(client, 'https://fake.example.com')

    assert client.closed is False


def test_checked_async_client_requires_close_surface() -> None:
    class NoClose:
        base_url = 'https://fake.example.com'

    with pytest.raises(TypeError, match=r'surface|close'):
        checked_async_client(NoClose(), 'https://fake.example.com')


def test_checked_async_client_validates_without_performing_cleanup() -> None:
    fake = FakeAsyncShotgun()
    assert checked_async_client(fake, 'https://fake.example.com') is fake
    with pytest.raises(ValueError, match='site'):
        checked_async_client(fake, 'https://other.example.com')
    assert fake.closed is False  # awaited cleanup belongs to the async backend


def test_checked_clients_do_not_require_optional_info() -> None:
    sync_client = FakeShotgun()
    async_client = FakeAsyncShotgun()

    assert not hasattr(sync_client, 'info')
    assert not hasattr(async_client, 'info')
    assert checked_sync_client(sync_client, 'https://fake.example.com') is sync_client
    assert checked_async_client(async_client, 'https://fake.example.com') is async_client


def test_checked_sync_client_closes_and_rejects_wrong_site() -> None:
    fake = FakeShotgun()
    with pytest.raises(ValueError, match='site'):
        checked_sync_client(fake, 'https://other.example.com')
    assert fake.closed is True


def test_checked_sync_client_closes_closable_result_missing_call_surface() -> None:
    class Closable:
        base_url = 'https://fake.example.com'

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    result = Closable()
    with pytest.raises(TypeError, match=r'surface|find'):
        checked_sync_client(result, 'https://fake.example.com')
    assert result.closed is True


def test_checked_sync_client_preserves_site_error_when_close_lookup_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingCloseLookup:
        base_url = 'https://fake.example.com'

        def __init__(self) -> None:
            self._close_lookups = 0
            self._delegate = FakeShotgun()

        def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - Delegate unknown provider attributes around a failing close descriptor.
            return getattr(self._delegate, name)

        @property
        def close(self) -> Callable[[], None]:
            self._close_lookups += 1
            if self._close_lookups > 1:
                raise RuntimeError('close descriptor failed')
            return lambda: None

    result = FailingCloseLookup()
    with pytest.raises(ValueError, match='different site'):
        checked_sync_client(result, 'https://other.example.com')

    assert any(record.message == 'client cleanup close failed' for record in caplog.records)


@pytest.mark.parametrize('method_name', REQUIRED_SHOTGUN_METHODS)
def test_checked_sync_client_rejects_async_call_surface(method_name: str) -> None:
    client = FakeShotgun()
    setattr(client, method_name, getattr(FakeAsyncShotgun(), method_name))

    with pytest.raises(TypeError, match='required synchronous client surface'):
        checked_sync_client(client, 'https://fake.example.com')

    assert client.closed is True


def test_checked_sync_client_rejects_async_close_surface() -> None:
    client: Any = FakeShotgun()
    client.close = AsyncMock()

    with pytest.raises(TypeError, match='required synchronous client surface'):
        checked_sync_client(client, 'https://fake.example.com')

    client.close.assert_not_called()


def test_checked_sync_client_rejects_awaitable_result_with_teaching_error() -> None:
    async def make() -> FakeShotgun:
        return FakeShotgun()

    coroutine = make()
    with pytest.raises(TypeError, match='async_creator'):
        checked_sync_client(coroutine, 'https://fake.example.com')


def test_checked_sync_client_rejects_malformed_site_safely() -> None:
    fake = FakeShotgun()
    fake.base_url = 'http://[::1'

    with pytest.raises(
        ValueError, match='creator returned a client for a different site'
    ) as excinfo:
        checked_sync_client(fake, 'https://fake.example.com')

    assert (fake.closed, str(excinfo.value)) == (
        True,
        'creator returned a client for a different site',
    )


@pytest.mark.parametrize('method_name', REQUIRED_SHOTGUN_METHODS)
def test_checked_sync_client_rejects_missing_call_surface(method_name: str) -> None:
    client = FakeShotgun()
    setattr(client, method_name, None)

    with pytest.raises(TypeError, match='required synchronous client surface'):
        checked_sync_client(client, 'https://fake.example.com')

    assert client.closed is True


def test_cleanup_cancellation_propagates_through_primary_error_cleanup() -> None:
    class CancelledCloseFake(FakeShotgun):
        def close(self) -> None:
            raise asyncio.CancelledError('cleanup cancelled')

    fake = CancelledCloseFake()

    with pytest.raises(asyncio.CancelledError, match='cleanup cancelled'):
        checked_sync_client(fake, 'https://other.example.com')


def test_cleanup_failure_does_not_mask_primary_error() -> None:
    fake = FakeShotgun()
    fake.close_errors.append(RuntimeError('cleanup exploded'))
    with pytest.raises(ValueError, match='site'):
        checked_sync_client(fake, 'https://other.example.com')


def test_close_quietly_skips_descriptor_backed_async_callable() -> None:
    bound_close = AsyncMock()

    class CallDescriptor:
        def __get__(self, instance: object | None, owner: type[object]) -> object:
            return self if instance is None else bound_close

        def __call__(self) -> None:
            raise AssertionError('class-level descriptor must not be called')

    class DescriptorCallable:
        __call__ = CallDescriptor()

    class Client:
        close: Any = DescriptorCallable()

    close_quietly(Client())

    bound_close.assert_not_called()


def test_close_quietly_swallows_close_failures() -> None:
    fake = FakeShotgun()
    fake.close_errors.append(RuntimeError('boom'))
    close_quietly(fake)
    close_quietly(object())


def test_credential_secrets_are_absent_from_repr() -> None:
    value = 'secret-token'
    creds = validate_static_credentials(
        script_name='tool', api_key='secret-key', session_token=None
    )
    token = validate_static_credentials(
        script_name=None,
        api_key=None,
        session_token=value,
    )
    assert 'secret-key' not in repr(creds)
    assert 'secret-token' not in repr(token)


def test_ensure_awaitable_rejects_sync_result_without_cleanup() -> None:
    fake = FakeShotgun()
    with pytest.raises(TypeError, match='creator'):
        ensure_awaitable(fake)
    assert fake.closed is False  # awaited cleanup belongs to the async backend


@pytest.mark.parametrize(
    ('script_name', 'api_key', 'session_token'),
    [
        ('tool', None, None),
        (None, 'k', None),
        ('tool', 'k', 'tok'),
        (None, None, None),
        ('', 'k', None),
        ('tool', '', None),
        (None, None, ''),
    ],
)
def test_invalid_credential_shapes_are_rejected(
    script_name: str | None, api_key: str | None, session_token: str | None
) -> None:
    with pytest.raises(ValueError, match=r'script_name|api_key|session_token|credential'):
        validate_static_credentials(
            script_name=script_name, api_key=api_key, session_token=session_token
        )


def test_normalize_site_is_idempotent_and_strips_trailing_slash() -> None:
    site = _normalize_site('https://example.com/base/')
    assert site == 'https://example.com/base'
    assert _normalize_site(site) == site


def test_script_pair_credentials_produce_provider_kwargs() -> None:
    creds = validate_static_credentials(script_name='tool', api_key='k', session_token=None)
    assert creds.provider_kwargs() == {'script_name': 'tool', 'api_key': 'k'}


def test_session_token_credentials_produce_provider_kwargs() -> None:
    value = 'tok'
    creds = validate_static_credentials(script_name=None, api_key=None, session_token=value)
    assert creds.provider_kwargs() == {'session_token': 'tok'}


def test_supplied_connection_check_ignores_missing_methods_and_close() -> None:
    class Partial:
        base_url = 'https://fake.example.com'

        def find(self, *args: object, **kwargs: object) -> list[object]:
            del args, kwargs
            return []

        async def close(self) -> None:
            pass

    create_engine(connection=Partial(), bind_default=False)  # pyright: ignore[reportArgumentType]
    create_async_engine(
        connection=AsyncMock(base_url='https://fake.example.com'), bind_default=False
    )


@pytest.mark.parametrize(
    'url',
    [
        'ftp://example.com',
        'example.com',
        'https://',
        'https://:443',
        'https://example.com#',
        'https://example.com?',
        'https://example.com:not-a-port',
        'https://example.com:99999',
        'https://user:secret-value@example.com',
        'https://user:secret-value@example.com\uff0fpath',
        'https://example.com?token=secret-value',
        'https://example.com#secret-value',
    ],
)
def test_validate_site_rejects_invalid_urls_without_echoing_them(url: str) -> None:
    with pytest.raises(ValueError, match='url') as excinfo:
        validate_site(url)
    assert 'secret-value' not in str(excinfo.value)
    assert url not in str(excinfo.value)


def test_validate_site_returns_canonical_form() -> None:
    assert validate_site('HTTPS://Studio.ShotGrid.autodesk.com/') == (
        'https://studio.shotgrid.autodesk.com'
    )


def test_wrong_form_cleanup_failure_does_not_mask_teaching_error() -> None:
    class ExplodingAwaitable:
        def __await__(self) -> Generator[None, None, None]:  # pragma: no cover - never awaited
            yield

        def close(self) -> None:
            raise RuntimeError('cleanup exploded')

    with pytest.raises(TypeError, match='async_creator'):
        checked_sync_client(ExplodingAwaitable(), 'https://fake.example.com')
