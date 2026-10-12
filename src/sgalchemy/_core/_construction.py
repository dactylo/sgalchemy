"""Validate factory inputs and provider client capabilities."""

import dataclasses
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import cast
from urllib.parse import urlsplit

from sgalchemy._shotgun import (
    REQUIRED_SHOTGUN_METHODS,
    AsyncClosableShotgunLike,
    ClosableShotgunLike,
)

_LOGGER = logging.getLogger('sgalchemy.engine')


@dataclasses.dataclass(frozen=True)
class StaticCredentials:
    """Carry one static credential selection with secrets omitted from repr.

    Obtain a validated selection through `validate_static_credentials()`.
    Exactly one nonempty session token or complete script-name/API-key pair is
    required; direct dataclass construction does not enforce that invariant.

    Secret fields remain absent from repr, while `provider_kwargs()` supplies
    the provider's original keyword names.
    """

    _script_name: str | None = None
    _api_key: str | None = dataclasses.field(default=None, repr=False)
    _session_token: str | None = dataclasses.field(default=None, repr=False)

    def provider_kwargs(self) -> dict[str, str]:
        """Return the exact provider constructor keywords for this selection.

        Returns:
            Exact provider constructor keyword mapping for the selected
            credential form.

        Raises:
            RuntimeError: If the validated credential selection is incomplete.
        """
        if self._session_token is not None:
            return {'session_token': self._session_token}
        if self._script_name is None or self._api_key is None:
            raise RuntimeError('validated credentials are incomplete')
        return {'script_name': self._script_name, 'api_key': self._api_key}


def _is_async_callable(value: Callable[..., object]) -> bool:
    """Return whether one callable declares an awaited call contract."""
    return inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(value.__call__)


def _normalize_site(url: str) -> str:
    """Return the canonical site spelling for equality comparisons."""
    parts = urlsplit(url)
    path = parts.path.rstrip('/')
    return f'{parts.scheme.lower()}://{parts.netloc.lower()}{path}'


def close_quietly(client: object) -> None:
    """Close one synchronously closable value, containing ordinary failures.

    Cancellation and interruption propagate with their language-level
    semantics.

    Args:
        client: Potentially closable client value.
    """
    try:
        close = getattr(client, 'close', None)
        if not callable(close) or _is_async_callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            retire = getattr(result, 'close', None)
            if callable(retire):
                retire()
    except Exception:
        _LOGGER.exception('client cleanup close failed')


def _require_surface(
    result: object,
    *,
    asynchronous: bool,
    cleanup: bool,
) -> None:
    """Validate the strategy-matched call surface, optionally closing on failure."""
    base_url = getattr(result, 'base_url', None)
    members = {name: getattr(result, name, None) for name in (*REQUIRED_SHOTGUN_METHODS, 'close')}
    missing = [name for name, member in members.items() if not callable(member)]
    mismatched = [
        name
        for name, member in members.items()
        if callable(member) and _is_async_callable(member) != asynchronous
    ]
    if isinstance(base_url, str) and not missing and not mismatched:
        return
    if cleanup:
        close_quietly(result)
    mode = 'asynchronous' if asynchronous else 'synchronous'
    raise TypeError(
        f'creator result does not implement the required {mode} client surface '
        '(base_url plus strategy-matched close and Shotgun call methods such as find)'
    )


def checked_async_client(result: object, site: str) -> AsyncClosableShotgunLike:
    """Validate one awaited async creator result before publication.

    Args:
        result: Value produced by awaiting an `async_creator` call.
        site: Canonical engine site.

    Returns:
        The validated async client.

    Raises:
        TypeError: If the result lacks the required surface.
        ValueError: If the client's site does not match the engine site.
    """
    _require_surface(result, asynchronous=True, cleanup=False)
    client = cast('AsyncClosableShotgunLike', result)
    try:
        if _normalize_site(client.base_url) == site:
            return client
    except ValueError:
        pass
    raise ValueError('async_creator returned a client for a different site')


def checked_sync_client(result: object, site: str) -> ClosableShotgunLike:
    """Validate one sync creator result before publication.

    Rejected results receive best-effort synchronous cleanup before the
    validation failure is raised. Ordinary cleanup failures are logged.

    Args:
        result: Value returned by a `creator` callable.
        site: Canonical engine site.

    Returns:
        The validated client, unchanged.

    Raises:
        TypeError: If the result is awaitable or lacks the required synchronous
            provider and close surface.
        ValueError: If the client's site does not match the engine site.
    """
    if inspect.isawaitable(result):
        close_quietly(result)
        raise TypeError(
            'creator returned an awaitable; pass an async callable as async_creator instead'
        )
    _require_surface(result, asynchronous=False, cleanup=True)
    client = cast('ClosableShotgunLike', result)
    try:
        if _normalize_site(client.base_url) == site:
            return client
    except ValueError:
        pass
    close_quietly(client)
    raise ValueError('creator returned a client for a different site')


def ensure_awaitable(result: object) -> Awaitable[object]:
    """Require an awaitable from an `async_creator` call.

    Performs no cleanup: a non-awaitable result may carry an async `close()`
    that only a caller running in the event loop can await. The async backend
    classifies the value and awaits its cleanup exactly once.

    Args:
        result: Raw return value of calling the `async_creator`.

    Returns:
        The awaitable, unchanged.

    Raises:
        TypeError: If the value is not awaitable.
    """
    if inspect.isawaitable(result):
        return result
    raise TypeError(
        'async_creator returned a non-awaitable; pass a synchronous callable as creator instead'
    )


def require_static_credential_https(site: str) -> None:
    """Reject plaintext transport for library-owned static credentials.

    Args:
        site: Canonical site URL for an engine-owned client.

    Raises:
        ValueError: If the canonical site does not use HTTPS.
    """
    if urlsplit(site).scheme != 'https':
        raise ValueError('static credentials require an https site URL')


def validate_site(url: str) -> str:
    """Validate one explicit site URL and return its canonical form.

    Args:
        url: Explicit factory site URL.

    Returns:
        Canonical site string.

    Raises:
        ValueError: If the URL is incomplete, uses an unsupported scheme, or
            carries userinfo, a query string, or a fragment. Messages never
            include the URL value.
    """
    if '?' in url or '#' in url:
        raise ValueError('url must not contain a query string or fragment')
    try:
        parts = urlsplit(url)
    except ValueError:
        raise ValueError('url must be a complete http or https site URL') from None
    if parts.scheme.lower() not in {'http', 'https'} or not parts.netloc:
        raise ValueError('url must be a complete http or https site URL')
    if '@' in parts.netloc:
        raise ValueError('url must not embed userinfo credentials')
    try:
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        raise ValueError('url must be a complete http or https site URL') from None
    if not hostname or (port is None and parts.netloc.endswith(':')):
        raise ValueError('url must be a complete http or https site URL')
    return _normalize_site(url)


def validate_static_credentials(
    *,
    script_name: str | None,
    api_key: str | None,
    session_token: str | None,
) -> StaticCredentials:
    """Validate exactly one nonempty static credential form.

    Args:
        script_name: Script identity name, paired with `api_key`.
        api_key: Script identity key, paired with `script_name`.
        session_token: JSON-RPC session token.

    Returns:
        Validated immutable credential selection.

    Raises:
        ValueError: If a value is empty, the script pair is incomplete, forms
            are combined, or no form is present. Messages never include values.
    """
    fields = {
        'api_key': api_key,
        'script_name': script_name,
        'session_token': session_token,
    }
    for name, value in fields.items():
        if value is not None and not value:
            raise ValueError(f'{name} must not be empty')
    has_script = script_name is not None or api_key is not None
    if has_script and (script_name is None or api_key is None):
        raise ValueError('script credentials require both script_name and api_key')
    if session_token is not None and has_script:
        raise ValueError('pass either script_name with api_key or session_token, not both')
    if session_token is None and not has_script:
        raise ValueError('url requires script_name with api_key or session_token')
    return StaticCredentials(
        _script_name=script_name, _api_key=api_key, _session_token=session_token
    )
