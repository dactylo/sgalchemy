"""Asynchronous engine, backend gateway, and schema lifecycle."""

import asyncio
import importlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, cast

from sgalchemy._core._async_backend import AsyncBackend, NativeAsyncBackend, ThreadedSyncBackend
from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._construction import (
    StaticCredentials,
    require_static_credential_https,
    validate_site,
    validate_static_credentials,
)
from sgalchemy._core._engine_common import (
    AUTH_RECOVERY_NOTE,
    CallProgress,
    EntityNamespace,
    add_note_once,
    looks_like_auth_fault,
    looks_like_schema_fault,
    provider_call_arguments,
    validated_schema,
)
from sgalchemy._core._registry import Registry, RegistryBinding, default_registry
from sgalchemy._errors import SgalchemyError
from sgalchemy._shotgun import (
    AsyncClosableShotgunLike,
    AsyncShotgunLike,
    ClosableShotgunLike,
    QueryResult,
    ShotgunLike,
)
from sgalchemy.schema._cache import (
    SchemaCache,
    load_cache,
    normalize_server_info,
    store_cache,
)
from sgalchemy.schema._cachedir import cache_file
from sgalchemy.schema._model import EntitySchema, build_schema
from sgalchemy.schema._policy import SchemaCachePolicy, TTLPolicy

_LOGGER = logging.getLogger('sgalchemy.engine')


class _AsyncAcquisitionBoundary:
    """Wrap ordinary creator failures without intercepting cancellation."""

    def __enter__(self) -> None:
        """Enter one creator acquisition boundary."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Wrap an ordinary `Exception` and propagate other `BaseException` subclasses."""
        del exc_type, traceback
        if isinstance(exc_value, Exception):
            raise _AsyncAcquisitionError(exc_value) from None
        return False


class _AsyncAcquisitionError(Exception):
    """Carry the original creator failure unchanged across the backend boundary."""

    def __init__(self, original: Exception) -> None:
        """Retain the original exception for identity-preserving unwrapping."""
        super().__init__()
        self._original = original


class AsyncEngine:
    """Manage a reusable async backend and one fixed Registry pairing.

    First async use binds the engine to the running event loop; later use on
    another loop raises `RuntimeError`. Tasks on that loop may share the
    engine. Await `load_schema()` before accessing `schema` or `entities`.

    Async context management binds the loop without preparing schema and
    disposes owned backend resources on exit without suppressing exceptions.
    Disposal preserves supplied connections and leaves the engine reusable.

    `site`, `registry`, `schema`, `entities`, `call()`, `load_schema()`, and
    `dispose()` are supported engine surfaces, along with the factory and
    lifecycle behavior. `call_with_progress()`, `execute_call()`,
    `load_binding()`, `prepared_binding()`, and `mark_schema_stale()` are
    package-internal hooks.

    Attributes:
        site: Canonical site identity of the backend.
    """

    def __init__(
        self,
        backend: AsyncBackend,
        *,
        cache_root: Path | None,
        policy: SchemaCachePolicy,
        echo: bool,
        registry: Registry,
    ) -> None:
        """Store validated backend and schema configuration without I/O.

        Args:
            backend: Awaited provider adapter that supplies the engine site.
            cache_root: Schema cache directory override, or `None` to use
                environment and platform defaults.
            policy: Policy deciding whether a loaded disk cache is fresh.
            echo: Log provider calls and arguments on `sgalchemy.engine`.
            registry: Registry permanently paired with this engine.
        """
        self.site = backend.site
        self._backend = backend
        self._cache_root = cache_root
        self._echo = echo
        self._loaded_generation = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._policy = policy
        self._registry = registry
        self._schema: dict[str, EntitySchema] | None = None
        self._schema_lock = asyncio.Lock()
        self._stale_generation = 0

    def _bind_loop(self) -> None:
        """Bind first use to one running event loop and reject another."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError('AsyncEngine cannot be shared across event loops')

    async def __aenter__(self) -> Self:
        """Bind this engine to the running loop and return it."""
        self._bind_loop()
        return self

    async def _supports(self, method: str) -> bool:
        """Probe backend support through acquisition-fault classification."""
        try:
            return await self._backend.supports(method)
        except _AsyncAcquisitionError as err:
            acquisition_fault = _classify_acquisition_fault(err)
        raise acquisition_fault

    async def dispose(self) -> None:
        """Close and forget owned clients; the engine stays usable.

        Drains admitted calls, closes engine-owned clients, and lazily
        recreates clients on the next operation. Supplied connections are never
        closed. Remains bound to this engine's event loop.

        Once started, resource reset continues if its awaiting caller is
        cancelled. Ordinary cleanup failures propagate and leave unclosed
        resources available for a later `dispose()` retry.
        """
        self._bind_loop()
        await self._backend.reset()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Reset owned backend resources without suppressing exceptions."""
        del exc_type, exc_value, traceback
        await self.dispose()

    def mark_schema_stale(self) -> None:
        """Advance stale generation without awaiting or losing concurrent marks."""
        self._stale_generation += 1

    async def call_with_progress(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - The runtime-selected backend determines the result.
        """Observe dispatch and resolved provider entry through the gateway.

        Args:
            method: Provider method name.
            args: Positional provider arguments.
            kwargs: Keyword provider arguments, excluding observation metadata.
            progress: Fresh per-call observation, or None for ordinary calls.

        Returns:
            Provider result without wrapping.
        """
        self._bind_loop()
        if self._echo:
            _LOGGER.info('%s args=%s kwargs=%s', method, args, kwargs)
        try:
            return await self._backend.call_with_progress(method, args, kwargs, progress)
        except _AsyncAcquisitionError as err:
            acquisition_fault = _classify_acquisition_fault(err)
        except Exception as err:
            if looks_like_auth_fault(err):
                add_note_once(err, AUTH_RECOVERY_NOTE)
            if looks_like_schema_fault(method, err):
                _LOGGER.warning('possible schema drift, marking cache stale: %s', err)
                self.mark_schema_stale()
            raise
        raise acquisition_fault

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - Preserve the arbitrary backend call surface and result.
        """Execute one backend call through echo and stale classification.

        Args:
            method: Provider method name.
            *args: Positional provider arguments.
            **kwargs: Keyword provider arguments.

        Returns:
            Provider result without wrapping.

        Raises:
            Exception: The original provider exception, unwrapped.
        """
        return await self.call_with_progress(method, args, kwargs, None)

    async def execute_call(self, call: FindCall | SummarizeCall) -> QueryResult:
        """Execute compiler output without normalizing its call shape.

        Args:
            call: Pure compiler output.

        Returns:
            Provider query result.
        """
        method, args, kwargs = provider_call_arguments(call)
        return await self.call(method, *args, **kwargs)

    @property
    def registry(self) -> Registry:
        """Return this engine's immutable Registry pairing.

        Returns:
            Registry selected during engine construction.
        """
        return self._registry

    def _prepared_schema(self) -> Mapping[str, EntitySchema] | None:
        """Return the Registry's current schema view without performing I/O."""
        return None if self._schema is None else self.registry.schema_snapshot()

    def prepared_binding(self) -> RegistryBinding | None:
        """Return the Registry's current binding snapshot without I/O.

        Returns:
            Current complete binding once this engine has prepared, else
            `None`.
        """
        return None if self._schema is None else self.registry.binding_snapshot()

    @property
    def schema(self) -> Mapping[str, EntitySchema]:
        """Return prepared schema or teach the explicit awaited initialization call.

        Returns:
            Published schema mapping.

        Raises:
            SgalchemyError: If the engine schema has not been loaded.
        """
        snapshot = self._prepared_schema()
        if snapshot is None:
            raise SgalchemyError(
                'async engine schema is not loaded; await engine.load_schema() first'
            )
        return snapshot

    @property
    def entities(self) -> EntityNamespace:
        """Return the prepared registry namespace without I/O.

        Returns:
            Namespace resolving classes from this engine's Registry.

        Raises:
            SgalchemyError: If the engine schema has not been loaded.
        """
        _ = self.schema
        return EntityNamespace(self.registry)

    async def load_schema(self, *, force: bool = False) -> Mapping[str, EntitySchema]:
        """Prepare schema and Registry state through an explicit awaited boundary.

        Args:
            force: Ignore a fresh cache and fetch from the backend.

        Returns:
            The Registry's current binding schemas after preparation.

        Raises:
            SgalchemyError: If the provider returns malformed schema data or
                the Registry is already bound to a different site.
            Exception: Original provider or schema-cache exception, unwrapped.
        """
        self._bind_loop()
        async with self._schema_lock:
            current_generation = self._stale_generation
            if self._schema is None or self._loaded_generation != current_generation or force:
                path = cache_file(self.site, root=self._cache_root)
                cache = None
                if not force and self._schema is None and current_generation == 0:
                    cache = await asyncio.to_thread(load_cache, path)
                if cache is None or not self._policy.is_fresh(cache):
                    raw_server_info = (
                        await self.call('info') if await self._supports('info') else None
                    )
                    cache = SchemaCache(
                        fields=await self.call('schema_read'),
                        entities=await self.call('schema_entity_read'),
                        fetched_at=datetime.now(UTC),
                        server_info=normalize_server_info(raw_server_info),
                    )
                    candidate = validated_schema(cache)
                    await asyncio.to_thread(store_cache, path, cache)
                else:
                    candidate = build_schema(cache)
                self.registry.bind(candidate, key=self.site)
                self._schema = candidate
                self._loaded_generation = current_generation
        return self.schema

    async def load_binding(self, *, force: bool = False) -> RegistryBinding:
        """Prepare schema and return the Registry's current binding snapshot.

        Args:
            force: Ignore a fresh cache and fetch from the backend.

        Returns:
            The Registry's current complete binding after preparation.

        Raises:
            SgalchemyError: If preparation fails to publish a binding or the
                Registry is already bound to a different site.
        """
        await self.load_schema(force=force)
        binding = self.prepared_binding()
        if binding is None:
            raise SgalchemyError(
                'async engine schema is not loaded; await engine.load_schema() first'
            )
        return binding


def _classify_acquisition_fault(err: _AsyncAcquisitionError) -> Exception:
    """Annotate and return the original creator exception for public delivery."""
    original = err._original  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if looks_like_auth_fault(original):
        add_note_once(original, AUTH_RECOVERY_NOTE)
    return original


def _create_native_backend(
    site: str | None,
    connection: AsyncShotgunLike | None,
    credentials: StaticCredentials | None,
    async_creator: Callable[[], Awaitable[AsyncClosableShotgunLike]] | None,
) -> AsyncBackend:
    """Validate and build the lazy native backend."""
    if connection is not None:
        return NativeAsyncBackend(validate_site(connection.base_url), connection=connection)
    if site is None:
        raise RuntimeError('validated native construction lost its site')
    if async_creator is not None:

        def create_from_async_creator() -> object:
            with _AsyncAcquisitionBoundary():
                result = async_creator()
            if not inspect.isawaitable(result):
                return result

            async def await_result() -> AsyncClosableShotgunLike:
                with _AsyncAcquisitionBoundary():
                    return await result

            return await_result()

        return NativeAsyncBackend(
            site,
            creator=create_from_async_creator,
        )
    if credentials is None:
        raise RuntimeError('validated native credentials are incomplete')
    kwargs = credentials.provider_kwargs()

    async def create_native() -> AsyncClosableShotgunLike:
        with _AsyncAcquisitionBoundary():
            try:
                provider = importlib.import_module('shotgun_api3')
                async_cls = provider.AsyncShotgun
            except (AttributeError, ImportError) as err:
                raise ImportError(
                    'native AsyncEngine requires sg-python-api[async]; '
                    "use strategy='threaded' for a synchronous provider"
                ) from err
            return cast('AsyncClosableShotgunLike', async_cls(site, **kwargs))

    return NativeAsyncBackend(site, creator=create_native)


def _create_threaded_backend(
    site: str | None,
    sync_connection: ShotgunLike | None,
    credentials: StaticCredentials | None,
    creator: Callable[[], ClosableShotgunLike] | None,
    *,
    max_workers: int | None,
) -> AsyncBackend:
    """Validate and build the lazy threaded backend at a resolved width."""
    if sync_connection is not None:
        resolved = 1 if max_workers is None else max_workers
        return ThreadedSyncBackend(
            validate_site(sync_connection.base_url),
            connection=sync_connection,
            max_workers=resolved,
        )
    if site is None:
        raise RuntimeError('validated threaded construction lost its site')
    resolved = 4 if max_workers is None else max_workers
    if creator is not None:

        def create_from_creator() -> ClosableShotgunLike:
            with _AsyncAcquisitionBoundary():
                return creator()

        return ThreadedSyncBackend(site, creator=create_from_creator, max_workers=resolved)
    if credentials is None:
        raise RuntimeError('validated threaded credentials are incomplete')
    kwargs = credentials.provider_kwargs()

    def create_sync() -> ClosableShotgunLike:
        with _AsyncAcquisitionBoundary():
            provider = importlib.import_module('shotgun_api3')
            return cast('ClosableShotgunLike', provider.Shotgun(site, **kwargs))

    return ThreadedSyncBackend(site, creator=create_sync, max_workers=resolved)


def _validate_factory_inputs(  # noqa: PLR0913 - Every option needs one validation site.
    strategy: str,
    *,
    url: str | None,
    connection: AsyncShotgunLike | None,
    sync_connection: ShotgunLike | None,
    creator: object,
    async_creator: object,
    has_static: bool,
    max_workers: int | None,
) -> None:
    """Validate strategy, source exclusivity, and sentinel options."""
    if strategy not in ('native', 'threaded'):
        raise ValueError("strategy must be 'native' or 'threaded'")
    if strategy == 'native':
        if sync_connection is not None or creator is not None:
            raise ValueError('native strategy does not accept sync_connection or creator')
        if max_workers is not None:
            raise ValueError('native strategy does not accept max_workers')
        supplied = connection is not None
        chosen = sum(1 for choice in (supplied, async_creator is not None, has_static) if choice)
    else:
        if connection is not None or async_creator is not None:
            raise ValueError('threaded strategy does not accept connection or async_creator')
        if max_workers is not None and max_workers < 1:
            raise ValueError('max_workers must be at least 1')
        supplied = sync_connection is not None
        chosen = sum(1 for choice in (supplied, creator is not None, has_static) if choice)
    if supplied:
        if url is not None or chosen != 1:
            raise ValueError(
                'a supplied connection cannot be combined with url, credentials, or creators'
            )
    elif url is None or chosen != 1:
        raise ValueError(
            'pass exactly one of a supplied connection, url with credentials, '
            'or url with a creator'
        )


def create_async_engine(  # noqa: PLR0913 - Mirror the synchronous factory surface.
    url: str | None = None,
    *,
    connection: AsyncShotgunLike | None = None,
    sync_connection: ShotgunLike | None = None,
    creator: Callable[[], ClosableShotgunLike] | None = None,
    async_creator: Callable[[], Awaitable[AsyncClosableShotgunLike]] | None = None,
    strategy: Literal['native', 'threaded'] = 'native',
    script_name: str | None = None,
    api_key: str | None = None,
    session_token: str | None = None,
    cache_root: Path | None = None,
    policy: SchemaCachePolicy | None = None,
    echo: bool = False,
    registry: Registry | None = None,
    bind_default: bool = True,
    max_workers: int | None = None,
) -> AsyncEngine:
    """Create a provider-neutral async engine without performing I/O.

    Choose exactly one source: a supplied connection, a URL with static
    credentials, or a URL with the strategy's creator. Supplied connections
    cannot be combined with a URL. Static authentication requires HTTPS and
    either a nonempty session token or a complete script-name/API-key pair.

    Creators must return fresh clients for the same site and logical identity
    throughout the engine's lifetime. Creation and provider imports remain
    lazy. Await `load_schema()` before accessing reflected entities.

    Args:
        url: Complete site URL for engine-owned clients.
        connection: Caller-owned native async connection; native strategy only.
        sync_connection: Caller-owned synchronous connection; threaded strategy
            only.
        creator: Callable building one fresh closable synchronous client per
            threaded worker.
        async_creator: Callable whose awaitable produces one fresh closable
            native async client.
        strategy: Native async or bounded threaded execution.
        script_name: Script credential paired with `api_key`.
        api_key: API key paired with `script_name`.
        session_token: JSON-RPC session token, exclusive with script
            credentials.
        cache_root: Schema cache directory override.
        policy: Cache freshness policy, or `None` for the default TTL policy.
        echo: Log provider calls and arguments on `sgalchemy.engine`.
        registry: Explicit Registry permanently paired with the engine.
        bind_default: When no Registry is supplied, select the process default
            if true or a fresh isolated Registry if false.
        max_workers: Threaded-only maximum concurrent provider calls; `None`
            selects one for a supplied connection and four for owned clients.
            A supplied connection must support the selected concurrency.

    Returns:
        Unprepared async engine that binds to its first-use event loop.

    Raises:
        ValueError: If the URL, strategy, sources, credentials, or options are
            inconsistent.
    """
    has_static = script_name is not None or api_key is not None or session_token is not None
    _validate_factory_inputs(
        strategy,
        url=url,
        connection=connection,
        sync_connection=sync_connection,
        creator=creator,
        async_creator=async_creator,
        has_static=has_static,
        max_workers=max_workers,
    )
    site = validate_site(url) if url is not None else None
    credentials = (
        validate_static_credentials(
            script_name=script_name,
            api_key=api_key,
            session_token=session_token,
        )
        if has_static
        else None
    )
    if credentials is not None:
        require_static_credential_https(cast('str', site))
    backend = (
        _create_native_backend(site, connection, credentials, async_creator)
        if strategy == 'native'
        else _create_threaded_backend(
            site,
            sync_connection,
            credentials,
            creator,
            max_workers=max_workers,
        )
    )
    if registry is None:
        registry = default_registry() if bind_default else Registry()
    return AsyncEngine(
        backend,
        cache_root=cache_root,
        policy=policy or TTLPolicy(),
        echo=echo,
        registry=registry,
    )
