"""Synchronous engine, backend gateway, and schema lifecycle."""

# pyright: reportPrivateUsage=false

import logging
import threading
import weakref
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._construction import (
    StaticCredentials,
    checked_sync_client,
    close_quietly,
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
from sgalchemy._shotgun import ClosableShotgunLike, QueryResult, ShotgunLike
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


class _ThreadSlot:
    """One thread's engine-owned client and its lifecycle accounting."""

    __slots__ = (
        '__weakref__',
        '_active',
        '_client',
        '_close_lock',
        '_finalizer',
        '_generation',
        '_retired',
    )

    def __init__(self, client: ClosableShotgunLike, generation: int) -> None:
        """Initialize lifecycle accounting for one engine-owned client."""
        self._active = 0
        self._client = client
        self._close_lock = threading.Lock()
        self._finalizer: weakref.finalize[..., _ThreadSlot] | None = None
        self._generation = generation
        self._retired = False


class Engine:
    """Manage synchronous provider clients and one fixed Registry pairing.

    Owned clients are created lazily per thread. Supplied connections remain
    caller-owned and are serialized when configured. Call `load_schema()`
    before accessing `schema` or `entities`.

    Disposal retires owned clients and permits later lazy recreation; it
    preserves the supplied connection and prepared Registry pairing.

    `site`, `connection`, `registry`, `schema`, `entities`, `call()`,
    `load_schema()`, and `dispose()` are supported engine surfaces, along with
    the factory and lifecycle behavior. `call_with_progress()`,
    `execute_call()`, `load_binding()`, `prepared_binding()`, and
    `mark_schema_stale()` are package-internal hooks.

    Attributes:
        site: Canonical site identity used for schema binding and caching.
    """

    def __init__(  # noqa: PLR0913 - Mirror the public factory's explicit options.
        self,
        *,
        site: str,
        url: str | None,
        credentials: StaticCredentials | None,
        connection: ShotgunLike | None,
        creator: Callable[[], ClosableShotgunLike] | None,
        cache_root: Path | None,
        policy: SchemaCachePolicy,
        echo: bool,
        registry: Registry,
        serialize: bool,
    ) -> None:
        """Store configuration already validated by `create_engine()`.

        Construction performs no provider or schema-cache I/O.

        Args:
            site: Canonical site identity used for binding and client checks.
            url: Provider construction URL, absent for a supplied connection.
            credentials: Validated static credentials, or `None` for another
                client source.
            connection: Caller-owned connection, or `None` for owned clients.
            creator: Lazy factory for a fresh closable client per thread.
            cache_root: Schema cache directory override, or `None` for
                environment and platform defaults.
            policy: Policy deciding whether a loaded disk cache is fresh.
            echo: Log provider calls and arguments on `sgalchemy.engine`.
            registry: Registry permanently paired with this engine.
            serialize: Serialize calls when a supplied connection is present.
        """
        self.site = site
        self._cache_root = cache_root
        self._call_lock = threading.Lock() if connection is not None and serialize else None
        self._creator = creator
        self._credentials = credentials
        self._disposal_lock = threading.Lock()
        self._echo = echo
        self._generation = 0
        self._local = threading.local()
        self._policy = policy
        self._retired_clients: dict[int, ClosableShotgunLike] = {}
        self._retired_slots: set[_ThreadSlot] = set()
        self._schema: dict[str, EntitySchema] | None = None
        self._schema_lock = threading.RLock()
        self._slot_refs: set[weakref.ref[_ThreadSlot]] = set()
        self._stale = False
        self._url = url
        self._wrapped = connection
        self._registry = registry

    def _close_finalized_client(self, client: ClosableShotgunLike) -> None:
        """Close a dead-thread client, retaining any failure for disposal."""
        try:
            client.close()
        except Exception:
            with self._disposal_lock:
                self._retired_clients[id(client)] = client
            _LOGGER.exception('client close failed; retained for disposal retry')

    def _close_retained_client(self, client: ClosableShotgunLike) -> None:
        """Claim and close one client retained after finalizer failure."""
        with self._disposal_lock:
            if self._retired_clients.get(id(client)) is not client:
                return
            del self._retired_clients[id(client)]
        try:
            client.close()
        except Exception:
            with self._disposal_lock:
                self._retired_clients[id(client)] = client
            raise

    def _close_slot(self, slot: _ThreadSlot, *, quiet: bool) -> None:
        """Close one owned slot exactly once, retaining it on any failure."""
        with slot._close_lock:
            finalizer = slot._finalizer
            if finalizer is None or not finalizer.alive:
                return
            if finalizer.detach() is None:
                return
            try:
                slot._client.close()
            except Exception:
                slot._finalizer = weakref.finalize(
                    slot,
                    Engine._finalize_client,
                    weakref.WeakMethod(self._close_finalized_client),
                    slot._client,
                )
                with self._disposal_lock:
                    self._retired_slots.add(slot)
                if quiet:
                    _LOGGER.exception('client close failed; retained for disposal retry')
                    return
                raise
            with self._disposal_lock:
                self._retired_slots.discard(slot)

    def _create_client(self) -> ClosableShotgunLike:
        """Construct one fresh engine-owned client for the calling thread."""
        if self._creator is not None:
            return checked_sync_client(self._creator(), self.site)
        if self._credentials is None:
            raise RuntimeError('engine has no owned-client construction source')
        import shotgun_api3  # noqa: PLC0415 - provider import must remain lazy

        constructor = cast('Callable[..., ClosableShotgunLike]', cast('Any', shotgun_api3).Shotgun)
        return constructor(cast('str', self._url), **self._credentials.provider_kwargs())

    @staticmethod
    def _finalize_client(
        close_ref: weakref.WeakMethod[Callable[[ClosableShotgunLike], None]],
        client: ClosableShotgunLike,
    ) -> None:
        """Close a dead-thread client through its engine when still alive."""
        close = close_ref()
        if close is None:
            close_quietly(client)
            return
        close(client)

    def _install_slot(self) -> _ThreadSlot:
        """Create, register, and publish a fresh slot for the calling thread."""
        with self._disposal_lock:
            generation = self._generation
        try:
            client = self._create_client()
        except Exception as err:
            if looks_like_auth_fault(err):
                add_note_once(err, AUTH_RECOVERY_NOTE)
            raise
        slot = _ThreadSlot(client, generation)
        slot._finalizer = weakref.finalize(
            slot,
            Engine._finalize_client,
            weakref.WeakMethod(self._close_finalized_client),
            client,
        )
        with self._disposal_lock:
            self._slot_refs.add(weakref.ref(slot))
        self._local.slot = slot
        return slot

    def _acquire(self) -> _ThreadSlot:
        """Return this thread's current-generation slot with one active call."""
        while True:
            slot: _ThreadSlot | None = getattr(self._local, 'slot', None)
            if slot is None:
                slot = self._install_slot()
            with self._disposal_lock:
                if slot._generation == self._generation and not slot._retired:
                    slot._active += 1
                    return slot
            self._local.slot = None
            self._close_slot(slot, quiet=True)

    def _invoke(
        self,
        client: ShotgunLike,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - The runtime-selected provider attribute determines the result.
        """Invoke one provider method without acquiring schema state."""
        provider = getattr(client, method)
        if self._echo:
            _LOGGER.info('%s args=%s kwargs=%s', method, args, kwargs)
        if progress is not None:
            progress.invoked.set()
        return provider(*args, **kwargs)

    def _prepared_schema(self) -> Mapping[str, EntitySchema] | None:
        """Return prepared schema state without performing I/O."""
        with self._schema_lock:
            if self._schema is None:
                return None
            return self._registry.schema_snapshot()

    def _release(self, slot: _ThreadSlot) -> None:
        """Finish one call, closing a retired slot after its last call."""
        with self._disposal_lock:
            slot._active -= 1
            deferred = slot._retired and slot._active == 0
        if deferred:
            self._close_slot(slot, quiet=True)

    @property
    def connection(self) -> ShotgunLike:
        """Return the supplied connection or this thread's owned client.

        Raw access bypasses engine locking and lifecycle accounting; quiesce
        raw use before `dispose()`.

        Returns:
            Connection used by the current thread.
        """
        if self._wrapped is not None:
            return self._wrapped
        slot: _ThreadSlot | None = getattr(self._local, 'slot', None)
        if slot is not None:
            with self._disposal_lock:
                stale = slot._generation != self._generation or slot._retired
            if stale:
                self._local.slot = None
                self._close_slot(slot, quiet=True)
                slot = None
        if slot is None:
            slot = self._install_slot()
        return slot._client

    def dispose(self) -> None:
        """Close and forget owned clients; the engine stays usable.

        Idle owned clients close immediately; clients with an active call are
        retired and closed by that call's completion. A close failure
        propagates and leaves disposal retryable for clients not yet closed.
        Supplied connections are never closed.
        """
        with self._disposal_lock:
            self._generation += 1
            retired_clients = list(self._retired_clients.values())
            retired = list(self._retired_slots)
            live = [slot for ref in self._slot_refs if (slot := ref()) is not None]
            idle: list[_ThreadSlot] = []
            for slot in live:
                if slot._active > 0:
                    slot._retired = True
                else:
                    idle.append(slot)
            self._slot_refs = {ref for ref in self._slot_refs if ref() is not None}
        for client in retired_clients:
            self._close_retained_client(client)
        for slot in [*retired, *idle]:
            self._close_slot(slot, quiet=False)

    def mark_schema_stale(self) -> None:
        """Mark prepared schema stale under the publication lock."""
        with self._schema_lock:
            self._stale = True

    def call_with_progress(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - Return the result of the runtime-selected provider attribute.
        """Execute a call, observing resolved provider entry after diagnostics.

        Args:
            method: Provider method name.
            args: Positional provider arguments.
            kwargs: Keyword provider arguments, excluding observation metadata.
            progress: Fresh per-call observation, or None for ordinary calls.

        Returns:
            Provider result without wrapping.
        """
        slot: _ThreadSlot | None = None
        wrapped = self._wrapped
        if wrapped is not None:
            client = wrapped
        else:
            try:
                slot = self._acquire()
            except Exception as err:
                if looks_like_auth_fault(err):
                    add_note_once(err, AUTH_RECOVERY_NOTE)
                raise
            client = slot._client
        try:
            if self._call_lock is None:
                return self._invoke(client, method, args, kwargs, progress)
            with self._call_lock:
                return self._invoke(client, method, args, kwargs, progress)
        except Exception as err:
            if looks_like_auth_fault(err):
                add_note_once(err, AUTH_RECOVERY_NOTE)
            if looks_like_schema_fault(method, err):
                _LOGGER.warning('possible schema drift, marking cache stale: %s', err)
                self.mark_schema_stale()
            raise
        finally:
            if slot is not None:
                self._release(slot)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - Preserve arbitrary provider arguments and results.
        """Execute one backend call through all runtime gateway behavior.

        Args:
            method: Provider method name.
            *args: Positional provider arguments.
            **kwargs: Keyword provider arguments.

        Returns:
            Provider result without wrapping.

        Raises:
            Exception: The original provider exception, unwrapped.
        """
        return self.call_with_progress(method, args, kwargs, None)

    def execute_call(self, call: FindCall | SummarizeCall) -> QueryResult:
        """Execute one compiled query call without changing its wire shape.

        Args:
            call: Pure compiler output.

        Returns:
            Provider query result.
        """
        method, args, kwargs = provider_call_arguments(call)
        return self.call(method, *args, **kwargs)

    def load_schema(self, *, force: bool = False) -> Mapping[str, EntitySchema]:
        """Prepare schema and registry state through an explicit I/O boundary.

        Args:
            force: Ignore prepared state and disk cache.

        Returns:
            Published entity schema mapping.

        Raises:
            SgalchemyError: If the provider returns malformed schema data or
                the Registry is already bound to a different site.
            Exception: Original provider or schema-cache exception, unwrapped.
        """
        with self._schema_lock:
            if self._schema is not None and not self._stale and not force:
                schemas = self._registry.schema_snapshot()
                if schemas is None:
                    raise RuntimeError('prepared engine Registry has no schema binding')
                return schemas
            path = cache_file(self.site, root=self._cache_root)
            cached = None if force or self._stale else load_cache(path)
            if cached is None or not self._policy.is_fresh(cached):
                info = getattr(self.connection, 'info', None)
                fields = self.call('schema_read')
                entities = self.call('schema_entity_read')
                raw_server_info = self.call('info') if callable(info) else None
                cached = SchemaCache(
                    fields=fields,
                    entities=entities,
                    fetched_at=datetime.now(UTC),
                    server_info=normalize_server_info(raw_server_info),
                )
                candidate = validated_schema(cached)
                store_cache(path, cached)
            else:
                candidate = build_schema(cached)
            self._registry.bind(candidate, key=self.site)
            self._schema = candidate
            self._stale = False
            schemas = self._registry.schema_snapshot()
            if schemas is None:
                raise RuntimeError('Registry bind did not publish schema')
            return schemas

    def load_binding(self, *, force: bool = False) -> RegistryBinding:
        """Prepare schema and return one complete Registry binding.

        Args:
            force: Ignore prepared state and disk cache.

        Returns:
            Complete immutable Registry binding.

        Raises:
            SgalchemyError: If the provider returns malformed schema data or
                the Registry is already bound to a different site.
            Exception: Original provider or schema-cache exception, unwrapped.
        """
        self.load_schema(force=force)
        binding = self._registry.binding_snapshot()
        if binding is None:
            raise RuntimeError('loaded schema did not publish a Registry binding')
        return binding

    def prepared_binding(self) -> RegistryBinding | None:
        """Return a prepared complete binding without performing I/O.

        Returns:
            Current Registry binding, or `None` before this engine prepares.
        """
        with self._schema_lock:
            if self._schema is None:
                return None
            return self._registry.binding_snapshot()

    @property
    def registry(self) -> Registry:
        """Return this engine's immutable Registry pairing.

        Returns:
            Registry selected during engine construction.
        """
        return self._registry

    @property
    def schema(self) -> Mapping[str, EntitySchema]:
        """Return prepared schema state without performing I/O.

        Returns:
            Published schema mapping.

        Raises:
            SgalchemyError: If the engine schema has not been loaded.
        """
        with self._schema_lock:
            if self._schema is None:
                raise SgalchemyError(
                    'engine schema is not loaded; call engine.load_schema() first'
                )
            schemas = self._registry.schema_snapshot()
            if schemas is None:
                raise RuntimeError('prepared engine Registry has no schema binding')
            return schemas

    @property
    def entities(self) -> EntityNamespace:
        """Return this engine's prepared entity namespace.

        Returns:
            Namespace resolving classes from this engine's registry.

        Raises:
            SgalchemyError: If the engine schema has not been loaded.
        """
        _ = self.schema
        return EntityNamespace(self.registry)


def create_engine(  # noqa: PLR0913 - Preserve the SQLAlchemy-style factory surface.
    url: str | None = None,
    *,
    connection: ShotgunLike | None = None,
    creator: Callable[[], ClosableShotgunLike] | None = None,
    script_name: str | None = None,
    api_key: str | None = None,
    session_token: str | None = None,
    cache_root: Path | None = None,
    policy: SchemaCachePolicy | None = None,
    echo: bool = False,
    registry: Registry | None = None,
    bind_default: bool = True,
    serialize: bool | None = None,
) -> Engine:
    """Create a lazy synchronous engine without performing I/O.

    Choose exactly one source: a supplied connection, a URL with static
    credentials, or a URL with a creator. A supplied connection cannot be
    combined with a URL. Static authentication requires HTTPS and either a
    nonempty session token or a complete script-name/API-key pair.

    A creator must return fresh closable clients for the same site and logical
    identity throughout the engine's lifetime. Call `load_schema()` before
    accessing reflected entities.

    Args:
        url: Complete site URL for engine-owned clients.
        connection: Existing caller-owned connection, never closed by the
            engine.
        creator: Callable building one fresh engine-owned client per thread.
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
        serialize: Supplied-connection-only serialization control. `None`
            enables serialization; false requires a client that supports
            concurrent shared-instance calls.

    Returns:
        Unprepared reusable engine.

    Raises:
        ValueError: If the URL, sources, credentials, or options are
            inconsistent.
    """
    has_static = script_name is not None or api_key is not None or session_token is not None
    sources = sum(
        1 for chosen in (connection is not None, creator is not None, has_static) if chosen
    )
    if connection is not None:
        if url is not None or sources != 1:
            raise ValueError('connection cannot be combined with url, credentials, or creator')
        site = validate_site(connection.base_url)
    else:
        if serialize is not None:
            raise ValueError('serialize is valid only with a supplied connection')
        if url is None or sources != 1:
            raise ValueError(
                'pass exactly one of connection, url with credentials, or url with creator'
            )
        site = validate_site(url)
    credentials = (
        validate_static_credentials(
            script_name=script_name, api_key=api_key, session_token=session_token
        )
        if has_static
        else None
    )
    if credentials is not None:
        require_static_credential_https(site)
    selected_registry = registry
    if selected_registry is None:
        selected_registry = default_registry() if bind_default else Registry()
    return Engine(
        site=site,
        url=url,
        credentials=credentials,
        connection=connection,
        creator=creator,
        cache_root=cache_root,
        policy=policy or TTLPolicy(),
        echo=echo,
        registry=selected_registry,
        serialize=serialize if serialize is not None else True,
    )
