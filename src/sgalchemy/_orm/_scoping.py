"""Synchronous session factories, thread-local scoping, and scripting sugar."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, Unpack, cast

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from sgalchemy._core._expressions import Filter, FilterGroup, RawFilters
    from sgalchemy.schema._policy import SchemaCachePolicy

from sgalchemy._core._construction import validate_site
from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._engine_common import add_note_once
from sgalchemy._core._registry import (
    Entity,
    Registry,
    can_replace_default_registry,
    commit_default_registry_transition,
    install_entity_finder,
    release_default_registry_transition,
    reserve_default_registry_transition,
)
from sgalchemy._core._result import Result
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import SgalchemyError
from sgalchemy._orm._scope_common import SuppressOrdinaryFailure as _SuppressFailure
from sgalchemy._orm._scope_common import record_close_failure
from sgalchemy._orm._session import Session, commit_session
from sgalchemy._shotgun import ClosableShotgunLike, ShotgunLike

_COMMIT_FAILURE_CLOSE_NOTE = (
    'sgalchemy: the factory closed the Session and discarded its local retry queue; '
    'callers must explicitly restage any desired unsent work on a usable Session'
)
_DEFAULT_SCOPED: list[ScopedSession] = []
_DEFAULT_SCOPED_LOCK = threading.Lock()
_EntityT = TypeVar('_EntityT', bound=Entity)
_LOGGER = logging.getLogger('sgalchemy.session')
# Keep public scoping annotations resolvable through `typing.get_type_hints()`.
_RUNTIME_ANNOTATION_TYPES = (Callable, Engine, Entity, Result, ShotgunLike)


class _CommitOnSuccess:
    """Commit a clean body and always close one factory-created Session."""

    def __init__(self, factory: Callable[[], Session]) -> None:
        """Store the factory without allocating a Session."""
        self._factory = factory
        self._session: Session | None = None

    def __enter__(self) -> Session:
        """Create and return one independent Session."""
        session = self._factory()
        self._session = session
        return session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Commit a clean body, preserve the primary fault, and always close."""
        del exc_type, traceback
        session = self._session
        if session is None:
            raise RuntimeError('commit_on_success() entered without a Session')
        try:
            if exc_value is not None:
                _close_preserving(session, exc_value)
                return False
            try:
                commit_session(session, wrap_ordinary_failure=True)
            except BaseException as primary:
                _close_preserving(session, primary)
                add_note_once(primary, _COMMIT_FAILURE_CLOSE_NOTE)
                raise
            else:
                session.close()
        finally:
            self._session = None
        return False


class _ConnectOptions(TypedDict, total=False):
    """Describe optional keyword arguments accepted by `connect()`."""

    creator: Callable[[], ClosableShotgunLike] | None
    script_name: str | None
    api_key: str | None
    session_token: str | None
    cache_root: Path | None
    policy: SchemaCachePolicy | None
    echo: bool
    registry: Registry | None
    bind_default: bool
    serialize: bool | None


class _SessionLocal(threading.local):
    """Hold the current thread's scoped Session."""

    _session: Session | None

    def __init__(self) -> None:
        """Initialize one thread's empty Session slot."""
        self._session = None


class ScopedSession:
    """Provide one lazily created `Session` per thread.

    Obtain instances from `scoped_session()`; direct construction is
    unsupported.
    Calling the registry returns the current thread's `Session`, and `remove()`
    closes and discards it.
    """

    def __init__(self, factory: Callable[[], Session]) -> None:
        """Store the Session factory and create thread-local state."""
        self._factory = factory
        self._local = _SessionLocal()

    def __call__(self) -> Session:
        """Return the current thread's Session, creating it when absent."""
        local = cast('_SessionLocal', object.__getattribute__(self, '_local'))
        session = local._session  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if session is None:
            factory = cast('Callable[[], Session]', object.__getattribute__(self, '_factory'))
            session = factory()
            local._session = session  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return session

    if not TYPE_CHECKING:

        def __getattribute__(self, name: str) -> Any:  # noqa: ANN401 - Preserve fallback compatibility for undeclared Session attributes.
            """Resolve registry names and delegate missing public names to the current session."""
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                if name.startswith('_'):
                    raise
                return getattr(self(), name)

    def add(self, obj: Entity) -> None:
        """Forward an entity creation stage to the current Session.

        Args:
            obj: Schema-bound pending entity from `Entity.new()`.
        """
        self().add(obj)

    def close(self) -> None:
        """Forward local-state cleanup to the current Session."""
        self().close()

    def commit(self, batch_size: int | None = None) -> None:
        """Forward staged work to the current Session's commit.

        Args:
            batch_size: Maximum provider operations per batch request.
        """
        self().commit(batch_size)

    def delete(self, obj: Entity) -> None:
        """Forward an entity deletion stage to the current Session.

        Args:
            obj: Pending or persistent entity to transition.
        """
        self().delete(obj)

    @property
    def engine(self) -> Engine:
        """Return the engine bound to the current Session.

        Returns:
            Bound engine.
        """
        return self().engine

    def execute(self, stmt: Select[None]) -> Result | dict[str, Any]:
        """Forward a projection or summary statement to the current Session.

        Args:
            stmt: Projection or summary statement.

        Returns:
            Projection result or raw summary payload.
        """
        return self().execute(stmt)

    def expire(self, obj: Entity) -> None:
        """Forward entity expiration to the current Session.

        Args:
            obj: Entity to expire.
        """
        self().expire(obj)

    def expire_all(self) -> None:
        """Forward complete identity-map expiration to the current Session."""
        self().expire_all()

    def get(self, cls: type[_EntityT], entity_id: int) -> _EntityT | None:
        """Forward an identity lookup to the current Session.

        Args:
            cls: Reflected entity class.
            entity_id: Positive server identifier.

        Returns:
            Identity-mapped entity or `None` when absent.
        """
        return self().get(cls, entity_id)

    def merge_dict(self, raw: Mapping[str, Any]) -> Entity:
        """Forward raw entity adoption to the current Session.

        Args:
            raw: Entity dictionary containing canonical type and id.

        Returns:
            Identity-mapped entity.
        """
        return self().merge_dict(raw)

    def refresh(self, obj: Entity) -> None:
        """Forward entity refresh to the current Session.

        Args:
            obj: Session-owned persistent entity.
        """
        self().refresh(obj)

    def remove(self) -> None:
        """Close and remove the current thread's Session when present.

        Close restores local dirty snapshots and detaches owned entities.
        """
        local = cast('_SessionLocal', object.__getattribute__(self, '_local'))
        session = local._session  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if session is None:
            return
        try:
            session.close()
        finally:
            local._session = None  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def revive(self, obj: Entity) -> None:
        """Forward an entity revival stage to the current Session.

        Args:
            obj: Persistent entity to transition.
        """
        self().revive(obj)

    def rollback(self) -> None:
        """Forward local-state rollback to the current Session."""
        self().rollback()

    def scalars(self, stmt: Select[_EntityT]) -> list[_EntityT]:
        """Forward an entity statement to the current Session.

        Args:
            stmt: Entity-select statement.

        Returns:
            Identity-mapped entities in provider order.
        """
        return self().scalars(stmt)


@dataclass(frozen=True)
class SessionFactory:
    """Create independent `Session` instances with frozen defaults.

    Obtain instances from `sessionmaker()`; direct construction is unsupported.

    Attributes:
        engine: Engine bound to every produced `Session`.
        max_implicit_queries: Maximum implicit lazy-load queries per `Session`.
    """

    engine: Engine
    max_implicit_queries: int | None

    def __call__(self) -> Session:
        """Create one independent Session."""
        return Session(
            self.engine,
            max_implicit_queries=self.max_implicit_queries,
        )

    def commit_on_success(self) -> _CommitOnSuccess:
        """Commit one clean body once, then always close its Session.

        A body failure skips commit. Ordinary commit failures use `CommitError`
        with four-part failure-time evidence; `BaseException` interruptions
        propagate unchanged. Automatic close discards the local retry queue.
        The manager adds no transaction or provider-write rollback.

        Returns:
            Single-use commit-on-success context manager.
        """
        return _CommitOnSuccess(self)


def _default_session() -> Session:
    """Return the current thread's explicitly configured default Session."""
    with _DEFAULT_SCOPED_LOCK:
        if not _DEFAULT_SCOPED:
            raise SgalchemyError(
                'no default session is configured; call sgalchemy.connect(...) first'
            )
        scoped = _DEFAULT_SCOPED[0]
    return scoped()


def _find_entities(
    cls: type[Entity],
    conditions: tuple[Filter | FilterGroup | RawFilters, ...],
    eq_filters: dict[str, object],
) -> list[Entity]:
    """Execute model-level finder sugar through the current default Session."""
    equalities = tuple(getattr(cls, name) == value for name, value in eq_filters.items())
    return _default_session().scalars(select(cls).where(*conditions, *equalities))


def _publish_default_session(
    registry: Registry,
    default_bound: bool,
    replacement: ScopedSession,
) -> None:
    """Atomically publish the Registry and scoped Session pairing."""

    def publish_replacement() -> None:
        _DEFAULT_SCOPED[:] = [replacement]

    with _DEFAULT_SCOPED_LOCK:
        commit_default_registry_transition(
            registry,
            default_bound=default_bound,
            publish=publish_replacement,
        )


def _record_close_failure(primary: BaseException, secondary: BaseException) -> None:
    """Annotate a primary fault or log the secondary close failure."""
    record_close_failure(primary, secondary, 'Session', _LOGGER)


def _close_preserving(session: Session, primary: BaseException) -> None:
    """Keep the primary on close `Exception`; propagate cleanup or diagnostic `BaseException`."""
    suppression = _SuppressFailure()
    with suppression:
        session.close()
    if suppression.error is not None:
        _record_close_failure(primary, suppression.error)


def _select_connection_registry(
    url: str | None,
    connection: ShotgunLike | None,
    force: bool,
    engine_kwargs: _ConnectOptions,
    reserved_registry: Registry,
) -> tuple[Registry, bool]:
    """Select the connection Registry while the transition is reserved."""
    registry_arg = engine_kwargs.get('registry')
    implicit_default = registry_arg is None and bool(engine_kwargs.get('bind_default', True))
    default_bound = implicit_default or registry_arg is reserved_registry
    requested_site = connection.base_url if connection is not None else url
    if not default_bound:
        return (registry_arg if registry_arg is not None else Registry(), False)
    if requested_site is not None:
        requested_site = validate_site(requested_site)
    bound_key = reserved_registry.bound_key
    different_site = (
        bound_key is not None and requested_site is not None and requested_site != bound_key
    )
    if not different_site:
        return reserved_registry, True
    if implicit_default and force and can_replace_default_registry(reserved_registry):
        return Registry(), True
    raise SgalchemyError(
        'the default registry is already bound to another site; '
        'use engine.entities, registry=Registry(), or a separate process'
    )


def _admit_connection(
    url: str | None,
    connection: ShotgunLike | None,
    force: bool,
    engine_kwargs: _ConnectOptions,
) -> tuple[Registry, bool, object]:
    """Reserve and route one connection before provider I/O."""
    with _DEFAULT_SCOPED_LOCK:
        if _DEFAULT_SCOPED and not force:
            raise SgalchemyError(
                'a default session is already configured; pass force=True to replace it'
            )
        reservation_owner = object()
        try:
            reserved_registry = reserve_default_registry_transition(reservation_owner)
            target, default_bound = _select_connection_registry(
                url,
                connection,
                force,
                engine_kwargs,
                reserved_registry,
            )
        except BaseException:
            release_default_registry_transition(reservation_owner)
            raise
        engine_kwargs['registry'] = target
    return target, default_bound, reservation_owner


def install_entity_finder_callback() -> None:
    """Install model finder sugar at the explicit ORM import boundary."""
    install_entity_finder(_find_entities)


def scoped_session(factory: Callable[[], Session]) -> ScopedSession:
    """Create a thread-local Session registry with public passthrough.

    Args:
        factory: Zero-argument callable creating independent Sessions.

    Returns:
        Callable thread-local registry with public `Session` attribute
        passthrough and local-state-reverting `remove()` cleanup.
    """
    return ScopedSession(factory)


def sessionmaker(
    engine: Engine,
    *,
    max_implicit_queries: int | None = None,
) -> SessionFactory:
    """Create the narrow supported Session factory for `engine`.

    The signature rejects unsupported defaults at construction. The returned
    factory provides `commit_on_success()` but intentionally has no `begin()`.

    Args:
        engine: Engine bound to every produced Session.
        max_implicit_queries: Maximum implicit lazy-load queries per Session.

    Returns:
        Zero-argument Session factory with frozen lazy-load budget.
    """
    return SessionFactory(engine, max_implicit_queries)


def connect(
    url: str | None = None,
    *,
    connection: ShotgunLike | None = None,
    force: bool = False,
    **engine_kwargs: Unpack[_ConnectOptions],
) -> Session:
    """Create, prepare, and install the process default scoped Session.

    Args:
        url: Site URL for credential-owned provider clients.
        connection: Existing caller-owned provider connection.
        force: Replace the installed default after preparing the new engine.
        **engine_kwargs: Additional arguments forwarded to `create_engine()`.

    Returns:
        Current thread's Session from the installed scoped registry.

    Raises:
        SgalchemyError: If admission is already in flight, replacement is not
            authorized, or the requested Registry and site are incompatible.
        ValueError: If arguments forwarded to `create_engine()` are invalid.
        Exception: Provider or schema-cache failure while preparing the engine.
    """
    target, default_bound, reservation_owner = _admit_connection(
        url,
        connection,
        force,
        engine_kwargs,
    )
    try:
        engine = create_engine(url, connection=connection, **engine_kwargs)
        engine.load_schema()
        replacement = scoped_session(sessionmaker(engine))
        replacement_session = replacement()
        _publish_default_session(target, default_bound, replacement)
    except BaseException:
        release_default_registry_transition(reservation_owner)
        raise
    else:
        return replacement_session
