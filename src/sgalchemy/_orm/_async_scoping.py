"""Asynchronous session factories and task-local scoping."""

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar, cast

from sgalchemy._core._async_engine import AsyncEngine
from sgalchemy._core._engine_common import add_note_once
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._registry import Entity
from sgalchemy._core._result import Result
from sgalchemy._core._statement import Select
from sgalchemy._errors import SgalchemyError
from sgalchemy._orm._async_session import AsyncSession, automatic_cleanup, commit_async_session
from sgalchemy._orm._scope_common import SuppressOrdinaryFailure as _SuppressFailure
from sgalchemy._orm._scope_common import record_close_failure

_COMMIT_FAILURE_CLOSE_NOTE = (
    'sgalchemy: the factory closed the AsyncSession and discarded its local retry queue; '
    'callers must explicitly restage any desired unsent work on a usable AsyncSession'
)
_EntityT = TypeVar('_EntityT', bound=Entity)
_LOGGER = logging.getLogger('sgalchemy.session')


class _AsyncCommitOnSuccess:
    """Commit a clean body and always close one factory-created AsyncSession."""

    def __init__(self, factory: Callable[[], AsyncSession]) -> None:
        """Store the factory without allocating an AsyncSession."""
        self._factory = factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        """Create and return one independent AsyncSession."""
        session = self._factory()
        self._session = session
        return session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Commit a clean body, preserve the primary fault, and always close."""
        del exc_type, traceback
        session = self._session
        if session is None:
            raise RuntimeError('commit_on_success() entered without an AsyncSession')
        try:
            if exc_value is not None:
                await _close_preserving(session, exc_value)
                return False
            try:
                await commit_async_session(session, wrap_ordinary_failure=True)
            except BaseException as primary:
                await _close_preserving(session, primary)
                add_note_once(primary, _COMMIT_FAILURE_CLOSE_NOTE)
                raise
            else:
                await session.close()
        finally:
            self._session = None
        return False


@dataclass(frozen=True)
class _ScopedEntry:
    """Store one exact task Session and its stable completion callback."""

    _session: AsyncSession
    _callback: Callable[[asyncio.Task[object]], None]


class AsyncScopedSession:
    """Provide one lazily created `AsyncSession` per running task.

    Obtain instances from `async_scoped_session()`; direct construction is
    unsupported. Calling the registry returns the current task's session, while
    `remove()` closes and discards it.
    """

    def __init__(self, factory: Callable[[], AsyncSession]) -> None:
        """Store the session factory and create an empty task registry."""
        self._factory = factory
        self._sessions: dict[asyncio.Task[object], _ScopedEntry] = {}

    def __call__(self) -> AsyncSession:
        """Return the current task's session, creating it when absent."""
        task: asyncio.Task[object] = object.__getattribute__(self, '_scope_task')()
        sessions = cast(
            'dict[asyncio.Task[object], _ScopedEntry]',
            object.__getattribute__(self, '_sessions'),
        )
        existing = sessions.get(task)
        if existing is not None:
            return existing._session  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        factory = cast(
            'Callable[[], AsyncSession]',
            object.__getattribute__(self, '_factory'),
        )
        session = factory()

        def cleanup(completed_task: asyncio.Task[object]) -> None:
            registry = cast(
                'dict[asyncio.Task[object], _ScopedEntry]',
                object.__getattribute__(self, '_sessions'),
            )
            if registry.get(completed_task) is not entry:
                return
            registry.pop(completed_task)
            automatic_cleanup(
                entry._session,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                task_name=completed_task.get_name(),
                cancelled=completed_task.cancelled(),
            )

        entry = _ScopedEntry(_session=session, _callback=cleanup)
        sessions[task] = entry
        task.add_done_callback(cleanup)
        return session

    if not TYPE_CHECKING:

        def __getattribute__(self, name: str) -> Any:  # noqa: ANN401 - Preserve fallback compatibility for undeclared AsyncSession attributes.
            """Resolve registry names and delegate missing public names to the current session."""
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                if name.startswith('_'):
                    raise
                return getattr(self(), name)

    def _scope_task(self) -> asyncio.Task[object]:
        """Return the current task or teach how to establish async scope."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None:
            raise SgalchemyError('async_scoped_session requires a running asyncio task')
        return task

    def add(self, obj: Entity) -> None:
        """Forward an entity creation stage to the current AsyncSession.

        Args:
            obj: Schema-bound pending entity from `Entity.new()`.
        """
        self().add(obj)

    async def close(self) -> None:
        """Forward local-state cleanup to the current AsyncSession."""
        await self().close()

    async def commit(self, batch_size: int | None = None) -> None:
        """Forward staged work to the current AsyncSession's commit.

        Args:
            batch_size: Maximum provider operations per batch request.
        """
        await self().commit(batch_size)

    def delete(self, obj: Entity) -> None:
        """Forward an entity deletion stage to the current AsyncSession.

        Args:
            obj: Pending or persistent entity to transition.
        """
        self().delete(obj)

    @property
    def engine(self) -> AsyncEngine:
        """Return the engine bound to the current AsyncSession.

        Returns:
            Bound async engine.
        """
        return self().engine

    async def execute(self, stmt: Select[None]) -> Result | dict[str, Any]:
        """Forward a projection or summary statement to the current AsyncSession.

        Args:
            stmt: Projection or summary statement.

        Returns:
            Projection result or raw summary payload.
        """
        return await self().execute(stmt)

    def expire(self, obj: Entity) -> None:
        """Forward entity expiration to the current AsyncSession.

        Args:
            obj: Entity to expire.
        """
        self().expire(obj)

    def expire_all(self) -> None:
        """Forward complete identity-map expiration to the current AsyncSession."""
        self().expire_all()

    async def get(self, cls: type[_EntityT], entity_id: int) -> _EntityT | None:
        """Forward an identity lookup to the current AsyncSession.

        Args:
            cls: Reflected entity class.
            entity_id: Positive server identifier.

        Returns:
            Identity-mapped entity or `None` when absent.
        """
        return await self().get(cls, entity_id)

    async def load(
        self,
        objects: Entity | Iterable[Entity],
        field: FieldExpr,
    ) -> None:
        """Forward explicit relationship loading to the current AsyncSession.

        Args:
            objects: One entity or an iterable of entities to load.
            field: Relationship field rooted at every object's entity type.
        """
        await self().load(objects, field)

    def merge_dict(self, raw: Mapping[str, Any]) -> Entity:
        """Forward raw entity adoption to the current AsyncSession.

        Args:
            raw: Entity dictionary containing canonical type and id.

        Returns:
            Identity-mapped entity.
        """
        return self().merge_dict(raw)

    async def refresh(self, obj: Entity) -> None:
        """Forward entity refresh to the current AsyncSession.

        Args:
            obj: Session-owned persistent entity.
        """
        await self().refresh(obj)

    async def remove(self) -> None:
        """Remove and close only the current task's session when present.

        Removal is early and idempotent. Close restores local dirty snapshots
        and detaches owned entities without provider I/O.
        """
        task: asyncio.Task[object] = object.__getattribute__(self, '_scope_task')()
        sessions = cast(
            'dict[asyncio.Task[object], _ScopedEntry]',
            object.__getattribute__(self, '_sessions'),
        )
        entry = sessions.pop(task, None)
        if entry is None:
            return
        task.remove_done_callback(entry._callback)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        await entry._session.close()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def revive(self, obj: Entity) -> None:
        """Forward an entity revival stage to the current AsyncSession.

        Args:
            obj: Persistent entity to transition.
        """
        self().revive(obj)

    def rollback(self) -> None:
        """Forward local-state rollback to the current AsyncSession."""
        self().rollback()

    async def scalars(self, stmt: Select[_EntityT]) -> list[_EntityT]:
        """Forward an entity statement to the current AsyncSession.

        Args:
            stmt: Entity-select statement.

        Returns:
            Identity-mapped entities in provider order.
        """
        return await self().scalars(stmt)


@dataclass(frozen=True)
class AsyncSessionFactory:
    """Create independent `AsyncSession` instances with a frozen engine.

    Obtain instances from `async_sessionmaker()`; direct construction is
    unsupported.

    Attributes:
        engine: Async engine bound to every produced `AsyncSession`.
    """

    engine: AsyncEngine

    def __call__(self) -> AsyncSession:
        """Create one independent AsyncSession."""
        return AsyncSession(self.engine)

    def commit_on_success(self) -> _AsyncCommitOnSuccess:
        """Commit one clean body once, then always close its AsyncSession.

        A body failure skips commit. Ordinary commit validation, planning, and
        execution failures use `CommitError` with four-part failure-time
        evidence. Session-access rejection before commit starts raises
        `SgalchemyError` unchanged; `BaseException` interruptions also
        propagate unchanged.

        Automatic close discards the local retry queue. The manager adds no
        transaction or provider-write rollback.

        Returns:
            Single-use commit-on-success context manager.
        """
        return _AsyncCommitOnSuccess(self)


def _record_close_failure(primary: BaseException, secondary: BaseException) -> None:
    """Annotate a primary fault or log the secondary close failure."""
    record_close_failure(primary, secondary, 'AsyncSession', _LOGGER)


async def _close_preserving(session: AsyncSession, primary: BaseException) -> None:
    """Keep the primary on close `Exception`; propagate cleanup or diagnostic `BaseException`."""
    suppression = _SuppressFailure()
    with suppression:
        await session.close()
    if suppression.error is not None:
        _record_close_failure(primary, suppression.error)


def async_scoped_session(
    factory: Callable[[], AsyncSession],
) -> AsyncScopedSession:
    """Create a current-task-only AsyncSession registry with passthrough.

    The registry accepts no custom scope function. Task completion pops the
    exact entry, then performs no-I/O cleanup after the event loop receives its
    next turn. That timing is best effort during abnormal shutdown, and
    ordering relative to awaiters of the owner is unspecified.

    Args:
        factory: Zero-argument callable creating independent AsyncSessions.

    Returns:
        Callable task-local registry with public `AsyncSession` passthrough,
        awaited `remove()`, and automatic completion cleanup.
    """
    return AsyncScopedSession(factory)


def async_sessionmaker(
    engine: AsyncEngine,
) -> AsyncSessionFactory:
    """Create the narrow supported AsyncSession factory for `engine`.

    The signature accepts no defaults. The returned factory provides
    `commit_on_success()` but intentionally has no `begin()`.

    Args:
        engine: AsyncEngine bound to every produced AsyncSession.

    Returns:
        Zero-argument AsyncSession factory.
    """
    return AsyncSessionFactory(engine)
