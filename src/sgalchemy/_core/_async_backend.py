"""Adapt native async clients and synchronous worker calls."""

import asyncio
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, Literal, ParamSpec, Protocol, TypeVar

from sgalchemy._core._construction import (
    checked_async_client,
    checked_sync_client,
    ensure_awaitable,
)
from sgalchemy._core._engine_common import CallProgress
from sgalchemy._shotgun import (
    AsyncClosableShotgunLike,
    AsyncShotgunLike,
    ClosableShotgunLike,
    ShotgunLike,
)

_LOGGER = logging.getLogger('sgalchemy.engine')
_P = ParamSpec('_P')
_R = TypeVar('_R')


class ThreadedSyncBackend:
    """Run synchronous provider calls in a bounded worker executor.

    Each worker lazily retains its own created client; a supplied connection is
    shared by all workers. Cancelling an awaiter does not stop submitted work.
    Reset drains the executor, closes owned clients, and permits reuse.

    Attributes:
        mode: The threaded execution strategy.
        site: Canonical site identity required of created clients.
    """

    mode: Literal['native', 'threaded'] = 'threaded'

    def __init__(
        self,
        site: str,
        *,
        connection: ShotgunLike | None = None,
        creator: Callable[[], ClosableShotgunLike] | None = None,
        max_workers: int,
    ) -> None:
        """Configure one client source and a bounded worker width.

        Args:
            site: Canonical site identity expected from created clients.
            connection: Caller-owned client shared by workers and preserved
                across resets.
            creator: Factory for one engine-owned closable client per worker.
            max_workers: Positive maximum concurrent worker calls. A supplied
                connection must support that concurrency.

        Raises:
            ValueError: If both or neither client source is supplied, or
                `max_workers` is less than one.
        """
        if (connection is None) == (creator is None):
            raise ValueError('threaded backend requires exactly one connection source')
        if max_workers < 1:
            raise ValueError('max_workers must be at least 1')
        self.site = site
        self._connection = connection
        self._creator = creator
        self._executor: ThreadPoolExecutor | None = None
        self._local = threading.local()
        self._max_workers = max_workers
        self._owned: dict[int, ClosableShotgunLike] = {}
        self._owned_lock = threading.Lock()
        self._retired_clients: list[ClosableShotgunLike] = []
        self._reset_gate = _ResetGate()

    def _client(self) -> ShotgunLike:
        """Return the supplied client or this worker's owned client."""
        if self._connection is not None:
            return self._connection
        client: ClosableShotgunLike | None = getattr(self._local, 'client', None)
        if client is None:
            if self._creator is None:
                raise RuntimeError('threaded backend has no client creator')
            client = checked_sync_client(self._creator(), self.site)
            with self._owned_lock:
                self._owned[threading.get_ident()] = client
            self._local.client = client
        return client

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Return the current executor, creating one lazily after a reset."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix='sgalchemy',
            )
        return self._executor

    def _invoke(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - The selected sync provider method determines the result.
        """Run one sync method on its owning worker."""
        provider = getattr(self._client(), method)
        if progress is not None:
            progress.invoked.set()
        return provider(*args, **kwargs)

    async def _perform_reset(self) -> None:
        """Run one complete reset independently of any awaiting caller."""
        executor = self._executor
        if executor is not None:
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=False,
            )
            self._executor = None
        with self._owned_lock:
            candidates = [*self._retired_clients, *self._owned.values()]
            self._owned.clear()
            self._retired_clients.clear()
        to_close: list[ClosableShotgunLike] = []
        for client in candidates:
            if not any(known is client for known in to_close):
                to_close.append(client)
        failed: list[ClosableShotgunLike] = []
        failure: Exception | None = None
        for client in to_close:
            try:
                client.close()
            except Exception as err:
                _LOGGER.debug('threaded client close failed during reset', exc_info=True)
                failed.append(client)
                if failure is None:
                    failure = err
        if failed:
            with self._owned_lock:
                self._retired_clients.extend(failed)
        if failure is not None:
            raise failure

    async def _submit_with_progress(
        self,
        progress: CallProgress | None,
        operation: Callable[_P, _R],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        """Observe executor acceptance before awaiting shielded work."""
        await self._reset_gate._wait()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        future = self._ensure_executor().submit(operation, *args, **kwargs)
        if progress is not None:
            progress.dispatched.set()
        return await asyncio.shield(asyncio.wrap_future(future))

    async def _submit(
        self,
        operation: Callable[_P, _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        """Submit work while shielding it from waiter cancellation."""
        return await self._submit_with_progress(None, operation, *args, **kwargs)

    def _supports(self, method: str) -> bool:
        """Check one method on the worker-owned client."""
        return callable(getattr(self._client(), method, None))

    async def call_with_progress(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - The runtime-selected provider determines the result.
        """Observe executor acceptance and worker-side provider entry.

        Set `dispatched` after successful submission and before awaiting.
        Workers set `invoked` after client acquisition and method lookup.
        A worker can start before submission returns; the events do not impose
        worker scheduling order.

        Args:
            method: Provider method name.
            args: Positional provider arguments.
            kwargs: Keyword provider arguments.
            progress: Fresh per-call events, or None for an unobserved call.

        Returns:
            Provider result without wrapping.

        Raises:
            asyncio.CancelledError: If the awaiting task is cancelled;
                already-submitted work continues in its worker.
            Exception: Client acquisition or provider failures propagate
                unchanged.
        """
        return await self._submit_with_progress(
            progress,
            self._invoke,
            method,
            args,
            kwargs,
            progress,
        )

    async def reset(self) -> None:
        """Drain the executor, close owned worker clients, and stay usable.

        The reset body runs in its own task: cancelling a caller awaiting
        `reset()` leaves the reset running to completion, and the
        reset-completion event always resolves.

        Ordinary shutdown or client-close failures propagate and leave cleanup
        retryable. If multiple clients fail to close, the first failure is
        raised and all failed clients are retained for a later reset.
        """
        await self._reset_gate._run(self._perform_reset)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    async def supports(self, method: str) -> bool:
        """Check one sync method on a worker-owned client.

        Args:
            method: Public provider method name.

        Returns:
            Whether the provider exposes a callable attribute named `method`.

        Raises:
            asyncio.CancelledError: If the awaiting task is cancelled; an
                already-submitted capability probe continues in its worker.
            Exception: Any ordinary failure raised by client acquisition or
                provider attribute inspection is propagated unchanged.
        """
        return bool(await self._submit(self._supports, method))


class _AsyncAcquisitionCancelled(asyncio.CancelledError):
    """Signal cancellation before a provider method was invoked."""


class AsyncBackend(Protocol):
    """Define awaited provider dispatch and reusable resource reset.

    Calls return provider results without wrapping. Reset drains admitted work,
    closes owned clients, and leaves the backend usable; supplied connections
    remain caller-owned. Progress metadata belongs to the adapter and must not
    reach provider arguments.

    Attributes:
        mode: Whether execution uses native awaits or synchronous workers.
        site: Canonical site identity exposed to the owning engine.
    """

    mode: Literal['native', 'threaded']
    site: str

    async def call_with_progress(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - The runtime-selected provider determines the result.
        """Observe one call without passing observation metadata to the provider.

        Args:
            method: Provider method name.
            args: Positional provider arguments.
            kwargs: Keyword provider arguments.
            progress: Fresh per-call events, or None for an unobserved call.

        Returns:
            Provider result without wrapping.
        """
        ...

    async def reset(self) -> None:
        """Drain admitted work and close owned clients; stay usable."""
        ...

    async def supports(self, method: str) -> bool:
        """Return whether the selected provider exposes one public method."""
        ...


class NativeAsyncBackend:
    """Adapt one lazy async client to awaited engine operations.

    Use the adapter on one event loop. Concurrent acquisition shares one
    creator task. Reset drains admitted calls and closes owned clients while
    preserving supplied connections and permitting later lazy acquisition.

    Attributes:
        mode: The native execution strategy.
        site: Canonical site identity required of created clients.
    """

    mode: Literal['native', 'threaded'] = 'native'

    def __init__(
        self,
        site: str,
        *,
        connection: AsyncShotgunLike | None = None,
        creator: Callable[[], object] | None = None,
    ) -> None:
        """Store exactly one client source without performing I/O.

        Args:
            site: Canonical site identity expected from created clients.
            connection: Caller-owned client preserved across resets.
            creator: Factory whose result is validated as an awaitable yielding
                an engine-owned closable client on first acquisition.

        Raises:
            ValueError: If both or neither client source is supplied.
        """
        if (connection is None) == (creator is None):
            raise ValueError('native backend requires exactly one connection source')
        self.site = site
        self._connection = connection
        self._creator = creator
        self._owned_connection: AsyncClosableShotgunLike | None = None
        self._active_calls = 0
        self._creating: asyncio.Task[AsyncClosableShotgunLike] | None = None
        self._creating_generation = 0
        self._generation = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._reset_gate = _ResetGate()
        self._stale: list[AsyncClosableShotgunLike] = []

    async def _create(self) -> AsyncClosableShotgunLike:
        """Invoke the awaited creator once and classify its result."""
        if self._creator is None:
            raise RuntimeError('native async backend has no client creator')
        value: object = self._creator()
        try:
            awaitable = ensure_awaitable(value)
        except TypeError:
            await _close_awaited(value)
            raise
        raw = await awaitable
        creating = asyncio.current_task()
        if creating is not None and creating.cancelling():
            await _close_awaited(raw)
            raise asyncio.CancelledError
        try:
            client = checked_async_client(raw, self.site)
        except Exception:
            await _close_awaited(raw)
            raise
        if self._creating_generation == self._generation and self._connection is None:
            self._owned_connection = client
            self._connection = client
        else:
            self._stale.append(client)
        return client

    def _invalidate_generation(self) -> None:
        """Retire in-flight acquisition before scheduling reset work."""
        self._generation += 1

    @staticmethod
    def _observe_creation_failure(task: asyncio.Task[AsyncClosableShotgunLike]) -> None:
        """Retrieve an ordinary creator failure without consuming its result."""
        if task.cancelled():
            return
        task.exception()

    async def _client(self) -> AsyncShotgunLike:
        """Return, await creation of, or coalesce onto the current client."""
        if self._connection is not None:
            return self._connection
        if self._creating is None or self._creating.cancelled():
            self._creating_generation = self._generation
            self._creating = asyncio.get_running_loop().create_task(self._create())
            self._creating.add_done_callback(self._observe_creation_failure)
        creating = self._creating
        try:
            await asyncio.wait({creating})
            return creating.result()
        except asyncio.CancelledError as err:
            raise _AsyncAcquisitionCancelled(*err.args) from err
        except Exception:
            if self._creating is creating:
                self._creating = None
            raise

    async def _perform_reset(self) -> None:
        """Run one complete reset independently of any awaiting caller."""
        await self._idle.wait()
        creating = self._creating
        if creating is not None and not creating.done():
            try:
                with suppress(Exception):
                    await asyncio.shield(creating)
            except asyncio.CancelledError:
                reset_task = asyncio.current_task()
                if not creating.cancelled() or (
                    reset_task is not None and reset_task.cancelling()
                ):
                    raise
        self._creating = None
        owned = self._owned_connection
        if owned is not None:
            self._stale.append(owned)
            self._owned_connection = None
            self._connection = None
        remaining: list[AsyncClosableShotgunLike] = []
        failure: Exception | None = None
        for client in self._stale:
            try:
                await client.close()
            except Exception as err:
                _LOGGER.debug('native client close failed during reset', exc_info=True)
                remaining.append(client)
                if failure is None:
                    failure = err
        self._stale = remaining
        if failure is not None:
            raise failure

    async def call_with_progress(
        self,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        progress: CallProgress | None,
    ) -> Any:  # noqa: ANN401 - The runtime-selected provider determines the result.
        """Observe resolved provider entry after acquisition and reset waits.

        Set `invoked` immediately before calling the resolved method, without
        an intervening await. Native calls never set `dispatched`.

        Args:
            method: Provider method name.
            args: Positional provider arguments.
            kwargs: Keyword provider arguments.
            progress: Fresh per-call events, or None for an unobserved call.

        Returns:
            Provider result without wrapping.

        Raises:
            _AsyncAcquisitionCancelled: If cancellation interrupts lazy client
                acquisition.
            asyncio.CancelledError: If cancellation interrupts a reset wait or
                the provider call after acquisition.
            TypeError: If the provider method does not return an awaitable.
            Exception: Other client acquisition or provider failures propagate
                unchanged.
        """
        await self._reset_gate._wait()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        self._active_calls += 1
        self._idle.clear()
        try:
            client = await self._client()
            provider = getattr(client, method)
            if progress is not None:
                progress.invoked.set()
            result = provider(*args, **kwargs)
            if not inspect.isawaitable(result):
                raise TypeError(
                    f'native async provider method {method!r} did not return an awaitable'
                )
            return await result
        finally:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._idle.set()

    async def reset(self) -> None:
        """Drain admitted calls, close owned clients, and stay usable.

        The reset body runs in its own task: cancelling a caller awaiting
        `reset()` leaves the reset running to completion, and the
        reset-completion event always resolves. A close failure propagates to
        awaiters and leaves a retry possible.
        """
        await self._reset_gate._run(self._perform_reset, before=self._invalidate_generation)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    async def supports(self, method: str) -> bool:
        """Return whether the async client exposes one public method.

        Args:
            method: Public provider method name.

        Returns:
            Whether the provider exposes a callable attribute named `method`.

        Raises:
            _AsyncAcquisitionCancelled: If cancellation interrupts lazy client
                acquisition.
            asyncio.CancelledError: If cancellation occurs while waiting for an
                active reset.
            Exception: Any ordinary failure raised by client acquisition or
                provider attribute inspection is propagated unchanged.
        """
        await self._reset_gate._wait()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        self._active_calls += 1
        self._idle.clear()
        try:
            return callable(getattr(await self._client(), method, None))
        finally:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._idle.set()


class _ResetGate:
    """Coalesce shielded resets and hold new calls until completion."""

    def __init__(self) -> None:
        """Start with admission open and no reset task."""
        self._done = asyncio.Event()
        self._done.set()
        self._lock = asyncio.Lock()
        self._resetting = False
        self._task: asyncio.Task[None] | None = None

    async def _perform(self, operation: Callable[[], Awaitable[None]]) -> None:
        """Reopen admission after reset success, failure, or cancellation."""
        try:
            await operation()
        finally:
            self._resetting = False
            self._done.set()

    async def _run(
        self,
        operation: Callable[[], Awaitable[None]],
        *,
        before: Callable[[], None] | None = None,
    ) -> None:
        """Close admission and share a reset task shielded from waiter cancellation."""
        async with self._lock:
            task = self._task
            if task is None or task.done():
                self._resetting = True
                self._done.clear()
                if before is not None:
                    before()
                task = asyncio.get_running_loop().create_task(self._perform(operation))
                self._task = task
        await asyncio.shield(task)

    async def _wait(self) -> None:
        """Wait until reset completion permits another call."""
        while self._resetting:
            await self._done.wait()


async def _close_awaited(client: object) -> None:
    """Await one cleanup close, containing only ordinary failures."""
    try:
        close = getattr(client, 'close', None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:
        _LOGGER.exception('client cleanup close failed')
