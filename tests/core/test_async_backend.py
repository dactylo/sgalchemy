import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import suppress
from typing import Any, ParamSpec, TypeVar, cast

import pytest
from _fakes import _FakeAsyncShotgun, _FakeShotgun, _FakeShotgunFactory

from sgalchemy._core import _async_backend
from sgalchemy._core._async_backend import NativeAsyncBackend, ThreadedSyncBackend
from sgalchemy._core._engine_common import CallProgress
from sgalchemy._shotgun import AsyncClosableShotgunLike, ClosableShotgunLike, ShotgunLike

_P = ParamSpec('_P')
_R = TypeVar('_R')


def test_async_creator_cleanup_cancellation_propagates_from_acquisition() -> None:
    async def scenario() -> None:
        close_attempts: list[bool] = []

        class Partial:
            base_url = 'https://fake.example.com'

            async def close(self) -> None:
                close_attempts.append(True)
                raise asyncio.CancelledError

        async def make() -> Partial:
            return Partial()

        backend = NativeAsyncBackend(
            'https://fake.example.com',
            creator=make,
        )
        with pytest.raises(_async_backend._AsyncAcquisitionCancelled):
            await backend.call_with_progress('schema_read', (), {}, None)
        assert close_attempts == [True]

    asyncio.run(scenario())


def test_async_creator_cleanup_descriptor_error_does_not_mask_site_error() -> None:
    async def scenario() -> None:
        target = _FakeAsyncShotgun()
        target.base_url = 'https://other.example.com'
        close_lookups: list[bool] = []

        class RaisingSecondCloseLookup:
            def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - Delegate arbitrary client attributes around a failing second close lookup.
                if name == 'close':
                    close_lookups.append(True)
                    if len(close_lookups) > 1:
                        raise RuntimeError('close descriptor failed')
                return getattr(target, name)

        async def make() -> RaisingSecondCloseLookup:
            return RaisingSecondCloseLookup()

        backend = NativeAsyncBackend(
            'https://fake.example.com',
            creator=make,
        )
        with pytest.raises(ValueError, match='site'):
            await backend.call_with_progress('schema_read', (), {}, None)
        assert close_lookups == [True, True]

    asyncio.run(scenario())


def test_async_creator_malformed_site_closes_result_once() -> None:
    async def scenario() -> None:
        client = _FakeAsyncShotgun()
        client.base_url = 'http://[::1'

        async def make() -> AsyncClosableShotgunLike:
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        with pytest.raises(
            ValueError, match='async_creator returned a client for a different site'
        ) as excinfo:
            await backend.call_with_progress('schema_read', (), {}, None)
        close_calls = [call for call in client._calls if call[0] == 'close']
        assert (client._closed, close_calls, str(excinfo.value)) == (
            True,
            [('close', (), {})],
            'async_creator returned a client for a different site',
        )

    asyncio.run(scenario())


def test_async_creator_missing_surface_is_closed_by_backend() -> None:
    async def scenario() -> None:
        closed: list[bool] = []

        class Partial:
            base_url = 'https://fake.example.com'

            async def close(self) -> None:
                closed.append(True)

        async def make() -> Partial:
            return Partial()

        backend = NativeAsyncBackend(
            'https://fake.example.com',
            creator=make,
        )
        with pytest.raises(TypeError, match=r'surface|find'):
            await backend.call_with_progress('schema_read', (), {}, None)
        assert closed == [True]

    asyncio.run(scenario())


def test_async_creator_site_mismatch_closes_result() -> None:
    async def scenario() -> None:
        client = _FakeAsyncShotgun()
        client.base_url = 'https://other.example.com'

        async def make() -> AsyncClosableShotgunLike:
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        with pytest.raises(ValueError, match='site'):
            await backend.call_with_progress('schema_read', (), {}, None)
        assert client._closed

    asyncio.run(scenario())


@pytest.mark.parametrize('threaded', [False, True])
def test_call_progress_marks_resolved_entry_without_consuming_provider_kwargs(
    threaded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        observation = CallProgress()
        missing = CallProgress()
        sync_client = _FakeShotgun()
        async_client = _FakeAsyncShotgun()

        def provider(*, progress: str) -> str:
            assert observation.invoked.is_set()
            return progress

        async def async_provider(*, progress: str) -> str:
            return provider(progress=progress)

        monkeypatch.setattr(sync_client, 'schema_read', provider)
        monkeypatch.setattr(async_client, 'schema_read', async_provider)
        if threaded:
            backend = ThreadedSyncBackend(
                sync_client.base_url, connection=cast('ShotgunLike', sync_client), max_workers=1
            )
        else:
            backend = NativeAsyncBackend(
                async_client.base_url, connection=cast('AsyncClosableShotgunLike', async_client)
            )
        try:
            with pytest.raises(AttributeError):
                await backend.call_with_progress('missing_method', (), {}, missing)
            assert not missing.invoked.is_set()
            assert missing.dispatched.is_set() is threaded
            result = await backend.call_with_progress(
                'schema_read', (), {'progress': 'wire'}, observation
            )
            assert result == 'wire'
            assert observation.dispatched.is_set() is threaded
            assert not missing.invoked.is_set()
        finally:
            await backend.reset()

    asyncio.run(scenario())


@pytest.mark.parametrize('threaded', [False, True])
@pytest.mark.parametrize('invalid', [False, True])
def test_call_progress_excludes_async_acquisition_and_validation_failure(
    threaded: bool,
    invalid: bool,
) -> None:
    async def scenario() -> None:
        def creator() -> ClosableShotgunLike:
            if not invalid:
                raise RuntimeError('creation failed')
            client = _FakeShotgun()
            client.base_url = 'https://different.example.com'
            return cast('ClosableShotgunLike', client)

        async def async_creator() -> AsyncClosableShotgunLike:
            if not invalid:
                raise RuntimeError('creation failed')
            client = _FakeAsyncShotgun()
            client.base_url = 'https://different.example.com'
            return cast('AsyncClosableShotgunLike', client)

        if threaded:
            backend = ThreadedSyncBackend(
                'https://fake.example.com', creator=creator, max_workers=1
            )
        else:
            backend = NativeAsyncBackend('https://fake.example.com', creator=async_creator)
        progress = CallProgress()
        try:
            with pytest.raises((ValueError, RuntimeError)):
                await backend.call_with_progress('schema_read', (), {}, progress)
            assert not progress.invoked.is_set()
            assert progress.dispatched.is_set() is threaded
        finally:
            await backend.reset()

    asyncio.run(scenario())


def test_calls_arriving_during_reset_wait_and_use_fresh_client() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []

        async def make() -> AsyncClosableShotgunLike:
            client = _FakeAsyncShotgun()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        await backend.call_with_progress('schema_read', (), {}, None)
        first = created[0]
        entered = asyncio.Event()
        release = asyncio.Event()
        first._gate('find', entered, release)
        admitted = asyncio.ensure_future(
            backend.call_with_progress('find', ('Shot', []), {}, None)
        )
        await entered.wait()
        resetting = asyncio.ensure_future(backend.reset())
        await asyncio.sleep(0)
        arriving = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await asyncio.sleep(0)
        assert not arriving.done()
        release.set()
        await admitted
        await resetting
        await arriving
        assert first._closed is True
        assert len(created) == 2

    asyncio.run(scenario())


def test_cancelling_creation_suppressed_by_creator_closes_result_and_retries() -> None:
    async def scenario() -> None:
        attempts: list[int] = []
        created: list[_FakeAsyncShotgun] = []
        entered = asyncio.Event()
        gate = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            attempts.append(1)
            client = _FakeAsyncShotgun()
            created.append(client)
            if len(attempts) == 1:
                entered.set()
                with suppress(asyncio.CancelledError):
                    await gate.wait()
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        waiter = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await entered.wait()
        creating = backend._creating
        assert creating is not None
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert backend._connection is None
        assert created[0]._closed
        await backend.call_with_progress('schema_read', (), {}, None)
        assert len(attempts) == 2

    asyncio.run(scenario())


def test_cancelling_creation_task_publishes_nothing_and_next_call_retries() -> None:
    async def scenario() -> None:
        attempts: list[int] = []
        gate = asyncio.Event()
        started = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            attempts.append(1)
            started.set()
            if len(attempts) == 1:
                await gate.wait()
            return cast('AsyncClosableShotgunLike', _FakeAsyncShotgun())

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        waiter = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await started.wait()
        creating = backend._creating
        assert creating is not None
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert backend._connection is None
        await backend.call_with_progress('schema_read', (), {}, None)
        assert len(attempts) == 2

    asyncio.run(scenario())


def test_cancelling_one_waiter_leaves_creation_running() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []
        gate = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            await gate.wait()
            client = _FakeAsyncShotgun()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        doomed = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        survivor = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await asyncio.sleep(0)
        doomed.cancel()
        gate.set()
        await survivor
        assert len(created) == 1

    asyncio.run(scenario())


def test_cancelling_reset_caller_leaves_reset_completing() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []

        async def make() -> AsyncClosableShotgunLike:
            client = _FakeAsyncShotgun()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        await backend.call_with_progress('schema_read', (), {}, None)
        first = created[0]
        entered = asyncio.Event()
        release = asyncio.Event()
        first._gate('find', entered, release)
        admitted = asyncio.ensure_future(
            backend.call_with_progress('find', ('Shot', []), {}, None)
        )
        await entered.wait()
        reset_waiter = asyncio.ensure_future(backend.reset())
        await asyncio.sleep(0)
        reset_waiter.cancel()
        arriving = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await asyncio.sleep(0)
        assert not arriving.done()
        release.set()
        await admitted
        await arriving
        assert first._closed is True
        assert len(created) == 2

    asyncio.run(scenario())


def test_close_awaited_propagates_cancellation() -> None:
    async def scenario() -> None:
        class Client:
            async def close(self) -> None:
                raise asyncio.CancelledError('cleanup cancelled')

        with pytest.raises(asyncio.CancelledError, match='cleanup cancelled'):
            await _async_backend._close_awaited(Client())

    asyncio.run(scenario())


def test_creator_failure_after_waiter_cancellation_is_observed_and_retried() -> None:
    async def scenario() -> None:
        attempts = 0
        entered = asyncio.Event()
        finished = asyncio.Event()
        release = asyncio.Event()
        sentinel = RuntimeError('issuer down after waiter cancellation')

        async def make() -> AsyncClosableShotgunLike:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                entered.set()
                await release.wait()
                finished.set()
                raise sentinel
            return cast('AsyncClosableShotgunLike', _FakeAsyncShotgun())

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        reported: list[dict[str, Any]] = []
        observed: RuntimeError | None = None
        loop.set_exception_handler(lambda _loop, context: reported.append(context))
        try:
            waiter = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
            await entered.wait()
            waiter.cancel()
            with pytest.raises(_async_backend._AsyncAcquisitionCancelled):
                await waiter
            release.set()
            await finished.wait()
            await asyncio.sleep(0)
            try:
                await backend.call_with_progress('schema_read', (), {}, None)
            except RuntimeError as err:
                observed = err
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_handler)

        assert reported == []
        assert observed is sentinel
        await backend.call_with_progress('schema_read', (), {}, None)
        assert attempts == 2

    asyncio.run(scenario())


def test_failed_creation_reaches_waiters_and_later_calls_retry() -> None:
    async def scenario() -> None:
        attempts: list[int] = []

        async def make() -> AsyncClosableShotgunLike:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('issuer down')
            return cast('AsyncClosableShotgunLike', _FakeAsyncShotgun())

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        with pytest.raises(RuntimeError, match='issuer down'):
            await backend.call_with_progress('schema_read', (), {}, None)
        await backend.call_with_progress('schema_read', (), {}, None)
        assert len(attempts) == 2

    asyncio.run(scenario())


def test_native_creator_is_awaited_and_coalesced() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []
        gate = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            await gate.wait()
            client = _FakeAsyncShotgun()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        first = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        second = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(first, second)
        assert len(created) == 1

    asyncio.run(scenario())


def test_native_reset_closes_owned_client_and_backend_stays_usable() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []

        async def make() -> AsyncClosableShotgunLike:
            client = _FakeAsyncShotgun()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        await backend.call_with_progress('schema_read', (), {}, None)
        await backend.reset()
        assert created[0]._closed is True
        await backend.call_with_progress('schema_read', (), {}, None)
        assert len(created) == 2

    asyncio.run(scenario())


def test_native_reset_drains_creation_after_its_waiter_is_cancelled() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []
        gate = asyncio.Event()
        started = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            started.set()
            await gate.wait()
            client = _FakeAsyncShotgun()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        waiter = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        resetting = asyncio.ensure_future(backend.reset())
        await asyncio.sleep(0)
        assert not resetting.done()
        gate.set()
        await resetting
        assert created[0]._closed

    asyncio.run(scenario())


def test_native_reset_treats_cancelled_creation_child_as_drained() -> None:
    async def scenario() -> None:
        attempts = 0
        gate = asyncio.Event()
        started = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                started.set()
                await gate.wait()
            return cast('AsyncClosableShotgunLike', _FakeAsyncShotgun())

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        waiter = asyncio.ensure_future(backend.call_with_progress('schema_read', (), {}, None))
        await started.wait()
        creating = backend._creating
        assert creating is not None
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        resetting = asyncio.ensure_future(backend.reset())
        await asyncio.sleep(0)
        reset_task = backend._reset_gate._task
        assert reset_task is not None
        await asyncio.sleep(0)
        creating.cancel()
        await resetting
        await backend.call_with_progress('schema_read', (), {}, None)
        assert attempts == 2

    asyncio.run(scenario())


def test_reset_completion_event_resolves_even_when_close_fails() -> None:
    async def scenario() -> None:
        client = _FakeAsyncShotgun()
        client._close_errors.append(RuntimeError('close failed'))

        async def make() -> AsyncClosableShotgunLike:
            return cast('AsyncClosableShotgunLike', client)

        backend = NativeAsyncBackend('https://fake.example.com', creator=make)
        await backend.call_with_progress('schema_read', (), {}, None)
        with pytest.raises(RuntimeError, match='close failed'):
            await backend.reset()
        await backend.reset()
        assert client._closed is True

    asyncio.run(scenario())


def test_sync_creator_result_raises_teaching_type_error_and_awaits_cleanup() -> None:
    async def scenario() -> None:
        created: list[_FakeAsyncShotgun] = []

        def wrong_form() -> _FakeAsyncShotgun:
            client = _FakeAsyncShotgun()
            created.append(client)
            return client

        backend = NativeAsyncBackend(
            'https://fake.example.com',
            creator=wrong_form,
        )
        with pytest.raises(TypeError, match='creator'):
            await backend.call_with_progress('schema_read', (), {}, None)
        assert created[0]._closed is True

    asyncio.run(scenario())


def test_threaded_cancellation_does_not_claim_sync_call_stopped(fake_sg: _FakeShotgun) -> None:
    async def scenario() -> None:
        entered = threading.Event()
        release = threading.Event()
        fake_sg._block_next('find', entered=entered, release=release)
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        task = None
        try:
            task = asyncio.create_task(
                backend.call_with_progress('find', ('Shot', [], ['id']), {}, None)
            )
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await backend.reset()  # drains the still-running sync call

    asyncio.run(scenario())


def test_threaded_creator_result_is_validated() -> None:
    async def scenario() -> None:
        factory = _FakeShotgunFactory()
        backend = ThreadedSyncBackend(
            'https://other.example.com', creator=cast('Any', factory), max_workers=1
        )
        try:
            with pytest.raises(ValueError, match='different site'):
                await backend.call_with_progress('schema_read', (), {}, None)
        finally:
            await backend.reset()
        assert factory._instances[0]._closed is True

    asyncio.run(scenario())


def test_threaded_creator_uses_one_connection_per_worker(
    fake_sg_factory: _FakeShotgunFactory,
) -> None:
    async def scenario() -> None:
        barrier = fake_sg_factory._block_workers('schema_read', parties=2)
        backend = ThreadedSyncBackend(
            'https://fake.example.com',
            creator=cast('Callable[[], ClosableShotgunLike]', fake_sg_factory),
            max_workers=2,
        )
        try:
            await asyncio.gather(
                backend.call_with_progress('schema_read', (), {}, None),
                backend.call_with_progress('schema_read', (), {}, None),
            )
            await asyncio.gather(
                backend.call_with_progress('schema_entity_read', (), {}, None),
                backend.call_with_progress('schema_entity_read', (), {}, None),
            )
        finally:
            barrier.abort()
            await backend.reset()
        assert len(fake_sg_factory._instances) == 2
        assert all(len(instance._thread_ids) == 1 for instance in fake_sg_factory._instances)
        assert all(instance._closed for instance in fake_sg_factory._instances)

    asyncio.run(scenario())


def test_threaded_progress_records_acceptance_before_awaiting(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        observation = CallProgress()
        executor = backend._ensure_executor()
        original = executor.submit

        def submit(
            operation: Callable[_P, _R],
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> Future[_R]:
            assert not observation.dispatched.is_set()
            result = original(operation, *args, **kwargs)
            assert not observation.dispatched.is_set()
            return result

        monkeypatch.setattr(executor, 'submit', submit)
        try:
            result = await backend.call_with_progress(
                'upload_thumbnail', ('Shot', 90, 'image.jpg'), {}, observation
            )
            assert result == 1
            assert observation.dispatched.is_set()
            assert observation.invoked.is_set()
        finally:
            await backend.reset()

    asyncio.run(scenario())


def test_threaded_rejected_submission_does_not_record_dispatch(
    fake_sg: _FakeShotgun,
) -> None:
    async def scenario() -> None:
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        backend._ensure_executor().shutdown()
        progress = CallProgress()
        try:
            with pytest.raises(RuntimeError, match='shutdown'):
                await backend.call_with_progress('schema_read', (), {}, progress)
            assert not progress.dispatched.is_set()
            assert not progress.invoked.is_set()
            assert fake_sg._calls == []
        finally:
            await backend.reset()

    asyncio.run(scenario())


def test_threaded_reset_cancellation_is_drained_and_retryable(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        await backend.call_with_progress('schema_read', (), {}, None)
        entered = threading.Event()
        release = threading.Event()
        release_observed = threading.Event()
        executor = vars(backend)['_executor']
        original_shutdown = executor.shutdown

        def blocking_shutdown(*, wait: bool, cancel_futures: bool = False) -> None:
            entered.set()
            if release.wait(timeout=2):
                release_observed.set()
            original_shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(executor, 'shutdown', blocking_shutdown)
        resetting = asyncio.create_task(backend.reset())
        assert await asyncio.to_thread(entered.wait, 2)
        arriving = asyncio.create_task(backend.call_with_progress('schema_read', (), {}, None))
        await asyncio.sleep(0)
        assert not arriving.done()
        resetting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resetting
        release.set()
        await arriving
        assert release_observed.is_set()

    asyncio.run(scenario())


def test_threaded_reset_closes_same_client_identity_once_per_attempt() -> None:
    async def scenario() -> None:
        client = _FakeShotgun()
        close_attempts = 0
        original_close = client.close

        def close() -> None:
            nonlocal close_attempts
            close_attempts += 1
            if close_attempts == 1:
                raise RuntimeError('close failed')
            original_close()

        client.close = close
        backend = ThreadedSyncBackend(
            'https://fake.example.com',
            creator=cast('Any', lambda: client),
            max_workers=1,
        )
        await backend.call_with_progress('schema_read', (), {}, None)
        with pytest.raises(RuntimeError, match='close failed'):
            await backend.reset()
        await backend.call_with_progress('schema_read', (), {}, None)
        await backend.reset()
        assert close_attempts == 2

    asyncio.run(scenario())


def test_threaded_reset_closes_worker_clients_and_recreates_executor() -> None:
    async def scenario() -> None:
        factory = _FakeShotgunFactory()
        backend = ThreadedSyncBackend(
            'https://fake.example.com', creator=cast('Any', factory), max_workers=2
        )
        await backend.call_with_progress('schema_read', (), {}, None)
        await backend.reset()
        assert factory._instances[0]._closed is True
        await backend.call_with_progress('schema_read', (), {}, None)
        assert len(factory._instances) == 2

    asyncio.run(scenario())


def test_threaded_reset_never_closes_supplied_connection(fake_sg: _FakeShotgun) -> None:
    async def scenario() -> None:
        backend = ThreadedSyncBackend(
            'https://fake.example.com', connection=fake_sg, max_workers=1
        )
        await backend.call_with_progress('schema_read', (), {}, None)
        await backend.reset()
        assert fake_sg._closed is False
        await backend.call_with_progress('schema_read', (), {}, None)

    asyncio.run(scenario())


def test_threaded_reset_retains_failed_close_across_thread_id_reuse() -> None:
    async def scenario() -> None:
        factory = _FakeShotgunFactory()
        backend = ThreadedSyncBackend(
            'https://fake.example.com', creator=cast('Any', factory), max_workers=1
        )
        await backend.call_with_progress('schema_read', (), {}, None)
        failing = factory._instances[0]
        failing._close_errors.append(RuntimeError('close failed'))
        with pytest.raises(RuntimeError, match='close failed'):
            await backend.reset()
        assert failing in backend._retired_clients
        await backend.call_with_progress('schema_read', (), {}, None)
        assert failing in backend._retired_clients
        await backend.reset()
        assert failing._closed is True
        assert factory._instances[1]._closed is True

    asyncio.run(scenario())


def test_threaded_reset_shutdown_failure_allows_retry(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        await backend.call_with_progress('schema_read', (), {}, None)
        executor = vars(backend)['_executor']
        original_shutdown = executor.shutdown
        attempts = 0

        def failing_once_shutdown(*, wait: bool, cancel_futures: bool) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('shutdown failed')
            original_shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(executor, 'shutdown', failing_once_shutdown)
        with pytest.raises(RuntimeError, match='shutdown failed'):
            await backend.reset()
        await backend.call_with_progress('schema_read', (), {}, None)
        await backend.reset()
        await backend.call_with_progress('schema_read', (), {}, None)
        await backend.reset()
        assert attempts == 2

    asyncio.run(scenario())


def test_threaded_reset_survives_caller_cancellation() -> None:
    async def scenario() -> None:
        factory = _FakeShotgunFactory()
        backend = ThreadedSyncBackend(
            'https://fake.example.com', creator=cast('Any', factory), max_workers=2
        )
        entered = threading.Event()
        release = threading.Event()
        pending: list[asyncio.Future[Any]] = []
        try:
            await backend.call_with_progress('schema_read', (), {}, None)
            client = factory._instances[0]
            client._block_next('find', entered=entered, release=release)
            admitted = asyncio.ensure_future(
                backend.call_with_progress('find', ('Shot', []), {}, None)
            )
            pending.append(admitted)
            assert await asyncio.to_thread(entered.wait, 5)
            reset_waiter = asyncio.ensure_future(backend.reset())
            pending.append(reset_waiter)
            await asyncio.sleep(0)
            reset_waiter.cancel()
            arriving = asyncio.ensure_future(
                backend.call_with_progress('schema_read', (), {}, None)
            )
            pending.append(arriving)
            await asyncio.sleep(0)
            assert not arriving.done()
            release.set()
            await admitted
            await arriving
            assert client._closed is True
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)
            await backend.reset()

    asyncio.run(scenario())


def test_threaded_supplied_connection_is_serialized(fake_sg: _FakeShotgun) -> None:
    async def scenario() -> None:
        fake_sg._raise_on_concurrent = True
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        await asyncio.gather(
            backend.call_with_progress('schema_read', (), {}, None),
            backend.call_with_progress('schema_entity_read', (), {}, None),
        )
        await backend.reset()
        assert fake_sg._max_active_calls == 1
        assert fake_sg._closed is False

    asyncio.run(scenario())


def test_threaded_supplied_connection_supports_opted_in_concurrency(
    fake_sg: _FakeShotgun,
) -> None:
    async def scenario() -> None:
        backend = ThreadedSyncBackend(
            'https://fake.example.com', connection=fake_sg, max_workers=2
        )
        entered_one = threading.Event()
        release = threading.Event()
        entered_two = threading.Event()
        fake_sg._block_next('find', entered=entered_one, release=release)
        fake_sg._block_next('find', entered=entered_two, release=release)
        first = asyncio.ensure_future(backend.call_with_progress('find', ('Shot', []), {}, None))
        second = asyncio.ensure_future(backend.call_with_progress('find', ('Shot', []), {}, None))
        await asyncio.to_thread(entered_one.wait, 5)
        await asyncio.to_thread(entered_two.wait, 5)
        release.set()
        await asyncio.gather(first, second)
        assert fake_sg._max_active_calls == 2
        await backend.reset()
        assert fake_sg._closed is False

    asyncio.run(scenario())


def test_threaded_supports_cancellation_leaves_queued_probe_running(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        call_entered = threading.Event()
        call_release = threading.Event()
        fake_sg._block_next('find', entered=call_entered, release=call_release)
        backend = ThreadedSyncBackend(fake_sg.base_url, connection=fake_sg, max_workers=1)
        probe_entered = threading.Event()
        fake_type = cast('type[object]', type(fake_sg))

        def observed_getattribute(instance: object, name: str) -> object:
            if instance is fake_sg and name == 'schema_read':
                probe_entered.set()
            return object.__getattribute__(instance, name)

        running = None
        checking = None
        try:
            running = asyncio.create_task(
                backend.call_with_progress('find', ('Shot', [], ['id']), {}, None)
            )
            assert await asyncio.to_thread(call_entered.wait, 2)
            monkeypatch.setattr(fake_type, '__getattribute__', observed_getattribute)
            executor = vars(backend)['_executor']
            original_submit = executor.submit
            submitted = threading.Event()

            def observed_submit(
                fn: Callable[_P, _R],
                /,
                *args: _P.args,
                **kwargs: _P.kwargs,
            ) -> Future[_R]:
                future = original_submit(fn, *args, **kwargs)
                submitted.set()
                return future

            monkeypatch.setattr(executor, 'submit', observed_submit)
            checking = asyncio.create_task(backend.supports('schema_read'))
            assert await asyncio.to_thread(submitted.wait, 2)
            checking.cancel()
            with pytest.raises(asyncio.CancelledError):
                await checking
            call_release.set()
            await running
            assert await asyncio.to_thread(probe_entered.wait, 2)
        finally:
            call_release.set()
            if checking is not None:
                if not checking.done():
                    checking.cancel()
                await asyncio.gather(checking, return_exceptions=True)
            try:
                if running is not None:
                    await running
            finally:
                await backend.reset()

    asyncio.run(scenario())
