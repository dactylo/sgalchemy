import abc
import asyncio
import functools
import gc
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never, cast

import pytest
from _fakes import _FakeAsyncShotgun
from _support import (
    _AsyncCallback,
    _AsyncExecutionEnvironment,
    _AsyncSessionFactory,
    assert_disjoint_commit_outcome,
    isolated_async_engine,
)

from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._registry import Entity, Registry, entity_state
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _async_scoping, _async_session, _uow
from sgalchemy._orm._async_scoping import async_scoped_session, async_sessionmaker
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._orm._uow import RecoveryAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

_ASYNC_SCOPED_SESSION_METHODS = (
    'add',
    'close',
    'commit',
    'delete',
    'execute',
    'expire',
    'expire_all',
    'get',
    'load',
    'merge_dict',
    'refresh',
    'revive',
    'rollback',
    'scalars',
)
_AUTOMATIC_CLEANUP_WARNING = (
    'automatic task cleanup detached entities, reverted local edits, '
    'and discarded uncommitted work'
)


class _DiagnosticSignal(BaseException):
    pass


def _async_execution_error_body(
    environment: _AsyncExecutionEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    error: CommitError,
) -> _AsyncCallback[[AsyncSession], None]:
    engine, fake_async_sg, tmp_path = environment
    if route == 'multi_batch':
        original_batch = fake_async_sg.batch
        calls = 0

        async def fail_second_batch(requests: list[dict[str, Any]]) -> list[Any]:
            nonlocal calls
            if calls == 1:
                fake_async_sg._errors['batch'] = [error]
            calls += 1
            return await original_batch(requests)

        monkeypatch.setattr(fake_async_sg, 'batch', fail_second_batch)
        sequence = engine.entities.Sequence.new(code='first')
        dependent = engine.entities.Shot.new(code='second', sg_sequence=sequence)

        async def stage_batch(session: AsyncSession) -> None:
            session.add(sequence)
            session.add(dependent)

        return stage_batch
    if route == 'revive':
        fake_async_sg._errors['revive'] = [error]

        async def stage_revive(session: AsyncSession) -> None:
            shot = session.merge_dict({'type': 'Shot', 'id': 37, 'image': 'old'})
            entity_state(shot).deleted = True
            session.revive(shot)

        return stage_revive
    upload_path = tmp_path / 'async-existing-commit-error.jpg'
    upload_path.write_bytes(b'image')
    fake_async_sg._errors['upload_thumbnail'] = [error]

    async def stage_upload(session: AsyncSession) -> None:
        shot = session.merge_dict({'type': 'Shot', 'id': 37, 'image': 'old'})
        shot.image = upload_path

    return stage_upload


async def _make_factory(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> tuple[AsyncEngine, _AsyncSessionFactory]:
    engine = isolated_async_engine(fake_async_sg, tmp_path)
    await engine.load_schema()
    return engine, async_sessionmaker(engine)


async def _run_commit_on_success(
    factory: _AsyncSessionFactory,
    body: _AsyncCallback[[AsyncSession], None],
) -> None:
    async with factory.commit_on_success() as session:
        await body(session)


async def _run_direct_commit(
    factory: _AsyncSessionFactory,
    body: _AsyncCallback[[AsyncSession], None],
) -> None:
    session = factory()
    try:
        await body(session)
        await session.commit()
    finally:
        await session.close()


def test_async_close_failure_annotates_base_exception_once_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary = _DiagnosticSignal('primary')
    secondary = RuntimeError('secondary')

    with caplog.at_level('ERROR', logger='sgalchemy.session'):
        _async_scoping._record_close_failure(primary, secondary)
        _async_scoping._record_close_failure(primary, secondary)

    assert primary.__notes__ == [
        'sgalchemy: AsyncSession.close() also failed: RuntimeError: secondary'
    ]
    assert caplog.records == []


def test_async_commit_on_success_cancellation_stays_primary_when_close_fails(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        close_fault = RuntimeError('close failed')
        original_close = AsyncSession.close

        async def failing_close(session: AsyncSession) -> None:
            await original_close(session)
            raise close_fault

        monkeypatch.setattr(AsyncSession, 'close', failing_close)
        entered, _release = fake_async_sg._block('batch')
        pending = engine.entities.Shot.new(code='cancelled')
        observed: list[asyncio.CancelledError] = []

        async def worker() -> None:
            async def stage(session: AsyncSession) -> None:
                session.add(pending)

            try:
                await _run_commit_on_success(factory, stage)
            except asyncio.CancelledError as fault:
                observed.append(fault)
                raise

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task

        fault = excinfo.value
        assert observed[0] is fault
        assert entity_state(pending).session is None
        assert entity_state(pending).indeterminate
        assert any(
            note == 'sgalchemy: AsyncSession.close() also failed: RuntimeError: close failed'
            for note in fault.__notes__
        )
        assert any('explicitly restage' in note for note in fault.__notes__)
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_async_commit_on_success_close_failure_propagates_inside_outer_except(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        close_fault = RuntimeError('close failed')
        original_close = AsyncSession.close

        async def failing_close(session: AsyncSession) -> None:
            await original_close(session)
            raise close_fault

        monkeypatch.setattr(AsyncSession, 'close', failing_close)
        outer = ValueError('outer failure')

        async def close_inside_outer_except() -> None:
            try:
                raise outer
            except ValueError:
                async with factory.commit_on_success() as session:
                    pending = engine.entities.Shot.new(code='outer-close')
                    session.add(pending)

        with pytest.raises(RuntimeError, match='close failed'):
            await close_inside_outer_except()

        assert getattr(outer, '__notes__', []) == []
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_async_commit_on_success_success_inside_outer_except_adds_no_note(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        outer = RuntimeError('outer failure')

        try:
            raise outer
        except RuntimeError:
            async with factory.commit_on_success() as session:
                pending = engine.entities.Shot.new(code='outer-context')
                session.add(pending)

        assert pending.id is not None
        assert getattr(outer, '__notes__', []) == []
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_async_suppress_failure_preserves_timeout_cancellation() -> None:
    async def scenario() -> None:
        async def time_out_suppression() -> None:
            async with asyncio.timeout(0):
                with _async_scoping._SuppressFailure():
                    await asyncio.sleep(0)

        with pytest.raises(TimeoutError):
            await time_out_suppression()

    asyncio.run(scenario())


def test_automatic_cleanup_counts_same_entity_work_after_commit_cancellation(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))
        upload_path = tmp_path / 'unsent-image-secret.jpg'
        upload_path.write_bytes(b'image')
        entered, _release = fake_async_sg._block('batch')
        escaped: list[tuple[AsyncSession, Entity]] = []

        async def worker() -> None:
            session = scoped()
            shot = session.merge_dict({'type': 'Shot', 'id': 81, 'description': 'original-secret'})
            shot.description = 'submitted-update-secret'
            shot.image = upload_path
            session.delete(shot)
            escaped.append((session, shot))
            await session.commit()

        task = asyncio.create_task(worker(), name='same-entity-cancel-worker')
        await entered.wait()
        calls_before_cleanup = list(fake_async_sg._calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        session, shot = escaped[0]
        state = entity_state(shot)
        assert state.indeterminate
        assert state.session is None
        assert fake_async_sg._calls == calls_before_cleanup
        with pytest.raises(SgalchemyError, match='new AsyncSession'):
            await session.commit()

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith(_AUTOMATIC_CLEANUP_WARNING)
    ]
    assert warnings == [
        (
            f"{_AUTOMATIC_CLEANUP_WARNING}: task='same-entity-cancel-worker' cancelled=True "
            'dirty=1 pending=0 deletes=1 revives=0'
        )
    ]
    assert all(
        secret not in warnings[0]
        for secret in (
            'original-secret',
            'submitted-update-secret',
            'unsent-image-secret.jpg',
            str(tmp_path),
        )
    )


def test_automatic_cleanup_does_not_suppress_unretrieved_task_exception(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))
        loop = asyncio.get_running_loop()
        handled = asyncio.Event()
        contexts: list[dict[str, Any]] = []
        fault = RuntimeError('abandoned worker failure')

        def exception_handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            contexts.append(context)
            handled.set()

        async def worker() -> Never:
            shot = scoped().merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
            shot.code = 'new'
            raise fault

        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(exception_handler)
        try:
            task = asyncio.create_task(worker(), name='abandoned-worker')
            await asyncio.sleep(0)
            del task
            for _ in range(3):
                await asyncio.sleep(0)
                gc.collect()
            await asyncio.wait_for(handled.wait(), timeout=1)
        finally:
            loop.set_exception_handler(previous_handler)

        assert any(context.get('exception') is fault for context in contexts)

    asyncio.run(scenario())


def test_automatic_cleanup_failure_is_logged_after_registry_pop(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fault = RuntimeError('cleanup failed')

    def fail_cleanup(_recovery: RecoveryAdapter) -> None:
        raise fault

    monkeypatch.setattr(_async_session, 'discard_local_state', fail_cleanup)

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))

        async def worker() -> None:
            scoped()

        task = asyncio.create_task(worker(), name='cleanup-failure-worker')
        await task
        await asyncio.sleep(0)

        assert object.__getattribute__(scoped, '_sessions') == {}

    with caplog.at_level('ERROR', logger='sgalchemy.session'):
        asyncio.run(scenario())

    errors = [record for record in caplog.records if record.levelname == 'ERROR']
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert errors[0].exc_info[1] is fault
    assert 'automatic task cleanup failed after releasing task scope' in errors[0].message


def test_automatic_cleanup_warns_when_completed_task_is_cancelled(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))
        started = asyncio.Event()
        escaped: list[Any] = []

        async def worker() -> None:
            shot = scoped().merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
            shot.code = 'new'
            escaped.append(shot)
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(worker(), name='cancelled-worker')
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        assert entity_state(escaped[0]).session is None
        assert entity_state(escaped[0]).loaded['code'] == 'old'

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith(_AUTOMATIC_CLEANUP_WARNING)
    ]
    assert len(warnings) == 1
    assert "task='cancelled-worker' cancelled=True" in warnings[0]


def test_automatic_cleanup_warns_when_completed_task_raises(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))
        escaped: list[Any] = []

        async def worker() -> Never:
            shot = scoped().merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
            shot.code = 'new'
            escaped.append(shot)
            raise RuntimeError('worker failed')

        task = asyncio.create_task(worker(), name='failing-worker')
        with pytest.raises(RuntimeError, match='worker failed'):
            await task
        await asyncio.sleep(0)

        assert entity_state(escaped[0]).session is None
        assert entity_state(escaped[0]).loaded['code'] == 'old'

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith(_AUTOMATIC_CLEANUP_WARNING)
    ]
    assert len(warnings) == 1
    assert "task='failing-worker' cancelled=False" in warnings[0]


def test_automatic_cleanup_warns_with_exact_unsubmitted_counts(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> tuple[Entity, Entity, Entity, Entity]:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))

        async def worker() -> tuple[Entity, Entity, Entity, Entity]:
            session = scoped()
            dirty = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'old-secret'})
            dirty.code = 'new-secret'
            pending = engine.entities.Shot.new(code='pending-secret')
            session.add(pending)
            deleted = session.merge_dict({'type': 'Shot', 'id': 2, 'code': 'deleted-secret'})
            session.delete(deleted)
            revived = session.merge_dict({'type': 'Shot', 'id': 3, 'code': 'revived-secret'})
            entity_state(revived).deleted = True
            session.revive(revived)
            return dirty, pending, deleted, revived

        task = asyncio.create_task(worker(), name='count-worker')
        entities = await task
        await asyncio.sleep(0)

        for obj in entities:
            assert entity_state(obj).session is None
        assert entity_state(entities[0]).loaded['code'] == 'old-secret'
        return entities

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.levelname == 'WARNING' and record.message.startswith(_AUTOMATIC_CLEANUP_WARNING)
    ]
    assert warnings == [
        (
            f"{_AUTOMATIC_CLEANUP_WARNING}: task='count-worker' cancelled=False "
            'dirty=1 pending=1 deletes=1 revives=1'
        )
    ]
    assert all(
        secret not in warnings[0]
        for secret in (
            'old-secret',
            'new-secret',
            'pending-secret',
            'deleted-secret',
            'revived-secret',
        )
    )


def test_automatic_cleanup_without_unsubmitted_work_does_not_warn(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        scoped = async_scoped_session(async_sessionmaker(engine))

        async def worker() -> AsyncSession:
            return scoped()

        task = asyncio.create_task(worker(), name='clean-worker')
        await task
        await asyncio.sleep(0)

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    assert not any(
        record.message.startswith(_AUTOMATIC_CLEANUP_WARNING) for record in caplog.records
    )


def test_automatic_task_scope_rejects_custom_scopefunc() -> None:
    def factory() -> AsyncSession:
        raise AssertionError('factory must not run')

    with pytest.raises(TypeError, match='scopefunc'):
        cast('Any', async_scoped_session)(factory, scopefunc=object)


def test_borrower_defers_automatic_cleanup_until_operation_finishes(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        entered, release = fake_async_sg._block('find_one')
        scoped = async_scoped_session(async_sessionmaker(engine))
        borrowers: list[asyncio.Task[Any]] = []

        async def owner() -> tuple[AsyncSession, Any]:
            session = scoped()
            pending = engine.entities.Shot.new(code='borrowed-secret')
            session.add(pending)
            borrower = asyncio.create_task(
                session.get(engine.entities.Shot, 1),
                name='borrower',
            )
            borrowers.append(borrower)
            await entered.wait()
            return session, pending

        owner_task = asyncio.create_task(owner(), name='owner-worker')
        session, pending = await owner_task
        await asyncio.sleep(0)

        entries = object.__getattribute__(scoped, '_sessions')
        unit_of_work = object.__getattribute__(session, '_unit_of_work')
        assert owner_task not in entries
        assert entity_state(pending).session is session
        assert unit_of_work.new == (pending,)

        release.set()
        await borrowers[0]

        assert entries == {}
        assert entity_state(pending).session is None
        assert unit_of_work.new == ()

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [record.message for record in caplog.records if record.levelname == 'WARNING']
    misuse = [message for message in warnings if 'cross-task' in message]
    discarded = [message for message in warnings if message.startswith(_AUTOMATIC_CLEANUP_WARNING)]
    assert len(misuse) == 1
    assert "task='owner-worker' cancelled=False" in misuse[0]
    assert len(discarded) == 1
    assert "task='owner-worker' cancelled=False" in discarded[0]
    assert 'dirty=0 pending=1 deletes=0 revives=0' in discarded[0]
    assert 'borrowed-secret' not in '\n'.join(warnings)


@pytest.mark.parametrize('route', ['multi_batch', 'revive', 'upload'])
@pytest.mark.parametrize('through_factory', [False, True], ids=['direct', 'factory'])
def test_commit_error_from_execution_passes_through_unchanged(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
    through_factory: bool,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        partitions = ([object()], [object()], [object()], [object()])
        cause = RuntimeError('provider evidence')
        existing = CommitError(
            'existing commit error',
            committed=partitions[0],
            staged=partitions[1],
            indeterminate=partitions[2],
            withdrawn=partitions[3],
        )
        existing.__cause__ = cause
        existing.add_note('provider note')
        body = _async_execution_error_body(
            (engine, fake_async_sg, tmp_path),
            monkeypatch,
            route,
            existing,
        )
        run_commit = _run_commit_on_success if through_factory else _run_direct_commit

        with pytest.raises(CommitError) as excinfo:
            await run_commit(factory, body)

        assert excinfo.value is existing
        assert existing.__cause__ is cause
        assert existing.committed is partitions[0]
        assert existing.staged is partitions[1]
        assert existing.indeterminate is partitions[2]
        assert existing.withdrawn is partitions[3]
        expected_notes = ('provider note',)
        if through_factory:
            expected_notes += (_async_scoping._COMMIT_FAILURE_CLOSE_NOTE,)
        assert tuple(existing.__notes__) == expected_notes
        if route == 'upload':
            names = [name for name, _args, _kwargs in fake_async_sg._calls]
            assert names.count('upload_thumbnail') == 1
            assert 'find' not in names

    asyncio.run(scenario())


@pytest.mark.parametrize('exception_type', [RuntimeError, KeyboardInterrupt])
def test_commit_on_success_body_failure_skips_commit_and_stays_primary(
    exception_type: type[BaseException],
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = exception_type('body failed')
        pending = engine.entities.Shot.new(code='discarded')
        close_calls: list[AsyncSession] = []
        original_close = AsyncSession.close

        async def tracking_close(session: AsyncSession) -> None:
            close_calls.append(session)
            await original_close(session)

        async def fail_body(session: AsyncSession) -> None:
            session.add(pending)
            raise fault

        monkeypatch.setattr(AsyncSession, 'close', tracking_close)
        with pytest.raises(exception_type) as excinfo:
            await _run_commit_on_success(factory, fail_body)

        assert excinfo.value is fault
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 0
        assert len(close_calls) == 1
        assert entity_state(pending).session is None
        assert not entity_state(pending).dirty

    asyncio.run(scenario())


def test_commit_on_success_body_failure_stays_primary_when_close_fails(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        body_fault = RuntimeError('body failed')
        close_fault = RuntimeError('close failed')
        original_close = AsyncSession.close
        close_calls = 0

        async def failing_close(session: AsyncSession) -> None:
            nonlocal close_calls
            close_calls += 1
            await original_close(session)
            raise close_fault

        monkeypatch.setattr(AsyncSession, 'close', failing_close)
        pending = engine.entities.Shot.new(code='discarded')

        async def fail_body(session: AsyncSession) -> None:
            session.add(pending)
            raise body_fault

        with pytest.raises(RuntimeError) as excinfo:
            await _run_commit_on_success(factory, fail_body)

        assert excinfo.value is body_fault
        assert close_calls == 1
        assert entity_state(pending).session is None
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 0
        assert any('close failed' in note for note in body_fault.__notes__)

    asyncio.run(scenario())


def test_commit_on_success_commit_base_exception_stays_raw_and_quarantines(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = SystemExit('commit interrupted')
        fake_async_sg._errors['batch'] = [fault]
        pending = engine.entities.Shot.new(code='interrupted')
        sessions: list[AsyncSession] = []

        async def stage(session: AsyncSession) -> None:
            sessions.append(session)
            session.add(pending)

        with pytest.raises(SystemExit) as excinfo:
            await _run_commit_on_success(factory, stage)

        assert excinfo.value is fault
        assert excinfo.value.__cause__ is None
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1
        assert entity_state(pending).session is None
        assert entity_state(pending).indeterminate
        assert object.__getattribute__(sessions[0], '_discarded_after_cancel')
        assert any('indeterminate' in note for note in fault.__notes__)
        assert any('explicitly restage' in note for note in fault.__notes__)

    asyncio.run(scenario())


def test_commit_on_success_commit_failure_stays_primary_when_close_fails(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        commit_fault = RuntimeError('commit failed')
        close_fault = RuntimeError('close failed')
        fake_async_sg._errors['batch'] = [commit_fault]
        original_close = AsyncSession.close
        close_calls = 0

        async def failing_close(session: AsyncSession) -> None:
            nonlocal close_calls
            close_calls += 1
            await original_close(session)
            raise close_fault

        monkeypatch.setattr(AsyncSession, 'close', failing_close)
        pending = engine.entities.Shot.new(code='unsent')

        async def stage(session: AsyncSession) -> None:
            session.add(pending)

        with pytest.raises(CommitError) as excinfo:
            await _run_commit_on_success(factory, stage)

        error = excinfo.value
        assert error.__cause__ is commit_fault
        assert close_calls == 1
        assert entity_state(pending).session is None
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1
        assert any('explicitly restage' in note for note in error.__notes__)
        assert any('close failed' in note for note in error.__notes__)

    asyncio.run(scenario())


def test_commit_on_success_commits_once_and_closes(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        close_calls: list[AsyncSession] = []
        original_close = AsyncSession.close

        async def tracking_close(session: AsyncSession) -> None:
            close_calls.append(session)
            await original_close(session)

        monkeypatch.setattr(AsyncSession, 'close', tracking_close)
        async with factory.commit_on_success() as session:
            pending = engine.entities.Shot.new(code='created')
            session.add(pending)

        assert pending.id is not None
        assert entity_state(pending).session is None
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1
        assert close_calls == [session]

    asyncio.run(scenario())


def test_commit_on_success_does_not_claim_to_undo_inner_commit(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = RuntimeError('body failed after explicit commit')
        pending = engine.entities.Shot.new(code='already committed')

        async def commit_then_fail(session: AsyncSession) -> None:
            session.add(pending)
            await session.commit()
            raise fault

        with pytest.raises(RuntimeError) as excinfo:
            await _run_commit_on_success(factory, commit_then_fail)

        assert excinfo.value is fault
        assert pending.id is not None
        assert entity_state(pending).session is None
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1
        assert not any('rollback' in note.lower() for note in getattr(fault, '__notes__', ()))

    asyncio.run(scenario())


def test_commit_on_success_keeps_multi_step_failure_partitions(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = RuntimeError('second batch fault')
        original_batch = fake_async_sg.batch
        calls = 0

        async def fail_second_batch(requests: list[dict[str, Any]]) -> list[Any]:
            nonlocal calls
            if calls == 1:
                fake_async_sg._errors['batch'] = [fault]
            calls += 1
            return await original_batch(requests)

        monkeypatch.setattr(fake_async_sg, 'batch', fail_second_batch)
        sequence = engine.entities.Sequence.new(code='first')
        dependent = engine.entities.Shot.new(code='second', sg_sequence=sequence)

        async def stage_dependency(session: AsyncSession) -> None:
            session.add(sequence)
            session.add(dependent)

        with pytest.raises(CommitError) as excinfo:
            await _run_commit_on_success(factory, stage_dependency)

        error = excinfo.value
        assert error.__cause__ is fault
        assert len(error.committed) == 1
        committed = error.committed[0]
        assert isinstance(committed, _uow._Op)
        assert committed.obj is sequence
        assert len(error.staged) == 1
        staged = error.staged[0]
        assert isinstance(staged, _uow._Op)
        assert staged.obj is dependent
        assert error.indeterminate == []
        assert error.withdrawn == []
        assert_disjoint_commit_outcome(error)
        assert calls == 2
        assert entity_state(sequence).session is None
        assert entity_state(dependent).session is None

    asyncio.run(scenario())


def test_commit_on_success_preserves_existing_commit_error(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        pending = engine.entities.Shot.new(code='existing evidence')
        cause = RuntimeError('provider evidence')
        existing = CommitError('existing commit error', staged=[pending])
        existing.__cause__ = cause
        attempts = 0

        async def fail_plan(_session_instance: AsyncSession) -> Never:
            nonlocal attempts
            attempts += 1
            raise existing

        monkeypatch.setattr(AsyncSession, '_plan_commit', fail_plan)

        async def stage(session: AsyncSession) -> None:
            session.add(pending)

        with pytest.raises(CommitError) as excinfo:
            await _run_commit_on_success(factory, stage)

        assert excinfo.value is existing
        assert existing.__cause__ is cause
        assert existing.committed == []
        assert existing.staged == [pending]
        assert existing.indeterminate == []
        assert existing.withdrawn == []
        assert attempts == 1
        assert entity_state(pending).session is None
        assert any('explicitly restage' in note for note in existing.__notes__)

    asyncio.run(scenario())


def test_commit_on_success_propagates_close_failure_when_it_is_sole_failure(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = RuntimeError('close failed')
        original_close = AsyncSession.close
        close_calls = 0

        async def failing_close(session: AsyncSession) -> None:
            nonlocal close_calls
            close_calls += 1
            await original_close(session)
            raise fault

        monkeypatch.setattr(AsyncSession, 'close', failing_close)
        pending = engine.entities.Shot.new(code='committed before close')

        async def stage(session: AsyncSession) -> None:
            session.add(pending)

        with pytest.raises(RuntimeError) as excinfo:
            await _run_commit_on_success(factory, stage)

        assert excinfo.value is fault
        assert close_calls == 1
        assert pending.id is not None
        assert entity_state(pending).session is None
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_commit_on_success_wraps_schema_preparation_failure_without_replanning(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        registry = Registry()
        source = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'source',
            registry=registry,
            bind_default=False,
        )
        await source.load_schema()
        destination_sg = type(fake_async_sg)()
        destination = create_async_engine(
            connection=destination_sg,
            cache_root=tmp_path / 'destination',
            registry=registry,
            bind_default=False,
        )
        factory = async_sessionmaker(destination)
        pending = source.entities.Shot.new(code='cold staged intent')
        fault = RuntimeError('schema preparation failed')
        destination_sg._errors['schema_read'] = [fault]

        async def stage(session: AsyncSession) -> None:
            session.add(pending)

        with pytest.raises(CommitError) as excinfo:
            await _run_commit_on_success(factory, stage)

        error = excinfo.value
        assert error.__cause__ is fault
        assert error.committed == []
        assert error.staged == [pending]
        assert error.indeterminate == []
        assert error.withdrawn == []
        assert_disjoint_commit_outcome(error)
        assert [name for name, _args, _kwargs in destination_sg._calls].count('schema_read') == 1
        assert not any(name == 'batch' for name, _args, _kwargs in destination_sg._calls)
        assert entity_state(pending).session is None

    asyncio.run(scenario())


def test_commit_on_success_wraps_sole_batch_failure_reports_unsent(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = RuntimeError('serialize')

        def fail_serialization(*_args: object) -> Never:
            raise fault

        monkeypatch.setattr(_async_session, 'serialize_chunk', fail_serialization)
        pending = engine.entities.Shot.new(code='unsent')

        async def stage(session: AsyncSession) -> None:
            session.add(pending)

        with pytest.raises(CommitError) as excinfo:
            await _run_commit_on_success(factory, stage)

        error = excinfo.value
        assert error.__cause__ is fault
        assert error.__notes__.count(_uow._BATCH_UNSENT_NOTE) == 1
        assert _uow.SOLE_BATCH_STAGED_NOTE not in error.__notes__
        assert error.committed == []
        assert len(error.staged) == 1
        staged = error.staged[0]
        assert isinstance(staged, _uow._Op)
        assert staged.obj is pending
        assert error.indeterminate == []
        assert error.withdrawn == []
        assert_disjoint_commit_outcome(error)
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 0
        assert entity_state(pending).session is None
        assert not entity_state(pending).dirty
        assert any(
            'if this session remains open and usable, a later commit may retry' in note
            for note in error.__notes__
        )
        assert any('explicitly restage' in note for note in error.__notes__)

    asyncio.run(scenario())


def test_commit_on_success_wraps_sole_batch_failure_with_failure_time_evidence(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, factory = await _make_factory(fake_async_sg, tmp_path)
        fault = RuntimeError('sole batch fault')
        fake_async_sg._errors['batch'] = [fault]
        pending = engine.entities.Shot.new(code='unsent')

        async def stage(session: AsyncSession) -> None:
            session.add(pending)

        with pytest.raises(CommitError) as excinfo:
            await _run_commit_on_success(factory, stage)

        error = excinfo.value
        assert error.__cause__ is fault
        assert error.__notes__.count(_uow._BATCH_OUTCOME_UNKNOWN_NOTE) == 1
        assert _uow.SOLE_BATCH_STAGED_NOTE not in error.__notes__
        assert error.committed == []
        assert len(error.staged) == 1
        staged = error.staged[0]
        assert isinstance(staged, _uow._Op)
        assert staged.obj is pending
        assert error.indeterminate == []
        assert error.withdrawn == []
        assert_disjoint_commit_outcome(error)
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('batch') == 1
        assert entity_state(pending).session is None
        assert not entity_state(pending).dirty
        assert any(
            'if this session remains open and usable, a later commit may retry' in note
            for note in error.__notes__
        )
        assert any('explicitly restage' in note for note in error.__notes__)

    asyncio.run(scenario())


@pytest.mark.parametrize('cleanup', ['close', 'context', 'rollback'])
def test_explicit_cleanup_does_not_emit_automatic_warning(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    cleanup: str,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))

        async def worker() -> Entity:
            session = scoped()
            if cleanup == 'context':
                async with session:
                    shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
                    shot.code = 'new'
                return shot
            shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
            shot.code = 'new'
            if cleanup == 'close':
                await scoped.close()
                assert scoped() is session
            else:
                session.rollback()
            return shot

        task = asyncio.create_task(worker(), name=f'explicit-{cleanup}-worker')
        shot = await task
        await asyncio.sleep(0)

        assert entity_state(shot).session is None
        assert entity_state(shot).loaded['code'] == 'old'

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    assert not any(
        record.message.startswith(_AUTOMATIC_CLEANUP_WARNING) for record in caplog.records
    )


def test_remove_is_early_idempotent_and_recreation_gets_fresh_cleanup(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))

        async def worker() -> tuple[AsyncSession, Entity, AsyncSession, Entity]:
            first = scoped()
            first_pending = engine.entities.Shot.new(code='first')
            first.add(first_pending)
            current_task = asyncio.current_task()
            assert current_task is not None
            entries = object.__getattribute__(scoped, '_sessions')
            stale_callback = entries[current_task]._callback
            await scoped.remove()
            await scoped.remove()
            second = scoped()
            second_pending = engine.entities.Shot.new(code='second')
            second.add(second_pending)
            stale_callback(current_task)
            assert scoped() is second
            return first, first_pending, second, second_pending

        task = asyncio.create_task(worker(), name='recreated-worker')
        first, first_pending, second, second_pending = await task
        await asyncio.sleep(0)

        assert first is not second
        assert entity_state(first_pending).session is None
        assert entity_state(second_pending).session is None

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith(_AUTOMATIC_CLEANUP_WARNING)
    ]
    assert len(warnings) == 1
    assert "task='recreated-worker' cancelled=False" in warnings[0]
    assert 'dirty=0 pending=1 deletes=0 revives=0' in warnings[0]


def test_scoped_remove_forgets_before_awaiting_close() -> None:
    async def scenario() -> None:
        class ClosingSession:
            async def close(self) -> None:
                assert scoped() is not self

        sessions: list[ClosingSession] = []

        def factory() -> ClosingSession:
            session = ClosingSession()
            sessions.append(session)
            return session

        scoped = async_scoped_session(cast('Callable[[], AsyncSession]', factory))
        first = scoped()
        await scoped.remove()
        assert sessions == [first, scoped()]
        await scoped.remove()

    asyncio.run(scenario())


def test_scoped_session_dynamic_fallback_delegates_unknown_public_method(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def future_public_method(
        session: AsyncSession,
        marker: object,
    ) -> tuple[AsyncSession, object]:
        return session, marker

    monkeypatch.setattr(
        AsyncSession,
        'future_public_method',
        future_public_method,
        raising=False,
    )

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        scoped = async_scoped_session(async_sessionmaker(engine))
        marker = object()

        assert 'future_public_method' not in _async_scoping.AsyncScopedSession.__dict__
        delegated: Any = scoped
        assert await delegated.future_public_method(marker) == (scoped(), marker)

        await scoped.remove()
        await engine.dispose()

    asyncio.run(scenario())


def test_scoped_session_explicit_method_signatures_match_async_session() -> None:
    for name in _ASYNC_SCOPED_SESSION_METHODS:
        scoped_method = _async_scoping.AsyncScopedSession.__dict__[name]
        session_method = AsyncSession.__dict__[name]
        scoped_parameters = tuple(inspect.signature(scoped_method).parameters.values())[1:]
        session_parameters = tuple(inspect.signature(session_method).parameters.values())[1:]

        assert [parameter.name for parameter in scoped_parameters] == [
            parameter.name for parameter in session_parameters
        ]
        assert [parameter.kind for parameter in scoped_parameters] == [
            parameter.kind for parameter in session_parameters
        ]
        assert [parameter.default for parameter in scoped_parameters] == [
            parameter.default for parameter in session_parameters
        ]
        assert inspect.iscoroutinefunction(scoped_method) is inspect.iscoroutinefunction(
            session_method
        )


def test_scoped_session_requires_a_task_key() -> None:
    factory_called = False

    def factory() -> AsyncSession:
        nonlocal factory_called
        factory_called = True
        raise AssertionError

    scoped = async_scoped_session(factory)

    with pytest.raises(SgalchemyError, match='running asyncio task'):
        scoped()
    assert not factory_called


def test_scoped_session_supports_abc_and_singledispatch_without_allocation() -> None:
    def factory() -> AsyncSession:
        raise AssertionError('factory must not run')

    class Marker(abc.ABC):
        @abc.abstractmethod
        def mark(self) -> None:
            pass

    @functools.singledispatch
    def describe(_value: object) -> str:
        return 'object'

    @describe.register(_async_scoping.AsyncScopedSession)
    def describe_scoped(_value: _async_scoping.AsyncScopedSession) -> str:
        return 'scoped'

    scoped = async_scoped_session(factory)

    assert not isinstance(scoped, Marker)
    assert describe(scoped) == 'scoped'


def test_scoped_session_underscore_probes_do_not_allocate() -> None:
    factory_calls = 0

    def factory() -> AsyncSession:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError('factory must not run')

    scoped = async_scoped_session(factory)

    for name in ('_state', '_missing', '__missing__'):
        with pytest.raises(AttributeError) as excinfo:
            getattr(scoped, name)
        assert type(excinfo.value) is AttributeError
        assert not hasattr(scoped, name)
    assert scoped.__class__ is _async_scoping.AsyncScopedSession
    assert scoped._factory is factory
    assert set(vars(scoped)) == {'_factory', '_sessions'}
    assert factory_calls == 0


def test_scoped_sessions_are_task_local(fake_async_sg: _FakeAsyncShotgun, tmp_path: Path) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        scoped = async_scoped_session(async_sessionmaker(engine))
        release = asyncio.Event()
        sessions: list[AsyncSession] = []
        sessions_ready = asyncio.Event()

        async def identity() -> AsyncSession:
            first = scoped()
            assert scoped() is first
            assert scoped.engine is engine
            sessions.append(first)
            if len(sessions) == 2:
                sessions_ready.set()
            await release.wait()
            return first

        tasks = (asyncio.create_task(identity()), asyncio.create_task(identity()))
        await sessions_ready.wait()
        assert sessions[0] is not sessions[1]
        release.set()
        left, right = await asyncio.gather(*tasks)
        assert (left, right) == tuple(sessions)
        await asyncio.sleep(0)
        await engine.dispose()

    asyncio.run(scenario())


def test_sessionmaker_has_no_begin_and_probe_does_not_allocate_or_perform_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    engine = create_async_engine(
        connection=fake_async_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    factory = async_sessionmaker(engine)
    calls_before = list(fake_async_sg._calls)

    assert not hasattr(factory, 'begin')
    assert fake_async_sg._calls == calls_before


def test_sessionmaker_rejects_defaults_and_creates_independent_sessions(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    engine = create_async_engine(
        connection=fake_async_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    factory = async_sessionmaker(engine)

    first = factory()
    second = factory()

    assert first is not second
    assert first.engine is engine
    assert second.engine is engine
    assert first._state is not second._state
    with pytest.raises(TypeError):
        cast('Any', async_sessionmaker)(engine, unsupported=True)


def test_task_completion_reverts_and_detaches_without_remove(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        scoped = async_scoped_session(async_sessionmaker(engine))

        async def worker() -> Entity:
            shot = scoped().merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
            shot.code = 'new'
            return shot

        task = asyncio.create_task(worker(), name='dirty-worker')
        shot = await task
        await asyncio.sleep(0)

        assert entity_state(shot).session is None
        assert entity_state(shot).loaded['code'] == 'old'

    asyncio.run(scenario())
