import asyncio
import copy
import datetime
import os
from pathlib import Path
from threading import Event, get_ident
from typing import TYPE_CHECKING, Any, Never, cast

import pytest
from _fakes import _FakeAsyncShotgun, _FakeShotgun
from _support import _preserve_signature, isolated_async_engine

from sgalchemy import _errors, inspect
from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._engine_common import CallProgress, EntityNamespace
from sgalchemy._core._registry import Entity, Registry, RegistryBinding, entity_state
from sgalchemy._core._statement import select
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _async_session, _uow
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._orm._uow import CommitOutcome
from sgalchemy._shotgun import (
    AsyncClosableShotgunLike,
    BatchItem,
    ClosableShotgunLike,
    WireRecord,
    WireRecords,
)

if TYPE_CHECKING:
    from sgalchemy._core._async_backend import NativeAsyncBackend, ThreadedSyncBackend


async def _cold_session(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> tuple[AsyncSession, AsyncEngine, _FakeAsyncShotgun]:
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
    return AsyncSession(destination), source, destination_sg


def _entities(engine: AsyncEngine) -> EntityNamespace:
    return engine.entities


def test_async_add_delete_revive_transitions_cancel_without_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='cancel-create')
        session.add(pending)
        session.delete(pending)
        assert entity_state(pending).session is None
        with pytest.raises(SgalchemyError, match='pending'):
            session.revive(pending)

        active = session.merge_dict({'type': 'Shot', 'id': 21, 'code': 'active'})
        session.delete(active)
        session.delete(active)
        session.revive(active)

        retired = session.merge_dict({'type': 'Shot', 'id': 22, 'code': 'retired'})
        entity_state(retired).deleted = True
        session.revive(retired)
        session.revive(retired)
        session.delete(retired)

        calls_before = len(fake_async_sg._calls)
        await session.commit()
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_ambiguous_upload_preserves_unrelated_work_and_error_snapshots(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        failed = session.merge_dict({'type': 'Shot', 'id': 90, 'code': 'failed'})
        unrelated = session.merge_dict({'type': 'Shot', 'id': 91, 'code': 'unrelated'})
        failed_path = tmp_path / 'async-failed-shared-state.jpg'
        unrelated_path = tmp_path / 'async-unrelated-shared-state.jpg'
        retry_path = tmp_path / 'async-retry-shared-state.jpg'
        for path in (failed_path, unrelated_path, retry_path):
            path.write_bytes(b'image')
        failed.image = failed_path
        unrelated.image = unrelated_path
        fault = RuntimeError('upload fault')
        fake_async_sg._errors['upload_thumbnail'] = [fault]
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 90, 'image': 'current-server'}]]

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        snapshots = (
            list(error.committed),
            list(error.staged),
            list(error.indeterminate),
            list(error.withdrawn),
        )
        assert error.committed == []
        assert error.staged == [(unrelated, 'image', str(unrelated_path))]
        assert error.indeterminate == [(failed, 'image', str(failed_path))]
        assert error.withdrawn == []
        assert failed.image == 'current-server'
        assert failed.code == 'failed'
        failed.description = 'still-usable'
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 91, 'image': 'unrelated-server'}],
            [
                {
                    'type': 'Shot',
                    'id': 90,
                    'code': 'failed',
                    'description': 'still-usable',
                    'image': 'refreshed-server',
                }
            ],
        ]

        await session.commit()
        await session.refresh(failed)
        failed.image = retry_path

        assert unrelated.image == 'unrelated-server'
        assert error.committed == snapshots[0]
        assert error.staged == snapshots[1]
        assert error.indeterminate == snapshots[2]
        assert error.withdrawn == snapshots[3]

    asyncio.run(scenario())


def test_async_canceled_pending_reference_rejected_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        sequence = _entities(engine).Sequence.new(code='canceled')
        shot = _entities(engine).Shot.new(code='dependent', sg_sequence=sequence)
        session.add(sequence)
        session.add(shot)
        session.delete(sequence)
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(SgalchemyError, match=r'pending relationship.*not staged'):
            await session.commit()
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_cold_commit_rejects_different_site_registry_without_schema_io(
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
        pending = _entities(source).Shot.new(code='bound-elsewhere')
        destination_sg = type(fake_async_sg)()
        destination_sg.base_url = 'https://other.example.com'
        destination = create_async_engine(
            connection=destination_sg,
            cache_root=tmp_path / 'destination',
            registry=registry,
            bind_default=False,
        )
        session = AsyncSession(destination)
        session.add(pending)

        with pytest.raises(SgalchemyError) as raised:
            await session.commit()
        assert str(raised.value) == (
            'registry is already bound to a different site (one site per Registry); '
            'pass registry=Registry() or bind_default=False to the engine factory '
            'for a second site'
        )

        assert destination_sg._calls == []

    asyncio.run(scenario())


def test_async_cold_local_preflight_rejects_missing_provenance_without_schema_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session, _source, destination_sg = await _cold_session(fake_async_sg, tmp_path)
        shot = session.merge_dict({'type': 'Shot', 'id': 41, 'description': 'before'})
        shot.description = 'after'
        entity_state(shot).dirty_types.pop('description')

        with pytest.raises(SgalchemyError, match='classification is missing'):
            await session.commit()

        assert destination_sg._calls == []

    asyncio.run(scenario())


def test_async_cold_local_preflight_rejects_pending_cycle_without_schema_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session, source, destination_sg = await _cold_session(fake_async_sg, tmp_path)
        shot = _entities(source).Shot.new(code='S')
        asset = _entities(source).Asset.new(code='A', shots=[shot])
        shot.assets = [asset]
        session.add(shot)
        session.add(asset)

        with pytest.raises(SgalchemyError, match='cycle'):
            await session.commit()

        assert destination_sg._calls == []

    asyncio.run(scenario())


def test_async_commit_cancellation_is_never_silently_retried(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='S')
        session.add(pending)
        entered, _release = fake_async_sg._block('batch')
        task = asyncio.create_task(session.commit())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        assert any('indeterminate' in note for note in excinfo.value.__notes__)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            await session.commit()
        with pytest.raises(SgalchemyError, match='indeterminate'):
            session.rollback()
        with pytest.raises(SgalchemyError, match='indeterminate'):
            _ = pending.code
        with pytest.raises(SgalchemyError, match='indeterminate'):
            pending.code = 'ambiguous'
        with pytest.raises(SgalchemyError, match='indeterminate'):
            pending.as_dict()
        with pytest.raises(SgalchemyError, match='indeterminate'):
            inspect(pending)
        assert repr(pending) == 'Shot(indeterminate)'
        await session.close()
        with pytest.raises(SgalchemyError, match='new AsyncSession'):
            await session.commit()
        replacement = AsyncSession(engine)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            replacement.add(pending)
        assert [name for name, _, _ in fake_async_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_async_confirmed_thumbnail_clear_needs_no_recovery_read(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict(
            {
                'type': 'Shot',
                'id': 51,
                'image': 'old-image',
                'filmstrip_image': 'old-filmstrip',
            }
        )
        state = entity_state(shot)
        state.refresh_required.update({'image', 'filmstrip_image'})
        state.unverified_writes.update({'image', 'filmstrip_image'})
        shot.image = None
        fake_async_sg._batch_results = [[{'type': 'Shot', 'id': 51}]]

        with caplog.at_level('WARNING', logger='sgalchemy.session'):
            await session.commit()

        assert state.loaded['image'] is None
        assert state.loaded['filmstrip_image'] is None
        assert state.refresh_required == set()
        assert state.unverified_writes == set()
        assert not [call for call in fake_async_sg._calls if call[0] == 'find']
        assert 'provider-response contract' in caplog.text

    asyncio.run(scenario())


def test_async_confirmed_upload_readback_failure_does_not_block_final_retirement(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 94})
        path = tmp_path / 'async-retire-after-read-failure.jpg'
        path.write_bytes(b'image')
        shot.image = path
        session.delete(shot)
        fake_async_sg._find_error = RuntimeError('readback fault')

        await session.commit()

        commit_calls = [
            call
            for call in fake_async_sg._calls
            if call[0] in {'upload_thumbnail', 'find', 'batch'}
        ]
        assert [call[0] for call in commit_calls] == ['upload_thumbnail', 'find', 'batch']
        assert entity_state(shot).deleted

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'dispose_before_commit',
    [False, True],
    ids=['constructed', 'disposed'],
)
def test_async_creator_cancellation_before_submission_keeps_commit_retriable(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    *,
    dispose_before_commit: bool,
) -> None:
    async def scenario() -> None:
        cache_root = tmp_path / 'creator-cancellation'
        source = create_async_engine(
            connection=fake_async_sg,
            cache_root=cache_root,
            registry=Registry(),
            bind_default=False,
        )
        await source.load_schema()
        created: list[_FakeAsyncShotgun] = []
        creator_started = asyncio.Event()
        release_creator = asyncio.Event()

        async def make() -> AsyncClosableShotgunLike:
            creator_started.set()
            await release_creator.wait()
            client = type(fake_async_sg)()
            created.append(client)
            return cast('AsyncClosableShotgunLike', client)

        engine = create_async_engine(
            fake_async_sg.base_url,
            async_creator=make,
            cache_root=cache_root,
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        assert created == []
        if dispose_before_commit:
            await engine.dispose()
        session = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='unsent')
        session.add(pending)
        committing = asyncio.create_task(session.commit())
        await creator_started.wait()
        committing.cancel()

        try:
            with pytest.raises(asyncio.CancelledError) as excinfo:
                await committing

            assert created == []
            assert all(
                'indeterminate' not in note for note in getattr(excinfo.value, '__notes__', ())
            )
            assert not entity_state(pending).indeterminate
            assert session.unit_of_work.new == (pending,)
            release_creator.set()
            await session.commit()
            assert pending.id is not None
            assert len(created) == 1
            assert [name for name, _, _ in created[0]._calls].count('batch') == 1
        finally:
            release_creator.set()
            await engine.dispose()

    asyncio.run(scenario())


def test_async_dependent_creates_and_persistent_links(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        project = session.merge_dict({'type': 'Project', 'id': 70, 'name': 'Demo'})
        sequence = _entities(engine).Sequence.new(code='SQ01')
        shot = _entities(engine).Shot.new(code='S1', project=project, sg_sequence=sequence)
        session.add(sequence)
        session.add(shot)
        await session.commit()
        batches = [call for call in fake_async_sg._calls if call[0] == 'batch']
        assert len(batches) == 2
        shot_data = batches[1][1][0][0]['data']
        assert shot_data['project'] == {'type': 'Project', 'id': 70}
        assert shot_data['sg_sequence'] == {'type': 'Sequence', 'id': sequence.id}

    asyncio.run(scenario())


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_async_detached_dirty_transition_commits_in_lifecycle_order(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    transition: str,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        source = AsyncSession(engine)
        obj = source.merge_dict({'type': 'Shot', 'id': 71, 'description': 'before'})
        await source.close()
        obj.description = 'after'
        entity_state(obj).deleted = transition == 'revive'
        session = AsyncSession(engine)
        getattr(session, transition)(obj)
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        commit_calls = fake_async_sg._calls[calls_before:]
        if transition == 'delete':
            assert [name for name, _, _ in commit_calls] == ['batch']
            assert [request['request_type'] for request in commit_calls[0][1][0]] == [
                'update',
                'delete',
            ]
        else:
            assert [name for name, _, _ in commit_calls] == ['revive', 'batch']
            assert [request['request_type'] for request in commit_calls[1][1][0]] == ['update']
        assert entity_state(obj).dirty == {}

    asyncio.run(scenario())


def test_async_dirty_delete_with_upload_defers_only_retirement(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        image_path = tmp_path / 'async-retiring-thumbnail.jpg'
        image_path.write_bytes(b'thumbnail')
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 73, 'description': 'before'})
        shot.description = 'after'
        shot.image = image_path
        session.delete(shot)
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 73, 'image': 'server-image'}]]
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        commit_calls = fake_async_sg._calls[calls_before:]
        assert [name for name, _, _ in commit_calls] == [
            'batch',
            'upload_thumbnail',
            'find',
            'batch',
        ]
        assert [request['request_type'] for request in commit_calls[0][1][0]] == ['update']
        assert commit_calls[1][1] == ('Shot', 73, str(image_path))
        assert [request['request_type'] for request in commit_calls[3][1][0]] == ['delete']

    asyncio.run(scenario())


def test_async_dirty_retired_object_never_loads_schema_for_recovery(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        retired = session.merge_dict({'type': 'Shot', 'id': 72, 'description': 'before'})
        session.delete(retired)
        await session.commit()
        retired.description = 'after'

        async def unexpected_load(*, force: bool = False) -> Never:
            del force
            raise AssertionError('dirty retired state must not load schema')

        monkeypatch.setattr(engine, 'load_binding', unexpected_load)
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(_errors.RetiredEntityWriteError, match=r'revive\(obj\).*rollback'):
            await session.commit()
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_failed_upload_current_value_is_loaded_but_unverified(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 89, 'image': 'old-image'})
        image_path = tmp_path / 'async-ambiguous-current.jpg'
        image_path.write_bytes(b'image')
        shot.image = image_path
        fault = RuntimeError('upload fault')
        fake_async_sg._errors['upload_thumbnail'] = [fault]
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 89, 'image': 'current-server'}]]

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        state = entity_state(shot)
        assert excinfo.value.__cause__ is fault
        assert excinfo.value.committed == []
        indeterminate = excinfo.value.indeterminate
        assert all(isinstance(item, tuple) for item in indeterminate)
        assert [item[1] for item in indeterminate if isinstance(item, tuple)] == ['image']
        assert state.loaded['image'] == 'current-server'
        assert state.refresh_required == set()
        assert state.unverified_writes == {'image'}
        assert [name for name, _, _ in fake_async_sg._calls].count('find') == 1

    asyncio.run(scenario())


def test_async_failed_upload_explicit_reassignment_submits_one_new_upload(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 93})
        first_path = tmp_path / 'async-failed-first.jpg'
        retry_path = tmp_path / 'async-explicit-new-intent.jpg'
        first_path.write_bytes(b'image')
        retry_path.write_bytes(b'image')
        shot.image = first_path
        fake_async_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

        with pytest.raises(CommitError):
            await session.commit()
        shot.image = retry_path
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 93, 'image': 'retry-server'}]]

        await session.commit()
        await session.commit()

        assert shot.image == 'retry-server'
        assert [name for name, _, _ in fake_async_sg._calls].count('upload_thumbnail') == 2

    asyncio.run(scenario())


def test_async_failed_upload_is_never_resent_by_plain_commit(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 92})
        path = tmp_path / 'async-never-resend.jpg'
        path.write_bytes(b'image')
        shot.image = path
        fake_async_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

        with pytest.raises(CommitError):
            await session.commit()
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert len(fake_async_sg._calls) == calls_before
        assert [name for name, _, _ in fake_async_sg._calls].count('upload_thumbnail') == 1

    asyncio.run(scenario())


def test_async_failed_upload_readback_preserves_callback_reassignment(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 96, 'image': 'old-image'})
        failed_path = tmp_path / 'async-ambiguous-callback.jpg'
        reassigned_path = tmp_path / 'async-callback-reassignment.jpg'
        failed_path.write_bytes(b'image')
        reassigned_path.write_bytes(b'image')
        shot.image = failed_path
        fault = RuntimeError('upload fault')
        fake_async_sg._errors['upload_thumbnail'] = [fault]

        async def find_with_reassignment(
            entity_type: str,
            filters: list[Any],
            fields: list[str] | None = None,
            **kwargs: object,
        ) -> WireRecords:
            await fake_async_sg._record('find', entity_type, filters, fields, **kwargs)
            shot.image = reassigned_path
            return [{'type': 'Shot', 'id': 96, 'image': 'current-server'}]

        monkeypatch.setattr(fake_async_sg, 'find', find_with_reassignment)

        with pytest.raises(CommitError):
            await session.commit()

        state = entity_state(shot)
        assert state.loaded['image'] == reassigned_path
        assert state.dirty == {'image': reassigned_path}
        assert state.refresh_required == set()
        assert state.unverified_writes == set()

    asyncio.run(scenario())


def test_async_failed_upload_withdraws_unsubmitted_same_owner_retirement(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    image_path = str(tmp_path / 'async-retiring-ambiguous.jpg')
    Path(image_path).write_bytes(b'image')

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        source = AsyncSession(engine)
        shot = source.merge_dict({'type': 'Shot', 'id': 77, 'code': 'retiring'})
        await source.close()
        shot.image = image_path
        session.delete(shot)
        fake_async_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        assert error.indeterminate == [(shot, 'image', image_path)]
        assert len(error.withdrawn) == 1
        assert getattr(error.withdrawn[0], 'request_type', None) == 'delete'
        calls_before = len(fake_async_sg._calls)
        await session.commit()
        assert len(fake_async_sg._calls) == calls_before
        retry_path = tmp_path / 'async-explicit-retry.jpg'
        retry_path.write_bytes(b'image')
        shot.image = retry_path
        await session.commit()
        assert [name for name, _, _ in fake_async_sg._calls].count('upload_thumbnail') == 2
        session.delete(shot)
        await session.commit()
        assert fake_async_sg._calls[-1][0] == 'batch'
        assert fake_async_sg._calls[-1][1][0][0]['request_type'] == 'delete'

    asyncio.run(scenario())


def test_async_false_revive_and_delete_results_are_successful_desired_states(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 75, 'code': 'active'})
        session.delete(shot)
        fake_async_sg._batch_results = [[False]]
        await session.commit()
        assert entity_state(shot).deleted

        async def already_active(entity_type: str, entity_id: int) -> bool:
            await fake_async_sg._record('revive', entity_type, entity_id)
            return False

        monkeypatch.setattr(fake_async_sg, 'revive', already_active)
        session.revive(shot)
        await session.commit()
        assert not entity_state(shot).deleted

    asyncio.run(scenario())


def test_async_indivisible_dirty_delete_rejects_too_small_batch_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 74, 'description': 'before'})
        shot.description = 'after'
        session.delete(shot)
        calls_before = len(fake_async_sg._calls)

        with pytest.raises(ValueError, match=r'batch_size.*minimum.*2'):
            await session.commit(batch_size=1)

        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_invalid_and_noop_transitions_leave_detached_entities_reusable(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)

        source = AsyncSession(engine)
        active = source.merge_dict({'type': 'Shot', 'id': 23, 'code': 'active'})
        await source.close()
        with pytest.raises(SgalchemyError, match='active entity cannot be revived'):
            session.revive(active)
        assert entity_state(active).session is None
        other = AsyncSession(engine)
        other.delete(active)
        assert entity_state(active).session is other

        source = AsyncSession(engine)
        retired = source.merge_dict({'type': 'Shot', 'id': 24, 'code': 'retired'})
        entity_state(retired).deleted = True
        await source.close()
        session.delete(retired)
        assert entity_state(retired).session is None
        other.revive(retired)
        assert entity_state(retired).session is other

    asyncio.run(scenario())


def test_async_later_upload_failure_reads_prior_confirmed_upload_once(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 88})
        image_path = tmp_path / 'async-confirmed-before-failure.jpg'
        filmstrip_path = tmp_path / 'async-ambiguous-after-success.jpg'
        image_path.write_bytes(b'image')
        filmstrip_path.write_bytes(b'image')
        shot.image = image_path
        shot.filmstrip_image = filmstrip_path
        fault = RuntimeError('filmstrip upload fault')
        fake_async_sg._errors['upload_filmstrip_thumbnail'] = [fault]
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 88,
                    'image': 'confirmed-server',
                    'filmstrip_image': 'current-server',
                }
            ]
        ]

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        assert error.__cause__ is fault
        assert all(isinstance(item, tuple) for item in error.committed)
        assert [item[1] for item in error.committed if isinstance(item, tuple)] == ['image']
        assert all(isinstance(item, tuple) for item in error.indeterminate)
        assert [item[1] for item in error.indeterminate if isinstance(item, tuple)] == [
            'filmstrip_image'
        ]
        state = entity_state(shot)
        assert state.loaded['image'] == 'confirmed-server'
        assert state.loaded['filmstrip_image'] == 'current-server'
        assert state.refresh_required == set()
        assert state.unverified_writes == {'filmstrip_image'}
        find_calls = [call for call in fake_async_sg._calls if call[0] == 'find']
        assert len(find_calls) == 1
        assert find_calls[0][1][:3] == (
            'Shot',
            [['id', 'in', [88]]],
            ['filmstrip_image', 'image'],
        )

    asyncio.run(scenario())


def test_async_later_upload_failure_reconciles_pending_readbacks_across_types(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 88})
        asset = session.merge_dict({'type': 'Asset', 'id': 89})
        filmstrip_path = tmp_path / 'async-confirmed-cross-type-filmstrip.jpg'
        image_path = tmp_path / 'async-failed-cross-type-image.jpg'
        filmstrip_path.write_bytes(b'image')
        image_path.write_bytes(b'image')
        shot.filmstrip_image = filmstrip_path
        asset.image = image_path
        fault = RuntimeError('later cross-type upload fault')
        fake_async_sg._errors['upload_thumbnail'] = [fault]
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 88, 'filmstrip_image': 'shot-server'}],
            [{'type': 'Asset', 'id': 89, 'image': 'asset-server'}],
        ]

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        assert all(isinstance(item, tuple) for item in error.committed)
        assert [item[1] for item in error.committed if isinstance(item, tuple)] == [
            'filmstrip_image'
        ]
        assert all(isinstance(item, tuple) for item in error.indeterminate)
        assert [item[1] for item in error.indeterminate if isinstance(item, tuple)] == ['image']
        find_calls = [call for call in fake_async_sg._calls if call[0] == 'find']
        assert [call[1][0] for call in find_calls] == ['Shot', 'Asset']
        assert shot.filmstrip_image == 'shot-server'
        assert asset.image == 'asset-server'
        assert entity_state(shot).unverified_writes == set()
        assert entity_state(asset).unverified_writes == {'image'}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'case',
    [
        (1, None, [], 0),
        (1, None, [True], 0),
        (2, None, [{'type': 'Shot', 'id': 9101, 'code': 'S0'}, True], 0),
        (2, 1, [], 1),
    ],
)
def test_async_malformed_successful_batch_is_not_automatically_retried(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    case: tuple[int, int | None, list[BatchItem], int],
) -> None:
    entity_count, batch_size, results, expected_staged = case

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        pending = [_entities(engine).Shot.new(code=f'S{index}') for index in range(entity_count)]
        for obj in pending:
            session.add(obj)
        fake_async_sg._batch_results = [results]
        with pytest.raises(CommitError) as excinfo:
            await session.commit(batch_size=batch_size)
        assert excinfo.value.committed == []
        submitted_count = entity_count - expected_staged
        submitted = pending[:submitted_count]
        unsubmitted = pending[submitted_count:]
        assert all(entity_state(obj).session is None for obj in submitted)
        assert all(entity_state(obj).access_guard is None for obj in submitted)
        assert all(entity_state(obj).session is session for obj in unsubmitted)
        assert len(excinfo.value.indeterminate) == submitted_count
        assert len(excinfo.value.staged) == expected_staged
        assert all(entity_state(obj).id is None for obj in pending[:submitted_count])
        assert not any(name == 'find' for name, _, _ in fake_async_sg._calls)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            _ = pending[0].code
        batches_before = [call for call in fake_async_sg._calls if call[0] == 'batch']
        await session.commit(batch_size=batch_size)
        batches_after = [call for call in fake_async_sg._calls if call[0] == 'batch']
        assert len(batches_after) == len(batches_before) + expected_staged
        if expected_staged:
            assert pending[-1].id is not None
        await session.close()
        assert all(entity_state(obj).session is None for obj in unsubmitted)

    asyncio.run(scenario())


def test_async_malformed_update_withdraws_followups_and_retries_only_unrelated(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    image_path = str(tmp_path / 'async-update-delete-ambiguous.jpg')
    Path(image_path).write_bytes(b'image')

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        source = AsyncSession(engine)
        affected = source.merge_dict({'type': 'Shot', 'id': 76, 'description': 'before'})
        await source.close()
        affected.description = 'after'
        affected.image = image_path
        session.delete(affected)
        independent = _entities(engine).Shot.new(code='async-independent')
        session.add(independent)
        fake_async_sg._batch_results = [[]]

        with pytest.raises(CommitError) as excinfo:
            await session.commit(batch_size=1)

        error = excinfo.value
        assert len(error.indeterminate) == 1
        assert getattr(error.indeterminate[0], 'request_type', None) == 'update'
        assert error.withdrawn[0] == (affected, 'image', image_path)
        assert getattr(error.withdrawn[1], 'request_type', None) == 'delete'
        assert len(error.staged) == 1
        assert getattr(error.staged[0], 'obj', None) is independent
        calls_before = len(fake_async_sg._calls)

        await session.commit(batch_size=1)

        retry_calls = fake_async_sg._calls[calls_before:]
        assert [name for name, _, _ in retry_calls] == ['batch']
        assert independent.id is not None
        await session.close()
        assert entity_state(affected).session is None

    asyncio.run(scenario())


def test_async_merge_delete_commit_prepares_schema_automatically(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 31})
        session.delete(shot)

        await session.commit()

        assert entity_state(shot).deleted
        names = [name for name, _args, _kwargs in fake_async_sg._calls]
        assert names.count('schema_read') == 1
        assert names.count('batch') == 1

    asyncio.run(scenario())


def test_async_missing_write_provenance_never_loads_schema_for_recovery(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 41, 'description': 'before'})
        shot.description = 'after'
        state = entity_state(shot)
        state.dirty_types.pop('description')

        async def unexpected_load(*, force: bool = False) -> Never:
            del force
            raise AssertionError('local corruption must not load schema')

        monkeypatch.setattr(engine, 'load_binding', unexpected_load)

        with pytest.raises(
            SgalchemyError,
            match='classification is missing',
        ):
            await session.commit()

        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 1
        assert [name for name, _, _ in fake_async_sg._calls].count('batch') == 0

    asyncio.run(scenario())


def test_async_multistep_failure_partitions_match_sync(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        for index in range(4):
            session.add(_entities(engine).Shot.new(code=f'S{index}'))
        fake_async_sg._fail_on_batch_index = 1
        with pytest.raises(CommitError) as excinfo:
            await session.commit(batch_size=2)
        assert len(excinfo.value.committed) == 2
        assert len(excinfo.value.staged) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('container', 'persistent'),
    [
        ('scalar', True),
        ('scalar', False),
        ('list', True),
        ('list', False),
    ],
)
def test_async_nested_foreign_registry_entities_rejected_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    container: str,
    persistent: bool,
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'one',
            registry=Registry(),
            bind_default=False,
        )
        other = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'two',
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        await other.load_schema()
        session = AsyncSession(engine)
        foreign_session = AsyncSession(other)
        if container == 'scalar':
            foreign = (
                foreign_session.merge_dict({'type': 'Sequence', 'id': 41, 'code': 'foreign'})
                if persistent
                else _entities(other).Sequence.new(code='foreign')
            )
            local = _entities(engine).Shot.new(code='local', sg_sequence=foreign)
        else:
            foreign = (
                foreign_session.merge_dict({'type': 'Shot', 'id': 42, 'code': 'foreign'})
                if persistent
                else _entities(other).Shot.new(code='foreign')
            )
            local = _entities(engine).Asset.new(code='local', shots=[foreign])
        session.add(local)
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(SgalchemyError, match='different registry'):
            await session.commit()
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_out_of_band_interruption_quarantines_submitted_write(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='S')
        session.add(pending)
        fake_async_sg._errors['batch'] = [SystemExit('out-of-band interruption')]
        with pytest.raises(SystemExit) as excinfo:
            await session.commit()
        assert any('indeterminate' in note for note in excinfo.value.__notes__)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            await session.commit()
        with pytest.raises(SgalchemyError, match='indeterminate'):
            _ = pending.code
        await session.close()
        replacement = AsyncSession(engine)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            replacement.add(pending)
        assert [name for name, _, _ in fake_async_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_async_pending_attachment_is_single_session_and_add_is_idempotent(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        other = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='one-create')
        session.add(pending)
        session.add(pending)
        with pytest.raises(SgalchemyError, match='another open session'):
            other.add(pending)
        persistent = session.merge_dict({'type': 'Shot', 'id': 31, 'code': 'owned'})
        with pytest.raises(SgalchemyError, match='another open session'):
            other.delete(persistent)
        await session.commit()
        requests = next(call for call in fake_async_sg._calls if call[0] == 'batch')[1][0]
        assert [request['request_type'] for request in requests] == ['create']

    asyncio.run(scenario())


def test_async_provider_commit_error_retires_attempted_upload_without_rewriting_error(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 97})
        image_path = tmp_path / 'async-provider-commit-error.jpg'
        filmstrip_path = tmp_path / 'async-provider-commit-error-filmstrip.jpg'
        image_path.write_bytes(b'image')
        filmstrip_path.write_bytes(b'image')
        shot.image = image_path
        shot.filmstrip_image = filmstrip_path
        session.delete(shot)
        cause = RuntimeError('provider cause')
        partitions = ([object()], [object()], [object()], [object()])
        fault = CommitError(
            'provider commit error',
            committed=partitions[0],
            staged=partitions[1],
            indeterminate=partitions[2],
            withdrawn=partitions[3],
        )
        fault.__cause__ = cause
        fake_async_sg._errors['upload_thumbnail'] = [fault]

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        assert excinfo.value is fault
        assert excinfo.value.__cause__ is cause
        assert excinfo.value.committed is partitions[0]
        assert excinfo.value.staged is partitions[1]
        assert excinfo.value.indeterminate is partitions[2]
        assert excinfo.value.withdrawn is partitions[3]
        assert entity_state(shot).dirty == {}
        assert session.unit_of_work.deleted == ()
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert len(fake_async_sg._calls) == calls_before
        assert [name for name, _, _ in fake_async_sg._calls].count('upload_thumbnail') == 1

    asyncio.run(scenario())


def test_async_quarantine_rejects_state_access_and_close_cleans_state(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        ambiguous = _entities(engine).Shot.new(code='ambiguous')
        unsubmitted = _entities(engine).Shot.new(code='unsubmitted')
        session.add(ambiguous)
        session.add(unsubmitted)
        fake_async_sg._errors['batch'] = [SystemExit('out-of-band interruption')]

        with pytest.raises(SystemExit):
            await session.commit(batch_size=1)

        assert session.engine is engine
        with pytest.raises(SgalchemyError, match='indeterminate'):
            session._identity_values()
        with pytest.raises(SgalchemyError, match='indeterminate'):
            _ = session.unit_of_work
        with pytest.raises(SgalchemyError, match='indeterminate'):
            session.detach_canceled(unsubmitted)

        await session.close()
        await session.close()

        ambiguous_state = entity_state(ambiguous)
        unsubmitted_state = entity_state(unsubmitted)
        assert ambiguous_state.indeterminate
        assert not unsubmitted_state.indeterminate
        assert ambiguous_state.session is None
        assert unsubmitted_state.session is None
        assert ambiguous_state.access_guard is None
        assert unsubmitted_state.access_guard is None
        with pytest.raises(SgalchemyError, match='new AsyncSession'):
            session._identity_values()
        replacement = AsyncSession(engine)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            replacement.add(ambiguous)
        replacement.add(unsubmitted)
        assert unsubmitted_state.session is replacement

    asyncio.run(scenario())


def test_async_recovery_base_exception_propagates_without_resend(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    class RecoveryAbort(BaseException):
        pass

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 23, 'description': 'old'})
        shot.description = 'new'
        fake_async_sg._batch_results = [[{'type': 'Shot', 'id': 23}]]
        fault = RecoveryAbort()
        fake_async_sg._find_error = fault

        with pytest.raises(RecoveryAbort) as excinfo:
            await session.commit()

        assert excinfo.value is fault
        state = entity_state(shot)
        assert state.dirty == {}
        assert state.clean == {}
        assert state.dirty_types == {}
        assert state.refresh_required == {'description'}
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_recovery_failure_logs_and_continues_later_steps(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        sequence = _entities(engine).Sequence.new(code='submitted-secret')
        dependent = _entities(engine).Shot.new(code='dependent', sg_sequence=sequence)
        session.add(sequence)
        session.add(dependent)
        fake_async_sg._batch_results = [
            [{'type': 'Sequence', 'id': 9501}],
            [
                {
                    'type': 'Shot',
                    'id': 9502,
                    'code': 'normalized-dependent',
                    'sg_sequence': {'type': 'Sequence', 'id': 9501},
                }
            ],
        ]
        fake_async_sg._find_error = RuntimeError('synthetic async recovery failure')

        with caplog.at_level('WARNING', logger='sgalchemy.session'):
            await session.commit()

        sequence_state = entity_state(sequence)
        assert sequence_state.dirty == {}
        assert sequence_state.clean == {}
        assert sequence_state.dirty_types == {}
        assert sequence_state.refresh_required == {'code'}
        assert dependent.code == 'normalized-dependent'
        assert len([call for call in fake_async_sg._calls if call[0] == 'find']) == 1
        errors = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(errors) == 1
        assert errors[0].exc_info is not None
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


@pytest.mark.parametrize('recovery', ['close', 'rollback'])
@pytest.mark.parametrize('before', ['refresh', 'unverified'])
def test_async_recovery_restores_field_provenance(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    recovery: str,
    before: str,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 40})
        state = entity_state(shot)
        if before == 'refresh':
            state.refresh_required.add('description')
        else:
            state.loaded['description'] = 'old'
            state.unverified_writes.add('description')

        shot.description = 'new'
        if recovery == 'close':
            await session.close()
        else:
            session.rollback()

        assert state.session is (None if recovery == 'close' else session)
        assert (state.access_guard is None) is (recovery == 'close')
        assert state.loaded.get('description') == (None if before == 'refresh' else 'old')
        assert state.refresh_required == ({'description'} if before == 'refresh' else set())
        assert state.unverified_writes == ({'description'} if before == 'unverified' else set())
        assert not state.dirty
        assert not state.clean
        assert not state.clean_refresh_required
        assert not state.clean_unverified_writes

    asyncio.run(scenario())


def test_async_retired_query_refresh_and_dirty_revival_commit(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 41, 'description': 'retired'}],
            [{'type': 'Shot', 'id': 41, 'description': 'refreshed'}],
            [{'type': 'Shot', 'id': 41, 'description': 'query-after-revive'}],
        ]
        shot = (
            await session.scalars(
                select(engine.entities.Shot).execution_options(retired_only=True)
            )
        )[0]
        await session.refresh(shot)
        assert [call for call in fake_async_sg._calls if call[0] == 'find'][-1][2][
            'retired_only'
        ] is True
        assert shot.description == 'refreshed'
        shot.description = 'local'
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(_errors.RetiredEntityWriteError, match=r'revive\(obj\)'):
            await session.commit()
        assert len(fake_async_sg._calls) == calls_before
        assert entity_state(shot).dirty == {'description': 'local'}
        session.revive(shot)
        assert (await session.scalars(select(engine.entities.Shot)))[0] is shot
        assert entity_state(shot).deleted is True
        assert session.unit_of_work.revived[0] is shot
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert [call[0] for call in fake_async_sg._calls[calls_before:]] == ['revive', 'batch']
        assert fake_async_sg._calls[-1][1][0] == [
            {
                'request_type': 'update',
                'entity_type': 'Shot',
                'entity_id': 41,
                'data': {'description': 'local'},
            }
        ]
        assert entity_state(shot).deleted is False
        assert entity_state(shot).dirty == {}
        assert session.unit_of_work.revived == ()
        await session.commit()
        assert [call[0] for call in fake_async_sg._calls[calls_before:]] == ['revive', 'batch']

    asyncio.run(scenario())


def test_async_revive_failure_remains_staged_without_indeterminate_state(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 12, 'code': 'retired'})
        entity_state(shot).deleted = True
        session.revive(shot)
        pending = _entities(engine).Shot.new(code='after-revive')
        session.add(pending)
        fault = RuntimeError('revive fault')
        fake_async_sg._errors['revive'] = [fault]
        with pytest.raises(CommitError) as excinfo:
            await session.commit()
        assert excinfo.value.__cause__ is fault
        assert excinfo.value.committed == []
        assert shot in excinfo.value.staged
        assert len(excinfo.value.staged) == 2
        assert excinfo.value.indeterminate == []
        assert _uow._BATCH_OUTCOME_UNKNOWN_NOTE not in excinfo.value.__notes__
        assert _uow._BATCH_UNSENT_NOTE not in excinfo.value.__notes__
        assert entity_state(shot).deleted
        await session.commit()
        assert not entity_state(shot).deleted
        assert pending.id is not None

    asyncio.run(scenario())


def test_async_revive_reconciliation_failure_is_normalized(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 13, 'code': 'retired'})
        entity_state(shot).deleted = True
        session.revive(shot)
        fault = SgalchemyError('revive reconciliation fault')

        def fail_registration(_obj: Entity) -> Never:
            raise fault

        monkeypatch.setattr(session, 'register_identity', fail_registration)

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        assert error.__cause__ is fault
        assert error.committed == []
        assert error.staged == [shot]
        assert error.indeterminate == []
        assert error.withdrawn == []
        assert [name for name, _, _ in fake_async_sg._calls].count('revive') == 1

    asyncio.run(scenario())


def test_async_schema_recovery_never_refreshes_after_first_submitted_write(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        registry = Registry()
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'first',
            registry=registry,
            bind_default=False,
        )
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = _entities(engine).Shot.new(code='planned')
        session.add(shot)
        captures = 0
        original_prepared_binding = engine.prepared_binding

        def tracking_prepared_binding() -> RegistryBinding | None:
            nonlocal captures
            captures += 1
            return original_prepared_binding()

        monkeypatch.setattr(engine, 'prepared_binding', tracking_prepared_binding)
        batch_entered, release_batch = fake_async_sg._block('batch')
        original_batch = fake_async_sg.batch

        async def batch_with_description(requests: WireRecords) -> list[BatchItem]:
            results = await original_batch(requests)
            assert isinstance(results[0], dict)
            results[0]['description'] = '2026-07-25'
            return results

        monkeypatch.setattr(fake_async_sg, 'batch', batch_with_description)
        committing = asyncio.create_task(session.commit())
        await batch_entered.wait()
        refreshed_schema = copy.deepcopy(schema_fixture)
        refreshed_schema['Shot']['description']['data_type']['value'] = 'date'
        refresher_sg = type(fake_async_sg)()

        async def refreshed_schema_read() -> dict[str, Any]:
            return copy.deepcopy(refreshed_schema)

        monkeypatch.setattr(refresher_sg, 'schema_read', refreshed_schema_read)
        refresher = create_async_engine(
            connection=refresher_sg,
            cache_root=tmp_path / 'refresher',
            registry=registry,
            bind_default=False,
        )
        await refresher.load_schema()
        release_batch.set()
        await committing

        assert captures == 1
        assert shot.description == '2026-07-25'
        refreshed = session.merge_dict({'type': 'Shot', 'id': 2, 'description': '2026-07-25'})
        assert refreshed.description == datetime.date(2026, 7, 25)
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 1

    asyncio.run(scenario())


def test_async_sole_batch_failure_passes_through_and_retries(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='sole-batch')
        session.add(pending)
        fault = RuntimeError('sole batch fault')
        fake_async_sg._errors['batch'] = [fault]
        with pytest.raises(RuntimeError) as excinfo:
            await session.commit()
        assert excinfo.value is fault
        assert excinfo.value.__cause__ is None
        assert any('single-batch' in note for note in excinfo.value.__notes__)
        assert entity_state(pending).dirty == {'code': 'sole-batch'}
        await session.commit()
        assert pending.id is not None
        assert [name for name, _, _ in fake_async_sg._calls].count('batch') == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'fault',
    [RuntimeError('local validation fault'), CommitError('local commit error')],
    ids=['runtime-error', 'commit-error'],
)
def test_async_sole_batch_local_failure_after_response_is_indeterminate(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    def fail_validation(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(_uow, '_validate_batch_results', fail_validation)

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        pending = _entities(engine).Shot.new(code='unconfirmed-local')
        session.add(pending)

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        assert error is not fault
        assert error.__cause__ is fault
        assert error.committed == []
        assert error.staged == []
        assert all(isinstance(item, _uow._Op) for item in error.indeterminate)
        assert [item.obj for item in error.indeterminate if isinstance(item, _uow._Op)] == [
            pending
        ]
        assert entity_state(pending).session is None
        assert entity_state(pending).access_guard is None
        assert entity_state(pending).id is None
        assert not any(name == 'find' for name, _, _ in fake_async_sg._calls)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            _ = pending.code

        await session.commit()
        assert [name for name, _, _ in fake_async_sg._calls].count('batch') == 1
        await session.close()

    asyncio.run(scenario())


def test_async_staging_rejects_foreign_registry(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'one',
            registry=Registry(),
            bind_default=False,
        )
        other = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'two',
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        await other.load_schema()
        session = AsyncSession(engine)
        foreign_session = AsyncSession(other)
        foreign = foreign_session.merge_dict({'type': 'Shot', 'id': 7, 'code': 'foreign'})
        pending = _entities(other).Shot.new(code='pending')
        with pytest.raises(SgalchemyError, match='different registry'):
            session.add(pending)
        with pytest.raises(SgalchemyError, match='different registry'):
            session.delete(foreign)
        with pytest.raises(SgalchemyError, match='different registry'):
            session.revive(foreign)

    asyncio.run(scenario())


def test_async_successful_revive_and_upload_are_not_repeated(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 7, 'code': 'existing'})
        entity_state(shot).deleted = True
        session.revive(shot)
        image_path = tmp_path / 'thumb.jpg'
        image_path.write_bytes(b'image')
        shot.image = image_path
        await session.commit()
        await session.commit()
        names = [name for name, _, _ in fake_async_sg._calls]
        assert names.count('revive') == 1
        assert names.count('upload_thumbnail') == 1
        assert not entity_state(shot).deleted
        assert not entity_state(shot).dirty

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('readback', 'raises'),
    [
        ([], False),
        ([{'type': 'Shot', 'id': 82}], False),
        ([{'type': 'Shot', 'id': '82', 'image': 'wrong-row'}], False),
        ([], True),
    ],
    ids=['missing-row', 'missing-field', 'malformed-row', 'read-exception'],
)
def test_async_successful_upload_readback_failure_requires_refresh(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readback: list[dict[str, object]],
    raises: bool,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 82, 'image': 'old-image'})
        image_path = tmp_path / 'async-successful-unreadable.jpg'
        image_path.write_bytes(b'image')
        shot.image = image_path
        fake_async_sg._find_results = [readback]
        if raises:
            fake_async_sg._find_error = RuntimeError('readback fault')
        confirmed: list[tuple[_uow.UploadPlanItem, CommitOutcome]] = []
        original_confirm = CommitOutcome._confirm

        def capture_confirm(
            outcome: CommitOutcome,
            items: tuple[Entity | _uow._Op | _uow.UploadPlanItem, ...],
        ) -> None:
            original_confirm(outcome, items)
            if isinstance(items[0], _uow.UploadPlanItem):
                confirmed.append((items[0], outcome))

        monkeypatch.setattr(CommitOutcome, '_confirm', capture_confirm)

        await session.commit()

        state = entity_state(shot)
        assert 'image' not in state.loaded
        assert state.refresh_required == {'image'}
        assert state.unverified_writes == set()
        assert len(confirmed) == 1
        upload, outcome = confirmed[0]
        assert outcome.committed[0] is upload
        assert outcome.indeterminate == []
        assert [name for name, _, _ in fake_async_sg._calls].count('find') == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('server_value', [None, 'server-image'])
def test_async_successful_upload_reads_exact_provider_value(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    server_value: object,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 81, 'image': 'old-image'})
        image_path = tmp_path / 'async-successful-upload.jpg'
        image_path.write_bytes(b'image')
        shot.image = image_path
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 81, 'image': server_value}]]

        await session.commit()

        state = entity_state(shot)
        assert state.loaded['image'] == server_value
        assert state.refresh_required == set()
        assert state.unverified_writes == set()
        assert state.dirty == {}
        find_calls = [call for call in fake_async_sg._calls if call[0] == 'find']
        assert len(find_calls) == 1
        assert find_calls[0][1][:3] == ('Shot', [['id', 'in', [81]]], ['image'])

    asyncio.run(scenario())


def test_async_successful_uploads_are_confirmed_before_later_awaits(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        first = session.merge_dict({'type': 'Shot', 'id': 83})
        second = session.merge_dict({'type': 'Shot', 'id': 84})
        first_path = tmp_path / 'async-first-immediate.jpg'
        second_path = tmp_path / 'async-second-immediate.jpg'
        first_path.write_bytes(b'image')
        second_path.write_bytes(b'image')
        first.image = first_path
        second.image = second_path
        fake_async_sg._find_results = [
            [
                {'type': 'Shot', 'id': 83, 'image': 'first-server'},
                {'type': 'Shot', 'id': 84, 'image': 'second-server'},
            ]
        ]
        original_upload = fake_async_sg.upload_thumbnail
        original_confirm = CommitOutcome._confirm
        confirmed: list[tuple[_uow.UploadPlanItem, CommitOutcome]] = []
        calls = 0

        def capture_confirm(
            outcome: CommitOutcome,
            items: tuple[Entity | _uow._Op | _uow.UploadPlanItem, ...],
        ) -> None:
            original_confirm(outcome, items)
            if isinstance(items[0], _uow.UploadPlanItem):
                confirmed.append((items[0], outcome))

        async def observe_second_upload(
            entity_type: str,
            entity_id: int,
            path: str,
        ) -> int:
            nonlocal calls
            assert type(path) is str
            calls += 1
            if calls == 2:
                state = entity_state(first)
                assert 'image' not in state.loaded
                assert state.refresh_required == {'image'}
                upload, outcome = confirmed[0]
                assert outcome.committed[0] is upload
            return await original_upload(entity_type, entity_id, path)

        monkeypatch.setattr(CommitOutcome, '_confirm', capture_confirm)
        monkeypatch.setattr(fake_async_sg, 'upload_thumbnail', observe_second_upload)

        await session.commit()

        assert first.image == 'first-server'
        assert second.image == 'second-server'

    asyncio.run(scenario())


def test_async_thin_echo_recovery_applies_normalized_value(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 23, 'description': 'old'})
        shot.description = 'submitted-secret'
        fake_async_sg._batch_results = [[{'type': 'Shot', 'id': 23}]]
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 23, 'description': 'normalized'}]]

        with caplog.at_level('WARNING', logger='sgalchemy.session'):
            await session.commit()

        assert shot.description == 'normalized'
        assert entity_state(shot).refresh_required == set()
        warnings = [record for record in caplog.records if record.levelname == 'WARNING']
        assert len(warnings) == 1
        assert 'provider-response contract' in warnings[0].getMessage()
        assert 'submitted-secret' not in warnings[0].getMessage()

    asyncio.run(scenario())


@pytest.mark.parametrize('discard', ['rollback', 'close'])
def test_async_thumbnail_clear_restores_only_ordinary_snapshot(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    discard: str,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict(
            {
                'type': 'Shot',
                'id': 48,
                'image': 'old-image',
                'filmstrip_image': 'old-filmstrip',
            }
        )
        shot.image = None

        if discard == 'close':
            await session.close()
        else:
            session.rollback()

        state = entity_state(shot)
        assert state.loaded['image'] == 'old-image'
        assert state.loaded['filmstrip_image'] == 'old-filmstrip'
        assert state.dirty == {}

    asyncio.run(scenario())


def test_async_upload_cancellation_is_raw_without_recovery_read(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 95})
        path = tmp_path / 'async-cancel-upload.jpg'
        path.write_bytes(b'image')
        shot.image = path
        entered, _release = fake_async_sg._block('upload_thumbnail')
        committing = asyncio.create_task(session.commit())
        await entered.wait()

        committing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await committing

        assert not [call for call in fake_async_sg._calls if call[0] == 'find']
        with pytest.raises(SgalchemyError, match='indeterminate'):
            _ = shot.code
        with pytest.raises(SgalchemyError, match='indeterminate'):
            await session.commit()
        assert [name for name, _, _ in fake_async_sg._calls].count('upload_thumbnail') == 1

    asyncio.run(scenario())


def test_async_upload_failure_partitions_and_clears_local_paths(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 13, 'code': 'upload'})
        image_path = tmp_path / 'thumb.jpg'
        filmstrip_path = tmp_path / 'filmstrip.jpg'
        image_path.write_bytes(b'image')
        filmstrip_path.write_bytes(b'image')
        shot.image = image_path
        shot.filmstrip_image = filmstrip_path
        session.delete(shot)
        fault = RuntimeError('upload fault')
        fake_async_sg._errors['upload_thumbnail'] = [fault]
        with pytest.raises(CommitError) as excinfo:
            await session.commit()
        assert excinfo.value.__cause__ is fault
        assert _uow._BATCH_OUTCOME_UNKNOWN_NOTE not in excinfo.value.__notes__
        assert _uow._BATCH_UNSENT_NOTE not in excinfo.value.__notes__
        assert excinfo.value.committed == []
        assert excinfo.value.staged == []
        assert all(isinstance(item, tuple) for item in excinfo.value.indeterminate)
        assert [item[1] for item in excinfo.value.indeterminate if isinstance(item, tuple)] == [
            'image'
        ]
        withdrawn_uploads = excinfo.value.withdrawn[:-1]
        assert all(isinstance(item, tuple) for item in withdrawn_uploads)
        assert [item[1] for item in withdrawn_uploads if isinstance(item, tuple)] == [
            'filmstrip_image'
        ]
        assert getattr(excinfo.value.withdrawn[-1], 'request_type', None) == 'delete'
        state = entity_state(shot)
        assert 'image' not in state.loaded
        assert 'filmstrip_image' not in state.loaded
        assert state.refresh_required == {'image'}
        assert state.unverified_writes == set()
        assert state.dirty == {}
        assert [name for name, _, _ in fake_async_sg._calls].count('find') == 1

    asyncio.run(scenario())


def test_async_upload_path_preflight_normalizes_custom_pathlike(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    class CustomPath:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def __fspath__(self) -> bytes:
            return self._value

    path = tmp_path / 'async-custom-path.jpg'
    path.write_bytes(b'image')

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 52, 'code': 'upload'})
        shot.image = CustomPath(os.fsencode(path))

        await session.commit()

        upload = next(call for call in fake_async_sg._calls if call[0] == 'upload_thumbnail')
        provider_path = upload[1][2]
        assert type(provider_path) is str
        assert os.fsencode(provider_path) == os.fsencode(path)

    asyncio.run(scenario())


@pytest.mark.parametrize('bad_path_kind', ['missing', 'directory', 'empty', 'unreadable'])
def test_async_upload_path_preflight_precedes_revive_and_preserves_staging(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_path_kind: str,
) -> None:
    if bad_path_kind == 'directory':
        path: object = tmp_path
    elif bad_path_kind == 'empty':
        path = ''
    else:
        path = tmp_path / f'async-{bad_path_kind}.jpg'
        if bad_path_kind == 'unreadable':
            path.write_bytes(b'image')
            real_open = _uow.os.open
            target_path = path

            @_preserve_signature(real_open)
            def fail_target_open(
                path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if path == str(target_path):
                    raise OSError('unreadable')
                return real_open(path, flags, mode, dir_fd=dir_fd)

            monkeypatch.setattr(_uow.os, 'open', fail_target_open)

    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        revived = session.merge_dict({'type': 'Shot', 'id': 49, 'code': 'revive'})
        entity_state(revived).deleted = True
        session.revive(revived)
        upload = session.merge_dict({'type': 'Shot', 'id': 50, 'code': 'upload'})
        upload.image = path
        calls_before = len(fake_async_sg._calls)

        with pytest.raises(SgalchemyError, match=r'Shot\.image.*readable regular file'):
            await session.commit()

        assert len(fake_async_sg._calls) == calls_before
        assert session.unit_of_work.revived == (revived,)
        assert entity_state(upload).dirty == {'image': path}

    asyncio.run(scenario())


@pytest.mark.parametrize('threaded', [False, True])
def test_async_unsent_upload_acquisition_failure_preserves_intent_for_retry(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    threaded: bool,
) -> None:
    async def scenario() -> None:
        failing = False
        fault = RuntimeError('upload client unavailable')

        def creator() -> ClosableShotgunLike:
            if failing:
                raise fault
            return cast('ClosableShotgunLike', fake_sg)

        async def async_creator() -> AsyncClosableShotgunLike:
            if failing:
                raise fault
            return cast('AsyncClosableShotgunLike', fake_async_sg)

        if threaded:
            engine = create_async_engine(
                fake_sg.base_url,
                creator=creator,
                strategy='threaded',
                max_workers=1,
                cache_root=tmp_path,
                registry=Registry(),
                bind_default=False,
            )
            provider = fake_sg
        else:
            engine = create_async_engine(
                fake_async_sg.base_url,
                async_creator=async_creator,
                cache_root=tmp_path,
                registry=Registry(),
                bind_default=False,
            )
            provider = fake_async_sg
        try:
            await engine.load_schema()
            session = AsyncSession(engine)
            shot = session.merge_dict({'type': 'Shot', 'id': 90, 'image': 'old'})
            path = tmp_path / 'unsent.jpg'
            path.write_bytes(b'image')
            state = entity_state(shot)
            state.refresh_required.add('image')
            state.unverified_writes.add('image')
            shot.image = path
            snapshot = (
                dict(state.dirty),
                dict(state.clean),
                dict(state.dirty_types),
                set(state.refresh_required),
                set(state.unverified_writes),
                set(state.clean_refresh_required),
                set(state.clean_unverified_writes),
            )
            await engine.dispose()
            failing = True
            provider._calls.clear()
            with pytest.raises(CommitError) as excinfo:
                await session.commit()
            assert excinfo.value.__cause__ is fault
            assert provider._calls == []
            assert (
                dict(state.dirty),
                dict(state.clean),
                dict(state.dirty_types),
                set(state.refresh_required),
                set(state.unverified_writes),
                set(state.clean_refresh_required),
                set(state.clean_unverified_writes),
            ) == snapshot
            assert state.dirty['image'] is path
            assert (excinfo.value.indeterminate, excinfo.value.withdrawn) == ([], [])
            failing = False
            provider._find_results = [[{'type': 'Shot', 'id': 90, 'image': 'uploaded'}]]
            await session.commit()
            assert [name for name, _, _ in provider._calls] == ['upload_thumbnail', 'find']
            assert shot.image == 'uploaded'
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_async_upload_preflight_does_not_block_loop(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        path = tmp_path / 'async-nonblocking-preflight.jpg'
        path.write_bytes(b'image')
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 53, 'code': 'upload'})
        shot.image = path

        exited = Event()
        release = Event()
        loop = asyncio.get_running_loop()
        probe_entered = asyncio.Event()
        loop_thread_id = get_ident()
        probe_thread_ids: list[int] = []
        real_open = _uow.os.open
        target_path = path

        @_preserve_signature(real_open)
        def blocking_target_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if path != str(target_path):
                return real_open(path, flags, mode, dir_fd=dir_fd)
            probe_thread_ids.append(get_ident())
            loop.call_soon_threadsafe(probe_entered.set)
            try:
                if not release.wait(timeout=1.0):
                    raise AssertionError('test did not release blocked upload preflight')
                return real_open(path, flags, mode, dir_fd=dir_fd)
            finally:
                exited.set()

        async def short_loop_timer() -> None:
            await probe_entered.wait()
            await asyncio.sleep(0.01)

        monkeypatch.setattr(_uow.os, 'open', blocking_target_open)
        committing = asyncio.create_task(session.commit())
        timer = asyncio.create_task(short_loop_timer())
        try:
            await asyncio.wait_for(timer, timeout=1.5)
            assert not exited.is_set()
            assert probe_thread_ids[0] != loop_thread_id
        finally:
            release.set()
            await asyncio.gather(committing, return_exceptions=True)
        await committing

    asyncio.run(scenario())


def test_async_upload_readbacks_group_entities_and_types(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        first = session.merge_dict({'type': 'Shot', 'id': 85})
        second = session.merge_dict({'type': 'Shot', 'id': 86})
        asset = session.merge_dict({'type': 'Asset', 'id': 87})
        paths = [
            tmp_path / name for name in ('async-first.jpg', 'async-second.jpg', 'async-asset.jpg')
        ]
        for path in paths:
            path.write_bytes(b'image')
        first.image, second.image, asset.image = paths
        fake_async_sg._find_results = [
            [
                {'type': 'Shot', 'id': 85, 'image': 'first-server'},
                {'type': 'Shot', 'id': 86, 'image': 'second-server'},
            ],
            [{'type': 'Asset', 'id': 87, 'image': 'asset-server'}],
        ]

        await session.commit()

        find_calls = [call for call in fake_async_sg._calls if call[0] == 'find']
        assert [call[1][0] for call in find_calls] == ['Shot', 'Asset']
        assert find_calls[0][1][1:3] == ([['id', 'in', [85, 86]]], ['image'])
        assert find_calls[1][1][1:3] == ([['id', 'in', [87]]], ['image'])
        assert first.image == 'first-server'
        assert second.image == 'second-server'
        assert asset.image == 'asset-server'

    asyncio.run(scenario())


def test_async_validated_chunk_hydration_failure_confirms_all_and_continues(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        first = _entities(engine).Sequence.new(code='first')
        second = _entities(engine).Sequence.new(code='second')
        dependent = _entities(engine).Shot.new(code='dependent', sg_sequence=second)
        session.add(first)
        session.add(second)
        session.add(dependent)
        fake_async_sg._batch_results = [
            [
                {'type': 'Sequence', 'id': 9301, 'code': 'normalized-first'},
                {'type': 'Sequence', 'id': 9302, 'code': 'normalized-second'},
            ],
            [
                {
                    'type': 'Shot',
                    'id': 9303,
                    'code': 'normalized-dependent',
                    'sg_sequence': {'type': 'Sequence', 'id': 9302},
                }
            ],
        ]
        new_queue = session.unit_of_work._new
        original_confirm = CommitOutcome._confirm
        original_hydrate = session.hydrate_record
        confirmed_before_hydration = False
        first_chunk: tuple[Entity | _uow._Op | _uow.UploadPlanItem, ...] = ()
        active_outcome: CommitOutcome | None = None

        def capture_confirm(
            outcome: CommitOutcome,
            items: tuple[Entity | _uow._Op | _uow.UploadPlanItem, ...],
        ) -> None:
            nonlocal active_outcome, first_chunk
            original_confirm(outcome, items)
            if len(items) == 2:
                active_outcome = outcome
                first_chunk = items

        def hydrate_with_fault(
            entity_type: str,
            record: WireRecord,
            binding: RegistryBinding | None,
        ) -> Entity:
            nonlocal confirmed_before_hydration
            if record['id'] == 9301:
                confirmed_before_hydration = (
                    active_outcome is not None
                    and active_outcome.contains_committed(first_chunk)
                    and session.unit_of_work._new is new_queue
                    and [id(obj) for obj in new_queue] == [id(dependent)]
                    and all(entity_state(obj).dirty == {} for obj in (first, second))
                )
                raise RuntimeError('synthetic async echo hydration failure')
            return original_hydrate(entity_type, record, binding)

        monkeypatch.setattr(CommitOutcome, '_confirm', capture_confirm)
        monkeypatch.setattr(session, 'hydrate_record', hydrate_with_fault)
        with caplog.at_level('ERROR', logger='sgalchemy.session'):
            await session.commit()

        assert confirmed_before_hydration
        first_state = entity_state(first)
        assert first_state.dirty == {}
        assert first_state.clean == {}
        assert first_state.dirty_types == {}
        assert 'code' not in first_state.loaded
        assert first_state.refresh_required == {'code'}
        assert second.code == 'normalized-second'
        assert dependent.code == 'normalized-dependent'
        assert len([call for call in fake_async_sg._calls if call[0] == 'batch']) == 2
        errors = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(errors) == 1
        assert errors[0].exc_info is not None
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_validated_chunk_phase_one_failure_keeps_confirmed_partition(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        first = _entities(engine).Sequence.new(code='first')
        second = _entities(engine).Sequence.new(code='second')
        dependent = _entities(engine).Shot.new(code='dependent', sg_sequence=second)
        session.add(first)
        session.add(second)
        session.add(dependent)
        fake_async_sg._batch_results = [
            [
                {'type': 'Sequence', 'id': 9401, 'code': 'first'},
                {'type': 'Sequence', 'id': 9402, 'code': 'second'},
            ]
        ]
        fault = RuntimeError('synthetic async phase one invariant failure')

        def fail_reconciliation(*_args: object) -> Never:
            raise fault

        monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', fail_reconciliation)

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        error = excinfo.value
        assert error.__cause__ is fault
        assert len(error.committed) == 2
        assert all(isinstance(item, _uow._Op) for item in error.committed)
        assert [item.obj for item in error.committed if isinstance(item, _uow._Op)] == [
            first,
            second,
        ]
        assert len(error.staged) == 1
        staged = error.staged[0]
        assert isinstance(staged, _uow._Op)
        assert staged.obj is dependent
        assert error.indeterminate == []
        assert error.withdrawn == []
        assert not any('nothing committed' in note for note in error.__notes__)

    asyncio.run(scenario())


def test_async_validated_chunk_phase_one_failure_retires_confirmed_updates(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        first = session.merge_dict({'type': 'Shot', 'id': 99, 'description': 'old-first'})
        second = session.merge_dict({'type': 'Shot', 'id': 100, 'description': 'old-second'})
        first.description = 'new-first'
        second.description = 'new-second'
        fake_async_sg._batch_results = [
            [
                {'type': 'Shot', 'id': 99, 'description': 'new-first'},
                {'type': 'Shot', 'id': 100, 'description': 'new-second'},
            ]
        ]
        fault = RuntimeError('synthetic async confirmed reconciliation failure')
        original_apply = _uow._apply_deterministic_thumbnail_clear

        def fail_reconciliation(*_args: object) -> Never:
            raise fault

        monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', fail_reconciliation)

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        assert excinfo.value.__cause__ is fault
        assert len(excinfo.value.committed) == 2
        assert excinfo.value.staged == []
        monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', original_apply)
        calls_before = len(fake_async_sg._calls)

        await session.commit()

        assert len(fake_async_sg._calls) == calls_before
        assert entity_state(first).dirty == {}
        assert entity_state(second).dirty == {}

    asyncio.run(scenario())


def test_automatic_cleanup_is_idempotent_and_preserves_quarantine(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        ambiguous = _entities(engine).Shot.new(code='ambiguous-secret')
        unsubmitted = _entities(engine).Shot.new(code='unsubmitted-secret')
        session.add(ambiguous)
        session.add(unsubmitted)
        fake_async_sg._errors['batch'] = [SystemExit('out-of-band interruption')]

        with pytest.raises(SystemExit):
            await session.commit(batch_size=1)

        calls_before = len(fake_async_sg._calls)
        _async_session.automatic_cleanup(
            session,
            task_name='quarantined-worker',
            cancelled=True,
        )
        _async_session.automatic_cleanup(
            session,
            task_name='quarantined-worker',
            cancelled=True,
        )

        ambiguous_state = entity_state(ambiguous)
        unsubmitted_state = entity_state(unsubmitted)
        assert ambiguous_state.indeterminate
        assert not unsubmitted_state.indeterminate
        assert ambiguous_state.session is None
        assert unsubmitted_state.session is None
        assert len(fake_async_sg._calls) == calls_before
        with pytest.raises(SgalchemyError, match='new AsyncSession'):
            session._identity_values()
        replacement = AsyncSession(engine)
        with pytest.raises(SgalchemyError, match='indeterminate'):
            replacement.add(ambiguous)
        replacement.add(unsubmitted)

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith(
            'automatic task cleanup detached entities, reverted local edits, '
            'and discarded uncommitted work'
        )
    ]
    assert warnings == [
        (
            'automatic task cleanup detached entities, reverted local edits, and discarded '
            "uncommitted work: task='quarantined-worker' cancelled=True dirty=0 pending=1 "
            'deletes=0 revives=0'
        )
    ]
    assert 'ambiguous-secret' not in warnings[0]
    assert 'unsubmitted-secret' not in warnings[0]


def test_automatic_cleanup_restores_interrupted_detached_dirty_revive(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        source = AsyncSession(engine)
        shot = source.merge_dict({'type': 'Shot', 'id': 101, 'description': 'old-revive-secret'})
        await source.close()
        shot.description = 'new-revive-secret'
        state = entity_state(shot)
        state.deleted = True
        session = AsyncSession(engine)
        session.revive(shot)
        fault = asyncio.CancelledError('revive interrupted')
        fake_async_sg._errors['revive'] = [fault]

        with pytest.raises(asyncio.CancelledError) as excinfo:
            await session.commit()

        assert excinfo.value is fault
        unit_of_work = object.__getattribute__(session, '_unit_of_work')
        assert unit_of_work.revived == ()
        assert state.session is session
        assert state.dirty == {'description': 'new-revive-secret'}
        assert state.indeterminate
        calls_before_cleanup = list(fake_async_sg._calls)

        _async_session.automatic_cleanup(
            session,
            task_name='detached-revive-worker',
            cancelled=True,
        )
        _async_session.automatic_cleanup(
            session,
            task_name='detached-revive-worker',
            cancelled=True,
        )

        assert fake_async_sg._calls == calls_before_cleanup
        assert [name for name, _, _ in fake_async_sg._calls].count('revive') == 1
        assert state.loaded['description'] == 'old-revive-secret'
        assert state.dirty == {}
        assert state.clean == {}
        assert state.session is None
        assert state.indeterminate
        with pytest.raises(SgalchemyError, match='new AsyncSession'):
            session._identity_values()

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        asyncio.run(scenario())

    warnings = [
        record.message
        for record in caplog.records
        if record.message.startswith(
            'automatic task cleanup detached entities, reverted local edits, '
            'and discarded uncommitted work'
        )
    ]
    assert warnings == [
        (
            'automatic task cleanup detached entities, reverted local edits, and discarded '
            "uncommitted work: task='detached-revive-worker' cancelled=True dirty=1 "
            'pending=0 deletes=0 revives=0'
        )
    ]
    assert 'old-revive-secret' not in warnings[0]
    assert 'new-revive-secret' not in warnings[0]


@pytest.mark.parametrize('threaded', [False, True])
def test_cancelled_commit_waiting_for_reset_preserves_upload_for_retry(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threaded: bool,
) -> None:
    async def scenario() -> None:
        engine = (
            create_async_engine(
                sync_connection=fake_sg,
                strategy='threaded',
                cache_root=tmp_path,
                registry=Registry(),
                bind_default=False,
            )
            if threaded
            else isolated_async_engine(fake_async_sg, tmp_path)
        )
        provider = fake_sg if threaded else fake_async_sg
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 90, 'image': 'old'})
        path = tmp_path / 'reset-wait.jpg'
        path.write_bytes(b'image')
        shot.image = path
        backend = cast('NativeAsyncBackend | ThreadedSyncBackend', engine._backend)
        waiting = asyncio.Event()
        original_wait = backend._reset_gate._done.wait

        async def observed_wait() -> bool:
            waiting.set()
            return await original_wait()

        monkeypatch.setattr(backend._reset_gate._done, 'wait', observed_wait)
        backend._reset_gate._resetting = True
        backend._reset_gate._done.clear()
        provider._calls.clear()
        committing = asyncio.create_task(session.commit())
        try:
            await asyncio.wait_for(waiting.wait(), 2)
            committing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await committing
            assert entity_state(shot).dirty['image'] is path
            assert not entity_state(shot).indeterminate
            assert provider._calls == []
            backend._reset_gate._resetting = False
            backend._reset_gate._done.set()
            provider._find_results = [[{'type': 'Shot', 'id': 90, 'image': 'uploaded'}]]
            await session.commit()
            assert shot.image == 'uploaded'
            assert [name for name, _, _ in provider._calls] == ['upload_thumbnail', 'find']
        finally:
            backend._reset_gate._resetting = False
            backend._reset_gate._done.set()
            await engine.dispose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'case',
    [(False, RuntimeError), (True, RuntimeError), (False, CommitError), (True, CommitError)],
)
def test_async_uninvoked_later_upload_recovers_confirmed_prefix_only(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[bool, type[RuntimeError | CommitError]],
) -> None:
    threaded, error_type = case

    async def scenario() -> None:
        engine = (
            create_async_engine(
                sync_connection=fake_sg,
                strategy='threaded',
                cache_root=tmp_path,
                registry=Registry(),
                bind_default=False,
            )
            if threaded
            else isolated_async_engine(fake_async_sg, tmp_path)
        )
        provider = fake_sg if threaded else fake_async_sg
        await engine.load_schema()
        session = AsyncSession(engine)
        objects = [session.merge_dict({'type': 'Shot', 'id': value}) for value in (88, 89, 90)]
        paths = [tmp_path / f'{value}.jpg' for value in (88, 89, 90)]
        for obj, path in zip(objects, paths, strict=True):
            path.write_bytes(b'image')
            obj.image = path
        original = provider.upload_thumbnail
        lookups = 0
        fault = error_type('upload method unavailable')

        def resolve(_client: object) -> object:
            nonlocal lookups
            lookups += 1
            if lookups == 2:
                raise fault
            return original

        monkeypatch.setattr(type(provider), 'upload_thumbnail', property(resolve))
        provider._calls.clear()
        provider._find_results = [[{'type': 'Shot', 'id': 88, 'image': 'confirmed'}]]
        try:
            with pytest.raises(CommitError) as excinfo:
                await session.commit()
            error = excinfo.value
            if isinstance(fault, CommitError):
                assert error is fault
            else:
                assert error.__cause__ is fault
                assert len(error.committed) == 1
                assert len(error.staged) == 2
            assert error.indeterminate == []
            assert error.withdrawn == []
            assert objects[0].image == 'confirmed'
            assert entity_state(objects[1]).dirty['image'] is paths[1]
            assert entity_state(objects[2]).dirty['image'] is paths[2]
            assert [name for name, _, _ in provider._calls] == ['upload_thumbnail', 'find']
            assert provider._calls[-1][1][:3] == ('Shot', [['id', 'in', [88]]], ['image'])
            provider._find_results = [
                [
                    {'type': 'Shot', 'id': 89, 'image': 'second'},
                    {'type': 'Shot', 'id': 90, 'image': 'third'},
                ]
            ]
            await session.commit()
            uploads = [args[1] for name, args, _ in provider._calls if name == 'upload_thumbnail']
            assert uploads == [88, 89, 90]
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_concurrent_task_cannot_commit_or_mutate_unit_of_work(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        session.add(_entities(engine).Shot.new(code='first'))
        entered, release = fake_async_sg._block('batch')
        committing = asyncio.create_task(session.commit())
        await entered.wait()
        second = _entities(engine).Shot.new(code='second')
        with pytest.raises(SgalchemyError, match='concurrent task'):
            session.add(second)
        assert entity_state(second).session is None
        with pytest.raises(SgalchemyError, match='concurrent task'):
            session.rollback()
        with pytest.raises(SgalchemyError, match='concurrent task'):
            await session.commit()
        release.set()
        await committing

    asyncio.run(scenario())


def test_interrupted_detached_delete_remains_owned_until_close(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        source = AsyncSession(engine)
        shot = source.merge_dict({'type': 'Shot', 'id': 41, 'code': 'detached'})
        await source.close()
        session = AsyncSession(engine)
        session.delete(shot)
        entered, _release = fake_async_sg._block('batch')
        committing = asyncio.create_task(session.commit())
        await entered.wait()
        committing.cancel()

        with pytest.raises(asyncio.CancelledError):
            await committing

        state = entity_state(shot)
        assert state.session is session
        assert any(candidate is shot for candidate in session._cleanup_anchors)
        await session.close()
        assert state.session is None
        assert state.access_guard is None

    asyncio.run(scenario())


def test_native_cancelled_upload_acquisition_preserves_intent(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        preparing = True

        async def creator() -> AsyncClosableShotgunLike:
            if not preparing:
                entered.set()
                await release.wait()
            return cast('AsyncClosableShotgunLike', fake_async_sg)

        engine = create_async_engine(
            fake_async_sg.base_url,
            async_creator=creator,
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 90, 'image': 'old'})
        path = tmp_path / 'native-acquisition.jpg'
        path.write_bytes(b'image')
        shot.image = path
        await engine.dispose()
        preparing = False
        fake_async_sg._calls.clear()
        committing = asyncio.create_task(session.commit())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            committing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await committing
            assert entity_state(shot).dirty['image'] is path
            assert not entity_state(shot).indeterminate
            assert fake_async_sg._calls == []
            release.set()
            fake_async_sg._find_results = [[{'type': 'Shot', 'id': 90, 'image': 'uploaded'}]]
            await session.commit()
            assert shot.image == 'uploaded'
            assert [name for name, _, _ in fake_async_sg._calls] == ['upload_thumbnail', 'find']
        finally:
            release.set()
            await engine.dispose()

    asyncio.run(scenario())


def test_threaded_cancelled_commit_is_drained_without_duplicate_submission(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        entered = Event()
        release = Event()
        fake_sg._block_next('batch', entered=entered, release=release)
        engine = create_async_engine(
            sync_connection=fake_sg,
            strategy='threaded',
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )
        task = None
        try:
            await engine.load_schema()
            session = AsyncSession(engine)
            pending = _entities(engine).Shot.new(code='S')
            session.add(pending)
            task = asyncio.create_task(session.commit())
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(SgalchemyError, match='indeterminate'):
                await session.commit()
            release.set()
            await session.close()
            replacement = AsyncSession(engine)
            with pytest.raises(SgalchemyError, match='indeterminate'):
                replacement.add(pending)
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await engine.dispose()
        assert [name for name, _, _ in fake_sg._calls].count('batch') == 1

    asyncio.run(scenario())


def test_threaded_queued_upload_cancellation_quarantines_before_provider_entry(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            sync_connection=fake_sg,
            strategy='threaded',
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 90, 'image': 'old'})
        path = tmp_path / 'queued.jpg'
        path.write_bytes(b'image')
        shot.image = path
        entered = Event()
        release = Event()
        fake_sg._block_next('schema_read', entered=entered, release=release)
        blocker = asyncio.create_task(engine.call('schema_read'))
        queued = asyncio.Event()
        observations: list[CallProgress] = []
        original = engine.call_with_progress

        async def observed_call(
            method: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            progress: CallProgress | None,
        ) -> object:
            if method == 'upload_thumbnail':
                assert progress is not None
                observations.append(progress)
                # The callback runs only after the gateway reaches its first await.
                asyncio.get_running_loop().call_soon(queued.set)
            return await original(method, args, kwargs, progress)

        monkeypatch.setattr(engine, 'call_with_progress', observed_call)
        committing = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            committing = asyncio.create_task(session.commit())
            await asyncio.wait_for(queued.wait(), 2)
            assert observations[0].dispatched.is_set()
            assert not observations[0].invoked.is_set()
            committing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await committing
            assert 'image' not in entity_state(shot).dirty
            quarantined_values = dict(entity_state(shot).loaded)
            release.set()
            await blocker
            await engine.dispose()
            assert observations[0].invoked.is_set()
            assert entity_state(shot).loaded == quarantined_values
            assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 1
            assert not any(name == 'find' for name, _, _ in fake_sg._calls)
            with pytest.raises(SgalchemyError, match='indeterminate'):
                await session.commit()
        finally:
            release.set()
            if committing is not None and not committing.done():
                committing.cancel()
            await blocker
            await engine.dispose()

    asyncio.run(scenario())
