from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Never, ParamSpec, TypeVar, cast

import pytest
from _fakes import _FakeShotgun
from _support import YieldCountingEntities, _SyncSessionBundle

from sgalchemy._core._compiler import FindCall
from sgalchemy._core._engine import create_engine
from sgalchemy._core._engine_common import AUTH_RECOVERY_NOTE
from sgalchemy._core._registry import Entity, Registry, RegistryBinding, entity_state
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _session, _uow
from sgalchemy._orm._session import Session
from sgalchemy._orm._uow import CommitOutcome
from sgalchemy._shotgun import BatchItem, ClosableShotgunLike, WireRecord, WireRecords

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T')


class _IterationCountingList(list[_T]):
    def __init__(self, values: Iterable[_T]) -> None:
        super().__init__(values)
        self._iterations = 0

    def __iter__(self) -> Iterator[_T]:
        self._iterations += 1
        return super().__iter__()


def _authentication_fault(message: str) -> RuntimeError:
    fault_type = cast('type[RuntimeError]', type('AuthenticationFault', (RuntimeError,), {}))
    return fault_type(message)


def test_ambiguous_upload_preserves_unrelated_work_and_error_snapshots(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    failed = session.merge_dict({'type': 'Shot', 'id': 90, 'code': 'failed'})
    unrelated = session.merge_dict({'type': 'Shot', 'id': 91, 'code': 'unrelated'})
    failed_path = tmp_path / 'failed-shared-state.jpg'
    unrelated_path = tmp_path / 'unrelated-shared-state.jpg'
    retry_path = tmp_path / 'retry-shared-state.jpg'
    for path in (failed_path, unrelated_path, retry_path):
        path.write_bytes(b'image')
    failed.image = failed_path
    unrelated.image = unrelated_path
    fault = RuntimeError('upload fault')
    fake_sg._errors['upload_thumbnail'] = [fault]
    fake_sg._find_results = [[{'type': 'Shot', 'id': 90, 'image': 'current-server'}]]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

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
    fake_sg._find_results = [
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

    session.commit()
    session.refresh(failed)
    failed.image = retry_path

    assert unrelated.image == 'unrelated-server'
    assert error.committed == snapshots[0]
    assert error.staged == snapshots[1]
    assert error.indeterminate == snapshots[2]
    assert error.withdrawn == snapshots[3]


def test_batch_partial_reconciliation_failure_keeps_confirmed_partition(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = engine.entities.Sequence.new(code='first')
    second = engine.entities.Sequence.new(code='second')
    dependent = engine.entities.Shot.new(code='dependent', sg_sequence=second)
    session.add(first)
    session.add(second)
    session.add(dependent)
    fake_sg._batch_results = [
        [
            {'type': 'Sequence', 'id': 9201, 'code': 'first'},
            {'type': 'Sequence', 'id': 9202, 'code': 'second'},
        ]
    ]
    fault = _authentication_fault('synthetic phase one authentication fault')

    def fail_reconciliation(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', fail_reconciliation)

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert type(error) is CommitError
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
    assert any('failed during batch wave 1 chunk 1' in note for note in error.__notes__)
    assert AUTH_RECOVERY_NOTE in error.__notes__
    assert not any('single-batch' in note for note in error.__notes__)
    assert not any('nothing committed' in note for note in error.__notes__)


def test_batch_reconciliation_error_teaches_fresh_session(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='bad-response')
    session.add(pending)
    fake_sg._batch_results = [[]]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert type(error) is CommitError
    assert type(error.__cause__) is _uow.BatchReconciliationError
    assert error.committed == []
    assert error.staged == []
    assert len(error.indeterminate) == 1
    indeterminate = error.indeterminate[0]
    assert isinstance(indeterminate, _uow._Op)
    assert indeterminate.obj is pending
    assert error.withdrawn == []
    assert any(
        'discard affected objects and reconcile server state through a fresh query '
        'in a new session' in note
        for note in error.__notes__
    )
    assert any('failed during batch wave 1 chunk 1' in note for note in error.__notes__)
    assert not any('single-batch' in note for note in error.__notes__)


def test_batch_validation_and_echoes_use_separate_staging_snapshots(
    sess: _SyncSessionBundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    deleted = session.merge_dict({'type': 'Shot', 'id': 99})
    session.delete(deleted)
    created = [engine.entities.Shot.new(code=f'new-{i}') for i in range(3)]
    for obj in created:
        session.add(obj)
    chunk = tuple(session._plan_commit()._waves[0])
    outcome = CommitOutcome([], list(chunk), [], [])
    snapshots: list[tuple[Entity, ...]] = []
    original_candidates = session._state._staged_identity_candidates
    original_apply = _uow._apply_deterministic_thumbnail_clear

    def capture_candidates() -> tuple[Entity, ...]:
        candidates = tuple(original_candidates())
        snapshots.append(candidates)
        return candidates

    def apply_result(operation: _uow._Op) -> None:
        assert session._state._hydration_index is None
        original_apply(operation)

    monkeypatch.setattr(session._state, '_staged_identity_candidates', capture_candidates)
    monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', apply_result)
    results: list[BatchItem] = [
        True if op.request_type == 'delete' else {'type': 'Shot', 'id': 20 + i, **op.data}
        for i, op in enumerate(chunk)
    ]

    reads = _uow.apply_batch_results(
        session, chunk, results, outcome, session._state.identity_map.__setitem__
    )

    assert reads == ()
    assert snapshots == [(deleted,), ()]
    assert len(outcome.committed) == 4
    assert outcome.staged == []
    assert session.unit_of_work.deleted == ()
    assert session.unit_of_work.new == ()
    assert all(session._state.identity_map[('Shot', obj.id)] is obj for obj in created)
    assert session._state._hydration_index is None


def test_chunk_failure_note_names_phase_and_batch_size(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    for index in range(4):
        session.add(engine.entities.Shot.new(code=f'note-{index}'))
    fake_sg._fail_on_batch_index = 1

    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=2)

    notes = excinfo.value.__notes__
    assert any('batch wave 1 chunk 2' in note for note in notes)
    assert any('batch_size=2' in note for note in notes)


def test_chunking_and_three_state_failure(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot_cls = engine.registry.entity_class('Shot')
    pending = [shot_cls.new(code=f'S{i}') for i in range(4)]
    for obj in pending:
        session.add(obj)
    fault = _authentication_fault('second batch authentication fault')
    fake_sg._errors['batch'] = [None, fault]

    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=2)

    error = excinfo.value
    assert type(error) is CommitError
    assert error.__cause__ is fault
    assert all(isinstance(item, _uow._Op) for item in error.committed)
    assert [item.obj for item in error.committed if isinstance(item, _uow._Op)] == pending[:2]
    assert all(isinstance(item, _uow._Op) for item in error.staged)
    assert [item.obj for item in error.staged if isinstance(item, _uow._Op)] == pending[2:]
    assert error.indeterminate == []
    assert error.withdrawn == []
    assert any(
        'if this session remains open and usable, a later commit may retry' in note
        for note in error.__notes__
    )
    assert error.__notes__.count(_uow._BATCH_OUTCOME_UNKNOWN_NOTE) == 1
    assert any('failed during batch wave 1 chunk 2' in note for note in error.__notes__)
    assert AUTH_RECOVERY_NOTE in error.__notes__
    assert not any('single-batch' in note for note in error.__notes__)
    session2_batches = [c for c in fake_sg._calls if c[0] == 'batch']
    assert all(len(c[1][0]) == 2 for c in session2_batches)


def test_confirmed_field_read_response_scans_staged_items_once(sess: _SyncSessionBundle) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    objects = [session.merge_dict({'type': 'Shot', 'id': i + 1}) for i in range(20)]
    staged = [session.merge_dict({'type': 'Shot', 'id': 1000 + i}) for i in range(20)]
    for obj in staged:
        session.delete(obj)
    queue = YieldCountingEntities(session.unit_of_work._deleted)
    session.unit_of_work._deleted = queue
    group = _uow.ConfirmedFieldReadGroup(
        entity_type='Shot',
        retired_only=False,
        binding=engine.prepared_binding(),
        _reads=tuple(_uow.ConfirmedFieldRead(obj, ('project',), ('project',)) for obj in objects),
    )
    records: WireRecords = [
        {'type': 'Shot', 'id': i + 1, 'project': {'type': 'Project', 'id': 1}} for i in range(20)
    ]

    _uow.apply_confirmed_field_readback(session, group, records)

    assert queue._visits == 20
    assert all(obj.project is objects[0].project for obj in objects)
    assert session._state._hydration_index is None


def test_confirmed_upload_readback_failure_does_not_block_final_retirement(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 94})
    path = tmp_path / 'retire-after-read-failure.jpg'
    path.write_bytes(b'image')
    shot.image = path
    session.delete(shot)
    fake_sg._find_error = RuntimeError('readback fault')

    session.commit()

    commit_calls = [
        call for call in fake_sg._calls if call[0] in {'upload_thumbnail', 'find', 'batch'}
    ]
    assert [call[0] for call in commit_calls] == ['upload_thumbnail', 'find', 'batch']
    assert entity_state(shot).deleted


def test_create_cycle_raises_before_io(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    a_cls = engine.registry.entity_class('Asset')
    shot_cls = engine.registry.entity_class('Shot')
    shot = shot_cls.new(code='S')
    asset = a_cls.new(code='A', shots=[shot])
    shot.assets = [asset]
    session.add(shot)
    session.add(asset)
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match='cycle'):
        session.commit()
    assert len(fake_sg._calls) == calls_before


def test_dependent_creates_split_into_waves(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    seq_cls = engine.registry.entity_class('Sequence')
    shot_cls = engine.registry.entity_class('Shot')
    seq = seq_cls.new(code='SEQ01')
    shot = shot_cls.new(code='S1', sg_sequence=seq)  # references pending create
    session.add(seq)
    session.add(shot)
    session.commit()
    batch_calls = [c for c in fake_sg._calls if c[0] == 'batch']
    assert len(batch_calls) == 2  # wave 1: seq, wave 2: shot with seq's fresh id
    shot_req = batch_calls[1][1][0][0]
    assert shot_req['data']['sg_sequence'] == {'type': 'Sequence', 'id': seq.id}


def test_dirty_delete_with_upload_defers_only_retirement(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    image_path = tmp_path / 'retiring-thumbnail.jpg'
    image_path.write_bytes(b'thumbnail')
    shot = session.merge_dict({'type': 'Shot', 'id': 16, 'description': 'before'})
    shot.description = 'after'
    shot.image = image_path
    session.delete(shot)
    fake_sg._find_results = [[{'type': 'Shot', 'id': 16, 'image': 'server-image'}]]
    calls_before = len(fake_sg._calls)

    session.commit()

    commit_calls = fake_sg._calls[calls_before:]
    assert [name for name, _, _ in commit_calls] == [
        'batch',
        'upload_thumbnail',
        'find',
        'batch',
    ]
    assert [request['request_type'] for request in commit_calls[0][1][0]] == ['update']
    assert commit_calls[1][1] == ('Shot', 16, str(image_path))
    assert [request['request_type'] for request in commit_calls[3][1][0]] == ['delete']


def test_dirty_delete_without_upload_is_one_indivisible_ordered_batch(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 15, 'description': 'before'})
    shot.description = 'after'
    session.delete(shot)
    calls_before = len(fake_sg._calls)

    session.commit()

    commit_calls = fake_sg._calls[calls_before:]
    assert [name for name, _, _ in commit_calls] == ['batch']
    assert commit_calls[0][1][0] == [
        {
            'request_type': 'update',
            'entity_type': 'Shot',
            'entity_id': 15,
            'data': {'description': 'after'},
        },
        {'request_type': 'delete', 'entity_type': 'Shot', 'entity_id': 15},
    ]


def test_dirty_indeterminate_owner_is_rejected_before_io(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 21, 'description': 'before'})
    shot.description = 'after'
    entity_state(shot).indeterminate = True
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='indeterminate'):
        session.commit()

    assert len(fake_sg._calls) == calls_before


def test_exact_guard_preserves_later_dirty_field_and_accepts_deep_echo(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    sequence = session.merge_dict({'type': 'Sequence', 'id': 31, 'code': 'old-sequence'})
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 23,
            'code': 'old-shot',
            'description': 'old-description',
            'sg_sequence': {'type': 'Sequence', 'id': 31},
        }
    )
    shot.description = 'submitted-description'

    def batch_with_later_assignment(requests: WireRecords) -> list[BatchItem]:
        fake_sg._record('batch', requests)
        assert requests[0]['data'] == {'description': 'submitted-description'}
        shot.code = 'later-local-code'
        return [
            {
                'type': 'Shot',
                'id': 23,
                'code': 'server-code',
                'description': 'normalized-description',
                'sg_sequence.Sequence.code': 'deep-sequence-code',
            }
        ]

    monkeypatch.setattr(fake_sg, 'batch', batch_with_later_assignment)

    session.commit()

    state = entity_state(shot)
    assert state.loaded['description'] == 'normalized-description'
    assert state.loaded['code'] == 'later-local-code'
    assert state.dirty == {'code': 'later-local-code'}
    assert state.dirty_types == {'code': 'text'}
    assert entity_state(sequence).loaded['code'] == 'deep-sequence-code'


def test_failed_upload_current_value_is_loaded_but_unverified(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 89, 'image': 'old-image'})
    image_path = tmp_path / 'ambiguous-current.jpg'
    image_path.write_bytes(b'image')
    shot.image = image_path
    fault = RuntimeError('upload fault')
    fake_sg._errors['upload_thumbnail'] = [fault]
    fake_sg._find_results = [[{'type': 'Shot', 'id': 89, 'image': 'current-server'}]]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    state = entity_state(shot)
    assert excinfo.value.__cause__ is fault
    assert excinfo.value.committed == []
    indeterminate = excinfo.value.indeterminate
    assert all(isinstance(item, tuple) for item in indeterminate)
    assert [item[1] for item in indeterminate if isinstance(item, tuple)] == ['image']
    assert state.loaded['image'] == 'current-server'
    assert state.refresh_required == set()
    assert state.unverified_writes == {'image'}
    assert any(
        'readable provider value marked unverified; refresh it to accept the current server state'
        in note
        for note in excinfo.value.__notes__
    )
    assert not any('before reading' in note for note in excinfo.value.__notes__)
    assert [name for name, _, _ in fake_sg._calls].count('find') == 1


def test_failed_upload_explicit_reassignment_submits_one_new_upload(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 93})
    first_path = tmp_path / 'failed-first.jpg'
    retry_path = tmp_path / 'explicit-new-intent.jpg'
    first_path.write_bytes(b'image')
    retry_path.write_bytes(b'image')
    shot.image = first_path
    fake_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

    with pytest.raises(CommitError):
        session.commit()
    shot.image = retry_path
    fake_sg._find_results = [[{'type': 'Shot', 'id': 93, 'image': 'retry-server'}]]

    session.commit()
    session.commit()

    assert shot.image == 'retry-server'
    assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 2


def test_failed_upload_is_never_resent_by_plain_commit(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 92})
    path = tmp_path / 'never-resend.jpg'
    path.write_bytes(b'image')
    shot.image = path
    fake_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

    with pytest.raises(CommitError):
        session.commit()
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before
    assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 1


def test_failed_upload_readback_preserves_callback_reassignment(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 96, 'image': 'old-image'})
    failed_path = tmp_path / 'ambiguous-callback.jpg'
    reassigned_path = tmp_path / 'callback-reassignment.jpg'
    failed_path.write_bytes(b'image')
    reassigned_path.write_bytes(b'image')
    shot.image = failed_path
    fault = RuntimeError('upload fault')
    fake_sg._errors['upload_thumbnail'] = [fault]

    def find_with_reassignment(
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
        **kwargs: object,
    ) -> WireRecords:
        fake_sg._record('find', entity_type, filters, fields, **kwargs)
        shot.image = reassigned_path
        return [{'type': 'Shot', 'id': 96, 'image': 'current-server'}]

    monkeypatch.setattr(fake_sg, 'find', find_with_reassignment)

    with pytest.raises(CommitError):
        session.commit()

    state = entity_state(shot)
    assert state.loaded['image'] == reassigned_path
    assert state.dirty == {'image': reassigned_path}
    assert state.refresh_required == set()
    assert state.unverified_writes == set()


def test_failed_upload_withdraws_unsubmitted_same_owner_retirement(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    image_path = str(tmp_path / 'retiring-ambiguous.jpg')
    Path(image_path).write_bytes(b'image')
    source = type(session)(engine)
    shot = source.merge_dict({'type': 'Shot', 'id': 19, 'code': 'retiring'})
    source.close()
    shot.image = image_path
    session.delete(shot)
    fake_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert error.committed == []
    assert error.staged == []
    assert error.indeterminate == [(shot, 'image', image_path)]
    assert len(error.withdrawn) == 1
    assert getattr(error.withdrawn[0], 'request_type', None) == 'delete'
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before
    retry_path = tmp_path / 'explicit-retry.jpg'
    retry_path.write_bytes(b'image')
    shot.image = retry_path
    session.commit()
    assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 2
    session.delete(shot)
    session.commit()
    assert fake_sg._calls[-1][0] == 'batch'
    assert fake_sg._calls[-1][1][0][0]['request_type'] == 'delete'


def test_false_revive_and_delete_results_are_successful_desired_states(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 18, 'code': 'active'})
    session.delete(shot)
    fake_sg._batch_results = [[False]]
    session.commit()
    assert entity_state(shot).deleted

    def already_active(entity_type: str, entity_id: int) -> bool:
        fake_sg._record('revive', entity_type, entity_id)
        return False

    monkeypatch.setattr(fake_sg, 'revive', already_active)
    session.revive(shot)
    session.commit()
    assert not entity_state(shot).deleted


def test_indivisible_dirty_delete_rejects_too_small_batch_before_io(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 17, 'description': 'before'})
    shot.description = 'after'
    session.delete(shot)
    calls_before = len(fake_sg._calls)

    with pytest.raises(ValueError, match=r'batch_size.*minimum.*2'):
        session.commit(batch_size=1)

    assert len(fake_sg._calls) == calls_before


def test_later_dirty_filmstrip_guards_confirmed_thumbnail_clear(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 26,
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )
    path = tmp_path / 'later-filmstrip.jpg'
    path.write_bytes(b'image')
    shot.image = None
    shot.filmstrip_image = path
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 26, 'image': None}]]
    fake_sg._find_results = [[{'type': 'Shot', 'id': 26, 'filmstrip_image': 'server-filmstrip'}]]

    session.commit()

    state = entity_state(shot)
    assert state.loaded['image'] is None
    assert state.loaded['filmstrip_image'] == 'server-filmstrip'
    assert state.dirty == {}
    assert [name for name, _, _ in fake_sg._calls].count('upload_filmstrip_thumbnail') == 1


def test_later_upload_failure_reads_prior_confirmed_upload_once(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 88})
    image_path = tmp_path / 'confirmed-before-failure.jpg'
    filmstrip_path = tmp_path / 'ambiguous-after-success.jpg'
    image_path.write_bytes(b'image')
    filmstrip_path.write_bytes(b'image')
    shot.image = image_path
    shot.filmstrip_image = filmstrip_path
    fault = RuntimeError('filmstrip upload fault')
    fake_sg._errors['upload_filmstrip_thumbnail'] = [fault]
    fake_sg._find_results = [
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
        session.commit()

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
    find_calls = [call for call in fake_sg._calls if call[0] == 'find']
    assert len(find_calls) == 1
    assert find_calls[0][1][:3] == (
        'Shot',
        [['id', 'in', [88]]],
        ['filmstrip_image', 'image'],
    )


def test_later_upload_failure_reconciles_pending_readbacks_across_types(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 88})
    asset = session.merge_dict({'type': 'Asset', 'id': 89})
    filmstrip_path = tmp_path / 'confirmed-cross-type-filmstrip.jpg'
    image_path = tmp_path / 'failed-cross-type-image.jpg'
    filmstrip_path.write_bytes(b'image')
    image_path.write_bytes(b'image')
    shot.filmstrip_image = filmstrip_path
    asset.image = image_path
    fault = RuntimeError('later cross-type upload fault')
    fake_sg._errors['upload_thumbnail'] = [fault]
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 88, 'filmstrip_image': 'shot-server'}],
        [{'type': 'Asset', 'id': 89, 'image': 'asset-server'}],
    ]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert all(isinstance(item, tuple) for item in error.committed)
    assert [item[1] for item in error.committed if isinstance(item, tuple)] == ['filmstrip_image']
    assert all(isinstance(item, tuple) for item in error.indeterminate)
    assert [item[1] for item in error.indeterminate if isinstance(item, tuple)] == ['image']
    find_calls = [call for call in fake_sg._calls if call[0] == 'find']
    assert [call[1][0] for call in find_calls] == ['Shot', 'Asset']
    assert shot.filmstrip_image == 'shot-server'
    assert asset.image == 'asset-server'
    assert entity_state(shot).unverified_writes == set()
    assert entity_state(asset).unverified_writes == {'image'}


def test_malformed_batch_quarantine_clears_provenance_and_blocks_dependents(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    sequence = engine.entities.Sequence.new(code='ambiguous')
    shot = engine.entities.Shot.new(code='blocked', sg_sequence=sequence)
    session.add(sequence)
    session.add(shot)
    fake_sg._batch_results = [[]]

    with pytest.raises(CommitError):
        session.commit(batch_size=1)

    assert entity_state(sequence).dirty == {}
    assert entity_state(sequence).clean == {}
    assert entity_state(sequence).dirty_types == {}
    assert entity_state(shot).dirty == {'code': 'blocked', 'sg_sequence': sequence}
    assert entity_state(shot).dirty_types == {'code': 'text', 'sg_sequence': 'entity'}
    assert not any(name == 'find' for name, _, _ in fake_sg._calls)
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match='indeterminate'):
        session.commit(batch_size=1)
    assert len(fake_sg._calls) == calls_before


def test_malformed_create_with_upload_withdraws_unsent_same_owner_work(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    image_path = str(tmp_path / 'create-ambiguous.jpg')
    Path(image_path).write_bytes(b'image')
    pending = engine.entities.Shot.new(code='ambiguous', image=image_path)
    session.add(pending)
    fake_sg._batch_results = [[]]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert error.committed == []
    assert error.staged == []
    assert len(error.indeterminate) == 1
    assert getattr(error.indeterminate[0], 'obj', None) is pending
    assert error.withdrawn == [(pending, 'image', image_path)]
    state = entity_state(pending)
    assert state.dirty == {}
    assert state.clean == {}
    assert state.dirty_types == {}
    assert state.loaded['image'] == image_path
    assert state.indeterminate
    assert state.session is None
    assert state.access_guard is None
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before
    session.close()
    assert state.session is None


def test_malformed_update_identity_is_quarantined_without_retry(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 23, 'description': 'old'})
    shot.description = 'new'
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 999, 'description': 'foreign'}]]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert (error.committed, error.staged, error.withdrawn) == ([], [], [])
    assert len(error.indeterminate) == 1
    operation = error.indeterminate[0]
    assert isinstance(operation, _uow._Op)
    assert (operation.obj, operation.request_type, operation.entity_id) == (shot, 'update', 23)
    assert ('Shot', 999) not in session._state.identity_map
    state = entity_state(shot)
    assert state.id == 23
    assert state.indeterminate
    assert state.dirty == {}
    with pytest.raises(SgalchemyError, match='indeterminate'):
        _ = shot.description

    calls_before_retry = list(fake_sg._calls)
    session.commit()
    assert fake_sg._calls == calls_before_retry


def test_malformed_update_withdraws_same_owner_upload_and_delete(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    affected = source.merge_dict({'type': 'Shot', 'id': 23, 'description': 'before'})
    source.close()
    affected.description = 'after'
    image_path = str(tmp_path / 'update-delete-ambiguous.jpg')
    Path(image_path).write_bytes(b'image')
    affected.image = image_path
    session.delete(affected)
    independent = engine.entities.Shot.new(code='independent-delete-case')
    session.add(independent)
    fake_sg._batch_results = [[]]

    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=1)

    error = excinfo.value
    assert error.committed == []
    assert len(error.staged) == 1
    assert getattr(error.staged[0], 'obj', None) is independent
    assert len(error.indeterminate) == 1
    assert getattr(error.indeterminate[0], 'request_type', None) == 'update'
    assert getattr(error.indeterminate[0], 'obj', None) is affected
    assert [getattr(item, 'request_type', None) for item in error.withdrawn] == [None, 'delete']
    assert error.withdrawn[0] == (affected, 'image', image_path)
    assert getattr(error.withdrawn[1], 'obj', None) is affected
    assert session.unit_of_work.deleted == ()
    affected_state = entity_state(affected)
    assert affected_state.dirty == {}
    assert affected_state.clean == {}
    assert affected_state.dirty_types == {}
    assert affected_state.loaded['description'] == 'after'
    assert affected_state.loaded['image'] == image_path
    assert affected_state.indeterminate
    assert entity_state(independent).dirty == {'code': 'independent-delete-case'}
    assert entity_state(independent).dirty_types == {'code': 'text'}
    calls_before = len(fake_sg._calls)

    session.commit(batch_size=1)

    retry_calls = fake_sg._calls[calls_before:]
    assert [name for name, _, _ in retry_calls] == ['batch']
    assert retry_calls[0][1][0] == [
        {
            'request_type': 'create',
            'entity_type': 'Shot',
            'data': {'code': 'independent-delete-case'},
        }
    ]
    assert independent.id is not None
    session.close()
    assert entity_state(affected).session is None


def test_multistep_local_failure_after_response_is_indeterminate(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    pending = [engine.entities.Shot.new(code=f'local-{index}') for index in range(2)]
    for obj in pending:
        session.add(obj)
    fault = _authentication_fault('local reconciliation authentication fault')
    original_validate = _uow._validate_batch_results
    calls = 0

    def fail_second_validation(call: Callable[_P, _R]) -> Callable[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise fault
            return call(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        _uow,
        '_validate_batch_results',
        fail_second_validation(original_validate),
    )

    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=1)

    error = excinfo.value
    assert type(error) is CommitError
    assert error.__cause__ is fault
    assert all(isinstance(item, _uow._Op) for item in error.committed)
    assert [item.obj for item in error.committed if isinstance(item, _uow._Op)] == pending[:1]
    assert error.staged == []
    assert all(isinstance(item, _uow._Op) for item in error.indeterminate)
    assert [item.obj for item in error.indeterminate if isinstance(item, _uow._Op)] == pending[1:]
    assert error.withdrawn == []
    assert any(
        'discard affected objects and reconcile server state' in note for note in error.__notes__
    )
    assert any('failed during batch wave 1 chunk 2' in note for note in error.__notes__)
    assert AUTH_RECOVERY_NOTE in error.__notes__
    assert not any('single-batch' in note for note in error.__notes__)


def test_multistep_serialization_failure_preserves_committed_evidence(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = engine.entities.Shot.new(code='first')
    second = engine.entities.Shot.new(code='second')
    session.add(first)
    session.add(second)
    serialize = _session.serialize_chunk
    calls = 0

    def fail_second(operations: tuple[_uow._Op, ...]) -> WireRecords:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('serialize')
        return serialize(operations)

    monkeypatch.setattr(_session, 'serialize_chunk', fail_second)

    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=1)

    error = excinfo.value
    assert [item.obj for item in error.committed if isinstance(item, _uow._Op)] == [first]
    assert [item.obj for item in error.staged if isinstance(item, _uow._Op)] == [second]
    assert error.indeterminate == []
    assert error.withdrawn == []
    assert error.__notes__.count(_uow._BATCH_UNSENT_NOTE) == 1
    assert [name for name, _, _ in fake_sg._calls].count('batch') == 1


def test_nested_pending_create_dependency_is_resolved_before_serialization(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    sequence = engine.entities.Sequence.new(code='nested')
    shot = engine.entities.Shot.new(
        code='dependent',
        sg_sequence={'container': [sequence]},
    )
    session.add(shot)
    session.add(sequence)

    session.commit()

    batch_calls = [call for call in fake_sg._calls if call[0] == 'batch']
    assert len(batch_calls) == 2
    dependent_request = batch_calls[1][1][0][0]
    assert dependent_request['data']['sg_sequence'] == {
        'container': [{'type': 'Sequence', 'id': sequence.id}]
    }


def test_planning_membership_preflight_uses_constant_staging_passes(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()

    def planned_deleted(size: int) -> tuple[Session, _IterationCountingList[Entity]]:
        planning_session = type(session)(engine)
        for entity_id in range(1, size + 1):
            obj = planning_session.merge_dict({'type': 'Shot', 'id': entity_id})
            planning_session.delete(obj)
        deleted = _IterationCountingList(planning_session.unit_of_work.deleted)
        planning_session.unit_of_work._deleted = deleted

        plan = planning_session._plan_commit()

        assert plan._waves
        return planning_session, deleted

    small_session, small_deleted = planned_deleted(8)
    large_session, large_deleted = planned_deleted(64)

    assert large_deleted._iterations == small_deleted._iterations
    small_session.close()
    large_session.close()


def test_planning_precomputes_repeated_exact_identity_memberships(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    binding = engine.prepared_binding()
    assert binding is not None
    candidates: list[Entity] = []
    for index in range(8):
        obj = session.merge_dict({'type': 'Shot', 'id': index + 1, 'description': 'old'})
        obj.description = 'new'
        candidates.append(obj)

    deleted = _IterationCountingList(candidates)
    _uow._persistent_write_operations(candidates, deleted, binding.schemas, binding)

    uploaded = _IterationCountingList([_uow.UploadPlanItem(candidates[0], 'image', 'new.jpg')])
    ordinary = _IterationCountingList([candidates[1]])
    remaining = _IterationCountingList(candidates)
    normal, final = _uow._remaining_retirement_operations(
        remaining,
        uploaded,
        ordinary,
        binding,
    )

    assert deleted._iterations == 1
    assert remaining._iterations == 1
    assert uploaded._iterations == 1
    assert ordinary._iterations == 1
    assert [operation.obj for operation in normal] == candidates[2:]
    assert len(final) == 1
    assert final[0].obj is candidates[0]


def test_provider_commit_error_retires_attempted_upload_without_rewriting_error(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 97})
    image_path = tmp_path / 'provider-commit-error.jpg'
    filmstrip_path = tmp_path / 'provider-commit-error-filmstrip.jpg'
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
    fake_sg._errors['upload_thumbnail'] = [fault]

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert excinfo.value.__cause__ is cause
    assert excinfo.value.committed is partitions[0]
    assert excinfo.value.staged is partitions[1]
    assert excinfo.value.indeterminate is partitions[2]
    assert excinfo.value.withdrawn is partitions[3]
    assert entity_state(shot).dirty == {}
    assert session.unit_of_work.deleted == ()
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before
    assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 1


def test_readme_thumbnail_assignment_clear_and_readback(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 88,
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )
    image_path = tmp_path / 'shot.jpg'
    filmstrip_path = tmp_path / 'shot-filmstrip.jpg'
    image_path.write_bytes(b'image')
    filmstrip_path.write_bytes(b'filmstrip')
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 88,
                'image': 'provider-image',
                'filmstrip_image': 'provider-filmstrip',
            }
        ]
    ]

    shot.image = image_path
    shot.filmstrip_image = filmstrip_path
    session.commit()

    assert shot.image == 'provider-image'
    assert shot.filmstrip_image == 'provider-filmstrip'

    shot.image = None
    session.commit()

    assert shot.image is None
    assert shot.filmstrip_image is None


def test_recovery_base_exception_propagates_without_resend(sess: _SyncSessionBundle) -> None:
    class RecoveryAbort(BaseException):
        pass

    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 23, 'description': 'old'})
    shot.description = 'new'
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 23}]]
    fault = RecoveryAbort()
    fake_sg._find_error = fault

    with pytest.raises(RecoveryAbort) as excinfo:
        session.commit()

    assert excinfo.value is fault
    state = entity_state(shot)
    assert state.dirty == {}
    assert state.clean == {}
    assert state.dirty_types == {}
    assert state.refresh_required == {'description'}
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before


def test_recovery_groups_by_batch_step_entity_type_and_retired_only(
    sess: _SyncSessionBundle,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    active = session.merge_dict({'type': 'Shot', 'id': 21, 'description': 'old-active'})
    asset = session.merge_dict({'type': 'Asset', 'id': 22, 'code': 'old-asset'})
    retired = session.merge_dict({'type': 'Shot', 'id': 23, 'description': 'old-retired'})
    active.description = 'active-secret'
    asset.code = 'asset-secret'
    retired.description = 'retired-secret'
    session.delete(retired)
    fake_sg._batch_results = [
        [
            {'type': 'Shot', 'id': 21},
            {'type': 'Asset', 'id': 22},
            {'type': 'Shot', 'id': 23},
            True,
        ]
    ]
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 21, 'description': 'active-normalized'}],
        [{'type': 'Asset', 'id': 22, 'code': 'asset-normalized'}],
        [{'type': 'Shot', 'id': 23, 'description': 'retired-normalized'}],
    ]

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        session.commit()

    finds = [call for call in fake_sg._calls if call[0] == 'find']
    assert len(finds) == 3
    assert [(call[1][0], call[2].get('retired_only', False)) for call in finds] == [
        ('Shot', False),
        ('Asset', False),
        ('Shot', True),
    ]
    assert active.description == 'active-normalized'
    assert asset.code == 'asset-normalized'
    assert entity_state(retired).loaded['description'] == 'retired-normalized'
    warnings = [record for record in caplog.records if record.levelname == 'WARNING']
    assert len(warnings) == 3
    assert all('provider-response contract' in record.getMessage() for record in warnings)
    assert all('secret' not in record.getMessage() for record in warnings)


def test_recovery_hydration_failure_restores_missing_field_marker(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 23, 'code': 'stable', 'description': 'old'})
    shot.description = 'new'
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 23}]]
    fake_sg._find_results = [[{'type': 'Shot', 'id': 23, 'description': 'normalized'}]]
    original_hydrate = session.hydrate_record

    def partially_hydrate(
        entity_type: str,
        record: WireRecord,
        binding: RegistryBinding | None,
    ) -> Entity:
        if 'description' not in record:
            return original_hydrate(entity_type, record, binding)
        state = entity_state(shot)
        state.loaded['description'] = 'partial'
        state.refresh_required.discard('description')
        raise RuntimeError('synthetic recovery hydration failure')

    monkeypatch.setattr(session, 'hydrate_record', partially_hydrate)

    session.commit()

    state = entity_state(shot)
    assert state.loaded == {'id': 23, 'code': 'stable'}
    assert state.refresh_required == {'description'}


def test_recovery_is_bounded_to_each_batch_step(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = session.merge_dict({'type': 'Shot', 'id': 21, 'description': 'old-first'})
    second = session.merge_dict({'type': 'Shot', 'id': 22, 'description': 'old-second'})
    first.description = 'first'
    second.description = 'second'
    fake_sg._batch_results = [
        [{'type': 'Shot', 'id': 21}],
        [{'type': 'Shot', 'id': 22}],
    ]
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 21, 'description': 'normalized-first'}],
        [{'type': 'Shot', 'id': 22, 'description': 'normalized-second'}],
    ]

    session.commit(batch_size=1)

    finds = [call for call in fake_sg._calls if call[0] == 'find']
    assert len(finds) == 2
    assert [call[1][1] for call in finds] == [
        [['id', 'in', [21]]],
        [['id', 'in', [22]]],
    ]


def test_recovery_rejects_noncanonical_identity(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 1, 'description': 'old'})
    shot.description = 'new'
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 1}]]
    fake_sg._find_results = [[{'type': 'Shot', 'id': True, 'description': 'wrong-identity'}]]

    session.commit()

    state = entity_state(shot)
    assert 'description' not in state.loaded
    assert state.refresh_required == {'description'}


def test_revive_failure_remains_staged_without_indeterminate_state(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 12, 'code': 'retired'})
    entity_state(shot).deleted = True
    session.revive(shot)
    pending = engine.entities.Shot.new(code='after-revive')
    session.add(pending)
    fault = RuntimeError('revive fault')
    fake_sg._errors['revive'] = [fault]
    with pytest.raises(CommitError) as excinfo:
        session.commit()
    assert excinfo.value.__cause__ is fault
    assert excinfo.value.committed == []
    assert shot in excinfo.value.staged
    assert len(excinfo.value.staged) == 2
    assert excinfo.value.indeterminate == []
    assert _uow._BATCH_OUTCOME_UNKNOWN_NOTE not in excinfo.value.__notes__
    assert _uow._BATCH_UNSENT_NOTE not in excinfo.value.__notes__
    assert entity_state(shot).deleted
    session.commit()
    assert not entity_state(shot).deleted
    assert pending.id is not None


def test_revive_reconciliation_failure_is_normalized(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 13, 'code': 'retired'})
    entity_state(shot).deleted = True
    session.revive(shot)
    fault = SgalchemyError('revive reconciliation fault')

    def fail_registration(_obj: Entity) -> Never:
        raise fault

    monkeypatch.setattr(session, 'register_identity', fail_registration)

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert error.__cause__ is fault
    assert error.committed == []
    assert error.staged == [shot]
    assert error.indeterminate == []
    assert error.withdrawn == []
    assert [name for name, _, _ in fake_sg._calls].count('revive') == 1


@pytest.mark.parametrize('size', [100, 200, 400])
@pytest.mark.parametrize('category', ['new', 'deleted'])
def test_single_chunk_staging_visits_are_linear(
    sess: _SyncSessionBundle,
    category: str,
    size: int,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    expected_objects: list[Entity] = []
    expected_requests: list[WireRecord] = []
    results: list[BatchItem] = []
    for index in range(size):
        if category == 'new':
            obj = engine.entities.Shot.new(code=f'shot-{index}')
            session.add(obj)
            expected_requests.append(
                {
                    'request_type': 'create',
                    'entity_type': 'Shot',
                    'data': {'code': f'shot-{index}'},
                }
            )
            results.append({'type': 'Shot', 'id': index + 1, 'code': f'shot-{index}'})
        else:
            obj = session.merge_dict({'type': 'Shot', 'id': index + 1})
            session.delete(obj)
            expected_requests.append(
                {'request_type': 'delete', 'entity_type': 'Shot', 'entity_id': index + 1}
            )
            results.append(True)
        expected_objects.append(obj)
    queue = YieldCountingEntities(expected_objects)
    setattr(session.unit_of_work, f'_{category}', queue)
    fake_sg._batch_results = [results]

    session.commit()

    visits = queue._visits
    assert getattr(session.unit_of_work, f'_{category}') is queue
    assert queue == []
    assert [call[1][0] for call in fake_sg._calls if call[0] == 'batch'] == [expected_requests]
    assert all(entity_state(obj).dirty == {} for obj in expected_objects)
    # Both categories need five planning passes and one compaction. Deletes
    # also enter the validation identity snapshot; the fresh echo snapshot is empty.
    assert visits <= (6 if category == 'new' else 7) * size


@pytest.mark.parametrize('size', [100, 200, 400])
def test_pending_relationship_planning_visits_are_linear(
    sess: _SyncSessionBundle,
    size: int,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    sequences = [engine.entities.Sequence.new(code=f'sequence-{index}') for index in range(size)]
    shots = [
        engine.entities.Shot.new(code=f'shot-{index}', sg_sequence=sequence)
        for index, sequence in enumerate(sequences)
    ]
    expected_objects = [*sequences, *shots]
    for obj in expected_objects:
        session.add(obj)
    queue = YieldCountingEntities(expected_objects)
    session.unit_of_work._new = queue
    calls_before = len(fake_sg._calls)

    plan = session._plan_commit()

    visits = queue._visits
    assert [[id(operation.obj) for operation in wave] for wave in plan._waves] == [
        [id(obj) for obj in sequences],
        [id(obj) for obj in shots],
    ]
    assert [id(obj) for obj in queue] == [id(obj) for obj in expected_objects]
    assert len(fake_sg._calls) == calls_before
    # Candidate collection, authoritative validation, schema participants and
    # two operation constructions need five passes, independent of link count.
    assert visits <= 5 * len(expected_objects)


@pytest.mark.parametrize('category', ['new', 'deleted'])
def test_repeated_chunk_retirement_preserves_order_without_rescanning(
    sess: _SyncSessionBundle,
    category: _uow._StagingCategory,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    objects: list[Entity] = []
    for index in range(6):
        if category == 'new':
            obj = engine.entities.Shot.new(code=f'shot-{index}')
            session.add(obj)
        else:
            obj = session.merge_dict({'type': 'Shot', 'id': index + 1})
            session.delete(obj)
        objects.append(obj)
    queue = YieldCountingEntities(objects)
    setattr(session.unit_of_work, f'_{category}', queue)
    confirmed = (objects[1], objects[3])
    expected = (objects[0], objects[2], objects[4], objects[5])
    chunk = tuple(
        _uow._Op(
            'create' if category == 'new' else 'delete',
            'Shot',
            entity_state(obj).id,
            dict(entity_state(obj).dirty),
            obj,
        )
        for obj in confirmed
    )

    _uow._retire_batch_chunk_provenance(session, chunk)

    assert queue._visits == len(objects)
    _uow._retire_batch_chunk_provenance(session, chunk)
    assert queue._visits == len(objects)
    assert getattr(session.unit_of_work, f'_{category}') is queue
    assert [id(obj) for obj in queue] == [id(obj) for obj in expected]
    unit_of_work = _uow._session_unit_of_work(session.unit_of_work)
    assert all(not unit_of_work._contains_staged(category, obj) for obj in confirmed)
    assert all(unit_of_work._contains_staged(category, obj) for obj in expected)


@pytest.mark.parametrize(
    ('retired_only', 'expected_extra'),
    [(False, {}), (True, {'retired_only': True})],
)
def test_confirmed_field_read_shape_is_sorted_and_unique(
    sess: _SyncSessionBundle,
    retired_only: bool,
    expected_extra: dict[str, bool],
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    later = session.merge_dict({'type': 'Shot', 'id': 22})
    earlier = session.merge_dict({'type': 'Shot', 'id': 11})
    group = _uow.ConfirmedFieldReadGroup(
        entity_type='Shot',
        retired_only=retired_only,
        binding=None,
        _reads=(
            _uow.ConfirmedFieldRead(
                later,
                ('sg_status_list', 'code'),
                ('sg_status_list',),
            ),
            _uow.ConfirmedFieldRead(
                earlier,
                ('description', 'code'),
                ('code', 'description'),
            ),
            _uow.ConfirmedFieldRead(later, ('code',), ('sg_status_list',)),
        ),
    )

    field_codes, missing_field_codes = _uow.confirmed_field_read_codes(group)
    call = _uow.confirmed_field_read_call(group)

    assert field_codes == ['code', 'description', 'sg_status_list']
    assert missing_field_codes == ['code', 'description', 'sg_status_list']
    assert isinstance(call, FindCall)
    assert call.entity_type == 'Shot'
    assert call.filters == [['id', 'in', [11, 22]]]
    assert call.fields == ['code', 'description', 'sg_status_list']
    assert call.extra == expected_extra


@pytest.mark.parametrize('echoes_image', [False, True])
def test_confirmed_thumbnail_clear_is_established_before_echo_hydration(
    sess: _SyncSessionBundle,
    caplog: pytest.LogCaptureFixture,
    echoes_image: bool,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 24,
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )
    state = entity_state(shot)
    state.refresh_required.update({'image', 'filmstrip_image'})
    state.unverified_writes.update({'image', 'filmstrip_image'})
    shot.image = None
    fake_sg._batch_results = [
        [
            {
                'type': 'Shot',
                'id': 24,
                **({'image': None} if echoes_image else {}),
            }
        ]
    ]

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        session.commit()

    assert state.loaded['image'] is None
    assert state.loaded['filmstrip_image'] is None
    assert state.refresh_required == set()
    assert state.unverified_writes == set()
    assert not [call for call in fake_sg._calls if call[0] == 'find']
    warnings = [record for record in caplog.records if record.levelname == 'WARNING']
    assert len(warnings) == (0 if echoes_image else 1)
    if warnings:
        assert 'image' in warnings[0].getMessage()


@pytest.mark.parametrize(
    'case',
    [
        (1, None, [], 0),
        (1, None, [True], 0),
        (2, None, [{'type': 'Shot', 'id': 9101, 'code': 'S0'}, True], 0),
        (2, 1, [], 1),
    ],
)
def test_malformed_successful_batch_is_not_automatically_retried(
    sess: _SyncSessionBundle,
    case: tuple[int, int | None, list[BatchItem], int],
) -> None:
    entity_count, batch_size, results, expected_staged = case
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = [engine.entities.Shot.new(code=f'S{index}') for index in range(entity_count)]
    for obj in pending:
        session.add(obj)
    fake_sg._batch_results = [results]
    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=batch_size)
    assert excinfo.value.committed == []
    submitted_count = entity_count - expected_staged
    assert len(excinfo.value.indeterminate) == submitted_count
    assert len(excinfo.value.staged) == expected_staged
    assert all(entity_state(obj).id is None for obj in pending[:submitted_count])
    with pytest.raises(SgalchemyError, match='indeterminate'):
        _ = pending[0].code
    batches_before = [call for call in fake_sg._calls if call[0] == 'batch']
    session.commit(batch_size=batch_size)
    batches_after = [call for call in fake_sg._calls if call[0] == 'batch']
    assert len(batches_after) == len(batches_before) + expected_staged
    if expected_staged:
        assert pending[-1].id is not None


@pytest.mark.parametrize(
    ('find_results', 'find_error'),
    [
        ([], None),
        ([{'type': 'Shot', 'id': 23}], None),
        ([{'type': 'Asset', 'id': 23, 'description': 'wrong-owner'}], None),
        ([], RuntimeError('synthetic recovery failure')),
    ],
    ids=['missing-row', 'missing-field', 'malformed-row', 'provider-error'],
)
def test_recovery_failure_keeps_only_missing_fields_refresh_required(
    sess: _SyncSessionBundle,
    caplog: pytest.LogCaptureFixture,
    find_results: WireRecords,
    find_error: Exception | None,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 23, 'code': 'stable', 'description': 'old'})
    shot.description = 'submitted-secret'
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 23}]]
    fake_sg._find_results = [find_results]
    fake_sg._find_error = find_error

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        session.commit()

    state = entity_state(shot)
    assert state.loaded == {'id': 23, 'code': 'stable'}
    assert state.refresh_required == {'description'}
    assert state.dirty == {}
    assert len([call for call in fake_sg._calls if call[0] == 'find']) == 1
    warnings = [record for record in caplog.records if record.levelname == 'WARNING']
    assert len(warnings) == 1
    assert 'description' in warnings[0].getMessage()
    assert 'submitted-secret' not in warnings[0].getMessage()
    if find_error is not None:
        errors = [record for record in caplog.records if record.levelname == 'ERROR']
        assert len(errors) == 1
        assert errors[0].exc_info is not None
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before


def test_sole_batch_failure_before_provider_call_reports_unsent(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='unsent')
    session.add(pending)
    fault = RuntimeError('serialize')

    def fail_serialization(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(_session, 'serialize_chunk', fail_serialization)

    with pytest.raises(RuntimeError) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert fault.__notes__ == [
        _uow.SOLE_BATCH_STAGED_NOTE,
        _uow._BATCH_UNSENT_NOTE,
        'sgalchemy: failed during batch wave 1 chunk 1',
    ]
    assert 'batch' not in [name for name, _, _ in fake_sg._calls]
    assert session.unit_of_work.new == (pending,)


@pytest.mark.parametrize(
    'fault',
    [
        _authentication_fault('unconfirmed local authentication fault'),
        RuntimeError('local validation fault'),
        CommitError('local commit error'),
    ],
    ids=['authentication-fault', 'runtime-error', 'commit-error'],
)
def test_sole_batch_local_failure_after_response_is_indeterminate(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
    fault: Exception,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='unconfirmed-local')
    session.add(pending)

    def fail_validation(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(_uow, '_validate_batch_results', fail_validation)

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    error = excinfo.value
    assert error is not fault
    assert error.__cause__ is fault
    assert error.staged == []
    assert all(isinstance(item, _uow._Op) for item in error.indeterminate)
    assert [item.obj for item in error.indeterminate if isinstance(item, _uow._Op)] == [pending]
    assert session.unit_of_work.new == ()
    assert entity_state(pending).indeterminate
    assert any(
        'discard affected objects and reconcile server state' in note for note in error.__notes__
    )
    if type(fault).__name__ == 'AuthenticationFault':
        assert AUTH_RECOVERY_NOTE in error.__notes__

    session.commit()
    assert [name for name, _, _ in fake_sg._calls].count('batch') == 1


def test_sole_batch_provider_failure_passes_through_and_retries(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='sole-batch')
    session.add(pending)
    fault = _authentication_fault('sole batch authentication fault')
    fake_sg._errors['batch'] = [fault]

    with pytest.raises(RuntimeError) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert type(excinfo.value).__name__ == 'AuthenticationFault'
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__notes__ == [
        AUTH_RECOVERY_NOTE,
        _uow.SOLE_BATCH_STAGED_NOTE,
        _uow._BATCH_OUTCOME_UNKNOWN_NOTE,
        'sgalchemy: failed during batch wave 1 chunk 1',
    ]
    assert session.unit_of_work.new == (pending,)
    assert session.unit_of_work.deleted == ()
    assert session.unit_of_work.revived == ()
    assert entity_state(pending).dirty == {'code': 'sole-batch'}
    session.commit()
    assert pending.id is not None
    assert [name for name, _, _ in fake_sg._calls].count('batch') == 2


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
def test_successful_upload_readback_failure_requires_refresh(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readback: list[dict[str, object]],
    raises: bool,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 82, 'image': 'old-image'})
    image_path = tmp_path / 'successful-unreadable.jpg'
    image_path.write_bytes(b'image')
    shot.image = image_path
    fake_sg._find_results = [readback]
    if raises:
        fake_sg._find_error = RuntimeError('readback fault')
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

    session.commit()

    state = entity_state(shot)
    assert 'image' not in state.loaded
    assert state.refresh_required == {'image'}
    assert state.unverified_writes == set()
    assert len(confirmed) == 1
    upload, outcome = confirmed[0]
    assert outcome.committed[0] is upload
    assert outcome.indeterminate == []
    assert [name for name, _, _ in fake_sg._calls].count('find') == 1


@pytest.mark.parametrize('server_value', [None, 'server-image'])
def test_successful_upload_reads_exact_provider_value(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    server_value: object,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 81, 'image': 'old-image'})
    image_path = tmp_path / 'successful-upload.jpg'
    image_path.write_bytes(b'image')
    shot.image = image_path
    fake_sg._find_results = [[{'type': 'Shot', 'id': 81, 'image': server_value}]]

    session.commit()

    state = entity_state(shot)
    assert state.loaded['image'] is server_value
    assert state.refresh_required == set()
    assert state.unverified_writes == set()
    assert state.dirty == {}
    find_calls = [call for call in fake_sg._calls if call[0] == 'find']
    assert len(find_calls) == 1
    assert find_calls[0][1][:3] == ('Shot', [['id', 'in', [81]]], ['image'])


def test_successful_uploads_are_confirmed_before_later_provider_calls(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = session.merge_dict({'type': 'Shot', 'id': 83})
    second = session.merge_dict({'type': 'Shot', 'id': 84})
    first_path = tmp_path / 'first-immediate.jpg'
    second_path = tmp_path / 'second-immediate.jpg'
    first_path.write_bytes(b'image')
    second_path.write_bytes(b'image')
    first.image = first_path
    second.image = second_path
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': 83, 'image': 'first-server'},
            {'type': 'Shot', 'id': 84, 'image': 'second-server'},
        ]
    ]
    original_upload = fake_sg.upload_thumbnail
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

    def observe_second_upload(
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
        return original_upload(entity_type, entity_id, path)

    monkeypatch.setattr(CommitOutcome, '_confirm', capture_confirm)
    monkeypatch.setattr(fake_sg, 'upload_thumbnail', observe_second_upload)

    session.commit()

    assert first.image == 'first-server'
    assert second.image == 'second-server'


def test_sync_batch_base_exception_after_confirmation_stays_committed(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='confirmed')
    session.add(pending)
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 9301, 'code': 'confirmed'}]]
    original_confirm = CommitOutcome._confirm
    fault = KeyboardInterrupt('echo hydration interrupted')
    confirmed_items: tuple[Entity | _uow._Op | _uow.UploadPlanItem, ...] = ()
    active_outcome: CommitOutcome | None = None

    def capture_confirm(
        outcome: CommitOutcome,
        items: tuple[Entity | _uow._Op | _uow.UploadPlanItem, ...],
    ) -> None:
        nonlocal active_outcome, confirmed_items
        original_confirm(outcome, items)
        active_outcome = outcome
        confirmed_items = items

    def interrupt_hydration(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(CommitOutcome, '_confirm', capture_confirm)
    monkeypatch.setattr(session, 'hydrate_record', interrupt_hydration)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert excinfo.value.__cause__ is None
    assert active_outcome is not None
    assert len(confirmed_items) == 1
    assert active_outcome.committed[0] is confirmed_items[0]
    assert active_outcome.indeterminate == []
    assert pending.id == 9301
    assert pending.code == 'confirmed'
    batch_calls = [call for call in fake_sg._calls if call[0] == 'batch']

    session.commit()

    assert [call for call in fake_sg._calls if call[0] == 'batch'] == batch_calls


def test_sync_batch_base_exception_before_submission_remains_retryable(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='not-submitted')
    session.add(pending)
    original_serialize = _session.serialize_chunk
    fault = KeyboardInterrupt('serialization interrupted')
    calls = 0

    def interrupt_first_serialization(call: Callable[_P, _R]) -> Callable[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise fault
            return call(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        _session,
        'serialize_chunk',
        interrupt_first_serialization(original_serialize),
    )

    with pytest.raises(KeyboardInterrupt) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert not entity_state(pending).indeterminate
    assert [call for call in fake_sg._calls if call[0] == 'batch'] == []

    session.commit()

    assert pending.id is not None
    assert [call for call in fake_sg._calls if call[0] == 'batch']


def test_sync_batch_base_exception_is_not_automatically_retried(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='interrupted')
    session.add(pending)
    original_batch = fake_sg.batch

    def interrupt_after_submission(requests: WireRecords) -> Never:
        original_batch(requests)
        raise KeyboardInterrupt('batch interrupted')

    monkeypatch.setattr(fake_sg, 'batch', interrupt_after_submission)

    with pytest.raises(KeyboardInterrupt):
        session.commit()

    assert entity_state(pending).indeterminate
    batch_calls = [call for call in fake_sg._calls if call[0] == 'batch']
    with pytest.raises(SgalchemyError, match='indeterminate'):
        session.commit()
    assert [call for call in fake_sg._calls if call[0] == 'batch'] == batch_calls


def test_sync_batch_lock_entry_interruption_remains_retryable(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='lock-entry-interrupted')
    session.add(pending)
    fault = KeyboardInterrupt('lock entry interrupted')
    entries = 0

    class InterruptFirstEntry:
        def __enter__(self) -> None:
            nonlocal entries
            entries += 1
            if entries == 1:
                raise fault

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(engine, '_call_lock', InterruptFirstEntry())

    with pytest.raises(KeyboardInterrupt) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert excinfo.value.__cause__ is None
    assert not entity_state(pending).indeterminate
    assert [call for call in fake_sg._calls if call[0] == 'batch'] == []

    session.commit()

    assert pending.id is not None
    assert [call for call in fake_sg._calls if call[0] == 'batch']


def test_sync_upload_base_exception_retires_local_retry_without_readback(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 98})
    image_path = tmp_path / 'system-exit-upload.jpg'
    filmstrip_path = tmp_path / 'system-exit-filmstrip.jpg'
    image_path.write_bytes(b'image')
    filmstrip_path.write_bytes(b'image')
    shot.image = image_path
    shot.filmstrip_image = filmstrip_path
    session.delete(shot)
    fault = SystemExit('upload interrupted')
    fake_sg._errors['upload_thumbnail'] = [fault]

    with pytest.raises(SystemExit) as excinfo:
        session.commit()

    assert excinfo.value is fault
    assert entity_state(shot).dirty == {}
    assert session.unit_of_work.deleted == ()
    assert not [call for call in fake_sg._calls if call[0] == 'find']
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before
    assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 1


def test_thin_echo_for_same_batch_retirement_uses_retired_read(
    sess: _SyncSessionBundle,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 23, 'description': 'old'})
    shot.description = 'new'
    session.delete(shot)
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 23}, True]]
    fake_sg._find_results = [[{'type': 'Shot', 'id': 23, 'description': 'normalized'}]]

    with caplog.at_level('WARNING', logger='sgalchemy.session'):
        session.commit()

    find = next(call for call in fake_sg._calls if call[0] == 'find')
    assert find[2]['retired_only'] is True
    assert entity_state(shot).loaded['description'] == 'normalized'
    assert 'provider-response contract' in caplog.text


def test_thumbnail_clear_survives_unrelated_echo_hydration_failure(
    sess: _SyncSessionBundle,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 25,
            'description': 'old-description',
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )
    shot.description = 'new-description'
    shot.image = None
    fake_sg._batch_results = [
        [
            {
                'type': 'Shot',
                'id': 25,
                'description': 'new-description',
                'image': None,
            }
        ]
    ]

    def fail_hydration(*_args: object) -> Never:
        raise RuntimeError('mixed hydration fault')

    monkeypatch.setattr(session, 'hydrate_record', fail_hydration)
    with caplog.at_level('ERROR', logger='sgalchemy.session'):
        session.commit()

    state = entity_state(shot)
    assert state.loaded['image'] is None
    assert state.loaded['filmstrip_image'] is None
    assert state.refresh_required == {'description'}
    assert 'description' not in state.loaded


@pytest.mark.parametrize('error_type', [RuntimeError, CommitError])
def test_uninvoked_later_upload_recovers_confirmed_prefix_only(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[RuntimeError | CommitError],
) -> None:
    session, engine, provider = sess
    engine.load_schema()
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
    with pytest.raises(CommitError) as excinfo:
        session.commit()
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
    session.commit()
    uploads = [args[1] for name, args, _ in provider._calls if name == 'upload_thumbnail']
    assert uploads == [88, 89, 90]


def test_unsent_upload_acquisition_failure_preserves_intent_for_retry(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    failing = False
    attempts = 0
    fault = RuntimeError('upload client unavailable')

    def creator() -> ClosableShotgunLike:
        nonlocal attempts
        attempts += 1
        if failing:
            raise fault
        return cast('ClosableShotgunLike', fake_sg)

    engine = create_engine(
        fake_sg.base_url,
        creator=creator,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    engine.load_schema()
    session = Session(engine)
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
    engine.dispose()
    failing = True
    fake_sg._calls.clear()
    with pytest.raises(CommitError) as excinfo:
        session.commit()
    assert excinfo.value.__cause__ is fault
    assert attempts == 2
    assert fake_sg._calls == []
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
    fake_sg._find_results = [[{'type': 'Shot', 'id': 90, 'image': 'uploaded'}]]
    session.commit()
    assert [name for name, _, _ in fake_sg._calls] == ['upload_thumbnail', 'find']
    assert shot.image == 'uploaded'
    engine.dispose()


def test_upload_ambiguity_clears_dirty_type_provenance(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 14, 'code': 'upload'})
    image_path = tmp_path / 'thumb.jpg'
    filmstrip_path = tmp_path / 'filmstrip.jpg'
    image_path.write_bytes(b'image')
    filmstrip_path.write_bytes(b'image')
    shot.image = image_path
    shot.filmstrip_image = filmstrip_path
    fake_sg._errors['upload_thumbnail'] = [RuntimeError('upload fault')]

    with pytest.raises(CommitError):
        session.commit()

    assert entity_state(shot).dirty_types == {}
    session.commit()
    assert entity_state(shot).dirty_types == {}
    shot.image = image_path
    assert entity_state(shot).dirty_types == {'image': 'image'}


def test_upload_failure_partitions_and_clears_local_paths(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 13, 'code': 'upload'})
    image_path = tmp_path / 'thumb.jpg'
    filmstrip_path = tmp_path / 'filmstrip.jpg'
    image_path.write_bytes(b'image')
    filmstrip_path.write_bytes(b'image')
    shot.image = image_path
    shot.filmstrip_image = filmstrip_path
    session.delete(shot)
    fault = RuntimeError('upload fault')
    fake_sg._errors['upload_thumbnail'] = [fault]
    with pytest.raises(CommitError) as excinfo:
        session.commit()
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
    assert any('refresh each affected thumbnail field' in note for note in excinfo.value.__notes__)
    assert not any(
        'readable provider value marked unverified' in note for note in excinfo.value.__notes__
    )
    assert any('explicitly restage withdrawn work' in note for note in excinfo.value.__notes__)
    state = entity_state(shot)
    assert 'image' not in state.loaded
    assert 'filmstrip_image' not in state.loaded
    assert state.refresh_required == {'image'}
    assert state.unverified_writes == set()
    assert state.dirty == {}
    assert [name for name, _, _ in fake_sg._calls].count('find') == 1


def test_upload_phase_runs_after_batches(sess: _SyncSessionBundle, tmp_path: Path) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot_cls = engine.registry.entity_class('Shot')
    image_path = tmp_path / 'thumb.jpg'
    image_path.write_bytes(b'image')
    shot = shot_cls.new(code='S1', image=image_path)
    session.add(shot)
    session.commit()
    names = [n for n, _, _ in fake_sg._calls]
    assert names.index('batch') < names.index('upload_thumbnail')
    _, args, _ = fake_sg._calls[names.index('upload_thumbnail')]
    assert args == ('Shot', shot.id, str(image_path))
    session.commit()
    assert [name for name, _, _ in fake_sg._calls].count('upload_thumbnail') == 1


def test_upload_readbacks_group_entities_and_types(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = session.merge_dict({'type': 'Shot', 'id': 85})
    second = session.merge_dict({'type': 'Shot', 'id': 86})
    asset = session.merge_dict({'type': 'Asset', 'id': 87})
    paths = [tmp_path / name for name in ('first.jpg', 'second.jpg', 'asset.jpg')]
    for path in paths:
        path.write_bytes(b'image')
    first.image, second.image, asset.image = paths
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': 85, 'image': 'first-server'},
            {'type': 'Shot', 'id': 86, 'image': 'second-server'},
        ],
        [{'type': 'Asset', 'id': 87, 'image': 'asset-server'}],
    ]

    session.commit()

    find_calls = [call for call in fake_sg._calls if call[0] == 'find']
    assert [call[1][0] for call in find_calls] == ['Shot', 'Asset']
    assert find_calls[0][1][1:3] == ([['id', 'in', [85, 86]]], ['image'])
    assert find_calls[1][1][1:3] == ([['id', 'in', [87]]], ['image'])
    assert first.image == 'first-server'
    assert second.image == 'second-server'
    assert asset.image == 'asset-server'


def test_validated_chunk_hydration_failure_confirms_all_and_continues(
    sess: _SyncSessionBundle,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = engine.entities.Sequence.new(code='first')
    second = engine.entities.Sequence.new(code='second')
    dependent = engine.entities.Shot.new(code='dependent', sg_sequence=second)
    session.add(first)
    session.add(second)
    session.add(dependent)
    fake_sg._batch_results = [
        [
            {'type': 'Sequence', 'id': 9101, 'code': 'normalized-first'},
            {'type': 'Sequence', 'id': 9102, 'code': 'normalized-second'},
        ],
        [
            {
                'type': 'Shot',
                'id': 9103,
                'code': 'normalized-dependent',
                'sg_sequence': {'type': 'Sequence', 'id': 9102},
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
        if record['id'] == 9101:
            confirmed_before_hydration = (
                active_outcome is not None
                and active_outcome.contains_committed(first_chunk)
                and session.unit_of_work._new is new_queue
                and [id(obj) for obj in new_queue] == [id(dependent)]
                and all(entity_state(obj).dirty == {} for obj in (first, second))
            )
            raise RuntimeError('synthetic echo hydration failure')
        return original_hydrate(entity_type, record, binding)

    monkeypatch.setattr(CommitOutcome, '_confirm', capture_confirm)
    monkeypatch.setattr(session, 'hydrate_record', hydrate_with_fault)
    with caplog.at_level('ERROR', logger='sgalchemy.session'):
        session.commit()

    assert confirmed_before_hydration
    first_state = entity_state(first)
    assert first_state.dirty == {}
    assert first_state.clean == {}
    assert first_state.dirty_types == {}
    assert 'code' not in first_state.loaded
    assert first_state.refresh_required == {'code'}
    assert second.code == 'normalized-second'
    assert dependent.code == 'normalized-dependent'
    assert len([call for call in fake_sg._calls if call[0] == 'batch']) == 2
    errors = [record for record in caplog.records if record.levelname == 'ERROR']
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before


def test_validated_chunk_reconciliation_failure_retires_confirmed_creates(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = engine.entities.Sequence.new(code='first')
    second = engine.entities.Sequence.new(code='second')
    session.add(first)
    session.add(second)
    fake_sg._batch_results = [
        [
            {'type': 'Sequence', 'id': 9501, 'code': 'first'},
            {'type': 'Sequence', 'id': 9502, 'code': 'second'},
        ]
    ]
    fault = RuntimeError('synthetic confirmed reconciliation failure')
    original_apply = _uow._apply_deterministic_thumbnail_clear

    def fail_reconciliation(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', fail_reconciliation)

    with pytest.raises(CommitError) as excinfo:
        session.commit()

    assert excinfo.value.__cause__ is fault
    assert len(excinfo.value.committed) == 2
    assert excinfo.value.staged == []
    monkeypatch.setattr(_uow, '_apply_deterministic_thumbnail_clear', original_apply)
    calls_before = len(fake_sg._calls)

    session.commit()

    assert len(fake_sg._calls) == calls_before
    assert session.unit_of_work.new == ()
    assert entity_state(first).dirty == {}
    assert entity_state(second).dirty == {}
