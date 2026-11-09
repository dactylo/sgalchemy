from collections.abc import Callable, Generator
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._options import lazyload
from sgalchemy._core._registry import Registry, entity_state, inspect
from sgalchemy._core._statement import select
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _session, _uow
from sgalchemy._orm._session import Session, commit_session
from sgalchemy._orm._uow import CommitOutcome
from sgalchemy._shotgun import BatchItem, WireRecord
from tests._fakes import FakeShotgun
from tests._support import (
    SAVED_BATCH_NOTE,
    UNFINISHED_BATCH_ERROR,
    UNFINISHED_BATCH_NOTE,
    AcceptedBatchCorpus,
    FailOnceDict,
    FailOnceSet,
    SyncSessionBundle,
    UnclassifiableError,
    assert_accepted_batch_error,
    assert_accepted_batch_settled,
    assert_cleanup_interrupted,
    assert_cleanup_log,
    assert_disjoint_commit_outcome,
    assert_outdated_corpus,
    assert_outdated_error,
    assert_unsent_work_pending,
    inject_faults,
    interrupt_accepted_release,
    plan_item_owners,
    stage_accepted_batch_corpus,
    wave_one_owners,
)

_OUTDATED = (
    'Shot(id=26) is outdated: SG saved your change, but sgalchemy could not '
    'update this copy; discard it and query the record again'
)


_PHASE_ONE_STEPS = [
    pytest.param((_uow, '_write_accepted_ids'), id='ids'),
    pytest.param((CommitOutcome, 'confirm'), id='partition'),
    pytest.param((_uow, '_retire_batch_chunk_provenance'), id='retirement'),
    pytest.param((_uow, '_apply_thumbnail_clears'), id='thumbnail-clear'),
    pytest.param((_uow, '_mark_accepted_field_trust'), id='missing-echo'),
    pytest.param((_uow, '_mark_accepted_retirements'), id='retired'),
    pytest.param((_uow, '_install_accepted_identities'), id='identity'),
]
_FAULTS = [
    pytest.param(RuntimeError, id='ordinary'),
    pytest.param(KeyboardInterrupt, id='interrupt'),
]


class _UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise ValueError('no text')


def _assert_retry(session: Session, fake: FakeShotgun, corpus: AcceptedBatchCorpus) -> None:
    assert_unsent_work_pending(corpus, session._unit_of_work)
    before = len(fake.calls)
    assert session.get(type(corpus.first), 9001) is corpus.first
    assert session.get(type(corpus.second), 9002) is corpus.second
    assert session.get(type(corpus.uploaded), 9003) is corpus.uploaded
    assert len(fake.calls) == before
    session.add(corpus.first)
    session.commit()
    calls = fake.calls[before:]
    batches = [args[0] for name, args, _ in calls if name == 'batch']
    assert batches == [
        [
            {
                'request_type': 'create',
                'entity_type': 'Shot',
                'data': {'code': 'dependent', 'sg_sequence': {'type': 'Sequence', 'id': 9001}},
            }
        ]
    ]
    uploads = {(name, args) for name, args, _ in calls if name.startswith('upload_')}
    assert uploads == {
        ('upload_thumbnail', ('Shot', 9003, str(corpus.thumbnail))),
        ('upload_filmstrip_thumbnail', ('Shot', 53, str(corpus.filmstrip))),
    }


def _fail_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(_uow._install_accepted_identities, repeat(RuntimeError('persistent'))),
    )


def _omit_description(fake: FakeShotgun, monkeypatch: pytest.MonkeyPatch) -> None:
    original = fake.batch

    def batch(requests: list[WireRecord]) -> list[Any]:
        results = original(requests)
        for result in results:
            if isinstance(result, dict):
                record = cast('WireRecord', result)
                if record.get('id') == 51:
                    record.pop('description', None)
        return results

    monkeypatch.setattr(fake, 'batch', batch)


def test_completion_interruption_inside_commit_keeps_original_as_context(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    first = RuntimeError('phase one')
    second = KeyboardInterrupt('completion')
    monkeypatch.setattr(
        _uow,
        '_mark_accepted_retirements',
        inject_faults(_uow._mark_accepted_retirements, [first, second]),
    )
    with pytest.raises(KeyboardInterrupt) as excinfo:
        session.commit()
    assert excinfo.value is second
    assert second.__context__ is first
    assert second.__notes__.index(SAVED_BATCH_NOTE) < second.__notes__.index(UNFINISHED_BATCH_NOTE)


def test_fault_in_later_step_keeps_its_own_note(
    sess: SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    fake_sg.errors['upload_filmstrip_thumbnail'] = [RuntimeError('upload')]
    with pytest.raises(CommitError) as excinfo:
        session.commit()
    assert (
        'sgalchemy: failed during upload_filmstrip_thumbnail for Shot(id=53).filmstrip_image'
        in excinfo.value.__notes__
    )
    assert SAVED_BATCH_NOTE not in excinfo.value.__notes__


def test_final_retirement_batch_fault_names_final_batch(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 80})
    path = tmp_path / 'image.jpg'
    path.write_bytes(b'image')
    shot.image = path
    session.delete(shot)
    monkeypatch.setattr(
        _uow,
        '_mark_accepted_retirements',
        inject_faults(_uow._mark_accepted_retirements, [RuntimeError('retirement')]),
    )
    with pytest.raises(CommitError) as excinfo:
        session.commit()
    assert (
        'sgalchemy: SG saved final retirement batch 1 before this failure; '
        'it counts as committed and a later commit will not send it again'
    ) in excinfo.value.__notes__
    assert entity_state(shot).deleted
    before = len(fake_sg.calls)
    session.commit()
    assert len(fake_sg.calls) == before


def test_interruption_during_gated_retry_propagates_raw(
    sess: SyncSessionBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    interruption = KeyboardInterrupt('retry')
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(
            _uow._install_accepted_identities,
            [RuntimeError('phase one'), RuntimeError('completion'), interruption],
        ),
    )
    with pytest.raises(CommitError):
        session.commit()
    with pytest.raises(KeyboardInterrupt) as excinfo:
        _ = corpus.unrelated.code
    assert excinfo.value is interruption


def test_later_chunk_fault_names_its_chunk(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(_uow._install_accepted_identities, [None, RuntimeError('chunk two')]),
    )
    with pytest.raises(CommitError) as excinfo:
        session.commit(batch_size=4)
    error = excinfo.value
    assert (
        'sgalchemy: SG saved batch wave 1 chunk 2 (batch_size=4) before this failure; '
        'it counts as committed and a later commit will not send it again'
    ) in error.__notes__
    assert not any('failed during' in note for note in error.__notes__)
    assert plan_item_owners(error.committed) == wave_one_owners(corpus)[:-1]
    assert plan_item_owners(error.staged)[0] is corpus.uploaded
    assert_disjoint_commit_outcome(error)


def test_next_operation_finishes_completion_and_keeps_unrelated_work(
    sess: SyncSessionBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(
            _uow._install_accepted_identities,
            [RuntimeError('phase one'), RuntimeError('completion')],
        ),
    )
    with pytest.raises(CommitError):
        session.commit()
    session.merge_dict({'type': 'Shot', 'id': 61})
    assert_accepted_batch_settled(corpus, session._unit_of_work, session._state._identity)
    assert_unsent_work_pending(corpus, session._unit_of_work)


def test_outdated_object_precedes_registry_and_ownership_errors(
    engine: Engine, fake_sg: FakeShotgun, tmp_path: Path
) -> None:
    engine.load_schema()
    session_a = Session(engine)
    session_b = Session(engine)
    shot = session_a.merge_dict({'type': 'Shot', 'id': 26, 'code': 'original'})
    other_engine = create_engine(
        connection=fake_sg, cache_root=tmp_path / 'other', registry=Registry(), bind_default=False
    )
    other_session = Session(other_engine)
    entity_state(shot).outdated = ('Shot', 26)
    provider_calls = len(fake_sg.calls)

    for action in (
        lambda: session_b.add(shot),
        lambda: session_b.delete(shot),
        lambda: session_a.refresh(shot),
        lambda: session_a.delete(shot),
        lambda: session_a.revive(shot),
        lambda: session_a.expire(shot),
        lambda: other_session.add(shot),
    ):
        with pytest.raises(SgalchemyError) as exc_info:
            action()
        assert str(exc_info.value) == _OUTDATED
    assert len(fake_sg.calls) == provider_calls


def test_outdated_relationship_value_rejected_at_commit(
    engine: Engine, fake_sg: FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine)
    sequence = engine.registry.entity_class('Sequence').new(code='SQ01')
    entity_state(sequence).outdated = ('Sequence', 7)
    shot = engine.registry.entity_class('Shot').new(sg_sequence=sequence)
    session.add(shot)
    provider_calls = len(fake_sg.calls)

    with pytest.raises(SgalchemyError) as exc_info:
        session.commit()

    assert str(exc_info.value) == _OUTDATED.replace('Shot(id=26)', 'Sequence(id=7)')
    assert len(fake_sg.calls) == provider_calls


def test_persistent_completion_fault_gates_session_and_entities(
    sess: SyncSessionBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = engine.registry.entity_class('Shot')
    fake_sg.find_results = [[{'type': 'Shot', 'id': 70, 'code': 'lazy'}]]
    lazy_shot = session.scalars(select(shot).options(lazyload(shot.sg_sequence)))[0]
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    before_repr = repr(corpus.unrelated)
    fault = RuntimeError('persistent')
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(_uow._install_accepted_identities, repeat(fault)),
    )
    with pytest.raises(CommitError):
        session.commit()
    before = len(fake_sg.calls)
    actions = (
        lambda: session.add(shot.new(code='later')),
        lambda: session.delete(corpus.unrelated),
        lambda: session.revive(corpus.deleted),
        lambda: session.refresh(corpus.unrelated),
        lambda: session.expire(corpus.unrelated),
        session.expire_all,
        lambda: session.get(shot, 60),
        lambda: session.merge_dict({'type': 'Shot', 'id': 61}),
        lambda: session.scalars(select(shot)),
        lambda: session.execute(select(shot.code)),
        session.commit,
        lambda: corpus.unrelated.code,
        lambda: setattr(corpus.unrelated, 'code', 'later'),
        corpus.unrelated.as_dict,
        lambda: inspect(corpus.unrelated),
        lambda: lazy_shot.sg_sequence,
    )
    for action in actions:
        with pytest.raises(SgalchemyError) as excinfo:
            action()
        assert str(excinfo.value) == UNFINISHED_BATCH_ERROR
        assert excinfo.value.__cause__ is fault
        assert len(fake_sg.calls) == before

    def fail_if_called(*_args: object) -> None:
        pytest.fail('repr attempted completion')

    with monkeypatch.context() as patch:
        patch.setattr(_uow, '_establish_accepted_batch', fail_if_called)
        assert repr(corpus.unrelated) == before_repr


@pytest.mark.parametrize(
    'boundary',
    [
        pytest.param((step.values[0], after), id=f'{step.id}-{after}')
        for step in _PHASE_ONE_STEPS
        for after in (False, True)
    ],
)
@pytest.mark.parametrize('make_fault', _FAULTS)
def test_phase_one_fault_settles_accepted_batch(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: tuple[tuple[object, str], bool],
    make_fault: Callable[[str], BaseException],
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    _omit_description(fake_sg, monkeypatch)
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    (owner, name), after = boundary
    fault = make_fault('phase one fault')
    monkeypatch.setattr(owner, name, inject_faults(getattr(owner, name), [fault], after=after))
    with pytest.raises(CommitError if isinstance(fault, Exception) else type(fault)) as excinfo:
        session.commit()
    assert_accepted_batch_error(excinfo.value, fault, corpus)
    assert_accepted_batch_settled(corpus, session._unit_of_work, session._state.identity_map)
    _assert_retry(session, fake_sg, corpus)


@pytest.mark.parametrize('position', ['id-write', 'partition', 'staging-index'])
@pytest.mark.parametrize('make_fault', _FAULTS)
def test_phase_one_mid_step_fault_settles_accepted_batch(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    position: str,
    make_fault: Callable[[str], BaseException],
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    _omit_description(fake_sg, monkeypatch)
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    fault = make_fault('phase one fault')
    if position == 'id-write':
        state = entity_state(corpus.first)
        state.loaded = FailOnceDict(state.loaded, 'id', fault)
    elif position == 'partition':
        monkeypatch.setattr(
            CommitOutcome,
            '_remove_staged',
            inject_faults(CommitOutcome._remove_staged, [fault], after=True),
        )
    else:
        session._unit_of_work._new_ids = FailOnceSet(session._unit_of_work._new_ids, fault)
    with pytest.raises(CommitError if isinstance(fault, Exception) else type(fault)) as excinfo:
        session.commit()
    assert_accepted_batch_error(excinfo.value, fault, corpus)
    assert_accepted_batch_settled(corpus, session._unit_of_work, session._state.identity_map)
    _assert_retry(session, fake_sg, corpus)


@pytest.mark.parametrize('boundary', ['after-store', 'before-release', 'after-release'])
@pytest.mark.parametrize('make_fault', _FAULTS)
def test_fault_around_record_lifetime(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    make_fault: Callable[[str], BaseException],
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    _omit_description(fake_sg, monkeypatch)
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    fault = make_fault('phase one fault')
    name = '_accept_batch' if boundary == 'after-store' else '_clear_accepted_record'
    monkeypatch.setattr(
        _uow, name, inject_faults(getattr(_uow, name), [fault], after=boundary != 'before-release')
    )
    established: list[bool] = []
    original = _uow._establish_accepted_batch

    def establish(
        recovery: _uow.RecoveryAdapter,
        batch: _uow._AcceptedBatch,
        outcome: CommitOutcome | None,
        results: list[BatchItem] | None,
    ) -> None:
        established.append(results is None)
        original(recovery, batch, outcome, results)

    monkeypatch.setattr(_uow, '_establish_accepted_batch', establish)
    with pytest.raises(CommitError if isinstance(fault, Exception) else type(fault)) as excinfo:
        session.commit()
    assert_accepted_batch_error(excinfo.value, fault, corpus)
    assert (True in established) == (boundary != 'after-release')
    assert_accepted_batch_settled(
        corpus,
        session._unit_of_work,
        session._state.identity_map,
        completion_ran=boundary != 'after-release',
    )
    _assert_retry(session, fake_sg, corpus)


@pytest.mark.parametrize('boundary', ['validation', 'before-store'])
@pytest.mark.parametrize('make_fault', _FAULTS)
def test_fault_before_acceptance_stays_indeterminate(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    make_fault: Callable[[str], BaseException],
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    fault = make_fault('before acceptance')
    name = '_validate_batch_results' if boundary == 'validation' else '_accept_batch'
    monkeypatch.setattr(
        _uow, name, inject_faults(getattr(_uow, name), [fault], after=boundary == 'validation')
    )
    with pytest.raises(CommitError if isinstance(fault, Exception) else type(fault)) as excinfo:
        session.commit()
    error = excinfo.value
    assert SAVED_BATCH_NOTE not in getattr(error, '__notes__', [])
    if isinstance(error, CommitError):
        assert error.committed == []
        assert plan_item_owners(error.indeterminate) == wave_one_owners(corpus)
        assert plan_item_owners(error.withdrawn) == [
            corpus.restripped,
            corpus.uploaded,
        ]
        assert 'sgalchemy: failed during batch wave 1 chunk 1' in error.__notes__
    else:
        assert error is fault
    assert all(entity_state(obj).indeterminate for obj in wave_one_owners(corpus))
    with pytest.raises(SgalchemyError, match='indeterminate'):
        session.add(corpus.first)
    before = len(fake_sg.calls)
    with pytest.raises(SgalchemyError):
        session.commit()
    assert len(fake_sg.calls) == before


@pytest.mark.parametrize('position', ['ids', 'partition', 'mid-partition', 'identity'])
def test_completion_failure_inside_commit_logs_and_keeps_original_primary(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    position: str,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    fault = RuntimeError('completion')
    first = fault
    if position == 'mid-partition':
        first = RuntimeError('phase one')
        monkeypatch.setattr(
            CommitOutcome,
            '_remove_staged',
            inject_faults(CommitOutcome._remove_staged, [first], after=True),
        )
    owner, name = (
        (CommitOutcome, 'confirm')
        if position == 'partition'
        else (_uow, '_write_accepted_ids' if position == 'ids' else '_install_accepted_identities')
    )
    monkeypatch.setattr(owner, name, inject_faults(getattr(owner, name), repeat(fault)))
    with (
        caplog.at_level('ERROR', logger='sgalchemy.session'),
        pytest.raises(CommitError) as excinfo,
    ):
        session.commit()
    error = excinfo.value
    assert error.__cause__ is first
    assert plan_item_owners(error.committed) == wave_one_owners(corpus)
    assert_disjoint_commit_outcome(error)
    assert error.__notes__.index(SAVED_BATCH_NOTE) < error.__notes__.index(UNFINISHED_BATCH_NOTE)
    records = [record for record in caplog.records if record.levelname == 'ERROR']
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert (
        records[0].getMessage()
        == 'sgalchemy could not finish updating objects after SG saved batch wave 1 chunk 1'
    )
    assert _uow.accepted_batch_pending(session._unit_of_work)


def test_retry_commit_after_unfinished_note_sends_only_unsent_work(
    sess: SyncSessionBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    first = RuntimeError('phase one')
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(_uow._install_accepted_identities, [first, RuntimeError('completion')]),
    )
    with pytest.raises(CommitError) as excinfo:
        session.commit()
    assert excinfo.value.__cause__ is first
    assert SAVED_BATCH_NOTE in excinfo.value.__notes__
    assert UNFINISHED_BATCH_NOTE in excinfo.value.__notes__
    before = len(fake_sg.calls)
    session.commit()
    calls = fake_sg.calls[before:]
    assert [args[0] for name, args, _ in calls if name == 'batch'] == [
        [
            {
                'request_type': 'create',
                'entity_type': 'Shot',
                'data': {'code': 'dependent', 'sg_sequence': {'type': 'Sequence', 'id': 9001}},
            }
        ]
    ]
    assert {(name, args) for name, args, _ in calls if name.startswith('upload_')} == {
        ('upload_thumbnail', ('Shot', 9003, str(corpus.thumbnail))),
        ('upload_filmstrip_thumbnail', ('Shot', 53, str(corpus.filmstrip))),
    }
    assert_accepted_batch_settled(corpus, session._unit_of_work, session._state._identity)
    assert session.unit_of_work.new == []
    assert 'image' not in entity_state(corpus.uploaded).dirty
    assert 'filmstrip_image' not in entity_state(corpus.restripped).dirty


@pytest.mark.parametrize('shape', ['sole-batch', 'corpus'])
@pytest.mark.parametrize('factory', [False, True])
def test_secondary_failure_still_reports_saved_batch(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    factory: bool,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    if shape == 'corpus':
        corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
        owners = wave_one_owners(corpus)
    else:
        obj = engine.registry.entity_class('Shot').new(code='only')
        session.add(obj)
        owners = [obj]
    first, diagnostic = RuntimeError('phase one'), RuntimeError('diagnostic')
    monkeypatch.setattr(
        _uow, '_mark_accepted_field_trust', inject_faults(_uow._mark_accepted_field_trust, [first])
    )
    monkeypatch.setattr(
        _session, 'build_commit_error', inject_faults(_session.build_commit_error, [diagnostic])
    )
    with pytest.raises(CommitError) as excinfo:
        commit_session(session, wrap_ordinary_failure=factory)
    error = excinfo.value
    assert error.__cause__ is diagnostic
    assert diagnostic.__context__ is first
    assert plan_item_owners(error.committed) == owners
    assert SAVED_BATCH_NOTE in error.__notes__
    assert not any(
        'failed during' in note or 'single-batch commit' in note for note in error.__notes__
    )
    before = len(fake_sg.calls)
    session.commit()
    batches = [args[0] for name, args, _ in fake_sg.calls[before:] if name == 'batch']
    if shape == 'sole-batch':
        assert batches == []
    else:
        assert batches == [
            [
                {
                    'request_type': 'create',
                    'entity_type': 'Shot',
                    'data': {'code': 'dependent', 'sg_sequence': {'type': 'Sequence', 'id': 9001}},
                }
            ]
        ]


@pytest.mark.parametrize(
    'mode', [(retained, factory) for retained in (False, True) for factory in (False, True)]
)
@pytest.mark.parametrize(
    'secondary_type', [RuntimeError, KeyboardInterrupt, _UnprintableError, UnclassifiableError]
)
def test_handler_failure_after_saved_batch_reaches_commit_exit(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: tuple[bool, bool],
    secondary_type: Callable[[str], BaseException],
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    _omit_description(fake_sg, monkeypatch)
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    retained, factory = mode
    first, secondary = RuntimeError('phase one'), secondary_type('cleanup')
    name = '_install_accepted_identities' if retained else '_mark_accepted_field_trust'
    monkeypatch.setattr(
        _uow, name, inject_faults(getattr(_uow, name), repeat(first) if retained else [first])
    )
    monkeypatch.setattr(
        Session,
        'recover_upload_fields',
        inject_faults(Session.recover_upload_fields, [secondary]),
    )
    with pytest.raises(
        CommitError if isinstance(secondary, Exception) else type(secondary)
    ) as excinfo:
        commit_session(session, wrap_ordinary_failure=factory)
    error = excinfo.value
    if isinstance(error, CommitError):
        assert error.__cause__ is secondary
        assert isinstance(secondary.__context__, CommitError)
        assert secondary.__context__.__cause__ is first
        assert plan_item_owners(error.committed) == wave_one_owners(corpus)
        assert plan_item_owners(error.staged) == [
            corpus.dependent,
            corpus.restripped,
            corpus.uploaded,
        ]
        assert_disjoint_commit_outcome(error)
        assert str(error) == (
            '<exception str() failed>' if isinstance(secondary, _UnprintableError) else 'cleanup'
        )
    else:
        assert error is secondary
        assert all(not entity_state(obj).indeterminate for obj in wave_one_owners(corpus))
    assert SAVED_BATCH_NOTE in error.__notes__
    assert not any('failed during' in note for note in error.__notes__)
    assert (UNFINISHED_BATCH_NOTE in error.__notes__) == retained
    assert _uow.accepted_batch_pending(session._unit_of_work) == retained
    if not retained:
        assert_accepted_batch_settled(corpus, session._unit_of_work, session._state.identity_map)
        _assert_retry(session, fake_sg, corpus)


@contextmanager
def _cleanup_scope(session: Session, entry: str) -> Generator[None]:
    if entry == 'context-exit':
        with session:
            yield
    else:
        yield
        if entry == 'rollback':
            session.rollback()
        else:
            session.close()


def _commit_and_cleanup(
    session: Session,
    entry: str,
    monkeypatch: pytest.MonkeyPatch | None = None,
    position: str = '',
) -> None:
    with _cleanup_scope(session, entry):
        with pytest.raises(CommitError):
            session.commit()
        if monkeypatch is not None:
            interrupt_accepted_release(monkeypatch, session, position)


@pytest.mark.parametrize('entry', ['rollback', 'close', 'context-exit'])
def test_cleanup_marks_unupdatable_objects_outdated(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    entry: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    _fail_completion(monkeypatch)
    with _cleanup_scope(session, entry), pytest.raises(CommitError):
        session.commit()
    assert_cleanup_log(caplog, entry)
    assert_outdated_corpus(corpus, session)
    with pytest.raises(SgalchemyError) as excinfo:
        Session(engine).add(corpus.first)
    assert_outdated_error(excinfo.value, 'Sequence(id=9001)')
    with pytest.raises(SgalchemyError) as excinfo:
        session.refresh(corpus.updated)
    assert_outdated_error(excinfo.value, 'Shot(id=51)')
    assert session._state.owns_entity(corpus.unrelated) == (entry == 'rollback')
    assert corpus.unrelated.code == 'unrelated'
    fake_sg.find_results = [[{'type': 'Shot', 'id': 51, 'code': 'server'}]]
    fresh = session.get(type(corpus.updated), 51)
    assert fresh is not corpus.updated
    assert fresh is not None
    assert fresh.code == 'server'


@pytest.mark.parametrize('entry', ['rollback', 'close', 'context-exit'])
@pytest.mark.parametrize('position', ['disposition', 'some-marked', 'provenance', 'detached'])
def test_interrupted_cleanup_resumes_release_without_completion(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    position: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    with monkeypatch.context() as faults:
        _fail_completion(faults)
        with pytest.raises(KeyboardInterrupt) as excinfo:
            _commit_and_cleanup(session, entry, faults, position)
        assert_cleanup_interrupted(excinfo.value, entry, session, corpus, position)
    with patch.object(
        _uow, '_establish_accepted_batch', wraps=_uow._establish_accepted_batch
    ) as spy:
        session.rollback()
    spy.assert_not_called()
    assert_outdated_corpus(corpus, session)
    fake_sg.find_results = [[{'type': 'Shot', 'id': 51, 'code': 'server'}]]
    fresh = session.get(type(corpus.updated), 51)
    assert fresh is not corpus.updated
    assert fresh is not None
    assert fresh.code == 'server'


@pytest.mark.parametrize('entry', ['rollback', 'close', 'context-exit'])
def test_interrupted_cleanup_completion_retries_on_next_cleanup(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(
            _uow._install_accepted_identities,
            [
                RuntimeError('phase one'),
                RuntimeError('completion'),
                KeyboardInterrupt('cleanup'),
            ],
        ),
    )
    with pytest.raises(KeyboardInterrupt) as excinfo:
        _commit_and_cleanup(session, entry)
    operation = 'rollback()' if entry == 'rollback' else 'close()'
    assert excinfo.value.__notes__ == [
        (
            f'sgalchemy: {operation} did not finish; call rollback() or close() on this session '
            'again before using it'
        )
    ]
    assert _uow.accepted_batch_pending(session._unit_of_work)
    with patch.object(
        _uow, '_establish_accepted_batch', wraps=_uow._establish_accepted_batch
    ) as spy:
        session.rollback()
    assert spy.call_count == 1
    assert not _uow.accepted_batch_pending(session._unit_of_work)
    assert entity_state(corpus.first).outdated is None
    assert session._state.identity_map[('Sequence', 9001)] is corpus.first


@pytest.mark.parametrize('entry', ['rollback', 'close'])
def test_interrupted_cleanup_completion_recovers_on_next_use(
    sess: SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    corpus = stage_accepted_batch_corpus(session, engine.registry, tmp_path)
    monkeypatch.setattr(
        _uow,
        '_install_accepted_identities',
        inject_faults(
            _uow._install_accepted_identities,
            [
                RuntimeError('phase one'),
                RuntimeError('completion'),
                KeyboardInterrupt('cleanup'),
            ],
        ),
    )
    with pytest.raises(KeyboardInterrupt, match='cleanup'):
        _commit_and_cleanup(session, entry)
    _assert_retry(session, fake_sg, corpus)


def test_sole_batch_phase_one_fault_raises_commit_error(
    sess: SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    obj = engine.registry.entity_class('Shot').new(code='only')
    session.add(obj)
    monkeypatch.setattr(
        _uow,
        '_mark_accepted_field_trust',
        inject_faults(_uow._mark_accepted_field_trust, [RuntimeError('phase one')]),
    )
    with pytest.raises(CommitError) as excinfo:
        session.commit()
    error = excinfo.value
    assert plan_item_owners(error.committed) == [obj]
    assert SAVED_BATCH_NOTE in error.__notes__
    assert not any('single-batch commit' in note for note in error.__notes__)
    before = len(fake_sg.calls)
    session.commit()
    assert len(fake_sg.calls) == before
