import abc
import copy
import functools
import gc
import importlib
import inspect
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import Any, Never, ParamSpec, TypeVar, cast
from weakref import ReferenceType, ref

import pytest
from _fakes import _FakeShotgun
from _support import (
    _SyncCallback,
    _SyncExecutionEnvironment,
    _SyncSessionFactory,
    assert_disjoint_commit_outcome,
    entity_namespace_scope,
    reset_default_registry_state,
)

import sgalchemy
from sgalchemy._core import _registry
from sgalchemy._core._engine import Engine
from sgalchemy._core._registry import (
    Entity,
    Registry,
    entity_registry,
    entity_state,
)
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _scoping, _session, _uow
from sgalchemy._orm._scoping import _ConnectOptions
from sgalchemy._orm._session import Session
from sgalchemy._shotgun import ShotgunLike

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SCOPED_SESSION_METHODS = (
    'add',
    'close',
    'commit',
    'delete',
    'execute',
    'expire',
    'expire_all',
    'get',
    'merge_dict',
    'refresh',
    'revive',
    'rollback',
    'scalars',
)


def _block_call(
    call: _SyncCallback[_P, _R],
    entered: Event,
    release: Event,
) -> _SyncCallback[_P, _R]:

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        entered.set()
        assert release.wait(timeout=10)
        return call(*args, **kwargs)

    return wrapper


def _connect(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    *,
    force: bool = False,
) -> Session:
    return sgalchemy.connect(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
        force=force,
    )


def _execution_error_body(
    environment: _SyncExecutionEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    error: CommitError,
) -> Callable[[Session], None]:
    engine, fake_sg, tmp_path = environment
    if route == 'multi_batch':
        original_batch = fake_sg.batch
        calls = 0

        def fail_second_batch(requests: list[dict[str, Any]]) -> list[Any]:
            nonlocal calls
            if calls == 1:
                fake_sg._errors['batch'] = [error]
            calls += 1
            return original_batch(requests)

        monkeypatch.setattr(fake_sg, 'batch', fail_second_batch)
        sequence = engine.entities.Sequence.new(code='first')
        dependent = engine.entities.Shot.new(code='second', sg_sequence=sequence)

        def stage_batch(session: sgalchemy.Session) -> None:
            session.add(sequence)
            session.add(dependent)

        return stage_batch
    if route == 'revive':
        fake_sg._errors['revive'] = [error]

        def stage_revive(session: sgalchemy.Session) -> None:
            shot = session.merge_dict({'type': 'Shot', 'id': 37, 'image': 'old'})
            entity_state(shot).deleted = True
            session.revive(shot)

        return stage_revive
    upload_path = tmp_path / 'existing-commit-error.jpg'
    upload_path.write_bytes(b'image')
    fake_sg._errors['upload_thumbnail'] = [error]

    def stage_upload(session: sgalchemy.Session) -> None:
        shot = session.merge_dict({'type': 'Shot', 'id': 37, 'image': 'old'})
        shot.image = upload_path

    return stage_upload


def _make_factory(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> tuple[Engine, _SyncSessionFactory]:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    engine.load_schema()
    return engine, sgalchemy.sessionmaker(engine)


def _replacement_schema(schema_fixture: dict[str, Any]) -> dict[str, Any]:
    replacement = copy.deepcopy(schema_fixture)
    new_field = copy.deepcopy(replacement['Shot']['description'])
    new_field['name']['value'] = 'New Field'
    replacement['Shot']['sg_new_field'] = new_field
    del replacement['Shot']['code']
    return replacement


def _run_commit_on_success(
    factory: _SyncSessionFactory,
    body: Callable[[Session], None],
) -> None:
    with factory.commit_on_success() as session:
        body(session)


def _run_direct_commit(
    factory: _SyncSessionFactory,
    body: Callable[[Session], None],
) -> None:
    session = factory()
    try:
        body(session)
        session.commit()
    finally:
        session.close()


def _use_replacement_schema(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    replacement = _replacement_schema(schema_fixture)
    monkeypatch.setattr(
        fake_sg,
        'schema_read',
        lambda: copy.deepcopy(replacement),
    )


@pytest.fixture(autouse=True)
def clean_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with suppress(SgalchemyError):
        _scoping._default_session().close()
    monkeypatch.setattr(_scoping, '_DEFAULT_SCOPED', [])
    reset_default_registry_state(monkeypatch)
    with entity_namespace_scope(('Asset', 'Shot')):
        yield
        with suppress(SgalchemyError):
            _scoping._default_session().close()


def test_admission_base_exception_after_reservation_rolls_back(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    def interrupted_selection(
        url: str | None,
        connection: ShotgunLike | None,
        force: bool,
        engine_kwargs: _ConnectOptions,
        reserved_registry: Registry,
    ) -> Never:
        del url, connection, force, engine_kwargs, reserved_registry
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_scoping, '_select_connection_registry', interrupted_selection)
        with pytest.raises(KeyboardInterrupt):
            sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)

    assert vars(_registry)['_DEFAULT_TRANSITION']['reserved'] is False
    retry = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
    assert retry is _scoping._default_session()


def test_admission_interruption_after_reservation_mutation_recovers_and_retries(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    original_reserve = _scoping.reserve_default_registry_transition

    def interrupt_after_reservation(*args: object) -> Registry:
        original_reserve(*args)
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            _scoping,
            'reserve_default_registry_transition',
            interrupt_after_reservation,
        )
        with pytest.raises(KeyboardInterrupt):
            sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)

    assert vars(_registry)['_DEFAULT_TRANSITION']['reserved'] is False
    retry = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
    assert retry is _scoping._default_session()


def test_admission_interruption_during_reservation_mutation_recovers_and_retries(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InterruptingTransition(dict[str, object]):
        def __init__(self) -> None:
            super().__init__(reserved=False)
            self._interrupt = True

        def __setitem__(self, key: str, value: object) -> None:
            super().__setitem__(key, value)
            if self._interrupt and key == 'reserved' and value is True:
                raise KeyboardInterrupt

    transition = InterruptingTransition()
    monkeypatch.setattr(_registry, '_DEFAULT_TRANSITION', transition)

    with pytest.raises(KeyboardInterrupt):
        sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)

    assert vars(_registry)['_DEFAULT_TRANSITION']['reserved'] is False
    assert transition['reserved'] is False
    transition._interrupt = False
    retry = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
    assert retry is _scoping._default_session()


@pytest.mark.parametrize('route', ['multi_batch', 'revive', 'upload'])
@pytest.mark.parametrize('through_factory', [False, True], ids=['direct', 'factory'])
def test_commit_error_from_execution_passes_through_unchanged(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
    through_factory: bool,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
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
    body = _execution_error_body(
        (engine, fake_sg, tmp_path),
        monkeypatch,
        route,
        existing,
    )
    run_commit = _run_commit_on_success if through_factory else _run_direct_commit

    with pytest.raises(CommitError) as excinfo:
        run_commit(factory, body)

    assert excinfo.value is existing
    assert existing.__cause__ is cause
    assert existing.committed is partitions[0]
    assert existing.staged is partitions[1]
    assert existing.indeterminate is partitions[2]
    assert existing.withdrawn is partitions[3]
    expected_notes = ('provider note',)
    if through_factory:
        expected_notes += (_scoping._COMMIT_FAILURE_CLOSE_NOTE,)
    assert tuple(existing.__notes__) == expected_notes
    if route == 'upload':
        names = [name for name, _args, _kwargs in fake_sg._calls]
        assert names.count('upload_thumbnail') == 1
        assert 'find' not in names


def test_commit_failure_after_same_site_refresh_preserves_transition_state(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    entities = importlib.import_module('sgalchemy.entities')
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    registry = first.engine.registry
    shot_cls = cast('type[Entity]', entities.Shot)
    pairing_before = list(vars(_registry)['_DEFAULT_SESSION_REGISTRIES'])
    replacement_sg = type(fake_sg)()
    _use_replacement_schema(replacement_sg, monkeypatch, schema_fixture)
    original_commit = _scoping.commit_default_registry_transition

    def reject_inside_real_commit(
        target: Registry,
        *,
        default_bound: bool,
        publish: Callable[[], None],
    ) -> None:
        _registry.release_default_registry_transition(_registry._DEFAULT_TRANSITION['owner'])
        original_commit(
            target,
            default_bound=default_bound,
            publish=publish,
        )

    monkeypatch.setattr(
        _scoping,
        'commit_default_registry_transition',
        reject_inside_real_commit,
    )

    with pytest.raises(SgalchemyError, match='no default registry transition'):
        sgalchemy.connect(
            connection=replacement_sg,
            cache_root=tmp_path / 'second',
            force=True,
        )

    assert _registry.default_registry() is registry
    assert vars(_registry)['_DEFAULT_SESSION_REGISTRIES'] == pairing_before
    assert _scoping._default_session() is first
    assert entities.Shot is shot_cls
    snapshot = registry.schema_snapshot()
    assert snapshot is not None
    assert 'sg_new_field' in snapshot['Shot'].fields
    refreshed = shot_cls.new(sg_new_field='refresh remains visible')
    first.add(refreshed)
    with pytest.raises(AttributeError, match='code'):
        refreshed.code = 'removed'

    monkeypatch.setattr(
        _scoping,
        'commit_default_registry_transition',
        original_commit,
    )
    retry = sgalchemy.connect(
        connection=replacement_sg,
        cache_root=tmp_path / 'second',
        force=True,
    )

    assert retry.engine.registry is registry
    assert _scoping._default_session() is retry
    assert entities.Shot is shot_cls
    assert vars(_registry)['_DEFAULT_SESSION_REGISTRIES'] == [registry]


def test_commit_failure_releases_transition_and_retry_succeeds(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current = _registry.default_registry()
    original = _scoping.commit_default_registry_transition
    attempts = 0

    def fail_once(
        registry: Registry,
        *,
        default_bound: bool,
        publish: Callable[[], None],
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('commit failed')
        original(registry, default_bound=default_bound, publish=publish)

    monkeypatch.setattr(_scoping, 'commit_default_registry_transition', fail_once)

    with pytest.raises(RuntimeError, match='commit failed'):
        sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)

    assert _registry.default_registry() is current
    with pytest.raises(SgalchemyError, match='connect'):
        _scoping._default_session()
    retry = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
    assert retry.engine.registry is current


@pytest.mark.parametrize('exception_type', [RuntimeError, KeyboardInterrupt])
def test_commit_on_success_body_failure_skips_commit_and_stays_primary(
    exception_type: type[BaseException],
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = exception_type('body failed')
    pending = engine.entities.Shot.new(code='discarded')
    close_calls: list[sgalchemy.Session] = []
    original_close = sgalchemy.Session.close

    def tracking_close(session: sgalchemy.Session) -> None:
        close_calls.append(session)
        original_close(session)

    def fail_body(session: sgalchemy.Session) -> None:
        session.add(pending)
        raise fault

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sgalchemy.Session, 'close', tracking_close)
        with pytest.raises(exception_type) as excinfo:
            _run_commit_on_success(factory, fail_body)

    assert excinfo.value is fault
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 0
    assert len(close_calls) == 1
    assert entity_state(pending).session is None
    assert not entity_state(pending).dirty


def test_commit_on_success_body_failure_stays_primary_when_close_fails(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    body_fault = RuntimeError('body failed')
    close_fault = RuntimeError('close failed')
    original_close = sgalchemy.Session.close
    close_calls = 0

    def failing_close(session: sgalchemy.Session) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(session)
        raise close_fault

    monkeypatch.setattr(sgalchemy.Session, 'close', failing_close)
    pending = engine.entities.Shot.new(code='discarded')

    def fail_body(session: sgalchemy.Session) -> None:
        session.add(pending)
        raise body_fault

    with pytest.raises(RuntimeError) as excinfo:
        _run_commit_on_success(factory, fail_body)

    assert excinfo.value is body_fault
    assert close_calls == 1
    assert entity_state(pending).session is None
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 0
    assert any('close failed' in note for note in body_fault.__notes__)


def test_commit_on_success_close_failure_propagates_inside_outer_except(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    close_fault = RuntimeError('close failed')
    original_close = sgalchemy.Session.close

    def failing_close(session: sgalchemy.Session) -> None:
        original_close(session)
        raise close_fault

    monkeypatch.setattr(sgalchemy.Session, 'close', failing_close)
    outer = ValueError('outer failure')

    def close_inside_outer_except() -> None:
        try:
            raise outer
        except ValueError:
            with factory.commit_on_success() as session:
                session.add(engine.entities.Shot.new(code='outer-close'))

    with pytest.raises(RuntimeError, match='close failed'):
        close_inside_outer_except()

    assert getattr(outer, '__notes__', []) == []
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1


def test_commit_on_success_commit_base_exception_stays_raw_and_closes(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = SystemExit('commit interrupted')
    fake_sg._errors['batch'] = [fault]
    pending = engine.entities.Shot.new(code='interrupted')

    with pytest.raises(SystemExit) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

    assert excinfo.value is fault
    assert excinfo.value.__cause__ is None
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1
    assert entity_state(pending).session is None
    assert entity_state(pending).indeterminate
    assert any('explicitly restage' in note for note in fault.__notes__)


def test_commit_on_success_commit_failure_stays_primary_when_close_fails(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    commit_fault = RuntimeError('commit failed')
    close_fault = RuntimeError('close failed')
    fake_sg._errors['batch'] = [commit_fault]
    original_close = sgalchemy.Session.close
    close_calls = 0

    def failing_close(session: sgalchemy.Session) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(session)
        raise close_fault

    monkeypatch.setattr(sgalchemy.Session, 'close', failing_close)
    pending = engine.entities.Shot.new(code='unsent')

    with pytest.raises(CommitError) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

    error = excinfo.value
    assert error.__cause__ is commit_fault
    assert close_calls == 1
    assert entity_state(pending).session is None
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1
    assert any('explicitly restage' in note for note in error.__notes__)
    assert any('close failed' in note for note in error.__notes__)


def test_commit_on_success_commits_once_and_closes(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    close_calls: list[sgalchemy.Session] = []
    original_close = sgalchemy.Session.close

    def tracking_close(session: sgalchemy.Session) -> None:
        close_calls.append(session)
        original_close(session)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sgalchemy.Session, 'close', tracking_close)
        with factory.commit_on_success() as session:
            pending = engine.entities.Shot.new(code='created')
            session.add(pending)

    assert pending.id is not None
    assert entity_state(pending).session is None
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1
    assert close_calls == [session]


def test_commit_on_success_does_not_claim_to_undo_inner_commit(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = RuntimeError('body failed after explicit commit')
    pending = engine.entities.Shot.new(code='already committed')

    def commit_then_fail(session: sgalchemy.Session) -> None:
        session.add(pending)
        session.commit()
        raise fault

    with pytest.raises(RuntimeError) as excinfo:
        _run_commit_on_success(factory, commit_then_fail)

    assert excinfo.value is fault
    assert pending.id is not None
    assert entity_state(pending).session is None
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1
    assert not any('rollback' in note.lower() for note in getattr(fault, '__notes__', ()))


def test_commit_on_success_keeps_multi_step_failure_partitions(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = RuntimeError('second batch fault')
    original_batch = fake_sg.batch
    calls = 0

    def fail_second_batch(requests: list[dict[str, Any]]) -> list[Any]:
        nonlocal calls
        if calls == 1:
            fake_sg._errors['batch'] = [fault]
        calls += 1
        return original_batch(requests)

    monkeypatch.setattr(fake_sg, 'batch', fail_second_batch)
    sequence = engine.entities.Sequence.new(code='first')
    dependent = engine.entities.Shot.new(code='second', sg_sequence=sequence)

    def stage_dependency(session: sgalchemy.Session) -> None:
        session.add(sequence)
        session.add(dependent)

    with pytest.raises(CommitError) as excinfo:
        _run_commit_on_success(factory, stage_dependency)

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


def test_commit_on_success_preserves_existing_commit_error(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    pending = engine.entities.Shot.new(code='existing evidence')
    cause = RuntimeError('provider evidence')
    existing = CommitError('existing commit error', staged=[pending])
    existing.__cause__ = cause
    attempts = 0

    def fail_plan(_session_instance: Session) -> Never:
        nonlocal attempts
        attempts += 1
        raise existing

    monkeypatch.setattr(sgalchemy.Session, '_plan_commit', fail_plan)

    with pytest.raises(CommitError) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

    assert excinfo.value is existing
    assert existing.__cause__ is cause
    assert existing.committed == []
    assert existing.staged == [pending]
    assert existing.indeterminate == []
    assert existing.withdrawn == []
    assert attempts == 1
    assert entity_state(pending).session is None
    assert any('explicitly restage' in note for note in existing.__notes__)


def test_commit_on_success_propagates_close_failure_when_it_is_sole_failure(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = RuntimeError('close failed')
    original_close = sgalchemy.Session.close
    close_calls = 0

    def failing_close(session: sgalchemy.Session) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(session)
        raise fault

    monkeypatch.setattr(sgalchemy.Session, 'close', failing_close)
    pending = engine.entities.Shot.new(code='committed before close')

    with pytest.raises(RuntimeError) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

    assert excinfo.value is fault
    assert close_calls == 1
    assert pending.id is not None
    assert entity_state(pending).session is None
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1


def test_commit_on_success_success_inside_outer_except_adds_no_note(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    outer = RuntimeError('outer failure')

    try:
        raise outer
    except RuntimeError:
        with factory.commit_on_success() as session:
            pending = engine.entities.Shot.new(code='outer-context')
            session.add(pending)

    assert pending.id is not None
    assert getattr(outer, '__notes__', []) == []
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1


def test_commit_on_success_wraps_schema_preparation_failure_without_replanning(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    registry = Registry()
    source = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'source',
        registry=registry,
        bind_default=False,
    )
    source.load_schema()
    destination_sg = type(fake_sg)()
    destination = sgalchemy.create_engine(
        connection=destination_sg,
        cache_root=tmp_path / 'destination',
        registry=registry,
        bind_default=False,
    )
    factory = sgalchemy.sessionmaker(destination)
    pending = source.entities.Shot.new(code='cold staged intent')
    fault = RuntimeError('schema preparation failed')
    destination_sg._errors['schema_read'] = [fault]

    with pytest.raises(CommitError) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

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


def test_commit_on_success_wraps_sole_batch_failure_reports_unsent(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = RuntimeError('serialize')

    def fail_serialization(*_args: object) -> Never:
        raise fault

    monkeypatch.setattr(_session, 'serialize_chunk', fail_serialization)
    pending = engine.entities.Shot.new(code='unsent')

    with pytest.raises(CommitError) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

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
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 0
    assert entity_state(pending).session is None
    assert not entity_state(pending).dirty
    assert any(
        'if this session remains open and usable, a later commit may retry' in note
        for note in error.__notes__
    )
    assert any('explicitly restage' in note for note in error.__notes__)


def test_commit_on_success_wraps_sole_batch_failure_with_failure_time_evidence(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory(fake_sg, tmp_path)
    fault = RuntimeError('sole batch fault')
    fake_sg._errors['batch'] = [fault]
    pending = engine.entities.Shot.new(code='unsent')

    with pytest.raises(CommitError) as excinfo:
        _run_commit_on_success(factory, lambda session: session.add(pending))

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
    assert [name for name, _args, _kwargs in fake_sg._calls].count('batch') == 1
    assert entity_state(pending).session is None
    assert not entity_state(pending).dirty
    assert any(
        'if this session remains open and usable, a later commit may retry' in note
        for note in error.__notes__
    )
    assert any('explicitly restage' in note for note in error.__notes__)


@pytest.mark.parametrize(
    'scenario',
    [
        (True, True, True, 'connection is establishing'),
        (True, False, True, 'force=True'),
        (False, False, False, 'connection is establishing'),
    ],
)
def test_concurrent_connect_combinations_admit_exactly_one(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    scenario: tuple[bool, bool, bool, str],
    tmp_path: Path,
) -> None:
    first_force, second_force, installed_first, loser_message = scenario
    if installed_first:
        sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'installed')
    active_sg = type(fake_sg)()
    rejected_sg = type(fake_sg)()
    third_sg = type(fake_sg)()
    entered = Event()
    release = Event()
    original = active_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=10)
        return original()

    monkeypatch.setattr(active_sg, 'schema_read', blocking_schema_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(
            sgalchemy.connect,
            connection=active_sg,
            cache_root=tmp_path / 'active',
            force=first_force,
        )
        try:
            assert entered.wait(timeout=2)
            loser = pool.submit(
                sgalchemy.connect,
                connection=rejected_sg,
                cache_root=tmp_path / 'loser',
                force=second_force,
            )
            with pytest.raises(SgalchemyError, match=loser_message):
                loser.result(timeout=2)
            with pytest.raises(SgalchemyError, match='connection is establishing'):
                sgalchemy.connect(
                    connection=third_sg,
                    cache_root=tmp_path / 'third',
                    force=True,
                )
        finally:
            release.set()
        installed = active.result(timeout=2)

    assert _scoping._default_session().engine is installed.engine
    assert rejected_sg._calls == []
    assert third_sg._calls == []


def test_concurrent_non_force_connect_rejects_loser_before_provider_io(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    schema_entered = Event()
    release_schema = Event()
    first_schema_read = fake_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        schema_entered.set()
        assert release_schema.wait(timeout=10)
        return first_schema_read()

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    create_engine = _scoping.create_engine
    engine_connections: list[object] = []

    def track_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            engine_connections.append(kwargs.get('connection'))
            return call(*args, **kwargs)

        return wrapper

    tracking_create_engine = track_call(create_engine)

    monkeypatch.setattr(_scoping, 'create_engine', tracking_create_engine)
    rejected_sg = type(fake_sg)()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_connect, fake_sg, tmp_path / 'first')
        try:
            assert schema_entered.wait(timeout=2)
            second = pool.submit(_connect, rejected_sg, tmp_path / 'second')
            with pytest.raises(SgalchemyError, match='connection is establishing'):
                second.result(timeout=2)
        finally:
            release_schema.set()
        installed = first.result(timeout=2)

    assert _scoping._default_session().engine is installed.engine
    assert engine_connections == [fake_sg]
    assert rejected_sg._calls == []


def test_connect_installs_default_session_and_finder_supports_aliases(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A'}]]
    session = _connect(fake_sg, tmp_path)
    assert session is _scoping._default_session()
    shot_cls = session.engine.registry.entity_class('Shot')

    shots = shot_cls.find(shot_cls.sg_cut_duration > 1, status_list='ip', vendor=None)

    assert shots[0].code == 'A'
    _name, args, _kwargs = next(call for call in fake_sg._calls if call[0] == 'find')
    assert args[1] == [
        ['sg_cut_duration', 'greater_than', 1],
        ['sg_status_list', 'is', 'ip'],
        ['sg_vendor', 'is', None],
    ]


def test_connect_rejects_second_install_before_io_and_force_preserves_old_session(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    first = _connect(fake_sg, tmp_path / 'first')
    shot_cls = first.engine.entities.Shot
    pending = shot_cls.new(code='pending')
    first.add(pending)
    replacement_sg = type(fake_sg)()

    with pytest.raises(SgalchemyError, match='force=True'):
        _connect(replacement_sg, tmp_path / 'second')
    assert replacement_sg._calls == []
    assert _scoping._default_session() is first
    assert entity_state(pending).session is first

    replacement = _connect(replacement_sg, tmp_path / 'second', force=True)

    assert _scoping._default_session() is replacement
    assert replacement.engine.connection is replacement_sg
    assert entity_state(pending).session is first
    later = shot_cls.new(code='later')
    first.add(later)
    assert entity_state(later).session is first


def test_default_session_stays_prompt_during_first_connect_provider_io(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    original = fake_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=10)
        return original()

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        connecting = pool.submit(
            sgalchemy.connect,
            connection=fake_sg,
            cache_root=tmp_path,
        )
        try:
            assert entered.wait(timeout=2)
            lookup = pool.submit(_scoping._default_session)
            with pytest.raises(SgalchemyError, match='connect'):
                lookup.result(timeout=2)
        finally:
            release.set()
        connecting.result(timeout=2)


def test_default_session_stays_prompt_during_force_provider_io(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    replacement_sg = type(fake_sg)()
    entered = Event()
    release = Event()
    original = replacement_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=10)
        return original()

    monkeypatch.setattr(replacement_sg, 'schema_read', blocking_schema_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replacing = pool.submit(
            sgalchemy.connect,
            connection=replacement_sg,
            cache_root=tmp_path / 'second',
            force=True,
        )
        try:
            assert entered.wait(timeout=2)
            lookup = pool.submit(_scoping._default_session)
            assert lookup.result(timeout=2).engine is first.engine
        finally:
            release.set()
        replacing.result(timeout=2)


def test_default_session_without_connect_teaches() -> None:
    with pytest.raises(SgalchemyError, match=r'sgalchemy\.connect'):
        _scoping._default_session()


def test_different_site_force_after_materialization_teaches_without_provider_io(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    entities = importlib.import_module('sgalchemy.entities')
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    registry = first.engine.registry
    shot_cls = entities.Shot
    replacement_sg = type(fake_sg)()
    replacement_sg.base_url = 'https://other.example.com'

    with pytest.raises(SgalchemyError, match=r'engine\.entities|registry=Registry'):
        sgalchemy.connect(
            connection=replacement_sg,
            cache_root=tmp_path / 'second',
            force=True,
        )

    assert replacement_sg._calls == []
    assert _registry.default_registry() is registry
    assert _scoping._default_session() is first
    assert entities.Shot is shot_cls


def test_different_site_force_with_empty_class_cache_publishes_fresh_registry(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    old_registry = first.engine.registry
    replacement_sg = type(fake_sg)()
    replacement_sg.base_url = 'https://other.example.com'

    replacement = sgalchemy.connect(
        connection=replacement_sg,
        cache_root=tmp_path / 'second',
        force=True,
    )

    assert replacement.engine.registry is not old_registry
    assert _registry.default_registry() is replacement.engine.registry
    assert _scoping._default_session() is replacement
    assert first.engine.registry is old_registry


def test_entity_find_without_installed_callback_teaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_registry, '_ENTITY_FINDERS', [])
    shot_cls = Registry().entity_class('Shot')
    with pytest.raises(SgalchemyError, match=r'sgalchemy\.connect'):
        shot_cls.find()


def test_entity_namespace_scope_clears_materialized_class(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    with entity_namespace_scope(('Shot',)) as entities:
        session = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
        try:
            _ = entities.Shot
        finally:
            session.close()

    assert 'Shot' not in vars(entities)


def test_explicit_default_registry_spelling_is_reserved_and_paired(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = _registry.default_registry()
    entered = Event()
    release = Event()
    original = fake_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=10)
        return original()

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        connecting = pool.submit(
            sgalchemy.connect,
            connection=fake_sg,
            cache_root=tmp_path,
            registry=registry,
        )
        try:
            assert entered.wait(timeout=2)
            with pytest.raises(SgalchemyError, match='connection is establishing'):
                _registry.set_default_registry(Registry())
        finally:
            release.set()
        installed = connecting.result(timeout=2)

    assert installed.engine.registry is registry
    with pytest.raises(SgalchemyError, match=r'connect\(force=True\)'):
        _registry.set_default_registry(Registry())


def test_failed_force_connect_keeps_old_default_and_state(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    first = _connect(fake_sg, tmp_path / 'first')
    pending = first.engine.entities.Shot.new(code='pending')
    first.add(pending)
    replacement_sg = type(fake_sg)()
    replacement_sg._errors['schema_read'] = [RuntimeError('schema unavailable')]

    with ThreadPoolExecutor(max_workers=1) as pool:
        failed = pool.submit(_connect, replacement_sg, tmp_path / 'second', force=True)
        with pytest.raises(RuntimeError, match='schema unavailable'):
            failed.result(timeout=2)

    assert _scoping._default_session() is first
    assert entity_state(pending).session is first


def test_failure_after_same_site_refresh_keeps_refresh_and_old_scoped_default(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    registry = first.engine.registry
    replacement_sg = type(fake_sg)()
    _use_replacement_schema(replacement_sg, monkeypatch, schema_fixture)
    real_scoped_session = _scoping.scoped_session

    class FailingScoped:
        def __call__(self) -> sgalchemy.Session:
            raise RuntimeError('session construction failed')

    def failing_scoped_session(_factory: Callable[[], sgalchemy.Session]) -> FailingScoped:
        return FailingScoped()

    monkeypatch.setattr(_scoping, 'scoped_session', failing_scoped_session)

    with pytest.raises(RuntimeError, match='session construction failed'):
        sgalchemy.connect(
            connection=replacement_sg,
            cache_root=tmp_path / 'second',
            force=True,
        )

    assert _scoping._default_session() is first
    snapshot = registry.schema_snapshot()
    assert snapshot is not None
    assert 'sg_new_field' in snapshot['Shot'].fields
    monkeypatch.setattr(_scoping, 'scoped_session', real_scoped_session)
    retry = sgalchemy.connect(
        connection=replacement_sg,
        cache_root=tmp_path / 'second',
        force=True,
    )
    assert retry.engine.registry is registry


def test_find_unknown_equality_field_fails_before_provider_find(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    session = _connect(fake_sg, tmp_path)
    shot_cls = session.engine.registry.entity_class('Shot')
    find_calls_before = sum(name == 'find' for name, _args, _kwargs in fake_sg._calls)

    with pytest.raises(AttributeError, match='no field or field alias'):
        shot_cls.find(not_a_field='value')

    assert sum(name == 'find' for name, _args, _kwargs in fake_sg._calls) == find_calls_before


def test_force_connect_does_not_call_old_session_close(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _connect(fake_sg, tmp_path / 'first')
    first_close = first.close
    close_attempted = False

    def failing_close() -> None:
        nonlocal close_attempted
        close_attempted = True
        raise RuntimeError('old Session close must remain caller-owned')

    monkeypatch.setattr(first, 'close', failing_close)
    try:
        replacement = _connect(
            type(fake_sg)(),
            tmp_path / 'second',
            force=True,
        )
    finally:
        monkeypatch.setattr(first, 'close', first_close)

    assert _scoping._default_session() is replacement
    assert not close_attempted
    first.close()


@pytest.mark.parametrize('registry_mode', ['explicit', 'private'])
def test_force_connect_nondefault_registry_preserves_process_default(
    fake_sg: _FakeShotgun,
    registry_mode: str,
    tmp_path: Path,
) -> None:
    _connect(fake_sg, tmp_path / 'first')
    process_default = _registry.default_registry()
    if registry_mode == 'explicit':
        replacement = sgalchemy.connect(
            connection=type(fake_sg)(),
            cache_root=tmp_path / 'second',
            force=True,
            registry=Registry(),
        )
    else:
        replacement = sgalchemy.connect(
            connection=type(fake_sg)(),
            cache_root=tmp_path / 'second',
            force=True,
            bind_default=False,
        )

    assert _registry.default_registry() is process_default
    assert replacement.engine.registry is not process_default


def test_force_connect_preserves_sessions_owned_by_other_threads(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    first = _connect(fake_sg, tmp_path / 'first')
    shot_cls = first.engine.entities.Shot
    main_pending = shot_cls.new(code='main pending')
    first.add(main_pending)
    replacement_sg = type(fake_sg)()
    worker_ready = Event()
    release_worker = Event()

    def use_old_worker_session() -> tuple[sgalchemy.Session, Entity, Entity]:
        old_session = _scoping._default_session()
        worker_pending = shot_cls.new(code='worker pending')
        old_session.add(worker_pending)
        worker_ready.set()
        assert release_worker.wait(timeout=10)
        assert entity_state(worker_pending).session is old_session
        later = shot_cls.new(code='worker later')
        old_session.add(later)
        assert _scoping._default_session().engine.connection is replacement_sg
        return old_session, worker_pending, later

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(use_old_worker_session)
        try:
            assert worker_ready.wait(timeout=2)
            replacement = _connect(
                replacement_sg,
                tmp_path / 'second',
                force=True,
            )
            assert entity_state(main_pending).session is first
            main_later = shot_cls.new(code='main later')
            first.add(main_later)
        finally:
            release_worker.set()
        worker_session, worker_pending, worker_later = worker.result(timeout=2)

    assert _scoping._default_session() is replacement
    assert entity_state(main_later).session is first
    assert entity_state(worker_pending).session is worker_session
    assert entity_state(worker_later).session is worker_session


def test_in_flight_old_query_finishes_while_force_replaces_future_default(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _connect(fake_sg, tmp_path / 'first')
    old_shot_cls = first.engine.entities.Shot
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'old'}]]
    find_entered = Event()
    release_find = Event()
    original_find = fake_sg.find
    closed_sessions: list[sgalchemy.Session] = []
    original_close = sgalchemy.Session.close

    blocking_find = _block_call(original_find, find_entered, release_find)

    def tracking_close(session: sgalchemy.Session) -> None:
        closed_sessions.append(session)
        original_close(session)

    monkeypatch.setattr(fake_sg, 'find', blocking_find)
    monkeypatch.setattr(sgalchemy.Session, 'close', tracking_close)

    def query_old_default() -> tuple[sgalchemy.Session, Entity]:
        old_session = _scoping._default_session()
        result = old_session.scalars(sgalchemy.select(old_shot_cls))
        return old_session, result[0]

    with ThreadPoolExecutor(max_workers=1) as pool:
        query = pool.submit(query_old_default)
        try:
            assert find_entered.wait(timeout=2)
            replacement_sg = type(fake_sg)()
            replacement_sg._find_results = [[{'type': 'Shot', 'id': 2, 'code': 'new'}]]
            replacement = _connect(
                replacement_sg,
                tmp_path / 'second',
                force=True,
            )
            replacement_shot_cls = replacement.engine.entities.Shot
            found = replacement_shot_cls.find()
            assert entity_state(found[0]).session is replacement
            assert _scoping._default_session() is replacement
        finally:
            release_find.set()
        old_session, old_shot = query.result(timeout=2)

    assert first not in closed_sessions
    assert old_session not in closed_sessions
    assert entity_state(old_shot).session is old_session
    assert old_shot.code == 'old'


def test_interrupted_connect_releases_admission_and_retry_succeeds(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    fake_sg._errors['schema_read'] = [KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)

    assert vars(_registry)['_DEFAULT_TRANSITION']['reserved'] is False
    with pytest.raises(SgalchemyError, match='connect'):
        _scoping._default_session()
    retry = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
    assert retry is _scoping._default_session()


def test_interrupted_force_connect_keeps_old_default_and_allows_retry(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    replacement_sg = type(fake_sg)()
    replacement_sg._errors['schema_read'] = [KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        sgalchemy.connect(
            connection=replacement_sg,
            cache_root=tmp_path / 'second',
            force=True,
        )

    assert vars(_registry)['_DEFAULT_TRANSITION']['reserved'] is False
    assert _scoping._default_session() is first
    retry = sgalchemy.connect(
        connection=type(fake_sg)(),
        cache_root=tmp_path / 'third',
        force=True,
    )
    assert retry is _scoping._default_session()


def test_preparation_failure_releases_transition_and_retry_succeeds(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    fake_sg._errors['schema_read'] = [RuntimeError('preparation failed')]

    with pytest.raises(RuntimeError, match='preparation failed'):
        sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)

    with pytest.raises(SgalchemyError, match='connect'):
        _scoping._default_session()
    replacement = Registry()
    _registry.set_default_registry(replacement)
    retry = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path)
    assert retry.engine.registry is replacement


def test_private_default_session_can_repair_pairing_before_materialization(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    first = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'first')
    process_registry = first.engine.registry
    private = Registry()
    replacement = sgalchemy.connect(
        connection=type(fake_sg)(),
        cache_root=tmp_path / 'private',
        force=True,
        registry=private,
    )

    with pytest.raises(SgalchemyError, match=r'connect\(force=True\)'):
        _registry.set_default_registry(Registry())
    _registry.set_default_registry(private)

    assert _registry.default_registry() is private
    assert replacement.engine.entities.Shot is private.entity_class('Shot')
    shot_cls = replacement.engine.entities.Shot
    pending = shot_cls.new(code='paired')
    replacement.add(pending)
    assert entity_state(pending).session is replacement
    assert process_registry is not private


def test_private_default_session_stays_split_until_force_recovery(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    entities = importlib.import_module('sgalchemy.entities')
    initial = sgalchemy.connect(connection=fake_sg, cache_root=tmp_path / 'initial')
    process_registry = initial.engine.registry
    private_registry = Registry()
    private = sgalchemy.connect(
        connection=type(fake_sg)(),
        cache_root=tmp_path / 'private',
        force=True,
        registry=private_registry,
    )
    outgoing_shot_cls = cast('type[Entity]', entities.Shot)
    foreign = outgoing_shot_cls.new(code='foreign')

    assert entity_registry(outgoing_shot_cls) is process_registry
    with pytest.raises(SgalchemyError, match='different registry'):
        private.add(foreign)
    assert entity_state(foreign).session is None
    with pytest.raises(SgalchemyError, match='cannot be replaced'):
        _registry.set_default_registry(private_registry)

    assert _registry.default_registry() is process_registry
    assert _scoping._default_session() is private
    assert private.engine.registry is private_registry
    assert entities.Shot is outgoing_shot_cls

    recovered = sgalchemy.connect(
        connection=type(fake_sg)(),
        cache_root=tmp_path / 'recovered',
        force=True,
    )

    assert _registry.default_registry() is process_registry
    assert _scoping._default_session() is recovered
    assert recovered.engine.registry is process_registry
    assert entities.Shot is outgoing_shot_cls
    recovered.add(foreign)
    assert entity_state(foreign).session is recovered


def test_same_site_force_reuses_registry_and_refreshes_old_sessions(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    entities = importlib.import_module('sgalchemy.entities')
    first = sgalchemy.connect(
        connection=fake_sg,
        cache_root=tmp_path / 'first',
    )
    registry = first.engine.registry
    shot_cls = cast('type[Entity]', entities.Shot)
    old_pending = shot_cls.new(code='old')
    first.add(old_pending)
    replacement_sg = type(fake_sg)()
    _use_replacement_schema(replacement_sg, monkeypatch, schema_fixture)
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'sg_new_field': 'old session'}]]
    replacement_sg._find_results = [[{'type': 'Shot', 'id': 2, 'sg_new_field': 'new session'}]]

    replacement = sgalchemy.connect(
        connection=replacement_sg,
        cache_root=tmp_path / 'second',
        force=True,
    )

    assert replacement.engine.registry is registry
    assert _registry.default_registry() is registry
    assert _scoping._default_session() is replacement
    assert entities.Shot is shot_cls
    assert replacement.engine.entities.Shot is shot_cls
    assert entity_state(old_pending).session is first
    old_pending.sg_new_field = 'refreshed'
    with pytest.raises(AttributeError, match='code'):
        old_pending.code = 'removed'
    new_pending = shot_cls.new(sg_new_field='new')
    replacement.add(new_pending)
    old_found = first.scalars(sgalchemy.select(shot_cls))[0]
    new_found = replacement.scalars(sgalchemy.select(shot_cls))[0]
    assert old_found.sg_new_field == 'old session'
    assert new_found.sg_new_field == 'new session'


def test_scoped_session_does_not_retain_dead_thread_session(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    scoped = sgalchemy.scoped_session(sgalchemy.sessionmaker(engine))

    def create_worker_session() -> ReferenceType[sgalchemy.Session]:
        return ref(scoped())

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_session = pool.submit(create_worker_session).result(timeout=2)

    gc.collect()
    assert worker_session() is None


def test_scoped_session_dynamic_fallback_delegates_unknown_public_method(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def future_public_method(
        session: sgalchemy.Session,
        marker: object,
    ) -> tuple[sgalchemy.Session, object]:
        return session, marker

    monkeypatch.setattr(
        sgalchemy.Session,
        'future_public_method',
        future_public_method,
        raising=False,
    )
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    scoped = sgalchemy.scoped_session(sgalchemy.sessionmaker(engine))
    marker = object()

    assert 'future_public_method' not in _scoping.ScopedSession.__dict__
    delegated: Any = scoped
    assert delegated.future_public_method(marker) == (scoped(), marker)

    scoped.remove()


def test_scoped_session_explicit_method_signatures_match_session() -> None:
    for name in _SCOPED_SESSION_METHODS:
        scoped_method = _scoping.ScopedSession.__dict__[name]
        session_method = sgalchemy.Session.__dict__[name]
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


def test_scoped_session_is_thread_local(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    scoped = sgalchemy.scoped_session(sgalchemy.sessionmaker(engine))
    main_session = scoped()

    def use_worker_session() -> tuple[sgalchemy.Session, sgalchemy.Session]:
        sessions = (scoped(), scoped())
        scoped.remove()
        return sessions

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_first, worker_second = pool.submit(use_worker_session).result(timeout=2)

    assert worker_first is worker_second
    assert worker_first is not main_session
    assert scoped() is main_session
    scoped.remove()


def test_scoped_session_supports_abc_and_singledispatch_without_allocation() -> None:
    def factory() -> sgalchemy.Session:
        raise AssertionError('factory must not run')

    class Marker(abc.ABC):
        @abc.abstractmethod
        def mark(self) -> None:
            pass

    @functools.singledispatch
    def describe(_value: object) -> str:
        return 'object'

    @describe.register(sgalchemy.ScopedSession)
    def describe_scoped(_value: sgalchemy.ScopedSession) -> str:
        return 'scoped'

    scoped = sgalchemy.scoped_session(factory)

    assert not isinstance(scoped, Marker)
    assert describe(scoped) == 'scoped'


def test_scoped_session_underscore_probes_do_not_allocate() -> None:
    factory_calls = 0

    def factory() -> sgalchemy.Session:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError('factory must not run')

    scoped = sgalchemy.scoped_session(factory)

    for name in ('_state', '_missing', '__missing__'):
        with pytest.raises(AttributeError) as excinfo:
            getattr(scoped, name)
        assert type(excinfo.value) is AttributeError
        assert not hasattr(scoped, name)
    assert scoped.__class__ is _scoping.ScopedSession
    assert scoped._factory is factory
    assert set(vars(scoped)) == {'_factory', '_local'}
    assert factory_calls == 0


def test_sessionmaker_and_scoped_passthrough_remove_reset_local_state(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    engine.load_schema()
    factory = sgalchemy.sessionmaker(engine)
    assert factory() is not factory()
    scoped = sgalchemy.scoped_session(factory)
    current = scoped()
    pending = engine.entities.Shot.new(code='pending')

    scoped.add(pending)

    assert scoped.engine is engine
    assert entity_state(pending).session is current
    scoped.remove()
    assert entity_state(pending).session is None
    assert scoped() is not current
    scoped.remove()


def test_sessionmaker_forwards_only_supported_defaults(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    factory = sgalchemy.sessionmaker(engine, max_implicit_queries=3)

    first = factory()
    second = factory()

    assert first is not second
    assert first.engine is engine
    assert first._max_implicit_queries == 3
    with pytest.raises(TypeError):
        cast('Any', sgalchemy.sessionmaker)(engine, unsupported=True)


def test_sessionmaker_has_no_begin_and_probe_does_not_allocate_or_perform_io(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    engine = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    factory = sgalchemy.sessionmaker(engine)
    calls_before = list(fake_sg._calls)

    assert not hasattr(factory, 'begin')
    assert fake_sg._calls == calls_before


@pytest.mark.parametrize('private', [False, True])
def test_setter_rejects_every_inflight_connection_transition(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    private: bool,
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    original = fake_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=10)
        return original()

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    target = Registry() if private else _registry.default_registry()
    kwargs: dict[str, Any] = {'registry': target} if private else {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        connecting = pool.submit(
            lambda: sgalchemy.connect(
                connection=fake_sg,
                cache_root=tmp_path,
                **kwargs,
            )
        )
        try:
            assert entered.wait(timeout=2)
            with pytest.raises(SgalchemyError, match='connection is establishing'):
                _registry.set_default_registry(target)
        finally:
            release.set()
        installed = connecting.result(timeout=2)

    assert installed.engine.registry is target
    assert _scoping._default_session().engine is installed.engine
    if private:
        assert _registry.default_registry() is not target


def test_standalone_default_binding_accepts_equivalent_connection_site_spelling(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    standalone = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'standalone',
    )
    standalone.load_schema()
    registry = standalone.registry
    equivalent = type(fake_sg)()
    equivalent.base_url = 'HTTPS://FAKE.EXAMPLE.COM/'

    same_site = sgalchemy.connect(
        connection=equivalent,
        cache_root=tmp_path / 'same',
    )

    assert same_site.engine.registry is registry


def test_standalone_default_binding_rejects_different_site_non_force(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    standalone = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'standalone',
    )
    standalone.load_schema()
    other = type(fake_sg)()
    other.base_url = 'https://other.example.com'

    with pytest.raises(SgalchemyError, match=r'engine\.entities|registry=Registry'):
        sgalchemy.connect(connection=other, cache_root=tmp_path / 'other')
    assert other._calls == []


def test_standalone_default_binding_routes_connect_by_site(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    standalone = sgalchemy.create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'standalone',
    )
    standalone.load_schema()
    registry = standalone.registry

    same_site = sgalchemy.connect(
        connection=type(fake_sg)(),
        cache_root=tmp_path / 'same',
    )

    assert same_site.engine.registry is registry


def test_standalone_same_site_bind_during_connect_is_an_ordinary_refresh(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    original = fake_sg.schema_read

    def blocking_schema_read() -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=10)
        return original()

    monkeypatch.setattr(fake_sg, 'schema_read', blocking_schema_read)
    registry = _registry.default_registry()
    standalone_sg = type(fake_sg)()
    standalone = sgalchemy.create_engine(
        connection=standalone_sg,
        cache_root=tmp_path / 'standalone',
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        connecting = pool.submit(
            sgalchemy.connect,
            connection=fake_sg,
            cache_root=tmp_path / 'connect',
        )
        try:
            assert entered.wait(timeout=2)
            standalone.load_schema()
        finally:
            release.set()
        installed = connecting.result(timeout=2)

    assert standalone.registry is registry
    assert installed.engine.registry is registry
    assert _registry.default_registry() is registry
    assert _scoping._default_session().engine is installed.engine


def test_sync_suppress_failure_does_not_suppress_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt), _scoping._SuppressFailure():
        raise KeyboardInterrupt
