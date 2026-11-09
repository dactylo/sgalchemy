import asyncio
import importlib
import sys
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, ParamSpec, Protocol, TypeAlias, TypeVar

import pytest

if TYPE_CHECKING:
    from sgalchemy._core._engine import Engine
    from tests._fakes import FakeAsyncShotgun, FakeShotgun

from sgalchemy._core import _registry
from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._entity import register_entity_class
from sgalchemy._core._registry import Entity, Registry, RegistryBinding, entity_state
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _uow
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._orm._session import Session
from sgalchemy._shotgun import AsyncShotgunLike, ShotgunLike, WireRecord
from sgalchemy.schema._cache import SchemaCache, normalize_server_info

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T', bound=type[object])

AUTHENTICATION_FAULT: type[Exception] = type('AuthenticationFault', (Exception,), {})
PROVIDER_DISTRIBUTIONS = {'api': 'sg-python-api', 'official': 'shotgun-api3'}
SAVED_BATCH_NOTE = (
    'sgalchemy: SG saved batch wave 1 chunk 1 before this failure; it counts as committed '
    'and a later commit will not send it again'
)
SCHEMA_ABSENT_MESSAGE = (
    'FutureVendor schema remains absent after one forced schema refresh; '
    'make the entity type visible on the site, then retry'
)
STAGED_RETRY_NOTE = (
    'sgalchemy: if this session remains open and usable, a later commit may retry only the '
    'staged partition'
)
STATIC_FORMS = (
    {'script_name': 'script', 'api_key': 'secret-key'},
    {'session_token': 'secret-token'},
)

UNFINISHED_BATCH_ERROR = (
    "SG saved a batch from this session's last commit, but sgalchemy could not finish "
    "updating the session's objects; call rollback() or close() on the session, then query"
    ' the affected records again. If you no longer have the session, discard every object '
    'it held, including objects you added to it or staged for deletion or revival.'
)
UNFINISHED_BATCH_NOTE = (
    'sgalchemy: SG saved that batch, but sgalchemy could not finish updating this '
    "session's objects; it tries again when the session is next used, rolled back, or "
    'closed'
)

AsyncCallback: TypeAlias = Callable[_P, Awaitable[_R]]
AsyncExecutionEnvironment: TypeAlias = 'tuple[AsyncEngine, FakeAsyncShotgun, Path]'
SyncCallback: TypeAlias = Callable[_P, _R]
SyncExecutionEnvironment: TypeAlias = 'tuple[Engine, FakeShotgun, Path]'
SyncSessionBundle: TypeAlias = 'tuple[Session, Engine, FakeShotgun]'


class AsyncSessionFactory(Protocol):
    def __call__(self) -> AsyncSession: ...
    def commit_on_success(self) -> AbstractAsyncContextManager[AsyncSession]: ...


class SyncSessionFactory(Protocol):
    def __call__(self) -> Session: ...
    def commit_on_success(self) -> AbstractContextManager[Session]: ...


class BrokenNoteError(RuntimeError):
    def add_note(self, note: str) -> None:
        raise RuntimeError(f'cannot attach {note}')


class InterruptingNoteAccess(BaseException):
    def __getattribute__(self, name: str) -> Any:  # noqa: ANN401 - Delegate BaseException attributes while interrupting note access.
        if name == '__notes__':
            raise KeyboardInterrupt
        return super().__getattribute__(name)


class YieldCountingEntities(list[_registry.Entity]):
    def __init__(self, values: list[_registry.Entity]) -> None:
        super().__init__(values)
        self.visits = 0

    def __iter__(self) -> Iterator[_registry.Entity]:
        for obj in super().__iter__():
            self.visits += 1
            yield obj


class _BrokenProviderFinder:
    def __init__(self, missing: str) -> None:
        self._missing = missing

    def find_spec(self, fullname: str, _path: object, _target: object = None) -> None:
        if fullname == 'shotgun_api3':
            raise ModuleNotFoundError(f'No module named {self._missing!r}', name=self._missing)


def assert_cleanup_log(caplog: pytest.LogCaptureFixture, entry: str) -> None:
    operation = 'rollback()' if entry == 'rollback' else 'close()'
    records = [record for record in caplog.records if record.levelname == 'ERROR']
    assert all(record.exc_info is not None for record in records)
    messages = [record.getMessage() for record in records]
    assert len(messages) == 2
    assert messages[1] == (
        f'{operation} could not finish updating objects after SG saved their batch; '
        'marking them outdated'
    )
    for message in messages:
        assert not any(
            term in message
            for term in (
                'acceptance record',
                'completion',
                'protective release',
                'unreconciled',
                'indeterminate',
            )
        )


def assert_disjoint_commit_outcome(error: CommitError) -> None:
    items = [*error.committed, *error.staged, *error.indeterminate, *error.withdrawn]
    assert len(items) == len({id(item) for item in items})


def assert_outdated_error(error: BaseException, label: str) -> None:
    assert str(error) == (
        f'{label} is outdated: SG saved your change, but sgalchemy could not '
        'update this copy; discard it and query the record again'
    )


def block_call(
    call: SyncCallback[_P, _R],
    entered: Event,
    release: Event,
) -> SyncCallback[_P, _R]:
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        entered.set()
        assert release.wait(timeout=10)
        return call(*args, **kwargs)

    return wrapper


def break_provider_import(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.delitem(sys.modules, 'shotgun_api3', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenProviderFinder(missing), *sys.meta_path])


def capture_hydration_bindings(
    session: Session | AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> list[RegistryBinding | None]:
    state = object.__getattribute__(
        session, 'state' if isinstance(session, AsyncSession) else '_state'
    )
    original_hydrate_record = state.hydrate_record
    bindings: list[RegistryBinding | None] = []

    def capture(
        entity_type_name: str,
        record: WireRecord,
        binding: RegistryBinding | None,
        *,
        retired: bool | None = None,
    ) -> _registry.Entity:
        bindings.append(binding)
        return original_hydrate_record(entity_type_name, record, binding, retired=retired)

    monkeypatch.setattr(state, 'hydrate_record', capture)
    return bindings


@contextmanager
def entity_namespace_scope(names: tuple[str, ...]) -> Generator[ModuleType, None, None]:
    entities = importlib.import_module('sgalchemy.entities')
    for name in names:
        vars(entities).pop(name, None)
    try:
        yield entities
    finally:
        for name in names:
            vars(entities).pop(name, None)


class AcceptedBatchCorpus(NamedTuple):
    updated: Entity
    cleared: Entity
    restripped: Entity
    deleted: Entity
    unrelated: Entity
    first: Entity
    second: Entity
    uploaded: Entity
    dependent: Entity
    thumbnail: Path
    filmstrip: Path


class _UnreadableTypeMeta(type):
    @property
    def __mro__(cls) -> tuple[type, ...]:
        raise ValueError('type attributes unavailable')


class UnclassifiableError(RuntimeError, metaclass=_UnreadableTypeMeta):
    pass


class FailOnceDict(dict[str, Any]):
    def __init__(self, values: dict[str, Any], key: str, fault: BaseException) -> None:
        super().__init__(values)
        self._key = key
        self._fault: BaseException | None = fault

    def __setitem__(self, key: str, value: object) -> None:
        if key == self._key and self._fault is not None:
            fault, self._fault = self._fault, None
            raise fault
        super().__setitem__(key, value)


class FailOnceSet(set[int]):
    def __init__(self, values: set[int], fault: BaseException) -> None:
        super().__init__(values)
        self._fault: BaseException | None = fault

    def difference_update(self, *others: Iterable[object]) -> None:
        if self._fault is not None:
            fault, self._fault = self._fault, None
            raise fault
        super().difference_update(*others)


def assert_unsent_work_pending(corpus: AcceptedBatchCorpus, unit_of_work: _uow.UnitOfWork) -> None:
    assert entity_state(corpus.unrelated).loaded['code'] == 'unrelated'
    assert entity_state(corpus.unrelated).dirty == {}
    assert unit_of_work.new == [corpus.dependent]
    assert unit_of_work._new_ids == {id(corpus.dependent)}
    assert entity_state(corpus.uploaded).dirty == {'image': corpus.thumbnail}
    state = entity_state(corpus.restripped)
    assert state.dirty['filmstrip_image'] == corpus.filmstrip
    assert state.clean['filmstrip_image'] is None
    assert 'filmstrip_image' not in state.clean_refresh_required | state.clean_unverified_writes


def fetch_schema(connection: ShotgunLike) -> SchemaCache:
    info = getattr(connection, 'info', None)
    raw_server_info = info() if callable(info) else None
    return SchemaCache(
        fields=connection.schema_read(),
        entities=connection.schema_entity_read(),
        fetched_at=datetime.now(UTC),
        server_info=normalize_server_info(raw_server_info),
    )


def inject_faults(
    original: Callable[_P, _R],
    faults: Iterable[BaseException | None],
    *,
    after: bool = False,
) -> Callable[_P, _R]:
    planned = iter(faults)

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        fault = next(planned, None)
        if fault is not None and not after:
            raise fault
        result = original(*args, **kwargs)
        if fault is not None:
            raise fault
        return result

    return wrapper


def assert_accepted_batch_settled(
    corpus: AcceptedBatchCorpus,
    unit_of_work: _uow.UnitOfWork,
    identity_map: dict[tuple[str, int], Entity],
    *,
    completion_ran: bool = True,
) -> None:
    submissions = [
        (corpus.updated, {'code', 'description'}),
        (corpus.cleared, {'image'}),
        (corpus.restripped, {'image'}),
        (corpus.deleted, {'code'}),
        (corpus.first, {'code'}),
        (corpus.second, {'code'}),
        (corpus.uploaded, {'code'}),
    ]
    for obj, fields in submissions:
        state = entity_state(obj)
        for provenance in (
            state.dirty,
            state.clean,
            state.dirty_types,
            state.clean_refresh_required,
            state.clean_unverified_writes,
        ):
            assert not fields.intersection(provenance)
        assert not fields.intersection(state.unverified_writes)
        expected: set[str] = (
            fields - {'image'}
            if completion_ran
            else ({'description'} if obj is corpus.updated else set())
        )
        assert fields.intersection(state.refresh_required) == expected
        assert state.id is not None
        assert identity_map[state.entity_type, state.id] is obj
        assert not state.indeterminate
    for obj, entity_id in zip(
        (corpus.first, corpus.second, corpus.uploaded), (9001, 9002, 9003), strict=True
    ):
        state = entity_state(obj)
        assert state.id == state.loaded['id'] == entity_id
        assert all(staged is not obj for staged in unit_of_work.new)
        assert id(obj) not in unit_of_work._new_ids
    for obj, fields in (
        (corpus.cleared, {'image', 'filmstrip_image'}),
        (corpus.restripped, {'image'}),
    ):
        state = entity_state(obj)
        for code in fields:
            assert state.loaded[code] is None
            assert code not in state.refresh_required | state.unverified_writes
    if not completion_ran:
        for obj, value in (
            (corpus.updated, 'new'),
            (corpus.deleted, 'gone'),
            (corpus.first, 'first'),
            (corpus.second, 'second'),
            (corpus.uploaded, 'uploaded'),
        ):
            assert entity_state(obj).loaded['code'] == value
    assert entity_state(corpus.deleted).deleted
    assert unit_of_work.deleted == []
    assert unit_of_work._deleted_ids == set()
    assert _uow.staged_for_identity(unit_of_work, ('Shot', 54)) == ()
    assert not _uow.accepted_batch_pending(unit_of_work)


def interrupt_accepted_release(
    monkeypatch: pytest.MonkeyPatch,
    session: Session | AsyncSession,
    position: str,
) -> None:
    state = session.state if isinstance(session, AsyncSession) else session._state
    unit_of_work = (
        session.raw_unit_of_work if isinstance(session, AsyncSession) else session._unit_of_work
    )
    owner, name = (
        (state, 'detach_canceled')
        if position == 'detached'
        else (_uow, '_retire_batch_operation_provenance')
        if position == 'provenance'
        else (_uow, '_mark_outdated')
    )
    faults = [KeyboardInterrupt()] if position == 'disposition' else [None, KeyboardInterrupt()]
    original = getattr(owner, name)
    interrupted = inject_faults(original, faults)

    def during_release(*args: object, **kwargs: object) -> object:
        batch = unit_of_work.accepted
        if batch is not None and batch.release_started:
            return interrupted(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, during_release)


def isolated_async_engine(connection: AsyncShotgunLike, cache_root: Path) -> AsyncEngine:
    return create_async_engine(
        connection=connection,
        cache_root=cache_root,
        registry=Registry(),
        bind_default=False,
    )


def mapped_entity(type_name: str) -> Callable[[_T], _T]:
    def decorate(cls: _T) -> _T:
        register_entity_class(cls, type_name, Registry())
        return cls

    return decorate


def plan_item_owners(items: list[object]) -> list[Entity]:
    owners: list[Entity] = []
    for item in items:
        assert isinstance(item, (Entity, _uow._Op, _uow._UploadPlanItem))
        owners.append(_uow._plan_item_owner(item))
    return owners


def preserve_signature(
    _call: Callable[_P, _R],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:

    def decorate(wrapper: Callable[_P, _R]) -> Callable[_P, _R]:
        return wrapper

    return decorate


def publish_on_drift(
    call: SyncCallback[_P, _R],
    registry: Registry,
    complete: RegistryBinding,
) -> tuple[SyncCallback[_P, _R], list[int]]:
    publications = [0]

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return call(*args, **kwargs)
        except SchemaDrift:
            publications[0] += 1
            registry.bind(dict(complete.schemas), key=complete.key)
            raise

    return wrapper, publications


def reset_default_registry_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_registry, '_DEFAULT_REGISTRIES', [])
    monkeypatch.setattr(_registry, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(_registry, '_DEFAULT_TRANSITION_OWNER', [])


def run_async(awaitable: Awaitable[_R], *, timeout: float = 10) -> _R:
    # Bound the whole rendezvous scenario. asyncio.run cancels and drains any
    # unfinished child tasks when the scenario fails or reaches its deadline.
    async def bounded() -> _R:
        async with asyncio.timeout(timeout):
            return await awaitable

    return asyncio.run(bounded())


def stage_accepted_batch_corpus(
    session: Session | AsyncSession,
    registry: Registry,
    tmp_path: Path,
) -> AcceptedBatchCorpus:
    sequence = registry.entity_class('Sequence')
    shot = registry.entity_class('Shot')
    thumbnail = tmp_path / 'accepted-thumbnail.jpg'
    filmstrip = tmp_path / 'accepted-filmstrip.jpg'
    for path in (thumbnail, filmstrip):
        path.write_bytes(b'image')
    updated = session.merge_dict({'type': 'Shot', 'id': 51, 'code': 'old', 'description': 'old'})
    cleared = session.merge_dict(
        {'type': 'Shot', 'id': 52, 'image': 'old-image', 'filmstrip_image': 'old-strip'}
    )
    restripped = session.merge_dict(
        {'type': 'Shot', 'id': 53, 'image': 'old-image', 'filmstrip_image': 'old-strip'}
    )
    deleted = session.merge_dict({'type': 'Shot', 'id': 54, 'code': 'doomed'})
    unrelated = session.merge_dict({'type': 'Shot', 'id': 60, 'code': 'unrelated'})
    updated.code = 'new'
    updated.description = 'new'
    cleared.image = None
    restripped.image = None
    restripped.filmstrip_image = filmstrip
    deleted.code = 'gone'
    session.delete(deleted)
    first = sequence.new(code='first')
    second = sequence.new(code='second')
    uploaded = shot.new(code='uploaded', image=thumbnail)
    dependent = shot.new(code='dependent', sg_sequence=first)
    for obj in (first, second, uploaded, dependent):
        session.add(obj)
    return AcceptedBatchCorpus(
        updated,
        cleared,
        restripped,
        deleted,
        unrelated,
        first,
        second,
        uploaded,
        dependent,
        thumbnail,
        filmstrip,
    )


def wave_one_owners(corpus: AcceptedBatchCorpus) -> list[Entity]:
    return [
        corpus.updated,
        corpus.cleared,
        corpus.restripped,
        corpus.deleted,
        corpus.deleted,
        corpus.first,
        corpus.second,
        corpus.uploaded,
    ]


def assert_accepted_batch_error(
    error: BaseException, fault: BaseException, corpus: AcceptedBatchCorpus
) -> None:
    if isinstance(fault, Exception):
        assert isinstance(error, CommitError)
        assert error.__cause__ is fault
        assert plan_item_owners(error.committed) == wave_one_owners(corpus)
        assert plan_item_owners(error.staged) == [
            corpus.dependent,
            corpus.restripped,
            corpus.uploaded,
        ]
        assert error.indeterminate == error.withdrawn == []
        assert_disjoint_commit_outcome(error)
    else:
        assert error is fault
    assert SAVED_BATCH_NOTE in error.__notes__
    assert not any('failed during' in note for note in error.__notes__)
    assert UNFINISHED_BATCH_NOTE not in error.__notes__


def assert_outdated_corpus(corpus: AcceptedBatchCorpus, session: Session | AsyncSession) -> None:
    state = session.state if isinstance(session, AsyncSession) else session._state
    unit_of_work = (
        session.raw_unit_of_work if isinstance(session, AsyncSession) else session._unit_of_work
    )
    expected = [
        ('Shot', 51),
        ('Shot', 52),
        ('Shot', 53),
        ('Shot', 54),
        ('Shot', 54),
        ('Sequence', 9001),
        ('Sequence', 9002),
        ('Shot', 9003),
    ]
    for obj, (entity_type, entity_id) in zip(wave_one_owners(corpus), expected, strict=True):
        assert repr(obj) == f'{entity_type}(id={entity_id}, outdated)'
        assert not any(obj is mapped for mapped in state.identity_map.values())
    for action, label in (
        (lambda: corpus.updated.code, 'Shot(id=51)'),
        (lambda: _registry.inspect(corpus.first), 'Sequence(id=9001)'),
        (lambda: session.delete(corpus.updated), 'Shot(id=51)'),
        (lambda: session.revive(corpus.deleted), 'Shot(id=54)'),
    ):
        with pytest.raises(SgalchemyError) as excinfo:
            action()
        assert_outdated_error(excinfo.value, label)
    assert not _uow.accepted_batch_pending(unit_of_work)
    assert not state.owns_entity(corpus.dependent)
    assert unit_of_work.new == []


def assert_release_pending(
    session: Session | AsyncSession,
    corpus: AcceptedBatchCorpus,
    position: str,
) -> None:
    unit_of_work = (
        session.raw_unit_of_work if isinstance(session, AsyncSession) else session._unit_of_work
    )
    assert _uow.accepted_batch_pending(unit_of_work)
    for action in (
        lambda: session.merge_dict({'type': 'Shot', 'id': 99}),
        lambda: corpus.unrelated.code,
    ):
        with pytest.raises(SgalchemyError) as excinfo:
            action()
        assert str(excinfo.value) == UNFINISHED_BATCH_ERROR
    marked = (
        wave_one_owners(corpus)
        if position in {'provenance', 'detached'}
        else [corpus.updated]
        if position == 'some-marked'
        else []
    )
    for obj in marked:
        state = entity_state(obj)
        assert state.outdated is not None
        with pytest.raises(SgalchemyError) as excinfo:
            obj.as_dict()
        entity_type, entity_id = state.outdated
        assert_outdated_error(excinfo.value, f'{entity_type}(id={entity_id})')


def assert_cleanup_interrupted(
    error: BaseException,
    entry: str,
    session: Session | AsyncSession,
    corpus: AcceptedBatchCorpus,
    position: str,
) -> None:
    operation = 'rollback()' if entry == 'rollback' else 'close()'
    assert getattr(error, '__notes__', []) == [
        (
            f'sgalchemy: {operation} did not finish; call rollback() or close() on this session '
            'again before using it'
        )
    ]
    assert_release_pending(session, corpus, position)
