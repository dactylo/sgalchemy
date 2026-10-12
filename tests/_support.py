import importlib
from collections.abc import Awaitable, Callable, Generator, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeAlias, TypeVar

import pytest

if TYPE_CHECKING:
    from _fakes import _FakeAsyncShotgun, _FakeShotgun

    from sgalchemy._core._engine import Engine

from sgalchemy._core import _registry
from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._entity import register_entity_class
from sgalchemy._core._registry import Registry, RegistryBinding
from sgalchemy._errors import CommitError
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._orm._session import Session
from sgalchemy._shotgun import AsyncShotgunLike, ShotgunLike, WireRecord
from sgalchemy.schema._cache import SchemaCache, normalize_server_info

_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T', bound=type[object])

AUTHENTICATION_FAULT: type[Exception] = type('AuthenticationFault', (Exception,), {})
PROVIDER_DISTRIBUTIONS = {'api': 'sg-python-api', 'official': 'shotgun-api3'}
SCHEMA_ABSENT_MESSAGE = (
    'FutureVendor schema remains absent after one forced schema refresh; '
    'make the entity type visible on the site, then retry'
)
STATIC_FORMS = (
    {'script_name': 'script', 'api_key': 'secret-key'},
    {'session_token': 'secret-token'},
)

_AsyncCallback: TypeAlias = Callable[_P, Awaitable[_R]]  # noqa: PYI047
_AsyncExecutionEnvironment: TypeAlias = (  # noqa: PYI047
    'tuple[AsyncEngine, _FakeAsyncShotgun, Path]'
)
_SyncCallback: TypeAlias = Callable[_P, _R]  # noqa: PYI047
_SyncExecutionEnvironment: TypeAlias = (  # noqa: PYI047
    'tuple[Engine, _FakeShotgun, Path]'
)
_SyncSessionBundle: TypeAlias = (  # noqa: PYI047
    'tuple[Session, Engine, _FakeShotgun]'
)


class _AsyncSessionFactory(Protocol):  # noqa: PYI046  # pyright: ignore[reportUnusedClass]
    def __call__(self) -> AsyncSession: ...
    def commit_on_success(self) -> AbstractAsyncContextManager[AsyncSession]: ...


class _SyncSessionFactory(Protocol):  # noqa: PYI046  # pyright: ignore[reportUnusedClass]
    def __call__(self) -> Session: ...
    def commit_on_success(self) -> AbstractContextManager[Session]: ...


class BrokenNoteError(RuntimeError):
    def add_note(self, note: str) -> None:
        raise RuntimeError(f'cannot attach {note}')


class YieldCountingEntities(list[_registry.Entity]):
    def __init__(self, values: list[_registry.Entity]) -> None:
        super().__init__(values)
        self._visits = 0

    def __iter__(self) -> Iterator[_registry.Entity]:
        for obj in super().__iter__():
            self._visits += 1
            yield obj


def _preserve_signature(  # pyright: ignore[reportUnusedFunction]
    _call: Callable[_P, _R],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:

    def decorate(wrapper: Callable[_P, _R]) -> Callable[_P, _R]:
        return wrapper

    return decorate


def assert_disjoint_commit_outcome(error: CommitError) -> None:
    items = [*error.committed, *error.staged, *error.indeterminate, *error.withdrawn]
    assert len(items) == len({id(item) for item in items})


def capture_hydration_bindings(
    session: Session | AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> list[RegistryBinding | None]:
    state = object.__getattribute__(session, '_state')
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


def fetch_schema(connection: ShotgunLike) -> SchemaCache:
    info = getattr(connection, 'info', None)
    raw_server_info = info() if callable(info) else None
    return SchemaCache(
        fields=connection.schema_read(),
        entities=connection.schema_entity_read(),
        fetched_at=datetime.now(UTC),
        server_info=normalize_server_info(raw_server_info),
    )


def isolated_async_engine(connection: AsyncShotgunLike, cache_root: Path) -> AsyncEngine:
    return create_async_engine(
        connection=connection,
        cache_root=cache_root,
        registry=Registry(),
        bind_default=False,
    )


def mapped_entity(type_name: str) -> Callable[[_T], _T]:
    def decorate(cls: _T) -> _T:
        register_entity_class(cls, type_name, None)
        return cls

    return decorate


def reset_default_registry_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_registry, '_DEFAULT_REGISTRIES', [])
    monkeypatch.setattr(_registry, '_DEFAULT_SESSION_REGISTRIES', [], raising=False)
    monkeypatch.setattr(_registry, '_DEFAULT_TRANSITION', {'reserved': False}, raising=False)
