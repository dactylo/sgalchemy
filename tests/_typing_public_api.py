from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, assert_type

from _support import mapped_entity

from sgalchemy import (
    AsyncEngine,
    AsyncScopedSession,
    AsyncSession,
    AsyncSessionFactory,
    AsyncShotgunLike,
    CommitError,
    Engine,
    Entity,
    Registry,
    Result,
    ScopedSession,
    Select,
    Session,
    SessionFactory,
    ShotgunLike,
    async_scoped_session,
    async_sessionmaker,
    connect,
    scoped_session,
    select,
    sessionmaker,
)
from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._expressions import FieldExpr, Summary, func
from sgalchemy._orm._state import validate_point_records
from sgalchemy._shotgun import ClosableShotgunLike
from sgalchemy.schema import EntitySchema, FieldSchema, TTLPolicy
from sgalchemy.testing import engine_over


@mapped_entity('Shot')
class _Shot(Entity):
    pass


async def _check_async_entity_types(  # pyright: ignore[reportUnusedFunction]
    session: AsyncSession,
) -> None:
    entity_statement = select(_Shot)
    projection_statement = select(FieldExpr('Shot', ('code',), None))

    assert_type(await session.get(_Shot, 1), _Shot | None)
    assert_type(await session.scalars(entity_statement), list[_Shot])
    assert_type(await session.execute(projection_statement), Result | dict[str, Any])


async def _check_async_scoped_types(  # pyright: ignore[reportUnusedFunction]
    session: AsyncSession,
) -> None:
    scoped = async_scoped_session(lambda: session)
    assert_type(scoped.engine, AsyncEngine)
    assert_type(await scoped.get(_Shot, 1), _Shot | None)
    assert_type(await scoped.scalars(select(_Shot)), list[_Shot])


def _check_commit_error_evidence(  # pyright: ignore[reportUnusedFunction]
    error: CommitError,
) -> None:
    assert_type(error.committed, list[object])
    assert_type(error.staged, list[object])
    assert_type(error.indeterminate, list[object])
    assert_type(error.withdrawn, list[object])


def _check_compile_capabilities(  # pyright: ignore[reportUnusedFunction]
    statement: Select[Entity],
    schemas: Mapping[str, EntitySchema],
) -> None:
    compiled: FindCall | SummarizeCall = statement.compile(schemas)
    assert_type(compiled, FindCall | SummarizeCall)


def _check_connect_options(  # pyright: ignore[reportUnusedFunction]
    connection: ShotgunLike,
    creator: Callable[[], ClosableShotgunLike],
    cache_root: Path,
    policy: TTLPolicy,
    registry: Registry,
) -> None:
    connect(
        'https://example.com',
        script_name='script',
        api_key='key',
        cache_root=cache_root,
        policy=policy,
        echo=True,
        registry=registry,
        bind_default=False,
    )
    connect('https://example.com', creator=creator)
    connect(
        connection=connection,
        force=True,
        cache_root=cache_root,
        serialize=False,
    )


def _check_connect_session_token(  # pyright: ignore[reportUnusedFunction]
    session_token: str,
) -> None:
    connect('https://example.com', session_token=session_token)


def _check_engine_over_options(  # pyright: ignore[reportUnusedFunction]
    connection: ShotgunLike,
    cache_root: Path,
    policy: TTLPolicy,
    registry: Registry,
) -> None:
    engine_over(
        connection,
        cache_root=cache_root,
        policy=policy,
        echo=True,
        registry=registry,
        bind_default=False,
        serialize=True,
    )


def _check_func_types(  # pyright: ignore[reportUnusedFunction]
    field: FieldExpr,
) -> None:
    assert_type(func.avg(field), Summary)
    assert_type(func.record_count(field), Summary)


def _check_public_annotation_types(  # pyright: ignore[reportUnusedFunction]
    engine: Engine,
    async_engine: AsyncEngine,
    async_connection: AsyncShotgunLike,
) -> None:
    factory = sessionmaker(engine)
    assert_type(factory, SessionFactory)
    assert_type(scoped_session(factory), ScopedSession)
    async_factory = async_sessionmaker(async_engine)
    assert_type(async_factory, AsyncSessionFactory)
    assert_type(async_scoped_session(async_factory), AsyncScopedSession)
    assert_type(engine.schema, Mapping[str, EntitySchema])
    assert_type(engine.schema['Shot'].fields, Mapping[str, FieldSchema])
    assert_type(async_connection, AsyncShotgunLike)


def _check_sync_entity_types(  # pyright: ignore[reportUnusedFunction]
    session: Session,
) -> None:
    entity_statement = select(_Shot).where(_Shot.code == 'A')
    projection_statement = select(FieldExpr('Shot', ('code',), None))
    records = ({'type': 'Shot', 'id': 1},)

    assert_type(_Shot.find(), list[_Shot])
    assert_type(entity_statement, Select[_Shot])
    assert_type(projection_statement, Select[None])
    assert_type(session.get(_Shot, 1), _Shot | None)
    assert_type(session.scalars(entity_statement), list[_Shot])
    assert_type(session.execute(projection_statement), Result | dict[str, Any])
    assert_type(
        validate_point_records(records, 'Shot', 1, allow_absent=False),
        dict[str, Any],
    )
    assert_type(
        validate_point_records(records, 'Shot', 1, allow_absent=True),
        dict[str, Any] | None,
    )


def _check_sync_scoped_types(  # pyright: ignore[reportUnusedFunction]
    session: Session,
) -> None:
    scoped = scoped_session(lambda: session)
    assert_type(scoped.engine, Engine)
    assert_type(scoped.get(_Shot, 1), _Shot | None)
    assert_type(scoped.scalars(select(_Shot)), list[_Shot])
