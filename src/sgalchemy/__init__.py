"""Build reflected queries and manage explicit entity reads and writes.

Create an engine, call its `load_schema()` method, and obtain reflected classes
from `engine.entities`. Async engines require awaited schema preparation.
Queries never autoflush staged changes; call `Session.commit()` or await
`AsyncSession.commit()` explicitly.

Factories use the process default Registry unless an explicit Registry or
`bind_default=False` selects isolation. Importing this package installs the
synchronous `Entity.find()` callback; `connect()` configures its default
session. Importing the package performs no provider or schema-cache I/O.
"""

from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._expressions import FieldExpr, and_, func, or_, raw
from sgalchemy._core._options import defer, joinedload, lazyload, load_only, selectinload
from sgalchemy._core._registry import (
    Entity,
    InspectionInfo,
    Registry,
    default_registry,
    inspect,
    set_default_registry,
)
from sgalchemy._core._result import Result, Row
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import (
    CommitError,
    CompilationError,
    RefreshRequiredError,
    RelationshipTargetUnavailableError,
    RetiredEntityWriteError,
    SchemaUnavailableError,
    SgalchemyError,
    StaleObjectError,
    UnfetchedFieldError,
)
from sgalchemy._orm import (
    AsyncScopedSession,
    AsyncSession,
    AsyncSessionFactory,
    ScopedSession,
    Session,
    SessionFactory,
    async_scoped_session,
    async_sessionmaker,
    connect,
    scoped_session,
    sessionmaker,
)
from sgalchemy._shotgun import AsyncShotgunLike, ShotgunLike

__all__ = [
    'AsyncEngine',
    'AsyncScopedSession',
    'AsyncSession',
    'AsyncSessionFactory',
    'AsyncShotgunLike',
    'CommitError',
    'CompilationError',
    'Engine',
    'Entity',
    'FieldExpr',
    'InspectionInfo',
    'RefreshRequiredError',
    'Registry',
    'RelationshipTargetUnavailableError',
    'Result',
    'RetiredEntityWriteError',
    'Row',
    'SchemaUnavailableError',
    'ScopedSession',
    'Select',
    'Session',
    'SessionFactory',
    'SgalchemyError',
    'ShotgunLike',
    'StaleObjectError',
    'UnfetchedFieldError',
    'and_',
    'async_scoped_session',
    'async_sessionmaker',
    'connect',
    'create_async_engine',
    'create_engine',
    'default_registry',
    'defer',
    'func',
    'inspect',
    'joinedload',
    'lazyload',
    'load_only',
    'or_',
    'raw',
    'scoped_session',
    'select',
    'selectinload',
    'sessionmaker',
    'set_default_registry',
]
