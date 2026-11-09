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
from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._engine_common import EntityNamespace
from sgalchemy._core._expressions import (
    FieldExpr,
    Filter,
    FilterGroup,
    Grouping,
    Ordering,
    RawFilters,
    Summary,
    and_,
    func,
    or_,
    raw,
)
from sgalchemy._core._options import (
    Defer,
    JoinedLoad,
    LazyLoad,
    LoadOnly,
    SelectinLoad,
    defer,
    joinedload,
    lazyload,
    load_only,
    selectinload,
)
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
from sgalchemy._orm._async_scoping import (
    AsyncScopedSession,
    AsyncSessionFactory,
    async_scoped_session,
    async_sessionmaker,
)
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._orm._scoping import (
    ScopedSession,
    SessionFactory,
    connect,
    scoped_session,
    sessionmaker,
)
from sgalchemy._orm._session import Session
from sgalchemy._shotgun import (
    AsyncClosableShotgunLike,
    AsyncShotgunLike,
    ClosableShotgunLike,
    ShotgunLike,
)

__all__ = [
    'AsyncClosableShotgunLike',
    'AsyncEngine',
    'AsyncScopedSession',
    'AsyncSession',
    'AsyncSessionFactory',
    'AsyncShotgunLike',
    'ClosableShotgunLike',
    'CommitError',
    'CompilationError',
    'Defer',
    'Engine',
    'Entity',
    'EntityNamespace',
    'FieldExpr',
    'Filter',
    'FilterGroup',
    'FindCall',
    'Grouping',
    'InspectionInfo',
    'JoinedLoad',
    'LazyLoad',
    'LoadOnly',
    'Ordering',
    'RawFilters',
    'RefreshRequiredError',
    'Registry',
    'RelationshipTargetUnavailableError',
    'Result',
    'RetiredEntityWriteError',
    'Row',
    'SchemaUnavailableError',
    'ScopedSession',
    'Select',
    'SelectinLoad',
    'Session',
    'SessionFactory',
    'SgalchemyError',
    'ShotgunLike',
    'StaleObjectError',
    'SummarizeCall',
    'Summary',
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
