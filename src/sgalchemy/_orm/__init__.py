"""Hold package-internal session modules and install the `Entity.find()` finder on import."""

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
    install_entity_finder_callback,
    scoped_session,
    sessionmaker,
)
from sgalchemy._orm._session import Session

__all__ = [
    'AsyncScopedSession',
    'AsyncSession',
    'AsyncSessionFactory',
    'ScopedSession',
    'Session',
    'SessionFactory',
    'async_scoped_session',
    'async_sessionmaker',
    'connect',
    'scoped_session',
    'sessionmaker',
]

install_entity_finder_callback()
