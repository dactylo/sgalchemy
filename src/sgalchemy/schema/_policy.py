"""Define schema cache freshness policies."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from sgalchemy.schema._cache import SchemaCache


@runtime_checkable
class SchemaCachePolicy(Protocol):
    """Decides whether a cached schema may be used without refetching."""

    def is_fresh(self, cache: SchemaCache) -> bool:
        """Return True if the cached schema may be used without refetching."""
        ...


@dataclass(frozen=True)
class TTLPolicy:
    """Treat a cache as fresh for a fixed number of hours after fetch."""

    hours: float = 24.0

    def is_fresh(self, cache: SchemaCache) -> bool:
        """Return True while the cache is younger than the TTL.

        Args:
            cache: The cached schema snapshot to evaluate.

        Returns:
            True if the cache age is below the configured TTL.
        """
        return datetime.now(UTC) - cache.fetched_at < timedelta(hours=self.hours)
