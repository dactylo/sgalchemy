from datetime import UTC, datetime, timedelta

from sgalchemy.schema import SchemaCachePolicy, TTLPolicy
from sgalchemy.schema._cache import SchemaCache


def _cache_aged(hours: float) -> SchemaCache:
    return SchemaCache(
        fields={}, entities={}, fetched_at=datetime.now(UTC) - timedelta(hours=hours)
    )


def test_ttl_policy() -> None:
    policy = TTLPolicy(hours=24)
    assert policy.is_fresh(_cache_aged(1))
    assert not policy.is_fresh(_cache_aged(25))
    assert isinstance(policy, SchemaCachePolicy)
