"""Validate and persist schema snapshots."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class SchemaCache:
    """Carry raw site-schema payloads and their fetch metadata.

    The dataclass freezes attribute assignment, not its nested dictionaries.
    Cache loading and TTL evaluation require a timezone-aware fetch timestamp.

    Attributes:
        fields: Raw `schema_read()` payload keyed by canonical entity type.
        entities: Raw `schema_entity_read()` metadata keyed by entity type.
        fetched_at: Time the schema snapshot was fetched.
        server_info: Optional dictionary returned by provider `info()`.
    """

    fields: dict[str, dict[str, Any]]
    entities: dict[str, dict[str, Any]]
    fetched_at: datetime
    server_info: dict[str, Any] | None = None


def _valid_entities(value: object) -> bool:
    """Return whether entity metadata has the shape consumed by `build_schema`."""
    if not isinstance(value, dict):
        return False
    entities = cast('dict[object, object]', value)
    for type_name, raw in entities.items():
        if not isinstance(type_name, str) or not isinstance(raw, dict):
            return False
        metadata = cast('dict[object, object]', raw)
        name = metadata.get('name', {})
        if not isinstance(name, dict) or not isinstance(
            cast('dict[object, object]', name).get('value', type_name), str
        ):
            return False
    return True


def _valid_field(raw: object) -> bool:
    """Return whether field metadata has every shape consumed by `build_schema`."""
    if not isinstance(raw, dict):
        return False
    field = cast('dict[object, object]', raw)
    data_type = field.get('data_type', {})
    name = field.get('name', {})
    properties = field.get('properties', {})
    base_valid = (
        isinstance(data_type, dict)
        and isinstance(cast('dict[object, object]', data_type).get('value', ''), str)
        and isinstance(name, dict)
        and isinstance(cast('dict[object, object]', name).get('value', ''), str)
        and isinstance(properties, dict)
    )
    if not base_valid:
        return False
    if 'editable' in field and not isinstance(field['editable'], dict):
        return False
    property_values = cast('dict[object, object]', properties)
    if 'valid_types' not in property_values:
        return True
    valid_types = property_values['valid_types']
    if not isinstance(valid_types, dict):
        return False
    values = cast('dict[object, object]', valid_types).get('value')
    return values is None or (
        isinstance(values, list)
        and all(isinstance(item, str) for item in cast('list[object]', values))
    )


def _valid_fields(value: object) -> bool:
    """Return whether field-schema mappings have the shape consumed by `build_schema`."""
    if not isinstance(value, dict):
        return False
    fields = cast('dict[object, object]', value)
    return all(
        isinstance(type_name, str)
        and isinstance(raw_fields, dict)
        and all(
            isinstance(code, str) and _valid_field(raw)
            for code, raw in cast('dict[object, object]', raw_fields).items()
        )
        for type_name, raw_fields in fields.items()
    )


def normalize_server_info(value: object) -> dict[str, Any] | None:
    """Return provider server information only when it is a dictionary.

    Args:
        value: Result from an optional provider `info()` method.

    Returns:
        The provider dictionary, or `None` for every other shape.
    """
    return cast('dict[str, Any]', value) if isinstance(value, dict) else None


def store_cache(path: Path, cache: SchemaCache) -> None:
    """Atomically write a schema cache file, creating parent directories.

    Args:
        path: Destination cache file.
        cache: Schema snapshot to serialize.

    Raises:
        OSError: If a cache directory or file operation fails.
        TypeError: If the cache contains a value JSON cannot serialize.
        ValueError: If the cache contains a reference cycle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'fields': cache.fields,
        'entities': cache.entities,
        'fetched_at': cache.fetched_at.isoformat(),
        'server_info': cache.server_info,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def validate_cache(cache: SchemaCache) -> None:
    """Validate schema mappings before building or storing a cache.

    Args:
        cache: Schema snapshot to validate.

    Raises:
        TypeError: If field or entity metadata is structurally malformed.
    """
    if not _valid_fields(cache.fields) or not _valid_entities(cache.entities):
        raise TypeError('cache schema mappings are malformed')


def _parse_cache_payload(payload: object) -> SchemaCache:
    """Parse one structurally validated JSON cache payload."""
    if not isinstance(payload, dict):
        raise TypeError('cache payload is not an object')
    values = cast('dict[object, object]', payload)
    fields = values.get('fields')
    entities = values.get('entities')
    fetched_raw = values.get('fetched_at')
    server_info = values.get('server_info')
    if not isinstance(fetched_raw, str):
        raise TypeError('cache fetched_at is not a string')
    fetched_at = datetime.fromisoformat(fetched_raw)
    if fetched_at.tzinfo is None:
        raise ValueError('cache fetched_at is not timezone-aware')
    if server_info is not None and not isinstance(server_info, dict):
        raise TypeError('cache server_info is not an object')
    cache = SchemaCache(
        fields=cast('dict[str, dict[str, Any]]', fields),
        entities=cast('dict[str, dict[str, Any]]', entities),
        fetched_at=fetched_at,
        server_info=cast('dict[str, Any] | None', server_info),
    )
    validate_cache(cache)
    return cache


def load_cache(path: Path) -> SchemaCache | None:
    """Load a cached schema; missing or corrupt files yield None, never an error.

    Args:
        path: Cache file to read.

    Returns:
        Parsed cache, or None when the file is missing or corrupt.
    """
    try:
        payload = cast('object', json.loads(path.read_text(encoding='utf-8')))
        return _parse_cache_payload(payload)
    except (OSError, ValueError, KeyError, TypeError):
        return None
