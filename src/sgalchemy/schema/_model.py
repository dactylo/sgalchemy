"""Parsed schema models and the data type map."""

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sgalchemy.schema._cache import SchemaCache

_DATA_TYPE_TO_PY: dict[str, type] = {
    'text': str,
    'list': str,
    'status_list': str,
    'uuid': str,
    'footage': str,
    'color': str,
    'password': str,
    'system_task_type': str,
    'number': int,
    'duration': int,
    'timecode': int,
    'percent': int,
    'float': float,
    'currency': float,
    'checkbox': bool,
    'date': datetime.date,
    'date_time': datetime.datetime,
    'entity': dict,
    'multi_entity': list,
    'tag_list': list,
    'addressing': list,
    'serializable': dict,
    'url': dict,
    'image': str,
}
_NAME_FIELD_CANDIDATES = ('code', 'name', 'content', 'title')


@dataclass(frozen=True)
class FieldSchema:
    """Describe one field reflected from a site's schema.

    Attributes:
        code: Canonical provider field code.
        data_type: Provider data-type name.
        python_type: Python type associated with the provider wire value;
            relationship hydration may expose Entity objects instead.
        api_readable: Whether the field has usable API data, excluding UI-only
            query and summary fields.
        valid_types: Canonical target types declared for a relationship.
        display_name: Reflected human-readable field name.
        editable: Cached editability advice, defaulting to true when metadata
            is absent or unusable. Provider permissions remain authoritative.
    """

    code: str
    data_type: str
    python_type: type
    api_readable: bool
    valid_types: tuple[str, ...]
    display_name: str
    editable: bool = True


@dataclass(frozen=True)
class EntitySchema:
    """Describe a reflected entity type and its canonical fields.

    Attributes:
        type_name: Canonical provider entity type.
        display_name: Reflected human-readable entity name.
        fields: Field schemas keyed by canonical provider code.
        name_field: Conventional field used for display and link names, or
            `None` when no supported name candidate exists.
    """

    type_name: str
    display_name: str
    fields: Mapping[str, FieldSchema]
    name_field: str | None

    def readable_field_codes(self) -> list[str]:
        """Return sorted codes of all API-readable fields.

        Returns:
            Field codes safe to request through the API, sorted for
            determinism.
        """
        return sorted(code for code, field in self.fields.items() if field.api_readable)


def _build_field(code: str, raw: dict[str, Any]) -> FieldSchema:
    """Parse one raw schema field dictionary into a `FieldSchema`."""
    data_type = raw.get('data_type', {}).get('value', '')
    editable = raw.get('editable', {}).get('value', True)
    properties = raw.get('properties', {})
    valid_types = tuple(properties.get('valid_types', {}).get('value', []) or [])
    api_readable = 'query' not in properties and data_type != 'summary'
    return FieldSchema(
        code=code,
        data_type=data_type,
        python_type=_DATA_TYPE_TO_PY.get(data_type, object),
        api_readable=api_readable,
        valid_types=valid_types,
        display_name=raw.get('name', {}).get('value', code),
        editable=editable if type(editable) is bool else True,
    )


def build_schema(cache: SchemaCache) -> dict[str, EntitySchema]:
    """Parse a raw schema cache into typed schema models.

    Args:
        cache: The raw cached `schema_read` and `schema_entity_read` payloads.

    Returns:
        Mapping of entity type name to its parsed `EntitySchema`.
    """
    schemas: dict[str, EntitySchema] = {}
    for type_name, raw_fields in cache.fields.items():
        fields = {code: _build_field(code, raw) for code, raw in raw_fields.items()}
        name_field = next(
            (candidate for candidate in _NAME_FIELD_CANDIDATES if candidate in fields), None
        )
        display_name = cache.entities.get(type_name, {}).get('name', {}).get('value', type_name)
        schemas[type_name] = EntitySchema(
            type_name=type_name,
            display_name=display_name,
            fields=fields,
            name_field=name_field,
        )
    return schemas
