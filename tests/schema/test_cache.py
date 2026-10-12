import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema

from sgalchemy.schema import _cache


def _valid_payload() -> dict[str, Any]:
    return {
        'fields': {
            'Shot': {
                'code': {
                    'data_type': {'value': 'text'},
                    'name': {'value': 'Code'},
                    'properties': {},
                }
            }
        },
        'entities': {'Shot': {'name': {'value': 'Shot'}}},
        'fetched_at': '2026-07-01T00:00:00+00:00',
        'server_info': None,
    }


def test_load_accepts_optional_member_shapes(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload['fields'] = {
        'Shot': {
            'code': {
                'data_type': {'value': 'text'},
                'editable': {'value': True},
                'name': {'value': 'Code'},
                'properties': {'valid_types': {'value': None}},
            },
            'description': {
                'data_type': {'value': 'text'},
                'name': {'value': 'Description'},
                'properties': {},
            },
        }
    }
    path = tmp_path / 'optional.json'
    path.write_text(json.dumps(payload), encoding='utf-8')

    assert _cache.load_cache(path) is not None


def test_load_missing_or_corrupt_is_none(tmp_path: Path) -> None:
    assert _cache.load_cache(tmp_path / 'nope.json') is None
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json', encoding='utf-8')
    assert _cache.load_cache(bad) is None


@pytest.mark.parametrize(
    'replacement',
    [
        ('fields', []),
        ('fields', {'Shot': []}),
        ('fields', {'Shot': {'code': []}}),
        (
            'fields',
            {'Shot': {'code': {'data_type': [], 'name': {}, 'properties': {}}}},
        ),
        (
            'fields',
            {
                'Shot': {
                    'code': {
                        'data_type': {'value': 'text'},
                        'name': {'value': 'Code'},
                        'properties': {'valid_types': None},
                    }
                }
            },
        ),
        (
            'fields',
            {
                'Shot': {
                    'code': {
                        'data_type': {'value': 'text'},
                        'name': {'value': 'Code'},
                        'properties': {'valid_types': ['Sequence']},
                    }
                }
            },
        ),
        (
            'fields',
            {
                'Shot': {
                    'code': {
                        'data_type': {'value': 'text'},
                        'editable': None,
                        'name': {'value': 'Code'},
                        'properties': {},
                    }
                }
            },
        ),
        (
            'fields',
            {
                'Shot': {
                    'code': {
                        'data_type': {'value': 'text'},
                        'editable': 'yes',
                        'name': {'value': 'Code'},
                        'properties': {},
                    }
                }
            },
        ),
        ('entities', []),
        ('fetched_at', 1),
        ('fetched_at', '2026-07-01T00:00:00'),
        ('server_info', []),
    ],
)
def test_load_valid_json_with_malformed_shapes_is_none(
    tmp_path: Path,
    replacement: tuple[str, Any],
) -> None:
    payload = _valid_payload()
    key, value = replacement
    payload[key] = value
    path = tmp_path / 'malformed.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert _cache.load_cache(path) is None


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        pytest.param({'version': [8, 64, 0]}, {'version': [8, 64, 0]}, id='dictionary'),
        pytest.param(['8.64.0'], None, id='non-dictionary'),
        pytest.param(None, None, id='none'),
    ],
)
def test_normalize_server_info_preserves_only_dictionaries(
    value: object,
    expected: dict[str, Any] | None,
) -> None:
    assert _cache.normalize_server_info(value) == expected


def test_round_trip(tmp_path: Path, fake_sg: _FakeShotgun) -> None:
    cache = fetch_schema(fake_sg)
    path = tmp_path / 'x.json'
    _cache.store_cache(path, cache)
    loaded = _cache.load_cache(path)
    assert loaded is not None
    assert loaded.fields == cache.fields
    assert loaded.fetched_at == cache.fetched_at
    assert loaded.server_info == cache.server_info


def test_store_replacement_failure_preserves_existing_cache(
    tmp_path: Path,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = fetch_schema(fake_sg)
    path = tmp_path / 'schema.json'
    _cache.store_cache(path, cache)
    original = path.read_bytes()
    failure = OSError('replacement failed')
    monkeypatch.setattr(Path, 'replace', Mock(side_effect=failure))

    with pytest.raises(OSError, match='replacement failed') as exc_info:
        _cache.store_cache(path, cache)

    assert exc_info.value is failure
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_store_serialization_failure_preserves_existing_cache(
    tmp_path: Path,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = fetch_schema(fake_sg)
    path = tmp_path / 'schema.json'
    _cache.store_cache(path, cache)
    original = path.read_bytes()
    failure = TypeError('serialization failed')
    monkeypatch.setattr(_cache.json, 'dump', Mock(side_effect=failure))

    with pytest.raises(TypeError) as exc_info:
        _cache.store_cache(path, cache)

    assert exc_info.value is failure
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_validate_cache_accepts_valid_schema_cache() -> None:
    payload = _valid_payload()
    cache = _cache.SchemaCache(
        fields=payload['fields'],
        entities=payload['entities'],
        fetched_at=datetime.now(UTC),
    )

    _cache.validate_cache(cache)


@pytest.mark.parametrize(
    ('fields', 'entities'),
    [
        pytest.param(
            {'Shot': {'code': ['schema-secret']}},
            {'Shot': {'name': {'value': 'Shot'}}},
            id='field-metadata',
        ),
        pytest.param(
            {
                'Shot': {
                    'code': {
                        'data_type': {'value': 'text'},
                        'name': {'value': 'Code'},
                        'properties': {},
                    }
                }
            },
            {'Shot': ['schema-secret']},
            id='entity-metadata',
        ),
        pytest.param(
            {
                'Shot': {
                    'project': {
                        'data_type': {'value': 'entity'},
                        'name': {'value': 'Project'},
                        'properties': {'valid_types': {'value': 'schema-secret'}},
                    }
                }
            },
            {'Shot': {'name': {'value': 'Shot'}}},
            id='valid-types-value',
        ),
    ],
)
def test_validate_cache_rejects_malformed_schema_shapes(
    fields: dict[str, Any],
    entities: dict[str, Any],
) -> None:
    cache = _cache.SchemaCache(
        fields=fields,
        entities=entities,
        fetched_at=datetime.now(UTC),
    )

    with pytest.raises(TypeError, match='schema mappings are malformed'):
        _cache.validate_cache(cache)
