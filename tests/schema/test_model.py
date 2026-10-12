import datetime
from pathlib import Path

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema

from sgalchemy.schema import _cache, _model


def test_build_schema_types_and_name_field(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    shot = schemas['Shot']
    assert shot.display_name == 'Shot'
    assert shot.name_field == 'code'
    assert shot.fields['sg_cut_duration'].python_type is int
    assert shot.fields['sg_due_date'].python_type is datetime.date
    assert shot.fields['sg_sequence'].valid_types == ('Sequence',)
    assert _model._DATA_TYPE_TO_PY['color'] is str
    assert _model._DATA_TYPE_TO_PY['password'] is str
    assert _model._DATA_TYPE_TO_PY['system_task_type'] is str


@pytest.mark.parametrize(
    ('raw_editable', 'expected'),
    [
        ({'value': False}, False),
        ({'value': True}, True),
        ({'value': 'false'}, True),
        ({}, True),
    ],
)
def test_field_editability_defaults_safely(
    fake_sg: _FakeShotgun,
    raw_editable: dict[str, object],
    expected: bool,
) -> None:
    cache = fetch_schema(fake_sg)
    cache.fields['Shot']['description']['editable'] = raw_editable
    field = _model.build_schema(cache)['Shot'].fields['description']
    assert field.editable is expected


def test_old_cached_schema_without_editability_remains_valid(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache = fetch_schema(fake_sg)
    cache.fields['Shot']['description'].pop('editable', None)
    path = tmp_path / 'old-schema.json'
    _cache.store_cache(path, cache)

    loaded = _cache.load_cache(path)

    assert loaded is not None
    assert _model.build_schema(loaded)['Shot'].fields['description'].editable


def test_pipeline_step_summary_fields_are_not_api_readable(fake_sg: _FakeShotgun) -> None:
    cache = fetch_schema(fake_sg)
    cache.fields['Shot']['step_1'] = {
        'data_type': {'value': 'summary'},
        'name': {'value': 'Animation'},
        'properties': {},
    }
    schemas = _model.build_schema(cache)
    assert not schemas['Shot'].fields['step_1'].api_readable
    assert 'step_1' not in schemas['Shot'].readable_field_codes()


def test_query_fields_are_not_api_readable(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    shot = schemas['Shot']
    assert not shot.fields['sg_qc_summary'].api_readable
    codes = shot.readable_field_codes()
    assert 'sg_qc_summary' not in codes
    assert 'code' in codes
    assert codes == sorted(codes)


def test_unknown_data_type_is_object_and_readable(fake_sg: _FakeShotgun) -> None:
    cache = fetch_schema(fake_sg)
    cache.fields['Shot']['sg_new'] = {
        'data_type': {'value': 'holo_volume'},
        'name': {'value': 'New Thing'},
        'properties': {},
    }
    schemas = _model.build_schema(cache)
    assert schemas['Shot'].fields['sg_new'].python_type is object
    assert schemas['Shot'].fields['sg_new'].api_readable
