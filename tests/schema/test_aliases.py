from _fakes import _FakeShotgun
from _support import fetch_schema

from sgalchemy.schema import _aliases, _model


def _entity(type_name: str, display_name: str, *field_codes: str) -> _model.EntitySchema:
    fields = {
        code: _model.FieldSchema(
            code=code,
            data_type='text',
            python_type=str,
            api_readable=True,
            valid_types=(),
            display_name=code,
        )
        for code in field_codes
    }
    return _model.EntitySchema(
        type_name=type_name,
        display_name=display_name,
        fields=fields,
        name_field=None,
    )


def _schemas(fake_sg: _FakeShotgun) -> dict[str, _model.EntitySchema]:
    return _model.build_schema(fetch_schema(fake_sg))


def test_entity_alias_from_display_name(fake_sg: _FakeShotgun) -> None:
    aliases, reports = _aliases.entity_aliases(_schemas(fake_sg))

    assert aliases['Vendor'] == 'CustomEntity03'
    assert aliases['Person'] == 'HumanUser'
    assert 'Shot' not in aliases
    assert not reports


def test_entity_alias_skips_invalid_self_and_collisions() -> None:
    schemas = {
        'AType': _entity('AType', 'Shared Name'),
        'Canonical': _entity('Canonical', 'Canonical'),
        'OtherType': _entity('OtherType', 'Canonical'),
        'SecondType': _entity('SecondType', 'Shared-Name'),
        'Unusable': _entity('Unusable', '---'),
    }

    aliases, reports = _aliases.entity_aliases(schemas)

    assert aliases == {'SharedName': 'AType'}
    assert len(reports) == 2
    assert any('OtherType' in report and 'Canonical' in report for report in reports)
    assert any('SecondType' in report and 'AType' in report for report in reports)


def test_entity_alias_skips_python_constants() -> None:
    schemas = {
        'AType': _entity('AType', 'None'),
        'BType': _entity('BType', 'True'),
        'CType': _entity('CType', 'False'),
    }

    aliases, reports = _aliases.entity_aliases(schemas)

    assert aliases == {}
    assert reports == []


def test_field_alias_collision_with_prefixed_canonical_field_is_reported() -> None:
    entity = _entity('Example', 'Example', 'sg_name', 'sg_sg_name')

    aliases, reports = _aliases.field_aliases(entity)

    assert aliases == {'name': 'sg_name'}
    assert len(reports) == 1
    assert 'sg_sg_name' in reports[0]


def test_field_alias_strip_and_collision(fake_sg: _FakeShotgun) -> None:
    schemas = _schemas(fake_sg)

    shot_aliases, shot_reports = _aliases.field_aliases(schemas['Shot'])
    assert shot_aliases['status_list'] == 'sg_status_list'
    assert shot_aliases['vendor'] == 'sg_vendor'
    assert not shot_reports

    asset_aliases, asset_reports = _aliases.field_aliases(schemas['Asset'])
    assert 'code' not in asset_aliases
    assert any('sg_code' in report for report in asset_reports)


def test_field_aliases_skip_python_and_entity_attribute_names() -> None:
    entity = _entity(
        'Example',
        'Example',
        'sg_class',
        'sg_new',
        'sg_mro',
        'sg__private',
        'sg_valid',
    )

    aliases, reports = _aliases.field_aliases(
        entity,
        reserved_names={'mro', 'new'},
    )

    assert aliases == {'valid': 'sg_valid'}
    assert len(reports) == 4
    assert all(name in ' '.join(reports) for name in ('class', 'mro', 'new', '_private'))
