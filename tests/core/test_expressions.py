from collections.abc import Callable
from dataclasses import fields
from typing import Any, get_origin, get_type_hints

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema

from sgalchemy._core import _expressions as ex
from sgalchemy._core._expressions import _SUMMARY_WIRE, func
from sgalchemy._errors import CompilationError
from sgalchemy.schema import _model


def _shot_field(code: str) -> ex.FieldExpr:
    return ex.FieldExpr('Shot', (code,), None)


def test_all_documented_grouping_modes() -> None:
    field = _shot_field('sg_due_date')

    assert {mode: field.by_(mode).mode for mode in ex._GROUPING_MODES} == {
        mode: mode for mode in ex._GROUPING_MODES
    }
    assert (
        frozenset(
            {
                'exact',
                'tens',
                'hundreds',
                'thousands',
                'tensofthousands',
                'hundredsofthousands',
                'millions',
                'day',
                'week',
                'month',
                'quarter',
                'year',
                'clustered_date',
                'oneday',
                'fivedays',
                'entitytype',
                'firstletter',
            }
        )
        == ex._GROUPING_MODES
    )


@pytest.mark.parametrize(
    ('name', 'wire_type'),
    [
        ('count', 'count'),
        ('sum', 'sum'),
        ('avg', 'average'),
        ('min', 'minimum'),
        ('max', 'maximum'),
        ('earliest', 'earliest'),
        ('latest', 'latest'),
        ('percentage', 'percentage'),
        ('status_percentage', 'status_percentage'),
        ('status_list', 'status_list'),
        ('checked', 'checked'),
        ('unchecked', 'unchecked'),
        ('record_count', 'record_count'),
    ],
)
def test_all_summary_spellings(name: str, wire_type: str) -> None:
    assert getattr(ex.func, name)(_shot_field('id')) == ex.Summary('id', wire_type, 'Shot')


def test_boolean_nesting_preserves_full_structure() -> None:
    code = _shot_field('code')
    group = ex.or_(code.startswith('AB'), ex.and_(code.contains('_'), code != 'X'))
    assert group == ex.FilterGroup(
        'any',
        (
            ex.Filter('code', 'starts_with', 'AB'),
            ex.FilterGroup(
                'all',
                (
                    ex.Filter('code', 'contains', '_'),
                    ex.Filter('code', 'is_not', 'X'),
                ),
            ),
        ),
    )


def test_comparison_operators() -> None:
    f = _shot_field('sg_cut_duration')
    assert (f == 5) == ex.Filter('sg_cut_duration', 'is', 5)
    assert (f != 5) == ex.Filter('sg_cut_duration', 'is_not', 5)
    assert (f > 5) == ex.Filter('sg_cut_duration', 'greater_than', 5)
    assert (f < 5) == ex.Filter('sg_cut_duration', 'less_than', 5)
    assert (f >= 5) == ex.Filter('sg_cut_duration', 'between', [5, None])
    assert (f <= 5) == ex.Filter('sg_cut_duration', 'between', [None, 5])


@pytest.mark.parametrize(
    ('method', 'args', 'operator', 'value', 'extra_values'),
    [
        ('in_', ('a', 'b'), 'in', ['a', 'b'], ()),
        ('not_in', ('a', 'b'), 'not_in', ['a', 'b'], ()),
        ('between', (1, 2), 'between', [1, 2], ()),
        ('is_', (None,), 'is', None, ()),
        ('contains', ('mid',), 'contains', 'mid', ()),
        ('startswith', ('pre',), 'starts_with', 'pre', ()),
        ('endswith', ('post',), 'ends_with', 'post', ()),
        ('in_last', (2, 'DAY'), 'in_last', 2, ('DAY',)),
        ('in_next', (3, 'WEEK'), 'in_next', 3, ('WEEK',)),
        ('name_contains', ('mid',), 'name_contains', 'mid', ()),
        ('not_contains', ('mid',), 'not_contains', 'mid', ()),
        ('not_between', (1, 2), 'not_between', [1, 2], ()),
        ('not_in_last', (2, 'DAY'), 'not_in_last', 2, ('DAY',)),
        ('not_in_next', (3, 'WEEK'), 'not_in_next', 3, ('WEEK',)),
        ('in_calendar_day', (-1,), 'in_calendar_day', -1, ()),
        ('in_calendar_week', (0,), 'in_calendar_week', 0, ()),
        ('in_calendar_month', (1,), 'in_calendar_month', 1, ()),
        ('in_calendar_year', (-1,), 'in_calendar_year', -1, ()),
        ('type_is', ('Shot',), 'type_is', 'Shot', ()),
        ('type_is_not', ('Asset',), 'type_is_not', 'Asset', ()),
        ('name_is', ('Exact',), 'name_is', 'Exact', ()),
        ('name_id', (88,), 'name_id', 88, ()),
        ('name_not_contains', ('bad',), 'name_not_contains', 'bad', ()),
        ('name_starts_with', ('pre',), 'name_starts_with', 'pre', ()),
        ('name_ends_with', ('post',), 'name_ends_with', 'post', ()),
    ],
)
def test_complete_method_operator_inventory(
    method: str,
    args: tuple[Any, ...],
    operator: str,
    value: object,
    extra_values: tuple[Any, ...],
) -> None:
    field = _shot_field('code')
    result = getattr(field, method)(*args)
    assert result == ex.Filter('code', operator, value, extra_values)


def test_dot_field_traversal_renders_path(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    seq = ex.FieldExpr('Shot', ('sg_sequence',), schemas['Shot'].fields['sg_sequence'])
    deep = seq.code == 'SEQ01'
    assert deep == ex.Filter('sg_sequence.Sequence.code', 'is', 'SEQ01')
    with pytest.raises(CompilationError):
        _ = ex.FieldExpr(
            'Shot', ('code',), schemas['Shot'].fields['code']
        ).anything  # text field has no relationships


def test_expression_exposes_compiler_contracts() -> None:
    field = _shot_field('code')
    assert field.path == ('code',)
    assert field.root_type == 'Shot'


def test_expression_outputs_carry_root_entity_stamps() -> None:
    field = _shot_field('code')

    assert field.is_('SHOT_001').entity_type == 'Shot'
    assert field.asc().entity_type == 'Shot'
    assert field.by_('exact').entity_type == 'Shot'
    assert ex.func.count(field).entity_type == 'Shot'


def test_field_expr_equality_still_builds_filter() -> None:
    condition = _shot_field('code') == 'AB'

    assert isinstance(condition, ex.Filter)
    assert (condition.path, condition.op, condition.value) == ('code', 'is', 'AB')


def test_filter_field_order_keeps_extra_values_before_entity_type() -> None:
    assert [field.name for field in fields(ex.Filter)] == [
        'path',
        'op',
        'value',
        'extra_values',
        'entity_type',
    ]


def test_func_type_declarations_match_runtime_summary_names() -> None:
    hints = get_type_hints(type(func))
    assert set(hints) == set(_SUMMARY_WIRE)
    assert all(get_origin(annotation) is Callable for annotation in hints.values())


def test_missing_target_field_teaches_refresh_then_valid_code(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    sequence = ex.FieldExpr(
        'Shot', ('sg_sequence',), schemas['Shot'].fields['sg_sequence'], schemas=schemas
    )

    with pytest.raises(
        CompilationError,
        match=(r'engine\.load_schema\(force=True\).*use a valid field code if it remains absent'),
    ):
        _ = sequence.not_a_field


def test_multi_entity_traversal_rejected(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    assets = ex.FieldExpr('Shot', ('assets',), schemas['Shot'].fields['assets'])
    with pytest.raises(CompilationError, match='multi-entity'):
        _ = assets.code  # the wire drills through single-entity links only


def test_multi_hop_traversal_chains_with_schemas_mapping(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    seq = ex.FieldExpr(
        'Shot', ('sg_sequence',), schemas['Shot'].fields['sg_sequence'], schemas=schemas
    )
    deep = seq.episode.code == 'EP01'  # two hops: type tokens inserted at each
    assert deep == ex.Filter('sg_sequence.Sequence.episode.Episode.code', 'is', 'EP01')
    with pytest.raises(CompilationError):
        _ = seq.nope  # unknown field on the hop target, caught at build time


def test_ordering_grouping_and_func() -> None:
    f = _shot_field('sg_due_date')
    assert f.desc() == ex.Ordering('sg_due_date', 'desc')
    assert f.by_month() == ex.Grouping('sg_due_date', 'month')
    assert f.by_('quarter') == ex.Grouping('sg_due_date', 'quarter')
    with pytest.raises(CompilationError, match='grouping mode'):
        f.by_('fortnight')
    assert ex.func.avg(f) == ex.Summary('sg_due_date', 'average', 'Shot')
    assert ex.func.count(_shot_field('id')) == ex.Summary('id', 'count', 'Shot')
    unsupported = 'median'
    with pytest.raises(AttributeError):
        getattr(ex.func, unsupported)  # runtime rejects unsupported summary names


def test_polymorphic_traversal_teaches_concrete_filter_or_loader(
    fake_sg: _FakeShotgun,
) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    entity = ex.FieldExpr('Version', ('entity',), schemas['Version'].fields['entity'])

    with pytest.raises(
        CompilationError,
        match=r'filter a concrete target type directly or use selectinload\(\)',
    ):
        _ = entity.code


def test_raw_filters_preserve_list_identity() -> None:
    filters: list[Any] = [['code', 'is', 'SHOT_001']]

    result = ex.raw(filters)

    assert result.filters is filters


def test_scalar_traversal_teaches_direct_filtering(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    code = ex.FieldExpr('Shot', ('code',), schemas['Shot'].fields['code'])

    with pytest.raises(CompilationError, match='filter the scalar field directly'):
        _ = code.anything


def test_unknown_target_teaches_schema_refresh(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    del schemas['CustomEntity03']
    vendor = ex.FieldExpr(
        'Shot', ('sg_vendor',), schemas['Shot'].fields['sg_vendor'], schemas=schemas
    )

    with pytest.raises(CompilationError, match=r'engine\.load_schema\(force=True\)'):
        _ = vendor.code
