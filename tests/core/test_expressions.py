from collections.abc import Callable
from dataclasses import fields
from typing import Any, get_origin, get_type_hints

import pytest

from sgalchemy._core import _expressions as ex
from sgalchemy._core._expressions import _SUMMARY_WIRE, func
from sgalchemy._errors import CompilationError
from sgalchemy.schema import _model
from tests._fakes import FakeShotgun
from tests._support import fetch_schema

_FILTER_TRUTH = (
    'filter conditions have no truth value; combine them with and_(), or_(), or several where() '
    "arguments instead of Python's and/or/not, and use field.in_(values) instead of Python's in"
)


def _bound_and_symbolic(fake_sg: FakeShotgun) -> tuple[ex.FieldExpr, ex.FieldExpr]:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    seq = ex.FieldExpr(
        'Shot', ('sg_sequence',), schemas['Shot'].fields['sg_sequence'], schemas=schemas
    )
    return seq.code, ex.FieldExpr('Shot', ('sg_sequence',), None).Sequence.code


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


def test_and_or_reject_non_conditions() -> None:
    prefix = 'takes filter conditions (Filter, FilterGroup, or RawFilters), not'
    with pytest.raises(TypeError) as excinfo:
        ex.and_(None)  # pyright: ignore[reportArgumentType]
    assert str(excinfo.value) == f'and_() {prefix} NoneType'

    invalid = True
    with pytest.raises(TypeError) as excinfo:
        ex.or_(invalid)  # pyright: ignore[reportArgumentType]
    assert str(excinfo.value) == (
        f"or_() {prefix} bool; Python's in, and, and or were already evaluated, "
        'so use field.in_(), and_(), or or_() instead'
    )


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


def test_dot_field_traversal_renders_path(fake_sg: FakeShotgun) -> None:
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


def test_field_hash_matches_structural_equality(fake_sg: FakeShotgun) -> None:
    bound, symbolic = _bound_and_symbolic(fake_sg)
    assert hash(bound) == hash(symbolic)
    assert symbolic in {bound, _shot_field('code')}
    assert {bound: 'kept'}[symbolic] == 'kept'
    assert ex.FieldExpr('Asset', ('code',), None) not in {
        _shot_field('code'),
        _shot_field('sg_status_list'),
    }


def test_field_to_field_comparison_has_structural_identity(fake_sg: FakeShotgun) -> None:
    bound, symbolic = _bound_and_symbolic(fake_sg)
    assert bool(bound == symbolic)
    assert not bool(bound != symbolic)
    assert not bool(_shot_field('code') == ex.FieldExpr('Asset', ('code',), None))
    assert _shot_field('code') in [_shot_field('sg_status_list'), _shot_field('code')]


@pytest.mark.parametrize(
    'condition',
    [
        _shot_field('code') == 'A',
        ex.and_(_shot_field('code') == 'A'),
        ex.raw([['code', 'is', 'A']]),
    ],
    ids=['filter', 'group', 'raw'],
)
def test_filter_conditions_have_no_truth_value(condition: object) -> None:
    with pytest.raises(TypeError) as excinfo:
        bool(condition)
    assert str(excinfo.value) == _FILTER_TRUTH


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


@pytest.mark.parametrize('method', ['in_', 'not_in'])
def test_in_accepts_one_collection_or_separate_values(method: str) -> None:
    build = getattr(_shot_field('sg_status_list'), method)
    assert build(['ip', 'rev']).value == ['ip', 'rev']
    assert build(('ip', 'rev')).value == ['ip', 'rev']
    assert sorted(build({'ip', 'rev'}).value) == ['ip', 'rev']
    assert sorted(build(frozenset({'ip', 'rev'})).value) == ['ip', 'rev']
    assert build('ip', 'rev').value == ['ip', 'rev']
    assert build('ip').value == ['ip']
    assert build().value == []


@pytest.mark.parametrize('method', ['in_', 'not_in'])
def test_in_expands_other_single_iterables_but_not_text_or_mappings(method: str) -> None:
    assert getattr(_shot_field('id'), method)(range(1, 4)).value == [1, 2, 3]
    assert getattr(_shot_field('code'), method)(c for c in 'ab').value == ['a', 'b']
    assert getattr(_shot_field('code'), method)({'a': 1, 'b': 2}.keys()).value == ['a', 'b']
    assert getattr(_shot_field('code'), method)(b'ab').value == [b'ab']
    assert getattr(_shot_field('code'), method)(bytearray(b'ab')).value == [bytearray(b'ab')]
    link = {'type': 'Sequence', 'id': 1}
    assert getattr(_shot_field('sg_sequence'), method)(link).value == [link]


@pytest.mark.parametrize('method', ['in_', 'not_in'])
def test_in_rejects_a_collection_mixed_with_other_arguments(method: str) -> None:
    build = getattr(_shot_field('sg_status_list'), method)
    for values in (['ip'], 'rev'), ('ip', ('rev',)):
        with pytest.raises(TypeError) as excinfo:
            build(*values)
        assert str(excinfo.value) == (
            f'{method}() takes either one iterable of values or the values as separate arguments, '
            'not both'
        )


def test_missing_target_field_teaches_refresh_then_valid_code(fake_sg: FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    sequence = ex.FieldExpr(
        'Shot', ('sg_sequence',), schemas['Shot'].fields['sg_sequence'], schemas=schemas
    )

    with pytest.raises(
        CompilationError,
        match=(r'engine\.load_schema\(force=True\).*use a valid field code if it remains absent'),
    ):
        _ = sequence.not_a_field


def test_multi_entity_traversal_rejected(fake_sg: FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    assets = ex.FieldExpr('Shot', ('assets',), schemas['Shot'].fields['assets'])
    with pytest.raises(CompilationError, match='multi-entity'):
        _ = assets.code  # the wire drills through single-entity links only


def test_multi_hop_traversal_chains_with_schemas_mapping(fake_sg: FakeShotgun) -> None:
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
    fake_sg: FakeShotgun,
) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    entity = ex.FieldExpr('Version', ('entity',), schemas['Version'].fields['entity'])

    with pytest.raises(
        CompilationError,
        match=r'filter a concrete target type directly or use selectinload\(\)',
    ):
        _ = entity.code


def test_python_and_or_not_and_in_raise_instead_of_dropping_conditions() -> None:
    code = _shot_field('code')
    first = code == 'A'
    second = code == 'B'
    with pytest.raises(TypeError, match='no truth value'):
        _ = first and second
    with pytest.raises(TypeError, match='no truth value'):
        _ = first or second
    with pytest.raises(TypeError, match='no truth value'):
        _ = not first
    with pytest.raises(TypeError, match='no truth value'):
        _ = code in ['A', 'B']


def test_raw_filters_preserve_list_identity() -> None:
    filters: list[Any] = [['code', 'is', 'SHOT_001']]

    result = ex.raw(filters)

    assert result.filters is filters


@pytest.mark.parametrize('method', ['in_last', 'in_next', 'not_in_last', 'not_in_next'])
def test_relative_date_units_are_checked_and_sent_uppercase(method: str) -> None:
    build = getattr(_shot_field('sg_due_date'), method)
    for unit in ('HOUR', 'day', 'Week', 'month', 'YEAR'):
        condition = build(2, unit)
        assert (condition.op, condition.value, condition.extra_values) == (
            method,
            2,
            (unit.upper(),),
        )
    for bad in ('DAYS', 3):
        with pytest.raises(CompilationError) as excinfo:
            build(2, bad)
        assert str(excinfo.value) == (
            f'unknown time unit {bad!r}; server-supported units: HOUR, DAY, WEEK, MONTH, YEAR '
            '(any letter case)'
        )


def test_scalar_traversal_teaches_direct_filtering(fake_sg: FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    code = ex.FieldExpr('Shot', ('code',), schemas['Shot'].fields['code'])

    with pytest.raises(CompilationError, match='filter the scalar field directly'):
        _ = code.anything


def test_unknown_target_teaches_schema_refresh(fake_sg: FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    del schemas['CustomEntity03']
    vendor = ex.FieldExpr(
        'Shot', ('sg_vendor',), schemas['Shot'].fields['sg_vendor'], schemas=schemas
    )

    with pytest.raises(CompilationError, match=r'engine\.load_schema\(force=True\)'):
        _ = vendor.code
