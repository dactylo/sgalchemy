from collections.abc import Mapping
from typing import get_type_hints

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema, mapped_entity

from sgalchemy._core import _expressions as ex
from sgalchemy._core import _statement
from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._options import (
    joinedload,
    load_only,
    selectinload,
)
from sgalchemy._core._registry import Entity, Registry
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import CompilationError
from sgalchemy.schema import _model
from sgalchemy.schema._model import EntitySchema


@mapped_entity('Shot')
class _Shot(Entity):
    pass


def _f(code: str) -> ex.FieldExpr:
    return ex.FieldExpr('Shot', (code,), None)


def test_builder_returns_copies() -> None:
    base = select(_f('code'))
    limited = base.limit(10).page(2)

    assert base.limit_value == 0
    assert base.page_value == 0
    assert (limited.limit_value, limited.page_value) == (10, 2)
    assert limited is not base


def test_compile_annotations_preserve_exact_capabilities() -> None:
    hints = get_type_hints(Select[Entity].compile)

    assert hints['schema'] == Mapping[str, EntitySchema] | None
    assert hints['return'] == FindCall | SummarizeCall


def test_compile_produces_expected_find_call(fake_sg: _FakeShotgun) -> None:
    schemas = _model.build_schema(fetch_schema(fake_sg))
    statement = select(ex.FieldExpr('Shot', ('code',), None)).limit(2)

    assert statement.compile(schemas) == FindCall(
        entity_type='Shot',
        filters=[],
        fields=['code'],
        order=[],
        limit=2,
        page=0,
        extra={},
    )


def test_compile_translates_internal_joinedload_schema_drift(fake_sg: _FakeShotgun) -> None:
    registry = Registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    registry.bind(schemas, key='https://fake.example.com')
    shot_cls = registry.entity_class('Shot')
    stale: dict[str, EntitySchema] = {
        name: schema for name, schema in schemas.items() if name != 'Sequence'
    }

    with pytest.raises(
        CompilationError,
        match=r'Shot\.sg_sequence.*Sequence.*refresh',
    ):
        select(shot_cls).options(joinedload(shot_cls.sg_sequence)).compile(stale)


def test_exec_options_are_owned_by_each_statement() -> None:
    options: dict[str, object] = {'retired_only': False}
    base = _statement.Select[None](exec_options=options)
    derived = base.limit(10)

    options['retired_only'] = True

    assert base.exec_options == {'retired_only': False}
    assert derived.exec_options == {'retired_only': False}
    assert derived.exec_options is not base.exec_options


def test_execution_options_accumulate_and_later_values_win() -> None:
    base = select(_f('code')).execution_options(retired_only=False, custom='kept')
    updated = base.execution_options(retired_only=True)

    assert base.exec_options == {'retired_only': False, 'custom': 'kept'}
    assert updated.exec_options == {'retired_only': True, 'custom': 'kept'}


def test_group_by_bare_field_stamps_owner_and_preserves_groupings() -> None:
    status = _f('sg_status_list')
    explicit = _f('created_at').by_month()
    stmt = select(_f('code')).group_by(explicit).group_by(status)

    assert stmt.groupings[0] is explicit
    grouping = stmt.groupings[1]
    assert (grouping.path, grouping.mode) == ('sg_status_list', 'exact')
    assert grouping.entity_type == 'Shot'


def test_select_entity_or_projected_columns() -> None:
    code = _f('code')
    summary = ex.func.count(code)

    entity_statement = select(_Shot)
    projected_statement = select(code, summary)

    assert entity_statement.entity is _Shot
    assert entity_statement.columns == ()
    assert projected_statement.entity is None
    assert projected_statement.columns[0] is code
    assert projected_statement.columns[1] is summary


def test_select_uses_identity_equality_and_hash() -> None:
    first = select(_f('code'))
    second = select(_f('code'))
    same = first

    assert first != second
    assert first == same
    assert hash(first) == object.__hash__(first)


def test_where_order_by_and_options_accumulate_without_validation() -> None:
    code = _f('code')
    duration = _f('sg_cut_duration')
    ordering = code.desc()
    first_option = joinedload(_f('sg_sequence'))
    second_option = selectinload(_f('assets'))
    third_option = load_only(code)

    stmt = (
        select(code)
        .where(code == 'AB', duration > 5)
        .where(duration < 100)
        .order_by(ordering)
        .options(first_option, second_option)
        .options(third_option)
    )

    assert len(stmt.conditions) == 3
    assert stmt.orderings == (ordering,)
    assert stmt.loader_options == (first_option, second_option, third_option)
