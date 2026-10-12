import pytest

from sgalchemy._core import _expressions as ex
from sgalchemy._core._options import (
    FIND_EXECUTION_OPTIONS,
    SUMMARIZE_EXECUTION_OPTIONS,
    Defer,
    JoinedLoad,
    LazyLoad,
    LoadOnly,
    SelectinLoad,
    defer,
    joinedload,
    load_only,
    selectinload,
)


def _f(code: str) -> ex.FieldExpr:
    return ex.FieldExpr('Shot', (code,), None)


def test_execution_option_inventories_are_exact() -> None:
    assert (
        frozenset({'retired_only', 'include_archived_projects', 'additional_filter_presets'})
        == FIND_EXECUTION_OPTIONS
    )
    assert frozenset({'include_archived_projects'}) == SUMMARIZE_EXECUTION_OPTIONS


@pytest.mark.parametrize(
    ('left', 'equal', 'unequal', 'extra', 'other_type'),
    [
        (
            Defer((_f('code'),)),
            Defer((_f('code'),)),
            Defer((_f('description'),)),
            Defer((_f('sg_status_list'),)),
            LoadOnly((_f('code'),)),
        ),
        (
            JoinedLoad((_f('code'),)),
            JoinedLoad((_f('code'),)),
            JoinedLoad((_f('description'),)),
            JoinedLoad((_f('sg_status_list'),)),
            SelectinLoad(_f('code')),
        ),
        (
            LazyLoad(_f('code')),
            LazyLoad(_f('code')),
            LazyLoad(_f('description')),
            LazyLoad(_f('sg_status_list')),
            SelectinLoad(_f('code')),
        ),
        (
            LoadOnly((_f('code'),)),
            LoadOnly((_f('code'),)),
            LoadOnly((_f('description'),)),
            LoadOnly((_f('sg_status_list'),)),
            Defer((_f('code'),)),
        ),
        (
            SelectinLoad(_f('code')),
            SelectinLoad(_f('code')),
            SelectinLoad(_f('description')),
            SelectinLoad(_f('sg_status_list')),
            LazyLoad(_f('code')),
        ),
    ],
    ids=['defer', 'joinedload', 'lazyload', 'load-only', 'selectinload'],
)
def test_loader_option_equality_is_boolean_and_structural(
    left: object,
    equal: object,
    unequal: object,
    extra: object,
    other_type: object,
) -> None:
    assert type(left == equal) is bool
    assert left == equal
    assert hash(left) == hash(equal)
    assert type(left == unequal) is bool
    assert left != unequal
    assert unequal not in [left, extra]
    assert type(left == other_type) is bool
    assert left != other_type


def test_loader_option_factories_preserve_fields_and_chain() -> None:
    sequence = _f('sg_sequence')
    project = _f('project')
    assets = _f('assets')
    code = _f('code')
    description = _f('description')

    joined = joinedload(sequence).joinedload(project)
    selected = selectinload(assets)
    only = load_only(code, description)
    deferred = defer(description, code)

    assert isinstance(joined, JoinedLoad)
    assert len(joined.paths) == 2
    assert all(
        actual is expected
        for actual, expected in zip(joined.paths, (sequence, project), strict=True)
    )
    assert isinstance(selected, SelectinLoad)
    assert selected.field is assets
    assert isinstance(only, LoadOnly)
    assert len(only.fields) == 2
    assert all(
        actual is expected
        for actual, expected in zip(only.fields, (code, description), strict=True)
    )
    assert isinstance(deferred, Defer)
    assert len(deferred.fields) == 2
    assert all(
        actual is expected
        for actual, expected in zip(deferred.fields, (description, code), strict=True)
    )
