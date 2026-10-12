import pytest
from _support import mapped_entity

from sgalchemy._core import _expressions as ex
from sgalchemy._core._compiler import SummarizeCall, compile_statement
from sgalchemy._core._options import load_only
from sgalchemy._core._registry import Entity
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import CompilationError
from sgalchemy.schema import _model


@mapped_entity('Asset')
class _AssetTarget(Entity):
    pass


@mapped_entity('Shot')
class _ShotTarget(Entity):
    pass


def _f(code: str, entity_type: str = 'Shot') -> ex.FieldExpr:
    return ex.FieldExpr(entity_type, (code,), None)


def test_aggregate_with_grouping_and_conditions_has_exact_wire_shape(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = (
        select(ex.func.count(_f('id')), ex.func.avg(_f('sg_cut_duration')))
        .where(_f('sg_status_list') == 'ip')
        .group_by(_f('sg_due_date').by_month())
        .execution_options(include_archived_projects=True)
    )

    call = compile_statement(stmt, schemas)

    assert call == SummarizeCall(
        entity_type='Shot',
        filters=[['sg_status_list', 'is', 'ip']],
        summary_fields=[
            {'field': 'id', 'type': 'count'},
            {'field': 'sg_cut_duration', 'type': 'average'},
        ],
        grouping=[{'field': 'sg_due_date', 'type': 'month', 'direction': 'asc'}],
        extra={'include_archived_projects': True},
    )


def test_condition_owner_must_match_summary_owner(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(ex.func.count(_f('id'))).where(_f('code', 'Asset') == 'tree')

    with pytest.raises(CompilationError, match=r'filter on Asset\.code'):
        compile_statement(stmt, schemas)


@pytest.mark.parametrize('entity', [_ShotTarget, _AssetTarget])
def test_direct_entity_and_summary_target_is_rejected(
    entity: type[Entity],
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = Select[Entity](entity=entity, columns=(ex.func.count(_f('id')),))

    with pytest.raises(
        CompilationError,
        match=r'summary statement cannot contain an entity target.*select\(func\.\*',
    ):
        compile_statement(stmt, schemas)


def test_display_name_alias_teaches_canonical_entity_name(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=r"alias of 'CustomEntity03'"):
        compile_statement(select(ex.func.count(_f('id', 'Vendor'))), schemas)


def test_group_only_statement_requires_summary_column(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='at least one func'):
        compile_statement(select().group_by(_f('code')), schemas)


def test_grouping_owner_must_match_summary_owner(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(ex.func.count(_f('id'))).group_by(_f('code', 'Asset'))

    with pytest.raises(CompilationError, match=r'group_by on Asset\.code'):
        compile_statement(stmt, schemas)


def test_schema_backed_grouping_rejects_invalid_path(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(ex.func.count(_f('id'))).group_by(
        ex.Grouping('sg_cut_duration.Shot.code', 'exact', 'Shot')
    )

    with pytest.raises(CompilationError, match='not an entity link'):
        compile_statement(stmt, schemas)


@pytest.mark.parametrize(
    'path',
    ['missing', 'code.Project.name', 'sg_sequence.Project.name'],
)
def test_schema_backed_summary_rejects_invalid_paths(
    path: str,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError):
        compile_statement(select(ex.Summary(path, 'count', 'Shot')), schemas)


def test_summarize_accepts_only_supported_execution_option(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    call = compile_statement(
        select(ex.func.count(_f('id'))).execution_options(include_archived_projects=False),
        schemas,
    )

    assert isinstance(call, SummarizeCall)
    assert call.extra == {'include_archived_projects': False}


@pytest.mark.parametrize(
    'statement',
    [
        select(ex.func.count(_f('id'))).order_by(_f('id').desc()),
        select(ex.func.count(_f('id'))).limit(1),
        select(ex.func.count(_f('id'))).limit(-1),
        select(ex.func.count(_f('id'))).page(2),
        select(ex.func.count(_f('id'))).page(-1),
        select(ex.func.count(_f('id'))).options(load_only(_f('id'))),
    ],
)
def test_summarize_rejects_every_find_only_statement_state(
    statement: Select[Entity | None],
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='summary statements do not support'):
        compile_statement(statement, schemas)


@pytest.mark.parametrize('option', ['retired_only', 'additional_filter_presets'])
def test_summarize_rejects_find_only_execution_options(
    option: str,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='unsupported execution options'):
        compile_statement(
            select(ex.func.count(_f('id'))).execution_options(**{option: True}), schemas
        )


@pytest.mark.parametrize(
    ('statement', 'message'),
    [
        (select(ex.func.count(_f('id')), _f('code')), 'accept only func'),
        (select(_f('code')).group_by(_f('code')), 'accept only func'),
    ],
)
def test_summarize_rejects_non_summary_columns(
    statement: Select[Entity | None],
    message: str,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=message):
        compile_statement(statement, schemas)


def test_summary_columns_must_share_one_entity_owner(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(ex.func.count(_f('id')), ex.func.count(_f('id', 'Asset')))

    with pytest.raises(CompilationError, match='span entity types'):
        compile_statement(stmt, schemas)


@pytest.mark.parametrize(
    'grouping',
    [
        ex.Grouping('', 'exact', 'Shot'),
        ex.Grouping('sg_sequence.Sequence', 'exact', 'Shot'),
        ex.Grouping('sg_sequence..code', 'exact', 'Shot'),
    ],
)
def test_symbolic_grouping_rejects_malformed_wire_paths(grouping: ex.Grouping) -> None:
    stmt = select(ex.func.count(_f('id'))).group_by(grouping)

    with pytest.raises(CompilationError, match='field path'):
        compile_statement(stmt)


@pytest.mark.parametrize(
    'summary',
    [
        ex.Summary('', 'count', 'Shot'),
        ex.Summary('sg_sequence.Sequence', 'count', 'Shot'),
        ex.Summary('sg_sequence..code', 'count', 'Shot'),
    ],
)
def test_symbolic_summary_rejects_malformed_wire_paths(summary: ex.Summary) -> None:
    with pytest.raises(CompilationError, match='field path'):
        compile_statement(select(summary))


@pytest.mark.parametrize('role', ['summary', 'grouping'])
def test_ui_only_fields_are_rejected(
    role: str,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    if role == 'summary':
        stmt = select(ex.func.count(_f('sg_qc_summary')))
    else:
        stmt = select(ex.func.count(_f('id'))).group_by(_f('sg_qc_summary'))

    with pytest.raises(CompilationError, match='UI-only'):
        compile_statement(stmt, schemas)


def test_ungrouped_summary_uses_none_grouping_and_is_deterministic(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(ex.func.sum(_f('sg_cut_duration')))

    first = compile_statement(stmt, schemas)
    second = compile_statement(stmt, schemas)

    assert first == second
    assert isinstance(first, SummarizeCall)
    assert first.grouping is None
    assert first.summary_fields == [{'field': 'sg_cut_duration', 'type': 'sum'}]
