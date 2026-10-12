from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeAlias, cast

import pytest
from _support import mapped_entity

from sgalchemy._core import _expressions as ex
from sgalchemy._core._compiler import FindCall, compile_statement
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._options import (
    Defer,
    JoinedLoad,
    LazyLoad,
    LoadOnly,
    SelectinLoad,
    defer,
    joinedload,
    lazyload,
    load_only,
    selectinload,
)
from sgalchemy._core._registry import Entity, Registry
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import CompilationError
from sgalchemy.schema import _model

_LoaderFactory: TypeAlias = Callable[[FieldExpr], JoinedLoad | LazyLoad | SelectinLoad]


@mapped_entity('Shot')
class _FakeShotClass(Entity):
    pass


@mapped_entity('Version')
class _FakeVersionClass(Entity):
    pass


def _compile_find(
    stmt: Select[Entity | None],
    schemas: dict[str, _model.EntitySchema] | None = None,
) -> FindCall:
    call = compile_statement(stmt, schemas)
    assert isinstance(call, FindCall)
    return call


def _f(
    code: str,
    schemas: dict[str, _model.EntitySchema] | None = None,
) -> ex.FieldExpr:
    schema = schemas['Shot'].fields[code] if schemas else None
    return ex.FieldExpr('Shot', (code,), schema)


def test_boolean_nesting_and_raw(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(_FakeShotClass).where(
        ex.or_(_f('code') == 'A', ex.and_(_f('code') == 'B', _f('sg_cut_duration') > 3)),
        ex.raw([['project', 'is', {'type': 'Project', 'id': 70}]]),
    )

    call = _compile_find(stmt, schemas)

    assert call.filters == [
        {
            'filter_operator': 'any',
            'filters': [
                ['code', 'is', 'A'],
                {
                    'filter_operator': 'all',
                    'filters': [
                        ['code', 'is', 'B'],
                        ['sg_cut_duration', 'greater_than', 3],
                    ],
                },
            ],
        },
        ['project', 'is', {'type': 'Project', 'id': 70}],
    ]


def test_column_path_must_end_on_field(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Shot', ('sg_sequence', 'Sequence'), None)

    with pytest.raises(CompilationError, match='must end on a field'):
        compile_statement(select(malformed), schemas)


def test_column_targets_deduplicate_in_first_seen_order(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    call = _compile_find(
        select(_f('code'), _f('id'), _f('code'), _f('description'), _f('id')),
        schemas,
    )

    assert call.fields == ['code', 'id', 'description']


@pytest.mark.parametrize(
    ('option', 'message'),
    [
        (load_only(_f('code')), r'load_only.*redundant'),
        (defer(_f('code')), r'defer.*redundant'),
        (joinedload(_f('sg_sequence')), r'relationship loaders.*entity selection'),
        (selectinload(_f('assets')), r'relationship loaders.*entity selection'),
    ],
)
def test_column_targets_reject_loader_options(
    option: Defer | JoinedLoad | LoadOnly | SelectinLoad,
    message: str,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=message):
        compile_statement(select(_f('code')).options(option), schemas)


def test_column_targets_render_paths_in_given_order(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    deep = ex.FieldExpr('Shot', ('sg_sequence', 'Sequence', 'code'), None)

    call = _compile_find(select(_f('code'), deep, _f('id')), schemas)

    assert call.entity_type == 'Shot'
    assert call.fields == ['code', 'sg_sequence.Sequence.code', 'id']


def test_cross_entity_expressions_rejected(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    asset_code = ex.FieldExpr('Asset', ('code',), None)

    with pytest.raises(CompilationError, match='cannot be attached'):
        compile_statement(select(_FakeShotClass).where(asset_code == 'x'), schemas)
    with pytest.raises(CompilationError, match='cannot be attached'):
        compile_statement(select(_FakeShotClass).order_by(asset_code.desc()), schemas)
    with pytest.raises(CompilationError, match='cannot be loaded'):
        compile_statement(select(_FakeShotClass).options(load_only(asset_code)), schemas)
    with pytest.raises(CompilationError, match='cannot be loaded'):
        compile_statement(select(_FakeShotClass).options(defer(asset_code)), schemas)


def test_defer_id_rejected(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=r'identity.*mandatory'):
        compile_statement(select(_FakeShotClass).options(defer(_f('id'))), schemas)


def test_defer_removes_terminal_joinedload_projection(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    deep_code = ex.FieldExpr(
        'Shot',
        ('sg_sequence', 'Sequence', 'code'),
        schemas['Sequence'].fields['code'],
    )
    call = _compile_find(
        select(_FakeShotClass).options(
            load_only(_f('code', schemas)),
            joinedload(_f('sg_sequence', schemas)),
            defer(deep_code),
        ),
        schemas,
    )

    assert call.fields[:3] == ['id', 'code', 'sg_sequence']
    assert 'sg_sequence.Sequence.code' not in call.fields
    assert 'sg_sequence.Sequence.id' in call.fields


def test_defer_retains_required_intermediate_joinedload_projection(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    episode = ex.FieldExpr(
        'Shot',
        ('sg_sequence', 'Sequence', 'episode'),
        schemas['Sequence'].fields['episode'],
    )
    option = joinedload(_f('sg_sequence', schemas)).joinedload(
        ex.FieldExpr('Sequence', ('episode',), schemas['Sequence'].fields['episode'])
    )

    call = _compile_find(
        select(_FakeShotClass).options(
            load_only(_f('code', schemas)),
            option,
            defer(episode),
        ),
        schemas,
    )

    assert 'sg_sequence.Sequence.episode' in call.fields
    assert 'sg_sequence.Sequence.episode.Episode.code' in call.fields


def test_direct_statement_rejects_entity_and_columns(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = Select[Entity](entity=_FakeShotClass, columns=(_f('code'),))

    with pytest.raises(CompilationError, match=r'both.*entity.*columns'):
        compile_statement(malformed, schemas)


def test_empty_statement_rejects_missing_target(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='at least one target'):
        compile_statement(Select[None](), schemas)


def test_entity_select_defaults_to_readable_fields(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(_FakeShotClass).where(_f('sg_status_list') == 'ip')

    call = compile_statement(stmt, schemas)

    assert isinstance(call, FindCall)
    assert call.entity_type == 'Shot'
    assert call.filters == [['sg_status_list', 'is', 'ip']]
    assert call.fields == schemas['Shot'].readable_field_codes()


@pytest.mark.parametrize(
    'statement',
    [
        select(_FakeShotClass).where(_f('nope') == 1),
        select(_FakeShotClass).order_by(_f('nope').desc()),
        select(_FakeShotClass).options(load_only(_f('nope'))),
        select(_FakeShotClass).options(defer(_f('nope'))),
        select(_f('nope')),
    ],
)
def test_every_find_field_position_rejects_unknown_field(
    statement: Select[Entity | None],
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='no field'):
        compile_statement(statement, schemas)


def test_factory_rejects_mixed_entity_and_column_targets(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=r'cannot mix.*entity.*field'):
        compile_statement(cast('Any', select)(_FakeShotClass, _f('code')), schemas)


def test_filter_path_rejects_empty_token(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Shot', ('sg_sequence', '', 'code'), None)

    with pytest.raises(CompilationError, match=r'non-empty.*tokens'):
        compile_statement(select(_FakeShotClass).where(malformed == 'x'), schemas)


@pytest.mark.parametrize(
    'options',
    [
        {'retired_only': True},
        {'include_archived_projects': False},
        {'additional_filter_presets': [{'preset_name': 'LATEST'}]},
    ],
)
def test_find_execution_option_matrix_is_preserved(
    options: dict[str, object],
    schemas: dict[str, _model.EntitySchema],
) -> None:
    call = _compile_find(select(_FakeShotClass).execution_options(**options), schemas)

    assert call.extra == options


def test_invalid_traversals_raise(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='cannot be traversed'):
        compile_statement(
            select(_FakeShotClass).where(
                ex.FieldExpr('Shot', ('code', 'Asset', 'code'), None) == 'x'
            ),
            schemas,
        )
    with pytest.raises(CompilationError, match='links to'):
        compile_statement(
            select(_FakeShotClass).where(
                ex.FieldExpr('Shot', ('sg_sequence', 'Asset', 'code'), None) == 'x'
            ),
            schemas,
        )
    with pytest.raises(CompilationError, match='multi-entity'):
        compile_statement(
            select(_FakeShotClass).where(
                ex.FieldExpr('Shot', ('assets', 'Asset', 'code'), None) == 'x'
            ),
            schemas,
        )


def test_joinedload_expands_every_hop_and_loads_base_link(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = select(_FakeShotClass).options(
        load_only(_f('code', schemas)),
        joinedload(_f('sg_sequence', schemas)).joinedload(
            ex.FieldExpr('Sequence', ('episode',), schemas['Sequence'].fields['episode'])
        ),
    )

    call = _compile_find(stmt, schemas)

    assert call.fields[:3] == ['id', 'code', 'sg_sequence']
    assert 'sg_sequence.Sequence.code' in call.fields
    assert 'sg_sequence.Sequence.episode.Episode.code' in call.fields


def test_joinedload_missing_single_target_signals_actionable_schema_drift(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stale = {name: schema for name, schema in schemas.items() if name != 'Sequence'}

    with pytest.raises(SchemaDrift) as excinfo:
        compile_statement(
            select(_FakeShotClass).options(joinedload(_f('sg_sequence', stale))),
            stale,
        )

    error = excinfo.value
    assert (error.entity_type, error.field_code, error.target_type) == (
        'Shot',
        'sg_sequence',
        'Sequence',
    )
    assert not hasattr(error, 'binding')


def test_joinedload_rejects_empty_chain(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=r'at least one.*field'):
        compile_statement(select(_FakeShotClass).options(JoinedLoad(())), schemas)


@pytest.mark.parametrize(
    'path',
    [(), ('sg_sequence', 'Sequence', 'code')],
)
def test_joinedload_rejects_malformed_hop(
    path: tuple[str, ...],
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Shot', path, None)

    with pytest.raises(CompilationError, match='one non-empty field-code token'):
        compile_statement(select(_FakeShotClass).options(joinedload(malformed)), schemas)


def test_joinedload_rejects_malformed_later_hop(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Sequence', (), None)
    option = joinedload(_f('sg_sequence', schemas)).joinedload(malformed)

    with pytest.raises(CompilationError, match='one non-empty field-code token'):
        compile_statement(select(_FakeShotClass).options(option), schemas)


def test_joinedload_rejects_multi_entity_link(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='selectinload'):
        compile_statement(
            select(_FakeShotClass).options(joinedload(_f('assets', schemas))), schemas
        )


def test_joinedload_rejects_multi_type_link(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    field = ex.FieldExpr('Version', ('entity',), schemas['Version'].fields['entity'])

    with pytest.raises(CompilationError, match='exactly one target type'):
        compile_statement(select(_FakeVersionClass).options(joinedload(field)), schemas)


def test_joinedload_rejects_ui_only_later_hop(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    sequence = schemas['Sequence']
    ui_link = replace(sequence.fields['episode'], api_readable=False)
    copied = {
        **schemas,
        'Sequence': replace(
            sequence,
            fields={**sequence.fields, 'episode': ui_link},
        ),
    }
    option = joinedload(_f('sg_sequence', copied)).joinedload(
        ex.FieldExpr('Sequence', ('episode',), ui_link)
    )

    with pytest.raises(CompilationError, match=r'Sequence\.episode.*UI-only'):
        compile_statement(select(_FakeShotClass).options(option), copied)


def test_joinedload_rejects_wrong_owner_on_later_hop(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    wrong_owner = ex.FieldExpr('Shot', ('episode',), schemas['Sequence'].fields['episode'])
    option = joinedload(_f('sg_sequence', schemas)).joinedload(wrong_owner)

    with pytest.raises(CompilationError, match=r'hop.*Shot.*Sequence'):
        compile_statement(select(_FakeShotClass).options(option), schemas)


def test_joinedload_requires_schema() -> None:
    with pytest.raises(CompilationError, match='requires a schema'):
        compile_statement(select(_FakeShotClass).options(joinedload(_f('sg_sequence'))))


def test_joinedload_shared_first_hop_preserves_distinct_chains(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    root = joinedload(_f('sg_sequence', schemas))
    deep = root.joinedload(
        ex.FieldExpr('Sequence', ('episode',), schemas['Sequence'].fields['episode'])
    )

    call = _compile_find(select(_FakeShotClass).options(root, deep, deep), schemas)

    assert 'sg_sequence.Sequence.code' in call.fields
    assert 'sg_sequence.Sequence.episode.Episode.code' in call.fields
    assert len(call.fields) == len(set(call.fields))


def test_load_only_and_defer_conflict_rejected(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=r'conflicting loader strategies.*code'):
        compile_statement(
            select(_FakeShotClass).options(load_only(_f('code')), defer(_f('code'))),
            schemas,
        )


def test_load_only_deduplicates_fields_and_preserves_id(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    call = _compile_find(
        select(_FakeShotClass).options(
            load_only(_f('code'), _f('id'), _f('code')),
            load_only(_f('description'), _f('id')),
        ),
        schemas,
    )

    assert call.fields == ['id', 'code', 'description']


def test_load_only_deep_field_projects_base_link(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    deep = ex.FieldExpr('Shot', ('sg_sequence', 'Sequence', 'code'), None)
    call = _compile_find(select(_FakeShotClass).options(load_only(deep)), schemas)

    assert call.fields == ['id', 'sg_sequence.Sequence.code', 'sg_sequence']


def test_load_only_defer_and_order(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    stmt = (
        select(_FakeShotClass)
        .options(load_only(_f('code', schemas)))
        .order_by(_f('code', schemas).desc())
        .limit(5)
        .page(2)
    )

    call = _compile_find(stmt, schemas)

    assert call.fields == ['id', 'code']
    assert call.order == [{'field_name': 'code', 'direction': 'desc'}]
    assert (call.limit, call.page) == (5, 2)
    deferred = _compile_find(select(_FakeShotClass).options(defer(_f('image'))), schemas)
    assert 'image' not in deferred.fields


@pytest.mark.parametrize('loader', [joinedload, selectinload])
def test_loader_options_reject_foreign_root(
    loader: _LoaderFactory,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    foreign = ex.FieldExpr('Asset', ('sg_vendor',), None)

    with pytest.raises(CompilationError, match='cannot be attached'):
        compile_statement(select(_FakeShotClass).options(loader(foreign)), schemas)


def test_loader_path_rejects_empty_token(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Shot', ('',), None)

    with pytest.raises(CompilationError, match=r'non-empty.*tokens'):
        compile_statement(select(_FakeShotClass).options(load_only(malformed)), schemas)


def test_mixed_entity_columns_raise(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='span entity types'):
        compile_statement(
            select(_f('code', schemas), ex.FieldExpr('Asset', ('code',), None)), schemas
        )


def test_opaque_root_query_teaches_schema_refresh(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    opaque_cls = Registry().entity_class_for('FutureVendor', binding=None)
    with pytest.raises(CompilationError, match=r'FutureVendor.*schema.*refresh'):
        compile_statement(select(opaque_cls), schemas)


def test_ordering_path_must_end_on_field(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Shot', ('sg_sequence', 'Sequence'), None)

    with pytest.raises(CompilationError, match='must end on a field'):
        compile_statement(select(_FakeShotClass).order_by(malformed.asc()), schemas)


def test_prebind_traversal_requires_rebuilt_expression(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    registry = Registry()
    shot = registry.entity_class('Shot')
    preserved = shot.sg_sequence.code
    registry.bind(schemas, key='https://example.invalid')

    with pytest.raises(
        CompilationError,
        match=(r'pre-bind traversal.*rebuild.*engine\.load_schema\(\).*or connect\(\)'),
    ):
        compile_statement(select(shot).where(preserved == 'SEQ01'), schemas)

    call = _compile_find(select(shot).where(shot.sg_sequence.code == 'SEQ01'), schemas)

    assert call.filters == [['sg_sequence.Sequence.code', 'is', 'SEQ01']]


@pytest.mark.parametrize(
    ('first', 'second'),
    [
        (joinedload, lazyload),
        (joinedload, selectinload),
        (selectinload, lazyload),
    ],
)
def test_relationship_loader_strategy_conflicts_are_rejected(
    first: _LoaderFactory,
    second: _LoaderFactory,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    field = _f('sg_sequence', schemas)

    with pytest.raises(
        CompilationError,
        match=r'conflicting relationship loader strategies.*Shot\.sg_sequence',
    ):
        compile_statement(
            select(_FakeShotClass).options(first(field), second(field)),
            schemas,
        )


@pytest.mark.parametrize('loader', [joinedload, selectinload])
def test_relationship_loaders_reject_ui_only_link(
    loader: _LoaderFactory,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    shot = schemas['Shot']
    ui_link = replace(
        shot.fields['sg_qc_summary'],
        data_type='entity',
        valid_types=('Sequence',),
    )
    schemas['Shot'] = replace(
        shot,
        fields={**shot.fields, 'sg_qc_summary': ui_link},
    )
    field = ex.FieldExpr('Shot', ('sg_qc_summary',), ui_link)

    with pytest.raises(CompilationError, match='UI-only'):
        compile_statement(select(_FakeShotClass).options(loader(field)), schemas)


def test_repeated_joinedload_deduplicates_projections(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    option = joinedload(_f('sg_sequence', schemas))

    call = _compile_find(
        select(_FakeShotClass).options(option, option),
        schemas,
    )

    assert len(call.fields) == len(set(call.fields))


def test_selectinload_adds_link_once(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    call = _compile_find(
        select(_FakeShotClass).options(
            load_only(_f('assets', schemas)), selectinload(_f('assets', schemas))
        ),
        schemas,
    )

    assert call.fields == ['id', 'assets']


def test_selectinload_rejects_malformed_field(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    malformed = ex.FieldExpr('Shot', (), None)

    with pytest.raises(CompilationError, match='one non-empty field-code token'):
        compile_statement(select(_FakeShotClass).options(SelectinLoad(malformed)), schemas)


def test_selectinload_requires_entity_link(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='not an entity link'):
        compile_statement(
            select(_FakeShotClass).options(selectinload(_f('code', schemas))), schemas
        )


def test_selectinload_requires_schema() -> None:
    with pytest.raises(CompilationError, match=r'requires a schema.*relationship'):
        compile_statement(select(_FakeShotClass).options(selectinload(_f('assets'))))


def test_symbolic_alias_class_teaches_canonical(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    @mapped_entity('Vendor')
    class _FakeVendorClass(Entity):
        pass

    with pytest.raises(CompilationError, match='CustomEntity03'):
        compile_statement(select(_FakeVendorClass), schemas)


def test_symbolic_entity_select_loads_only_id() -> None:
    call = _compile_find(select(_FakeShotClass))

    assert call.fields == ['id']


@pytest.mark.parametrize(
    ('method', 'expected'),
    [
        ('in_last', ['sg_due_date', 'in_last', 2, 'DAY']),
        ('in_next', ['sg_due_date', 'in_next', 2, 'DAY']),
        ('not_in_last', ['sg_due_date', 'not_in_last', 2, 'DAY']),
        ('not_in_next', ['sg_due_date', 'not_in_next', 2, 'DAY']),
    ],
)
def test_temporal_filters_compile_to_flat_wire_arguments(
    method: str,
    expected: list[Any],
    schemas: dict[str, _model.EntitySchema],
) -> None:
    condition = getattr(_f('sg_due_date'), method)(2, 'DAY')

    call = _compile_find(select(_FakeShotClass).where(condition), schemas)

    assert call.filters == [expected]


def test_ui_only_field_rejected(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match='UI-only'):
        compile_statement(select(_FakeShotClass).where(_f('sg_qc_summary') == 'x'), schemas)


def test_unknown_entity_type_rejected(
    schemas: dict[str, _model.EntitySchema],
) -> None:
    @mapped_entity('Missing')
    class _MissingClass(Entity):
        pass

    with pytest.raises(CompilationError, match="unknown entity type 'Missing'"):
        compile_statement(select(_MissingClass), schemas)


@pytest.mark.parametrize('key', ['unsupported', 'include_archived_projects_typo'])
def test_unknown_execution_option_rejected(
    key: str,
    schemas: dict[str, _model.EntitySchema],
) -> None:
    with pytest.raises(CompilationError, match=key):
        compile_statement(select(_FakeShotClass).execution_options(**{key: True}), schemas)
