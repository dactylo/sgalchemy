"""Compile statements into provider call arguments."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never, cast

from sgalchemy._core._entity import entity_registry as registered_entity_registry
from sgalchemy._core._entity import entity_type as registered_entity_type
from sgalchemy._core._expressions import (
    FieldExpr,
    Filter,
    FilterGroup,
    RawFilters,
    Summary,
)
from sgalchemy._core._options import (
    FIND_EXECUTION_OPTIONS,
    SUMMARIZE_EXECUTION_OPTIONS,
    Defer,
    JoinedLoad,
    LazyLoad,
    LoadOnly,
    SelectinLoad,
    normalize_relationship_loaders,
)
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._errors import CompilationError
from sgalchemy.schema._aliases import entity_aliases
from sgalchemy.schema._model import EntitySchema, FieldSchema

if TYPE_CHECKING:
    from sgalchemy._core._registry import Entity
    from sgalchemy._core._statement import Select


@dataclass(frozen=True)
class FindCall:
    """Carry the complete arguments for one provider `find()` call.

    Attributes:
        entity_type: Canonical API entity type.
        filters: Compiled provider filter list.
        fields: Ordered wire field paths to project.
        order: Provider field-name and direction descriptors.
        limit: Maximum returned records; zero requests no limit.
        page: One-based provider page; zero requests all pages.
        extra: Additional validated provider keyword options.
    """

    entity_type: str
    filters: list[Any]
    fields: list[str]
    order: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    limit: int = 0
    page: int = 0
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class SummarizeCall:
    """Carry the complete arguments for one provider `summarize()` call.

    Attributes:
        entity_type: Canonical API entity type.
        filters: Compiled provider filter list.
        summary_fields: Provider field and summary-type descriptors.
        grouping: Ordered grouping descriptors, or `None` to omit grouping.
        extra: Additional validated provider keyword options.
    """

    entity_type: str
    filters: list[Any]
    summary_fields: list[dict[str, str]]
    grouping: list[dict[str, str]] | None = None
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


def _check_exec_options(opts: Mapping[str, object], allowed: frozenset[str]) -> None:
    """Reject execution options unsupported by the statement kind."""
    unknown = sorted(set(opts) - allowed)
    if unknown:
        raise CompilationError(
            f'unsupported execution options {unknown}; supported here: {sorted(allowed)}'
        )


def _deduplicate_fields(fields: list[str]) -> list[str]:
    """Return fields in stable first-seen order without duplicates."""
    return list(dict.fromkeys(fields))


def _known_entity_type(
    entity_type: str,
    schemas: Mapping[str, EntitySchema] | None,
    *,
    opaque_class: bool = False,
) -> None:
    """Reject unknown types and teach the canonical name for display-name aliases."""
    if schemas is None or entity_type in schemas:
        return
    alias_map, _ = entity_aliases(schemas)
    canonical = alias_map.get(entity_type)
    if canonical is not None:
        raise CompilationError(
            f'{entity_type!r} is the display-name alias of {canonical!r}; aliases '
            f'resolve from a bound registry - import the class after engine creation, '
            f'or use the canonical name'
        )
    if opaque_class:
        raise CompilationError(
            f'entity type {entity_type!r} is absent from the captured schema; refresh '
            'schema with engine.load_schema(force=True) before querying this opaque class'
        )
    raise CompilationError(f'unknown entity type {entity_type!r}')


def _summary_entity_type(
    stmt: 'Select[Entity | None]', schemas: Mapping[str, EntitySchema] | None
) -> str:
    """Resolve summary columns to one canonical entity owner."""
    types = sorted({column.entity_type for column in stmt.columns if isinstance(column, Summary)})
    if len(types) != 1:
        raise CompilationError(
            f'summary columns span entity types {types}; one summarize() call covers '
            f'a single entity type'
        )
    entity_type = types[0]
    _known_entity_type(entity_type, schemas)
    return entity_type


def _validate_api_readable(
    entity_type: str,
    code: str,
    schema: FieldSchema | None,
    location: str,
) -> None:
    """Reject API-unreadable schema fields before compilation."""
    if schema is not None and not schema.api_readable:
        raise CompilationError(
            f'{entity_type}.{code} is a UI-only query/summary field; it returns no usable '
            f'data through the API (in {location})'
        )


def _validate_loader_field(field_expr: FieldExpr, loader_name: str) -> str:
    """Return one loader field code after validating its structural shape."""
    path = field_expr.path
    if len(path) != 1 or not path[0] or '.' in path[0]:
        raise CompilationError(
            f'{loader_name} requires each hop to be one non-empty field-code token; '
            f'all loader tokens must be non-empty; '
            f'got {path!r}'
        )
    return path[0]


def _deep_fields(
    opt: JoinedLoad,
    entity_type: str,
    schemas: Mapping[str, EntitySchema],
    defers: Collection[str],
) -> list[str]:
    """Expand a joinedload chain while honoring terminal deep defers."""
    out: list[str] = []
    prefix_parts: list[str] = []
    current = entity_type
    for hop_index, hop in enumerate(opt.paths):
        code = _validate_loader_field(hop, 'joinedload')
        if hop.root_type != current:
            raise CompilationError(
                f'joinedload hop {hop.root_type}.{code} cannot follow a link to {current}'
            )
        entity = schemas.get(current)
        if entity is None:
            raise CompilationError(f'unknown entity type {current!r} in joinedload')
        fschema = entity.fields.get(code)
        _validate_api_readable(current, code, fschema, 'joinedload')
        if fschema is None or fschema.data_type not in ('entity', 'multi_entity'):
            raise CompilationError(f'{current}.{code} is not an entity link field')
        if fschema.data_type == 'multi_entity':
            raise CompilationError(
                f'{current}.{code} is a multi-entity field; joinedload traverses to-one '
                f'links only (use selectinload)'
            )
        if len(fschema.valid_types) != 1:
            raise CompilationError(
                f'{current}.{code} links to {fschema.valid_types}; joinedload needs '
                f'exactly one target type (use selectinload)'
            )
        target = fschema.valid_types[0]
        target_entity = schemas.get(target)
        if target_entity is None:
            raise SchemaDrift(current, code, target)
        prefix_parts.extend((code, target))
        current = target
        prefix = '.'.join(prefix_parts)
        required_link = (
            _validate_loader_field(opt.paths[hop_index + 1], 'joinedload')
            if hop_index + 1 < len(opt.paths)
            else None
        )
        for sub in target_entity.readable_field_codes():
            projection = f'{prefix}.{sub}'
            if projection not in defers or sub == required_link:
                out.append(projection)
    return out


def _validate_path_grammar(path: str) -> list[str]:
    """Return path tokens after validating the wire dot-path grammar."""
    parts = path.split('.')
    if not path or any(not part for part in parts):
        raise CompilationError(f'field path {path!r} must contain non-empty dot-separated tokens')
    if len(parts) % 2 == 0:
        raise CompilationError(
            f'field path {path!r} must alternate field.TargetType.field and must end on a field; '
            'this may be a pre-bind traversal from before schema binding; rebuild the expression '
            'after engine.load_schema() or connect()'
        )
    return parts


def _validate_path(
    path: str,
    entity_type: str,
    schemas: Mapping[str, EntitySchema],
) -> None:
    """Walk a dot-field path, raising on unknown or invalid hops."""
    current = entity_type
    parts = _validate_path_grammar(path)
    index = 0
    while index < len(parts):
        entity = schemas.get(current)
        if entity is None:
            raise CompilationError(f'unknown entity type {current!r} in {path!r}')
        code = parts[index]
        fschema = entity.fields.get(code)
        if fschema is None:
            raise CompilationError(f'{current} has no field {code!r} (in {path!r})')
        _validate_api_readable(current, code, fschema, repr(path))
        if index + 1 < len(parts):
            if fschema.data_type == 'multi_entity':
                raise CompilationError(
                    f'{current}.{code} is a multi-entity field; deep filters drill through '
                    f'single-entity links only (in {path!r})'
                )
            if fschema.data_type != 'entity':
                raise CompilationError(
                    f'{current}.{code} is a {fschema.data_type} field, not an entity link; '
                    f'it cannot be traversed (in {path!r})'
                )
            target = parts[index + 1]
            if target not in fschema.valid_types:
                raise CompilationError(
                    f'{current}.{code} links to {list(fschema.valid_types)}, not {target!r} '
                    f'(in {path!r})'
                )
            current = target
            index += 2
        else:
            index += 1


def _apply_joinedload(
    opt: JoinedLoad,
    entity_type: str,
    fields: list[str],
    schemas: Mapping[str, EntitySchema] | None,
    defers: Collection[str],
) -> None:
    """Append one joined-load projection after validating its root."""
    if not opt.paths:
        raise CompilationError('joinedload requires at least one relationship field')
    _validate_loader_field(opt.paths[0], 'joinedload')
    root = opt.paths[0].root_type
    if root != entity_type:
        raise CompilationError(
            f'joinedload of {root}.{opt.paths[0].filter_field} cannot be attached '
            f'to a {entity_type} statement'
        )
    if schemas is None:
        raise CompilationError('joinedload requires a schema to expand deep fields')
    _validate_path(opt.paths[0].filter_field, entity_type, schemas)
    base = opt.paths[0].path[0]
    if base not in fields:
        fields.append(base)
    fields.extend(_deep_fields(opt, entity_type, schemas, defers))


def _column_target_and_fields(
    stmt: 'Select[Entity | None]',
    schemas: Mapping[str, EntitySchema] | None,
) -> tuple[str, list[str]]:
    """Resolve a column selection to one entity type and its wire paths."""
    for opt in stmt.loader_options:
        if isinstance(opt, LoadOnly):
            raise CompilationError(
                'load_only is redundant for a column selection; select exactly the fields needed'
            )
        if isinstance(opt, Defer):
            raise CompilationError(
                'defer is redundant for a column selection; omit unwanted columns from select()'
            )
        raise CompilationError(
            'relationship loaders require an entity selection, not a column selection'
        )
    if not stmt.columns:
        raise CompilationError('select() requires at least one target entity or field')
    if not all(isinstance(target, FieldExpr) for target in stmt.columns):
        raise CompilationError(
            'find select() cannot mix entity targets with field columns; choose one entity '
            'or only fields'
        )
    columns = cast('tuple[FieldExpr, ...]', stmt.columns)
    types = {column.root_type for column in columns}
    if len(types) != 1:
        raise CompilationError(
            f'select() columns span entity types {sorted(types)}; a statement '
            f'compiles to one find() over a single entity type'
        )
    entity_type = columns[0].root_type
    _known_entity_type(entity_type, schemas)
    paths = [column.filter_field for column in columns]
    for path in paths:
        _validate_path_grammar(path)
        if schemas is not None:
            _validate_path(path, entity_type, schemas)
    return entity_type, _deduplicate_fields(paths)


def _compile_conditions(
    conds: tuple[Filter | FilterGroup | RawFilters, ...],
    entity_type: str,
    schemas: Mapping[str, EntitySchema] | None,
) -> list[Any]:
    """Compile condition objects to the wire filter-list shape."""
    out: list[Any] = []
    for cond in conds:
        match cond:
            case Filter():
                if cond.entity_type is not None and cond.entity_type != entity_type:
                    raise CompilationError(
                        f'filter on {cond.entity_type}.{cond.path} cannot be attached to a '
                        f'{entity_type} statement'
                    )
                _validate_path_grammar(cond.path)
                if schemas is not None:
                    _validate_path(cond.path, entity_type, schemas)
                out.append([cond.path, cond.op, cond.value, *cond.extra_values])
            case RawFilters():
                out.extend(cond.filters)
            case FilterGroup():
                out.append(
                    {
                        'filter_operator': cond.op,
                        'filters': _compile_conditions(cond.items, entity_type, schemas),
                    }
                )
            case _:
                assert_never(cond)
    return out


def _validate_field_loaders(
    stmt: 'Select[Entity | None]',
    entity_type: str,
    schemas: Mapping[str, EntitySchema] | None,
) -> None:
    """Validate ownership and paths for load-only and defer options."""
    for opt in stmt.loader_options:
        if isinstance(opt, (LoadOnly, Defer)):
            for field_expr in opt.fields:
                if field_expr.root_type != entity_type:
                    raise CompilationError(
                        f'{field_expr.root_type}.{field_expr.filter_field} cannot be loaded '
                        f'on a {entity_type} statement'
                    )
                _validate_path_grammar(field_expr.filter_field)
                if schemas is not None:
                    _validate_path(field_expr.filter_field, entity_type, schemas)


def _validate_summary_target(stmt: 'Select[Entity | None]') -> None:
    """Reject entity targets that a summary call cannot represent."""
    if stmt.entity is not None:
        raise CompilationError(
            'a summary statement cannot contain an entity target; use select(func.*(...)) '
            'without an entity target'
        )


def _compile_summarize(
    stmt: 'Select[Entity | None]', schemas: Mapping[str, EntitySchema] | None
) -> SummarizeCall:
    """Compile one summary statement to exact `summarize()` call arguments."""
    _validate_summary_target(stmt)
    summaries = [column for column in stmt.columns if isinstance(column, Summary)]
    if len(summaries) != len(stmt.columns):
        raise CompilationError(
            'summary statements accept only func.* columns; mixing plain fields with '
            'aggregates has no server-side meaning'
        )
    if not summaries:
        raise CompilationError('group_by requires at least one func.* summary column')
    if stmt.orderings or stmt.limit_value or stmt.page_value or stmt.loader_options:
        raise CompilationError(
            'summary statements do not support order_by(), limit(), page(), or loader options'
        )
    entity_type = _summary_entity_type(stmt, schemas)
    _check_exec_options(stmt.exec_options, SUMMARIZE_EXECUTION_OPTIONS)
    filters = _compile_conditions(stmt.conditions, entity_type, schemas)
    for grouping in stmt.groupings:
        if grouping.entity_type is not None and grouping.entity_type != entity_type:
            raise CompilationError(
                f'group_by on {grouping.entity_type}.{grouping.path} cannot be attached '
                f'to a {entity_type} statement'
            )
    for summary in summaries:
        _validate_path_grammar(summary.path)
        if schemas is not None:
            _validate_path(summary.path, entity_type, schemas)
    for grouping in stmt.groupings:
        _validate_path_grammar(grouping.path)
        if schemas is not None:
            _validate_path(grouping.path, entity_type, schemas)
    grouping = [
        {'field': item.path, 'type': item.mode, 'direction': 'asc'} for item in stmt.groupings
    ] or None
    return SummarizeCall(
        entity_type=entity_type,
        filters=filters,
        summary_fields=[
            {'field': summary.path, 'type': summary.wire_type} for summary in summaries
        ],
        grouping=grouping,
        extra=dict(stmt.exec_options),
    )


def validate_relationship_field(
    field_expr: FieldExpr,
    loader_name: str,
    entity_type: str,
    schemas: Mapping[str, EntitySchema],
) -> str:
    """Return a readable direct relationship's validated field code.

    Args:
        field_expr: Single-token relationship expression to validate.
        loader_name: Loader name used in failure diagnostics.
        entity_type: Canonical entity type that must own the expression.
        schemas: Captured schema mapping used for validation.

    Returns:
        Direct provider field code.

    Raises:
        CompilationError: If the expression is not a readable direct entity or
            multi-entity link rooted at the expected type.
    """
    code = _validate_loader_field(field_expr, loader_name)
    if field_expr.root_type != entity_type:
        raise CompilationError(
            f'{loader_name} of {field_expr.root_type}.{code} cannot be attached '
            f'to a {entity_type} statement'
        )
    _validate_path(code, entity_type, schemas)
    fschema = schemas[entity_type].fields[code]
    if fschema.data_type not in ('entity', 'multi_entity'):
        raise CompilationError(
            f'{entity_type}.{code} is not an entity link; {loader_name} loads relationships '
            '(use load_only for plain fields)'
        )
    return code


def _apply_selectinload(
    opt: SelectinLoad | LazyLoad,
    entity_type: str,
    fields: list[str],
    schemas: Mapping[str, EntitySchema] | None,
) -> None:
    """Append one select-in-load link after validating its root and type."""
    loader_name = 'lazyload' if isinstance(opt, LazyLoad) else 'selectinload'
    if schemas is None:
        raise CompilationError(
            f'{loader_name} requires a schema to prove that the field is a relationship'
        )
    code = validate_relationship_field(opt.field, loader_name, entity_type, schemas)
    if code not in fields:
        fields.append(code)


def _apply_relationship_loaders(
    stmt: 'Select[Entity]',
    entity_type: str,
    fields: list[str],
    schemas: Mapping[str, EntitySchema] | None,
    defers: Collection[str],
) -> None:
    """Append relationship-loader projections in option order."""
    for opt in normalize_relationship_loaders(stmt.loader_options):
        if isinstance(opt, JoinedLoad):
            _apply_joinedload(opt, entity_type, fields, schemas, defers)
        else:
            _apply_selectinload(opt, entity_type, fields, schemas)


def _entity_target_and_fields(
    stmt: 'Select[Entity]',
    schemas: Mapping[str, EntitySchema] | None,
) -> tuple[str, list[str]]:
    """Resolve an entity selection and apply its loader projections."""
    target = cast('type[object]', stmt.entity)
    entity_type = registered_entity_type(target)
    _known_entity_type(
        entity_type,
        schemas,
        opaque_class=registered_entity_registry(target) is not None,
    )
    load_only = [opt for opt in stmt.loader_options if isinstance(opt, LoadOnly)]
    defers = {
        field.filter_field
        for opt in stmt.loader_options
        if isinstance(opt, Defer)
        for field in opt.fields
    }
    if 'id' in defers:
        raise CompilationError('defer(id) is invalid because entity identity is mandatory')
    loaded = {field.filter_field for opt in load_only for field in opt.fields}
    conflicts = sorted(loaded & defers)
    if conflicts:
        raise CompilationError(
            f'conflicting loader strategies name fields in both load_only and defer: {conflicts}'
        )
    _validate_field_loaders(stmt, entity_type, schemas)
    if load_only:
        fields = [
            'id',
            *(field_expr.filter_field for opt in load_only for field_expr in opt.fields),
        ]
        fields.extend(
            field_expr.path[0]
            for opt in load_only
            for field_expr in opt.fields
            if len(field_expr.path) > 1
        )
    elif schemas is not None:
        fields = schemas[entity_type].readable_field_codes()
    else:
        fields = ['id']
    fields = [field_code for field_code in fields if field_code not in defers]
    _apply_relationship_loaders(stmt, entity_type, fields, schemas, defers)
    return entity_type, _deduplicate_fields(fields)


def _target_and_fields(
    stmt: 'Select[Entity | None]',
    schemas: Mapping[str, EntitySchema] | None,
) -> tuple[str, list[str]]:
    """Resolve the target entity and deterministic find projection."""
    if stmt.entity is not None and stmt.columns:
        raise CompilationError(
            'a find statement cannot contain both an entity target and columns; choose one shape'
        )
    if stmt.entity is None:
        return _column_target_and_fields(stmt, schemas)
    return _entity_target_and_fields(cast('Select[Entity]', stmt), schemas)


def compile_statement(
    stmt: 'Select[Entity | None]',
    schemas: Mapping[str, EntitySchema] | None = None,
) -> FindCall | SummarizeCall:
    """Compile a statement to exact API-call arguments without I/O.

    Args:
        stmt: The statement to compile.
        schemas: Parsed site schema for validation and field expansion.
            `None` permits symbolic compilation but cannot expand relationship
            loaders.

    Returns:
        A find or summarize call value object.

    Raises:
        CompilationError: On unsupported options or untranslatable shapes.
        SchemaDrift: If a joined-load target is absent from the captured schema
            and the caller must decide whether to refresh and recompile.
    """
    is_summary = bool(stmt.groupings) or any(
        isinstance(column, Summary) for column in stmt.columns
    )
    if is_summary:
        return _compile_summarize(stmt, schemas)

    entity_type, fields = _target_and_fields(stmt, schemas)
    _check_exec_options(stmt.exec_options, FIND_EXECUTION_OPTIONS)
    filters = _compile_conditions(stmt.conditions, entity_type, schemas)
    for ordering in stmt.orderings:
        if ordering.entity_type is not None and ordering.entity_type != entity_type:
            raise CompilationError(
                f'order_by on {ordering.entity_type}.{ordering.path} cannot be attached '
                f'to a {entity_type} statement'
            )
        _validate_path_grammar(ordering.path)
        if schemas is not None:
            _validate_path(ordering.path, entity_type, schemas)
    order = [
        {'field_name': ordering.path, 'direction': ordering.direction}
        for ordering in stmt.orderings
    ]
    return FindCall(
        entity_type=entity_type,
        filters=filters,
        fields=fields,
        order=order,
        limit=stmt.limit_value,
        page=stmt.page_value,
        extra=dict(stmt.exec_options),
    )
