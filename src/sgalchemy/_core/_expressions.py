"""Build field expressions, filters, and groupings."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Literal

from sgalchemy._errors import CompilationError
from sgalchemy.schema._model import EntitySchema, FieldSchema

# Server-supported grouping modes (summarize() docs); validity per mode depends on the
# grouped field's data type - the server is the authority, we only reject unknown names.
_GROUPING_MODES = frozenset(
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

_SUMMARY_WIRE = {
    'count': 'count',
    'sum': 'sum',
    'avg': 'average',
    'min': 'minimum',
    'max': 'maximum',
    'earliest': 'earliest',
    'latest': 'latest',
    'percentage': 'percentage',
    'status_percentage': 'status_percentage',
    'status_list': 'status_list',
    'checked': 'checked',
    'unchecked': 'unchecked',
    'record_count': 'record_count',
}


@dataclass(frozen=True)
class Filter:
    """Carry one provider filter condition and optional root ownership.

    Attributes:
        path: Rendered wire field path.
        op: Provider filter operator.
        value: First operator argument, which may itself be a collection.
        extra_values: Additional operator arguments emitted after `value`.
        entity_type: Root entity stamp used for ownership validation, or `None`
            for an unstamped hand-built filter. Excluded from equality.
    """

    path: str
    op: str
    value: object
    extra_values: tuple[object, ...] = ()
    entity_type: str | None = dc_field(default=None, compare=False)


@dataclass(frozen=True)
class FilterGroup:
    """Nested any/all group compiling to a `filter_operator` block."""

    op: Literal['any', 'all']
    items: tuple['Filter | FilterGroup | RawFilters', ...]


@dataclass(frozen=True)
class Grouping:
    """Carry a provider grouping descriptor and optional root ownership.

    Attributes:
        path: Rendered wire field path to group.
        mode: Provider grouping mode.
        entity_type: Root entity stamp used for ownership validation, or `None`
            for an unstamped grouping. Excluded from equality.
    """

    path: str
    mode: str
    entity_type: str | None = dc_field(default=None, compare=False)


@dataclass(frozen=True)
class Ordering:
    """Carry a provider sort descriptor and optional root ownership.

    Attributes:
        path: Rendered wire field path to sort.
        direction: Ascending or descending provider ordering.
        entity_type: Root entity stamp used for ownership validation, or `None`
            for an unstamped ordering. Excluded from equality.
    """

    path: str
    direction: Literal['asc', 'desc']
    entity_type: str | None = dc_field(default=None, compare=False)


class FieldExpr:
    """Class-level field reference for filters, ordering, grouping, and link traversal.

    Comparisons compile `==`, `!=`, `>`, and `<` to `is`, `is_not`,
    `greater_than`, and `less_than`. Inclusive `>=` and `<=` bounds use
    open-ended `between` filters. Private names raise `AttributeError`;
    schema-bound traversal raises `CompilationError` for non-single-entity
    links, ambiguous target types, and unavailable target entities or fields.
    """

    def __init__(
        self,
        entity_type: str,
        path: tuple[str, ...],
        schema: FieldSchema | None,
        *,
        schemas: Mapping[str, EntitySchema] | None = None,
        root_type: str | None = None,
    ) -> None:
        """Bind a rendered field path to its entity type and optional schema.

        Args:
            entity_type: API entity type at the current traversal hop.
            path: Wire path components from the root entity.
            schema: Schema for the current field, or `None` when unbound.
            schemas: Full site schema mapping for validated traversal.
            root_type: Root API entity type, defaulting to `entity_type`.
        """
        self._entity_type = entity_type
        self._root_type = root_type or entity_type
        self._path = path
        self._schema = schema
        self._schemas = schemas

    def __hash__(self) -> int:
        """Hash by entity type and path.

        Returns:
            Stable hash of the current entity type and path.
        """
        return hash((self._entity_type, self._path))

    @property
    def filter_field(self) -> str:
        """Return the dot-field path sent to the API.

        Returns:
            Dot-separated wire field path.
        """
        return '.'.join(self._path)

    def __getattr__(self, name: str) -> 'FieldExpr':
        """Traverse a relationship one hop deeper using wire dot syntax.

        Args:
            name: Target field code or symbolic path component.

        Returns:
            Field expression bound to the deeper path.

        Raises:
            AttributeError: If `name` is private.
            CompilationError: If bound schema does not permit the traversal.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        schema = self._schema
        if schema is not None:
            if schema.data_type == 'multi_entity':
                raise CompilationError(
                    f'{self.filter_field} is a multi-entity field; deep filters drill '
                    f'through single-entity links only (filter the target type directly, '
                    f'or selectinload() the relationship)'
                )
            if schema.data_type != 'entity':
                raise CompilationError(
                    f'{self.filter_field} is a {schema.data_type} field; only single-entity '
                    f'link fields support dot traversal; filter the scalar field directly'
                )
            if len(schema.valid_types) != 1:
                raise CompilationError(
                    f'{self.filter_field} links to multiple types '
                    f'{schema.valid_types}; filter a concrete target type directly or use '
                    f'selectinload()'
                )
            target = schema.valid_types[0]
            next_schema: FieldSchema | None = None
            if self._schemas is not None:
                target_entity = self._schemas.get(target)
                if target_entity is None:
                    raise CompilationError(
                        f'unknown entity type {target!r} behind {self.filter_field}; call '
                        f'engine.load_schema(force=True) to refresh schema state'
                    )
                next_schema = target_entity.fields.get(name)
                if next_schema is None:
                    raise CompilationError(
                        f'{target} has no field {name!r} (behind {self.filter_field}); call '
                        f'engine.load_schema(force=True), then use a valid field code if it '
                        f'remains absent'
                    )
            return FieldExpr(
                target,
                (*self._path, target, name),
                next_schema,
                schemas=self._schemas,
                root_type=self._root_type,
            )
        return FieldExpr(self._entity_type, (*self._path, name), None, root_type=self._root_type)

    def _filter(self, op: str, value: object, *extra_values: object) -> Filter:
        """Build a filter stamped with the root entity type."""
        return Filter(
            self.filter_field,
            op,
            value,
            tuple(extra_values),
            entity_type=self._root_type,
        )

    def __eq__(self, other: object) -> Filter:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Compile to `is`.

        Args:
            other: Required field value.

        Returns:
            Filter using the `is` wire operator.
        """
        return self._filter('is', other)

    def __ge__(self, other: object) -> Filter:
        """Compile to open-ended `between`.

        Args:
            other: Inclusive lower bound.

        Returns:
            Filter using an open-ended `between` wire operator.
        """
        return self._filter('between', [other, None])  # LIVE-PIN: bound inclusivity

    def __gt__(self, other: object) -> Filter:
        """Compile to `greater_than`.

        Args:
            other: Exclusive lower bound.

        Returns:
            Filter using the `greater_than` wire operator.
        """
        return self._filter('greater_than', other)

    def __le__(self, other: object) -> Filter:
        """Compile to open-ended `between`.

        Args:
            other: Inclusive upper bound.

        Returns:
            Filter using an open-ended `between` wire operator.
        """
        return self._filter('between', [None, other])  # LIVE-PIN: bound inclusivity

    def __lt__(self, other: object) -> Filter:
        """Compile to `less_than`.

        Args:
            other: Exclusive upper bound.

        Returns:
            Filter using the `less_than` wire operator.
        """
        return self._filter('less_than', other)

    def __ne__(self, other: object) -> Filter:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Compile to `is_not`.

        Args:
            other: Rejected field value.

        Returns:
            Filter using the `is_not` wire operator.
        """
        return self._filter('is_not', other)

    def asc(self) -> Ordering:
        """Build an ascending order item.

        Returns:
            Ascending ordering for this field.
        """
        return Ordering(self.filter_field, 'asc', entity_type=self._root_type)

    def between(self, lo: object, hi: object) -> Filter:
        """Compile to `between`.

        Args:
            lo: Inclusive lower bound.
            hi: Inclusive upper bound.

        Returns:
            Filter using the `between` wire operator.
        """
        return self._filter('between', [lo, hi])

    def by_(self, mode: str) -> Grouping:
        """Group by a server-supported mode.

        Args:
            mode: Server-supported grouping mode.

        Returns:
            Grouping for this field and mode.

        Raises:
            CompilationError: If `mode` is not server-supported.
        """
        if mode not in _GROUPING_MODES:
            raise CompilationError(
                f'unknown grouping mode {mode!r}; server-supported modes: '
                f'{sorted(_GROUPING_MODES)}'
            )
        return Grouping(self.filter_field, mode, entity_type=self._root_type)

    def by_day(self) -> Grouping:
        """Group by calendar day.

        Returns:
            Calendar-day grouping for this field.
        """
        return Grouping(self.filter_field, 'day', entity_type=self._root_type)

    def by_first_letter(self) -> Grouping:
        """Group by first letter.

        Returns:
            First-letter grouping for this field.
        """
        return Grouping(self.filter_field, 'firstletter', entity_type=self._root_type)

    def by_month(self) -> Grouping:
        """Group by calendar month.

        Returns:
            Calendar-month grouping for this field.
        """
        return Grouping(self.filter_field, 'month', entity_type=self._root_type)

    def by_week(self) -> Grouping:
        """Group by calendar week.

        Returns:
            Calendar-week grouping for this field.
        """
        return Grouping(self.filter_field, 'week', entity_type=self._root_type)

    def contains(self, value: object) -> Filter:
        """Compile to `contains`.

        Args:
            value: Value that the field must contain.

        Returns:
            Filter using the `contains` wire operator.
        """
        return self._filter('contains', value)

    def desc(self) -> Ordering:
        """Build a descending order item.

        Returns:
            Descending ordering for this field.
        """
        return Ordering(self.filter_field, 'desc', entity_type=self._root_type)

    def endswith(self, value: str) -> Filter:
        """Compile to `ends_with`.

        Args:
            value: Required field suffix.

        Returns:
            Filter using the `ends_with` wire operator.
        """
        return self._filter('ends_with', value)

    def in_(self, *values: object) -> Filter:
        """Compile to `in`.

        Args:
            *values: Accepted field values.

        Returns:
            Filter using the `in` wire operator.
        """
        return self._filter('in', list(values))

    def in_calendar_day(self, offset: int) -> Filter:
        """Compile to `in_calendar_day`.

        Args:
            offset: Calendar-day offset from today.

        Returns:
            Filter using the `in_calendar_day` wire operator.
        """
        return self._filter('in_calendar_day', offset)

    def in_calendar_month(self, offset: int) -> Filter:
        """Compile to `in_calendar_month`.

        Args:
            offset: Calendar-month offset from the current month.

        Returns:
            Filter using the `in_calendar_month` wire operator.
        """
        return self._filter('in_calendar_month', offset)

    def in_calendar_week(self, offset: int) -> Filter:
        """Compile to `in_calendar_week`.

        Args:
            offset: Calendar-week offset from the current week.

        Returns:
            Filter using the `in_calendar_week` wire operator.
        """
        return self._filter('in_calendar_week', offset)

    def in_calendar_year(self, offset: int) -> Filter:
        """Compile to `in_calendar_year`.

        Args:
            offset: Calendar-year offset from the current year.

        Returns:
            Filter using the `in_calendar_year` wire operator.
        """
        return self._filter('in_calendar_year', offset)

    def in_last(self, count: int, unit: str) -> Filter:
        """Compile to `in_last`.

        Args:
            count: Number of time units.
            unit: Server time unit name.

        Returns:
            Filter using the `in_last` wire operator.
        """
        return self._filter('in_last', count, unit)

    def in_next(self, count: int, unit: str) -> Filter:
        """Compile to `in_next`.

        Args:
            count: Number of time units.
            unit: Server time unit name.

        Returns:
            Filter using the `in_next` wire operator.
        """
        return self._filter('in_next', count, unit)

    def is_(self, value: object) -> Filter:
        """Compile to `is`.

        Args:
            value: Required field value.

        Returns:
            Filter using the `is` wire operator.
        """
        return self._filter('is', value)

    def name_contains(self, value: str) -> Filter:
        """Compile to `name_contains`.

        Args:
            value: Value the linked entity name must contain.

        Returns:
            Filter using the `name_contains` wire operator.
        """
        return self._filter('name_contains', value)

    def name_ends_with(self, value: str) -> Filter:
        """Compile to `name_ends_with`.

        Args:
            value: Required linked entity name suffix.

        Returns:
            Filter using the `name_ends_with` wire operator.
        """
        return self._filter('name_ends_with', value)

    def name_id(self, value: object) -> Filter:
        """Compile to `name_id`.

        Args:
            value: Linked entity identifier.

        Returns:
            Filter using the `name_id` wire operator.
        """
        return self._filter('name_id', value)

    def name_is(self, value: str) -> Filter:
        """Compile to `name_is`.

        Args:
            value: Required linked entity name.

        Returns:
            Filter using the `name_is` wire operator.
        """
        return self._filter('name_is', value)

    def name_not_contains(self, value: str) -> Filter:
        """Compile to `name_not_contains`.

        Args:
            value: Value the linked entity name must not contain.

        Returns:
            Filter using the `name_not_contains` wire operator.
        """
        return self._filter('name_not_contains', value)

    def name_starts_with(self, value: str) -> Filter:
        """Compile to `name_starts_with`.

        Args:
            value: Required linked entity name prefix.

        Returns:
            Filter using the `name_starts_with` wire operator.
        """
        return self._filter('name_starts_with', value)

    def not_between(self, lo: object, hi: object) -> Filter:
        """Compile to `not_between`.

        Args:
            lo: Excluded lower bound.
            hi: Excluded upper bound.

        Returns:
            Filter using the `not_between` wire operator.
        """
        return self._filter('not_between', [lo, hi])

    def not_contains(self, value: object) -> Filter:
        """Compile to `not_contains`.

        Args:
            value: Value that the field must not contain.

        Returns:
            Filter using the `not_contains` wire operator.
        """
        return self._filter('not_contains', value)

    def not_in(self, *values: object) -> Filter:
        """Compile to `not_in`.

        Args:
            *values: Rejected field values.

        Returns:
            Filter using the `not_in` wire operator.
        """
        return self._filter('not_in', list(values))

    def not_in_last(self, count: int, unit: str) -> Filter:
        """Compile to `not_in_last`.

        Args:
            count: Number of time units.
            unit: Server time unit name.

        Returns:
            Filter using the `not_in_last` wire operator.
        """
        return self._filter('not_in_last', count, unit)

    def not_in_next(self, count: int, unit: str) -> Filter:
        """Compile to `not_in_next`.

        Args:
            count: Number of time units.
            unit: Server time unit name.

        Returns:
            Filter using the `not_in_next` wire operator.
        """
        return self._filter('not_in_next', count, unit)

    @property
    def path(self) -> tuple[str, ...]:
        """Return path components for compiler and loader ownership checks.

        Returns:
            Immutable wire path components.
        """
        return self._path

    @property
    def root_type(self) -> str:
        """Return the root entity type that owns this expression.

        Returns:
            Root API entity type.
        """
        return self._root_type

    def startswith(self, value: str) -> Filter:
        """Compile to `starts_with`.

        Args:
            value: Required field prefix.

        Returns:
            Filter using the `starts_with` wire operator.
        """
        return self._filter('starts_with', value)

    def type_is(self, entity_type: str) -> Filter:
        """Compile to `type_is`.

        Args:
            entity_type: Required linked entity type.

        Returns:
            Filter using the `type_is` wire operator.
        """
        return self._filter('type_is', entity_type)

    def type_is_not(self, entity_type: str) -> Filter:
        """Compile to `type_is_not`.

        Args:
            entity_type: Rejected linked entity type.

        Returns:
            Filter using the `type_is_not` wire operator.
        """
        return self._filter('type_is_not', entity_type)


@dataclass(frozen=True)
class RawFilters:
    """Raw SG filter-list syntax passed through verbatim."""

    filters: list[Any]


@dataclass(frozen=True)
class Summary:
    """Describe one aggregate column for a provider summarize call.

    Attributes:
        path: Rendered wire field path to aggregate.
        wire_type: Provider summary-type spelling.
        entity_type: Root entity type that owns the aggregate.
    """

    path: str
    wire_type: str
    entity_type: str


class _Func:
    """Aggregate namespace containing only server-supported summary types."""

    count: Callable[[FieldExpr], Summary]
    sum: Callable[[FieldExpr], Summary]
    avg: Callable[[FieldExpr], Summary]
    min: Callable[[FieldExpr], Summary]
    max: Callable[[FieldExpr], Summary]
    earliest: Callable[[FieldExpr], Summary]
    latest: Callable[[FieldExpr], Summary]
    percentage: Callable[[FieldExpr], Summary]
    status_percentage: Callable[[FieldExpr], Summary]
    status_list: Callable[[FieldExpr], Summary]
    checked: Callable[[FieldExpr], Summary]
    unchecked: Callable[[FieldExpr], Summary]
    record_count: Callable[[FieldExpr], Summary]

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Callable[[FieldExpr], Summary]:
            """Return the factory for a server-supported summary type."""
            wire = _SUMMARY_WIRE.get(name)
            if wire is None:
                raise AttributeError(
                    f'func.{name} does not exist: the server supports only {sorted(_SUMMARY_WIRE)}'
                )

            def make(field: FieldExpr) -> Summary:
                return Summary(field.filter_field, wire, field.root_type)

            return make


def and_(*items: 'Filter | FilterGroup | RawFilters') -> FilterGroup:
    """Combine conditions so all must match.

    Args:
        *items: Filters, nested groups, or raw filter lists.

    Returns:
        All-items filter group.
    """
    return FilterGroup('all', tuple(items))


def or_(*items: 'Filter | FilterGroup | RawFilters') -> FilterGroup:
    """Combine conditions so any may match.

    Args:
        *items: Filters, nested groups, or raw filter lists.

    Returns:
        Any-item filter group.
    """
    return FilterGroup('any', tuple(items))


def raw(filters: list[Any]) -> RawFilters:
    """Pass raw SG filter-list syntax through unchanged.

    Args:
        filters: Raw filter list to retain verbatim.

    Returns:
        Wrapper holding the original list.
    """
    return RawFilters(filters)


func = _Func()
