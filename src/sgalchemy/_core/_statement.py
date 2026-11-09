"""Build immutable entity, projection, and summary statements."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Self, TypeVar, cast, overload

from sgalchemy._core._compiler import FindCall, SummarizeCall, compile_statement
from sgalchemy._core._expressions import (
    FieldExpr,
    Filter,
    FilterGroup,
    Grouping,
    Ordering,
    RawFilters,
    Summary,
    require_conditions,
)
from sgalchemy._core._options import Defer, JoinedLoad, LazyLoad, LoadOnly, SelectinLoad
from sgalchemy._core._registry import Entity
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._errors import CompilationError
from sgalchemy.schema._model import EntitySchema

_EntityT = TypeVar('_EntityT', bound=Entity)
_LoaderOption = Defer | JoinedLoad | LazyLoad | LoadOnly | SelectinLoad
_SelectT_co = TypeVar('_SelectT_co', bound=Entity | None, covariant=True)


@dataclass(frozen=True, eq=False)
class Select(Generic[_SelectT_co]):
    """Describe a query whose builder methods return modified copies.

    Compilation validates the selected shape and options without I/O.
    Frozen attributes and read-only execution-option storage do not deeply
    freeze values held inside filters or other supplied objects.

    Attributes:
        entity: Reflected class for an entity selection, otherwise `None`.
        columns: Field projections or aggregate columns for a non-entity
            selection; plain fields and aggregates cannot be mixed.
        conditions: Filter conditions combined with AND.
        orderings: Provider sort descriptors in priority order.
        groupings: Ordered summarize grouping descriptors.
        limit_value: Provider result limit; zero requests no limit.
        page_value: One-based provider page; zero requests all pages.
        loader_options: Field and relationship loading instructions.
        exec_options: Read-only outer mapping of provider execution flags.
    """

    entity: type[_SelectT_co] | None = None
    columns: tuple[FieldExpr | Summary, ...] = ()
    conditions: tuple[Filter | FilterGroup | RawFilters, ...] = ()
    orderings: tuple[Ordering, ...] = ()
    groupings: tuple[Grouping, ...] = ()
    limit_value: int = 0
    page_value: int = 0
    loader_options: tuple[_LoaderOption, ...] = ()
    exec_options: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        """Freeze execution options into deterministic read-only storage."""
        object.__setattr__(self, 'exec_options', MappingProxyType(dict(self.exec_options)))

    def compile(
        self,
        schema: Mapping[str, EntitySchema] | None = None,
    ) -> FindCall | SummarizeCall:
        """Compile to an API call descriptor without I/O.

        Args:
            schema: Optional schema information used during compilation.

        Returns:
            A `FindCall` or `SummarizeCall` containing the compiled arguments.

        Raises:
            CompilationError: If the statement cannot be compiled to an API
                call.
        """
        try:
            return compile_statement(self, schema)
        except SchemaDrift as error:
            raise CompilationError(
                f'{error.entity_type}.{error.field_code} target '
                f'{error.target_type!r} is absent from the captured schema; refresh '
                'schema before compiling this joinedload'
            ) from None

    def execution_options(self, **flags: object) -> Self:
        """Add backend execution flags.

        Args:
            **flags: Backend flags validated for the statement kind at compile
                time.

        Returns:
            Statement copy containing the merged execution flags.
        """
        return dataclasses.replace(self, exec_options={**self.exec_options, **flags})

    def group_by(self, *groupings: Grouping | FieldExpr) -> Self:
        """Add groupings, treating bare fields as exact groupings.

        Args:
            *groupings: Explicit groupings or bare fields.

        Returns:
            Statement copy containing the accumulated groupings.

        Raises:
            TypeError: If an item is neither a field nor a grouping.
        """
        resolved = tuple(_grouping(grouping) for grouping in groupings)
        return dataclasses.replace(self, groupings=(*self.groupings, *resolved))

    def limit(self, n: int) -> Self:
        """Set the provider result limit.

        Args:
            n: Maximum number of results; zero requests no limit.

        Returns:
            Statement copy containing the result limit.
        """
        return dataclasses.replace(self, limit_value=n)

    def options(self, *opts: _LoaderOption) -> Self:
        """Add loader options.

        Args:
            *opts: Loader options to append.

        Returns:
            Statement copy containing the accumulated loader options.

        Raises:
            TypeError: If an item is not a loader option.
        """
        resolved = tuple(_loader_option(option) for option in opts)
        return dataclasses.replace(self, loader_options=(*self.loader_options, *resolved))

    def order_by(self, *orderings: Ordering | FieldExpr) -> Self:
        """Add order-by items.

        Args:
            *orderings: Explicit orderings or bare fields to sort ascending.

        Returns:
            Statement copy containing the accumulated orderings.

        Raises:
            TypeError: If an item is neither a field nor an ordering.
        """
        resolved = tuple(_ordering(ordering) for ordering in orderings)
        return dataclasses.replace(self, orderings=(*self.orderings, *resolved))

    def page(self, n: int) -> Self:
        """Set the SG-native result page.

        Args:
            n: One-based result page, or zero to request all pages.

        Returns:
            Statement copy containing the result page.
        """
        return dataclasses.replace(self, page_value=n)

    def where(self, *conds: Filter | FilterGroup | RawFilters) -> Self:
        """Add filter conditions that are ANDed together.

        Args:
            *conds: Filter conditions to append.

        Returns:
            Statement copy containing the accumulated conditions.

        Raises:
            TypeError: If an item is not a filter condition.
        """
        require_conditions('where()', conds)
        return dataclasses.replace(self, conditions=(*self.conditions, *conds))


def _grouping(value: object) -> Grouping:
    """Resolve a field or grouping to a provider grouping descriptor."""
    if isinstance(value, Grouping):
        return value
    if isinstance(value, FieldExpr):
        return Grouping(value.filter_field, 'exact', entity_type=value.root_type)
    raise TypeError(
        'group_by() takes fields or groupings from field.by_() and its shortcuts, '
        f'not {type(value).__name__}'
    )


def _loader_option(value: object) -> _LoaderOption:
    """Validate a loader option and guide bare fields to a loader."""
    if isinstance(value, _LoaderOption):
        return value
    if isinstance(value, FieldExpr):
        raise TypeError(
            'options() takes loader options, not a bare field; wrap the field in a loader such as '
            'selectinload() or load_only()'
        )
    raise TypeError(
        'options() takes loader options such as selectinload() or load_only(), '
        f'not {type(value).__name__}'
    )


def _ordering(value: object) -> Ordering:
    """Resolve a field or ordering to a provider sort descriptor."""
    if isinstance(value, Ordering):
        return value
    if isinstance(value, FieldExpr):
        return value.asc()
    raise TypeError(
        'order_by() takes fields or orderings from field.asc() and field.desc(), '
        f'not {type(value).__name__}'
    )


@overload
def select(target: type[_EntityT], /) -> Select[_EntityT]: ...


@overload
def select(*targets: FieldExpr | Summary) -> Select[None]: ...


def select(*targets: type[object] | FieldExpr | Summary) -> Select[Entity] | Select[None]:
    """Build an entity, field-projection, or aggregate statement.

    Args:
        *targets: One entity class, one or more field expressions, or one or
            more summary expressions. These selection shapes cannot be mixed.

    Returns:
        New statement whose supported shape is validated during compilation.
    """
    if len(targets) == 1 and isinstance(targets[0], type):
        return Select(entity=cast('type[Entity]', targets[0]))
    return Select[None](columns=cast('tuple[FieldExpr | Summary, ...]', targets))
