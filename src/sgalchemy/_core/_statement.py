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
)
from sgalchemy._core._options import Defer, JoinedLoad, LazyLoad, LoadOnly, SelectinLoad
from sgalchemy._core._registry import Entity
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._errors import CompilationError
from sgalchemy.schema._model import EntitySchema

_EntityT = TypeVar('_EntityT', bound=Entity)
_LoaderOption = Defer | JoinedLoad | LazyLoad | LoadOnly | SelectinLoad
_SelectT_co = TypeVar('_SelectT_co', bound=Entity | None, covariant=True)


def _empty_exec_options() -> dict[str, object]:
    """Create an empty execution-option mapping."""
    return {}


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
    exec_options: Mapping[str, object] = field(default_factory=_empty_exec_options)

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
        """
        resolved = tuple(
            grouping
            if isinstance(grouping, Grouping)
            else Grouping(
                grouping.filter_field,
                'exact',
                entity_type=grouping.root_type,
            )
            for grouping in groupings
        )
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
        """
        return dataclasses.replace(self, loader_options=(*self.loader_options, *opts))

    def order_by(self, *orderings: Ordering) -> Self:
        """Add order-by items.

        Args:
            *orderings: Ordering expressions to append.

        Returns:
            Statement copy containing the accumulated orderings.
        """
        return dataclasses.replace(self, orderings=(*self.orderings, *orderings))

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
        """
        return dataclasses.replace(self, conditions=(*self.conditions, *conds))


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
