"""Define loader options and accepted execution options."""

from collections.abc import Iterable
from dataclasses import dataclass

from sgalchemy._core._expressions import FieldExpr
from sgalchemy._errors import CompilationError

FIND_EXECUTION_OPTIONS = frozenset(
    {'retired_only', 'include_archived_projects', 'additional_filter_presets'}
)
SUMMARIZE_EXECUTION_OPTIONS = frozenset({'include_archived_projects'})


@dataclass(frozen=True, eq=False)
class Defer:
    """Remove fields from the default loaded set."""

    fields: tuple[FieldExpr, ...]

    def __eq__(self, other: object) -> bool:
        """Compare deferred fields by structural field identity."""
        if type(other) is not Defer:
            return False
        return tuple(map(_field_signature, self.fields)) == tuple(
            map(_field_signature, other.fields)
        )

    def __hash__(self) -> int:
        """Hash deferred fields by structural field identity."""
        return hash((Defer, tuple(map(_field_signature, self.fields))))


@dataclass(frozen=True, eq=False)
class JoinedLoad:
    """Project a to-one relationship chain through one primary query.

    Missing target schema may trigger one pre-query refresh and complete
    statement recompilation. Each hop must have exactly one reflected target
    type; multi-entity links require a different loader.

    Attributes:
        paths: Ordered direct relationship fields, each rooted at the previous
            hop's target type. These are individual hops, not one deep path.
    """

    paths: tuple[FieldExpr, ...]

    def __eq__(self, other: object) -> bool:
        """Compare relationship paths by structural field identity."""
        if type(other) is not JoinedLoad:
            return False
        return tuple(map(_field_signature, self.paths)) == tuple(
            map(_field_signature, other.paths)
        )

    def __hash__(self) -> int:
        """Hash relationship paths by structural field identity."""
        return hash((JoinedLoad, tuple(map(_field_signature, self.paths))))

    def joinedload(self, field: FieldExpr) -> 'JoinedLoad':
        """Extend the chain one hop deeper.

        Args:
            field: Relationship field for the next hop.

        Returns:
            Loader option containing the extended relationship chain.
        """
        return JoinedLoad((*self.paths, field))


@dataclass(frozen=True, eq=False)
class LazyLoad:
    """Opt-in sync attribute I/O across one originating result set.

    The load reserves its complete per-target-type query count against the
    Session's `max_implicit_queries` budget before provider I/O.
    """

    field: FieldExpr

    def __eq__(self, other: object) -> bool:
        """Compare lazy-loaded fields by structural field identity."""
        if type(other) is not LazyLoad:
            return False
        return _field_signature(self.field) == _field_signature(other.field)

    def __hash__(self) -> int:
        """Hash a lazy-loaded field by structural field identity."""
        return hash((LazyLoad, _field_signature(self.field)))


@dataclass(frozen=True, eq=False)
class LoadOnly:
    """Restrict loaded fields to the given set (plus id)."""

    fields: tuple[FieldExpr, ...]

    def __eq__(self, other: object) -> bool:
        """Compare selected fields by structural field identity."""
        if type(other) is not LoadOnly:
            return False
        return tuple(map(_field_signature, self.fields)) == tuple(
            map(_field_signature, other.fields)
        )

    def __hash__(self) -> int:
        """Hash selected fields by structural field identity."""
        return hash((LoadOnly, tuple(map(_field_signature, self.fields))))


@dataclass(frozen=True, eq=False)
class SelectinLoad:
    """Batched target reads for one relationship across a whole result set."""

    field: FieldExpr

    def __eq__(self, other: object) -> bool:
        """Compare select-in fields by structural field identity."""
        if type(other) is not SelectinLoad:
            return False
        return _field_signature(self.field) == _field_signature(other.field)

    def __hash__(self) -> int:
        """Hash a select-in field by structural field identity."""
        return hash((SelectinLoad, _field_signature(self.field)))


def _field_signature(field: FieldExpr) -> tuple[str, tuple[str, ...]]:
    """Return the structural identity of a field expression."""
    return (field.root_type, field.path)


def defer(*fields: FieldExpr) -> Defer:
    """Exclude the named fields from the default load.

    Args:
        *fields: Fields to exclude.

    Returns:
        Deferred-field loader option.
    """
    return Defer(tuple(fields))


def joinedload(field: FieldExpr) -> JoinedLoad:
    """Project a to-one relationship through one primary provider query.

    Args:
        field: Relationship field to load.

    Returns:
        Deep-field loader option.
    """
    return JoinedLoad((field,))


def lazyload(field: FieldExpr) -> LazyLoad:
    """Opt into bounded synchronous I/O on first `field` access.

    Args:
        field: Relationship field to load lazily.

    Returns:
        Deferred relationship loader option.
    """
    return LazyLoad(field)


def load_only(*fields: FieldExpr) -> LoadOnly:
    """Load only the named fields (plus id).

    Args:
        *fields: Fields to load.

    Returns:
        Restricted-field loader option.
    """
    return LoadOnly(tuple(fields))


def normalize_relationship_loaders(
    options: Iterable[object],
) -> tuple[JoinedLoad | LazyLoad | SelectinLoad, ...]:
    """Coalesce relationship loaders and reject conflicting strategies.

    Args:
        options: Statement loader options in user-specified order.

    Returns:
        Relationship options with exact structural duplicates removed.

    Raises:
        CompilationError: If one root relationship has conflicting strategies.
    """
    normalized: list[JoinedLoad | LazyLoad | SelectinLoad] = []
    seen: set[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]] = set()
    strategies: dict[tuple[str, str], str] = {}
    for option in options:
        if isinstance(option, JoinedLoad):
            strategy = 'joinedload'
            fields = option.paths
        elif isinstance(option, LazyLoad):
            strategy = 'lazyload'
            fields = (option.field,)
        elif isinstance(option, SelectinLoad):
            strategy = 'selectinload'
            fields = (option.field,)
        else:
            continue
        signature = (strategy, tuple((field.root_type, field.path) for field in fields))
        if signature in seen:
            continue
        if fields:
            first = fields[0]
            key = (first.root_type, first.filter_field)
            previous = strategies.get(key)
            if previous is not None and previous != strategy:
                raise CompilationError(
                    f'conflicting relationship loader strategies for {key[0]}.{key[1]}: '
                    f'{previous} and {strategy}; choose exactly one of joinedload(), '
                    'selectinload(), or lazyload()'
                )
            strategies[key] = strategy
        seen.add(signature)
        normalized.append(option)
    return tuple(normalized)


def selectinload(field: FieldExpr) -> SelectinLoad:
    """Load observed relationship targets with one find per target type.

    Args:
        field: Relationship field to load.

    Returns:
        Batched relationship loader option.
    """
    return SelectinLoad(field)
