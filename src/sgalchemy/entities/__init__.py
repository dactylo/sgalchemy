"""Expose entity classes from the process default Registry.

Before schema binding, non-private names create symbolic classes. After
binding, dynamic lookup accepts canonical names and supported display aliases;
unknown names raise `AttributeError`. Dynamic lookup rejects private names in
either state; existing module attributes follow ordinary Python lookup.

Canonical classes resolved after binding are cached as module attributes.
Display aliases are resolved against the current binding on each lookup.
Directory enumeration combines module globals with names known to the Registry.
Namespace access performs no provider I/O.
"""

from sgalchemy._core._registry import Entity as _Entity
from sgalchemy._core._registry import default_registry as _default_registry
from sgalchemy._core._registry import (
    resolve_default_entity_class as _resolve_default_entity_class,
)


def __dir__() -> list[str]:
    """List module globals and registry-known entity names."""
    return sorted(set(globals()) | _default_registry().known_names())


def __getattr__(name: str) -> type[_Entity]:
    """Resolve an entity class from the default registry."""
    if name.startswith('_'):
        raise AttributeError(name)

    def cache(cls: type[_Entity]) -> None:
        globals()[name] = cls

    return _resolve_default_entity_class(name, cache)
