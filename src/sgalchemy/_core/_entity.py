"""Store collision-free metadata for reflected entity classes."""

from weakref import WeakKeyDictionary

_ENTITY_REGISTRIES: WeakKeyDictionary[type[object], object] = WeakKeyDictionary()
_ENTITY_TYPES: WeakKeyDictionary[type[object], str] = WeakKeyDictionary()


def entity_registry(cls: type[object]) -> object | None:
    """Return the opaque registry that owns an entity class.

    Args:
        cls: Reflected or explicitly marked entity class.

    Returns:
        Opaque owning registry, or `None` when the class has no registry.
    """
    return _ENTITY_REGISTRIES.get(cls)


def entity_type(cls: type[object]) -> str:
    """Return the registered API entity type.

    Args:
        cls: Reflected or explicitly marked entity class.

    Returns:
        Canonical or symbolic API entity type.

    Raises:
        TypeError: If the class is not an sgalchemy entity target.
    """
    try:
        return _ENTITY_TYPES[cls]
    except KeyError:
        raise TypeError(f'{cls!r} is not a registered sgalchemy entity class') from None


def register_entity_class(
    cls: type[object],
    type_name: str,
    registry: object | None,
) -> None:
    """Register collision-free metadata for an entity class.

    Args:
        cls: Entity class to register.
        type_name: Canonical or symbolic API type name.
        registry: Opaque owning registry, or `None` for a compiler test target.
    """
    _ENTITY_TYPES[cls] = type_name
    if registry is not None:
        _ENTITY_REGISTRIES[cls] = registry
    else:
        _ENTITY_REGISTRIES.pop(cls, None)
