import pytest

from sgalchemy._core._entity import (
    entity_registry,
    entity_type,
    register_entity_class,
)


def test_entity_registry_returns_none_for_unregistered_class() -> None:
    class Unregistered:
        pass

    assert entity_registry(Unregistered) is None


def test_entity_type_rejects_unregistered_class() -> None:
    class Unregistered:
        pass

    with pytest.raises(TypeError, match='not a registered sgalchemy entity class'):
        entity_type(Unregistered)


def test_register_entity_class_stores_type_and_registry_outside_class() -> None:
    class Asset:
        pass

    registry = object()
    register_entity_class(Asset, 'Asset', registry)

    assert entity_type(Asset) == 'Asset'
    assert entity_registry(Asset) is registry
    assert 'entity_type' not in vars(Asset)
    assert 'entity_registry' not in vars(Asset)


def test_register_entity_class_with_none_clears_stale_registry() -> None:
    class Asset:
        pass

    register_entity_class(Asset, 'Asset', object())
    register_entity_class(Asset, 'Asset', None)

    assert entity_type(Asset) == 'Asset'
    assert entity_registry(Asset) is None
