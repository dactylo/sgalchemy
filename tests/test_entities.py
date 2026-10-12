import importlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from typing import Any

import pytest
from _fakes import _FakeShotgun
from _support import entity_namespace_scope, fetch_schema, reset_default_registry_state

from sgalchemy._core import _registry
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema import _model


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    reset_default_registry_state(monkeypatch)
    with entity_namespace_scope(('Asset', 'CustomEntity03', 'Shot')):
        yield


def test_alias_stays_live_through_repoint_and_drop(fake_sg: _FakeShotgun) -> None:
    ents = importlib.import_module('sgalchemy.entities')

    registry = _registry.default_registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    vendor = ents.Vendor
    assert _registry.entity_type(vendor) == 'Vendor'
    assert 'Vendor' not in vars(ents)

    registry.bind(schemas, key=fake_sg.base_url)
    assert ents.Vendor is ents.CustomEntity03
    assert 'Vendor' not in vars(ents)

    repointed = {
        **schemas,
        'Asset': replace(schemas['Asset'], display_name='Vendor'),
        'CustomEntity03': replace(schemas['CustomEntity03'], display_name='Custom Entity 03'),
    }
    registry.bind(repointed, key=fake_sg.base_url)
    assert ents.Vendor is ents.Asset
    assert 'Vendor' not in vars(ents)

    dropped = {**repointed, 'Asset': replace(schemas['Asset'], display_name='Asset')}
    registry.bind(dropped, key=fake_sg.base_url)
    with pytest.raises(AttributeError, match='unknown entity'):
        _ = ents.Vendor
    assert 'Vendor' not in vars(ents)


def test_binding_during_symbolic_lookup_uses_one_registry_generation(
    fake_sg: _FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    ents = importlib.import_module('sgalchemy.entities')

    registry = _registry.default_registry()
    schemas = _model.build_schema(fetch_schema(fake_sg))
    original = registry.entity_class
    first = True
    calls = 0

    def lookup(name: str) -> type[_registry.Entity]:
        nonlocal calls, first
        calls += 1
        cls = original(name)
        if first:
            first = False
            registry.bind(schemas, key=fake_sg.base_url)
        return cls

    monkeypatch.setattr(registry, 'entity_class', lookup)
    symbolic = ents.Vendor

    assert calls == 1
    assert _registry.entity_type(symbolic) == 'Vendor'
    assert 'Vendor' not in vars(ents)
    assert ents.Vendor is ents.CustomEntity03
    assert 'Vendor' not in vars(ents)


def test_bound_unknown_and_leading_underscore_raise(fake_sg: _FakeShotgun) -> None:
    ents = importlib.import_module('sgalchemy.entities')

    with pytest.raises(AttributeError):
        _ = ents._Private
    _registry.default_registry().bind(
        _model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url
    )
    with pytest.raises(AttributeError, match='unknown entity'):
        _ = ents.NoSuchType


def test_canonical_lookup_after_binding_is_cached(fake_sg: _FakeShotgun) -> None:
    ents = importlib.import_module('sgalchemy.entities')

    _registry.default_registry().bind(
        _model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url
    )
    shot = ents.Shot
    assert ents.Shot is shot
    assert vars(ents)['Shot'] is shot


def test_canonical_lookup_before_binding_stays_live_and_stable() -> None:
    ents = importlib.import_module('sgalchemy.entities')

    shot = ents.Shot
    assert _registry.entity_type(shot) == 'Shot'
    assert ents.Shot is shot
    assert 'Shot' not in vars(ents)


def test_concurrent_namespace_lookups_share_identity() -> None:
    ents = importlib.import_module('sgalchemy.entities')

    start = Barrier(8)

    def lookup() -> type[Any]:
        start.wait(timeout=2)
        return ents.Shot

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lookup) for _index in range(8)]
        classes = [future.result(timeout=2) for future in futures]
    assert all(cls is classes[0] for cls in classes)


def test_default_registry_replacement_freezes_after_class_access() -> None:
    replacement = _registry.Registry()
    _registry.set_default_registry(replacement)
    assert _registry.default_registry() is replacement
    ents = importlib.import_module('sgalchemy.entities')

    _ = ents.Shot
    with pytest.raises(SgalchemyError, match='cannot be replaced'):
        _registry.set_default_registry(_registry.Registry())


def test_dir_hides_default_generation_plumbing() -> None:
    ents = importlib.import_module('sgalchemy.entities')

    plumbing_names = {
        'default_registry',
        'resolve_default_entity_class',
    }
    assert plumbing_names.isdisjoint(dir(ents))


def test_dir_reports_globals_and_registry_names(fake_sg: _FakeShotgun) -> None:
    ents = importlib.import_module('sgalchemy.entities')

    _ = ents.SymbolicOnly
    assert 'SymbolicOnly' in dir(ents)
    assert 'Vendor' not in dir(ents)
    _registry.default_registry().bind(
        _model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url
    )
    names = dir(ents)
    assert 'Shot' in names
    assert 'CustomEntity03' in names
    assert 'Vendor' in names
    assert '__getattr__' in names


def test_entity_namespace_scope_clears_materialized_class(fake_sg: _FakeShotgun) -> None:
    with entity_namespace_scope(('Shot',)) as entities:
        _registry.default_registry().bind(
            _model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url
        )
        _ = entities.Shot

    assert 'Shot' not in vars(entities)


def test_materialization_wins_before_prefreeze_default_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ents = importlib.import_module('sgalchemy.entities')

    current = _registry.default_registry()
    replacement = _registry.Registry()
    entered = Event()
    release = Event()
    commit_attempted = Event()
    original_build = object.__getattribute__(current, '_build')

    def blocking_build(canonical: str) -> type[_registry.Entity]:
        entered.set()
        assert release.wait(timeout=10)
        return original_build(canonical)

    monkeypatch.setattr(current, '_build', blocking_build)
    owner = object()
    _registry.reserve_default_registry_transition(owner)

    def commit() -> None:
        commit_attempted.set()
        _registry.commit_default_registry_transition(
            replacement,
            default_bound=True,
            publish=lambda: None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        materializing = pool.submit(lambda: ents.Shot)
        assert entered.wait(timeout=2)
        committing = pool.submit(commit)
        assert commit_attempted.wait(timeout=2)
        assert not committing.done()
        release.set()
        shot_cls = materializing.result(timeout=2)
        with pytest.raises(SgalchemyError, match='cannot be replaced'):
            committing.result(timeout=2)

    _registry.release_default_registry_transition(owner)
    assert _registry.default_registry() is current
    assert _registry.entity_registry(shot_cls) is current


def test_prefreeze_default_commit_wins_before_materialization() -> None:
    ents = importlib.import_module('sgalchemy.entities')

    replacement = _registry.Registry()
    publish_entered = Event()
    release_publish = Event()
    lookup_attempted = Event()
    _registry.reserve_default_registry_transition(object())

    def publish() -> None:
        publish_entered.set()
        assert release_publish.wait(timeout=10)

    def lookup() -> type[_registry.Entity]:
        lookup_attempted.set()
        return ents.Shot

    with ThreadPoolExecutor(max_workers=2) as pool:
        committing = pool.submit(
            _registry.commit_default_registry_transition,
            replacement,
            default_bound=True,
            publish=publish,
        )
        assert publish_entered.wait(timeout=2)
        materializing = pool.submit(lookup)
        assert lookup_attempted.wait(timeout=2)
        assert not materializing.done()
        release_publish.set()
        committing.result(timeout=2)
        shot_cls = materializing.result(timeout=2)

    assert _registry.default_registry() is replacement
    assert _registry.entity_registry(shot_cls) is replacement
