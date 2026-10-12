import os
import subprocess
import sys
from pathlib import Path

import sgalchemy

_REQUIRED_PUBLIC = [
    'AsyncEngine',
    'AsyncScopedSession',
    'AsyncSession',
    'AsyncSessionFactory',
    'AsyncShotgunLike',
    'CommitError',
    'CompilationError',
    'Engine',
    'Entity',
    'FieldExpr',
    'InspectionInfo',
    'RefreshRequiredError',
    'Registry',
    'RelationshipTargetUnavailableError',
    'Result',
    'RetiredEntityWriteError',
    'Row',
    'SchemaUnavailableError',
    'ScopedSession',
    'Select',
    'Session',
    'SessionFactory',
    'SgalchemyError',
    'ShotgunLike',
    'StaleObjectError',
    'UnfetchedFieldError',
    'and_',
    'async_scoped_session',
    'async_sessionmaker',
    'connect',
    'create_async_engine',
    'create_engine',
    'default_registry',
    'defer',
    'func',
    'inspect',
    'joinedload',
    'lazyload',
    'load_only',
    'or_',
    'raw',
    'scoped_session',
    'select',
    'selectinload',
    'sessionmaker',
    'set_default_registry',
]


def _run_nondefault_to_default_transition(tmp_path: Path, registry_mode: str) -> None:
    cache_root = tmp_path / registry_mode
    script = """
import os
from pathlib import Path

import sgalchemy
import sgalchemy.entities as entities
from sgalchemy._core._registry import (
    Registry,
    default_registry,
    entity_registry,
    entity_state,
)
from sgalchemy._orm._scoping import _default_session


class FakeShotgun:
    def __init__(self, base_url):
        self.base_url = base_url
        self.calls = []
        self.find_results = []

    def find(self, entity_type, filters, fields=None, **kwargs):
        self.calls.append(('find', entity_type, filters, fields, kwargs))
        return self.find_results.pop(0)

    def schema_read(self):
        self.calls.append(('schema_read',))
        return {
            'Shot': {
                'id': {
                    'data_type': {'value': 'number'},
                    'name': {'value': 'Id'},
                    'properties': {},
                },
                'code': {
                    'data_type': {'value': 'text'},
                    'name': {'value': 'Code'},
                    'properties': {},
                },
            }
        }

    def schema_entity_read(self):
        self.calls.append(('schema_entity_read',))
        return {'Shot': {'name': {'value': 'Shot'}}}


cache_root = Path(os.environ['TEST_CACHE_ROOT'])
registry_mode = os.environ['TEST_REGISTRY_MODE']
process_registry = default_registry()
shot = entities.Shot
assert entity_registry(shot) is process_registry
assert not process_registry.is_bound

old_connection = FakeShotgun('https://old.example.com')
first_kwargs = (
    {'registry': Registry()}
    if registry_mode == 'explicit'
    else {'bind_default': False}
)
old_session = sgalchemy.connect(
    connection=old_connection,
    cache_root=cache_root / 'old',
    **first_kwargs,
)
old_registry = old_session.engine.registry
old_shot = old_session.engine.entities.Shot
old_pending = old_shot.new(code='old pending')
old_session.add(old_pending)
assert old_registry is not process_registry
assert default_registry() is process_registry
assert not process_registry.is_bound
assert entity_registry(shot) is process_registry

new_connection = FakeShotgun('https://new.example.com')
new_connection.find_results = [[{'type': 'Shot', 'id': 2, 'code': 'new'}]]
replacement = sgalchemy.connect(
    connection=new_connection,
    cache_root=cache_root / 'new',
    force=True,
)

assert replacement.engine.registry is process_registry
assert default_registry() is process_registry
assert process_registry.is_bound
assert _default_session() is replacement
assert entities.Shot is shot
assert entity_registry(shot) is process_registry
new_pending = shot.new(code='new pending')
replacement.add(new_pending)
assert entity_state(new_pending).session is replacement

old_connection.find_results = [[{'type': 'Shot', 'id': 1, 'code': 'old'}]]
old_found = old_session.scalars(
    sgalchemy.select(old_shot).where(old_shot.code == 'old')
)
assert old_found[0].code == 'old'
assert entity_state(old_found[0]).session is old_session
old_later = old_shot.new(code='old later')
old_session.add(old_later)
assert entity_state(old_pending).session is old_session
assert entity_state(old_later).session is old_session

new_found = shot.find(code='new')
assert new_found[0].code == 'new'
assert entity_state(new_found[0]).session is replacement
assert [call[0] for call in old_connection.calls].count('find') == 1
assert [call[0] for call in new_connection.calls].count('find') == 1
"""
    env = dict(os.environ)
    source_root = Path(__file__).parents[1] / 'src'
    env['PYTHONPATH'] = os.pathsep.join(
        part for part in (str(source_root), env.get('PYTHONPATH')) if part
    )
    env['TEST_CACHE_ROOT'] = str(cache_root)
    env['TEST_REGISTRY_MODE'] = registry_mode
    subprocess.run([sys.executable, '-c', script], env=env, check=True)


def test_all_is_exact_sorted_and_importable() -> None:
    assert sgalchemy.__all__ == _REQUIRED_PUBLIC
    assert sgalchemy.__all__ == sorted(sgalchemy.__all__)
    for name in sgalchemy.__all__:
        assert getattr(sgalchemy, name) is not None


def test_first_force_connect_binds_lazy_default_in_fresh_process(tmp_path: Path) -> None:
    cache_root = tmp_path / 'fresh-force'
    script = """
import os
from pathlib import Path

import sgalchemy
import sgalchemy.entities as entities
from sgalchemy._core._registry import default_registry, entity_registry


class FakeShotgun:
    base_url = 'https://fresh-force.example.com'

    def schema_read(self):
        return {
            'Shot': {
                'id': {
                    'data_type': {'value': 'number'},
                    'name': {'value': 'Id'},
                    'properties': {},
                },
                'code': {
                    'data_type': {'value': 'text'},
                    'name': {'value': 'Code'},
                    'properties': {},
                },
            }
        }

    def schema_entity_read(self):
        return {'Shot': {'name': {'value': 'Shot'}}}


registry = default_registry()
shot = entities.Shot
assert entity_registry(shot) is registry
session = sgalchemy.connect(
    connection=FakeShotgun(),
    cache_root=Path(os.environ['TEST_CACHE_ROOT']),
    force=True,
)
assert session.engine.registry is registry
assert default_registry() is registry
assert entities.Shot is shot
pending = shot.new(code='bound')
session.add(pending)
"""
    env = dict(os.environ)
    source_root = Path(__file__).parents[1] / 'src'
    env['PYTHONPATH'] = os.pathsep.join(
        part for part in (str(source_root), env.get('PYTHONPATH')) if part
    )
    env['TEST_CACHE_ROOT'] = str(cache_root)
    subprocess.run([sys.executable, '-c', script], env=env, check=True)


def test_force_connect_from_explicit_session_binds_lazy_default_registry(
    tmp_path: Path,
) -> None:
    _run_nondefault_to_default_transition(tmp_path, 'explicit')


def test_force_connect_from_private_session_binds_lazy_default_registry(
    tmp_path: Path,
) -> None:
    _run_nondefault_to_default_transition(tmp_path, 'private')


def test_import_installs_entity_finder_without_provider_or_cache_io(tmp_path: Path) -> None:
    cache_root = tmp_path / 'must-not-exist'
    script = """
import sys
from typing import get_type_hints

import sgalchemy
from sgalchemy._core import _registry
from sgalchemy._core._registry import Registry, install_entity_finder
from sgalchemy._orm._session import Session

assert len(_registry._ENTITY_FINDERS) == 1

targets = (
    sgalchemy.Entity.find,
    sgalchemy.connect,
    sgalchemy.inspect,
    sgalchemy.scoped_session,
    sgalchemy.sessionmaker,
    sgalchemy.default_registry,
    sgalchemy.set_default_registry,
    install_entity_finder,
    Registry.binding_snapshot,
    Registry.entity_class_for,
    Registry.schema_snapshot,
    Session.execute_call,
    Session.hydrate_record,
)
for target in targets:
    assert get_type_hints(target)
assert get_type_hints(sgalchemy.InspectionInfo)
assert 'shotgun_api3' not in sys.modules
"""
    env = dict(os.environ)
    source_root = Path(__file__).parents[1] / 'src'
    env['PYTHONPATH'] = os.pathsep.join(
        part for part in (str(source_root), env.get('PYTHONPATH')) if part
    )
    env['SGALCHEMY_SCHEMA_CACHE'] = str(cache_root)
    subprocess.run([sys.executable, '-c', script], env=env, check=True)
    assert not cache_root.exists()
