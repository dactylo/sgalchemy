"""Testing helpers for fake and Mockgun backends.

Mockgun capabilities vary by provider. Skip tests for capabilities the selected
provider does not support, such as unsupported filters or query options.

Mockgun implements `batch()` as sequential, non-transactional requests.
It must not be used to assert production batch atomicity.
"""

import copy
import pickle
import shutil
import tempfile
import weakref
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, TypedDict, Unpack, cast

from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._registry import Registry
from sgalchemy._errors import SgalchemyError
from sgalchemy._shotgun import AsyncShotgunLike, ShotgunLike
from sgalchemy.schema._cache import SchemaCache
from sgalchemy.schema._policy import SchemaCachePolicy

__all__ = ['async_mockgun_connection', 'engine_over', 'mockgun_engine']


class _EngineOverOptions(TypedDict, total=False):
    """Describe optional keyword arguments accepted by `engine_over()`."""

    policy: SchemaCachePolicy | None
    echo: bool
    registry: Registry | None
    bind_default: bool
    serialize: bool | None


class _AsyncMockgunFactory(Protocol):
    """Typed boundary for async Mockgun's unannotated class object."""

    def __call__(self, base_url: str) -> AsyncShotgunLike:
        """Construct one async Mockgun connection."""
        ...


class _MockgunFactory(Protocol):
    """Typed boundary for Mockgun's unannotated class object."""

    def __call__(self, base_url: str) -> ShotgunLike:
        """Construct one Mockgun connection."""
        ...

    def set_schema_paths(self, schema_path: Path, entity_path: Path) -> None:
        """Set Mockgun's process-global schema paths."""
        ...


class _MockgunModule(Protocol):
    """Typed boundary for the synchronous Mockgun package surface."""

    Shotgun: _MockgunFactory


class _AsyncMockgunModule(_MockgunModule, Protocol):
    """Typed boundary for Mockgun packages with async support."""

    AsyncShotgun: _AsyncMockgunFactory


def _mockgun_field(data_type: str, name: str) -> dict[str, Any]:
    """Build field metadata needed by Mockgun's constructor."""
    return {
        'data_type': {'value': data_type},
        'name': {'value': name},
        'properties': {'default_value': {'value': None}},
    }


def _prepare_mockgun_schema(schema_cache: SchemaCache, cache_root: Path) -> tuple[Path, Path]:
    """Serialize one schema cache in Mockgun's supported format."""
    cache_root.mkdir(parents=True, exist_ok=True)
    schema_path = cache_root / 'mockgun_schema.pickle'
    entity_path = cache_root / 'mockgun_schema_entity.pickle'
    fields = copy.deepcopy(schema_cache.fields)
    entities = copy.deepcopy(schema_cache.entities)
    for entity_fields in fields.values():
        for field in entity_fields.values():
            field.setdefault('properties', {}).setdefault(
                'default_value',
                {'value': None},
            )
    fields.setdefault(
        'EventLogEntry',
        {
            'description': _mockgun_field('text', 'Description'),
            'event_type': _mockgun_field('text', 'Event Type'),
            'id': _mockgun_field('number', 'Id'),
        },
    )
    entities.setdefault('EventLogEntry', {'name': {'value': 'Event Log Entry'}})
    with schema_path.open('wb') as stream:
        pickle.dump(fields, stream)
    with entity_path.open('wb') as stream:
        pickle.dump(entities, stream)
    return schema_path, entity_path


def _reset_mockgun_schema_factory(mockgun_module: _MockgunModule) -> None:
    """Clear a legacy schema factory's present path-keyed cache fields."""
    schema_module = getattr(mockgun_module, 'schema', None)
    schema_factory = getattr(schema_module, 'SchemaFactory', None)
    for attribute in (
        '_schema_cache',
        '_schema_cache_path',
        '_schema_entity_cache',
        '_schema_entity_cache_path',
    ):
        if hasattr(schema_factory, attribute):
            setattr(schema_factory, attribute, None)


def async_mockgun_connection(
    schema_cache: SchemaCache,
    cache_root: str | PathLike[str],
) -> AsyncShotgunLike:
    """Build the selected provider's configured async Mockgun client.

    Writes Mockgun schema pickles and replaces its process-global schema paths,
    affecting other Mockgun instances. The returned connection is caller-owned;
    passing it to an async engine does not transfer ownership. Prepare that
    engine's schema explicitly.

    Args:
        schema_cache: Raw field/entity schema payloads to install in Mockgun.
        cache_root: Schema cache directory override, as a path or string.

    Returns:
        Configured `mockgun.AsyncShotgun` through the consumed async protocol.

    Raises:
        SgalchemyError: If the selected provider lacks async Mockgun support.
        OSError: If the schema directory or pickle files cannot be created or
            written.
    """
    try:
        mockgun = cast('_AsyncMockgunModule', import_module('shotgun_api3.lib.mockgun'))
    except (ImportError, AttributeError) as err:
        raise SgalchemyError(
            "async Mockgun requires sg-python-api[async]; install the 'sgalchemy[async]' extra"
        ) from err
    try:
        async_cls = mockgun.AsyncShotgun
    except AttributeError as err:
        raise SgalchemyError(
            'selected provider lacks mockgun.AsyncShotgun; use sg-python-api[async]'
        ) from err
    schema_path, entity_path = _prepare_mockgun_schema(schema_cache, Path(cache_root))
    mockgun.Shotgun.set_schema_paths(schema_path, entity_path)
    return async_cls('https://mockgun.example.com')


def engine_over(
    connection: ShotgunLike,
    *,
    cache_root: str | PathLike[str] | None = None,
    **kw: Unpack[_EngineOverOptions],
) -> Engine:
    """Create an isolated engine over a caller-provided test connection.

    Each call defaults to a fresh `Registry`, `bind_default=False`, and a
    temporary schema cache removed when the engine is collected.
    Caller-supplied engine arguments override these isolation defaults;
    `bind_default=True` without `registry` opts into the process default.

    Args:
        connection: Fake or test provider implementing the synchronous Shotgun
            API surface.
        cache_root: Schema cache directory override, as a path or string.
        **kw: Additional arguments forwarded to `create_engine()`.

    Returns:
        Lazy synchronous engine wrapping `connection`.

    Raises:
        ValueError: If `connection.base_url` is not a valid site URL.
        OSError: If the default temporary cache directory cannot be created.
    """
    engine_kw: _EngineOverOptions = {'registry': Registry(), 'bind_default': False}
    engine_kw.update(kw)
    if kw.get('bind_default') is True and 'registry' not in kw:
        engine_kw.pop('registry')
    root = (
        Path(cache_root)
        if cache_root is not None
        else Path(tempfile.mkdtemp(prefix='sgalchemy-engine-over-'))
    )
    try:
        engine = create_engine(connection=connection, cache_root=root, **engine_kw)
    except Exception:
        if cache_root is None:
            shutil.rmtree(root, ignore_errors=True)
        raise
    if cache_root is None:
        weakref.finalize(engine, shutil.rmtree, root, ignore_errors=True)
    return engine


def mockgun_engine(schema_cache: SchemaCache, cache_root: str | PathLike[str]) -> Engine:
    """Create an isolated prepared engine backed by the selected Mockgun.

    Replaces Mockgun's process-global schema paths and clears legacy schema
    factory caches when present. Other Mockgun instances share those settings.
    The engine receives a fresh Registry and does not bind process defaults.

    Args:
        schema_cache: Schema snapshot used to initialize Mockgun.
        cache_root: Schema cache directory override, as a path or string.

    Returns:
        Prepared synchronous engine backed by an in-memory Mockgun connection.

    Raises:
        SgalchemyError: If the selected provider's Mockgun module is
            unavailable or schema preparation rejects its payload.
        OSError: If the schema directory or pickle files cannot be created or
            written.
        Exception: Other provider setup or schema-preparation failures,
            unwrapped.
    """
    try:
        mockgun = cast('_MockgunModule', import_module('shotgun_api3.lib.mockgun'))
    except ImportError as err:
        raise SgalchemyError(
            "Mockgun is unavailable; install the 'sgalchemy[official]' extra"
        ) from err

    root = Path(cache_root)
    schema_path, entity_path = _prepare_mockgun_schema(schema_cache, root)
    _reset_mockgun_schema_factory(mockgun)
    mockgun_factory = mockgun.Shotgun
    mockgun_factory.set_schema_paths(schema_path, entity_path)
    connection = mockgun_factory('https://mockgun.example.com')
    engine = create_engine(
        connection=connection,
        cache_root=root,
        registry=Registry(),
        bind_default=False,
    )
    engine.load_schema(force=True)
    return engine
