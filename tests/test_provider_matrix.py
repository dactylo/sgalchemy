import asyncio
import os
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes import _FakeShotgun
from _support import PROVIDER_DISTRIBUTIONS, fetch_schema

from sgalchemy._core._async_engine import create_async_engine
from sgalchemy._core._engine import create_engine
from sgalchemy._core._registry import Registry
from sgalchemy._core._statement import select
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._orm._session import Session
from sgalchemy._shotgun import (
    REQUIRED_SHOTGUN_METHODS,
    AsyncClosableShotgunLike,
    ClosableShotgunLike,
)
from sgalchemy.testing import mockgun_engine


def _installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


def test_required_provider_method_inventory_is_exact() -> None:
    assert REQUIRED_SHOTGUN_METHODS == (
        'batch',
        'find',
        'find_one',
        'revive',
        'schema_entity_read',
        'schema_read',
        'summarize',
        'upload_filmstrip_thumbnail',
        'upload_thumbnail',
    )


@pytest.mark.provider_contract
def test_selected_provider_construction_matrix(tmp_path: Path) -> None:
    installed = [name for name, dist in PROVIDER_DISTRIBUTIONS.items() if _installed(dist)]
    if not installed:
        pytest.skip('no optional provider distribution is installed')
    if len(installed) > 1:
        pytest.fail('both provider distributions are installed; select exactly one')

    provider = cast('Any', import_module('shotgun_api3'))
    site = 'https://example.shotgrid.autodesk.com'
    value = 'matrix-token'
    provider.Shotgun(site, script_name='matrix', api_key='k', connect=False).close()
    provider.Shotgun(site, session_token=value, connect=False).close()

    def make_client() -> ClosableShotgunLike:
        return provider.Shotgun(site, session_token=value, connect=False)

    engine = create_engine(site, creator=make_client)
    assert engine.connection.base_url == site
    engine.dispose()

    async def threaded_scenario() -> None:
        threaded = create_async_engine(
            site, strategy='threaded', creator=make_client, cache_root=tmp_path
        )
        assert await cast('Any', threaded)._backend.supports('find')
        await threaded.dispose()

    asyncio.run(threaded_scenario())

    async def native_scenario() -> None:
        native = create_async_engine(site, session_token=value, cache_root=tmp_path)
        supports = cast('Any', native)._supports
        if installed == ['api']:
            assert await supports('find')

            async def make_async() -> AsyncClosableShotgunLike:
                return provider.AsyncShotgun(site, session_token=value)

            with_creator = create_async_engine(site, async_creator=make_async, cache_root=tmp_path)
            assert await cast('Any', with_creator)._supports('find')
            await with_creator.dispose()
            await native.dispose()
        else:
            with pytest.raises(ImportError, match='sg-python-api'):
                await supports('find')

    asyncio.run(native_scenario())


@pytest.mark.provider_contract
def test_selected_provider_sync_and_threaded_contract(
    fake_sg: _FakeShotgun, tmp_path: Path
) -> None:
    required = os.environ.get('SGALCHEMY_REQUIRE_PROVIDER')
    if required is not None and required not in PROVIDER_DISTRIBUTIONS:
        pytest.fail(f'unknown SGALCHEMY_REQUIRE_PROVIDER value: {required!r}')
    installed = [name for name, dist in PROVIDER_DISTRIBUTIONS.items() if _installed(dist)]
    if required is not None and required not in installed:
        pytest.fail(
            f'required provider distribution is unavailable: {PROVIDER_DISTRIBUTIONS[required]}'
        )
    if len(installed) > 1:
        pytest.fail('both provider distributions are installed; select exactly one')
    if not installed:
        pytest.skip('no optional provider distribution is installed')

    sync_engine = mockgun_engine(fetch_schema(fake_sg), cache_root=tmp_path / 'sync')
    sync_engine.load_schema()
    sync_session = Session(sync_engine)
    sync_shot_cls = cast('Any', sync_engine.entities.Shot)
    sync_obj = sync_shot_cls.new(code='sync-provider')
    sync_session.add(sync_obj)
    sync_session.commit()
    sync_found = sync_session.scalars(
        select(sync_shot_cls).where(sync_shot_cls.id == sync_obj.id)
    )[0]
    sync_found.code = 'sync-provider-updated'
    sync_session.commit()
    sync_session.close()
    fresh_sync_session = Session(sync_engine)
    sync_persisted = fresh_sync_session.scalars(
        select(sync_shot_cls).where(sync_shot_cls.id == sync_obj.id)
    )[0]
    assert sync_persisted.code == 'sync-provider-updated'
    fresh_sync_session.close()

    async def threaded_contract() -> None:
        async_engine = create_async_engine(
            sync_connection=sync_engine.connection,
            strategy='threaded',
            cache_root=tmp_path / 'threaded',
            registry=Registry(),
            bind_default=False,
        )
        await async_engine.load_schema()
        async_session = AsyncSession(async_engine)
        async_shot_cls = cast('Any', async_engine.entities.Shot)
        async_obj = async_shot_cls.new(code='threaded-provider')
        async_session.add(async_obj)
        await async_session.commit()
        async_found = (
            await async_session.scalars(
                select(async_shot_cls).where(async_shot_cls.id == async_obj.id)
            )
        )[0]
        async_found.code = 'threaded-provider-updated'
        await async_session.commit()
        await async_session.close()
        fresh_async_session = AsyncSession(async_engine)
        async_persisted = (
            await fresh_async_session.scalars(
                select(async_shot_cls).where(async_shot_cls.id == async_obj.id)
            )
        )[0]
        assert async_persisted.code == 'threaded-provider-updated'
        await fresh_async_session.close()
        await async_engine.dispose()

    asyncio.run(threaded_contract())
