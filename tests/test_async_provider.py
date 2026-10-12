import asyncio
import os
from pathlib import Path
from typing import Protocol, cast

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema

from sgalchemy._core._async_engine import create_async_engine
from sgalchemy._core._registry import Registry
from sgalchemy._core._statement import select
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy.testing import async_mockgun_connection


class _AsyncMockgunModule(Protocol):
    AsyncShotgun: type[object]


class _AsyncProviderModule(Protocol):
    AsyncShotgun: type[object]


@pytest.mark.native_async
def test_native_provider_async_mockgun_round_trip(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    required = os.environ.get('SGALCHEMY_REQUIRE_NATIVE_ASYNC') == '1'
    try:
        import shotgun_api3  # noqa: PLC0415 - Optional provider may be absent.
        from shotgun_api3.lib import mockgun  # noqa: PLC0415 - Optional provider may be absent.

        async_cls = cast('_AsyncProviderModule', shotgun_api3).AsyncShotgun
        async_mockgun_cls = cast('_AsyncMockgunModule', mockgun).AsyncShotgun
    except (ImportError, AttributeError) as err:
        if required:
            pytest.fail(f'required native async provider is unavailable: {err}')
        pytest.skip(f'native async provider is unavailable: {err}')
    assert async_cls is not None
    assert async_mockgun_cls is not None

    async def scenario() -> None:
        connection = async_mockgun_connection(fetch_schema(fake_sg), tmp_path)
        engine = create_async_engine(
            connection=connection,
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        shot_cls = engine.entities.Shot
        session = AsyncSession(engine)
        pending = shot_cls.new(code='native-A')
        session.add(pending)
        await session.commit()
        found = (await session.scalars(select(shot_cls).where(shot_cls.id == pending.id)))[0]
        found.code = 'native-B'
        await session.commit()
        await session.close()
        fresh_session = AsyncSession(engine)
        persisted = (
            await fresh_session.scalars(select(shot_cls).where(shot_cls.id == pending.id))
        )[0]
        assert persisted.code == 'native-B'
        await fresh_session.close()
        await engine.dispose()

    asyncio.run(scenario())
