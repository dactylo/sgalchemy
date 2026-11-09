import asyncio

import pytest

from sgalchemy._core import _registry
from sgalchemy.schema import _model
from tests._fakes import FakeShotgun
from tests._support import (
    entity_namespace_scope,
    fetch_schema,
    reset_default_registry_state,
    run_async,
)


def test_entity_namespace_scope_clears_materialized_class(
    fake_sg: FakeShotgun, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_default_registry_state(monkeypatch)
    with entity_namespace_scope(('Shot',)) as entities:
        _registry.default_registry().bind(
            _model.build_schema(fetch_schema(fake_sg)), key=fake_sg.base_url
        )
        _ = entities.Shot

    assert 'Shot' not in vars(entities)


def test_run_async_times_out_and_drains_blocked_tasks() -> None:
    tasks: list[asyncio.Task[None]] = []
    drained: list[bool] = []

    async def scenario() -> None:
        entered = asyncio.Event()

        async def worker() -> None:
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                drained.append(True)

        tasks.append(asyncio.create_task(worker()))
        await entered.wait()
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        run_async(scenario(), timeout=0.01)

    assert drained == [True]
    assert len(tasks) == 1
    assert tasks[0].cancelled()
