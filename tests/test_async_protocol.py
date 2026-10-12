import asyncio
import inspect
from typing import Any, TypedDict, get_type_hints

import pytest
from _fakes import _FakeAsyncShotgun, _FakeShotgun

from sgalchemy._shotgun import (
    REQUIRED_SHOTGUN_METHODS,
    AsyncClosableShotgunLike,
    AsyncShotgunLike,
    ShotgunLike,
)


class _FindDefaultKwargs(TypedDict):
    additional_filter_presets: list[dict[str, Any]] | None
    filter_operator: str | None
    include_archived_projects: bool
    limit: int
    order: list[dict[str, str]] | None
    page: int
    retired_only: bool


class _SummarizeDefaultKwargs(TypedDict):
    filter_operator: str | None
    grouping: list[dict[str, str]] | None
    include_archived_projects: bool


def test_async_protocol_matches_complete_consumed_sync_signatures() -> None:
    for name in REQUIRED_SHOTGUN_METHODS:
        sync_member = getattr(ShotgunLike, name)
        async_member = getattr(AsyncShotgunLike, name)
        sync_signature = inspect.signature(sync_member)
        async_signature = inspect.signature(async_member)
        assert tuple(sync_signature.parameters) == tuple(async_signature.parameters)
        for parameter_name in sync_signature.parameters:
            sync_parameter = sync_signature.parameters[parameter_name]
            async_parameter = async_signature.parameters[parameter_name]
            assert async_parameter.kind is sync_parameter.kind
            assert async_parameter.default == sync_parameter.default

        sync_hints = get_type_hints(sync_member)
        async_hints = get_type_hints(async_member)
        assert async_hints == sync_hints
        if 'path' in sync_hints:
            assert sync_hints['path'] is str


def test_fake_async_find_records_explicit_defaults(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
) -> None:
    kwargs: _FindDefaultKwargs = {
        'order': None,
        'filter_operator': None,
        'limit': 0,
        'retired_only': False,
        'page': 0,
        'include_archived_projects': True,
        'additional_filter_presets': None,
    }
    asyncio.run(fake_async_sg.find('Shot', [], ['id'], **kwargs))
    fake_sg.find('Shot', [], ['id'], **kwargs)
    expected: list[tuple[str, tuple[Any, ...], _FindDefaultKwargs]] = [
        ('find', ('Shot', [], ['id']), kwargs)
    ]
    assert fake_async_sg._calls == expected
    assert fake_sg._calls == expected


def test_fake_async_shotgun_blocks_deterministically(fake_async_sg: _FakeAsyncShotgun) -> None:
    async def scenario() -> None:
        entered, release = fake_async_sg._block('find')
        task = asyncio.create_task(fake_async_sg.find('Shot', [], ['id']))
        await entered.wait()
        assert not task.done()
        release.set()
        await task

    asyncio.run(scenario())


def test_fake_async_shotgun_matches_sync_call_transcript(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
) -> None:
    asyncio.run(fake_async_sg.find('Shot', [], ['id']))
    fake_sg.find('Shot', [], ['id'])
    asyncio.run(fake_async_sg.summarize('Shot', [], [{'field': 'id', 'type': 'count'}]))
    fake_sg.summarize('Shot', [], [{'field': 'id', 'type': 'count'}])
    assert fake_async_sg._calls == fake_sg._calls


def test_fake_async_shotgun_records_awaited_calls(fake_async_sg: _FakeAsyncShotgun) -> None:
    async def scenario() -> None:
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A'}]]
        rows = await fake_async_sg.find('Shot', [], ['id', 'code'])
        assert rows[0]['code'] == 'A'
        assert fake_async_sg._calls[-1][0] == 'find'

    asyncio.run(scenario())


def test_fake_async_shotgun_satisfies_protocol(fake_async_sg: _FakeAsyncShotgun) -> None:
    assert isinstance(fake_async_sg, AsyncShotgunLike)
    assert isinstance(fake_async_sg, AsyncClosableShotgunLike)


def test_fake_async_summarize_records_explicit_defaults(
    fake_async_sg: _FakeAsyncShotgun,
    fake_sg: _FakeShotgun,
) -> None:
    kwargs: _SummarizeDefaultKwargs = {
        'filter_operator': None,
        'grouping': None,
        'include_archived_projects': True,
    }
    summary_fields = [{'field': 'id', 'type': 'count'}]
    asyncio.run(fake_async_sg.summarize('Shot', [], summary_fields, **kwargs))
    fake_sg.summarize('Shot', [], summary_fields, **kwargs)
    expected: list[tuple[str, tuple[Any, ...], _SummarizeDefaultKwargs]] = [
        ('summarize', ('Shot', [], summary_fields), kwargs)
    ]
    assert fake_async_sg._calls == expected
    assert fake_sg._calls == expected


def test_fake_sync_shotgun_close_marks_closed_and_raises_canned_errors() -> None:
    fake = _FakeShotgun()
    assert fake._closed is False
    fake.close()
    assert fake._closed is True
    failing = _FakeShotgun()
    failing._close_errors.append(RuntimeError('close failed'))
    with pytest.raises(RuntimeError, match='close failed'):
        failing.close()
    assert failing._closed is False
