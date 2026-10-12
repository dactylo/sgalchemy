import asyncio
import copy
import datetime
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Never, ParamSpec, TypeVar, cast

import pytest
from _fakes import _FakeAsyncShotgun
from _support import YieldCountingEntities, _AsyncCallback, isolated_async_engine

from sgalchemy._core._async_engine import create_async_engine
from sgalchemy._core._options import joinedload, lazyload, selectinload
from sgalchemy._core._registry import Entity, Registry, RegistryBinding, entity_state
from sgalchemy._core._statement import select
from sgalchemy._errors import CompilationError, RelationshipTargetUnavailableError, SgalchemyError
from sgalchemy._orm._async_session import AsyncSession
from sgalchemy._shotgun import WireRecord, WireRecords

_P = ParamSpec('_P')
_R = TypeVar('_R')


def _find_calls(
    fake_async_sg: _FakeAsyncShotgun,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    return [call for call in fake_async_sg._calls if call[0] == 'find']


def _replace_call(
    _call: _AsyncCallback[_P, _R],
    replacement: Callable[[str, tuple[object, ...], dict[str, object]], Awaitable[_R]],
) -> _AsyncCallback[_P, _R]:
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        entity_type = cast('str', kwargs['entity_type'] if 'entity_type' in kwargs else args[0])
        return await replacement(entity_type, tuple(args), dict(kwargs))

    return wrapper


def _track_call(call: _AsyncCallback[_P, _R]) -> tuple[_AsyncCallback[_P, _R], list[None]]:
    captures: list[None] = []

    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        captures.append(None)
        return await call(*args, **kwargs)

    return wrapper, captures


def test_async_explicit_load_fetches_active_and_retired_parents(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        active = session.merge_dict({'type': 'Shot', 'id': 1})
        retired = session.merge_dict({'type': 'Shot', 'id': 2})
        entity_state(retired).deleted = True
        original_find = fake_async_sg.find

        async def lifecycle_find(
            entity_type: str,
            filters: list[Any],
            fields: list[str] | None = None,
            **kwargs: object,
        ) -> WireRecords:
            if entity_type == 'Shot':
                parent_id = 2 if kwargs.get('retired_only') else 1
                fake_async_sg._find_results = [
                    [
                        {
                            'type': 'Shot',
                            'id': parent_id,
                            'project': {'type': 'Project', 'id': 10},
                        }
                    ]
                ]
            else:
                fake_async_sg._find_results = [
                    [{'type': 'Project', 'id': 10, 'name': 'Project One'}]
                ]
            return await original_find(
                entity_type, filters, fields, retired_only=bool(kwargs.get('retired_only', False))
            )

        monkeypatch.setattr(fake_async_sg, 'find', lifecycle_find)
        await session.load((obj for obj in [active, retired]), engine.entities.Shot.project)

        assert active.project is retired.project
        assert retired.project.name == 'Project One'
        assert entity_state(retired).deleted
        calls = _find_calls(fake_async_sg)
        assert [(call[1][1], call[2].get('retired_only', False)) for call in calls[:2]] == [
            ([['id', 'in', [1]]], False),
            ([['id', 'in', [2]]], True),
        ]
        assert len(calls) == 3

    asyncio.run(scenario())


def test_async_explicit_load_keeps_operation_owned_while_awaiting(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict(
            {'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 10}]}
        )
        entered, release = fake_async_sg._block('find')
        fake_async_sg._find_results = [[{'type': 'Asset', 'id': 10, 'code': 'A'}]]

        loading = asyncio.create_task(session.load(shot, shot_cls.assets))
        await entered.wait()
        with pytest.raises(SgalchemyError, match='concurrent task'):
            session.expire(shot)
        release.set()
        await loading

        assert shot.assets[0].code == 'A'

    asyncio.run(scenario())


def test_async_explicit_load_recovers_unknown_target_without_repeating_parent(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict({'type': 'Shot', 'id': 1})
        refreshed_raw = copy.deepcopy(schema_fixture)
        refreshed_raw['FutureVendor'] = copy.deepcopy(refreshed_raw['CustomEntity03'])
        opaque_before_refresh: Entity | None = None
        opaque_cls_before_refresh: type[Entity] | None = None

        async def refreshed_schema_read() -> dict[str, Any]:
            nonlocal opaque_before_refresh, opaque_cls_before_refresh
            await fake_async_sg._record('schema_read')
            opaque_before_refresh = cast(
                'Entity',
                entity_state(shot).loaded['sg_vendor'],
            )
            opaque_cls_before_refresh = type(opaque_before_refresh)
            return refreshed_raw

        monkeypatch.setattr(fake_async_sg, 'schema_read', refreshed_schema_read)
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'sg_vendor': {'type': 'FutureVendor', 'id': 9},
                }
            ],
            [{'type': 'FutureVendor', 'id': 9, 'code': 'FV-9'}],
        ]

        await session.load(shot, shot_cls.sg_vendor)

        assert shot.sg_vendor is opaque_before_refresh
        assert type(shot.sg_vendor) is opaque_cls_before_refresh
        assert engine.registry.entity_class('FutureVendor') is opaque_cls_before_refresh
        assert shot.sg_vendor.code == 'FV-9'
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == [
            'Shot',
            'FutureVendor',
        ]

    asyncio.run(scenario())


def test_async_explicit_load_rejects_foreign_objects_and_wrong_root(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'one',
            registry=Registry(),
            bind_default=False,
        )
        other = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'two',
            registry=Registry(),
            bind_default=False,
        )
        await engine.load_schema()
        await other.load_schema()
        session = AsyncSession(engine)
        asset_cls = engine.entities.Asset
        shot_cls = engine.entities.Shot
        own = session.merge_dict({'type': 'Shot', 'id': 1, 'assets': []})
        foreign = AsyncSession(other).merge_dict({'type': 'Shot', 'id': 2, 'assets': []})
        foreign_asset = AsyncSession(other).merge_dict({'type': 'Asset', 'id': 3, 'shots': []})
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(SgalchemyError, match='different registry'):
            await session.load(foreign, shot_cls.assets)
        with pytest.raises(SgalchemyError, match='field root'):
            await session.load(own, asset_cls.shots)
        with pytest.raises(SgalchemyError, match='different registry'):
            await session.load([own, foreign_asset], asset_cls.shots)
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('field_name', 'parent_row'),
    [
        (
            'deep',
            {'type': 'Shot', 'id': 1, 'project.Project.name': 'Project One'},
        ),
        ('scalar', {'type': 'Shot', 'id': 1, 'code': 'shot-1'}),
    ],
)
def test_async_explicit_load_rejects_invalid_fields_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    field_name: str,
    parent_row: dict[str, Any],
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict({'type': 'Shot', 'id': 1})
        field = shot_cls.project.name if field_name == 'deep' else shot_cls.code
        fake_async_sg._find_results = [[parent_row]]
        schema_reads_before = [call for call in fake_async_sg._calls if call[0] == 'schema_read']

        with pytest.raises(SgalchemyError) as excinfo:
            await session.load(shot, field)
        await engine.load_schema()

        state = entity_state(shot)
        assert _find_calls(fake_async_sg) == []
        assert state.loaded == {'id': 1}
        assert all('.' not in key for key in state.loaded)
        assert [call for call in fake_async_sg._calls if call[0] == 'schema_read'] == (
            schema_reads_before
        )
        assert isinstance(excinfo.value, CompilationError)

    asyncio.run(scenario())


def test_async_explicit_load_uses_one_binding_for_complete_batch(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        initial_schema = copy.deepcopy(schema_fixture)
        initial_schema['Asset']['code']['data_type']['value'] = 'date'
        initial_schema['Shot']['code']['data_type']['value'] = 'date'

        async def read_initial_schema() -> dict[str, Any]:
            return copy.deepcopy(initial_schema)

        monkeypatch.setattr(fake_async_sg, 'schema_read', read_initial_schema)
        registry = Registry()
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'first',
            registry=registry,
            bind_default=False,
        )
        await engine.load_schema()
        session = AsyncSession(engine)
        version_cls = engine.entities.Version
        versions = [
            session.merge_dict({'type': 'Version', 'id': 1, 'entity': {'type': 'Shot', 'id': 10}}),
            session.merge_dict(
                {'type': 'Version', 'id': 2, 'entity': {'type': 'Asset', 'id': 20}}
            ),
        ]
        fake_async_sg._find_results = [
            [{'type': 'Asset', 'id': 20, 'code': '2026-07-24'}],
            [{'type': 'Shot', 'id': 10, 'code': '2026-07-25'}],
        ]
        first_target_entered = asyncio.Event()
        release_first_target = asyncio.Event()
        original_find = fake_async_sg.find

        def gate_call(call: _AsyncCallback[_P, _R]) -> _AsyncCallback[_P, _R]:
            async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                entity_type = cast(
                    'str', kwargs['entity_type'] if 'entity_type' in kwargs else args[0]
                )
                records = await call(*args, **kwargs)
                if entity_type == 'Asset':
                    first_target_entered.set()
                    await release_first_target.wait()
                return records

            return wrapper

        gated_find = gate_call(original_find)

        monkeypatch.setattr(fake_async_sg, 'find', gated_find)
        original_load_binding = engine.load_binding
        tracking_load_binding, captures = _track_call(original_load_binding)

        monkeypatch.setattr(engine, 'load_binding', tracking_load_binding)
        loading = asyncio.create_task(session.load(versions, version_cls.entity))
        await first_target_entered.wait()

        refresher_sg = type(fake_async_sg)()
        refresher = create_async_engine(
            connection=refresher_sg,
            cache_root=tmp_path / 'refresher',
            registry=registry,
            bind_default=False,
        )
        await refresher.load_schema()
        release_first_target.set()
        await loading

        assert versions[0].entity.code == datetime.date(2026, 7, 25)
        assert versions[1].entity.code == datetime.date(2026, 7, 24)
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == ['Asset', 'Shot']
        assert len(captures) == 1

    asyncio.run(scenario())


def test_async_explicit_load_valid_direct_relationship_control(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict({'type': 'Shot', 'id': 1})
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'project': {'type': 'Project', 'id': 10},
                }
            ],
            [{'type': 'Project', 'id': 10, 'name': 'Project One'}],
        ]

        await session.load(shot, shot_cls.project)

        assert shot.project.name == 'Project One'
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == ['Shot', 'Project']

    asyncio.run(scenario())


def test_async_lazyload_rejected_before_io(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(SgalchemyError, match='cannot await') as excinfo:
            await session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))
        assert 'selectinload()' in str(excinfo.value)
        assert 'await session.load(...)' in str(excinfo.value)
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_load_fetches_only_absent_relationships(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        absent = session.merge_dict({'type': 'Shot', 'id': 1})
        empty = session.merge_dict({'type': 'Shot', 'id': 2, 'assets': []})
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 10}]}],
            [{'type': 'Asset', 'id': 10, 'code': 'A'}],
        ]

        result = await session.load([absent, empty], shot_cls.assets)

        assert result is None
        assert absent.assets[0].code == 'A'
        assert empty.assets == []
        assert _find_calls(fake_async_sg)[0][1][1] == [['id', 'in', [1]]]

    asyncio.run(scenario())


def test_async_load_pending_only_targets_capture_schema_without_record_io(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        asset_cls = engine.entities.Asset
        pending_asset = asset_cls.new(code='pending')
        shot = session.merge_dict({'type': 'Shot', 'id': 2, 'assets': []})
        shot.assets = [pending_asset]
        calls_before = list(fake_async_sg._calls)
        original_load_binding = engine.load_binding
        tracking_load_binding, captures = _track_call(original_load_binding)

        monkeypatch.setattr(engine, 'load_binding', tracking_load_binding)

        await session.load(shot, shot_cls.assets)

        assert len(captures) == 1
        assert shot.assets == [pending_asset]
        assert entity_state(shot).dirty == {'assets': [pending_asset]}
        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())


def test_async_load_present_empty_values_capture_one_schema(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        empty_list = session.merge_dict({'type': 'Shot', 'id': 1, 'assets': []})
        empty_none = session.merge_dict({'type': 'Shot', 'id': 2, 'assets': None})
        calls_before = list(fake_async_sg._calls)
        original_load_binding = engine.load_binding
        tracking_load_binding, captures = _track_call(original_load_binding)

        monkeypatch.setattr(engine, 'load_binding', tracking_load_binding)

        await session.load([], shot_cls.assets)
        assert len(captures) == 0
        await session.load([empty_list, empty_none], shot_cls.assets)

        assert len(captures) == 1
        assert empty_list.assets == []
        assert empty_none.assets is None
        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'value',
    [
        {'type': 'Asset', 'id': 2},
        [{'type': 'Asset', 'id': 2}, 'corrupt'],
    ],
)
def test_async_load_rejects_corrupt_link_shapes_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    value: object,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict({'type': 'Shot', 'id': 1, 'assets': value})
        calls_before = list(fake_async_sg._calls)

        with pytest.raises(SgalchemyError, match=r'Shot\.assets.*Entity'):
            await session.load(shot, shot_cls.assets)

        assert _find_calls(fake_async_sg) == []
        assert fake_async_sg._calls[: len(calls_before)] == calls_before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    'parent_rows',
    [[], [{'type': 'Shot', 'id': 1}]],
    ids=['missing-parent-row', 'omitted-relationship-field'],
)
def test_async_load_rejects_incomplete_parent_response(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    parent_rows: list[dict[str, Any]],
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'parent'})
        fake_async_sg._find_results = [parent_rows]

        with pytest.raises(
            SgalchemyError,
            match=r'Shot\.assets.*incomplete provider response',
        ):
            await session.load(shot, shot_cls.assets)

        state = entity_state(shot)
        assert state.loaded == {'id': 1, 'code': 'parent'}
        assert state.dirty == {}
        find_calls = _find_calls(fake_async_sg)
        assert len(find_calls) == 1
        assert find_calls[0][1][0] == 'Shot'

    asyncio.run(scenario())


def test_async_load_rejects_pending_parent_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        persistent = session.merge_dict({'type': 'Shot', 'id': 1})
        pending = shot_cls.new(code='pending')
        session.add(pending)
        calls_before = list(fake_async_sg._calls)

        with pytest.raises(SgalchemyError, match='without a server identity'):
            await session.load([persistent, pending], shot_cls.assets)

        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())


def test_async_load_retries_missing_targets_after_new_binding(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        note_cls = engine.entities.Note
        note = session.merge_dict(
            {
                'type': 'Note',
                'id': 1,
                'note_links': [
                    {'type': 'FutureB', 'id': 20},
                    {'type': 'FutureA', 'id': 10},
                ],
            }
        )
        links = list(entity_state(note).loaded['note_links'])

        with pytest.raises(
            RelationshipTargetUnavailableError,
            match=r'Note\.note_links.*FutureA.*FutureB.*valid_types.*Shot.*Version',
        ):
            await session.load(note, note_cls.note_links)

        assert all(
            current is original
            for current, original in zip(
                entity_state(note).loaded['note_links'],
                links,
                strict=True,
            )
        )
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert _find_calls(fake_async_sg) == []

        failed = engine.prepared_binding()
        assert failed is not None
        future_a = replace(
            failed.schemas['Sequence'],
            type_name='FutureA',
            display_name='Future A',
        )
        future_b = replace(
            failed.schemas['CustomEntity03'],
            type_name='FutureB',
            display_name='Future B',
        )
        engine.registry.bind(
            {**failed.schemas, 'FutureA': future_a, 'FutureB': future_b},
            key=failed.key,
        )
        fake_async_sg._find_results = [
            [{'type': 'FutureA', 'id': 10, 'code': 'A'}],
            [{'type': 'FutureB', 'id': 20, 'code': 'B'}],
        ]

        await session.load(note, note_cls.note_links)

        assert all(
            current is original for current, original in zip(note.note_links, links, strict=True)
        )
        assert [link.code for link in note.note_links] == ['B', 'A']
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == [
            'FutureA',
            'FutureB',
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize('other_strategy', ['joined', 'selectin'])
def test_async_mixed_lazyload_rejected_before_normalization(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    other_strategy: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        if other_strategy == 'joined':
            options = (joinedload(shot_cls.assets), lazyload(shot_cls.assets))
        else:
            options = (lazyload(shot_cls.assets), selectinload(shot_cls.assets))
        calls_before = list(fake_async_sg._calls)

        async def fail_load_binding(*, force: bool = False) -> Never:
            del force
            raise AssertionError('mixed lazy loading must fail before schema capture')

        monkeypatch.setattr(engine, 'load_binding', fail_load_binding)

        with pytest.raises(SgalchemyError, match='cannot await') as excinfo:
            await session.scalars(select(shot_cls).options(*options))

        assert 'selectinload()' in str(excinfo.value)
        assert 'await session.load(...)' in str(excinfo.value)
        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())


@pytest.mark.parametrize('explicit', [False, True])
def test_async_loader_responses_use_separate_staged_identity_scopes(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path, explicit: bool
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        objects = [session.merge_dict({'type': 'Shot', 'id': i + 1}) for i in range(20)]
        staged = [session.merge_dict({'type': 'Shot', 'id': 1000 + i}) for i in range(20)]
        for obj in staged:
            session.delete(obj)
        queue = YieldCountingEntities(session.unit_of_work._deleted)
        session.unit_of_work._deleted = queue
        fake_async_sg._find_results = [
            [
                {'type': 'Shot', 'id': i + 1, 'assets': [{'type': 'Asset', 'id': i + 1}]}
                for i in range(20)
            ],
            [{'type': 'Asset', 'id': i + 1, 'code': f'asset-{i}'} for i in range(20)],
        ]
        if explicit:
            await session.load(objects, engine.entities.Shot.assets)
        else:
            returned = await session.scalars(
                select(engine.entities.Shot).options(selectinload(engine.entities.Shot.assets))
            )
            assert all(left is right for left, right in zip(returned, objects, strict=True))
        assert objects[0].assets[0].code == 'asset-0'
        assert objects[-1].assets[0].code == 'asset-19'
        assert queue._visits == 40
        assert session._state._hydration_index is None

    asyncio.run(scenario())


def test_async_multiple_selectins_preflight_union_before_secondary_io(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        refreshed_raw = copy.deepcopy(schema_fixture)
        refreshed_raw['FutureA'] = copy.deepcopy(refreshed_raw['Sequence'])

        async def refreshed_schema_read() -> dict[str, Any]:
            await fake_async_sg._record('schema_read')
            return refreshed_raw

        monkeypatch.setattr(fake_async_sg, 'schema_read', refreshed_schema_read)
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'sg_sequence': {'type': 'FutureA', 'id': 8},
                    'sg_vendor': {'type': 'FutureB', 'id': 9},
                }
            ]
        ]
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot

        with pytest.raises(
            RelationshipTargetUnavailableError,
            match=r'Shot\.sg_vendor.*FutureB',
        ):
            await session.scalars(
                select(shot_cls).options(
                    selectinload(shot_cls.sg_sequence),
                    selectinload(shot_cls.sg_vendor),
                )
            )

        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == ['Shot']

    asyncio.run(scenario())


def test_async_multiple_selectins_sort_distinct_target_order(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Version',
                    'id': 1,
                    'entity': {'type': 'Shot', 'id': 10},
                    'user': {'type': 'HumanUser', 'id': 1},
                }
            ],
            [{'type': 'HumanUser', 'id': 1, 'name': 'B'}],
            [{'type': 'Shot', 'id': 10, 'code': 'A'}],
        ]
        session = AsyncSession(engine)
        version_cls = engine.entities.Version

        version = (
            await session.scalars(
                select(version_cls).options(
                    selectinload(version_cls.entity),
                    selectinload(version_cls.user),
                )
            )
        )[0]

        assert [call[1][0] for call in _find_calls(fake_async_sg)] == [
            'Version',
            'HumanUser',
            'Shot',
        ]
        assert (version.entity.code, version.user.name) == ('A', 'B')

    asyncio.run(scenario())


def test_async_multiple_selectins_union_shared_target_reads(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        schema_fixture['Version']['reviewer'] = copy.deepcopy(schema_fixture['Version']['user'])

        async def schema_read() -> dict[str, Any]:
            await fake_async_sg._record('schema_read')
            return copy.deepcopy(schema_fixture)

        monkeypatch.setattr(fake_async_sg, 'schema_read', schema_read)
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Version',
                    'id': 1,
                    'user': {'type': 'HumanUser', 'id': 2},
                    'reviewer': {'type': 'HumanUser', 'id': 1},
                },
                {
                    'type': 'Version',
                    'id': 2,
                    'user': {'type': 'HumanUser', 'id': 1},
                    'reviewer': {'type': 'HumanUser', 'id': 2},
                },
            ],
            [
                {'type': 'HumanUser', 'id': 1, 'name': 'A'},
                {'type': 'HumanUser', 'id': 2, 'name': 'B'},
            ],
        ]
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        version_cls = engine.entities.Version

        versions = await session.scalars(
            select(version_cls).options(
                selectinload(version_cls.user),
                selectinload(version_cls.reviewer),
            )
        )

        target_calls = [call for call in _find_calls(fake_async_sg) if call[1][0] == 'HumanUser']
        assert len(target_calls) == 1
        assert target_calls[0][1][1] == [['id', 'in', [1, 2]]]
        assert [(version.user.name, version.reviewer.name) for version in versions] == [
            ('B', 'A'),
            ('A', 'B'),
        ]

    asyncio.run(scenario())


def test_async_scalars_keeps_entry_binding_through_selectin_refresh(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        initial_schema = copy.deepcopy(schema_fixture)
        initial_schema['Asset']['code']['data_type']['value'] = 'date'

        async def read_initial_schema() -> dict[str, Any]:
            return copy.deepcopy(initial_schema)

        monkeypatch.setattr(fake_async_sg, 'schema_read', read_initial_schema)
        registry = Registry()
        engine = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'first',
            registry=registry,
            bind_default=False,
        )
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 10}]}],
            [{'type': 'Asset', 'id': 10, 'code': '2026-07-25'}],
        ]
        entered, release = fake_async_sg._block('find')

        loading = asyncio.create_task(
            session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))
        )
        await entered.wait()
        refresher_sg = type(fake_async_sg)()
        refresher = create_async_engine(
            connection=refresher_sg,
            cache_root=tmp_path / 'refresher',
            registry=registry,
            bind_default=False,
        )
        await refresher.load_schema()
        release.set()
        shots = await loading

        assert shots[0].assets[0].code == datetime.date(2026, 7, 25)
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == ['Shot', 'Asset']

    asyncio.run(scenario())


def test_async_selectin_and_explicit_load(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 10}]}],
            [{'type': 'Asset', 'id': 10, 'code': 'A'}],
            [{'type': 'Shot', 'id': 2, 'assets': [{'type': 'Asset', 'id': 20}]}],
            [{'type': 'Asset', 'id': 20, 'code': 'B'}],
        ]
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        eager = await session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))
        assert eager[0].assets[0].code == 'A'
        plain = await session.scalars(select(shot_cls))
        await session.load(plain, shot_cls.assets)
        assert plain[0].assets[0].code == 'B'

    asyncio.run(scenario())


def test_async_selectin_concurrent_target_cancellation_cancels_reads(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        version_cls = engine.entities.Version
        target_types = ('Asset', 'HumanUser', 'Shot')
        entered = {target_type: asyncio.Event() for target_type in target_types}
        cancelled = {target_type: asyncio.Event() for target_type in target_types}
        original_find = fake_async_sg.find

        async def replace_find(
            entity_type: str,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> WireRecords:
            await fake_async_sg._record('find', *args, **kwargs)
            if entity_type == 'Version':
                return [
                    {
                        'type': 'Version',
                        'id': 1,
                        'entity': {'type': 'Shot', 'id': 10},
                        'user': {'type': 'HumanUser', 'id': 1},
                    },
                    {
                        'type': 'Version',
                        'id': 2,
                        'entity': {'type': 'Asset', 'id': 20},
                        'user': {'type': 'HumanUser', 'id': 1},
                    },
                ]
            entered[entity_type].set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled[entity_type].set()
                raise
            raise AssertionError('blocked target read completed without cancellation')

        blocked_find = _replace_call(original_find, replace_find)

        monkeypatch.setattr(fake_async_sg, 'find', blocked_find)
        loading = asyncio.create_task(
            session.scalars(
                select(version_cls).options(
                    selectinload(version_cls.entity),
                    selectinload(version_cls.user),
                )
            )
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in entered.values())),
                timeout=1,
            )
        except BaseException:
            loading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loading
            raise

        loading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loading

        assert all(event.is_set() for event in cancelled.values())

    asyncio.run(scenario())


def test_async_selectin_concurrent_targets_hydrate_in_owner_task(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        version_cls = engine.entities.Version
        target_types = ('Asset', 'HumanUser', 'Shot')
        entered = {target_type: asyncio.Event() for target_type in target_types}
        release = {target_type: asyncio.Event() for target_type in target_types}
        target_rows = {
            'Asset': [{'type': 'Asset', 'id': 20, 'code': 'asset'}],
            'HumanUser': [{'type': 'HumanUser', 'id': 1, 'name': 'user'}],
            'Shot': [{'type': 'Shot', 'id': 10, 'code': 'shot'}],
        }
        original_find = fake_async_sg.find

        async def replace_find(
            entity_type: str,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> WireRecords:
            await fake_async_sg._record('find', *args, **kwargs)
            if entity_type == 'Version':
                return [
                    {
                        'type': 'Version',
                        'id': 1,
                        'entity': {'type': 'Shot', 'id': 10},
                        'user': {'type': 'HumanUser', 'id': 1},
                    },
                    {
                        'type': 'Version',
                        'id': 2,
                        'entity': {'type': 'Asset', 'id': 20},
                        'user': {'type': 'HumanUser', 'id': 1},
                    },
                ]
            entered[entity_type].set()
            await release[entity_type].wait()
            return target_rows[entity_type]

        gated_find = _replace_call(original_find, replace_find)

        monkeypatch.setattr(fake_async_sg, 'find', gated_find)
        owner_hydrations: list[tuple[str, asyncio.Task[Any] | None]] = []
        original_hydrate_record = session.hydrate_record

        def tracking_hydrate_record(
            entity_type_name: str,
            record: WireRecord,
            binding: RegistryBinding | None,
        ) -> Entity:
            if entity_type_name in target_types:
                owner_hydrations.append((entity_type_name, asyncio.current_task()))
            return original_hydrate_record(entity_type_name, record, binding)

        monkeypatch.setattr(session, 'hydrate_record', tracking_hydrate_record)
        loading = asyncio.create_task(
            session.scalars(
                select(version_cls).options(
                    selectinload(version_cls.entity),
                    selectinload(version_cls.user),
                )
            )
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in entered.values())),
                timeout=1,
            )
        finally:
            for event in release.values():
                event.set()
            await loading

        assert [call[1][0] for call in _find_calls(fake_async_sg)] == [
            'Version',
            *target_types,
        ]
        assert [target_type for target_type, _task in owner_hydrations] == list(target_types)
        assert all(task is loading for _target_type, task in owner_hydrations)

    asyncio.run(scenario())


def test_async_selectin_earlier_success_hydrates_before_ordered_target_failure(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        version_cls = engine.entities.Version
        asset = session.merge_dict({'type': 'Asset', 'id': 20})
        fault = RuntimeError('shot failed')
        shot_finished = asyncio.Event()
        original_find = fake_async_sg.find

        async def replace_find(
            entity_type: str,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> WireRecords:
            await fake_async_sg._record('find', *args, **kwargs)
            if entity_type == 'Version':
                return [
                    {'type': 'Version', 'id': 1, 'entity': {'type': 'Shot', 'id': 10}},
                    {'type': 'Version', 'id': 2, 'entity': {'type': 'Asset', 'id': 20}},
                ]
            if entity_type == 'Shot':
                shot_finished.set()
                raise fault
            await shot_finished.wait()
            return [{'type': 'Asset', 'id': 20, 'code': 'asset'}]

        later_failure_finishes_first = _replace_call(original_find, replace_find)

        monkeypatch.setattr(fake_async_sg, 'find', later_failure_finishes_first)

        with pytest.raises(RuntimeError) as excinfo:
            await asyncio.wait_for(
                session.scalars(select(version_cls).options(selectinload(version_cls.entity))),
                timeout=1,
            )

        assert excinfo.value is fault
        assert asset.code == 'asset'

    asyncio.run(scenario())


def test_async_selectin_ordered_target_failure_ignores_reverse_completion(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        version_cls = engine.entities.Version
        first_fault = RuntimeError('asset failed')
        later_fault = RuntimeError('shot failed')
        shot_finished = asyncio.Event()
        completions: list[str] = []
        original_find = fake_async_sg.find

        async def replace_find(
            entity_type: str,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> WireRecords:
            await fake_async_sg._record('find', *args, **kwargs)
            if entity_type == 'Version':
                return [
                    {'type': 'Version', 'id': 1, 'entity': {'type': 'Shot', 'id': 10}},
                    {'type': 'Version', 'id': 2, 'entity': {'type': 'Asset', 'id': 20}},
                ]
            if entity_type == 'Shot':
                completions.append(entity_type)
                shot_finished.set()
                raise later_fault
            await shot_finished.wait()
            completions.append(entity_type)
            raise first_fault

        reversed_failures = _replace_call(original_find, replace_find)

        monkeypatch.setattr(fake_async_sg, 'find', reversed_failures)

        with pytest.raises(RuntimeError) as excinfo:
            await asyncio.wait_for(
                session.scalars(select(version_cls).options(selectinload(version_cls.entity))),
                timeout=1,
            )

        assert excinfo.value is first_fault
        assert completions == ['Shot', 'Asset']
        assert any(
            'raised while executing' in note and 'Asset' in note
            for note in getattr(first_fault, '__notes__', [])
        )

    asyncio.run(scenario())


def test_async_selectin_recovers_unknown_target_after_one_force_refresh(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        refreshed_raw = copy.deepcopy(schema_fixture)
        refreshed_raw['FutureVendor'] = copy.deepcopy(refreshed_raw['CustomEntity03'])

        async def refreshed_schema_read() -> dict[str, Any]:
            await fake_async_sg._record('schema_read')
            return refreshed_raw

        monkeypatch.setattr(fake_async_sg, 'schema_read', refreshed_schema_read)
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'sg_vendor': {'type': 'FutureVendor', 'id': 9},
                }
            ],
            [{'type': 'FutureVendor', 'id': 9, 'code': 'FV-9'}],
        ]
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot

        shot = (await session.scalars(select(shot_cls).options(selectinload(shot_cls.sg_vendor))))[
            0
        ]

        opaque = shot.sg_vendor
        assert opaque.code == 'FV-9'
        assert session.merge_dict({'type': 'FutureVendor', 'id': 9}) is opaque
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert [call[1][0] for call in _find_calls(fake_async_sg)] == [
            'Shot',
            'FutureVendor',
        ]

    asyncio.run(scenario())


def test_async_selectin_rejects_returned_exception_payload(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        returned_error = RuntimeError('provider returned this object')
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'assets': [{'type': 'Asset', 'id': 10}],
                }
            ],
            returned_error,
        ]

        with pytest.raises(SgalchemyError, match='find payload') as excinfo:
            await session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))

        assert excinfo.value is not returned_error

    asyncio.run(scenario())


def test_readme_explicit_async_relationship_load_call(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        shot_cls = engine.entities.Shot
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'assets': [{'type': 'Asset', 'id': 10}],
                }
            ],
            [{'type': 'Asset', 'id': 10, 'code': 'asset-a'}],
        ]

        async with AsyncSession(engine) as session:
            shot = session.merge_dict({'type': 'Shot', 'id': 1})
            await session.load(shot, shot_cls.assets)

            assert shot.assets[0].code == 'asset-a'

        await engine.dispose()

    asyncio.run(scenario())
