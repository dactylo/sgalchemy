from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any, Never, ParamSpec, TypeVar, cast

import pytest
from _support import (
    AUTHENTICATION_FAULT,
    SCHEMA_ABSENT_MESSAGE,
    BrokenNoteError,
    YieldCountingEntities,
    _preserve_signature,
    capture_hydration_bindings,
    isolated_async_engine,
)

from sgalchemy import inspect
from sgalchemy._core._async_engine import AsyncEngine, create_async_engine
from sgalchemy._core._compiler import FindCall
from sgalchemy._core._options import joinedload
from sgalchemy._core._registry import Entity, Registry, RegistryBinding, entity_state
from sgalchemy._core._result import Result
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import CommitError, CompilationError, SchemaUnavailableError, SgalchemyError
from sgalchemy._orm import _async_session as async_session_module
from sgalchemy._orm._async_session import AsyncSession

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from _fakes import _FakeAsyncShotgun
    from _support import _SyncCallback

    from sgalchemy._core._compiler import SummarizeCall
    from sgalchemy._orm._uow import CommitPlan
    from sgalchemy.schema._model import EntitySchema

_P = ParamSpec('_P')
_R = TypeVar('_R')


async def _async_engine_with_symbolic_vendor(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> tuple[AsyncEngine, type[Entity]]:
    registry = Registry()
    vendor_cls = registry.entity_class('FutureVendor')
    engine = create_async_engine(
        connection=fake_async_sg,
        cache_root=tmp_path,
        registry=registry,
        bind_default=False,
    )
    await engine.load_schema()
    return engine, vendor_cls


def test_async_auth_upload_failure_notes_unknown_outcome_on_commit_error(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 9, 'image': 'old-image'})
        image_path = tmp_path / 'async-auth-fail.jpg'
        image_path.write_bytes(b'image')
        shot.image = image_path
        fault = AUTHENTICATION_FAULT('session expired')
        fake_async_sg._errors.setdefault('upload_thumbnail', []).append(fault)

        with pytest.raises(CommitError) as excinfo:
            await session.commit()

        joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
        assert excinfo.value.__cause__ is fault
        assert 'upload outcome' in joined
        assert 'session.refresh' in joined
        assert 'remain staged' not in joined

    asyncio.run(scenario())


def test_async_broken_add_note_does_not_mask_provider_error(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.registry.entity_class('Shot')
        error = BrokenNoteError('provider unavailable')
        fake_async_sg._errors.setdefault('find', []).append(error)

        with pytest.raises(BrokenNoteError) as excinfo:
            await session.scalars(select(shot_cls))

        assert excinfo.value is error

    asyncio.run(scenario())


def test_async_cold_stale_disk_schema_forces_one_refresh_before_commit(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        stale_sg = type(fake_async_sg)()
        stale_schema = {name: fields for name, fields in schema_fixture.items() if name != 'Shot'}

        async def stale_schema_read() -> dict[str, Any]:
            await stale_sg._record('schema_read')
            return copy.deepcopy(stale_schema)

        monkeypatch.setattr(stale_sg, 'schema_read', stale_schema_read)
        warmer = create_async_engine(
            connection=stale_sg,
            cache_root=tmp_path,
            registry=Registry(),
            bind_default=False,
        )
        await warmer.load_schema()
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 33})
        session.delete(shot)
        original_load = engine.load_binding
        force_values: list[bool] = []

        @_preserve_signature(original_load)
        async def tracking_load(*, force: bool = False) -> RegistryBinding:
            force_values.append(force)
            return await original_load(force=force)

        monkeypatch.setattr(engine, 'load_binding', tracking_load)

        await session.commit()

        assert force_values == [False, True]
        names = [name for name, _, _ in fake_async_sg._calls]
        assert names.count('schema_read') == 1
        assert names.count('schema_entity_read') == 1
        assert names.count('batch') == 1
        assert entity_state(shot).deleted

    asyncio.run(scenario())


def test_async_commit_schema_recovery_forces_once_when_schema_remains_invalid(
    caplog: pytest.LogCaptureFixture,
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        complete = engine.prepared_binding()
        assert complete is not None
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 32})
        session.delete(shot)
        engine.registry.bind(
            {name: schema for name, schema in complete.schemas.items() if name != 'Shot'},
            key=complete.key,
        )
        stale_raw = {name: schema for name, schema in schema_fixture.items() if name != 'Shot'}

        async def stale_schema_read() -> dict[str, Any]:
            await fake_async_sg._record('schema_read')
            return copy.deepcopy(stale_raw)

        monkeypatch.setattr(fake_async_sg, 'schema_read', stale_schema_read)

        with (
            caplog.at_level('INFO', logger='sgalchemy.engine'),
            pytest.raises(SgalchemyError, match=r'entity type.*refreshing schema once'),
        ):
            await session.commit()

        names = [name for name, _, _ in fake_async_sg._calls]
        assert names.count('schema_read') == 2
        assert names.count('batch') == 0
        assert caplog.messages.count('commit schema drift: refreshing schema once') == 1

    asyncio.run(scenario())


def test_async_commit_schema_recovery_reuses_concurrent_binding_without_io(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        complete = engine.prepared_binding()
        assert complete is not None
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 31})
        session.delete(shot)
        engine.registry.bind(
            {name: schema for name, schema in complete.schemas.items() if name != 'Shot'},
            key=complete.key,
        )
        original_plan_commit: _SyncCallback[
            [AsyncSession, RegistryBinding | None, Mapping[tuple[str, int], Entity]],
            CommitPlan,
        ] = async_session_module.plan_commit
        publications = 0

        def publish_on_drift(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                try:
                    return call(*args, **kwargs)
                except SchemaDrift:
                    nonlocal publications
                    publications += 1
                    engine.registry.bind(dict(complete.schemas), key=complete.key)
                    raise

            return wrapper

        publish_during_failed_plan = publish_on_drift(original_plan_commit)

        monkeypatch.setattr(
            async_session_module,
            'plan_commit',
            publish_during_failed_plan,
        )

        await session.commit()

        names = [name for name, _, _ in fake_async_sg._calls]
        assert entity_state(shot).deleted
        assert publications == 1
        assert names.count('schema_read') == 1
        assert names.count('batch') == 1

    asyncio.run(scenario())


def test_async_execute_call_accepts_captured_binding_witness(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        call = FindCall('Shot', [], ['id'])
        fake_async_sg._find_results = [[{'id': 1}]]
        binding = engine.prepared_binding()

        assert await session.execute_call(call, binding) == [{'id': 1}]

    asyncio.run(scenario())


def test_async_execute_projection_returns_mapping_and_attribute_rows(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 1, 'code': 'A', 'sg_cut_duration': 42}]
        ]
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        result = await session.execute(select(shot_cls.code, shot_cls.sg_cut_duration))
        assert isinstance(result, Result)
        assert result[0]['code'] == 'A'
        assert result[0].sg_cut_duration == 42
        assert list(result[0]) == ['code', 'sg_cut_duration']

    asyncio.run(scenario())


def test_async_execute_rejects_entity_statement_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        calls_before = len(fake_async_sg._calls)
        with pytest.raises(TypeError, match='scalars'):
            await session.execute(cast('Any', select(engine.entities.Shot)))
        assert len(fake_async_sg._calls) == calls_before

    asyncio.run(scenario())


def test_async_get_identity_hit_does_not_capture_schema(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'cached'})
        calls_before = list(fake_async_sg._calls)

        async def fail_load_binding(*, force: bool = False) -> Never:
            del force
            raise AssertionError('identity-map hit must not prepare schema')

        monkeypatch.setattr(engine, 'load_binding', fail_load_binding)

        assert await session.get(shot_cls, 1) is shot
        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())


def test_async_get_rejects_mismatched_identity_before_registration(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 2, 'code': 'wrong'}]]

        with pytest.raises(SgalchemyError, match=r'requested Shot\(id=1\).*Shot\(id=2\)'):
            await session.get(shot_cls, 1)

        assert session._identity_values() == []

    asyncio.run(scenario())


def test_async_get_type_absent_after_refresh_raises_teaching_error(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, vendor_cls = await _async_engine_with_symbolic_vendor(fake_async_sg, tmp_path)
        session = AsyncSession(engine)
        reads_before = [name for name, _args, _kwargs in fake_async_sg._calls].count('schema_read')

        with pytest.raises(SchemaUnavailableError) as excinfo:
            await session.get(vendor_cls, 5)

        assert str(excinfo.value) == SCHEMA_ABSENT_MESSAGE
        calls = [name for name, _args, _kwargs in fake_async_sg._calls]
        assert calls.count('schema_read') == reads_before + 1
        assert calls.count('find_one') == 0

    asyncio.run(scenario())


def test_async_get_unknown_type_forces_one_refresh_then_succeeds(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, vendor_cls = await _async_engine_with_symbolic_vendor(fake_async_sg, tmp_path)
        extended = copy.deepcopy(schema_fixture)
        extended['FutureVendor'] = {
            'id': {'data_type': {'value': 'number'}, 'name': {'value': 'Id'}, 'properties': {}},
            'code': {'data_type': {'value': 'text'}, 'name': {'value': 'Code'}, 'properties': {}},
        }
        schema_reads: list[bool] = []

        async def extended_read() -> dict[str, Any]:
            schema_reads.append(True)
            return copy.deepcopy(extended)

        monkeypatch.setattr(fake_async_sg, 'schema_read', extended_read)
        fake_async_sg._find_results = [[{'type': 'FutureVendor', 'id': 5, 'code': 'V5'}]]
        session = AsyncSession(engine)
        hydration_bindings = capture_hydration_bindings(session, monkeypatch)

        found = await session.get(vendor_cls, 5)

        assert found is not None
        assert found.id == 5
        assert len(hydration_bindings) == 1
        assert hydration_bindings[0] is engine.prepared_binding()
        assert len(schema_reads) == 1
        assert [name for name, _args, _kwargs in fake_async_sg._calls].count('find_one') == 1

    asyncio.run(scenario())


def test_async_hydration_scope_is_rebuilt_after_provider_await(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        first = session.merge_dict({'type': 'Shot', 'id': 1})
        session.delete(first)
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 1}]]
        await session.scalars(select(engine.entities.Shot).execution_options(retired_only=True))
        assert inspect(first).deleted is False
        entered, release = fake_async_sg._block('find')
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 1}]]
        reading = asyncio.create_task(
            session.scalars(select(engine.entities.Shot).execution_options(retired_only=True))
        )
        await entered.wait()
        assert session._state._hydration_index is None
        # Inject a staging change at the response boundary to check hydration scope timing.
        session._unit_of_work._deleted.clear()
        release.set()
        objects = await reading
        assert objects[0] is first
        assert inspect(first).deleted is True
        assert session._state._hydration_index is None

        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 2, 'project': {'type': 'Vendor', 'id': 1}}]
        ]
        with pytest.raises(SgalchemyError, match='display alias'):
            await session.scalars(select(engine.entities.Shot))
        assert session._state._hydration_index is None
        session.unit_of_work._revived.append(first)
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 1}]]
        await session.scalars(select(engine.entities.Shot))
        assert inspect(first).deleted is True

    asyncio.run(scenario())


def test_async_joinedload_force_refreshes_before_one_primary_query(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        complete = engine.prepared_binding()
        assert complete is not None
        engine.registry.bind(
            {name: schema for name, schema in complete.schemas.items() if name != 'Sequence'},
            key=complete.key,
        )
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'sg_sequence': {'type': 'Sequence', 'id': 7},
                    'sg_sequence.Sequence.code': 'SEQ',
                }
            ]
        ]
        session = AsyncSession(engine)
        shot_cls = engine.registry.entity_class('Shot')

        shot = (await session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence))))[
            0
        ]

        assert shot.sg_sequence.code == 'SEQ'
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert [name for name, _, _ in fake_async_sg._calls].count('find') == 1

    asyncio.run(scenario())


def test_async_joinedload_remaining_drift_forces_once_without_query(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        complete = engine.prepared_binding()
        assert complete is not None
        engine.registry.bind(
            {name: schema for name, schema in complete.schemas.items() if name != 'Sequence'},
            key=complete.key,
        )
        stale_raw = {name: schema for name, schema in schema_fixture.items() if name != 'Sequence'}

        async def stale_schema_read() -> dict[str, Any]:
            await fake_async_sg._record('schema_read')
            return stale_raw

        monkeypatch.setattr(fake_async_sg, 'schema_read', stale_schema_read)
        session = AsyncSession(engine)
        shot_cls = engine.registry.entity_class('Shot')

        with pytest.raises(
            CompilationError,
            match=r'Shot\.sg_sequence.*Sequence.*refresh.*once',
        ):
            await session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence)))

        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 2
        assert [name for name, _, _ in fake_async_sg._calls].count('find') == 0

    asyncio.run(scenario())


def test_async_joinedload_reuses_concurrently_published_binding_without_io(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        complete = engine.prepared_binding()
        assert complete is not None
        engine.registry.bind(
            {name: schema for name, schema in complete.schemas.items() if name != 'Sequence'},
            key=complete.key,
        )
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'sg_sequence': {'type': 'Sequence', 'id': 7},
                    'sg_sequence.Sequence.code': 'SEQ',
                }
            ]
        ]
        session = AsyncSession(engine)
        shot_cls = engine.registry.entity_class('Shot')
        statement = select(shot_cls).options(joinedload(shot_cls.sg_sequence))
        compile_statement: _SyncCallback[
            [Select[Entity | None], Mapping[str, EntitySchema]],
            FindCall | SummarizeCall,
        ] = async_session_module.compile_statement
        publications = 0

        def publish_on_drift(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
                try:
                    return call(*args, **kwargs)
                except SchemaDrift:
                    nonlocal publications
                    publications += 1
                    engine.registry.bind(dict(complete.schemas), key=complete.key)
                    raise

            return wrapper

        publish_during_failed_compile = publish_on_drift(compile_statement)

        monkeypatch.setattr(
            async_session_module,
            'compile_statement',
            publish_during_failed_compile,
        )

        shot = (await session.scalars(statement))[0]

        assert shot.sg_sequence.code == 'SEQ'
        assert publications == 1
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 1
        assert [name for name, _, _ in fake_async_sg._calls].count('find') == 1

    asyncio.run(scenario())


def test_async_load_parent_fetch_preserves_dirty_relationship(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        absent = session.merge_dict({'type': 'Shot', 'id': 1})
        dirty = session.merge_dict({'type': 'Shot', 'id': 2, 'assets': []})
        local_asset = session.merge_dict({'type': 'Asset', 'id': 20})
        dirty.assets = [local_asset]
        fake_async_sg._find_results = [
            [
                {'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 10}]},
                {'type': 'Shot', 'id': 2, 'assets': [{'type': 'Asset', 'id': 99}]},
            ],
            [
                {'type': 'Asset', 'id': 10, 'code': 'fetched'},
                {'type': 'Asset', 'id': 20, 'code': 'local-link'},
            ],
        ]

        await session.load([absent, dirty], shot_cls.assets)

        assert dirty.assets == [local_asset]
        assert absent.assets[0].code == 'fetched'
        find_calls = [call for call in fake_async_sg._calls if call[0] == 'find']
        assert find_calls[0][1][1] == [['id', 'in', [1]]]
        assert find_calls[0][1][2] == ['assets']

    asyncio.run(scenario())


def test_async_load_rejects_detached_same_registry_parent_before_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        owner = AsyncSession(engine)
        shot_cls = engine.entities.Shot
        shot = owner.merge_dict(
            {
                'type': 'Shot',
                'id': 1,
                'assets': [{'type': 'Asset', 'id': 10}],
            }
        )
        await owner.close()
        borrower = AsyncSession(engine)
        fake_async_sg._find_results = [[{'type': 'Asset', 'id': 10, 'code': 'server'}]]
        calls_before = list(fake_async_sg._calls)

        with pytest.raises(SgalchemyError, match='owned by this session'):
            await borrower.load(shot, shot_cls.assets)

        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())


def test_async_merge_dict_captures_once_and_record_hydration_passes_binding_through(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        binding = engine.prepared_binding()
        captures = 0

        def capture_binding() -> RegistryBinding | None:
            nonlocal captures
            captures += 1
            return binding

        monkeypatch.setattr(engine, 'prepared_binding', capture_binding)
        session = AsyncSession(engine)
        shot = session.merge_dict(
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7, 'code': 'SEQ'},
            },
        )
        session.hydrate_record(
            'Shot',
            {'type': 'Shot', 'id': 1, 'code': 'updated'},
            binding,
        )

        assert shot.sg_sequence.code == 'SEQ'
        assert shot.code == 'updated'
        assert captures == 1

    asyncio.run(scenario())


def test_async_merge_dict_rejects_non_canonical_root_identity(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        with pytest.raises(SgalchemyError, match='positive integer id'):
            session.merge_dict({'type': 'Shot', 'id': True})
        with pytest.raises(SgalchemyError, match='positive integer id'):
            session.merge_dict({'type': 'Shot', 'code': 'no-id'})
        with pytest.raises(SgalchemyError, match="non-empty string 'type'"):
            session.merge_dict({'id': 5, 'code': 'no-type'})

    asyncio.run(scenario())


def test_async_no_work_commit_does_not_load_schema(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        session = AsyncSession(engine)

        async def unexpected_load(*, force: bool = False) -> Never:
            del force
            raise AssertionError('no-work commit must not load schema')

        monkeypatch.setattr(engine, 'load_binding', unexpected_load)

        await session.commit()

    asyncio.run(scenario())


def test_async_read_auth_failure_carries_read_recovery_note(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.registry.entity_class('Shot')
        fault = AUTHENTICATION_FAULT('session expired')
        fake_async_sg._errors.setdefault('find', []).append(fault)

        with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
            await session.scalars(select(shot_cls))

        assert excinfo.value is fault
        joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
        assert 'rerun the read' in joined
        assert 'remain staged' not in joined

    asyncio.run(scenario())


def test_async_refetch_skips_deep_fields_under_dirty_relationship(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict(
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.code': 'old-server-value',
            }
        )
        replacement = session.merge_dict({'type': 'Sequence', 'id': 8, 'code': 'local-value'})
        shot.sg_sequence = replacement
        session.merge_dict(
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.code': 'new-server-value',
            }
        )
        assert shot.sg_sequence is replacement
        assert replacement.code == 'local-value'

    asyncio.run(scenario())


def test_async_refresh_clears_only_returned_field_markers_without_probe_io(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'old'})
        state = entity_state(shot)
        state.loaded.pop('code')
        state.refresh_required.update({'code', 'description'})
        state.unverified_writes.update({'sg_status_list', 'sg_cut_duration'})
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 1,
                    'code': 'new',
                    'sg_status_list': 'ip',
                }
            ]
        ]

        await session.refresh(shot)

        assert shot.code == 'new'
        assert shot.sg_status_list == 'ip'
        assert state.refresh_required == {'description'}
        assert state.unverified_writes == {'sg_cut_duration'}
        calls_after_refresh = list(fake_async_sg._calls)
        info = inspect(shot)
        assert repr(shot) == "Shot(id=1, code='new')"
        assert shot.as_dict()['code'] == 'new'
        assert info.refresh_required == frozenset({'description'})
        assert info.unverified_writes == frozenset({'sg_cut_duration'})
        assert fake_async_sg._calls == calls_after_refresh

    asyncio.run(scenario())


def test_async_refresh_opaque_object_succeeds_after_one_forced_refresh(
    fake_async_sg: _FakeAsyncShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _vendor_cls = await _async_engine_with_symbolic_vendor(fake_async_sg, tmp_path)
        extended = copy.deepcopy(schema_fixture)
        extended['FutureVendor'] = {
            'id': {'data_type': {'value': 'number'}, 'name': {'value': 'Id'}, 'properties': {}},
            'code': {'data_type': {'value': 'text'}, 'name': {'value': 'Code'}, 'properties': {}},
        }
        schema_reads: list[bool] = []

        async def extended_read() -> dict[str, Any]:
            schema_reads.append(True)
            return copy.deepcopy(extended)

        monkeypatch.setattr(fake_async_sg, 'schema_read', extended_read)
        session = AsyncSession(engine)
        opaque = session.merge_dict({'type': 'FutureVendor', 'id': 9})
        hydration_bindings = capture_hydration_bindings(session, monkeypatch)
        fake_async_sg._find_results = [[{'type': 'FutureVendor', 'id': 9, 'code': 'V9'}]]

        await session.refresh(opaque)

        assert opaque.code == 'V9'
        assert len(hydration_bindings) == 1
        assert hydration_bindings[0] is engine.prepared_binding()
        assert len(schema_reads) == 1

    asyncio.run(scenario())


def test_async_refresh_opaque_object_teaches_after_one_refresh(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _vendor_cls = await _async_engine_with_symbolic_vendor(fake_async_sg, tmp_path)
        session = AsyncSession(engine)
        opaque = session.merge_dict({'type': 'FutureVendor', 'id': 9})
        reads_before = [name for name, _args, _kwargs in fake_async_sg._calls].count('schema_read')

        with pytest.raises(SchemaUnavailableError) as excinfo:
            await session.refresh(opaque)

        assert str(excinfo.value) == SCHEMA_ABSENT_MESSAGE
        calls = [name for name, _args, _kwargs in fake_async_sg._calls]
        assert calls.count('schema_read') == reads_before + 1
        assert calls.count('find') == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ('rows', 'message'),
    [
        ([{'type': 'Shot', 'id': 2, 'code': 'wrong'}], r'requested Shot\(id=1\)'),
        (
            [
                {'type': 'Shot', 'id': 1, 'code': 'first'},
                {'type': 'Shot', 'id': 1, 'code': 'second'},
            ],
            'exactly one.*2',
        ),
    ],
)
def test_async_refresh_rejects_mismatched_or_multiple_rows_without_mutation(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot = session.merge_dict({'type': 'Shot', 'id': 1, 'code': 'before'})
        session.expire(shot)
        state = entity_state(shot)
        loaded_before = dict(state.loaded)
        fake_async_sg._find_results = [rows]

        with pytest.raises(SgalchemyError, match=message):
            await session.refresh(shot)

        assert state.loaded == loaded_before
        assert state.expired is True
        assert session._identity_values() == [shot]

    asyncio.run(scenario())


def test_async_refresh_retired_entity_requests_retired_only(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        retired = session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'old-retired'})
        active = session.merge_dict({'type': 'Shot', 'id': 42, 'code': 'old-active'})
        session.delete(retired)
        await session.commit()
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 41, 'code': 'retired'}],
            [{'type': 'Shot', 'id': 42, 'code': 'active'}],
        ]

        await session.refresh(retired)

        refresh_find = [call for call in fake_async_sg._calls if call[0] == 'find'][-1]
        assert refresh_find[2]['retired_only'] is True
        assert retired.code == 'retired'

        await session.refresh(active)

        refresh_find = [call for call in fake_async_sg._calls if call[0] == 'find'][-1]
        assert 'retired_only' not in refresh_find[2]
        assert active.code == 'active'

    asyncio.run(scenario())


def test_async_retired_query_marks_root_deleted(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        fake_async_sg._find_results = [[{'type': 'Shot', 'id': 41, 'code': 'retired'}]]
        shot = (
            await session.scalars(
                select(engine.entities.Shot).execution_options(retired_only=True)
            )
        )[0]
        assert inspect(shot).deleted is True
        session.revive(shot)
        assert len(session.unit_of_work.revived) == 1
        assert session.unit_of_work.revived[0] is shot

    asyncio.run(scenario())


def test_async_scalars_identity_and_dirty_merge(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 1, 'code': 'A'}],
            [{'type': 'Shot', 'id': 1, 'code': 'server-change'}],
        ]
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        session = AsyncSession(engine)
        await engine.load_schema()
        shot_cls = engine.entities.Shot
        shot = (await session.scalars(select(shot_cls)))[0]
        shot.code = 'local-change'
        again = (await session.scalars(select(shot_cls)))[0]
        assert again is shot
        assert shot.code == 'local-change'

    asyncio.run(scenario())


@pytest.mark.parametrize('retired', [False, True])
@pytest.mark.parametrize('child_baseline', [None, False, True])
def test_async_scalars_retirement_observation_is_root_only(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    retired: bool,
    child_baseline: bool | None,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        child = None
        if child_baseline is not None:
            child = session.merge_dict({'type': 'Sequence', 'id': 51, 'code': 'before'})
            entity_state(child).deleted = child_baseline
        fake_async_sg._find_results = [
            [
                {
                    'type': 'Shot',
                    'id': 41,
                    'sg_sequence': {'type': 'Sequence', 'id': 51},
                    'sg_sequence.Sequence.code': 'linked',
                }
            ]
        ]

        shot = (
            await session.scalars(
                select(engine.entities.Shot).execution_options(retired_only=retired)
            )
        )[0]

        assert inspect(shot).deleted is retired
        assert isinstance(shot.sg_sequence, Entity)
        assert inspect(shot.sg_sequence).deleted is bool(child_baseline)
        assert shot.sg_sequence.code == 'linked'
        if child is not None:
            assert shot.sg_sequence is child

    asyncio.run(scenario())


@pytest.mark.parametrize('retired', [False, True])
@pytest.mark.parametrize(
    ('baseline', 'transition'),
    [(None, None), (False, None), (True, None), (False, 'delete'), (True, 'revive')],
)
def test_async_scalars_retirement_observation_preserves_staged_intent(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
    retired: bool,
    baseline: bool | None,
    transition: str | None,
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        original = None
        if baseline is not None:
            original = session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'before'})
            state = entity_state(original)
            state.deleted = baseline
            original.code = 'local'
            if transition is not None:
                getattr(session, transition)(original)
            state.refresh_required.update({'code', 'description', 'sg_status_list'})
            state.unverified_writes.update({'code', 'description', 'sg_status_list'})
        deleted_before = tuple(session.unit_of_work.deleted)
        revived_before = tuple(session.unit_of_work.revived)
        fake_async_sg._find_results = [
            [{'type': 'Shot', 'id': 41, 'code': 'server', 'description': 'refreshed'}]
        ]

        shot = (
            await session.scalars(
                select(engine.entities.Shot).execution_options(retired_only=retired)
            )
        )[0]

        assert inspect(shot).deleted is (baseline if transition is not None else retired)
        assert tuple(map(id, session.unit_of_work.deleted)) == tuple(map(id, deleted_before))
        assert tuple(map(id, session.unit_of_work.revived)) == tuple(map(id, revived_before))
        if original is not None:
            assert shot is original
            state = entity_state(shot)
            assert state.dirty == {'code': 'local'}
            assert state.loaded['code'] == 'local'
            assert state.refresh_required == {'code', 'sg_status_list'}
            assert state.unverified_writes == {'code', 'sg_status_list'}
        else:
            assert shot.code == 'server'
        assert shot.description == 'refreshed'
        session.merge_dict({'type': 'Shot', 'id': 41, 'description': 'merged'})
        assert inspect(shot).deleted is (baseline if transition is not None else retired)

    asyncio.run(scenario())


@pytest.mark.parametrize('size', [100, 200, 400])
def test_async_response_hydration_bounds_staged_item_visits(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path, size: int
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        deleted = [session.merge_dict({'type': 'Shot', 'id': 1000 + i}) for i in range(size)]
        revived = [session.merge_dict({'type': 'Shot', 'id': 2000 + i}) for i in range(size)]
        for obj in deleted:
            session.delete(obj)
        for obj in revived:
            entity_state(obj).deleted = True
            session.revive(obj)
        deleted_queue = YieldCountingEntities(session.unit_of_work._deleted)
        revived_queue = YieldCountingEntities(session.unit_of_work._revived)
        session.unit_of_work._deleted = deleted_queue
        session.unit_of_work._revived = revived_queue
        fake_async_sg._find_results = [
            [
                {'type': 'Shot', 'id': i + 1, 'project': {'type': 'Project', 'id': i + 1}}
                for i in range(size)
            ]
        ]

        objects = await session.scalars(select(engine.entities.Shot))

        assert len(objects) == size
        assert len(session._identity_values()) == 4 * size
        assert objects[0].project.id == 1
        assert deleted_queue._visits + revived_queue._visits <= 6 * size

    asyncio.run(scenario())


def test_async_shape_mismatch_and_foreign_registry_parity(
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
        foreign_session = AsyncSession(other)
        foreign_cls = other.entities.Shot
        foreign = foreign_session.merge_dict({'type': 'Shot', 'id': 9, 'code': 'foreign'})

        contradictory = session.merge_dict({'type': 'Shot', 'id': 1, 'code': ['not', 'scalar']})
        assert contradictory.code == ['not', 'scalar']
        await engine.load_schema()
        assert [name for name, _, _ in fake_async_sg._calls].count('schema_read') == 3

        with pytest.raises(SgalchemyError, match='different registry'):
            await session.get(foreign_cls, 9)
        with pytest.raises(SgalchemyError, match='different registry'):
            await session.scalars(select(foreign_cls))
        with pytest.raises(SgalchemyError, match='different registry'):
            session.expire(foreign)
        with pytest.raises(SgalchemyError, match='different registry'):
            await session.refresh(foreign)

    asyncio.run(scenario())


def test_async_shared_access_guard_rejects_foreign_task_use(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        entered, release = fake_async_sg._block('find')
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        stmt = select(engine.entities.Shot)
        existing = session.merge_dict({'type': 'Shot', 'id': 99, 'code': 'existing'})
        other = session.merge_dict({'type': 'Shot', 'id': 100, 'code': 'other'})
        assert entity_state(existing).access_guard is entity_state(other).access_guard
        first = asyncio.create_task(session.scalars(stmt))
        await entered.wait()
        with pytest.raises(SgalchemyError, match='concurrent task'):
            await session.scalars(stmt)
        with pytest.raises(SgalchemyError, match='concurrent task'):
            session.expire(existing)
        with pytest.raises(SgalchemyError, match='concurrent task'):
            _ = existing.code
        with pytest.raises(SgalchemyError, match='concurrent task'):
            existing.code = 'blocked'
        with pytest.raises(SgalchemyError, match='concurrent task'):
            existing.as_dict()
        with pytest.raises(SgalchemyError, match='concurrent task'):
            inspect(existing)
        release.set()
        await first

    asyncio.run(scenario())


def test_async_sole_batch_auth_failure_notes_state_local_facts_only(
    fake_async_sg: _FakeAsyncShotgun, tmp_path: Path
) -> None:
    async def scenario() -> None:
        engine = isolated_async_engine(fake_async_sg, tmp_path)
        await engine.load_schema()
        session = AsyncSession(engine)
        shot_cls = engine.registry.entity_class('Shot')
        session.add(shot_cls.new(code='auth-fail'))
        fault = AUTHENTICATION_FAULT('session expired')
        fake_async_sg._errors.setdefault('batch', []).append(fault)

        with pytest.raises(AUTHENTICATION_FAULT) as excinfo:
            await session.commit()

        assert excinfo.value is fault
        joined = '\n'.join(getattr(excinfo.value, '__notes__', []))
        assert 'its server outcome is unknown' in joined
        assert 'remain staged' in joined
        assert 'no automatic retry' in joined
        assert 'nothing committed' not in joined
        assert 'did not execute' not in joined
        assert sum('no automatic retry' in note for note in excinfo.value.__notes__) == 1

    asyncio.run(scenario())


def test_async_unprepared_hydration_rejects_alias_known_by_shared_registry(
    fake_async_sg: _FakeAsyncShotgun,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        registry = Registry()
        prepared = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'prepared',
            registry=registry,
            bind_default=False,
        )
        await prepared.load_schema()
        unprepared = create_async_engine(
            connection=fake_async_sg,
            cache_root=tmp_path / 'unprepared',
            registry=registry,
            bind_default=False,
        )
        session = AsyncSession(unprepared)
        calls_before = list(fake_async_sg._calls)
        alias = {'type': 'Vendor', 'id': 7, 'code': 'raw'}
        operations = (
            lambda: session.merge_dict(alias),
            lambda: session.hydrate_record('Vendor', alias, None),
        )

        for operation in operations:
            with pytest.raises(SgalchemyError, match=r'Vendor.*alias.*CustomEntity03'):
                operation()
            assert session._identity_values() == []

        future = session.merge_dict({'type': 'FutureVendor', 'id': 7, 'code': 'raw'})
        assert type(future).__name__ == 'FutureVendor'
        assert fake_async_sg._calls == calls_before

    asyncio.run(scenario())
