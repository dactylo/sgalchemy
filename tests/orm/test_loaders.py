from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import pytest
from _support import YieldCountingEntities

from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._options import joinedload, lazyload, load_only, selectinload
from sgalchemy._core._registry import (
    Entity,
    EntityState,
    Registry,
    RegistryBinding,
    entity_state,
    initialize_entity,
)
from sgalchemy._core._schema_drift import SchemaDrift
from sgalchemy._core._statement import Select, select
from sgalchemy._errors import (
    CompilationError,
    RelationshipTargetUnavailableError,
    SgalchemyError,
    StaleObjectError,
    UnfetchedFieldError,
)
from sgalchemy._orm import _session as session_module
from sgalchemy._orm._loaders import LazyContext
from sgalchemy._orm._session import Session

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from _fakes import _FakeShotgun
    from _support import _SyncCallback

    from sgalchemy._core._compiler import FindCall, SummarizeCall
    from sgalchemy.schema._model import EntitySchema

_P = ParamSpec('_P')
_R = TypeVar('_R')


def _find_calls(fake_sg: _FakeShotgun) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    return [call for call in fake_sg._calls if call[0] == 'find']


def _publish_without(engine: Engine, type_name: str) -> tuple[RegistryBinding, RegistryBinding]:
    complete = engine.load_binding()
    engine.registry.bind(
        {name: schema for name, schema in complete.schemas.items() if name != type_name},
        key=complete.key,
    )
    stale = engine.prepared_binding()
    assert stale is not None
    return complete, stale


def test_close_clears_every_lazy_context_without_io(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': 4, 'code': 'D', 'assets': [{'type': 'Asset', 'id': 103}]},
            {'type': 'Shot', 'id': 5, 'code': 'E', 'assets': [{'type': 'Asset', 'id': 104}]},
        ]
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shots = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))
    linked_assets = [entity_state(shot).loaded['assets'][0] for shot in shots]
    calls_before = len(fake_sg._calls)

    session.close()

    assert all(entity_state(shot).lazy == {} for shot in shots)
    assert all(entity_state(shot).session is None for shot in shots)
    assert all(entity_state(asset).session is None for asset in linked_assets)
    assert linked_assets[0].as_dict() == {'type': 'Asset', 'id': 103}
    assert len(fake_sg._calls) == calls_before


def test_conflicting_relationship_loaders_raise_before_io(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    statement = select(shot_cls).options(
        selectinload(shot_cls.sg_sequence),
        lazyload(shot_cls.sg_sequence),
    )

    with pytest.raises(
        CompilationError,
        match=r'conflicting relationship loader strategies.*Shot\.sg_sequence',
    ):
        session.scalars(statement)

    assert fake_sg._calls == []


@pytest.mark.parametrize('deep_only', [False, True])
def test_deep_load_hydrates_fields_without_post_query(
    engine: Engine, fake_sg: _FakeShotgun, deep_only: bool
) -> None:
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'code': 'A',
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.code': 'SEQ',
            }
        ]
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    option = (
        load_only(FieldExpr('Shot', ('sg_sequence', 'Sequence', 'code'), None))
        if deep_only
        else joinedload(shot_cls.sg_sequence)
    )
    shot = session.scalars(select(shot_cls).options(option))[0]

    assert shot.sg_sequence.code == 'SEQ'
    assert len([call for call in fake_sg._calls if call[0] == 'find']) == 1


def test_dirty_relationship_rehydration_preserves_value_and_context(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 101}]}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    original_context = entity_state(shot).lazy['assets']
    local_asset = session.merge_dict({'type': 'Asset', 'id': 200, 'code': 'local'})
    shot.assets = [local_asset]

    reused = session.scalars(select(shot_cls))[0]

    state = entity_state(reused)
    assert state.loaded['assets'] == [local_asset]
    assert state.dirty['assets'] == [local_asset]
    assert state.lazy['assets'] is original_context
    assert len(_find_calls(fake_sg)) == 2


def test_dirty_selectin_fault_retains_lazy_context_and_value(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError('eager relationship load failed')
    original_find = fake_sg.find
    call_count = 0

    def track_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                fake_sg._find_error = error
            return call(*args, **kwargs)

        return wrapper

    find = track_call(original_find)

    monkeypatch.setattr(fake_sg, 'find', find)
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 101}]}],
        [{'type': 'Asset', 'id': 200, 'code': 'retried'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    original_context = entity_state(shot).lazy['assets']
    local_asset = session.merge_dict({'type': 'Asset', 'id': 200, 'code': 'local'})
    shot.assets = [local_asset]

    with pytest.raises(RuntimeError) as excinfo:
        session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))

    state = entity_state(shot)
    assert excinfo.value is error
    assert state.loaded['assets'] == [local_asset]
    assert state.dirty['assets'] == [local_asset]
    assert state.lazy['assets'] is original_context
    assert local_asset.code == 'local'
    calls_after_failure = len(_find_calls(fake_sg))

    assert shot.assets == [local_asset]
    assert local_asset.code == 'retried'
    assert len(_find_calls(fake_sg)) == calls_after_failure + 1
    assert 'assets' not in state.lazy
    assert state.dirty['assets'] == [local_asset]


def test_dirty_selectin_retires_only_participating_lazy_entries(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]},
            {'type': 'Shot', 'id': 2, 'assets': [{'type': 'Asset', 'id': 102}]},
        ],
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 101}]}],
        [{'type': 'Asset', 'id': 200, 'code': 'eager'}],
        [{'type': 'Asset', 'id': 102, 'code': 'retained'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    original_results = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))
    original_context = entity_state(original_results[0]).lazy['assets']
    assert entity_state(original_results[1]).lazy['assets'] is original_context
    local_asset = session.merge_dict({'type': 'Asset', 'id': 200, 'code': 'local'})
    original_results[0].assets = [local_asset]

    reused = session.scalars(
        select(shot_cls).where(shot_cls.id == 1).options(selectinload(shot_cls.assets))
    )[0]
    calls_before_access = len(_find_calls(fake_sg))

    assert reused is original_results[0]
    assert entity_state(reused).loaded['assets'] == [local_asset]
    assert entity_state(reused).dirty['assets'] == [local_asset]
    assert local_asset.code == 'eager'
    assert reused.assets == [local_asset]
    assert len(_find_calls(fake_sg)) == calls_before_access
    assert 'assets' not in entity_state(reused).lazy
    assert entity_state(original_results[1]).lazy['assets'] is original_context

    assert original_results[1].assets[0].code == 'retained'
    assert _find_calls(fake_sg)[-1][1][1] == [['id', 'in', [102]]]


def test_duplicate_lazyload_executes_once(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Asset', 'id': 100, 'code': 'loaded'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(
        select(shot_cls).options(
            lazyload(shot_cls.assets),
            lazyload(shot_cls.assets),
        )
    )[0]

    assert shot.assets[0].code == 'loaded'
    assert len(_find_calls(fake_sg)) == 2


def test_duplicate_selectinload_executes_once(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Asset', 'id': 100, 'code': 'loaded'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(
        select(shot_cls).options(
            selectinload(shot_cls.assets),
            selectinload(shot_cls.assets),
        )
    )[0]

    assert shot.assets[0].code == 'loaded'
    assert len(_find_calls(fake_sg)) == 2


def test_empty_lazyload_on_unprepared_engine_stays_local(
    engine: Engine,
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    state = EntityState(
        entity_type='Shot',
        id=1,
        loaded={'assets': []},
        session=session,
    )
    initialize_entity(shot, state)
    context = LazyContext(session, [shot], lazyload(shot_cls.assets))
    state.lazy['assets'] = context

    assert shot.assets == []

    assert state.lazy == {}
    assert engine.prepared_binding() is None
    assert fake_sg._calls == []
    assert list(tmp_path.iterdir()) == []


def test_expired_lazy_relationship_raises_before_resolving(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}]]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    original_context = entity_state(shot).lazy['assets']
    session.expire(shot)
    calls_before = len(_find_calls(fake_sg))

    with pytest.raises(StaleObjectError):
        _ = shot.assets

    assert len(_find_calls(fake_sg)) == calls_before
    assert entity_state(shot).lazy['assets'] is original_context


def test_failed_lazy_target_refresh_memo_uses_binding_identity(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    engine.load_schema()
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_vendor': {'type': 'FutureVendor', 'id': 9},
            }
        ],
        [{'type': 'FutureVendor', 'id': 9, 'code': 'FV-9'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(lazyload(shot_cls.sg_vendor)))[0]
    opaque = entity_state(shot).loaded['sg_vendor']
    context = entity_state(shot).lazy['sg_vendor']

    for _ in range(2):
        with pytest.raises(
            RelationshipTargetUnavailableError,
            match=r'Shot\.sg_vendor.*FutureVendor',
        ):
            _ = shot.sg_vendor
        assert entity_state(shot).lazy['sg_vendor'] is context

    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Shot']

    failed = engine.prepared_binding()
    assert failed is not None
    future_schema = replace(
        failed.schemas['CustomEntity03'],
        type_name='FutureVendor',
        display_name='Future Vendor',
    )
    engine.registry.bind(
        {**failed.schemas, 'FutureVendor': future_schema},
        key=failed.key,
    )

    assert shot.sg_vendor is opaque
    assert opaque.code == 'FV-9'
    assert 'sg_vendor' not in entity_state(shot).lazy
    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Shot', 'FutureVendor']


def test_invalid_joinedloads_and_opaque_root_never_refresh_schema(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    engine.load_schema()
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    version_cls = engine.registry.entity_class('Version')
    opaque_cls = engine.registry.entity_class_for('FutureVendor', binding=None)
    statements = [
        select(opaque_cls),
        select(shot_cls).options(joinedload(FieldExpr('Shot', (), None))),
        select(shot_cls).options(joinedload(shot_cls.code)),
        select(shot_cls).options(joinedload(shot_cls.assets)),
        select(version_cls).options(joinedload(version_cls.entity)),
    ]

    for statement in statements:
        with pytest.raises(CompilationError):
            session.scalars(statement)

    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 1
    assert _find_calls(fake_sg) == []


def test_joinedload_chain_reports_first_missing_intermediate_after_refresh(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    complete, _stale = _publish_without(engine, 'Sequence')
    stale_raw = {
        name: schema
        for name, schema in schema_fixture.items()
        if name not in {'Episode', 'Sequence'}
    }

    def stale_schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return stale_raw

    monkeypatch.setattr(fake_sg, 'schema_read', stale_schema_read)
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    sequence_cls = engine.registry.entity_class_for('Sequence', binding=complete)
    option = joinedload(shot_cls.sg_sequence).joinedload(sequence_cls.episode)

    with pytest.raises(CompilationError) as excinfo:
        session.scalars(select(shot_cls).options(option))

    message = str(excinfo.value)
    assert 'Shot.sg_sequence' in message
    assert 'Sequence' in message
    assert 'Sequence.episode' not in message
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert _find_calls(fake_sg) == []


def test_joinedload_force_refreshes_once_then_runs_one_primary_query(
    engine: Engine,
    fake_sg: _FakeShotgun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _publish_without(engine, 'Sequence')
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.code': 'SEQ',
            }
        ]
    ]

    with caplog.at_level('INFO', logger='sgalchemy.engine'):
        shot = session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence)))[0]

    assert shot.sg_sequence.code == 'SEQ'
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert len(_find_calls(fake_sg)) == 1
    assert 'refreshing schema' in caplog.text


def test_joinedload_recompiles_after_concurrent_binding_publication_without_io(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete, stale = _publish_without(engine, 'Sequence')
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    statement = select(shot_cls).options(joinedload(shot_cls.sg_sequence))
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.code': 'SEQ',
            }
        ]
    ]
    compile_statement: _SyncCallback[
        [Select[Entity | None], Mapping[str, EntitySchema]],
        FindCall | SummarizeCall,
    ] = session_module.compile_statement
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

    monkeypatch.setattr(session_module, 'compile_statement', publish_during_failed_compile)

    shot = session.scalars(statement)[0]

    assert shot.sg_sequence.code == 'SEQ'
    assert publications == 1
    assert stale is not engine.prepared_binding()
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 1
    assert len(_find_calls(fake_sg)) == 1


def test_joinedload_remaining_drift_raises_after_one_force_without_query(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    _publish_without(engine, 'Sequence')
    stale_raw = {name: schema for name, schema in schema_fixture.items() if name != 'Sequence'}

    def stale_schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return stale_raw

    monkeypatch.setattr(fake_sg, 'schema_read', stale_schema_read)
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    with pytest.raises(
        CompilationError,
        match=r'Shot\.sg_sequence.*Sequence.*refresh.*once',
    ):
        session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence)))

    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert _find_calls(fake_sg) == []


def test_lazy_then_joined_reuse_keeps_eager_data_without_post_query(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'sg_sequence': {'type': 'Sequence', 'id': 7}}],
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'Sequence', 'id': 7},
                'sg_sequence.Sequence.code': 'joined',
            }
        ],
        [{'type': 'Sequence', 'id': 7, 'code': 'redundant'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    first = session.scalars(select(shot_cls).options(lazyload(shot_cls.sg_sequence)))[0]
    reused = session.scalars(select(shot_cls).options(joinedload(shot_cls.sg_sequence)))[0]
    calls_before_access = len(_find_calls(fake_sg))

    assert reused is first
    assert reused.sg_sequence.code == 'joined'
    assert len(_find_calls(fake_sg)) == calls_before_access
    assert 'sg_sequence' not in entity_state(reused).lazy


def test_lazy_then_plain_rehydration_clears_stale_context(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 101}]}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    first = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    reused = session.scalars(select(shot_cls))[0]
    calls_before_access = len(_find_calls(fake_sg))

    assert reused is first
    assert reused.assets[0].as_dict() == {'type': 'Asset', 'id': 101}
    assert len(_find_calls(fake_sg)) == calls_before_access
    assert 'assets' not in entity_state(reused).lazy


def test_lazy_then_refresh_clears_stale_context(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 101}]}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    session.refresh(shot)
    calls_before_access = len(_find_calls(fake_sg))

    assert shot.assets[0].as_dict() == {'type': 'Asset', 'id': 101}
    assert len(_find_calls(fake_sg)) == calls_before_access
    assert 'assets' not in entity_state(shot).lazy


def test_lazy_then_selectin_reuse_keeps_eager_data_without_post_query(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}],
        [{'type': 'Asset', 'id': 100, 'code': 'eager'}],
        [{'type': 'Asset', 'id': 100, 'code': 'redundant'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    first = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    reused = session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))[0]
    calls_before_access = len(_find_calls(fake_sg))

    assert reused is first
    assert reused.assets[0].code == 'eager'
    assert len(_find_calls(fake_sg)) == calls_before_access
    assert 'assets' not in entity_state(reused).lazy


def test_lazyload_batches_on_first_touch_and_respects_budget(
    engine: Engine,
    fake_sg: _FakeShotgun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': 1, 'code': 'A', 'assets': [{'type': 'Asset', 'id': 100}]},
            {'type': 'Shot', 'id': 2, 'code': 'B', 'assets': [{'type': 'Asset', 'id': 101}]},
        ],
        [{'type': 'Asset', 'id': 100, 'code': 'a'}, {'type': 'Asset', 'id': 101, 'code': 'b'}],
    ]
    session = Session(engine, max_implicit_queries=1)
    shot_cls = engine.registry.entity_class('Shot')
    shots = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))
    primary_find = next(call for call in fake_sg._calls if call[0] == 'find')
    calls_before = len(fake_sg._calls)

    assert 'assets' in primary_find[1][2]
    with caplog.at_level('INFO', logger='sgalchemy.session'):
        first_assets = shots[0].assets

    assert first_assets[0].code == 'a'
    assert len(fake_sg._calls) == calls_before + 1
    assert any('lazy load: Shot.assets' in record.message for record in caplog.records)
    assert shots[1].assets[0].code == 'b'
    assert len(fake_sg._calls) == calls_before + 1

    session2 = Session(engine, max_implicit_queries=0)
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 3, 'code': 'C', 'assets': [{'type': 'Asset', 'id': 102}]}]
    ]
    shot3 = session2.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))[0]
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match=r'max_implicit_queries.*Shot\.assets'):
        _ = shot3.assets
    assert len(fake_sg._calls) == calls_before


def test_lazyload_counts_actual_target_type_calls(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [
        [
            {
                'type': 'Note',
                'id': 1,
                'note_links': [
                    {'type': 'Shot', 'id': 10},
                    {'type': 'Version', 'id': 20},
                ],
            }
        ],
        [{'type': 'Shot', 'id': 10, 'code': 'A'}],
        [{'type': 'Version', 'id': 20, 'code': 'v1'}],
        [{'type': 'Note', 'id': 2, 'note_links': []}],
        [
            {
                'type': 'Note',
                'id': 3,
                'note_links': [{'type': 'Shot', 'id': 30}],
            }
        ],
    ]
    session = Session(engine, max_implicit_queries=2)
    note_cls = engine.registry.entity_class('Note')
    first = session.scalars(select(note_cls).options(lazyload(note_cls.note_links)))[0]
    assert [link.code for link in first.note_links] == ['A', 'v1']

    empty = session.scalars(select(note_cls).options(lazyload(note_cls.note_links)))[0]
    calls_before_empty = len(fake_sg._calls)
    assert empty.note_links == []
    assert len(fake_sg._calls) == calls_before_empty

    last = session.scalars(select(note_cls).options(lazyload(note_cls.note_links)))[0]
    calls_before_budget_error = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match='needs 1 implicit query'):
        _ = last.note_links
    assert len(fake_sg._calls) == calls_before_budget_error


def test_lazyload_overlapping_results_preserve_newer_context(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 101}]},
            {'type': 'Shot', 'id': 2, 'assets': [{'type': 'Asset', 'id': 102}]},
        ],
        [
            {'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 111}]},
            {'type': 'Shot', 'id': 3, 'assets': [{'type': 'Asset', 'id': 103}]},
        ],
        [{'type': 'Asset', 'id': 102, 'code': 'old-only'}],
        [
            {'type': 'Asset', 'id': 103, 'code': 'new-third'},
            {'type': 'Asset', 'id': 111, 'code': 'new-first'},
        ],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    old_results = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))
    old_context = entity_state(old_results[1]).lazy['assets']
    new_results = session.scalars(select(shot_cls).options(lazyload(shot_cls.assets)))
    new_context = entity_state(new_results[0]).lazy['assets']
    assert new_results[0] is old_results[0]
    assert new_context is not old_context

    assert old_results[1].assets[0].code == 'old-only'
    assert _find_calls(fake_sg)[-1][1][1] == [['id', 'in', [102]]]
    assert entity_state(new_results[0]).lazy.get('assets') is new_context
    assert entity_state(new_results[1]).lazy.get('assets') is new_context

    assert new_results[0].assets[0].code == 'new-first'
    assert _find_calls(fake_sg)[-1][1][1] == [['id', 'in', [103, 111]]]
    assert new_results[1].assets[0].code == 'new-third'


def test_lazyload_pending_only_targets_use_no_budget_or_io(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    engine.load_schema()
    session = Session(engine, max_implicit_queries=0)
    shot_cls = engine.registry.entity_class('Shot')
    pending_asset = engine.registry.entity_class('Asset').new(code='pending')
    shot = session.merge_dict({'type': 'Shot', 'id': 4, 'assets': []})
    shot.assets = [pending_asset]
    state = entity_state(shot)
    context = LazyContext(session, [shot], lazyload(shot_cls.assets))
    state.lazy['assets'] = context
    calls_before = list(fake_sg._calls)

    assert shot.assets == [pending_asset]

    assert state.lazy == {}
    assert state.dirty == {'assets': [pending_asset]}
    assert fake_sg._calls == calls_before


def test_lazyload_reserves_all_target_type_calls_before_io(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    fake_sg._find_results = [
        [
            {
                'type': 'Note',
                'id': 1,
                'note_links': [
                    {'type': 'Shot', 'id': 10},
                    {'type': 'Version', 'id': 20},
                ],
            }
        ]
    ]
    session = Session(engine, max_implicit_queries=1)
    note_cls = engine.registry.entity_class('Note')
    note = session.scalars(select(note_cls).options(lazyload(note_cls.note_links)))[0]
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match=r'2 implicit queries.*Note\.note_links'):
        _ = note.note_links

    assert len(fake_sg._calls) == calls_before


def test_lazyload_schema_preparation_fault_has_session_context(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    session = Session(engine)
    asset_cls = engine.registry.entity_class('Asset')
    asset = asset_cls.__new__(asset_cls)
    initialize_entity(asset, EntityState(entity_type='Asset', id=2, session=session))
    shot_cls = engine.registry.entity_class('Shot')
    shot = shot_cls.__new__(shot_cls)
    state = EntityState(
        entity_type='Shot',
        id=1,
        loaded={'assets': [asset]},
        session=session,
    )
    initialize_entity(shot, state)
    context = LazyContext(session, [shot], lazyload(shot_cls.assets))
    state.lazy['assets'] = context
    error = RuntimeError('schema unavailable')
    fake_sg._errors['schema_read'] = [error]

    with pytest.raises(RuntimeError) as excinfo:
        _ = shot.assets

    assert excinfo.value is error
    assert error.__notes__ == ['sgalchemy: raised while lazy-loading Shot.assets']
    assert state.lazy['assets'] is context
    assert [name for name, _, _ in fake_sg._calls] == ['schema_read']


def test_lazyload_single_option_keeps_its_target_read(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    fake_sg._find_results = [
        [{'type': 'Version', 'id': 1, 'user': {'type': 'HumanUser', 'id': 2}}],
        [{'type': 'HumanUser', 'id': 2, 'name': 'B'}],
    ]
    session = Session(engine)
    version_cls = engine.registry.entity_class('Version')
    version = session.scalars(select(version_cls).options(lazyload(version_cls.user)))[0]

    assert version.user.name == 'B'
    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Version', 'HumanUser']
    assert _find_calls(fake_sg)[1][1][1] == [['id', 'in', [2]]]


@pytest.mark.parametrize('lazy', [False, True])
def test_loader_response_has_its_own_staged_identity_scope(
    engine: Engine, fake_sg: _FakeShotgun, lazy: bool
) -> None:
    engine.load_schema()
    session = Session(engine)
    staged = [session.merge_dict({'type': 'Shot', 'id': 1000 + i}) for i in range(20)]
    for obj in staged:
        session.delete(obj)
    queue = YieldCountingEntities(session.unit_of_work._deleted)
    session.unit_of_work._deleted = queue
    fake_sg._find_results = [
        [
            {'type': 'Shot', 'id': i + 1, 'assets': [{'type': 'Asset', 'id': i + 1}]}
            for i in range(20)
        ],
        [{'type': 'Asset', 'id': i + 1, 'code': f'asset-{i}'} for i in range(20)],
    ]
    loader = lazyload if lazy else selectinload

    objects = session.scalars(
        select(engine.entities.Shot).options(loader(engine.entities.Shot.assets))
    )

    if lazy:
        assert queue._visits == 20
    assert objects[0].assets[0].code == 'asset-0'
    assert objects[-1].assets[0].code == 'asset-19'
    assert queue._visits == 40
    assert session._state._hydration_index is None


def test_multiple_selectins_fail_union_preflight_before_any_secondary_find(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    engine.load_schema()
    refreshed_raw = copy.deepcopy(schema_fixture)
    refreshed_raw['FutureA'] = copy.deepcopy(refreshed_raw['Sequence'])

    def refreshed_schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return refreshed_raw

    monkeypatch.setattr(fake_sg, 'schema_read', refreshed_schema_read)
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'FutureA', 'id': 8},
                'sg_vendor': {'type': 'FutureB', 'id': 9},
            }
        ]
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    with pytest.raises(
        RelationshipTargetUnavailableError,
        match=r'Shot\.sg_vendor.*FutureB',
    ):
        session.scalars(
            select(shot_cls).options(
                selectinload(shot_cls.sg_sequence),
                selectinload(shot_cls.sg_vendor),
            )
        )

    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Shot']


def test_multiple_selectins_preserve_distinct_target_order(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [
            {
                'type': 'Version',
                'id': 1,
                'entity': {'type': 'Shot', 'id': 10},
                'user': {'type': 'HumanUser', 'id': 1},
            }
        ],
        [{'type': 'Shot', 'id': 10, 'code': 'A'}],
        [{'type': 'HumanUser', 'id': 1, 'name': 'B'}],
    ]
    session = Session(engine)
    version_cls = engine.registry.entity_class('Version')

    version = session.scalars(
        select(version_cls).options(
            selectinload(version_cls.entity),
            selectinload(version_cls.user),
        )
    )[0]

    assert [call[1][0] for call in _find_calls(fake_sg)] == [
        'Version',
        'Shot',
        'HumanUser',
    ]
    assert (version.entity.code, version.user.name) == ('A', 'B')


def test_multiple_selectins_retire_completed_lazy_marker_before_later_failure(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError('later target failed')
    original_find = fake_sg.find
    call_count = 0

    def track_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nonlocal call_count
            call_count += 1
            if call_count == 4:
                fake_sg._find_error = error
            return call(*args, **kwargs)

        return wrapper

    find = track_call(original_find)

    monkeypatch.setattr(fake_sg, 'find', find)
    parent = {
        'type': 'Shot',
        'id': 1,
        'sg_sequence': {'type': 'Sequence', 'id': 7},
        'assets': [{'type': 'Asset', 'id': 100}],
    }
    fake_sg._find_results = [
        [parent],
        [parent],
        [{'type': 'Sequence', 'id': 7, 'code': 'loaded'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(
        select(shot_cls).options(
            lazyload(shot_cls.sg_sequence),
            lazyload(shot_cls.assets),
        )
    )[0]
    state = entity_state(shot)
    assert 'sg_sequence' in state.lazy
    asset_context = state.lazy['assets']
    shot.sg_sequence = state.loaded['sg_sequence']
    shot.assets = state.loaded['assets']

    with pytest.raises(RuntimeError) as excinfo:
        session.scalars(
            select(shot_cls).options(
                selectinload(shot_cls.sg_sequence),
                selectinload(shot_cls.assets),
            )
        )

    assert excinfo.value is error
    assert 'sg_sequence' not in state.lazy
    assert state.lazy['assets'] is asset_context
    calls_before_access = len(_find_calls(fake_sg))
    assert [call[1][0] for call in _find_calls(fake_sg)] == [
        'Shot',
        'Shot',
        'Sequence',
        'Asset',
    ]
    assert shot.sg_sequence.code == 'loaded'
    assert len(_find_calls(fake_sg)) == calls_before_access


def test_multiple_selectins_union_preflight_once_and_share_refreshed_binding(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    engine.load_schema()
    refreshed_raw = copy.deepcopy(schema_fixture)
    refreshed_raw['FutureA'] = copy.deepcopy(refreshed_raw['Sequence'])
    refreshed_raw['FutureB'] = copy.deepcopy(refreshed_raw['CustomEntity03'])

    def refreshed_schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return refreshed_raw

    monkeypatch.setattr(fake_sg, 'schema_read', refreshed_schema_read)
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_sequence': {'type': 'FutureA', 'id': 8},
                'sg_vendor': {'type': 'FutureB', 'id': 9},
            }
        ],
        [{'type': 'FutureA', 'id': 8, 'code': 'A'}],
        [{'type': 'FutureB', 'id': 9, 'code': 'B'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    shot = session.scalars(
        select(shot_cls).options(
            selectinload(shot_cls.sg_sequence),
            selectinload(shot_cls.sg_vendor),
        )
    )[0]

    assert (shot.sg_sequence.code, shot.sg_vendor.code) == ('A', 'B')
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert [call[1][0] for call in _find_calls(fake_sg)] == [
        'Shot',
        'FutureA',
        'FutureB',
    ]


def test_multiple_selectins_union_shared_target_reads(
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    schema_fixture['Version']['reviewer'] = copy.deepcopy(schema_fixture['Version']['user'])

    def schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return copy.deepcopy(schema_fixture)

    monkeypatch.setattr(fake_sg, 'schema_read', schema_read)
    fake_sg._find_results = [
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
    engine = create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    session = Session(engine)
    version_cls = engine.registry.entity_class('Version')

    versions = session.scalars(
        select(version_cls).options(
            selectinload(version_cls.user),
            selectinload(version_cls.reviewer),
        )
    )

    target_calls = [call for call in _find_calls(fake_sg) if call[1][0] == 'HumanUser']
    assert len(target_calls) == 1
    assert target_calls[0][1][1] == [['id', 'in', [1, 2]]]
    assert [(version.user.name, version.reviewer.name) for version in versions] == [
        ('B', 'A'),
        ('A', 'B'),
    ]


def test_selectin_empty_results_and_refs_issue_no_post_find(
    engine: Engine, fake_sg: _FakeShotgun
) -> None:
    session = Session(engine)
    note_cls = engine.registry.entity_class('Note')
    fake_sg._find_results = [[], [{'type': 'Note', 'id': 1, 'note_links': []}]]

    assert session.scalars(select(note_cls).options(selectinload(note_cls.note_links))) == []
    empty_links = session.scalars(select(note_cls).options(selectinload(note_cls.note_links)))[0]

    assert empty_links.note_links == []
    assert len([call for call in fake_sg._calls if call[0] == 'find']) == 2


def test_selectin_is_one_find_per_target_type(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [
        [
            {
                'type': 'Note',
                'id': 1,
                'subject': 's1',
                'note_links': [
                    {'type': 'Shot', 'id': 10},
                    {'type': 'Version', 'id': 20},
                ],
            },
            {
                'type': 'Note',
                'id': 2,
                'subject': 's2',
                'note_links': [{'type': 'Shot', 'id': 11}],
            },
        ],
        [{'type': 'Shot', 'id': 10, 'code': 'A'}, {'type': 'Shot', 'id': 11, 'code': 'B'}],
        [{'type': 'Version', 'id': 20, 'code': 'v1'}],
    ]
    session = Session(engine)
    note_cls = engine.registry.entity_class('Note')

    notes = session.scalars(select(note_cls).options(selectinload(note_cls.note_links)))

    find_calls = [call for call in fake_sg._calls if call[0] == 'find']
    assert len(find_calls) == 3
    assert find_calls[1][1][0] == 'Shot'
    assert find_calls[1][1][1] == [['id', 'in', [10, 11]]]
    assert find_calls[1][1][2] == engine.schema['Shot'].readable_field_codes()
    assert find_calls[2][1][0] == 'Version'
    assert notes[0].note_links[0].code == 'A'
    assert notes[0].note_links[1].code == 'v1'


def test_selectin_preflights_all_observed_targets_before_secondary_query(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    engine.load_schema()
    fake_sg._find_results = [
        [
            {
                'type': 'Note',
                'id': 1,
                'note_links': [
                    {'type': 'FutureB', 'id': 20},
                    {'type': 'FutureA', 'id': 10},
                ],
            }
        ]
    ]
    session = Session(engine)
    note_cls = engine.registry.entity_class('Note')

    with pytest.raises(
        RelationshipTargetUnavailableError,
        match=r'Note\.note_links.*FutureA.*FutureB.*valid_types.*Shot.*Version',
    ):
        session.scalars(select(note_cls).options(selectinload(note_cls.note_links)))

    note = session.merge_dict({'type': 'Note', 'id': 1})
    links = entity_state(note).loaded['note_links']
    assert [entity_state(link).entity_type for link in links] == ['FutureB', 'FutureA']
    assert [type(link) for link in links] == [
        engine.registry.entity_class_for('FutureB', binding=None),
        engine.registry.entity_class_for('FutureA', binding=None),
    ]
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Note']


def test_selectin_preserves_secondary_provider_fault(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError('secondary find failed')
    original_find = fake_sg.find
    call_count = 0

    def track_call(call: _SyncCallback[_P, _R]) -> _SyncCallback[_P, _R]:
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise error
            return call(*args, **kwargs)

        return wrapper

    find = track_call(original_find)

    monkeypatch.setattr(fake_sg, 'find', find)
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 100}]}]]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    with pytest.raises(RuntimeError) as excinfo:
        session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))

    assert excinfo.value is error
    assert any('FindCall' in note for note in error.__notes__)


def test_selectin_recovers_exact_observed_opaque_target_for_secondary_phase(
    engine: Engine,
    fake_sg: _FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    schema_fixture: dict[str, Any],
) -> None:
    engine.load_schema()
    refreshed_raw = copy.deepcopy(schema_fixture)
    refreshed_raw['FutureVendor'] = copy.deepcopy(refreshed_raw['CustomEntity03'])

    def refreshed_schema_read() -> dict[str, Any]:
        fake_sg._record('schema_read')
        return refreshed_raw

    monkeypatch.setattr(fake_sg, 'schema_read', refreshed_schema_read)
    fake_sg._find_results = [
        [
            {
                'type': 'Shot',
                'id': 1,
                'sg_vendor': {'type': 'FutureVendor', 'id': 9},
            }
        ],
        [{'type': 'FutureVendor', 'id': 9, 'code': 'FV-9'}],
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    shot = session.scalars(select(shot_cls).options(selectinload(shot_cls.sg_vendor)))[0]

    opaque = shot.sg_vendor
    assert opaque.code == 'FV-9'
    assert session.merge_dict({'type': 'FutureVendor', 'id': 9}) is opaque
    assert [name for name, _, _ in fake_sg._calls].count('schema_read') == 2
    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Shot', 'FutureVendor']


def test_selectin_rejects_corrupt_non_entity_link_values_before_secondary_query(
    engine: Engine,
    fake_sg: _FakeShotgun,
) -> None:
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 1, 'assets': [{'type': 'Asset', 'id': 2}, 'corrupt']}]
    ]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')

    with pytest.raises(SgalchemyError, match=r'Shot\.assets.*Entity'):
        session.scalars(select(shot_cls).options(selectinload(shot_cls.assets)))

    assert [call[1][0] for call in _find_calls(fake_sg)] == ['Shot']


def test_unloaded_relationship_raises_not_queries(engine: Engine, fake_sg: _FakeShotgun) -> None:
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A'}]]
    session = Session(engine)
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls).options(load_only(shot_cls.code)))[0]
    calls_before = len(fake_sg._calls)

    with pytest.raises(UnfetchedFieldError) as excinfo:
        _ = shot.assets

    assert 'selectinload' in str(excinfo.value)
    assert len(fake_sg._calls) == calls_before
