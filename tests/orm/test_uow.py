import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Never, Self, cast

import pytest
from _fakes import _FakeShotgun
from _support import _preserve_signature, _SyncSessionBundle

from sgalchemy import _errors
from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._registry import (
    Entity,
    EntityState,
    Registry,
    entity_state,
    initialize_entity,
)
from sgalchemy._core._statement import select
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm import _uow
from sgalchemy._orm._session import Session
from sgalchemy._orm._uow import CommitOutcome, plan_commit
from sgalchemy.schema._model import EntitySchema, FieldSchema

if TYPE_CHECKING:
    from _collections_abc import dict_items


class _BytesPath:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __fspath__(self) -> bytes:
        return os.fsencode(self._path)


class _CountingEntityList(list[Entity]):
    _visits = 0

    def __iter__(self) -> Iterator[Entity]:
        for obj in super().__iter__():
            self._visits += 1
            yield obj


class _CountingIdentityMap(dict[tuple[str, int], Entity]):
    _items_calls = 0

    def items(self) -> 'dict_items[tuple[str, int], Entity]':
        self._items_calls += 1
        return super().items()


class _CountingTuple(tuple[_uow._PlanItem, ...]):
    __slots__ = ()

    _iterations: ClassVar[int] = 0

    def __new__(cls, values: tuple[_uow._PlanItem, ...]) -> Self:
        instance = super().__new__(cls, values)
        cls._iterations = 0
        return instance

    def __iter__(self) -> Iterator[_uow._PlanItem]:
        type(self)._iterations += 1
        return super().__iter__()


def _cold_session(fake_sg: _FakeShotgun, tmp_path: Path) -> tuple[Session, Engine, _FakeShotgun]:
    registry = Registry()
    source = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'source',
        registry=registry,
        bind_default=False,
    )
    source.load_schema()
    destination_sg = type(fake_sg)()
    destination = create_engine(
        connection=destination_sg,
        cache_root=tmp_path / 'destination',
        registry=registry,
        bind_default=False,
    )
    return Session(destination), source, destination_sg


def test_add_delete_revive_transitions_cancel_without_io(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='cancel-create')
    session.add(pending)
    session.delete(pending)
    assert entity_state(pending).session is None
    with pytest.raises(SgalchemyError, match='pending'):
        session.revive(pending)

    active = session.merge_dict({'type': 'Shot', 'id': 21, 'code': 'active'})
    session.delete(active)
    session.delete(active)
    session.revive(active)

    retired = session.merge_dict({'type': 'Shot', 'id': 22, 'code': 'retired'})
    entity_state(retired).deleted = True
    session.revive(retired)
    session.revive(retired)
    session.delete(retired)

    calls_before = len(fake_sg._calls)
    session.commit()
    assert len(fake_sg._calls) == calls_before


def test_canceled_pending_reference_rejected_before_io(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    sequence = engine.entities.Sequence.new(code='canceled')
    shot = engine.entities.Shot.new(code='dependent', sg_sequence=sequence)
    session.add(sequence)
    session.add(shot)
    session.delete(sequence)
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match=r'pending relationship.*not staged'):
        session.commit()
    assert len(fake_sg._calls) == calls_before


def test_cold_commit_prepares_destination_and_commits_upload(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    registry = Registry()
    source = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'source',
        registry=registry,
        bind_default=False,
    )
    source.load_schema()
    pending = source.entities.Shot.new(code='upload', image=str(tmp_path / 'image.jpg'))
    (tmp_path / 'image.jpg').write_bytes(b'image')

    destination_sg = type(fake_sg)()
    destination = create_engine(
        connection=destination_sg,
        cache_root=tmp_path / 'destination',
        registry=registry,
        bind_default=False,
    )
    session = Session(destination)
    session.add(pending)

    session.commit()

    names = [name for name, _, _ in destination_sg._calls]
    assert names.count('schema_read') == 1
    assert names.count('batch') == 1
    assert names.count('upload_thumbnail') == 1
    assert names.count('find') == 1


def test_cold_local_preflight_rejects_missing_dirty_provenance_without_schema_io(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    session, _source, destination_sg = _cold_session(fake_sg, tmp_path)
    shot = session.merge_dict({'type': 'Shot', 'id': 41, 'description': 'before'})
    shot.description = 'after'
    entity_state(shot).dirty_types.pop('description')

    with pytest.raises(SgalchemyError, match='classification is missing'):
        session.commit()

    assert destination_sg._calls == []


def test_cold_local_preflight_rejects_pending_dependency_cycle_without_schema_io(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    session, source, destination_sg = _cold_session(fake_sg, tmp_path)
    shot = source.entities.Shot.new(code='S')
    asset = source.entities.Asset.new(code='A', shots=[shot])
    shot.assets = [asset]
    session.add(shot)
    session.add(asset)

    with pytest.raises(SgalchemyError, match='cycle'):
        session.commit()

    assert destination_sg._calls == []


def test_commit_compiles_one_batch_with_dirty_fields_only(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A', 'description': 'd'}]]
    shot_cls = engine.registry.entity_class('Shot')
    shot = session.scalars(select(shot_cls))[0]
    shot.description = 'new'
    project = session.merge_dict({'type': 'Project', 'id': 70, 'name': 'Demo'})
    new_shot = shot_cls.new(code='B_010', project=project)
    session.add(new_shot)
    session.commit()
    batch_calls = [c for c in fake_sg._calls if c[0] == 'batch']
    assert len(batch_calls) == 1
    requests = batch_calls[0][1][0]
    assert {
        'request_type': 'update',
        'entity_type': 'Shot',
        'entity_id': 1,
        'data': {'description': 'new'},
    } in requests
    create = next(r for r in requests if r['request_type'] == 'create')
    assert create['data'] == {'code': 'B_010', 'project': {'type': 'Project', 'id': 70}}
    assert new_shot.id == 9001  # first create id minted by FakeShotgun's counter
    assert not shot._sg_state.dirty
    assert not shot._sg_state.dirty_types


def test_commit_outcome_contains_only_exact_committed_items() -> None:
    earlier = cast('_uow._PlanItem', object())
    active = cast('_uow._PlanItem', object())
    outcome = CommitOutcome(
        committed=[earlier],
        staged=[active],
        indeterminate=[],
        withdrawn=[],
    )

    assert not outcome.contains_committed((active,))
    assert outcome.contains_committed((earlier,))


def test_commit_outcome_removes_exact_items_with_one_items_pass() -> None:
    class AlwaysEqual:
        def __eq__(self, _other: object) -> bool:
            return True

        def __hash__(self) -> int:
            raise TypeError

    keep_first = cast('_uow._PlanItem', object())
    remove_first = cast('_uow._PlanItem', AlwaysEqual())
    equal_but_distinct = cast('_uow._PlanItem', AlwaysEqual())
    remove_second = cast('_uow._PlanItem', object())
    keep_last = cast('_uow._PlanItem', object())
    items = _CountingTuple((remove_first, remove_second))
    outcome = CommitOutcome(
        committed=[],
        staged=[keep_first, remove_first, equal_but_distinct, remove_second, keep_last],
        indeterminate=[],
        withdrawn=[],
    )

    outcome._remove_staged(items)

    assert len(outcome.staged) == 3
    assert outcome.staged[0] is keep_first
    assert outcome.staged[1] is equal_but_distinct
    assert outcome.staged[2] is keep_last
    assert items._iterations == 1


def test_context_exit_discards_dirty_and_pending_work(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    with session:
        shot = session.merge_dict({'type': 'Shot', 'id': 11, 'code': 'before'})
        shot.code = 'after'
        pending = engine.entities.Shot.new(code='pending')
        session.add(pending)
        session.delete(shot)
    assert shot.code == 'before'
    assert not entity_state(shot).dirty_types
    assert not entity_state(pending).dirty_types
    assert entity_state(pending).session is None
    calls_before = len(fake_sg._calls)
    session.commit()
    assert len(fake_sg._calls) == calls_before


def test_create_result_rejects_existing_identity_collision(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    existing = session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'existing'})
    pending = engine.entities.Shot.new(code='pending')
    session.add(pending)
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 41, 'code': 'pending'}]]

    with pytest.raises(CommitError, match='already tracks a different') as excinfo:
        session.commit()
    assert excinfo.value.committed == []
    assert excinfo.value.staged == []
    assert len(excinfo.value.indeterminate) == 1
    assert excinfo.value.withdrawn == []
    assert session._state.identity_map == {('Shot', 41): existing}
    assert session._identity_values() == [existing]
    assert entity_state(existing).session is session
    pending_state = entity_state(pending)
    assert pending_state.id is None
    assert pending_state.loaded == {'code': 'pending'}
    assert pending_state.dirty == {}
    assert pending_state.clean == {}
    assert pending_state.dirty_types == {}
    assert pending_state.indeterminate


def test_create_result_rejects_existing_identity_collision_after_owner_drift(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    existing = session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'existing'})
    entity_state(existing).id = 42
    pending = engine.entities.Shot.new(code='pending')
    session.add(pending)
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 41, 'code': 'pending'}]]

    with pytest.raises(CommitError, match='already tracks a different') as excinfo:
        session.commit()
    assert excinfo.value.committed == []
    assert excinfo.value.staged == []
    assert len(excinfo.value.indeterminate) == 1
    assert excinfo.value.withdrawn == []
    assert session._state.identity_map == {('Shot', 41): existing}
    assert session._state.identity_map[('Shot', 41)] is existing
    assert entity_state(existing).id == 42
    assert entity_state(existing).session is session
    pending_state = entity_state(pending)
    assert pending_state.id is None
    assert pending_state.loaded == {'code': 'pending'}
    assert pending_state.dirty == {}
    assert pending_state.indeterminate


def test_create_result_rejects_staged_identity_collision(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    existing = source.merge_dict({'type': 'Shot', 'id': 43, 'code': 'existing'})
    source.close()
    session.delete(existing)
    pending = engine.entities.Shot.new(code='pending')
    session.add(pending)
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 43, 'code': 'pending'}, True]]

    with pytest.raises(CommitError, match='already tracks a different') as excinfo:
        session.commit()

    assert excinfo.value.committed == []
    assert excinfo.value.staged == []
    assert len(excinfo.value.indeterminate) == 2
    assert excinfo.value.withdrawn == []
    assert session._identity_values() == [existing]
    assert entity_state(pending).id is None
    assert entity_state(pending).indeterminate
    assert entity_state(existing).indeterminate


def test_create_results_reject_duplicate_identity_collision(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    first = engine.entities.Shot.new(code='first')
    second = engine.entities.Shot.new(code='second')
    session.add(first)
    session.add(second)
    fake_sg._batch_results = [
        [
            {'type': 'Shot', 'id': 42, 'code': 'first'},
            {'type': 'Shot', 'id': 42, 'code': 'second'},
        ]
    ]

    with pytest.raises(CommitError, match='duplicate identity') as excinfo:
        session.commit()
    assert excinfo.value.committed == []
    assert excinfo.value.staged == []
    assert len(excinfo.value.indeterminate) == 2
    assert excinfo.value.withdrawn == []
    assert session._state.identity_map == {}
    assert session._identity_values() == []
    assert entity_state(first).id is None
    assert entity_state(second).id is None
    assert entity_state(first).indeterminate
    assert entity_state(second).indeterminate


def test_custom_image_schema_cannot_opt_field_into_uow_write_semantics() -> None:
    field = FieldSchema(
        code='sg_custom_thumbnail',
        data_type='image',
        python_type=str,
        api_readable=True,
        valid_types=(),
        display_name='Custom Thumbnail',
    )
    schema = EntitySchema(
        type_name='Shot',
        display_name='Shot',
        fields={field.code: field},
        name_field=None,
    )
    shot = Entity.__new__(Entity)
    initialize_entity(
        shot,
        EntityState(
            entity_type='Shot',
            id=47,
            dirty={field.code: '/renders/custom.jpg'},
            dirty_types={field.code: 'image'},
        ),
    )

    with pytest.raises(SgalchemyError, match=r'unsupported image upload field code'):
        _uow._operation_fields(schema, shot)


def test_delete_and_revive_stage(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    fake_sg._find_results = [[{'type': 'Shot', 'id': 3, 'code': 'C'}]]
    shot = session.scalars(select(engine.registry.entity_class('Shot')))[0]
    session.delete(shot)
    session.commit()
    requests = next(c for c in fake_sg._calls if c[0] == 'batch')[1][0]
    assert requests == [{'request_type': 'delete', 'entity_type': 'Shot', 'entity_id': 3}]
    session.revive(shot)
    session.commit()
    assert ('revive', ('Shot', 3), {}) in fake_sg._calls
    session.commit()
    assert [name for name, _, _ in fake_sg._calls].count('revive') == 1


def test_dirty_retired_object_never_loads_schema_for_recovery(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    retired = session.merge_dict({'type': 'Shot', 'id': 81, 'description': 'before'})
    session.delete(retired)
    session.commit()
    retired.description = 'after'

    def unexpected_load(*, force: bool = False) -> Never:
        del force
        raise AssertionError('dirty retired state must not load schema')

    monkeypatch.setattr(engine, 'load_binding', unexpected_load)
    calls_before = len(fake_sg._calls)

    with pytest.raises(_errors.RetiredEntityWriteError, match=r'revive\(obj\).*rollback'):
        session.commit()

    assert len(fake_sg._calls) == calls_before


def test_entity_reference_collection_tolerates_shared_cyclic_containers() -> None:
    linked = Entity.__new__(Entity)
    initialize_entity(linked, EntityState(entity_type='Sequence', id=7))
    shared = [linked]
    cyclic: dict[str, Any] = {'left': shared, 'right': shared}
    cyclic['self'] = cyclic

    assert _uow._entity_references(cyclic) == [linked]


def test_foreign_registry_entities_rejected(sess: _SyncSessionBundle, tmp_path: Path) -> None:
    session, _engine, fake_sg = sess
    other = create_engine(
        connection=fake_sg, cache_root=tmp_path, registry=Registry(), bind_default=False
    )
    other.load_schema()
    foreign_cls = other.entities.Shot
    foreign_session = type(session)(other)
    foreign = foreign_session.merge_dict({'type': 'Shot', 'id': 41, 'code': 'X'})
    pending = foreign_cls.new(code='pending')

    with pytest.raises(SgalchemyError, match='different registry'):
        session.get(foreign_cls, 41)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.scalars(select(foreign_cls))
    with pytest.raises(SgalchemyError, match='different registry'):
        session.expire(foreign)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.refresh(foreign)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.add(pending)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.delete(foreign)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.revive(foreign)


def test_invalid_and_noop_transitions_do_not_attach_detached_entities(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()

    source = type(session)(engine)
    active = source.merge_dict({'type': 'Shot', 'id': 23, 'code': 'active'})
    source.close()
    with pytest.raises(SgalchemyError, match='active entity cannot be revived'):
        session.revive(active)
    assert entity_state(active).session is None
    other = type(session)(engine)
    other.delete(active)
    assert entity_state(active).session is other

    source = type(session)(engine)
    retired = source.merge_dict({'type': 'Shot', 'id': 24, 'code': 'retired'})
    entity_state(retired).deleted = True
    source.close()
    session.delete(retired)
    assert entity_state(retired).session is None
    other.revive(retired)
    assert entity_state(retired).session is other


def test_merge_delete_commit_prepares_schema_automatically(
    sess: _SyncSessionBundle,
) -> None:
    session, _engine, fake_sg = sess
    shot = session.merge_dict({'type': 'Shot', 'id': 31})
    session.delete(shot)

    session.commit()

    assert entity_state(shot).deleted
    names = [name for name, _args, _kwargs in fake_sg._calls]
    assert names.count('schema_read') == 1
    assert names.count('batch') == 1


def test_pending_attachment_is_single_session_and_add_is_idempotent(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    other = type(session)(engine)
    pending = engine.entities.Shot.new(code='one-create')
    session.add(pending)
    session.add(pending)
    with pytest.raises(SgalchemyError, match='another open session') as excinfo:
        other.add(pending)
    assert 'close' in str(excinfo.value)
    assert 'rollback' not in str(excinfo.value)
    persistent = session.merge_dict({'type': 'Shot', 'id': 31, 'code': 'owned'})
    with pytest.raises(SgalchemyError, match='another open session'):
        other.delete(persistent)
    session.commit()
    requests = next(call for call in fake_sg._calls if call[0] == 'batch')[1][0]
    assert [request['request_type'] for request in requests] == ['create']


def test_pending_thumbnail_clear_produces_no_image_request(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='clear', image=None)
    session.add(pending)

    session.commit()

    batch = next(call for call in fake_sg._calls if call[0] == 'batch')
    assert batch[1][0][0]['data'] == {'code': 'clear'}
    assert not any(name.startswith('upload_') for name, _, _ in fake_sg._calls)


def test_plan_commit_enumerates_identity_map_once_per_attempt(sess: _SyncSessionBundle) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    for entity_id in range(1, 21):
        shot = session.merge_dict({'type': 'Shot', 'id': entity_id, 'code': 'old'})
        shot.code = f'new-{entity_id}'
    counting = _CountingIdentityMap(session._state.identity_map)
    session._state._identity = counting

    plan = session._plan_commit()

    assert plan._waves
    assert counting._items_calls == 1


def test_planning_accepts_repeated_acyclic_shared_containers(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
) -> None:
    session, engine, fake_sg = sess
    schema_fixture['Shot']['description']['data_type']['value'] = 'serializable'
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 23, 'code': 'shared'})
    shared = {'value': 1}
    payload = {'left': shared, 'right': shared, 'nested': [shared]}
    shot.description = payload
    binding = session.load_binding('raised while planning')

    plan = plan_commit(session, binding, session._state.identity_map)

    assert plan._waves
    assert _uow._wire_value(payload) == payload


def test_retired_query_refresh_and_dirty_revival_commit(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    fake_sg._find_results = [
        [{'type': 'Shot', 'id': 41, 'description': 'retired'}],
        [{'type': 'Shot', 'id': 41, 'description': 'refreshed'}],
        [{'type': 'Shot', 'id': 41, 'description': 'query-after-revive'}],
    ]
    shot = session.scalars(select(engine.entities.Shot).execution_options(retired_only=True))[0]
    session.refresh(shot)
    assert [call for call in fake_sg._calls if call[0] == 'find'][-1][2]['retired_only'] is True
    assert shot.description == 'refreshed'
    shot.description = 'local'
    calls_before = len(fake_sg._calls)
    with pytest.raises(_errors.RetiredEntityWriteError, match=r'revive\(obj\)'):
        session.commit()
    assert len(fake_sg._calls) == calls_before
    assert entity_state(shot).dirty == {'description': 'local'}
    session.revive(shot)
    assert session.scalars(select(engine.entities.Shot))[0] is shot
    assert entity_state(shot).deleted is True
    assert session.unit_of_work.revived[0] is shot
    calls_before = len(fake_sg._calls)

    session.commit()

    assert [call[0] for call in fake_sg._calls[calls_before:]] == ['revive', 'batch']
    assert fake_sg._calls[-1][1][0] == [
        {
            'request_type': 'update',
            'entity_type': 'Shot',
            'entity_id': 41,
            'data': {'description': 'local'},
        }
    ]
    assert entity_state(shot).deleted is False
    assert entity_state(shot).dirty == {}
    assert session.unit_of_work.revived == ()
    session.commit()
    assert [call[0] for call in fake_sg._calls[calls_before:]] == ['revive', 'batch']


def test_rollback_discards_pending_identities_in_one_pass(sess: _SyncSessionBundle) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    persistent = {
        ('Shot', entity_id): session.merge_dict({'type': 'Shot', 'id': entity_id, 'code': 'old'})
        for entity_id in range(1, 21)
    }
    pending = [engine.entities.Shot.new(code=f'new-{index}') for index in range(20)]
    for obj in pending:
        session.add(obj)
    counting = _CountingIdentityMap(session._state.identity_map)
    session._state._identity = counting

    session.rollback()

    assert counting._items_calls <= 1
    assert all(entity_state(obj).session is None for obj in pending)
    assert counting == persistent


def test_rollback_restores_snapshots(sess: _SyncSessionBundle, tmp_path: Path) -> None:
    session, engine, fake_sg = sess
    fake_sg._find_results = [[{'type': 'Shot', 'id': 1, 'code': 'A'}]]
    shot = session.scalars(select(engine.registry.entity_class('Shot')))[0]
    shot.code = 'CHANGED'
    pending = engine.registry.entity_class('Shot').new(code='NEW', image=str(tmp_path / 'new.jpg'))
    session.add(pending)
    session.delete(shot)
    session.rollback()
    assert shot.code == 'A'
    assert not entity_state(shot).dirty_types
    assert not entity_state(pending).dirty_types
    assert entity_state(pending).session is None
    session.commit()
    assert not [c for c in fake_sg._calls if c[0] in {'batch', 'upload_thumbnail'}]


def test_schema_refresh_display_name_change_preserves_write_classification(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    image_path = tmp_path / 'schema-refresh.jpg'
    image_path.write_bytes(b'image')
    pending = engine.entities.Shot.new(code='upload', image=image_path)
    session.add(pending)
    schema_fixture['Shot']['image']['name']['value'] = 'New Thumbnail Label'
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema(force=True)
    calls_before = len(fake_sg._calls)

    session.commit()
    assert [name for name, _, _ in fake_sg._calls[calls_before:]] == [
        'batch',
        'upload_thumbnail',
        'find',
    ]


def test_second_assignment_preserves_first_field_provenance(sess: _SyncSessionBundle) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 42, 'description': 'old'})
    state = entity_state(shot)
    state.unverified_writes.add('description')

    shot.description = 'first'
    shot.description = 'second'

    assert state.clean == {'description': 'old'}
    assert state.clean_unverified_writes == {'description'}
    session.rollback()
    assert shot.description == 'old'
    assert state.unverified_writes == {'description'}


def test_shared_staged_intent_snapshot_preserves_lifecycle_order(sess: _SyncSessionBundle) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    revived = session.merge_dict({'type': 'Shot', 'id': 1, 'description': 'same'})
    entity_state(revived).deleted = True
    session.revive(revived)
    revived.description = 'changed'
    dirty = session.merge_dict({'type': 'Shot', 'id': 2, 'description': 'same'})
    dirty.description = 'changed'
    pending = engine.entities.Shot.new(description='same')
    session.add(pending)
    deleted = session.merge_dict({'type': 'Shot', 'id': 3, 'description': 'same'})
    session.delete(deleted)

    staged = _uow.staged_intent_snapshot(session.unit_of_work, session._identity_values())

    expected = (revived, dirty, pending, deleted)
    assert len(staged) == len(expected)
    assert all(
        actual is expected_item for actual, expected_item in zip(staged, expected, strict=True)
    )


def test_snapshot_alias_does_not_restore_caller_mutation(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
) -> None:
    session, engine, fake_sg = sess
    schema_fixture['Shot']['description']['data_type']['value'] = 'serializable'
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema(force=True)
    payload = {'values': [1]}
    shot = session.merge_dict({'type': 'Shot', 'id': 41, 'description': payload})

    shot.description = {'values': [2]}
    payload['values'].append(3)
    session.rollback()

    assert shot.description == {'values': [1]}
    assert shot.description is not payload


def test_sole_batch_fault_preserves_staging(sess: _SyncSessionBundle) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    pending = engine.entities.Shot.new(code='retry')
    session.add(pending)
    fault = RuntimeError('provider failed')
    fake_sg._errors['batch'] = [fault]
    with pytest.raises(RuntimeError) as excinfo:
        session.commit()
    assert excinfo.value is fault
    assert entity_state(pending).dirty == {'code': 'retry'}
    assert entity_state(pending).dirty_types == {'code': 'text'}
    session.commit()
    assert pending.id == 9001


@pytest.mark.parametrize(
    ('transition', 'category'), [('add', 'new'), ('delete', 'deleted'), ('revive', 'revived')]
)
def test_staging_categories_expose_read_only_snapshots(
    sess: _SyncSessionBundle,
    transition: str,
    category: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    if transition == 'add':
        obj = engine.entities.Shot.new(code='pending')
    else:
        fake_sg._find_results = [[{'type': 'Shot', 'id': 71, 'code': 'persistent'}]]
        obj = session.scalars(
            select(engine.entities.Shot).execution_options(retired_only=transition == 'revive')
        )[0]
    getattr(session, transition)(obj)
    unit_of_work = session.unit_of_work
    snapshot = cast('tuple[Entity, ...]', getattr(unit_of_work, category))

    with pytest.raises(AttributeError):
        cast('Any', snapshot).append(obj)
    with pytest.raises(AttributeError):
        setattr(unit_of_work, category, ())
    assert type(snapshot) is tuple
    assert len(snapshot) == 1
    assert snapshot[0] is obj

    session.rollback()

    assert getattr(unit_of_work, category) == ()
    assert len(snapshot) == 1
    assert snapshot[0] is obj


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_interrupted_clean_transition_rejects_retry_before_io(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    fake_sg._find_results = [[{'type': 'Shot', 'id': 71, 'code': 'clean'}]]
    obj = session.scalars(
        select(engine.entities.Shot).execution_options(retired_only=transition == 'revive')
    )[0]
    assert not entity_state(obj).dirty
    getattr(session, transition)(obj)
    method = 'batch' if transition == 'delete' else 'revive'
    fake_sg._errors[method] = [KeyboardInterrupt('submitted transition interrupted')]

    with pytest.raises(KeyboardInterrupt, match='submitted transition interrupted'):
        session.commit()

    assert entity_state(obj).indeterminate
    assert not entity_state(obj).dirty
    assert (
        getattr(session.unit_of_work, 'deleted' if transition == 'delete' else 'revived')[0] is obj
    )
    calls_before = len(fake_sg._calls)
    assert fake_sg._calls[-1][0] == method
    with pytest.raises(SgalchemyError, match='indeterminate'):
        session.commit()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('category', ['new', 'deleted', 'revived'])
def test_bulk_staging_removal_preserves_exact_survivors_and_membership(
    sess: _SyncSessionBundle,
    category: _uow._StagingCategory,
) -> None:
    _session, engine, _fake_sg = sess
    engine.load_schema()
    objects = [engine.entities.Shot.new(code='same-code') for _ in range(5)]
    absent = engine.entities.Shot.new(code='same-code')
    unit_of_work = _uow._SessionUnitOfWork()
    for obj in objects:
        unit_of_work._append_staged(category, obj)
    queue = getattr(unit_of_work, f'_{category}')
    expected = [objects[0], objects[2], objects[4]]

    unit_of_work._discard_staged_many(category, [objects[3], absent, objects[1], objects[3]])

    assert getattr(unit_of_work, f'_{category}') is queue
    assert [id(obj) for obj in queue] == [id(obj) for obj in expected]
    assert [unit_of_work._contains_staged(category, obj) for obj in objects] == [
        True,
        False,
        True,
        False,
        True,
    ]
    unit_of_work._discard_staged_many(category, [objects[1], objects[3]])
    assert [id(obj) for obj in queue] == [id(obj) for obj in expected]
    assert unit_of_work._append_staged(category, objects[1])
    assert [id(obj) for obj in queue] == [id(obj) for obj in [*expected, objects[1]]]


@pytest.mark.parametrize('count', [20, 40])
def test_retired_write_validation_visits_revival_intent_once(
    sess: _SyncSessionBundle, count: int
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    revived = _CountingEntityList()
    for entity_id in range(1, count + 1):
        obj = session.merge_dict({'type': 'Shot', 'id': entity_id, 'code': 'old'})
        entity_state(obj).deleted = True
        obj.code = 'new'
        revived.append(obj)
    dirty = list(revived)
    revived._visits = 0

    _uow._validate_retired_write_intent(dirty, revived)

    assert revived._visits == count


@pytest.mark.parametrize('operation', ['delete', 'add'])
def test_staging_mapped_and_pending_objects_skips_identity_map_scans(
    sess: _SyncSessionBundle,
    operation: str,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    persistent = [
        session.merge_dict({'type': 'Shot', 'id': entity_id, 'code': 'old'})
        for entity_id in range(1, 41)
    ]
    counting = _CountingIdentityMap(session._state.identity_map)
    session._state._identity = counting

    if operation == 'delete':
        for obj in persistent:
            session.delete(obj)
    else:
        for index in range(40):
            session.add(engine.entities.Shot.new(code=f'new-{index}'))

    assert counting._items_calls == 0


def test_staging_preflight_rejects_duplicate_identity_between_staged_objects_before_io(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    staged: list[Entity] = []
    for code in ('first', 'second'):
        source = type(session)(engine)
        obj = source.merge_dict({'type': 'Shot', 'id': 41, 'code': code})
        source.close()
        session.attach_for_staging(obj)
        staged.append(obj)
    session.unit_of_work._deleted.extend(staged)
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError) as excinfo:
        session.commit()

    assert str(excinfo.value) == (
        'session already tracks a different Shot(id=41) object; use '
        'session.get(type(obj), obj.id) to retrieve it, or close this session '
        'before staging the detached copy'
    )
    assert len(fake_sg._calls) == calls_before


def test_staging_preflight_rejects_identity_map_collision_before_io(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    staged = source.merge_dict({'type': 'Shot', 'id': 41, 'code': 'staged'})
    source.close()
    session.attach_for_staging(staged)
    session.unit_of_work._deleted.append(staged)
    source = type(session)(engine)
    tracked = source.merge_dict({'type': 'Shot', 'id': 41, 'code': 'tracked'})
    source.close()
    session._state.identity_map[('Shot', 41)] = tracked
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError) as excinfo:
        session.commit()

    assert str(excinfo.value) == (
        'session already tracks a different Shot(id=41) object; use '
        'session.get(type(obj), obj.id) to retrieve it, or close this session '
        'before staging the detached copy'
    )
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('transition', ['add', 'delete', 'revive'])
def test_supported_bulk_staging_uses_exact_membership_index(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()

    def reject_scan(_items: list[Entity], _obj: Entity) -> Never:
        raise AssertionError('supported staging scanned an ordered list')

    monkeypatch.setattr(_uow, '_contains_identity', reject_scan)
    objects: list[Entity] = []
    for index in range(64):
        if transition == 'add':
            obj = engine.entities.Shot.new(code=f'pending-{index}')
        else:
            obj = session.merge_dict({'type': 'Shot', 'id': index + 1})
            entity_state(obj).deleted = transition == 'revive'
        getattr(session, transition)(obj)
        objects.append(obj)

    categories = {'add': 'new', 'delete': 'deleted', 'revive': 'revived'}
    staged = getattr(session.unit_of_work, categories[transition])
    assert all(actual is expected for actual, expected in zip(staged, objects, strict=True))


@pytest.mark.parametrize('before', ['refresh', 'unverified'])
def test_close_restores_field_provenance_before_detaching(
    sess: _SyncSessionBundle, before: str
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 41})
    state = entity_state(shot)
    if before == 'refresh':
        state.refresh_required.add('description')
    else:
        state.loaded['description'] = 'old'
        state.unverified_writes.add('description')

    shot.description = 'new'
    session.close()

    assert state.session is None
    assert state.access_guard is None
    assert state.loaded.get('description') == (None if before == 'refresh' else 'old')
    assert state.refresh_required == ({'description'} if before == 'refresh' else set())
    assert state.unverified_writes == ({'description'} if before == 'unverified' else set())
    assert not state.dirty
    assert not state.clean
    assert not state.clean_refresh_required
    assert not state.clean_unverified_writes


def test_thumbnail_clear_filters_batch_echo_and_applies_coupled_clear(
    sess: _SyncSessionBundle,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 49,
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )
    shot.image = None
    fake_sg._batch_results = [
        [
            {
                'type': 'Shot',
                'id': 49,
                'image': 'stale-image-echo',
                'filmstrip_image': 'stale-filmstrip-echo',
            }
        ]
    ]

    session.commit()

    state = entity_state(shot)
    assert state.loaded['image'] is None
    assert state.loaded['filmstrip_image'] is None
    assert not [call for call in fake_sg._calls if call[0] == 'find']


def test_thumbnail_clear_preserves_later_dirty_filmstrip(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 50,
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )
    path = tmp_path / 'later-filmstrip.jpg'
    path.write_bytes(b'image')
    shot.image = None
    shot.filmstrip_image = path
    fake_sg._batch_results = [[{'type': 'Shot', 'id': 50}]]
    fake_sg._find_results = [[{'type': 'Shot', 'id': 50, 'filmstrip_image': 'server-filmstrip'}]]

    session.commit()

    state = entity_state(shot)
    assert state.loaded['image'] is None
    assert state.loaded['filmstrip_image'] == 'server-filmstrip'
    assert [name for name, _, _ in fake_sg._calls].count('upload_filmstrip_thumbnail') == 1


def test_thumbnail_upload_accepts_bytes_pathlike(sess: _SyncSessionBundle, tmp_path: Path) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    path = tmp_path / 'bytes-path.jpg'
    path.write_bytes(b'image')
    shot = session.merge_dict({'type': 'Shot', 'id': 7, 'image': 'old'})
    shot.image = _BytesPath(path)

    session.commit()

    upload = next(call for call in fake_sg._calls if call[0] == 'upload_thumbnail')
    provider_path = upload[1][2]
    assert type(provider_path) is str
    assert os.fsencode(provider_path) == os.fsencode(path)


def test_unique_entities_does_not_scan_accumulated_results(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session, engine, _fake_sg = sess
    engine.load_schema()
    candidates = [engine.entities.Shot.new(code=f'Shot-{index}') for index in range(64)]
    contains_calls = 0
    original_contains = _uow._contains_identity

    def count_contains(items: list[Entity], obj: Entity) -> bool:
        nonlocal contains_calls
        contains_calls += 1
        return original_contains(items, obj)

    monkeypatch.setattr(_uow, '_contains_identity', count_contains)

    unique = _uow._unique_entities(candidates)

    assert len(unique) == len(candidates)
    assert contains_calls == 0


def test_unique_entities_preserves_first_occurrence_by_exact_identity(
    sess: _SyncSessionBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session, engine, _fake_sg = sess
    engine.load_schema()
    shot_cls = engine.entities.Shot
    first = shot_cls.new(code='equal')
    second = shot_cls.new(code='equal')
    third = shot_cls.new(code='equal')

    def entities_are_equal(_left: object, right: object) -> bool:
        return isinstance(right, Entity)

    monkeypatch.setattr(type(first), '__eq__', entities_are_equal)

    assert first == second
    unique = _uow._unique_entities([first, second, first, third, second])
    assert len(unique) == 3
    assert all(
        actual is expected for actual, expected in zip(unique, (first, second, third), strict=True)
    )


def test_upload_field_commits_after_pending_create(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    image_path = tmp_path / 'image.jpg'
    image_path.write_bytes(b'image')
    pending = engine.entities.Shot.new(code='upload', image=str(image_path))
    session.add(pending)
    calls_before = len(fake_sg._calls)
    session.commit()
    assert [name for name, _, _ in fake_sg._calls[calls_before:]] == [
        'batch',
        'upload_thumbnail',
        'find',
    ]
    assert entity_state(pending).dirty == {}
    assert entity_state(pending).dirty_types == {}


def test_upload_path_preflight_normalizes_custom_pathlike(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    class CustomPath:
        def __init__(self, value: str) -> None:
            self._value = value

        def __fspath__(self) -> str:
            return self._value

    session, engine, fake_sg = sess
    engine.load_schema()
    path = tmp_path / 'custom-path.jpg'
    path.write_bytes(b'image')
    shot = session.merge_dict({'type': 'Shot', 'id': 47, 'code': 'upload'})
    shot.image = CustomPath(str(path))

    session.commit()

    upload = next(call for call in fake_sg._calls if call[0] == 'upload_thumbnail')
    assert upload[1] == ('Shot', 47, str(path))


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='os.mkfifo is unavailable')
def test_upload_path_preflight_rejects_fifo_without_blocking(
    sess: _SyncSessionBundle,
    tmp_path: Path,
) -> None:
    if sys.platform == 'win32':
        pytest.skip('os.mkfifo is unavailable')

    session, engine, fake_sg = sess
    engine.load_schema()
    path = tmp_path / 'thumbnail.fifo'
    os.mkfifo(path)
    shot = session.merge_dict({'type': 'Shot', 'id': 48, 'code': 'upload'})
    shot.image = path
    calls_before = len(fake_sg._calls)
    blocked = False

    with ThreadPoolExecutor(max_workers=1) as pool:
        committing = pool.submit(session.commit)
        try:
            committing.result(timeout=1)
        except SgalchemyError:
            pass
        except FutureTimeoutError:
            blocked = True
            writer = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            os.close(writer)
            with pytest.raises(SgalchemyError):
                committing.result(timeout=2)
        else:
            pytest.fail('FIFO upload path was accepted')

    assert not blocked, 'FIFO preflight blocked waiting for a writer'
    assert len(fake_sg._calls) == calls_before
    assert entity_state(shot).dirty == {'image': path}


def test_upload_path_preflight_rejects_zero_byte_file(
    sess: _SyncSessionBundle, tmp_path: Path
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    empty = tmp_path / 'empty.jpg'
    empty.write_bytes(b'')
    shot = session.merge_dict({'type': 'Shot', 'id': 48, 'code': 'upload'})
    shot.image = empty
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='nonempty'):
        session.commit()

    assert len(fake_sg._calls) == calls_before
    assert entity_state(shot).dirty == {'image': empty}


@pytest.mark.parametrize('batch_size', [None, 1, 20])
def test_validate_batch_size_accepts_supported_values(batch_size: int | None) -> None:
    _uow.validate_batch_size(batch_size)


@pytest.mark.parametrize('batch_size', [True, False, 0, -1, 1.5, '1'])
def test_validate_batch_size_rejects_nonpositive_or_non_int_values(batch_size: object) -> None:
    with pytest.raises(ValueError, match='positive integer or None'):
        _uow.validate_batch_size(cast('int | None', batch_size))


@pytest.mark.parametrize('destination_site', ['same', 'different'])
def test_cold_shared_registry_commit_handles_site_ownership_before_writes(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    destination_site: str,
) -> None:
    registry = Registry()
    source = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'source',
        registry=registry,
        bind_default=False,
    )
    source.load_schema()
    pending = source.entities.Shot.new(code='bound-elsewhere')

    destination_sg = type(fake_sg)()
    if destination_site == 'different':
        destination_sg.base_url = 'https://other.example.com'
    destination = create_engine(
        connection=destination_sg,
        cache_root=tmp_path / 'destination',
        registry=registry,
        bind_default=False,
    )
    session = Session(destination)
    session.add(pending)
    if destination_site == 'same':
        session.commit()
        names = [name for name, _, _ in destination_sg._calls]
        assert names.count('schema_read') == 1
        assert names.count('batch') == 1
    else:
        with pytest.raises(SgalchemyError) as raised:
            session.commit()
        assert str(raised.value) == (
            'registry is already bound to a different site (one site per Registry); '
            'pass registry=Registry() or bind_default=False to the engine factory '
            'for a second site'
        )
        assert destination_sg._calls == []


@pytest.mark.parametrize('work_kind', ['create', 'delete', 'revive', 'update'])
def test_cold_shared_registry_commit_prepares_schema_for_all_work(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    work_kind: str,
) -> None:
    registry = Registry()
    source = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'source-all-work',
        registry=registry,
        bind_default=False,
    )
    source.load_schema()
    destination_sg = type(fake_sg)()
    destination = create_engine(
        connection=destination_sg,
        cache_root=tmp_path / 'destination-all-work',
        registry=registry,
        bind_default=False,
    )
    session = Session(destination)
    if work_kind == 'create':
        obj = source.entities.Shot.new(code='pending')
        session.add(obj)
    else:
        obj = session.merge_dict({'type': 'Shot', 'id': 64, 'code': 'before'})
        if work_kind == 'delete':
            session.delete(obj)
        elif work_kind == 'revive':
            entity_state(obj).deleted = True
            session.revive(obj)
        else:
            obj.code = 'after'

    session.commit()

    names = [name for name, _, _ in destination_sg._calls]
    assert names.count('schema_read') == 1
    assert names.count('batch') == (work_kind != 'revive')
    assert names.count('revive') == (work_kind == 'revive')


@pytest.mark.parametrize('shape', ['list', 'mapping', 'mixed'])
def test_commit_rejects_cyclic_container_values_before_provider_io(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
    shape: str,
) -> None:
    session, engine, fake_sg = sess
    schema_fixture['Shot']['description']['data_type']['value'] = 'serializable'
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 21, 'code': 'cycle'})
    cyclic: dict[str, Any]
    if shape == 'list':
        loop: list[Any] = []
        loop.append(loop)
        cyclic = {'root': loop}
    elif shape == 'mapping':
        cyclic = {}
        cyclic['self'] = cyclic
    else:
        loop = []
        cyclic = {'loop': loop}
        loop.append(cyclic)
    shot.description = cyclic
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='reference cycle'):
        session.commit()

    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('recovery', ['close', 'rollback'])
@pytest.mark.parametrize('category', ['new', 'deleted', 'revived'])
@pytest.mark.parametrize('ownership', ['detached', 'other_session'])
def test_corrupt_staging_recovery_preserves_unowned_object(
    sess: _SyncSessionBundle,
    recovery: str,
    category: str,
    ownership: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    obj = source.merge_dict({'type': 'Shot', 'id': 96, 'code': 'before'})
    if ownership == 'detached':
        source.close()
    obj.code = 'after'
    state = entity_state(obj)
    loaded_before = dict(state.loaded)
    dirty_before = dict(state.dirty)
    clean_before = dict(state.clean)
    dirty_types_before = dict(state.dirty_types)
    lazy_before = dict(state.lazy)
    session_before = state.session
    access_guard_before = state.access_guard
    identity_before = source._identity_values()
    getattr(session.unit_of_work, f'_{category}').append(obj)
    calls_before = len(fake_sg._calls)

    getattr(session, recovery)()

    assert session.unit_of_work.new == ()
    assert session.unit_of_work.deleted == ()
    assert session.unit_of_work.revived == ()
    assert state.loaded == loaded_before
    assert state.dirty == dirty_before
    assert state.clean == clean_before
    assert state.dirty_types == dirty_types_before
    assert state.lazy == lazy_before
    assert state.session is session_before
    assert state.access_guard is access_guard_before
    assert source._identity_values() == identity_before
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_detached_persistent_transition_exact_object_idempotence_and_inverse(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    obj = source.merge_dict({'type': 'Shot', 'id': 56, 'code': 'detached'})
    entity_state(obj).deleted = transition == 'revive'
    source.close()

    getattr(session, transition)(obj)
    getattr(session, transition)(obj)
    staged = (
        session.unit_of_work.deleted if transition == 'delete' else session.unit_of_work.revived
    )
    assert staged == (obj,)

    inverse = 'revive' if transition == 'delete' else 'delete'
    calls_before = len(fake_sg._calls)
    getattr(session, inverse)(obj)
    assert session.unit_of_work.deleted == ()
    assert session.unit_of_work.revived == ()
    assert entity_state(obj).session is None
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_detached_persistent_transition_joins_identity_lifecycle_after_success(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    obj = source.merge_dict({'type': 'Shot', 'id': 51, 'code': 'before'})
    entity_state(obj).deleted = transition == 'revive'
    source.close()

    getattr(session, transition)(obj)
    assert session._identity_values() == []
    session.commit()
    assert session._identity_values() == [obj]

    obj.code = 'after'
    if entity_state(obj).deleted:
        session.revive(obj)
    calls_before = len(fake_sg._calls)
    session.commit()
    later_requests = next(
        call[1][0] for call in fake_sg._calls[calls_before:] if call[0] == 'batch'
    )
    assert later_requests == [
        {
            'request_type': 'update',
            'entity_type': 'Shot',
            'entity_id': 51,
            'data': {'code': 'after'},
        }
    ]

    obj.code = 'rollback'
    session.rollback()
    assert obj.code == 'after'
    assert not entity_state(obj).dirty

    obj.code = 'close'
    session.close()
    assert obj.code == 'after'
    assert entity_state(obj).session is None
    other = type(session)(engine)
    if entity_state(obj).deleted:
        other.revive(obj)
    else:
        other.delete(obj)
    assert entity_state(obj).session is other


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_detached_persistent_transition_rejects_identity_collision_before_io(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    tracked = session.merge_dict({'type': 'Shot', 'id': 61, 'code': 'tracked'})
    source = type(session)(engine)
    duplicate = source.merge_dict({'type': 'Shot', 'id': 61, 'code': 'duplicate'})
    entity_state(duplicate).deleted = transition == 'revive'
    source.close()
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='already tracks a different'):
        getattr(session, transition)(duplicate)
    assert len(fake_sg._calls) == calls_before
    assert entity_state(duplicate).session is None
    assert session._identity_values() == [tracked]


@pytest.mark.parametrize(
    ('first_transition', 'second_transition'),
    [
        ('delete', 'delete'),
        ('delete', 'revive'),
        ('revive', 'delete'),
        ('revive', 'revive'),
    ],
)
def test_detached_persistent_transition_rejects_staged_copy_collision_before_io(
    sess: _SyncSessionBundle,
    first_transition: str,
    second_transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    objects: list[Entity] = []
    for transition, code in [(first_transition, 'first'), (second_transition, 'second')]:
        source = type(session)(engine)
        obj = source.merge_dict({'type': 'Shot', 'id': 66, 'code': code})
        entity_state(obj).deleted = transition == 'revive'
        source.close()
        objects.append(obj)
    first, second = objects

    getattr(session, first_transition)(first)
    deleted_before = session.unit_of_work.deleted
    revived_before = session.unit_of_work.revived
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match='already tracks a different'):
        getattr(session, second_transition)(second)
    assert session.unit_of_work.deleted == deleted_before
    assert session.unit_of_work.revived == revived_before
    assert entity_state(first).session is session
    assert entity_state(second).session is None
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_detached_transition_commits_valid_dirty_fields_in_lifecycle_order(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    obj = source.merge_dict({'type': 'Shot', 'id': 80, 'description': 'before'})
    source.close()
    obj.description = 'after'
    entity_state(obj).deleted = transition == 'revive'
    getattr(session, transition)(obj)
    calls_before = len(fake_sg._calls)

    session.commit()

    commit_calls = fake_sg._calls[calls_before:]
    if transition == 'delete':
        assert commit_calls == [
            (
                'batch',
                (
                    [
                        {
                            'request_type': 'update',
                            'entity_type': 'Shot',
                            'entity_id': 80,
                            'data': {'description': 'after'},
                        },
                        {
                            'request_type': 'delete',
                            'entity_type': 'Shot',
                            'entity_id': 80,
                        },
                    ],
                ),
                {},
            )
        ]
    else:
        assert commit_calls == [
            ('revive', ('Shot', 80), {}),
            (
                'batch',
                (
                    [
                        {
                            'request_type': 'update',
                            'entity_type': 'Shot',
                            'entity_id': 80,
                            'data': {'description': 'after'},
                        }
                    ],
                ),
                {},
            ),
        ]
    state = entity_state(obj)
    assert state.dirty == {}
    assert state.dirty_types == {}


@pytest.mark.parametrize(
    'case',
    [
        ('delete', 'foreign', 'different registry'),
        ('delete', 'pending', 'pending relationship entity is not staged'),
        ('revive', 'foreign', 'different registry'),
        ('revive', 'pending', 'pending relationship entity is not staged'),
    ],
)
def test_detached_transition_dirty_graph_rejected_before_io(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    case: tuple[str, str, str],
) -> None:
    session, engine, fake_sg = sess
    transition, relationship_kind, error = case
    engine.load_schema()
    source = type(session)(engine)
    obj = source.merge_dict({'type': 'Shot', 'id': 77, 'code': 'detached'})
    source.close()
    if relationship_kind == 'foreign':
        other_engine = create_engine(
            connection=fake_sg,
            cache_root=tmp_path / f'foreign-{transition}',
            registry=Registry(),
            bind_default=False,
        )
        other_engine.load_schema()
        relationship = type(session)(other_engine).merge_dict(
            {'type': 'Sequence', 'id': 78, 'code': 'foreign'}
        )
    else:
        relationship = engine.entities.Sequence.new(code='unstaged')
    obj.sg_sequence = relationship
    entity_state(obj).deleted = transition == 'revive'
    getattr(session, transition)(obj)
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match=error):
        session.commit()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize(
    'case',
    [
        ('delete', 'remove_field', r'no longer contains Shot\.description'),
        ('delete', 'make_ui_only', 'no longer API-writable'),
        ('delete', 'missing_provenance', 'classification is missing'),
        ('delete', 'type_drift', 'assigned type'),
        ('revive', 'remove_field', r'no longer contains Shot\.description'),
        ('revive', 'make_ui_only', 'no longer API-writable'),
        ('revive', 'missing_provenance', 'classification is missing'),
        ('revive', 'type_drift', 'assigned type'),
    ],
)
def test_detached_transition_schema_recovery_separates_local_corruption(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
    case: tuple[str, str, str],
) -> None:
    session, engine, fake_sg = sess
    transition, drift, error = case
    engine.load_schema()
    source = type(session)(engine)
    obj = source.merge_dict({'type': 'Shot', 'id': 79, 'description': 'before'})
    source.close()
    obj.description = 'after'
    entity_state(obj).deleted = transition == 'revive'
    getattr(session, transition)(obj)
    if drift == 'missing_provenance':
        entity_state(obj).dirty_types.pop('description')
    else:
        if drift == 'remove_field':
            schema_fixture['Shot'].pop('description')
        elif drift == 'make_ui_only':
            schema_fixture['Shot']['description']['properties']['query'] = {
                'value': {'filters': []}
            }
        else:
            schema_fixture['Shot']['description']['data_type']['value'] = 'image'
        fake_sg.schema_read = lambda: schema_fixture
        engine.load_schema(force=True)
    schema_reads_before = [name for name, _, _ in fake_sg._calls].count('schema_entity_read')
    writes_before = [
        call for call in fake_sg._calls if call[0] in {'batch', 'revive', 'upload_thumbnail'}
    ]

    with pytest.raises(SgalchemyError, match=error):
        session.commit()
    schema_reads = [name for name, _, _ in fake_sg._calls].count('schema_entity_read')
    assert schema_reads == schema_reads_before + (drift != 'missing_provenance')
    assert [
        call for call in fake_sg._calls if call[0] in {'batch', 'revive', 'upload_thumbnail'}
    ] == writes_before


@pytest.mark.parametrize('recovery', ['close', 'rollback'])
def test_indeterminate_entity_allows_internal_session_recovery(
    sess: _SyncSessionBundle,
    recovery: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    obj = session.merge_dict({'type': 'Shot', 'id': 95, 'code': 'before'})
    obj.code = 'after'
    session.delete(obj)
    state = entity_state(obj)
    state.indeterminate = True
    calls_before = len(fake_sg._calls)

    getattr(session, recovery)()

    assert session.unit_of_work.deleted == ()
    assert state.loaded['code'] == 'before'
    assert state.dirty == {}
    assert state.indeterminate
    assert state.session is (None if recovery == 'close' else session)
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('transition', ['add', 'delete', 'revive'])
def test_indeterminate_entity_rejects_every_staging_mode_before_attachment(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    if transition == 'add':
        obj = engine.entities.Shot.new(code='ambiguous')
    else:
        source = type(session)(engine)
        obj = source.merge_dict({'type': 'Shot', 'id': 94, 'code': 'ambiguous'})
        entity_state(obj).deleted = transition == 'revive'
        source.close()
    state = entity_state(obj)
    state.indeterminate = True
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='indeterminate server outcome'):
        getattr(session, transition)(obj)

    assert state.session is None
    assert session.unit_of_work.new == ()
    assert session.unit_of_work.deleted == ()
    assert session.unit_of_work.revived == ()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('container', ['mapping', 'tuple'])
def test_nested_container_foreign_entities_rejected_before_io(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    container: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    other = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'other-nested',
        registry=Registry(),
        bind_default=False,
    )
    other.load_schema()
    foreign = type(session)(other).merge_dict({'type': 'Shot', 'id': 43, 'code': 'foreign'})
    nested = {'value': foreign} if container == 'mapping' else (foreign,)
    local = engine.entities.Shot.new(code='local', description=nested)
    session.add(local)
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.commit()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize(
    ('container', 'persistent'),
    [
        ('scalar', True),
        ('scalar', False),
        ('list', True),
        ('list', False),
    ],
)
def test_nested_foreign_registry_entities_rejected_before_io(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    container: str,
    persistent: bool,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    other = create_engine(
        connection=fake_sg,
        cache_root=tmp_path / 'other',
        registry=Registry(),
        bind_default=False,
    )
    other.load_schema()
    if container == 'scalar':
        foreign: Entity = (
            type(session)(other).merge_dict({'type': 'Sequence', 'id': 41, 'code': 'foreign'})
            if persistent
            else other.entities.Sequence.new(code='foreign')
        )
        local = engine.entities.Shot.new(code='local', sg_sequence=foreign)
    else:
        foreign = (
            type(session)(other).merge_dict({'type': 'Shot', 'id': 42, 'code': 'foreign'})
            if persistent
            else other.entities.Shot.new(code='foreign')
        )
        local = engine.entities.Asset.new(code='local', shots=[foreign])
    session.add(local)
    calls_before = len(fake_sg._calls)
    with pytest.raises(SgalchemyError, match='different registry'):
        session.commit()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('with_clean_identity', [False, True])
def test_noop_commit_does_not_require_schema_or_io(
    sess: _SyncSessionBundle,
    with_clean_identity: bool,
) -> None:
    session, _engine, fake_sg = sess
    if with_clean_identity:
        session.merge_dict({'type': 'Shot', 'id': 33, 'code': 'clean'})
    plan = plan_commit(session, None, session._state.identity_map)
    assert plan._revives == []
    assert plan._waves == []
    assert plan._uploads == []

    session.commit()
    assert fake_sg._calls == []


@pytest.mark.parametrize(
    ('before', 'expected_loaded', 'expected_refresh', 'expected_unverified'),
    cast(
        'list[tuple[str, dict[str, str], set[str], set[str]]]',
        [
            ('loaded', {'description': 'old'}, set(), set()),
            ('refresh', {}, {'description'}, set()),
            ('unfetched', {}, set(), set()),
            ('unverified', {'description': 'old'}, set(), {'description'}),
        ],
    ),
)
def test_rollback_restores_field_provenance(
    sess: _SyncSessionBundle,
    before: str,
    expected_loaded: dict[str, str],
    expected_refresh: set[str],
    expected_unverified: set[str],
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict({'type': 'Shot', 'id': 40})
    state = entity_state(shot)
    if before in {'loaded', 'unverified'}:
        state.loaded['description'] = 'old'
    if before == 'refresh':
        state.refresh_required.add('description')
    if before == 'unverified':
        state.unverified_writes.add('description')

    shot.description = 'new'
    assert not state.refresh_required
    assert not state.unverified_writes
    session.rollback()

    assert {key: value for key, value in state.loaded.items() if key != 'id'} == expected_loaded
    assert state.refresh_required == expected_refresh
    assert state.unverified_writes == expected_unverified


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_schema_recovery_forces_once_for_removed_entity_before_io(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    obj = session.merge_dict({'type': 'Shot', 'id': 68, 'code': 'stale'})
    entity_state(obj).deleted = transition == 'revive'
    getattr(session, transition)(obj)
    schema_fixture.pop('Shot')
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema(force=True)
    schema_reads_before = [name for name, _, _ in fake_sg._calls].count('schema_entity_read')

    with pytest.raises(SgalchemyError, match='no longer contains entity') as excinfo:
        session.commit()
    message = str(excinfo.value)
    assert 'refreshing schema once' in message
    assert 'session.rollback()' in message
    assert [name for name, _, _ in fake_sg._calls].count('schema_entity_read') == (
        schema_reads_before + 1
    )
    assert not [call for call in fake_sg._calls if call[0] == 'batch']


@pytest.mark.parametrize(
    'case',
    [
        ('remove_entity', 'image', '/renders/image.jpg', 'no longer contains entity'),
        ('remove_field', 'image', '/renders/image.jpg', r'no longer contains Shot\.image'),
        ('make_ui_only', 'description', 'stale write', 'no longer API-writable'),
        ('make_noneditable', 'description', 'stale write', 'noneditable'),
        ('image_to_text', 'image', '/renders/image.jpg', 'assigned type'),
        ('text_to_image', 'description', '/renders/image.jpg', 'assigned type'),
    ],
)
def test_schema_recovery_forces_once_for_staged_write_drift_before_io(
    sess: _SyncSessionBundle,
    schema_fixture: dict[str, Any],
    case: tuple[str, str, str, str],
) -> None:
    session, engine, fake_sg = sess
    change, field_code, value, error = case
    engine.load_schema()
    pending = engine.entities.Shot.new(**{field_code: value})
    session.add(pending)
    if change == 'remove_entity':
        schema_fixture.pop('Shot')
    elif change == 'remove_field':
        schema_fixture['Shot'].pop(field_code)
    elif change == 'make_ui_only':
        schema_fixture['Shot'][field_code]['properties']['query'] = {'value': {'filters': []}}
    elif change == 'make_noneditable':
        schema_fixture['Shot'][field_code]['editable'] = {'value': False}
    else:
        data_type = 'text' if change == 'image_to_text' else 'image'
        schema_fixture['Shot'][field_code]['data_type']['value'] = data_type
    fake_sg.schema_read = lambda: schema_fixture
    engine.load_schema(force=True)
    schema_reads_before = [name for name, _, _ in fake_sg._calls].count('schema_entity_read')

    with pytest.raises(SgalchemyError, match=error) as excinfo:
        session.commit()
    message = str(excinfo.value)
    assert 'refreshing schema once' in message
    assert 'session.rollback()' in message
    assert [name for name, _, _ in fake_sg._calls].count('schema_entity_read') == (
        schema_reads_before + 1
    )
    assert not [call for call in fake_sg._calls if call[0] == 'batch']


@pytest.mark.parametrize('transition', ['delete', 'revive'])
def test_staged_persistent_transition_rechecks_identity_collision_before_io(
    sess: _SyncSessionBundle,
    transition: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    source = type(session)(engine)
    staged = source.merge_dict({'type': 'Shot', 'id': 71, 'code': 'staged'})
    entity_state(staged).deleted = transition == 'revive'
    source.close()
    getattr(session, transition)(staged)
    source = type(session)(engine)
    tracked = source.merge_dict({'type': 'Shot', 'id': 71, 'code': 'tracked'})
    source.close()
    session._state.identity_map[('Shot', 71)] = tracked
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='already tracks a different'):
        session.commit()
    assert len(fake_sg._calls) == calls_before
    assert session._identity_values() == [tracked]
    assert entity_state(staged).session is session


@pytest.mark.parametrize(
    ('first_transition', 'corrupted_list'),
    [
        ('delete', 'deleted'),
        ('delete', 'revived'),
        ('revive', 'revived'),
    ],
)
def test_staged_persistent_transition_rechecks_staging_collisions_before_io(
    sess: _SyncSessionBundle,
    first_transition: str,
    corrupted_list: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    objects: list[Entity] = []
    for transition, code in [
        (first_transition, 'first'),
        ('revive' if corrupted_list == 'revived' else 'delete', 'second'),
    ]:
        source = type(session)(engine)
        obj = source.merge_dict({'type': 'Shot', 'id': 76, 'code': code})
        entity_state(obj).deleted = transition == 'revive'
        source.close()
        objects.append(obj)
    first, second = objects
    getattr(session, first_transition)(first)
    getattr(session.unit_of_work, f'_{corrupted_list}').append(second)
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match='already tracks a different'):
        session.commit()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize(
    ('corruption', 'error'),
    [
        ('foreign_new', 'different registry'),
        ('other_session_deleted', 'another open session'),
        ('retired_new', 'retired and has staged writes'),
    ],
)
def test_staging_preflight_retains_registry_ownership_and_retired_write_checks(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    corruption: str,
    error: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    if corruption == 'foreign_new':
        foreign_engine = create_engine(
            connection=fake_sg,
            cache_root=tmp_path / 'foreign',
            registry=Registry(),
            bind_default=False,
        )
        foreign_engine.load_schema()
        obj = foreign_engine.entities.Shot.new(code='foreign')
        session.unit_of_work._new.append(obj)
    elif corruption == 'other_session_deleted':
        other = type(session)(engine)
        obj = other.merge_dict({'type': 'Shot', 'id': 92, 'code': 'owned'})
        session.unit_of_work._deleted.append(obj)
    else:
        obj = engine.entities.Shot.new(code='retired-pending')
        session.add(obj)
        entity_state(obj).deleted = True
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match=error):
        session.commit()
    assert len(fake_sg._calls) == calls_before


@pytest.mark.parametrize('discard', ['rollback', 'close'])
def test_thumbnail_clear_restores_only_ordinary_snapshot(
    sess: _SyncSessionBundle,
    discard: str,
) -> None:
    session, engine, _fake_sg = sess
    engine.load_schema()
    shot = session.merge_dict(
        {
            'type': 'Shot',
            'id': 44,
            'image': 'old-image',
            'filmstrip_image': 'old-filmstrip',
        }
    )

    shot.image = None
    getattr(session, discard)()

    state = entity_state(shot)
    assert state.loaded['image'] == 'old-image'
    assert state.loaded['filmstrip_image'] == 'old-filmstrip'
    assert state.dirty == {}


@pytest.mark.parametrize(
    ('field_code', 'method_name'),
    [
        ('image', 'upload_thumbnail'),
        ('filmstrip_image', 'upload_filmstrip_thumbnail'),
    ],
)
def test_thumbnail_upload_steps_select_provider_method(
    field_code: str,
    method_name: str,
) -> None:
    shot = Entity.__new__(Entity)
    initialize_entity(shot, EntityState(entity_type='Shot', id=47))
    path = Path('/renders/thumbnail.jpg')
    upload = _uow.UploadPlanItem(shot, field_code, path)
    plan = _uow.CommitPlan(_revives=[], _waves=[], _uploads=[upload], binding=None)

    steps = _uow.commit_steps(plan, None)

    assert len(steps) == 1
    assert steps[0].kind == method_name
    assert steps[0]._upload is upload
    assert steps[0].items == (upload,)


@pytest.mark.parametrize('bad_path_kind', ['missing', 'directory', 'empty', 'unreadable'])
def test_upload_path_preflight_precedes_revive_and_preserves_staging(
    sess: _SyncSessionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_path_kind: str,
) -> None:
    session, engine, fake_sg = sess
    engine.load_schema()
    revived = session.merge_dict({'type': 'Shot', 'id': 45, 'code': 'revive'})
    entity_state(revived).deleted = True
    session.revive(revived)
    upload = session.merge_dict({'type': 'Shot', 'id': 46, 'code': 'upload'})
    if bad_path_kind == 'directory':
        path: object = tmp_path
    elif bad_path_kind == 'empty':
        path = ''
    else:
        path = tmp_path / f'{bad_path_kind}.jpg'
        if bad_path_kind == 'unreadable':
            path.write_bytes(b'image')
            real_open = _uow.os.open
            target_path = path

            @_preserve_signature(real_open)
            def fail_target_open(
                path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                flags: int,
                mode: int = 0o777,
                *,
                dir_fd: int | None = None,
            ) -> int:
                if path == str(target_path):
                    raise OSError('unreadable')
                return real_open(path, flags, mode, dir_fd=dir_fd)

            monkeypatch.setattr(_uow.os, 'open', fail_target_open)
    upload.image = path
    calls_before = len(fake_sg._calls)

    with pytest.raises(SgalchemyError, match=r'Shot\.image.*readable regular file'):
        session.commit()

    assert len(fake_sg._calls) == calls_before
    assert session.unit_of_work.revived == (revived,)
    assert entity_state(upload).dirty == {'image': path}


def test_wire_value_emits_canonical_links_and_preserves_containers() -> None:
    linked = Entity.__new__(Entity)
    initialize_entity(
        linked,
        EntityState(entity_type='Sequence', id=7, loaded={'code': 'SQ01', 'name': 'Sequence'}),
    )
    value = {'mapping': {'link': linked}, 'list': [linked], 'tuple': (linked,)}
    canonical = {'type': 'Sequence', 'id': 7}

    assert _uow._wire_value(value) == {
        'mapping': {'link': canonical},
        'list': [canonical],
        'tuple': (canonical,),
    }


def test_wire_value_rejects_cycles_without_recursion_error() -> None:
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    with pytest.raises(SgalchemyError) as exc_info:
        _uow._wire_value(cyclic)

    assert str(exc_info.value) == (
        'container value contains a reference cycle; cyclic containers cannot be '
        'represented as wire data, so break the cycle before staging or serializing'
    )
