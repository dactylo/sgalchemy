import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._expressions import FieldExpr
from sgalchemy._core._registry import Entity, Registry
from sgalchemy._core._statement import select
from sgalchemy._orm._session import Session

_LIVE_KEYS = (
    'SGALCHEMY_LIVE_API_KEY',
    'SGALCHEMY_LIVE_CREATE_DATA',
    'SGALCHEMY_LIVE_ENTITY_TYPE',
    'SGALCHEMY_LIVE_NAME_FIELD',
    'SGALCHEMY_LIVE_SCRIPT_NAME',
    'SGALCHEMY_LIVE_SITE',
)


class _LiveEngine:
    def __init__(self) -> None:
        self.registry = _LiveRegistry()

    def load_schema(self, *, force: bool) -> None:
        _ = force


class _LiveRecord:
    def __init__(self, entity_id: int | None) -> None:
        self.id = entity_id


class _LiveEntity:
    code = object()

    @classmethod
    def new(cls, **_data: object) -> _LiveRecord:
        return _LiveRecord(None)


class _LiveRegistry:
    def entity_class(self, _entity_type: str) -> type[_LiveEntity]:
        return _LiveEntity


class _LiveStatement:
    def where(self, _condition: object) -> '_LiveStatement':
        return self


class _LiveSession:
    def __init__(
        self,
        *,
        assigned_id_before_error: int | None = None,
        commit_error: Exception | None = None,
        rows: list[_LiveRecord] | None = None,
    ) -> None:
        self._added: _LiveRecord | None = None
        self._assigned_id_before_error = assigned_id_before_error
        self._closed = False
        self._commit_error = commit_error
        self._committed = False
        self._deleted: _LiveRecord | None = None
        self.engine = object()
        self._rows = rows or []

    def add(self, obj: _LiveRecord) -> None:
        self._added = obj

    def close(self) -> None:
        self._closed = True

    def commit(self) -> None:
        if self._commit_error is not None:
            if self._added is not None and self._assigned_id_before_error is not None:
                self._added.id = self._assigned_id_before_error
            raise self._commit_error
        self._committed = True

    def delete(self, obj: _LiveRecord) -> None:
        self._deleted = obj

    def scalars(self, _statement: _LiveStatement) -> list[_LiveRecord]:
        return self._rows


def _cleanup_live_entity(
    engine: Engine,
    entity_cls: type[Entity],
    field: FieldExpr,
    unique_names: tuple[str, str],
    persisted: Entity | None,
) -> int | None:
    cleanup_obj = persisted
    if cleanup_obj is None:
        lookup = Session(engine)
        try:
            for unique_name in unique_names:
                matches = lookup.scalars(select(entity_cls).where(field == unique_name))
                if matches:
                    cleanup_obj = matches[0]
                    break
        finally:
            lookup.close()
    if cleanup_obj is None or cleanup_obj.id is None:
        return None
    created_id = cleanup_obj.id
    cleanup = Session(engine)
    try:
        cleanup.delete(cleanup_obj)
        cleanup.commit()
    finally:
        cleanup.close()
    return created_id


def _require_live_credentials() -> tuple[str, str, str]:
    missing = [key for key in _LIVE_KEYS if not os.environ.get(key)]
    required = os.environ.get('SGALCHEMY_REQUIRE_LIVE') == '1'
    if missing:
        message = f'missing live configuration: {missing}'
        if required:
            pytest.fail(message)
        pytest.skip(message)
    return (
        os.environ['SGALCHEMY_LIVE_SITE'],
        os.environ['SGALCHEMY_LIVE_SCRIPT_NAME'],
        os.environ['SGALCHEMY_LIVE_API_KEY'],
    )


def _run_live_contract(engine: Engine) -> None:
    required = os.environ.get('SGALCHEMY_REQUIRE_LIVE') == '1'
    try:
        engine.load_schema(force=True)
    except ImportError as err:
        if required:
            pytest.fail(f'required live provider is unavailable: {err}')
        pytest.skip(f'live provider is unavailable: {err}')
    entity_type = os.environ['SGALCHEMY_LIVE_ENTITY_TYPE']
    name_field = os.environ['SGALCHEMY_LIVE_NAME_FIELD']
    create_data = cast('dict[str, Any]', json.loads(os.environ['SGALCHEMY_LIVE_CREATE_DATA']))
    entity_cls = engine.registry.entity_class(entity_type)
    session = Session(engine)
    first_name = f'sgalchemy-contract-{uuid4().hex}'
    second_name = f'{first_name}-updated'
    create_data[name_field] = first_name
    obj = entity_cls.new(**create_data)
    session.add(obj)
    field = getattr(entity_cls, name_field)
    created_id: int | None
    fresh: Session | None = None
    persisted: Entity | None = None
    try:
        session.commit()
        found = session.scalars(select(entity_cls).where(field == first_name))[0]
        setattr(found, name_field, second_name)
        session.commit()
        session.close()
        fresh = Session(engine)
        persisted = fresh.scalars(select(entity_cls).where(field == second_name))[0]
        assert getattr(persisted, name_field) == second_name
    finally:
        session.close()
        if fresh is not None:
            fresh.close()
        created_id = _cleanup_live_entity(
            engine,
            entity_cls,
            field,
            (second_name, first_name),
            persisted,
        )
    assert created_id is not None
    postcheck = Session(engine)
    try:
        remaining = postcheck.scalars(select(entity_cls).where(entity_cls.id == created_id))
    finally:
        postcheck.close()
    assert remaining == []


@pytest.fixture
def live_case(tmp_path: Path) -> Engine:
    site, script_name, api_key = _require_live_credentials()
    return create_engine(
        site,
        script_name=script_name,
        api_key=api_key,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )


@pytest.mark.parametrize(
    'assigned_id_before_error',
    [
        pytest.param(None, id='before-remote-create'),
        pytest.param(41, id='assigned-id'),
    ],
)
def test_live_create_failure_attempts_fresh_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    assigned_id_before_error: int | None,
) -> None:
    original = _LiveSession(
        assigned_id_before_error=assigned_id_before_error,
        commit_error=RuntimeError('create reconciliation failed'),
    )
    discovered = _LiveRecord(42)
    lookup = _LiveSession(rows=[discovered])
    cleanup = _LiveSession()
    sessions = iter((original, lookup, cleanup))

    def create_session(_engine: object) -> _LiveSession:
        return next(sessions)

    def create_statement(_entity_cls: type[_LiveEntity]) -> _LiveStatement:
        return _LiveStatement()

    monkeypatch.setattr(sys.modules[__name__], 'Session', create_session)
    monkeypatch.setattr(sys.modules[__name__], 'select', create_statement)
    monkeypatch.setenv('SGALCHEMY_LIVE_CREATE_DATA', '{}')
    monkeypatch.setenv('SGALCHEMY_LIVE_ENTITY_TYPE', 'TestEntity')
    monkeypatch.setenv('SGALCHEMY_LIVE_NAME_FIELD', 'code')
    engine = cast('Engine', _LiveEngine())

    with pytest.raises(RuntimeError, match='create reconciliation failed'):
        _run_live_contract(engine)

    assert original._closed
    assert original._added is not None
    assert original._added.id == assigned_id_before_error
    assert lookup._closed
    assert cleanup._deleted is discovered
    assert cleanup._deleted is not original._added
    assert cleanup._committed
    assert cleanup._closed


@pytest.mark.live
def test_live_create_query_update_cleanup(
    live_case: Engine,
) -> None:
    try:
        _run_live_contract(live_case)
    finally:
        live_case.dispose()


@pytest.mark.live
def test_live_session_token_engine_runs_the_contract(tmp_path: Path) -> None:
    site, script_name, api_key = _require_live_credentials()
    shotgun_api3 = importlib.import_module('shotgun_api3')

    issuer = shotgun_api3.Shotgun(site, script_name=script_name, api_key=api_key)
    try:
        token = issuer.get_session_token()
    finally:
        close = getattr(issuer, 'close', None)
        if callable(close):
            close()
    engine = create_engine(
        site,
        session_token=token,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )
    try:
        _run_live_contract(engine)
    finally:
        engine.dispose()
