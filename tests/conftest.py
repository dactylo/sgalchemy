import copy
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from _fakes import (
    _SCHEMA_FIXTURE,
    _FakeAsyncShotgun,
    _FakeShotgun,
    _FakeShotgunFactory,
    _SchemaFixture,
)
from _support import PROVIDER_DISTRIBUTIONS, _SyncSessionBundle, fetch_schema

from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._core._registry import Registry
from sgalchemy._orm._session import Session
from sgalchemy.schema import _model

_REQUIRED_NODEIDS: set[str] = set()
_UNEXPECTED_REQUIRED_SKIPS: set[str] = set()


def _installed_provider_modes() -> set[str]:
    installed: set[str] = set()
    for mode, distribution in PROVIDER_DISTRIBUTIONS.items():
        try:
            version(distribution)
        except PackageNotFoundError:
            continue
        installed.add(mode)
    return installed


def _validate_required_provider_environment(provider: str | None, native: bool) -> None:
    if native and provider not in (None, 'api'):
        raise pytest.UsageError('native async mode is compatible only with provider mode api')
    expected: set[str] = {'api'} if native else ({provider} if provider is not None else set())
    if expected and _installed_provider_modes() != expected:
        names = sorted(PROVIDER_DISTRIBUTIONS[mode] for mode in expected)
        raise pytest.UsageError(
            f'required provider environment must contain exactly {names}, with no other '
            'provider distribution installed'
        )


def _required_markers() -> set[str]:
    markers: set[str] = set()
    if os.environ.get('SGALCHEMY_REQUIRE_LIVE') == '1':
        markers.add('live')
    native = os.environ.get('SGALCHEMY_REQUIRE_NATIVE_ASYNC') == '1'
    if native:
        markers.add('native_async')
    provider = os.environ.get('SGALCHEMY_REQUIRE_PROVIDER')
    if provider is not None:
        if provider not in PROVIDER_DISTRIBUTIONS:
            raise pytest.UsageError(f'unknown SGALCHEMY_REQUIRE_PROVIDER value: {provider!r}')
        markers.add('provider_contract')
    _validate_required_provider_environment(provider, native)
    return markers


@pytest.fixture
def engine(fake_sg: _FakeShotgun, tmp_path: Path) -> Engine:
    return create_engine(
        connection=fake_sg,
        cache_root=tmp_path,
        registry=Registry(),
        bind_default=False,
    )


@pytest.fixture
def fake_async_sg() -> _FakeAsyncShotgun:
    return _FakeAsyncShotgun()


@pytest.fixture
def fake_sg() -> _FakeShotgun:
    return _FakeShotgun()


@pytest.fixture
def fake_sg_factory() -> _FakeShotgunFactory:
    return _FakeShotgunFactory()


def pytest_collection_finish(session: pytest.Session) -> None:
    for marker in sorted(_required_markers()):
        nodeids = {item.nodeid for item in session.items if item.get_closest_marker(marker)}
        if not nodeids:
            raise pytest.UsageError(f'required test mode collected zero {marker!r} tests')
        _REQUIRED_NODEIDS.update(nodeids)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and report.nodeid in _REQUIRED_NODEIDS:
        _UNEXPECTED_REQUIRED_SKIPS.add(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _UNEXPECTED_REQUIRED_SKIPS and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def schemas(fake_sg: _FakeShotgun) -> dict[str, _model.EntitySchema]:
    return _model.build_schema(fetch_schema(fake_sg))


@pytest.fixture
def sess(fake_sg: _FakeShotgun, tmp_path: Path) -> _SyncSessionBundle:
    engine = create_engine(
        connection=fake_sg, cache_root=tmp_path, registry=Registry(), bind_default=False
    )
    return Session(engine), engine, fake_sg


@pytest.fixture
def schema_fixture() -> _SchemaFixture:
    return copy.deepcopy(_SCHEMA_FIXTURE)
