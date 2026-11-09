import asyncio
from typing import Never

import pytest

from sgalchemy._orm import _scoping
from sgalchemy._orm._scope_common import SuppressOrdinaryFailure, record_close_failure
from tests._support import InterruptingNoteAccess


class _ControlSignal(BaseException):
    pass


def _raise_close_failure() -> Never:
    raise RuntimeError('secondary')


@pytest.mark.parametrize('hook', ['__name__', '__str__', '__traceback__'])
@pytest.mark.parametrize('interrupted', [False, True])
def test_shared_close_failure_governs_foreign_diagnostic_hooks(
    caplog: pytest.LogCaptureFixture,
    hook: str,
    interrupted: bool,
) -> None:
    fault = _ControlSignal('interrupted') if interrupted else ValueError('diagnostic failed')

    class FailureType(type):
        def __getattribute__(cls, name: str) -> object:
            if name == hook:
                raise fault
            return super().__getattribute__(name)

    class CloseError(Exception, metaclass=FailureType):
        def __getattribute__(self, name: str) -> object:
            if name == hook:
                raise fault
            return super().__getattribute__(name)

        def __str__(self) -> str:
            if hook == '__str__':
                raise fault
            return super().__str__()

    class UnannotatableError(Exception):
        def add_note(self, note: str) -> None:
            raise ValueError(note)

    primary = UnannotatableError('primary')
    secondary = CloseError('secondary')
    with caplog.at_level('ERROR', logger='sgalchemy.session'):
        if interrupted:
            with pytest.raises(_ControlSignal) as excinfo:
                record_close_failure(primary, secondary, 'Session', _scoping._LOGGER)
            assert excinfo.value is fault
            assert not caplog.records
        else:
            record_close_failure(primary, secondary, 'Session', _scoping._LOGGER)
            record = caplog.records[-1]
            assert record.getMessage() == (
                'Session.close() failed while preserving the primary exception'
            )
            assert record.exc_info is None


def test_shared_close_failure_logs_secondary_when_primary_cannot_be_annotated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class UnannotatableFailure(BaseException):
        def add_note(self, note: str) -> None:
            raise RuntimeError(f'cannot attach {note}')

    primary = UnannotatableFailure('primary')
    try:
        _raise_close_failure()
    except RuntimeError as secondary:
        expected_exc_info = (type(secondary), secondary, secondary.__traceback__)
        with caplog.at_level('ERROR', logger='sgalchemy.session'):
            record_close_failure(primary, secondary, 'Session', _scoping._LOGGER)

    record = caplog.records[-1]
    assert record.getMessage() == ('Session.close() failed while preserving the primary exception')
    assert record.exc_info == expected_exc_info


def test_shared_close_failure_propagates_note_access_interruption() -> None:
    with pytest.raises(KeyboardInterrupt):
        record_close_failure(
            InterruptingNoteAccess('primary'),
            RuntimeError('secondary'),
            'Session',
            _scoping._LOGGER,
        )


def test_shared_suppress_failure_records_and_suppresses_ordinary_failure() -> None:
    fault = RuntimeError('ordinary failure')
    suppression = SuppressOrdinaryFailure()

    with suppression:
        raise fault

    assert suppression.error is fault


def test_shared_suppress_failure_records_without_suppressing_base_exception() -> None:
    fault = _ControlSignal('control signal')
    suppression = SuppressOrdinaryFailure()

    with pytest.raises(_ControlSignal) as excinfo, suppression:
        raise fault

    assert excinfo.value is fault
    assert suppression.error is fault


def test_suppress_failure_preserves_timeout_cancellation() -> None:
    async def scenario() -> None:
        async def time_out_suppression() -> None:
            async with asyncio.timeout(0):
                with SuppressOrdinaryFailure():
                    await asyncio.sleep(0)

        with pytest.raises(TimeoutError):
            await time_out_suppression()

    asyncio.run(scenario())
