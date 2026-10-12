from typing import Any, Never

import pytest

from sgalchemy._orm import _scoping
from sgalchemy._orm._scope_common import SuppressOrdinaryFailure, record_close_failure


class _ControlSignal(BaseException):
    pass


def _raise_close_failure() -> Never:
    raise RuntimeError('secondary')


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
    class InterruptingNoteAccess(BaseException):
        def __getattribute__(self, name: str) -> Any:  # noqa: ANN401 - Delegate BaseException attributes while interrupting note access.
            if name == '__notes__':
                raise KeyboardInterrupt
            return super().__getattribute__(name)

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
