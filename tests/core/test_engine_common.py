from typing import Any

import pytest
from _support import AUTHENTICATION_FAULT, BrokenNoteError

from sgalchemy._core._engine_common import add_auth_note, add_note_once


class _DiagnosticSignal(BaseException):
    pass


class _InterruptingNoteAccess(BaseException):
    def __getattribute__(self, name: str) -> Any:  # noqa: ANN401 - Delegate BaseException attributes while interrupting note access.
        if name == '__notes__':
            raise KeyboardInterrupt
        return super().__getattribute__(name)


def test_add_auth_note_annotates_only_authentication_faults() -> None:
    note = 'sgalchemy: recover credentials'
    target = RuntimeError('target')

    add_auth_note(RuntimeError('ordinary'), target, note)
    assert getattr(target, '__notes__', []) == []

    add_auth_note(AUTHENTICATION_FAULT('expired'), target, note)
    assert target.__notes__ == [note]


def test_add_note_once_propagates_note_access_interruption() -> None:
    with pytest.raises(KeyboardInterrupt):
        add_note_once(_InterruptingNoteAccess('interrupted'), 'diagnostic context')


def test_add_note_once_returns_false_for_ordinary_annotation_failure() -> None:
    assert not add_note_once(BrokenNoteError('provider unavailable'), 'diagnostic context')


@pytest.mark.parametrize('error_type', [RuntimeError, _DiagnosticSignal])
def test_add_note_once_supports_all_failures_without_duplicates(
    error_type: type[BaseException],
) -> None:
    error = error_type('fault')
    note = 'sgalchemy: diagnostic context'

    assert add_note_once(error, note)
    assert add_note_once(error, note)
    assert error.__notes__ == [note]
