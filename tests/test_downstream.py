import errno
import io
import os
import sys

import pytest

from sgalchemy import _downstream
from sgalchemy._downstream import _is_downstream_closure, _silence_descriptor, write_stdout


class _FakeOs:
    O_WRONLY = os.O_WRONLY
    devnull = os.devnull

    def __init__(self, *, redirect_error: OSError | None) -> None:
        self.events: list[tuple[str, int]] = []
        self.redirect_error = redirect_error

    def close(self, fd: int) -> None:
        self.events.append(('close', fd))

    def dup2(self, source: int, _target: int) -> None:
        self.events.append(('dup2', source))
        if self.redirect_error is not None:
            raise self.redirect_error

    def open(self, _path: str, _flags: int) -> int:
        return 37


class _DescriptorStream(io.StringIO):
    def fileno(self) -> int:
        return 1


class _RecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class _FailingStream(io.StringIO):
    def __init__(self, error: OSError, *, fail_on: str) -> None:
        super().__init__()
        self.error = error
        self.fail_on = fail_on

    def flush(self) -> None:
        if self.fail_on == 'flush':
            raise self.error
        super().flush()

    def write(self, text: str) -> int:
        if self.fail_on == 'write':
            raise self.error
        return super().write(text)


def test_broken_pipe_is_closure_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    for windows in (False, True):
        monkeypatch.setattr(_downstream, '_WINDOWS', windows)
        assert _is_downstream_closure(BrokenPipeError())


def test_einval_is_closure_only_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    error = OSError(errno.EINVAL, 'Invalid argument')
    monkeypatch.setattr(_downstream, '_WINDOWS', True)
    assert _is_downstream_closure(error)
    monkeypatch.setattr(_downstream, '_WINDOWS', False)
    assert not _is_downstream_closure(error)


def test_silence_descriptor_closes_devnull_and_propagates_redirect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeOs(redirect_error=OSError('injected redirect failure'))
    monkeypatch.setattr(_downstream, 'os', fake)

    with pytest.raises(OSError, match='injected redirect failure'):
        _silence_descriptor(_DescriptorStream())
    assert fake.events == [('dup2', 37), ('close', 37)]


def test_silence_descriptor_ignores_streams_without_descriptors() -> None:
    assert _silence_descriptor(io.StringIO()) is None


def test_silence_descriptor_redirects_a_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOs(redirect_error=None)
    monkeypatch.setattr(_downstream, 'os', fake)

    assert _silence_descriptor(_DescriptorStream()) is None
    assert fake.events == [('dup2', 37), ('close', 37)]


def test_unrelated_errno_is_not_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_downstream, '_WINDOWS', True)
    assert not _is_downstream_closure(OSError(errno.EBADF, 'bad'))


@pytest.mark.parametrize('fail_on', ['write', 'flush'])
def test_write_stdout_silences_stdout_after_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
    fail_on: str,
) -> None:
    stream = _FailingStream(BrokenPipeError(), fail_on=fail_on)
    silenced: list[io.StringIO] = []
    monkeypatch.setattr(_downstream, '_silence_descriptor', silenced.append)
    monkeypatch.setattr(sys, 'stdout', stream)

    write_stdout('text')

    assert silenced == [stream]


@pytest.mark.parametrize('fail_on', ['write', 'flush'])
def test_write_stdout_propagates_other_errors(
    monkeypatch: pytest.MonkeyPatch,
    fail_on: str,
) -> None:
    error = OSError(errno.EIO, 'I/O error')
    stream = _FailingStream(error, fail_on=fail_on)
    silenced: list[io.StringIO] = []
    monkeypatch.setattr(_downstream, '_silence_descriptor', silenced.append)
    monkeypatch.setattr(sys, 'stdout', stream)

    with pytest.raises(OSError, match='I/O error') as raised:
        write_stdout('text')

    assert raised.value is error
    assert silenced == []


def test_write_stdout_writes_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _RecordingStream()
    monkeypatch.setattr(sys, 'stdout', stream)

    write_stdout('hello')

    assert stream.getvalue() == 'hello'
    assert stream.flushes == 1
