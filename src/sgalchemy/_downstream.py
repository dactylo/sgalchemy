"""Write standard output and absorb closure by its downstream reader."""

import errno
import os
import sys
from typing import TextIO

_WINDOWS = sys.platform == 'win32'


def _is_downstream_closure(error: OSError) -> bool:
    """Report whether the error means the downstream reader closed the stream."""
    return isinstance(error, BrokenPipeError) or (_WINDOWS and error.errno == errno.EINVAL)


def _silence_descriptor(stream: TextIO) -> None:
    """Point a stream's descriptor at the null device when it has one."""
    try:
        target = stream.fileno()
    except (OSError, ValueError):
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, target)
    finally:
        os.close(devnull)


def write_stdout(text: str) -> None:
    """Write and flush standard output, silencing a closed reader.

    Args:
        text: Complete output of a finished command.

    Raises:
        OSError: If writing or flushing fails for a reason other than closure.
    """
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except OSError as error:
        if not _is_downstream_closure(error):
            raise
        _silence_descriptor(sys.stdout)
