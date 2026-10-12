"""Shared failure diagnostics for synchronous and asynchronous scoping."""

from logging import Logger
from types import TracebackType

from sgalchemy._core._engine_common import add_note_once


class SuppressOrdinaryFailure:
    """Record context failures while suppressing only ordinary exceptions.

    After exit, `error` contains the body failure, including an unsuppressed
    `BaseException`, or `None` when the body completed successfully.
    """

    def __init__(self) -> None:
        """Initialize without an observed failure."""
        self.error: BaseException | None = None

    def __enter__(self) -> None:
        """Enter the suppression boundary."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Record the body failure and suppress only ordinary exceptions."""
        del exc_type, traceback
        self.error = exc_value
        return isinstance(exc_value, Exception)


def record_close_failure(
    primary: BaseException,
    secondary: BaseException,
    owner_name: str,
    logger: Logger,
) -> None:
    """Annotate a primary fault or log the secondary close failure.

    Args:
        primary: Failure that must remain primary.
        secondary: Failure raised while closing the scoped owner.
        owner_name: Session type whose close operation failed.
        logger: Logger for a secondary failure that cannot be annotated.
    """
    note = f'sgalchemy: {owner_name}.close() also failed: {type(secondary).__name__}: {secondary}'
    if not add_note_once(primary, note):
        logger.error(
            '%s.close() failed while preserving the primary exception',
            owner_name,
            exc_info=(type(secondary), secondary, secondary.__traceback__),
        )
