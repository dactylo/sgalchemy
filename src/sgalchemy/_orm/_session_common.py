"""Share pure session result, refresh-call, and commit-failure handling."""

from typing import cast

from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._engine_common import (
    AUTH_READ_NOTE,
    AUTH_RECOVERY_NOTE,
    add_auth_note,
    add_note_once,
    looks_like_auth_fault,
)
from sgalchemy._core._registry import EntityState
from sgalchemy._core._result import Result, Row
from sgalchemy._errors import CommitError, SgalchemyError
from sgalchemy._orm._uow import (
    SOLE_BATCH_STAGED_NOTE,
    CommitOutcome,
    CommitStep,
    UnitOfWork,
    accepted_batch_reported,
    batch_failure_note,
    build_commit_error,
)
from sgalchemy._shotgun import QueryResult, WireRecords
from sgalchemy.schema._model import EntitySchema


def annotate_read_error(error: Exception, context: str) -> None:
    """Attach operation context and authentication recovery guidance to a fault."""
    add_note_once(error, f'sgalchemy: {context}')
    if looks_like_auth_fault(error):
        add_note_once(error, AUTH_READ_NOTE)


def commit_error_with_batch_status(
    err: Exception,
    step: CommitStep | None,
    outcome: CommitOutcome,
    unit_of_work: UnitOfWork,
    *,
    invoked: bool,
) -> CommitError:
    """Wrap an ordinary commit failure with batch-status and auth recovery notes.

    Args:
        err: Ordinary failure raised during commit execution.
        step: Step active when `err` was raised, or `None` between steps.
        outcome: Failure-time commit partitions.
        unit_of_work: Session staging state holding any accepted batch.
        invoked: Whether the failed provider call started.

    Returns:
        Commit error with failure-time partitions and the applicable notes.
    """
    commit_error = build_commit_error(err, step, outcome, unit_of_work)
    if note := batch_failure_note(step, outcome, unit_of_work, invoked=invoked):
        add_note_once(commit_error, note)
    add_auth_note(err, commit_error, AUTH_RECOVERY_NOTE)
    return commit_error


def projection_result(call: FindCall, records: WireRecords) -> Result:
    """Build projection rows in requested field order from validated records."""
    return Result(Row({field: record.get(field) for field in call.fields}) for record in records)


def refresh_call(state: EntityState, schema: EntitySchema) -> FindCall:
    """Build a point refresh in the entity's current retirement scope."""
    return FindCall(
        entity_type=state.entity_type,
        filters=[['id', 'is', state.id]],
        fields=schema.readable_field_codes(),
        limit=1,
        extra={'retired_only': True} if state.deleted else {},
    )


def route_batch_failure(  # noqa: PLR0913 - Routing needs the failure evidence and the caller's raw-passthrough mode.
    err: Exception,
    outcome: CommitOutcome,
    unit_of_work: UnitOfWork,
    *,
    sole_batch_step: CommitStep | None,
    invoked: bool,
    raw_passthrough: bool,
) -> CommitError | None:
    """Route a sole-batch failure to its caller, or wrap any other failure.

    Args:
        err: Ordinary failure raised during commit execution.
        outcome: Failure-time partitions and the authoritative current step.
        unit_of_work: Session staging state holding any accepted batch.
        sole_batch_step: The plan's only step when it is a batch, else `None`.
        invoked: Whether the failed provider call started.
        raw_passthrough: Whether the caller re-raises a returned-`None` failure
            unwrapped; when false, the caller's existing wrap path builds the
            `CommitError`.

    Returns:
        `None` for an unaccepted sole batch, with `err` annotated only for raw
        passthrough, or the wrapped failure.
    """
    step = outcome.current_step
    if sole_batch_step is not None and not accepted_batch_reported(unit_of_work, step, outcome):
        if raw_passthrough:
            add_note_once(err, SOLE_BATCH_STAGED_NOTE)
            if note := batch_failure_note(step, outcome, unit_of_work, invoked=invoked):
                add_note_once(err, note)
            add_note_once(err, f'sgalchemy: failed during {sole_batch_step.description}')
            add_auth_note(err, err, AUTH_RECOVERY_NOTE)
        return None
    return commit_error_with_batch_status(err, step, outcome, unit_of_work, invoked=invoked)


def validate_query_payload(call: FindCall | SummarizeCall, payload: object) -> QueryResult:
    """Return a checked query payload or reject its provider outer shape.

    Args:
        call: Compiled find or summarize call that produced `payload`.
        payload: Raw provider response for `call`.

    Returns:
        The unchanged payload: a record list for a find call, or a summary
        mapping for a summarize call.

    Raises:
        SgalchemyError: If the payload does not match the compiled call's
            shape.
    """
    if isinstance(call, FindCall):
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in cast('list[object]', payload)
        ):
            raise SgalchemyError('find payload must be a list of dictionaries')
    elif not isinstance(payload, dict):
        raise SgalchemyError('summarize payload must be a dictionary')
    return cast('QueryResult', payload)
