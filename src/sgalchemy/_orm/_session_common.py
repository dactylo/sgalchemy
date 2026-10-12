"""Share pure session result and refresh-call handling."""

from typing import cast

from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._engine_common import AUTH_READ_NOTE, add_note_once, looks_like_auth_fault
from sgalchemy._core._registry import EntityState
from sgalchemy._core._result import Result, Row
from sgalchemy._errors import SgalchemyError
from sgalchemy._shotgun import QueryResult, WireRecords
from sgalchemy.schema._model import EntitySchema


def annotate_read_error(error: Exception, context: str) -> None:
    """Attach operation context and authentication recovery guidance to a fault."""
    add_note_once(error, f'sgalchemy: {context}')
    if looks_like_auth_fault(error):
        add_note_once(error, AUTH_READ_NOTE)


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


def validate_query_payload(call: FindCall | SummarizeCall, payload: object) -> QueryResult:
    """Return a checked query payload or reject its provider outer shape.

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
