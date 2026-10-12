"""Shared synchronous and asynchronous engine behavior."""

import re
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Event
from typing import Any

from sgalchemy._core._compiler import FindCall, SummarizeCall
from sgalchemy._core._registry import Entity, Registry
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema._cache import SchemaCache, validate_cache
from sgalchemy.schema._model import EntitySchema, build_schema

AUTH_READ_NOTE = (
    'sgalchemy: authentication was rejected during a read; no automatic retry was '
    'attempted and no write was sent. A multi-phase read may retain trustworthy '
    'already-hydrated entities. Refresh credentials, then rerun the read; creator '
    'engines call engine.dispose() first.'
)
AUTH_RECOVERY_NOTE = (
    'sgalchemy: authentication was rejected; no automatic retry was attempted. '
    'Refresh credentials (creator engines call engine.dispose(); static-credential '
    'engines need a new engine and session; caller-owned connections must be '
    'repaired or replaced), then make an operation-specific retry or '
    'reconciliation decision.'
)
AUTH_UPLOAD_NOTE = (
    'sgalchemy: authentication was rejected during an upload; the upload outcome '
    'is unknown and it will not be re-attempted automatically. After credentials '
    'recover, call session.refresh(entity) or query through a fresh session to '
    'reconcile the field.'
)
_SCHEMA_FAULT_PATTERNS = (
    re.compile(r'\b(?:unknown|invalid)\s+(?:field|entity(?:\s+type)?)\b', re.IGNORECASE),
    re.compile(
        r'\b(?:field|entity(?:\s+type)?)\s+[^:;,\n]+\s+'
        r"(?:does not exist|doesn't exist|is unknown|is invalid)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r'\bAPI\s+[A-Za-z_]\w*\(\)\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+'
        r"\s+(?:does not exist|doesn't exist)\b",
        re.IGNORECASE,
    ),
)
_SCHEMA_SENSITIVE_METHODS = frozenset({'batch', 'find', 'find_one', 'summarize'})


@dataclass
class CallProgress:
    """Observe one provider call with monotonic, thread-safe events.

    `dispatched` records executor acceptance before the caller awaits.
    `invoked` records resolved provider entry after acquisition and
    diagnostics.
    Never clear or reuse either event for another call. Workers update only
    these observations, never Session state.
    """

    dispatched: Event = field(default_factory=Event)
    invoked: Event = field(default_factory=Event)


class EntityNamespace:
    """Provide no-I/O entity-class access through one prepared Registry."""

    def __init__(self, registry: Registry) -> None:
        """Bind the namespace to one Registry."""
        self._registry = registry

    def __getattr__(self, name: str) -> type[Entity]:
        """Resolve a canonical entity name or bound alias without I/O.

        Args:
            name: Canonical entity type or display-name alias in the Registry.

        Returns:
            Stable class owned by the namespace's Registry.

        Raises:
            SgalchemyError: If the Registry has no prepared schema binding.
            AttributeError: If the binding does not contain the requested name.
        """
        binding = self._registry.binding_snapshot()
        if binding is None:
            raise SgalchemyError('engine schema is not loaded; load schema before entity access')
        return self._registry.entity_class_for(name, binding=binding)


def add_note_once(err: BaseException, note: str) -> bool:
    """Attach one note unless it is already present.

    Args:
        err: Failure to annotate.
        note: Diagnostic note to attach.

    Returns:
        Whether the note was already present or was attached successfully.

    Raises:
        BaseException: If note inspection or attachment raises a non-ordinary
            control-flow exception outside `Exception`; it is propagated
            unchanged.
    """
    succeeded = False
    with suppress(Exception):
        notes = getattr(err, '__notes__', ())
        if note in notes:
            return True
        err.add_note(note)
        succeeded = True
    return succeeded


def looks_like_auth_fault(err: Exception) -> bool:
    """Return whether an exception is structurally an authentication fault.

    Args:
        err: Provider exception to classify.

    Returns:
        Whether the exception type inherits from a class named
        `AuthenticationFault`.
    """
    return any(cls.__name__ == 'AuthenticationFault' for cls in type(err).__mro__)


def add_auth_note(source: Exception, target: Exception, note: str) -> None:
    """Attach phase guidance when the source is an authentication fault.

    Args:
        source: Provider failure to classify.
        target: Failure that should receive recovery guidance.
        note: Recovery guidance for this operation phase.
    """
    if looks_like_auth_fault(source):
        add_note_once(target, note)


def looks_like_schema_fault(method: str, err: Exception) -> bool:
    """Return whether a backend fault suggests cached-schema drift.

    Args:
        method: Provider method that raised.
        err: Original provider exception.

    Returns:
        Whether the fault should invalidate prepared schema state.
    """
    if isinstance(err, OSError) or method not in _SCHEMA_SENSITIVE_METHODS:
        return False
    return any(pattern.search(str(err)) is not None for pattern in _SCHEMA_FAULT_PATTERNS)


def provider_call_arguments(
    call: FindCall | SummarizeCall,
) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    """Map compiler output to a provider method and its wire arguments.

    Args:
        call: Pure compiler output, whose nested wire values remain unchanged.

    Returns:
        Method name, positional arguments, and a fresh keyword mapping.
    """
    if isinstance(call, FindCall):
        return (
            'find',
            (call.entity_type, call.filters, call.fields),
            dict(order=call.order, limit=call.limit, page=call.page, **call.extra),
        )
    kwargs = dict(call.extra)
    if call.grouping is not None:
        kwargs['grouping'] = call.grouping
    return 'summarize', (call.entity_type, call.filters, call.summary_fields), kwargs


def validated_schema(cache: SchemaCache) -> dict[str, EntitySchema]:
    """Build provider schema only after validating its complete wire shape.

    Raises:
        SgalchemyError: If provider schema is malformed and cannot be
            published.
    """
    try:
        validate_cache(cache)
        return build_schema(cache)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise SgalchemyError(
            'provider returned malformed schema; cache was not updated'
        ) from error
