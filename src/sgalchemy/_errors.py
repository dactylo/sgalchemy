"""Exceptions for schema, query, and commit failures."""


class SgalchemyError(Exception):
    """Root of all sgalchemy exceptions."""


class CommitError(SgalchemyError):
    """Commit failure with four disjoint failure-time outcome snapshots.

    Partition elements are opaque diagnostic evidence. Their fields, tuple
    positions, and concrete classes are unsupported; inspect partition lengths
    or compare retained elements by object identity.

    Attributes:
        committed: Provider-confirmed plan items that must not be retried.
        staged: Plan items retained in the local retry queue, including a
            submitted batch whose provider call raised. Its server outcome may
            be unknown; reconcile before retrying such a batch.
        indeterminate: Submitted plan items whose outcome cannot be reconciled
            safely; never silently retried.
        withdrawn: Never-submitted same-owner follow-up plan items removed from
            automatic retry and requiring explicit restaging.
    """

    def __init__(
        self,
        message: str,
        *,
        committed: list[object] | None = None,
        staged: list[object] | None = None,
        indeterminate: list[object] | None = None,
        withdrawn: list[object] | None = None,
    ) -> None:
        """Initialize with the four disjoint outcome partitions.

        Args:
            message: Human-readable failure description.
            committed: Provider-confirmed plan items that must not be retried.
            staged: Plan items retained in the local retry queue, including a
                submitted batch whose provider call raised. Its server outcome
                may be unknown; reconcile before retrying such a batch.
            indeterminate: Submitted plan items whose outcome cannot be
                reconciled safely; never silently retried.
            withdrawn: Never-submitted same-owner follow-up plan items removed
                from automatic retry and requiring explicit restaging.
        """
        super().__init__(message)
        self.committed = committed or []
        self.staged = staged or []
        self.indeterminate = indeterminate or []
        self.withdrawn = withdrawn or []


class CompilationError(SgalchemyError):
    """Statement cannot be translated to a supported API call.

    Failed compilation prevents the primary query. Session execution may have
    already performed schema I/O, while direct `Select.compile()` remains
    I/O-free.
    """


class RefreshRequiredError(SgalchemyError):
    """Field lacks a trustworthy value; refresh or explicitly reassign it.

    This integrity error intentionally does not inherit `AttributeError`.
    """


class RelationshipTargetUnavailableError(SgalchemyError):
    """Observed relationship targets remain absent after one schema refresh.

    Make the target schemas visible, then retry the explicit load.
    """


class RetiredEntityWriteError(SgalchemyError):
    """Dirty retired entity requires `revive()` intent or local rollback."""


class SchemaUnavailableError(SgalchemyError):
    """Opaque entity schema is unavailable for field access or mutation.

    This unresolved-schema error intentionally does not inherit
    `AttributeError`.
    """


class StaleObjectError(SgalchemyError):
    """Expired attribute access; the message names `session.refresh(obj)`."""


class UnfetchedFieldError(SgalchemyError, AttributeError):
    """Field was not loaded; the message names the loader option that would load it."""
