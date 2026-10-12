from sgalchemy import _errors


def test_commit_error_carries_four_disjoint_partitions() -> None:
    upload = (object(), 'image', object())
    err = _errors.CommitError(
        'boom',
        committed=[upload],
        staged=[2],
        indeterminate=[3],
        withdrawn=[4],
    )
    assert (err.committed, err.staged, err.indeterminate, err.withdrawn) == (
        [upload],
        [2],
        [3],
        [4],
    )


def test_hierarchy() -> None:
    for exc in (
        _errors.CompilationError,
        _errors.RefreshRequiredError,
        _errors.RelationshipTargetUnavailableError,
        _errors.SchemaUnavailableError,
        _errors.UnfetchedFieldError,
        _errors.StaleObjectError,
        _errors.CommitError,
        _errors.RetiredEntityWriteError,
    ):
        assert issubclass(exc, _errors.SgalchemyError)
    assert not issubclass(_errors.RefreshRequiredError, AttributeError)
    assert not issubclass(_errors.RelationshipTargetUnavailableError, AttributeError)
    assert not issubclass(_errors.SchemaUnavailableError, AttributeError)
    assert issubclass(_errors.UnfetchedFieldError, AttributeError)
