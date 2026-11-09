# Sessions and writes

A session (`Session`) keeps one object per SG record and holds your changes locally until you call
`commit()`. This page covers the everyday write lifecycle, what to do when a commit fails, and the
details of thumbnails, field trust, and session factories. If you have not run a first query yet,
start with the [README quick start](../README.md#quick-start).

| If you want to...                                     | Start here                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------- |
| Create, change, retire, or revive records             | [Everyday lifecycle](#everyday-lifecycle)                     |
| Upload or clear thumbnails                            | [Thumbnails](#thumbnails)                                     |
| Create sessions from a factory or commit in a block   | [Session factories](#session-factories)                       |
| Know whether a commit is all-or-nothing               | [When a commit is atomic](#when-a-commit-is-atomic)           |
| Decide what to do after a commit fails                | [Recover from a failed commit](#recover-from-a-failed-commit) |
| Handle `RefreshRequiredError` or unverified values    | [Field trust](#field-trust)                                   |
| Check which session methods are supported             | [Supported session methods](#supported-session-methods)       |
| Understand how sgalchemy checks a successful response | [Response reconciliation](#response-reconciliation)           |

## Everyday lifecycle

Stage changes on objects the session owns, then send them with one `commit()`:

```python
from sgalchemy import Session

with Session(engine) as session:
    shot = session.get(Shot, 1234)
    if shot is not None:
        shot.description = 'Locked cut'
    session.add(Shot.new(code='sq010_sh030'))
    session.commit()
```

Nothing reaches SG until `commit()`; queries never send staged changes. The session tracks these
local operations:

- `Shot.new(**fields)` builds a pending entity, validated against the schema, and
  `session.add(obj)` stages its creation. It also re-attaches an entity kept from a closed session
  so later field assignments are sent by `commit()`. Assigning to such an entity before `add()`
  raises an error.
- Assigning a field on a loaded entity stages an update.
- `session.delete(obj)` stages retirement. SG retires records rather than erasing them, and
  `session.revive(obj)` stages the revival of a retired record. `delete()` also cancels a staged
  create or revival, and `revive()` cancels a staged delete.
- `session.commit()` sends all staged work. See [When a commit is atomic](#when-a-commit-is-atomic)
  before you rely on all-or-nothing behavior.
- `session.rollback()` restores each changed field to its previous state and discards staged
  creates, deletes, and revives. It changes only local state and cannot undo a write that SG
  already applied.
- `session.close()`, which also runs when the `with` block ends, does the same local cleanup,
  detaches the session's entities, and clears its identity map. It never commits. An entity you
  keep after closing sees its uncommitted edits reverted, not preserved. The closed session can be
  used again.

### Retired records

Query retired records with `select(Shot).execution_options(retired_only=True)`. To change a retired
record, stage `session.revive(obj)` along with the edit. Otherwise `commit()` raises
`RetiredEntityWriteError` before any schema or provider I/O.

When an object has both a field change and a staged delete, the update is normally sent before the
delete in the same batch. If the object also has a thumbnail upload, the commit sends ordinary
updates first, then the upload and its readback, and finally the delete in a separate retirement
batch.

When the provider's `revive()` or a batch delete returns `False`, sgalchemy treats the record as
already in the requested state: already active or already retired. A missing record or a permission
failure still raises the provider's exception.

### Editable fields

The reflected schema records whether each field is editable, and sgalchemy rejects a local
assignment to a field that the schema marks noneditable. Fields with no `editable` metadata count
as editable. This check uses cached schema, so it can be stale: if a field has become editable,
call `engine.load_schema(force=True)` and assign it again. SG's role permissions remain the final
authority, and the provider's own permission exception reaches you unchanged.

## Thumbnails

Assign a local `str` or `os.PathLike` path to the conventional `image` or `filmstrip_image` field,
then commit. Assignment performs no I/O.

```python
from pathlib import Path

shot.image = Path('shot.jpg')
shot.filmstrip_image = Path('shot-filmstrip.jpg')
session.commit()

# Clear both thumbnails.
shot.image = None
session.commit()
```

Commit checks that every path names a readable, nonempty file before its first provider write, so a
bad path fails before anything is sent. It then calls the provider's dedicated upload operation for
each field. After any upload attempt, the entity no longer holds your local path as though it were
the provider's value.

Uploaded fields are read back in bounded groups. The value you see afterward is SG's current value,
which may be absent, temporary, or final.

### Clear thumbnails

To clear thumbnails, assign `image=None`; it clears both. Assigning `filmstrip_image=None` raises
an error, because the provider silently ignores it. A confirmed clear sets both local values to
`None` and removes both fields from `refresh_required` and `unverified_writes`. If
`filmstrip_image` has an unsent assignment when the clear is confirmed, such as a new path in the
same commit, the clear leaves that value and its markers alone. Assigning `image=None` also
discards a `filmstrip_image` path you assigned before it in the same session, so that upload is
never sent. To clear both thumbnails and upload a new filmstrip in one commit, assign `image=None`
first and the new `filmstrip_image` path after it.

### Recover from an upload failure

An ordinary upload exception raises `CommitError`, removes the local path, and performs at most one
field-level readback. If that read succeeds, the field holds SG's current value as unverified; if
it fails, reading the field raises [`RefreshRequiredError`](#field-trust).

## Session factories

Factories create sessions with shared settings:

| Factory or registry                                  | Sessions and accepted settings                               |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `sessionmaker(engine, *, max_implicit_queries=None)` | Independent synchronous sessions; optional lazy-query budget |
| `async_sessionmaker(engine)`                         | Independent async sessions; no defaults accepted             |
| `scoped_session(factory)`                            | One session per thread                                       |
| `async_scoped_session(factory)`                      | One session per asyncio task; no custom `scopefunc`          |

See [Task-scoped sessions](async.md#task-scoped-sessions) for async cleanup.

Unsupported arguments fail when you construct the factory. Both scoped registries are callable and
pass public session methods through to the current session.

### Commit on success

Each factory offers `commit_on_success()`, a context manager that commits once and always attempts
to close the session:

```python
from sgalchemy import AsyncEngine, Engine, async_sessionmaker, sessionmaker


async def commit_examples(engine: Engine, async_engine: AsyncEngine) -> None:
    factory = sessionmaker(engine)
    with factory.commit_on_success() as session:
        session.add(engine.entities.Shot.new(code='sq010_sh020'))

    async_factory = async_sessionmaker(async_engine)
    async with async_factory.commit_on_success() as session:
        session.add(async_engine.entities.Shot.new(code='sq010_sh030'))
```

When the block finishes cleanly, the manager performs one ordinary commit, then closes the session.
If the block raises, it skips the commit. It is not a transaction: it adds no retry or rollback of
provider writes, and an explicit commit inside the block can send writes that a later failure in
the block cannot undo. Factories provide no `begin()`, because SG cannot keep a transaction promise
across revives, batches, uploads, readbacks, and retirement batches.

If the factory's commit raises `CommitError`, its [four partitions](#recover-from-a-failed-commit)
describe the instant that commit failed. When the automatic close finishes, it has discarded the
session's retry queue, so restage any `staged` work you still want on a usable session.

### If automatic close fails

If closing the session fails with an ordinary error after a body or commit failure, the original
failure remains the one you receive. If a note says the factory's `Session.close()` or
`AsyncSession.close()` did not finish, call `close()` on the session again. The close failure is
attached as a note when possible, or logged on the `sgalchemy.session` logger. A close failure on
its own propagates. If close is interrupted while handling a body or commit failure, the
interruption propagates with that earlier failure as its context.

## When a commit is atomic

`commit()` compiles staged changes into transactional provider `batch()` calls. A commit is atomic
only when it compiles to exactly one `batch()` call. SG offers no transaction across several calls,
and sgalchemy does not add one, so a commit that needs several calls can fail partway.

A commit makes its calls in this order:

1. One `revive()` call for each staged revival.
2. Batches of creates, updates, and deletes. A create that links to another new record waits for a
   later batch, so it can reference that record's server ID.
3. One upload call for each thumbnail, followed by a readback of the uploaded fields.
4. A final retirement batch for deleted records that also had thumbnail uploads.

`commit(batch_size=n)` splits each batch into calls of at most `n` requests, which can turn a
single-batch commit into several calls. The size must fit every group of requests that cannot be
split; an ordinary update and delete of the same record need at least two.

Before its first write, a commit may load the schema or force one refresh, then rebuild and
revalidate the whole plan. It never refreshes the schema after the first write starts.

## Recover from a failed commit

How a failure is reported depends on when it happens and how many calls the commit needed:

- **Before any write:** local validation failures, such as `RetiredEntityWriteError`, an unusable
  thumbnail path, or an invalid `batch_size`, raise before sgalchemy sends anything. A
  schema-loading failure also reaches you before any write.
- **One `batch()` call:** an ordinary provider failure raises the provider's own exception, not
  `CommitError`, and every operation stays staged in the session. Its notes include
  `sgalchemy: single-batch commit; all operations remain staged in this session`.
- **Any other failure after writing starts:** the commit raises `CommitError`. This includes every
  failure in a commit that needs several calls, and a successful batch response that sgalchemy
  cannot reconcile safely. When a provider call failed, its exception is the `CommitError`'s cause.

`CommitError` carries four disjoint snapshots of the plan at the moment of failure:

| Attribute       | Contains                                                                                             | Retried by `commit()`   |
| --------------- | ---------------------------------------------------------------------------------------------------- | ----------------------- |
| `committed`     | Work SG confirmed                                                                                    | No                      |
| `staged`        | Work still queued for retry, including a submitted batch that raised before its outcome was known    | Yes                     |
| `indeterminate` | Submitted writes whose result cannot be established safely                                           | Never                   |
| `withdrawn`     | Later work for the same records that was never sent, held back because an earlier outcome is unknown | No; restage it yourself |

Treat partition elements as opaque diagnostic evidence. You can check a partition's length, or test
whether it contains an element you kept a reference to, by identity:
`any(item is kept for item in partition)`. sgalchemy does not promise the elements' fields, tuple
positions, or classes, and the partitions cannot map a failure back to the entities you staged. Use
the checks for each kind of work below instead.

### Decide whether to retry

The exception's notes say whether a failed batch reached the provider:

- `sgalchemy: the failed batch was not sent to the provider` means a retry is safe.
- `sgalchemy: the failed batch call started but returned no usable result; its server outcome is unknown`
  means SG may have applied the batch even though the call raised. A plain `commit()` resubmits any
  of its work that remains staged.
- `sgalchemy: SG saved batch wave 1 chunk 1 before this failure; it counts as committed and a later commit will not send it again`
  means SG saved that batch before the failure. Nothing in it needs redoing, and a later plain
  `commit()` sends only work that was never sent. Some fields sent in that batch may need a
  refresh; reading one raises `RefreshRequiredError`, which says what to do.

When the outcome is unknown, reconcile each kind of staged work before you commit again:

- **Creates:** `inspect(obj).id is None` identifies a create without a confirmed server ID. If your
  application has a field whose value is unique, query SG by it. Call `session.delete(obj)` to
  cancel a staged create that SG already applied.
- **Updates:** a retry resends the staged values. That is harmless only if nobody changed those
  fields on the server since the failure; otherwise the retry overwrites their edits. Use
  `rollback()` to discard the staged values instead.
- **Deletes and revives:** repeating one returns the requested state, so a retry is harmless. Call
  `session.revive(obj)` to cancel a staged delete you no longer want.

After reconciling, restage `withdrawn` work explicitly. Discard objects whose outcome is
indeterminate, which show as `Shot(indeterminate)`, and reconcile those records through a fresh
query in a new session. An interruption such as `KeyboardInterrupt` while a provider call is in
flight also leaves that call's work indeterminate.

A failed thumbnail upload follows the field-level recovery in [Thumbnails](#thumbnails).
Cancellation of an async commit is described in
[Async usage](async.md#cancellation-and-interrupted-writes).

## Field trust

After an uncertain write, sgalchemy may not know a field's current value. It marks such fields so
that you never read a guess as though it came from SG:

- A **refresh-required** field has no trustworthy current value. Reading it raises
  `RefreshRequiredError`. Call `session.refresh(obj)`, or `await session.refresh(obj)` in async
  code, or assign the field again as new intent. Other fields remain usable.
- An **unverified** field holds a current value that sgalchemy read back from SG after an ambiguous
  write, but sgalchemy cannot tell whether your write produced it.

`RefreshRequiredError` does not inherit `AttributeError`, so `hasattr()` and
`getattr(obj, name, default)` cannot hide this condition. `UnfetchedFieldError` does inherit
`AttributeError`, because a field that a query did not load is a safe, expected absence; to ask
whether a field exists, use the class or schema metadata.

`inspect(obj)` exposes `refresh_required` and `unverified_writes` without I/O. `as_dict()` omits
refresh-required fields and includes loaded unverified values. `session.refresh(obj)` reloads
readable fields without overwriting your unsent changes. It clears the markers only on fields it
refreshed; fields with unsent changes and fields SG omitted keep theirs. Rollback and close restore
each changed field's previous state, including whether it was loaded, unfetched, refresh-required,
or unverified.

`session.expire(obj)` and `session.expire_all()` mark entities as expired. Reading a field of an
expired entity raises `StaleObjectError` until you call `session.refresh(obj)`.

## Reference

### Supported session methods

| Method or attribute | `Session`     | `AsyncSession` |
| ------------------- | ------------- | -------------- |
| `add()`             | Call directly | Call directly  |
| `close()`           | Call directly | Await          |
| `commit()`          | Call directly | Await          |
| `delete()`          | Call directly | Call directly  |
| `engine`            | Read property | Read property  |
| `execute()`         | Call directly | Await          |
| `expire()`          | Call directly | Call directly  |
| `expire_all()`      | Call directly | Call directly  |
| `get()`             | Call directly | Await          |
| `load()`            | Not available | Await          |
| `merge_dict()`      | Call directly | Call directly  |
| `refresh()`         | Call directly | Await          |
| `revive()`          | Call directly | Call directly  |
| `rollback()`        | Call directly | Call directly  |
| `scalars()`         | Call directly | Await          |
| Context manager     | `with`        | `async with`   |

The directly called async methods perform no I/O. Scoped registries also provide `remove()`: call
it directly on `ScopedSession`, and await it on `AsyncScopedSession`.

Other non-underscore session attributes are package-internal and unsupported.

The factories return `SessionFactory` and `AsyncSessionFactory`, and the scoped registries are
`ScopedSession` and `AsyncScopedSession`. Use these names for annotations; constructing them
directly is unsupported.

### Response reconciliation

A successful `batch()` response is validated as a whole before sgalchemy applies any of it. Use the
following diagnostics to decide what to do next.

#### Untrustworthy or interrupted response

If the response is structurally untrustworthy, submitted operations become indeterminate and later
work for the same records is withdrawn. A failure or interruption as the response arrives can also
leave the batch indeterminate. Indeterminate work is never sent again; follow
[failed-commit recovery](#recover-from-a-failed-commit).

#### Confirmed write could not be loaded into the object

An `ERROR` on `sgalchemy.session` saying `confirmed-write echo hydration failed for Shot(id=51)`
means SG saved the write, but sgalchemy could not apply the returned record. Do not repeat the
write. Reading a submitted field may raise `RefreshRequiredError`, which says what to do.

#### SG saved the batch before an exception

The saved-batch note in [Decide whether to retry](#decide-whether-to-retry) means the write
succeeded, even for an interruption such as `KeyboardInterrupt`. These further messages explain
recovery:

| Message or object state                                                                                                           | What to do                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `sgalchemy: SG saved that batch, but sgalchemy could not finish updating this session's objects; ...`                             | Carry on as usual; sgalchemy tries again on the next use, rollback, or close |
| Error beginning `SG saved a batch from this session's last commit, but sgalchemy could not finish updating the session's objects` | Call `rollback()` or `close()`, then query the affected records again        |
| `sgalchemy: rollback() did not finish` or `sgalchemy: close() did not finish`                                                     | Call `rollback()` or `close()` again                                         |
| `Shot(id=26, outdated)`                                                                                                           | Discard the object and query its record again; the new object is usable      |

A successful retry during ordinary session use preserves other staged work. `rollback()` and
`close()` discard it as usual.

#### Submitted fields missing from the response

A trustworthy response that omits submitted fields causes a `provider-response contract violation`
warning on `sgalchemy.session`. sgalchemy reads the current values at most once for each batch
call, entity type, provider query chunk, and active or retired lookup. If that read fails, only
those fields become refresh-required. It does not retry the write, mark the whole entity
untrustworthy, or repeat the read.
