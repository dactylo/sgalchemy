# Async usage

This page covers sgalchemy in asyncio code: creating an async engine, using sessions across tasks,
loading relationships, and handling cancellation. The async API mirrors the synchronous one shown
in the [README quick start](../README.md#quick-start), with every provider call at an awaited
boundary so that attribute access never hides I/O.

## Create an async engine

`create_async_engine()` builds an `AsyncEngine`. Choose a strategy according to your installed
provider extra:

| Strategy                         | Provider extras               | Supplied client    | Creator          |
| -------------------------------- | ----------------------------- | ------------------ | ---------------- |
| `strategy='native'`, the default | `async`                       | `connection=`      | `async_creator=` |
| `strategy='threaded'`            | `official`, `api`, or `async` | `sync_connection=` | `creator=`       |

The native strategy calls the provider's async client directly. The threaded strategy runs a
synchronous client in worker threads. Both accept the static credentials described in
[Connections and authentication](authentication.md#choose-a-construction-path). A native
`async_creator` builds the engine's one async client; a threaded `creator` builds one client per
worker.

```python
import asyncio
import os

from sgalchemy import AsyncSession, create_async_engine, select, selectinload


async def main() -> None:
    engine = create_async_engine(
        'https://example.shotgunstudio.com',
        session_token=os.environ['SHOTGUN_SESSION_TOKEN'],
        strategy='threaded',
    )
    try:
        await engine.load_schema()
        Shot = engine.entities.Shot

        async with AsyncSession(engine) as session:
            statement = select(Shot).options(selectinload(Shot.assets))
            shots = await session.scalars(statement)
            await session.load(shots, Shot.sg_sequence)
    finally:
        await engine.dispose()


asyncio.run(main())
```

This example uses the threaded strategy, so it works with any of the three provider extras. With
`[async]`, use `strategy='native'` for native async calls.

Constructing the engine performs no network or cache I/O; `await engine.load_schema()` is the
explicit boundary. For threaded engines, `max_workers` is the maximum number of concurrent provider
calls. It defaults to 1 with a supplied `sync_connection` and 4 with engine-owned clients. Passing
a larger value with a supplied connection asserts that your client is thread-safe.

An `AsyncEngine` binds to the event loop that first uses it; using it from another loop raises
`RuntimeError`. Share it between tasks on that loop, but do not move it between loops.
`async with engine:` binds the loop without loading the schema and disposes the engine-owned
clients on exit.

## Use one session per task

Use one `AsyncSession` per concurrent task, or `async_scoped_session()` for task-local sessions. A
session operation requires a running asyncio task, and a session raises `SgalchemyError` if a
second task uses it while one of its operations is active.

`AsyncSession` awaits the methods that perform I/O, such as `scalars()`, `commit()`, and `close()`;
see [Supported session methods](sessions.md#supported-session-methods) for the full list.

The model-level finder `Shot.find()` uses the synchronous default session that
`sgalchemy.connect()` installs. It has no async form and does not change its return type inside an
event loop; async code uses `await session.scalars(select(...))` instead.

## Load relationships

Async relationship loading is always explicit. Add `selectinload()` or `joinedload()` to the query,
or call `await session.load(objects, relationship)` after it. An async session rejects a statement
that uses `lazyload()`, because an ordinary attribute read cannot await I/O. See
[Queries and relationships](relationships.md#choose-a-relationship-loader) for what each loader
costs.

## Cancellation and interrupted writes

Cancellation behaves differently in each strategy:

- **Native:** `asyncio.CancelledError` propagates through the provider's async operation.
- **Threaded:** cancellation stops waiting for the result, but it cannot stop a synchronous
  provider call that has already started.

If cancellation or another `BaseException` arrives while a write is in flight, sgalchemy cannot
know whether SG applied it. That step becomes indeterminate, the session is quarantined, and only
the affected entities become unusable. Reconcile those records with a fresh query, close the
quarantined session, and create a new session. The quarantined session refuses further work and
stays unusable after you close it.

An ordinary exception from a thumbnail upload is handled differently: it uses the field-level
recovery described in [Thumbnails](sessions.md#thumbnails) and leaves unrelated entity state
usable.

## Task-scoped sessions

`async_scoped_session(factory)` gives each asyncio task its own session, keyed by
`asyncio.current_task()`. Using the registry outside a running task raises `SgalchemyError`.

- `await scoped.remove()` closes and removes only the calling task's session. It is optional, can
  run early, and is safe to call more than once.
- `await scoped.close()` closes the session's local state but leaves the task's registry entry in
  place until removal or task completion.

Automatic cleanup never commits. Call `commit()` or `rollback()` explicitly whenever the local
outcome matters.

When an owning task completes, the registry removes its entry, then cleans up without provider I/O:
it restores changed fields, detaches entities, and discards identity and staged state. If unsent
work existed, sgalchemy logs one `WARNING` on `sgalchemy.session` beginning
`automatic task cleanup detached entities, reverted local edits, and discarded uncommitted work`.
The warning includes the task name, its cancellation status, and counts of changed entities,
pending creates, deletes, and revives, but never field values.

If another task is borrowing the session when cleanup runs, sgalchemy postpones the destructive
cleanup until that operation ends and logs one misuse warning. Cleanup failures are logged and
never hide an unretrieved task exception.

Cleanup runs after the owning task finishes and the event loop runs at least one more iteration.
Its order relative to tasks awaiting the owner is unspecified, and it is best effort during
abnormal interpreter or event-loop shutdown. If you need an entity's current local values after the
owning task's next `await`, call `as_dict()` inside the owning task and return that snapshot.
