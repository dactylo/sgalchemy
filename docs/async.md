# Async usage

Use `AsyncSession` to query and write from asyncio code. Provider calls happen at awaited
boundaries; attribute reads never hide I/O. Start with the
[README quick start](../README.md#quick-start) for the synchronous lifecycle.

- [Create an async engine](#create-an-async-engine) and [load relationships](#load-relationships).
- [Use one session per task](#use-one-session-per-task), or use
  [task-scoped sessions](#task-scoped-sessions).
- Look up [engine strategies](#engine-strategies-and-lifetime) or
  [recover from cancellation](#cancellation-and-interrupted-writes).

## Create an async engine

Use `create_async_engine()` and await the session methods that call SG. This example uses
`strategy='threaded'`, which works with the `official`, `api`, or `async` provider extra:

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

The query returns shots with their assets loaded; `session.load()` then loads their sequences.
Constructing an engine performs no network or cache I/O. `await engine.load_schema()` loads the
schema, and `await engine.dispose()` releases the engine-owned clients when you finish.

For native async calls or a client your application supplies, see
[Engine strategies and lifetime](#engine-strategies-and-lifetime).

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

## Task-scoped sessions

`async_scoped_session(factory)` gives each asyncio task its own session, keyed by
`asyncio.current_task()`. Using the registry outside a running task raises `SgalchemyError`.

- `await scoped.remove()` closes and removes only the calling task's session. It is optional, can
  run early, and is safe to call more than once.
- `await scoped.close()` closes the session's local state but leaves the task's registry entry in
  place until removal or task completion.

Automatic cleanup never commits. Call `commit()` or `rollback()` explicitly whenever the local
outcome matters.

### Automatic task cleanup

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

## Engine strategies and lifetime

`create_async_engine()` builds an `AsyncEngine`. The strategy follows the source keyword;
`strategy=` is needed only to state it explicitly. Choose a provider extra according to the
strategy. With the `async` extra, use `strategy='native'` for native async calls.

| Strategy                                                     | Provider extras               |
| ------------------------------------------------------------ | ----------------------------- |
| native (no threaded keyword)                                 | `async`                       |
| threaded (`sync_connection=`, `creator=`, or `max_workers=`) | `official`, `api`, or `async` |

The native strategy calls the provider's async client directly. The threaded strategy runs a
synchronous client in worker threads. Both accept the static credentials described in
[Connections and authentication](authentication.md#choose-a-construction-path). A native
`connection=` supplies an async client and `async_creator=` builds the engine's one async client.
For threaded use, `sync_connection=` supplies a sync client and `creator=` builds one client per
worker. Each created client needs the surface described under
[construction paths](authentication.md#choose-a-construction-path): native clients use async
methods, including `close()`, and threaded clients use synchronous ones.

For threaded engines, `max_workers` is the maximum number of concurrent provider calls. It defaults
to 1 with a supplied `sync_connection` and 4 with engine-owned clients. Passing a larger value with
a supplied connection asserts that your client is thread-safe.

An `AsyncEngine` binds to the event loop that first uses it; using it from another loop raises
`RuntimeError`. Share it between tasks on that loop, but do not move it between loops.
`async with engine:` binds the loop without loading the schema and disposes the engine-owned
clients on exit.

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

If the exception carries the note `sgalchemy: SG saved ... before this failure`, SG saved that
batch before the interruption. It counts as committed, the session is not quarantined, and
[Response reconciliation](sessions.md#response-reconciliation) says what to do next.

An ordinary exception from a thumbnail upload is handled differently: it uses the field-level
recovery described in [Thumbnails](sessions.md#thumbnails) and leaves unrelated entity state
usable.
