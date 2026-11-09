# Connections and authentication

This page helps you choose how sgalchemy connects to SG and what to do when SG rejects a
credential. If you have not run a first query yet, start with the
[README quick start](../README.md#quick-start).

An engine holds the API clients, credentials, and reflected schema. Sessions call SG through their
engine. Start with a [session token](#connect-with-a-session-token), or
[choose another construction path](#choose-a-construction-path) if your application manages
clients. For an existing connection failure, go to
[credential recovery](#recover-from-a-rejected-credential).

## Connect with a session token

Supply a token issued by your application's authentication service, then load the schema:

```python
import os

from sgalchemy import create_engine

engine = create_engine(
    'https://example.shotgunstudio.com',
    session_token=os.environ['SHOTGUN_SESSION_TOKEN'],
)
try:
    engine.load_schema()
    Shot = engine.entities.Shot
finally:
    engine.dispose()
```

After `load_schema()`, the engine exposes the site's entity classes. See the
[README quick start](../README.md#quick-start) to query them with a session.

## Choose a construction path

Pass exactly one source of clients. These are the four paths, in the order most applications should
prefer them:

```python
create_engine(site, session_token=token)
create_engine(site, script_name=name, api_key=key)
create_engine(connection=existing_client)
create_engine(site, creator=make_client)
```

| Path                     | Who owns the client                | How a new credential takes effect                         |
| ------------------------ | ---------------------------------- | --------------------------------------------------------- |
| `session_token`          | The engine                         | Create a new engine and session                           |
| `script_name`, `api_key` | The engine                         | Create a new engine and session                           |
| `connection`             | Your application                   | Repair your client, or replace it and create a new engine |
| `creator`                | The engine, built by your callable | Refresh your credential source, then call `dispose()`     |

### Static credentials

A **session token** (`session_token`) is the preferred credential: a JSON-RPC session token issued
outside sgalchemy, by any issuer. It is a bearer secret with a sliding server-side lifetime and no
per-token revocation, so keep it out of code and logs. The CLI reads it from
`SGALCHEMY_SESSION_TOKEN` when `--script-name` is omitted.

A **script key** (`script_name` with `api_key`) provides a long-lived script identity for
deployments that manage keys directly. Rotate keys through your studio's normal process.

Both paths require an `https://` site URL: the engine refuses to send credentials in plain text.
The engine builds one client per thread from the installed provider.

`create_async_engine()` accepts the same static credentials. Its supplied-client and creator
options depend on the async strategy; see [Async usage](async.md#engine-strategies-and-lifetime).

## Use an existing client

Pass `connection=existing_client` when your application owns the client. The engine never closes it
and serializes its calls through that client by default. Pass `serialize=False` only when your
client documents that one instance is safe to share between threads.

## Build clients with a creator

Pass `creator=make_client` for proxies, custom CA bundles, custom providers, or credentials your
application refreshes. The callable builds one fresh engine-owned client per thread. Have each call
read the currently valid credential from your refreshed source; clients on different threads may
share the same credential.

Every returned client must meet these requirements:

- It belongs to the same site and logical identity for the engine's lifetime. sgalchemy rejects a
  client for a different site.
- It provides a string `base_url`, a `close()` method, and the provider methods sgalchemy calls,
  such as `find` and `batch`. When the engine first builds a client, it raises `TypeError` if any
  are missing.

See [Recovery examples](#examples) for a creator that reads a refreshed token.

## Dispose of clients

`engine.dispose()` closes and forgets the engine-owned clients. The engine remains usable: the next
operation creates fresh clients. Await `engine.dispose()` for an async engine. Disposal never
closes a supplied connection and never refreshes a credential itself; a creator engine's next
client uses whatever credential your creator reads.

## Recover from a rejected credential

When SG rejects a credential, sgalchemy does not retry automatically. If the provider raises its
`AuthenticationFault`, the exception you receive carries a note beginning
`sgalchemy: authentication was rejected` that summarizes these recovery steps:

- **Creator engines:** refresh your credential source, then call `engine.dispose()` (or await it)
  so the next client reads the new credential.
- **Static-credential engines:** obtain new credentials, then create a new engine and session.
  Credentials cannot be replaced inside an existing engine.
- **Supplied connections:** repair or replace your own client. A replacement client needs a new
  engine and session.

Recovering the credential does not tell you whether the failed operation reached the server. Make
an operation-specific decision before you retry: a read can simply run again, but a failed write or
upload may already be applied. Reconcile a failed thumbnail upload with `session.refresh()` rather
than committing it again, and follow
[Recover from a failed commit](sessions.md#recover-from-a-failed-commit) for other writes.

### Examples

These examples use an application-owned `is_authentication_failure()` check, because providers do
not expose one reliable, universal authentication exception. Names such as `credential_source`,
`run_operation()`, and `ProviderClient` stand for your application's own code and provider client
class.

For a creator engine, refresh the source before disposing, so the next creator call obtains the
current credential:

```python
def make_client():
    return ProviderClient(
        site,
        session_token=credential_source.current_token(),
    )


engine = create_engine(site, creator=make_client)
session = Session(engine)

try:
    run_operation(session)
except Exception as error:
    if not is_authentication_failure(error):
        raise
    credential_source.refresh()
    engine.dispose()
    if operation_was_upload:
        session.refresh(entity)  # Reconcile the field; do not recommit it.
    else:
        decide_retry_or_reconcile(error)
```

Async code follows the same steps, awaiting `async_engine.dispose()` and
`async_session.refresh(entity)`.

With static credentials, create a new engine and session, reconcile any uncertain outcome, and
restage only the changes you still intend:

```python
try:
    run_operation(session)
except Exception as error:
    if not is_authentication_failure(error):
        raise
    new_token = obtain_new_session_token()
    new_engine = create_engine(site, session_token=new_token)
    new_session = Session(new_engine)
    reconcile_then_restage(new_session)
```

A supplied client remains your application's responsibility. Repairing that same object keeps the
engine's binding but still requires an operation-specific decision. Replacing the client requires a
new engine and session:

```python
try:
    run_operation(session)
except Exception as error:
    if not is_authentication_failure(error):
        raise
    replacement_client = replace_client(existing_client)
    new_engine = create_engine(connection=replacement_client)
    new_session = Session(new_engine)
    reconcile_then_continue(new_session)
```
