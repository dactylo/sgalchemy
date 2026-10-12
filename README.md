# sgalchemy

sgalchemy is a SQLAlchemy-style ORM for Autodesk Flow Production Tracking. It gives pipeline code
typed entity classes reflected from your site's schema, `select()` queries with relationship
loaders, and a `Session` that keeps one object per record and sends your changes only when you call
`commit()`. You decide when the network is used: reading an attribute never makes a hidden provider
call.

This documentation uses SG for Flow Production Tracking, formerly named Shotgun and ShotGrid.

## Why use it?

Calling the provider API directly means working with dictionaries: you choose field lists for each
`find()`, track which records changed, and assemble `batch()` requests yourself. sgalchemy handles
that bookkeeping while keeping every provider call explicit.

The SQLAlchemy concepts are familiar, but SG is not a relational database, and sgalchemy follows
the provider where the two differ:

- Schema loading is explicit. Call `load_schema()` before you use entity classes; sgalchemy caches
  the schema on disk so later runs can skip the fetch.
- Queries never send pending changes. `commit()` is the only write boundary.
- Reading a field or relationship that a query did not load raises an error instead of calling the
  network. Synchronous code can opt into `lazyload()`.
- A commit is atomic only when it compiles to one transactional `batch()` call. Revives, thumbnail
  uploads, and creates that link to other new records need extra calls, so a failure can leave part
  of a commit applied. [Sessions and writes](docs/sessions.md) explains how to recover.

## Installation

You need Python 3.11 or newer and one compatible SG API provider. From a checkout, install
sgalchemy with the extra for your provider. For Autodesk's provider, run:

```sh
python -m pip install '.[official]'
```

Choose exactly one extra, and replace `official` in the command to pick another:

- `official` installs Autodesk's `shotgun-api3` for synchronous and threaded async engines.
- `api` installs `sg-python-api` for synchronous and threaded async engines.
- `async` installs `sg-python-api` with its native async client, which also supports native async
  engines.

If your application already supplies a compatible API client, install the base package with
`python -m pip install .` and install the client separately.

## Quick start

This example queries in-progress shots and creates one shot. It authenticates with a session token
that your application supplies in `SHOTGUN_SESSION_TOKEN`; see
[Connections and authentication](docs/authentication.md) for other credentials. Replace the site
URL and the status and code values with your studio's, and add any other fields your site requires
for a new shot. Run it against a test site: the commit creates a real record.

```python
import os

from sgalchemy import Session, create_engine, select

engine = create_engine(
    'https://example.shotgunstudio.com',
    session_token=os.environ['SHOTGUN_SESSION_TOKEN'],
)
try:
    engine.load_schema()
    from sgalchemy.entities import Shot

    with Session(engine) as session:
        shots = session.scalars(select(Shot).where(Shot.sg_status_list == 'ip'))
        print(f'{len(shots)} shots in progress')

        new_shot = Shot.new(code='sq010_sh020')
        session.add(new_shot)
        session.commit()
        print(new_shot)
finally:
    engine.dispose()
```

The script prints the number of matching shots, then the new shot with its server ID, such as
`Shot(id=1234, code='sq010_sh020')`.

`load_schema()` fetches the site schema, or reads a fresh cached copy, before you import entity
classes. The query does not send the pending create; `session.commit()` does. Leaving the `with`
block closes the session and discards any work you did not commit. If a commit fails after a
provider call, [check what was applied](docs/sessions.md#recover-from-a-failed-commit) before you
retry.

To give your type checker the reflected fields of `Shot`, generate site-specific stubs; see
[typed stubs](docs/tooling.md#generate-typed-stubs).

## Documentation

| If you want to...                                           | Read                                                            |
| ----------------------------------------------------------- | --------------------------------------------------------------- |
| Choose credentials and recover when SG rejects them         | [Connections and authentication](docs/authentication.md)        |
| Write queries and load related entities                     | [Queries and relationships](docs/relationships.md)              |
| Commit changes, upload thumbnails, or use session factories | [Sessions and writes](docs/sessions.md)                         |
| Recover from a failed or partial commit                     | [Failed commits](docs/sessions.md#recover-from-a-failed-commit) |
| Use sgalchemy from asyncio code                             | [Async usage](docs/async.md)                                    |
| Type-check with stubs, manage the schema cache, or test     | [Typing, schema cache, and testing](docs/tooling.md)            |
| See changes by version                                      | [Changelog](CHANGELOG.md)                                       |

## Development

See the [development guide](docs/development.md) for the required tools, CI, and dependency locks.
With those tools installed, run:

```sh
make setup
make check
```

The checks never contact a live SG site unless you configure the opt-in live test.

## License

sgalchemy is available under the [MIT License](LICENSE). It is an independent project and is not
affiliated with or endorsed by Autodesk. Flow Production Tracking is a product of Autodesk, Inc.
