# Typing, schema cache, and testing

This page covers the tools around the library: type stubs generated from your site's schema, the
`sgalchemy` command-line interface, the schema cache, annotation types, and testing pipeline code
without a live SG site. If you have not run a first query yet, start with the
[README quick start](../README.md#quick-start).

| If you want to...                                  | Start here                                          |
| -------------------------------------------------- | --------------------------------------------------- |
| Give your type checker your site's entity fields   | [Generate typed stubs](#generate-typed-stubs)       |
| Find, override, or refresh the cached schema       | [Schema cache](#schema-cache)                       |
| Test pipeline code with a fake provider or Mockgun | [Test your pipeline code](#test-your-pipeline-code) |
| Look up a CLI command, option, or exit status      | [Command-line reference](#command-line-reference)   |
| Annotate factories, sessions, or provider wrappers | [Annotation types](#annotation-types)               |

## Generate typed stubs

Reflected entity classes are built at run time, so a type checker cannot see their fields. Generate
editor and type-checker stubs from your site's schema.

For a script identity, supply the key in `SGALCHEMY_API_KEY` or enter it at the hidden terminal
prompt. In automation, inject the variable through your secret configuration. See
[Credentials and schema sources](#credentials-and-schema-sources) for session tokens and offline
cache files.

```sh
sgalchemy stubs -o typings/ \
    --site https://example.shotgunstudio.com \
    --script-name "$SHOTGUN_SCRIPT_NAME"
```

Point Pyright at the generated directory with `stubPath` in `pyrightconfig.json`:

```json
{
  "stubPath": "typings"
}
```

For mypy, set `mypy_path`:

```ini
[mypy]
mypy_path = typings
```

Keep the complete generated output under that path: the `sgalchemy/entities/` subtree and the
`sgalchemy/_core/_registry.pyi` companion work together. Generation owns these paths, removes
entity stubs for types that no longer exist, and preserves other files in the generated `_core`
directory.

## Use the typed classes

Import typed classes from `sgalchemy.entities` after an engine bound to the default Registry has
loaded the schema:

```python
from sgalchemy import Engine, Session, select


def shots_in_progress(engine: Engine):
    engine.load_schema()
    from sgalchemy.entities import Shot

    with Session(engine) as session:
        return session.scalars(select(Shot).where(Shot.sg_status_list == 'ip'))
```

For a second site, permissive fields, aliases, or thumbnail annotations, see
[Typing details](#typing-details).

## Keep stubs current

Regenerate stubs after the site schema changes. A connected `sgalchemy stubs` run can reuse a fresh
cached schema, so run `sgalchemy schema refresh` first:

```sh
site=https://example.shotgunstudio.com
sgalchemy schema refresh --site "$site" --script-name "$SHOTGUN_SCRIPT_NAME"
sgalchemy stubs -o typings/ --site "$site" --script-name "$SHOTGUN_SCRIPT_NAME"
```

To detect drift without rewriting files, for example in CI, run `sgalchemy stubs --check` with the
same source and output arguments.

## Schema cache

sgalchemy caches each site's schema on disk for 24 hours by default, so engines created within that
time skip the schema fetch. The cache lives in the platform user cache directory:

| Platform | Directory                                                                                             |
| -------- | ----------------------------------------------------------------------------------------------------- |
| macOS    | `~/Library/Caches/sgalchemy`                                                                          |
| Linux    | `$XDG_CACHE_HOME/sgalchemy` when that variable holds an absolute path, otherwise `~/.cache/sgalchemy` |
| Windows  | `%LOCALAPPDATA%\sgalchemy\Cache`                                                                      |

Set `SGALCHEMY_SCHEMA_CACHE` to a nonempty directory to override the location. Engine factories
also accept `cache_root` for a directory override and `policy` for a different freshness policy;
the default is `TTLPolicy(hours=24.0)` from `sgalchemy.schema`.

To fetch a new snapshot immediately, call `engine.load_schema(force=True)` or run
`sgalchemy schema refresh`. When a provider error suggests that the cached schema is out of date,
such as an unknown field, sgalchemy logs `possible schema drift, marking cache stale` on the
`sgalchemy.engine` logger, and the next operation fetches a fresh schema instead of reading the
cache.

## Test your pipeline code

`sgalchemy.testing.engine_over()` builds an engine over a fake that implements the synchronous
provider API surface (`ShotgunLike`). Use it to test pipeline code without a live site:

```python
from sgalchemy.testing import engine_over

engine = engine_over(fake_connection, cache_root=temporary_cache)
engine.load_schema()
Shot = engine.entities.Shot
```

Each `engine_over()` call uses a fresh Registry and does not bind the process default, so separate
fake sites stay isolated. Keyword arguments for `create_engine()`, including `registry` and
`bind_default`, override these defaults; `bind_default=True` without `registry` uses the process
default Registry. Its schema cache lives in a temporary directory unless you pass `cache_root=`.

### Mockgun

For integration tests, `sgalchemy.testing.mockgun_engine()` builds an isolated engine over Mockgun,
the in-memory test double included with the provider package (`shotgun_api3.lib.mockgun`), and
returns it with its schema already loaded. Mockgun has limits to keep in mind:

- Its schema paths are class-level, process-global state, so creating a Mockgun engine changes the
  paths shared by every Mockgun instance in the process.
- It lacks `summarize()`, several query options and filter shapes, and transactional `batch()`
  behavior. It runs batch requests one after another, so do not use it to test commit atomicity.

Mockgun helpers need an explicit `SchemaCache` snapshot and a directory for their schema files.
Given an authenticated synchronous provider connection `sg`, build the snapshot from its raw schema
responses:

```python
from datetime import UTC, datetime

from sgalchemy.schema import SchemaCache
from sgalchemy.testing import mockgun_engine

schema_cache = SchemaCache(
    fields=sg.schema_read(),
    entities=sg.schema_entity_read(),
    fetched_at=datetime.now(UTC),
)
engine = mockgun_engine(schema_cache, temporary_cache)
```

For async tests with the `[async]` extra, `sgalchemy.testing.async_mockgun_connection()` installs
the same kind of snapshot for the provider's `mockgun.AsyncShotgun`. Pass the returned connection
to `create_async_engine(connection=...)`, then call `await engine.load_schema()`:

```python
from sgalchemy import AsyncEngine, create_async_engine
from sgalchemy.testing import async_mockgun_connection


async def async_mockgun_engine(schema_cache, temporary_cache) -> AsyncEngine:
    connection = async_mockgun_connection(schema_cache, temporary_cache)
    engine = create_async_engine(
        connection=connection, cache_root=temporary_cache, bind_default=False
    )
    await engine.load_schema()
    return engine
```

Your test owns the returned connection. The engine does not close it.

## Command-line reference

The `sgalchemy` command has two subcommands. Run either one with `--help` for its full usage.
`sgalchemy --version` prints the installed version, and `python -m sgalchemy` runs the same
command.

| Command                                                    | What it does                                                       |
| ---------------------------------------------------------- | ------------------------------------------------------------------ |
| `sgalchemy stubs -o DIR --site URL [--script-name NAME]`   | Generate stubs from the site's schema, reusing a fresh cached copy |
| `sgalchemy stubs -o DIR --from-cache FILE`                 | Generate stubs from a saved schema cache file, without credentials |
| `sgalchemy schema refresh --site URL [--script-name NAME]` | Fetch the schema now and rewrite its cache file                    |

### Stub options

`sgalchemy stubs` accepts these options:

- `-o DIR`, `--out DIR` (required): root directory for the generated stubs.
- `--check`: compare the existing stubs with the schema without writing files.
- `--permissive`: allow unknown attributes in generated classes.
- `--no-entity-aliases`: omit display-name entity aliases.
- `--no-field-aliases`: omit field aliases with the `sg_` prefix removed.

### Credentials and schema sources

With `--script-name`, commands that connect to SG read the API key from `SGALCHEMY_API_KEY`. If it
is unset or empty, an interactive terminal prompts for the key without displaying it. In
automation, inject the variable through your existing secret configuration; without it, unattended
commands fail instead of prompting. Without `--script-name`, the CLI reads a session token from
`SGALCHEMY_SESSION_TOKEN`. An explicit `--script-name` takes precedence over an ambient token. The
CLI does not accept API keys or session tokens as command-line arguments.
`sgalchemy stubs --from-cache` needs no credentials and cannot be combined with `--site` or
`--script-name`.

### Output and exit status

`stubs` first prints a `schema:` line identifying its source and, for a saved cache, its fetch
time. It then prints a `wrote PATH` line for each file and an `aliases skipped:` line. With
`--check`, it prints `stubs are fresh` and exits 0, or `stub drift detected` and exits 1.
`schema refresh` prints `refreshed schema cache: PATH`; that path is a cache file you can pass to
`--from-cache`.

Usage errors print the selected command's usage and an error line. Other anticipated errors, such
as invalid site setup, provider or network failure, and failed stub writes, print one stderr line
starting with `error:`. Unexpected exceptions keep their traceback and exit 1. As described above,
`--check` drift prints to stdout. The exit status tells you what happened:

| Exit | Meaning                                                                                  |
| ---- | ---------------------------------------------------------------------------------------- |
| 0    | Success, including current stubs with `--check`                                          |
| 1    | Stub drift, schema-loading failure, stub/cache write failure, or an unexpected exception |
| 2    | Invalid arguments, missing credentials/provider, or an unreadable cache file             |
| 130  | Interruption, including Ctrl-C at the API key prompt                                     |

Schema-loading errors, including rejected authentication or an unreachable site, begin
`error: could not load the site schema: ...`. Stub-write errors begin
`error: could not write stubs: ...`. Both exit 1; a failed cache write during `schema refresh` also
exits 1.

An invalid HTTPS site URL or empty `--script-name` exits 2. A missing provider extra reports
`error: no SG API provider is installed; ...` and also exits 2.

## Typing details

Engines bind to the default Registry unless you pass `bind_default=False` or your own `registry`.
An engine using a separate Registry does not populate these package-level classes, and its
`engine.entities.Shot` is typed as `type[Entity]` rather than the generated `Shot`. To use a second
site in the same process, pass `registry=Registry()` or `bind_default=False` to that site's engine
factory. One Registry can bind to only one site.

Stubs are strict by default: unknown fields and unknown or invalid field writes are type errors.
`--permissive` allows unknown fields as `Any` while keeping known fields typed. Dynamic APIs such
as `Shot.new(**fields)` still accept schema field names at run time.

By default, stubs also include aliases: entity classes under their display names in CapWords, such
as `Vendor` for a custom entity whose display name is Vendor, and fields without their `sg_`
prefix, such as `status_list` for `sg_status_list`. An alias that would collide with or shadow
another name is skipped and listed on the command's `aliases skipped:` line.

Thumbnail fields need narrowing. Reading `image` or `filmstrip_image` has the type
`str | PathLike[str] | PathLike[bytes] | None`, because a pending write keeps the path object you
assigned, while a value read from SG is a string or `None`. Narrow the value before using
string-only methods. Both fields accept string and `PathLike` paths; only `image` accepts `None`,
which clears both thumbnails.

## Annotation types

| Import from        | Names                                   | Use                                                                         |
| ------------------ | --------------------------------------- | --------------------------------------------------------------------------- |
| `sgalchemy`        | `SessionFactory`, `AsyncSessionFactory` | Session factories                                                           |
| `sgalchemy`        | `ScopedSession`, `AsyncScopedSession`   | Scoped session registries                                                   |
| `sgalchemy`        | `ShotgunLike`, `AsyncShotgunLike`       | Provider wrappers and fakes; structural implementation needs no inheritance |
| `sgalchemy.schema` | `EntitySchema`, `FieldSchema`           | Read-only schema descriptions                                               |
