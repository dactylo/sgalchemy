# Queries and relationships

This page shows how to query entities and load related records with explicit control over network
calls. If you have not loaded a schema and run a first query yet, start with the
[README quick start](../README.md#quick-start).

Choose [entity objects](#select-entities), [rows or summaries](#select-fields-and-summaries), then
[choose a relationship loader](#choose-a-relationship-loader) for the related fields you need.

## Select entities

After `engine.load_schema()` has prepared the reflected classes, build a `select()` statement and
run it through a session:

```python
from sgalchemy import load_only, select

statement = (
    select(Shot)
    .where(Shot.sg_status_list.in_('ip', 'rev'))
    .order_by(Shot.code.asc())
    .limit(50)
    .options(load_only(Shot.code, Shot.sg_status_list))
)
shots = session.scalars(statement)
```

You can also write `Shot.sg_status_list.in_(['ip', 'rev'])`; both forms build the same filter.

`session.scalars()` returns a list of entity objects. It does not send pending writes; call
`commit()` for those. Use `session.execute()` for statements that select individual fields or
summaries instead of entities.

## The loading rule

By default, reading a field or relationship that a query did not load raises an error instead of
calling SG. Queries and explicit loads control provider I/O. Synchronous code can opt into
[`lazyload()`](#lazy-loading), which loads a relationship on its first attribute read.

- **Fields:** a query loads every API-readable field of the selected type unless `load_only()` or
  `defer()` narrows it. Reading a field that was left out raises `UnfetchedFieldError`, whose
  message says how to load it.
- **Relationships:** a relationship field loads as linked entity objects. Each link carries the
  target's type and ID and the fields the provider sends with the link, such as `name`. Reading any
  other field of the target raises `UnfetchedFieldError` until a relationship loader reads the
  target records.

`UnfetchedFieldError` inherits `AttributeError`, because an unloaded field is an expected, harmless
absence. Other field errors, such as `RefreshRequiredError`, deliberately do not; see
[field trust](sessions.md#field-trust).

## Select fields and summaries

Select fields instead of an entity class when you want a table of values rather than objects.
`session.execute()` returns a `Result`, which is a list of `Row` objects. A row is a read-only
mapping keyed by field path, and a plain field name also works as an attribute:

```python
from sgalchemy import or_, select

statement = (
    select(Shot.code, Shot.sg_status_list, Shot.sg_sequence.code)
    .where(or_(Shot.sg_status_list == 'ip', Shot.sg_status_list == 'rev'))
    .order_by(Shot.code.asc())
)
for row in session.execute(statement):
    print(row.code, row.sg_status_list, row['sg_sequence.Sequence.code'])
```

A field reached through a link is keyed by its wire path, `link.TargetType.field`, so read it with
`row[...]`.

Wrap a field in `func` to summarize instead, and add `group_by()` to get one summary per group.
`session.execute()` then returns the provider's summary payload unchanged. See
[Query reference](#query-reference) for the supported summary types:

```python
from sgalchemy import func, select

statement = select(func.count(Shot.id)).group_by(Shot.sg_status_list)
payload = session.execute(statement)
payload['summaries']  # {'id': 3}
payload['groups'][0]  # {'group_name': 'ip', 'group_value': 'ip', 'summaries': {'id': 2}}
```

## Filter, order, and page results

`where()` requires all of its conditions to match. Nest `and_()` and `or_()` for other
combinations, and use `raw()` to pass a filter list in SG syntax unchanged, such as
`raw([['sg_status_list', 'is_not', 'omt']])`. Entity and field statements also accept `limit()` and
`page()`, which selects a one-based page of results; summary statements reject ordering, limits,
and pages.

## Query reference

`func` accepts the following server summary types:

- `count`
- `record_count`
- `sum`
- `avg`
- `min`
- `max`
- `earliest`
- `latest`
- `percentage`
- `status_percentage`
- `status_list`
- `checked`
- `unchecked`

## Choose a relationship loader

Add a loader to the statement's `options()` for each relationship you intend to read:

| Loader                                   | Use it when                                            | Provider calls                                                          |
| ---------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------- |
| `joinedload(Shot.sg_sequence)`           | The link is to-one and has exactly one target type     | None extra: target fields join the same `find()`                        |
| `selectinload(Shot.assets)`              | The link is multi-entity or can point to several types | After the query, one `find()` per target type seen in the results       |
| `lazyload(Shot.assets)`                  | Synchronous code reads the relationship only sometimes | On first access, one `find()` per target type, for the whole result set |
| `await session.load(shots, Shot.assets)` | Async code decides to load after the query             | One `find()` per target type, after reading any missing link fields     |

`joinedload()` and `selectinload()` work in synchronous and async sessions. `lazyload()` is for
synchronous sessions only, and `session.load()` exists only on `AsyncSession`; see
[Async usage](async.md#load-relationships).

### Joined loading

`joinedload()` adds the target's fields to the primary query as deep-field paths. It is not a
server-side join, and it adds no provider call. Each hop must be a to-one link with exactly one
target type; a multi-entity link or a link to several types raises `CompilationError` that points
you to `selectinload()`. To follow a chain of links, extend the option one hop at a time with its
`.joinedload()` method.

### Select-in loading

`selectinload()` collects the linked IDs from the whole result set, then reads them with one
batched `find()` per observed target type, never one query per row. Several `selectinload()`
options on one statement share those reads.

### Lazy loading

`lazyload()` makes the first read of the relationship, on any object from the query, load it for
every object in that originating result. A loop over many shots therefore costs one query per
target type, not one per shot:

```python
from sgalchemy import Session, lazyload, select

with Session(engine, max_implicit_queries=5) as session:
    shots = session.scalars(select(Shot).options(lazyload(Shot.assets)))
    for shot in shots:
        print(shot.code, [asset.code for asset in shot.assets])
```

Because these calls happen inside attribute reads, you can cap them. `max_implicit_queries` on
`Session()` or `sessionmaker()` limits the lazy-load queries one session may run; the default,
`None`, sets no limit. Each lazy load reserves its whole query count before any I/O, and a load
that would exceed the budget raises `SgalchemyError` without calling SG.

### Explicit async loading

`await session.load(objects, relationship)` loads exactly one relationship for one entity or an
iterable of entities, and returns `None`:

```python
from sgalchemy import AsyncEngine, AsyncSession, Entity


async def load_assets(engine: AsyncEngine, Shot: type[Entity]) -> None:
    async with AsyncSession(engine) as session:
        shot = session.merge_dict({'type': 'Shot', 'id': 1})
        await session.load(shot, Shot.assets)
        assets = shot.assets
```

It validates every parent before any I/O and rejects pending parents that have no server ID yet. It
reads the relationship field only for parents that lack it, keeps unsent local changes to that
field, and treats a loaded `None` or `[]` as an empty relationship. It then reads the targets with
one `find()` per observed type. A link value that is not an entity raises `SgalchemyError`.

## Links to types missing from the schema

A link can point to an entity type that your reflected schema does not contain. sgalchemy still
turns a link with a valid type and positive ID into a stable entity object, called an opaque link.
Opaque links support identity, `repr()`, `inspect()`, `as_dict()`, and use as a value in another
relationship write. Fields sent with the link, such as `id` and `name`, remain readable.

Reading any other field of an opaque link raises `SchemaUnavailableError`. Unlike
`UnfetchedFieldError`, it does not inherit `AttributeError`, so `hasattr()` cannot hide it.
Querying, creating, or changing that type directly is unsupported until the schema contains it; a
direct query raises `CompilationError`.

Relationship loaders try once to recover a missing target type:

- `joinedload()` may force one schema refresh and recompile the statement before its single primary
  query.
- `selectinload()`, `lazyload()`, and `session.load()` first check every target type actually
  present in the loaded links. If one is missing, the load may force one schema refresh before any
  follow-up query.

If a target type is still missing after that refresh, `joinedload()` raises `CompilationError`
before its query. The other loaders raise `RelationshipTargetUnavailableError`, naming the
relationship and the missing target types; parent links, their order, and opaque object identity
stay intact. Make those types visible in the reflected schema, then retry the explicit load. A lazy
load that failed this way raises again without another refresh until the engine's schema is
reloaded.
