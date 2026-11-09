from __future__ import annotations

import asyncio
import copy
import inspect
import threading
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from types import CoroutineType

    from sgalchemy._shotgun import WireRecord, WireRecords

__all__ = [
    'SCHEMA_FIXTURE',
    'FakeAsyncShotgun',
    'FakeShotgun',
    'FakeShotgunFactory',
    'SchemaFixture',
]

_CallTranscript: TypeAlias = tuple[str, tuple[Any, ...], dict[str, Any]]
_P = ParamSpec('_P')
_R = TypeVar('_R')
SchemaFixture: TypeAlias = dict[str, dict[str, Any]]

_ENTITY = {'editable': {'value': True}, 'visible': {'value': True}}
_ENTITY_FIXTURE: dict[str, dict[str, Any]] = {
    name: dict(_ENTITY, name={'value': display})
    for name, display in [
        ('Project', 'Project'),
        ('Sequence', 'Sequence'),
        ('Episode', 'Episode'),
        ('Shot', 'Shot'),
        ('Asset', 'Asset'),
        ('Version', 'Version'),
        ('HumanUser', 'Person'),
        ('Note', 'Note'),
        ('CustomEntity03', 'Vendor'),
    ]
}
SCHEMA_FIXTURE: SchemaFixture = {
    'Project': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'name': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Project Name'},
            'properties': {},
        },
    },
    'Sequence': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Sequence Code'},
            'properties': {},
        },
        'episode': {
            'data_type': {'value': 'entity'},
            'name': {'value': 'Episode'},
            'properties': {'valid_types': {'value': ['Episode']}},
        },
    },
    'Episode': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Episode Code'},
            'properties': {},
        },
    },
    'Shot': {
        'id': {
            'data_type': {'value': 'number'},
            'editable': {'value': False},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Shot Code'},
            'properties': {},
        },
        'description': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Description'},
            'properties': {},
        },
        'sg_status_list': {
            'data_type': {'value': 'status_list'},
            'name': {'value': 'Status'},
            'properties': {},
        },
        'sg_cut_duration': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Cut Duration'},
            'properties': {},
        },
        'sg_due_date': {
            'data_type': {'value': 'date'},
            'name': {'value': 'Due Date'},
            'properties': {},
        },
        'sg_sequence': {
            'data_type': {'value': 'entity'},
            'name': {'value': 'Sequence'},
            'properties': {'valid_types': {'value': ['Sequence']}},
        },
        'sg_vendor': {
            'data_type': {'value': 'entity'},
            'name': {'value': 'Vendor'},
            'properties': {'valid_types': {'value': ['CustomEntity03']}},
        },
        'assets': {
            'data_type': {'value': 'multi_entity'},
            'name': {'value': 'Assets'},
            'properties': {'valid_types': {'value': ['Asset']}},
        },
        'image': {
            'data_type': {'value': 'image'},
            'editable': {'value': False},
            'name': {'value': 'Thumbnail'},
            'properties': {},
        },
        'filmstrip_image': {
            'data_type': {'value': 'image'},
            'editable': {'value': False},
            'name': {'value': 'Filmstrip'},
            'properties': {},
        },
        'project': {
            'data_type': {'value': 'entity'},
            'name': {'value': 'Project'},
            'properties': {'valid_types': {'value': ['Project']}},
        },
        'sg_qc_summary': {
            'data_type': {'value': 'text'},
            'name': {'value': 'QC Summary'},
            'properties': {'query': {'value': {'filters': []}}},
        },
    },
    'Asset': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'image': {
            'data_type': {'value': 'image'},
            'name': {'value': 'Thumbnail'},
            'properties': {},
        },
        'code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Asset Code'},
            'properties': {},
        },
        'sg_code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Legacy Code'},
            'properties': {},
        },
        'shots': {
            'data_type': {'value': 'multi_entity'},
            'name': {'value': 'Shots'},
            'properties': {'valid_types': {'value': ['Shot']}},
        },
    },
    'Version': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Version Code'},
            'properties': {},
        },
        'entity': {
            'data_type': {'value': 'entity'},
            'name': {'value': 'Link'},
            'properties': {'valid_types': {'value': ['Shot', 'Asset']}},
        },
        'user': {
            'data_type': {'value': 'entity'},
            'name': {'value': 'Artist'},
            'properties': {'valid_types': {'value': ['HumanUser']}},
        },
    },
    'HumanUser': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'name': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Name'},
            'properties': {},
        },
    },
    'Note': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'subject': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Subject'},
            'properties': {},
        },
        'note_links': {
            'data_type': {'value': 'multi_entity'},
            'name': {'value': 'Links'},
            'properties': {'valid_types': {'value': ['Shot', 'Version']}},
        },
    },
    'CustomEntity03': {
        'id': {
            'data_type': {'value': 'number'},
            'name': {'value': 'Id'},
            'properties': {},
        },
        'code': {
            'data_type': {'value': 'text'},
            'name': {'value': 'Vendor Code'},
            'properties': {},
        },
    },
}


# Failure deadline for test coordination, not an assertion about provider speed.
_WORKER_TIMEOUT_SECONDS = 10


def _normalize_transcript(
    method: Callable[..., object],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, str, tuple[Any, ...], dict[str, Any]]:
    signature = inspect.signature(method)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    positional_names = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ][1:]
    transcript_names = positional_names[:3]
    transcript_args = tuple(bound.arguments[name] for name in transcript_names)
    supplied_positionals = len(args) - 1
    transcript_kwargs = {
        name: bound.arguments[name] for name in positional_names[3:supplied_positionals]
    }
    transcript_kwargs.update(
        (name, value) for name, value in kwargs.items() if name not in transcript_names
    )
    return args[0], method.__name__, transcript_args, transcript_kwargs


def _record_async_call(
    method: Callable[_P, Coroutine[Any, Any, _R]],
) -> Callable[_P, CoroutineType[Any, Any, _R]]:

    @wraps(method)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        instance, name, transcript_args, transcript_kwargs = _normalize_transcript(
            method, args, kwargs
        )
        await instance.record(name, *transcript_args, **transcript_kwargs)
        return await method(*args, **kwargs)

    return wrapper


class FakeAsyncShotgun:
    base_url = 'https://fake.example.com'

    def __init__(self, find_results: list[WireRecords] | None = None) -> None:
        self.calls: list[_CallTranscript] = []
        self.close_errors: list[Exception] = []
        self.closed = False
        self.errors: dict[str, list[BaseException | None]] = {}
        self.find_results: list[Any] = list(find_results or [])
        self.batch_results: list[list[Any]] = []
        self.fail_on_batch_index: int | None = None
        self.find_error: BaseException | None = None
        self._blocks: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}
        self._next_id = 9000

    def block(self, method: str) -> tuple[asyncio.Event, asyncio.Event]:
        events = (asyncio.Event(), asyncio.Event())
        self._blocks[method] = events
        return events

    @_record_async_call
    async def find(  # noqa: PLR0913, PLR0917
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: list[dict[str, Any]] | None = None,
    ) -> WireRecords:
        del (
            entity_type,
            filters,
            fields,
            order,
            filter_operator,
            limit,
            retired_only,
            page,
            include_archived_projects,
            additional_filter_presets,
        )
        if self.find_error is not None:
            error, self.find_error = self.find_error, None
            raise error
        return self.find_results.pop(0) if self.find_results else []

    def gate(self, method: str, entered: asyncio.Event, release: asyncio.Event) -> None:
        self._blocks[method] = (entered, release)

    async def record(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))
        block = self._blocks.get(name)
        if block is not None:
            entered, release = block
            entered.set()
            await release.wait()
        errors = self.errors.get(name, [])
        if errors and (error := errors.pop(0)) is not None:
            raise error

    async def batch(self, requests: list[WireRecord]) -> list[Any]:
        index = sum(1 for name, _, _ in self.calls if name == 'batch')
        await self.record('batch', requests)
        if self.fail_on_batch_index == index:
            raise RuntimeError('batch failed (FakeAsyncShotgun)')
        if self.batch_results:
            return self.batch_results.pop(0)
        out: list[Any] = []
        for request in requests:
            if request['request_type'] == 'create':
                self._next_id += 1
                out.append(
                    {
                        'type': request['entity_type'],
                        'id': self._next_id,
                        **request['data'],
                    }
                )
            elif request['request_type'] == 'update':
                out.append(
                    {
                        'type': request['entity_type'],
                        'id': request['entity_id'],
                        **request['data'],
                    }
                )
            else:
                out.append(True)
        return out

    async def close(self) -> None:
        await self.record('close')
        if self.close_errors:
            raise self.close_errors.pop(0)
        self.closed = True

    async def find_one(
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
    ) -> WireRecord | None:
        await self.record('find_one', entity_type, filters, fields)
        rows: WireRecords = self.find_results.pop(0) if self.find_results else []
        return rows[0] if rows else None

    async def revive(self, entity_type: str, entity_id: int) -> bool:
        await self.record('revive', entity_type, entity_id)
        return True

    async def schema_entity_read(self) -> SchemaFixture:
        await self.record('schema_entity_read')
        return copy.deepcopy(_ENTITY_FIXTURE)

    async def schema_read(self) -> SchemaFixture:
        await self.record('schema_read')
        return copy.deepcopy(SCHEMA_FIXTURE)

    @_record_async_call
    async def summarize(  # noqa: PLR0913, PLR0917
        self,
        entity_type: str,
        filters: list[Any],
        summary_fields: list[dict[str, str]],
        filter_operator: str | None = None,
        grouping: list[dict[str, str]] | None = None,
        include_archived_projects: bool = True,
    ) -> WireRecord:
        del (
            entity_type,
            filters,
            summary_fields,
            filter_operator,
            grouping,
            include_archived_projects,
        )
        return {'summaries': {}, 'groups': []}

    async def upload_filmstrip_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
    ) -> int:
        await self.record('upload_filmstrip_thumbnail', entity_type, entity_id, path)
        return 1

    async def upload_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
    ) -> int:
        await self.record('upload_thumbnail', entity_type, entity_id, path)
        return 1


def _record_sync_call(method: Callable[_P, _R]) -> Callable[_P, _R]:

    @wraps(method)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        instance, name, transcript_args, transcript_kwargs = _normalize_transcript(
            method, args, kwargs
        )
        instance.record(name, *transcript_args, **transcript_kwargs)
        return method(*args, **kwargs)

    return wrapper


class FakeShotgun:
    base_url = 'https://fake.example.com'

    def __init__(
        self,
        find_results: list[WireRecords] | None = None,
        *,
        worker_barriers: dict[str, threading.Barrier] | None = None,
    ) -> None:
        self.calls: list[_CallTranscript] = []
        self.close_errors: list[Exception] = []
        self.closed = False
        self.errors: dict[str, list[BaseException | None]] = {}
        self.find_results: list[Any] = list(find_results or [])
        self.batch_results: list[list[Any]] = []
        self.fail_on_batch_index: int | None = None
        self.find_error: BaseException | None = None  # raised (once) by the next find()
        self.max_active_calls = 0
        self.raise_on_concurrent = False
        self.thread_ids: set[int] = set()
        self._active_calls = 0
        self._blocks: dict[str, list[tuple[threading.Event, threading.Event]]] = {}
        self._next_id = 9000  # monotonic create ids; never collide across batch calls
        self._state_lock = threading.Lock()
        self._worker_barriers = dict(worker_barriers or {})

    def block_next(
        self,
        method: str,
        *,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        with self._state_lock:
            self._blocks.setdefault(method, []).append((entered, release))

    def close(self) -> None:
        if self.close_errors:
            raise self.close_errors.pop(0)
        self.closed = True

    @_record_sync_call
    def find(  # noqa: PLR0913, PLR0917 - Preserve provider call compatibility.
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
        order: list[dict[str, str]] | None = None,
        filter_operator: str | None = None,
        limit: int = 0,
        retired_only: bool = False,
        page: int = 0,
        include_archived_projects: bool = True,
        additional_filter_presets: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> WireRecords:
        del (
            entity_type,
            filters,
            fields,
            order,
            filter_operator,
            limit,
            retired_only,
            page,
            include_archived_projects,
            additional_filter_presets,
            kwargs,
        )
        if self.find_error is not None:
            err, self.find_error = self.find_error, None
            raise err
        return self.find_results.pop(0) if self.find_results else []

    def record(self, name: str, *args: object, **kwargs: object) -> None:
        with self._state_lock:
            self.calls.append((name, args, kwargs))
            self.thread_ids.add(threading.get_ident())
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
            concurrent = self.raise_on_concurrent and self._active_calls > 1
            blocks = self._blocks.get(name, [])
            block = blocks.pop(0) if blocks else None
            barrier = self._worker_barriers.pop(name, None)
            errors = self.errors.get(name, [])
            error = errors.pop(0) if errors else None
        try:
            if concurrent:
                raise RuntimeError('concurrent fake Shotgun call')
            if block is not None:
                entered, release = block
                entered.set()
                if not release.wait(timeout=_WORKER_TIMEOUT_SECONDS):
                    raise TimeoutError(f'timed out waiting to release fake {name}')
            if barrier is not None:
                barrier.wait(timeout=_WORKER_TIMEOUT_SECONDS)
            if error is not None:
                raise error
        finally:
            with self._state_lock:
                self._active_calls -= 1

    def batch(self, requests: list[WireRecord]) -> list[Any]:
        index = sum(1 for name, _, _ in self.calls if name == 'batch')
        self.record('batch', requests)
        if self.fail_on_batch_index == index:
            raise RuntimeError('batch failed (FakeShotgun)')
        if self.batch_results:
            return self.batch_results.pop(0)
        out: list[Any] = []
        for req in requests:
            if req['request_type'] == 'create':
                self._next_id += 1
                out.append({'type': req['entity_type'], 'id': self._next_id, **req['data']})
            elif req['request_type'] == 'update':
                out.append({'type': req['entity_type'], 'id': req['entity_id'], **req['data']})
            else:
                out.append(True)
        return out

    def find_one(
        self,
        entity_type: str,
        filters: list[Any],
        fields: list[str] | None = None,
    ) -> WireRecord | None:
        self.record('find_one', entity_type, filters, fields)
        rows: WireRecords = self.find_results.pop(0) if self.find_results else []
        return rows[0] if rows else None

    def revive(self, entity_type: str, entity_id: int) -> bool:
        self.record('revive', entity_type, entity_id)
        return True

    def schema_entity_read(self) -> SchemaFixture:
        self.record('schema_entity_read')
        return copy.deepcopy(_ENTITY_FIXTURE)

    def schema_read(self) -> SchemaFixture:
        self.record('schema_read')
        return copy.deepcopy(SCHEMA_FIXTURE)

    @_record_sync_call
    def summarize(  # noqa: PLR0913, PLR0917 - Preserve provider call compatibility.
        self,
        entity_type: str,
        filters: list[Any],
        summary_fields: list[dict[str, str]],
        filter_operator: str | None = None,
        grouping: list[dict[str, str]] | None = None,
        include_archived_projects: bool = True,
        **kwargs: object,
    ) -> WireRecord:
        del (
            entity_type,
            filters,
            summary_fields,
            filter_operator,
            grouping,
            include_archived_projects,
            kwargs,
        )
        return {'summaries': {}, 'groups': []}

    def upload_filmstrip_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
    ) -> int:
        self.record('upload_filmstrip_thumbnail', entity_type, entity_id, path)
        return 1

    def upload_thumbnail(
        self,
        entity_type: str,
        entity_id: int,
        path: str,
    ) -> int:
        self.record('upload_thumbnail', entity_type, entity_id, path)
        return 1


class FakeShotgunFactory:
    def __init__(self) -> None:
        self.instances: list[FakeShotgun] = []
        self._worker_barriers: dict[str, threading.Barrier] = {}

    def __call__(self) -> FakeShotgun:
        instance = FakeShotgun(worker_barriers=self._worker_barriers)
        self.instances.append(instance)
        return instance

    def block_workers(self, method: str, parties: int) -> threading.Barrier:
        barrier = threading.Barrier(parties)
        self._worker_barriers[method] = barrier
        return barrier
