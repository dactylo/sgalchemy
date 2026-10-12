import ast
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema

from sgalchemy import _stubgen, _stubgen_registry
from sgalchemy._core._registry import Registry
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema import _cache, _model

_NONDEFAULT_PROFILES = [
    pytest.param(False, True, False, id='no-entity-aliases'),
    pytest.param(True, False, False, id='no-field-aliases'),
    pytest.param(True, True, True, id='permissive'),
    pytest.param(False, False, False, id='no-aliases'),
    pytest.param(False, True, True, id='no-entity-aliases-permissive'),
    pytest.param(True, False, True, id='no-field-aliases-permissive'),
    pytest.param(False, False, True, id='no-aliases-permissive'),
]


def _annotation(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def _consumer_cache() -> _cache.SchemaCache:
    fields: dict[str, dict[str, Any]] = {}
    for code, data_type, targets in (
        ('id', 'number', []),
        ('code', 'text', []),
        ('sg_sequence', 'entity', ['Sequence']),
        ('sg_sequences', 'multi_entity', ['Sequence']),
        ('image', 'image', []),
        ('filmstrip_image', 'image', []),
        ('sg_uploaded_movie', 'image', []),
    ):
        fields[code] = {
            'data_type': {'value': data_type},
            'properties': {'valid_types': {'value': targets}},
        }
    return _cache.SchemaCache(
        fields={'Shot': fields, 'Sequence': {'code': fields['code']}},
        entities={'Shot': {'name': {'value': 'Take'}}, 'Sequence': {}},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _run_pyright(config_path: Path) -> subprocess.CompletedProcess[str]:
    # G3: controlled executable/arguments; all consumer and config files are test-owned.
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pyright',
            '--project',
            str(config_path),
            '--pythonpath',
            sys.executable,
            '--outputjson',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _signature(node: ast.FunctionDef) -> tuple[object, ...]:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    positional_defaults = [False] * (len(positional) - len(args.defaults)) + [True] * len(
        args.defaults
    )
    parameters = [
        (arg.arg, _annotation(arg.annotation), has_default)
        for arg, has_default in zip(positional, positional_defaults, strict=True)
    ]
    parameters.extend(
        (arg.arg, _annotation(arg.annotation), default is not None)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    decorators = tuple(
        ast.unparse(decorator)
        for decorator in node.decorator_list
        if ast.unparse(decorator) in {'classmethod', 'staticmethod', 'property'}
    )
    return (
        tuple(parameters),
        None if args.vararg is None else (args.vararg.arg, _annotation(args.vararg.annotation)),
        None if args.kwarg is None else (args.kwarg.arg, _annotation(args.kwarg.annotation)),
        len(args.posonlyargs),
        len(args.kwonlyargs),
        _annotation(node.returns),
        decorators,
    )


def test_check_stubs_detects_crlf_only_drift(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path)
    target = tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi'
    target.write_bytes(target.read_bytes().replace(b'\n', b'\r\n'))

    assert _stubgen.check_stubs(cache, tmp_path) is False


def test_check_stubs_detects_display_name_drift(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path)
    cache.entities['CustomEntity03']['name']['value'] = 'Supplier'  # alias source changed
    assert not _stubgen.check_stubs(cache, tmp_path)


def test_check_stubs_detects_drift(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path)
    assert _stubgen.check_stubs(cache, tmp_path)
    (tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi').unlink()
    assert not _stubgen.check_stubs(cache, tmp_path)
    _stubgen.generate_stubs(cache, tmp_path)
    cache.fields['Shot']['sg_new'] = {
        'data_type': {'value': 'text'},
        'name': {'value': 'New'},
        'properties': {},
    }
    assert not _stubgen.check_stubs(cache, tmp_path)
    assert not _stubgen.check_stubs(cache, tmp_path / 'nowhere')


def test_check_stubs_rejects_and_regeneration_removes_obsolete_file(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path)
    obsolete = tmp_path / 'sgalchemy' / 'entities' / '_obsolete_entity.pyi'
    obsolete.write_text('class ObsoleteEntity: ...\n')
    assert not _stubgen.check_stubs(cache, tmp_path)
    _stubgen.generate_stubs(cache, tmp_path)
    assert not obsolete.exists()
    assert _stubgen.check_stubs(cache, tmp_path)


def test_check_stubs_rejects_invalid_utf8(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path)
    target = tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi'
    target.write_bytes(b'\xff')

    assert _stubgen.check_stubs(cache, tmp_path) is False


def test_check_stubs_rejects_same_hash_body_mutation(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path)
    target = tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi'
    original = target.read_text(encoding='utf-8')
    target.write_text(
        original.replace('class Shot(Entity):', 'class Shot(Entity):\n    stale: int'),
        encoding='utf-8',
    )

    assert _stubgen.check_stubs(cache, tmp_path) is False


def test_flags_control_aliases_and_permissive(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(
        cache, tmp_path, entity_aliases=False, field_aliases=False, permissive=True
    )
    init = (tmp_path / 'sgalchemy' / 'entities' / '__init__.pyi').read_text()
    shot = (tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi').read_text()
    assert 'Vendor' not in init
    assert '\n    status_list:' not in shot
    assert '__getattr__' in shot


def test_generate_emits_descriptor_annotations(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache = fetch_schema(fake_sg)
    report = _stubgen.generate_stubs(cache, tmp_path)
    for stub in (tmp_path / 'sgalchemy' / 'entities').glob('*.pyi'):
        ast.parse(stub.read_text())  # every generated stub is valid Python syntax
    init = (tmp_path / 'sgalchemy' / 'entities' / '__init__.pyi').read_text()
    shot = (tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi').read_text()
    types_stub = (tmp_path / 'sgalchemy' / 'entities' / '_types.pyi').read_text()
    assert "_ReadT_co = TypeVar('_ReadT_co', covariant=True)" in types_stub
    assert "_WriteT_contra = TypeVar('_WriteT_contra', contravariant=True)" in types_stub
    assert 'class _SGField(Generic[_ReadT_co, _WriteT_contra]):' in types_stub
    assert 'def __set__(self, obj: Entity, value: _WriteT_contra) -> None: ...' in types_stub
    assert 'fetched-at:' not in init
    assert f'schema-hash: {report._hash}' in init
    assert 'code: _SGField[str | None, str | None]' in shot
    assert (
        'image: _SGField[str | os.PathLike[str] | os.PathLike[bytes] | None, '
        'str | os.PathLike[str] | os.PathLike[bytes] | None]' in shot
    )
    assert (
        'filmstrip_image: _SGField[str | os.PathLike[str] | os.PathLike[bytes] | None, '
        'str | os.PathLike[str] | os.PathLike[bytes]]' in shot
    )
    assert 'sg_sequence: _SGField[Sequence | None, Sequence | None]' in shot
    assert 'status_list: _SGField[str | None, str | None]' in shot  # field alias annotation
    assert 'import os' in shot
    assert 'sg_qc_summary' not in shot  # UI-only field excluded
    assert 'Vendor = CustomEntity03' in init  # entity alias
    assert '__getattr__' not in shot  # strict by default
    assert [
        line
        for line in shot.splitlines()
        if line.startswith('from sgalchemy.entities.') and '_types import' not in line
    ] == [
        'from sgalchemy.entities._asset import Asset',
        'from sgalchemy.entities._custom_entity03 import CustomEntity03',
        'from sgalchemy.entities._project import Project',
        'from sgalchemy.entities._sequence import Sequence',
    ]


def test_generate_permissive_stub_imports_any(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(cache, tmp_path, permissive=True)

    shot = (tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi').read_text()
    assert 'from typing import Any' in shot.splitlines()


def test_generate_rejects_colliding_entity_paths(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache = fetch_schema(fake_sg)
    valid_out = tmp_path / 'valid'
    report = _stubgen.generate_stubs(cache, valid_out)
    cache.fields['SHOT'] = cache.fields['Shot']
    cache.entities['SHOT'] = {'name': {'value': 'Uppercase shot'}}
    collision_digest = _stubgen._schema_hash(cache)
    for path in report.written:
        text = path.read_text(encoding='utf-8')
        path.write_text(text.replace(report._hash, collision_digest), encoding='utf-8')

    assert not _stubgen.check_stubs(cache, valid_out)

    collision_out = tmp_path / 'collision'
    with pytest.raises(SgalchemyError) as excinfo:
        _stubgen.generate_stubs(cache, collision_out)

    message = str(excinfo.value)
    assert 'Shot' in message
    assert 'SHOT' in message
    assert '_shot.pyi' in message
    assert not collision_out.exists()


def test_generate_skips_unusable_field_aliases(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache = fetch_schema(fake_sg)
    for code in ('sg_class', 'sg_new', 'sg_valid'):
        cache.fields['Shot'][code] = {
            'data_type': {'value': 'text'},
            'name': {'value': code},
            'properties': {},
        }
    registry = Registry()
    registry.bind(_model.build_schema(cache), key=fake_sg.base_url)
    binding = registry.binding_snapshot()
    assert binding is not None
    runtime_aliases = binding.field_aliases['Shot']
    result = _stubgen.generate_stubs(cache, tmp_path)
    for stub in (tmp_path / 'sgalchemy' / 'entities').glob('*.pyi'):
        ast.parse(stub.read_text())
    shot_stub = (tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi').read_text()

    assert runtime_aliases['valid'] == 'sg_valid'
    assert 'class' not in runtime_aliases
    assert 'new' not in runtime_aliases
    assert '\n    class:' not in shot_stub
    assert '\n    new:' not in shot_stub
    assert '\n    valid:' in shot_stub
    assert '\n    sg_class:' in shot_stub
    assert '\n    sg_new:' in shot_stub
    assert any("'class'" in report for report in result.skipped_aliases)
    assert any("'new'" in report for report in result.skipped_aliases)


def test_generate_then_check_writes_exact_utf8_bytes_under_windows_newlines(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = fetch_schema(fake_sg)

    def write_text_with_windows_newlines(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        translated = data.replace('\n', '\r\n') if newline is None else data
        path.write_bytes(translated.encode(encoding or 'utf-8', errors or 'strict'))
        return len(data)

    monkeypatch.setattr(Path, 'write_text', write_text_with_windows_newlines)
    report = _stubgen.generate_stubs(cache, tmp_path)

    assert _stubgen.check_stubs(cache, tmp_path)
    assert all(b'\r\n' not in path.read_bytes() for path in report.written)


def test_registry_companion_mode_switch_preserves_unowned_core_files(tmp_path: Path) -> None:
    cache = _consumer_cache()
    companion = tmp_path / 'sgalchemy' / '_core' / '_registry.pyi'
    for permissive in (False, True, False):
        _stubgen.generate_stubs(cache, tmp_path, permissive=permissive)
        contents = companion.read_text()
        assert ('def __getattr__(cls,' in contents) is permissive
        assert ('def __getattr__(self,' in contents) is permissive
        assert ('def __setattr__(self,' in contents) is permissive
        assert _stubgen.check_stubs(cache, tmp_path, permissive=permissive)
        assert not _stubgen.check_stubs(cache, tmp_path, permissive=not permissive)
        unrelated = companion.parent / '_unrelated.pyi'
        unrelated.write_bytes(b'# Preserve this user-owned file.\r\n')
        obsolete = tmp_path / 'sgalchemy' / 'entities' / '_obsolete.pyi'
        obsolete.write_bytes(b'class Obsolete: ...\n')
        _stubgen.generate_stubs(cache, tmp_path, permissive=permissive)
        assert not obsolete.exists()
        assert unrelated.read_bytes() == b'# Preserve this user-owned file.\r\n'
        assert _stubgen.check_stubs(cache, tmp_path, permissive=permissive)


@pytest.mark.parametrize('permissive', [False, True])
def test_registry_companion_tracks_runtime_signatures(*, permissive: bool) -> None:
    runtime = ast.parse(
        (
            Path(__file__).resolve().parents[1] / 'src' / 'sgalchemy' / '_core' / '_registry.py'
        ).read_text(encoding='utf-8')
    )
    stub = ast.parse(_stubgen_registry.render_registry_stub('', permissive=permissive))
    runtime_classes = {node.name: node for node in runtime.body if isinstance(node, ast.ClassDef)}
    stub_classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
    supported_classes = {name for name in runtime_classes if not name.startswith('_')}
    supported_classes.update({'_EntityMeta', '_LazyResolver'})
    assert stub_classes.keys() == supported_classes

    for name, stub_class in stub_classes.items():
        runtime_class = runtime_classes[name]
        runtime_fields = {
            node.target.id: _annotation(node.annotation)
            for node in runtime_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        stub_fields = {
            node.target.id: _annotation(node.annotation)
            for node in stub_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert stub_fields == runtime_fields, name
        runtime_methods = {
            node.name: _signature(node)
            for node in runtime_class.body
            if isinstance(node, ast.FunctionDef)
        }
        if not permissive:
            if name == '_EntityMeta':
                runtime_methods.pop('__getattr__')
            elif name == 'Entity':
                runtime_methods.pop('__getattr__')
                runtime_methods.pop('__setattr__')
        stub_methods = {
            node.name: _signature(node)
            for node in stub_class.body
            if isinstance(node, ast.FunctionDef)
        }
        assert stub_methods == runtime_methods, name

    runtime_functions = {
        node.name: _signature(node)
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')
    }
    stub_functions = {
        node.name: _signature(node) for node in stub.body if isinstance(node, ast.FunctionDef)
    }
    assert stub_functions == runtime_functions


@pytest.mark.parametrize(
    ('fixture', 'permissive'),
    [
        ('strict_valid', False),
        ('strict_valid', True),
        ('strict_invalid', False),
        ('strict_invalid', True),
        ('permissive', True),
    ],
)
def test_generated_consumers(tmp_path: Path, fixture: str, *, permissive: bool) -> None:
    stub_path = tmp_path / 'typings'
    _stubgen.generate_stubs(_consumer_cache(), stub_path, permissive=permissive)
    source = (Path(__file__).parent / 'data' / 'stubgen' / f'{fixture}.py.txt').read_text()
    if permissive and fixture == 'strict_invalid':
        source = '\n'.join(line for line in source.splitlines() if 'unknown_field' not in line)
    consumer = tmp_path / 'consumer.py'
    consumer.write_text(source)
    config = tmp_path / 'pyrightconfig.json'
    config.write_text(
        json.dumps(
            {
                'include': [consumer.name],
                'extraPaths': [str(Path(__file__).resolve().parents[1] / 'src')],
                'stubPath': str(stub_path),
                'pythonVersion': '3.11',
                'typeCheckingMode': 'strict',
            }
        )
    )
    result = _run_pyright(config)
    assert not result.stderr, result.stderr
    payload = cast('dict[str, Any]', json.loads(result.stdout))
    assert payload['summary']['filesAnalyzed'] == 1, result.stdout
    diagnostics = cast('list[dict[str, Any]]', payload['generalDiagnostics'])
    expected = {
        line: text.split('# error: ')[1].strip()
        for line, text in enumerate(source.splitlines())
        if '# error: ' in text
    }
    if not expected:
        assert result.returncode == 0, result.stdout
        assert diagnostics == [], result.stdout
        return
    assert result.returncode == 1, result.stdout
    actual: dict[int, set[str]] = {}
    for diagnostic in diagnostics:
        assert Path(diagnostic['file']) == consumer, result.stdout
        line = diagnostic['range']['start']['line']
        assert line in expected, result.stdout
        rule = diagnostic['rule']
        assert rule == expected[line] or rule in {
            'reportUnknownMemberType',
            'reportUnknownArgumentType',
            'reportUnknownVariableType',
        }, result.stdout
        actual.setdefault(line, set()).add(rule)
    assert all(rule in actual.get(line, set()) for line, rule in expected.items()), result.stdout


@pytest.mark.parametrize(
    ('entity_aliases', 'field_aliases', 'permissive'),
    _NONDEFAULT_PROFILES,
)
def test_check_stubs_requires_exact_generation_profile(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    *,
    entity_aliases: bool,
    field_aliases: bool,
    permissive: bool,
) -> None:
    cache = fetch_schema(fake_sg)
    _stubgen.generate_stubs(
        cache,
        tmp_path,
        entity_aliases=entity_aliases,
        field_aliases=field_aliases,
        permissive=permissive,
    )

    assert not _stubgen.check_stubs(cache, tmp_path)
    assert _stubgen.check_stubs(
        cache,
        tmp_path,
        entity_aliases=entity_aliases,
        field_aliases=field_aliases,
        permissive=permissive,
    )


@pytest.mark.parametrize('permissive', [False, True])
@pytest.mark.parametrize('mutation', ['missing', 'changed', 'crlf', 'invalid-utf8'])
def test_registry_companion_drift(tmp_path: Path, *, permissive: bool, mutation: str) -> None:
    cache = _consumer_cache()
    report = _stubgen.generate_stubs(cache, tmp_path, permissive=permissive)
    companion = tmp_path / 'sgalchemy' / '_core' / '_registry.pyi'
    assert companion in report.written
    assert set(report.written) == {
        companion,
        tmp_path / 'sgalchemy' / 'entities' / '__init__.pyi',
        tmp_path / 'sgalchemy' / 'entities' / '_types.pyi',
        tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi',
        tmp_path / 'sgalchemy' / 'entities' / '_sequence.pyi',
    }
    content = companion.read_bytes()
    ast.parse(content)
    if mutation == 'missing':
        companion.unlink()
    elif mutation == 'changed':
        companion.write_bytes(content.replace(b'-> Self:', b'-> Entity:'))
    elif mutation == 'crlf':
        companion.write_bytes(content.replace(b'\n', b'\r\n'))
    else:
        companion.write_bytes(b'\xff')
    assert not _stubgen.check_stubs(cache, tmp_path, permissive=permissive)
    _stubgen.generate_stubs(cache, tmp_path, permissive=permissive)
    assert _stubgen.check_stubs(cache, tmp_path, permissive=permissive)
    assert companion.read_bytes() == content


def test_strict_stub_marks_server_owned_fields_unwritable(tmp_path: Path) -> None:
    _stubgen.generate_stubs(_consumer_cache(), tmp_path)
    shot = (tmp_path / 'sgalchemy' / 'entities' / '_shot.pyi').read_text()

    assert 'from typing import Never' in shot
    assert 'id: _SGField[int | None, Never]' in shot
    assert 'sg_uploaded_movie: _SGField[str | None, Never]' in shot


def test_stub_generation_ignores_fetch_time(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache_a = fetch_schema(fake_sg)
    cache_b = replace(cache_a, fetched_at=cache_a.fetched_at + timedelta(days=1))
    first_out = tmp_path / 'first'
    second_out = tmp_path / 'second'

    first = _stubgen.generate_stubs(cache_a, first_out)
    assert _stubgen.check_stubs(cache_b, first_out)
    second = _stubgen.generate_stubs(cache_b, second_out)

    assert {path.relative_to(first_out): path.read_bytes() for path in first.written} == {
        path.relative_to(second_out): path.read_bytes() for path in second.written
    }
