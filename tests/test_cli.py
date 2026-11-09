import argparse
import getpass
import importlib.metadata
import sys
import warnings
from collections.abc import Iterator
from datetime import UTC
from pathlib import Path
from types import ModuleType
from typing import Never, cast

import pytest

from sgalchemy import Engine, _cli, create_engine
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema import _cache
from sgalchemy.schema._cachedir import cache_file
from tests._fakes import FakeShotgun
from tests._support import fetch_schema
from tests.cli_helpers import invoke_cli

_VALID_COMMAND_ARGUMENTS = ['stubs', '-o', 'stubs', '--from-cache', 'schema.json']

_MISSING_PROVIDER = (
    'error: no SG API provider is installed; '
    'install sgalchemy with the official, api, or async extra\n'
)

_NONDEFAULT_FLAG_PROFILES = [
    pytest.param(['--no-entity-aliases'], id='no-entity-aliases'),
    pytest.param(['--no-field-aliases'], id='no-field-aliases'),
    pytest.param(['--permissive'], id='permissive'),
    pytest.param(['--no-entity-aliases', '--no-field-aliases'], id='no-aliases'),
    pytest.param(
        ['--no-entity-aliases', '--permissive'],
        id='no-entity-aliases-permissive',
    ),
    pytest.param(
        ['--no-field-aliases', '--permissive'],
        id='no-field-aliases-permissive',
    ),
    pytest.param(
        ['--no-entity-aliases', '--no-field-aliases', '--permissive'],
        id='no-aliases-permissive',
    ),
]
_REFRESH = ['schema', 'refresh', '--site', 'https://example.com', '--script-name', 'script']
_REFRESH_USAGE = 'usage: sgalchemy schema refresh [-h] --site URL [--script-name NAME]\n'


def _cache_file(fake_sg: FakeShotgun, tmp_path: Path) -> Path:
    cache = fetch_schema(fake_sg)
    path = tmp_path / 'schema.json'
    _cache.store_cache(path, cache)
    return path


def _connected_command(command: str, tmp_path: Path) -> list[str]:
    if command == 'refresh':
        return list(_REFRESH)
    return [
        'stubs',
        '-o',
        str(tmp_path / 'stubs'),
        '--site',
        'https://example.com',
        '--script-name',
        'script',
    ]


def _fake_provider(schema: object) -> ModuleType:
    class _Shotgun:
        def __init__(self, base_url: str, **_credentials: str) -> None:
            self.base_url = base_url

        def close(self) -> None:
            pass

        def schema_entity_read(self) -> dict[str, object]:
            return {}

        def schema_read(self) -> object:
            if isinstance(schema, Exception):
                raise schema
            return schema

    provider = ModuleType('shotgun_api3')
    provider.__dict__['Shotgun'] = _Shotgun
    return provider


def _fixed_version(_name: str) -> str:
    return '9.8.7'


def _help_entries(
    parser: argparse.ArgumentParser, prefix: str
) -> Iterator[tuple[str, str | None]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = cast('argparse._SubParsersAction[argparse.ArgumentParser]', action)
            visible = [choice.dest for choice in subparsers._choices_actions]
            for choice in subparsers._choices_actions:
                yield f'{prefix} {choice.dest}', choice.help
            for name in visible:
                yield from _help_entries(subparsers.choices[name], f'{prefix} {name}')
        elif action.help is not argparse.SUPPRESS:
            label = '/'.join(action.option_strings) or action.dest
            yield f'{prefix} {label}', action.help


def _named_error(name: str, base: type[Exception] = Exception) -> type[Exception]:
    return type(name, (base,), {})


def _record_engine(
    monkeypatch: pytest.MonkeyPatch, fake_sg: FakeShotgun
) -> list[tuple[str, dict[str, str]]]:
    creations: list[tuple[str, dict[str, str]]] = []

    def fake(site: str, **credentials: str) -> Engine:
        creations.append((site, credentials))
        return create_engine(connection=fake_sg, bind_default=False)

    monkeypatch.setattr(_cli, 'create_engine', fake)
    return creations


def _starts_in_lowercase_style(text: str) -> bool:
    first = text.split(maxsplit=1)[0]
    is_acronym = first.upper() == first and any(character.isalpha() for character in first)
    return text[0].islower() or is_acronym


def _use_site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('SGALCHEMY_API_KEY', 'key')
    monkeypatch.setenv('SGALCHEMY_SCHEMA_CACHE', str(tmp_path / 'cache'))


def test_check_reports_drift_on_stdout(fake_sg: FakeShotgun, tmp_path: Path) -> None:
    cache_path = _cache_file(fake_sg, tmp_path)
    command = ['stubs', '--check', '-o', str(tmp_path / 'stubs'), '--from-cache', str(cache_path)]

    cache = _cache.load_cache(cache_path)
    assert cache is not None
    fetched = cache.fetched_at.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')
    assert invoke_cli(command) == (
        1,
        f'schema: {cache_path} (fetched {fetched})\nstub drift detected\n',
        '',
    )


@pytest.mark.parametrize('flags', _NONDEFAULT_FLAG_PROFILES)
def test_check_uses_generation_flag_profile(
    fake_sg: FakeShotgun,
    tmp_path: Path,
    flags: list[str],
) -> None:
    cache_path = _cache_file(fake_sg, tmp_path)
    out = tmp_path / 'stubs'
    command = ['stubs', '-o', str(out), '--from-cache', str(cache_path)]

    assert _cli.main([*command, *flags]) == 0
    assert _cli.main([*command, '--check']) == 1
    assert _cli.main([*command, '--check', *flags]) == 0


def test_connected_stubs_generate_and_check_with_one_schema_load_each(
    fake_sg: FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_site(monkeypatch, tmp_path)
    creations = _record_engine(monkeypatch, fake_sg)

    def no_cache_read(_path: Path) -> Never:
        raise AssertionError('connected stubs must use the loaded schema')

    monkeypatch.setattr(_cli, 'load_cache', no_cache_read)
    command = _connected_command('stubs', tmp_path)
    assert invoke_cli(command)[0] == 0
    assert invoke_cli([*command, '--check'])[0] == 0
    assert creations == [
        ('https://example.com', {'script_name': 'script', 'api_key': 'key'}),
        ('https://example.com', {'script_name': 'script', 'api_key': 'key'}),
    ]
    assert sum(name == 'schema_read' for name, _, _ in fake_sg.calls) == 1


def test_connected_stubs_use_the_loaded_schema_when_the_cache_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_sg: FakeShotgun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_site(monkeypatch, tmp_path)
    (tmp_path / 'cache').write_text('blocked', encoding='utf-8')
    _record_engine(monkeypatch, fake_sg)
    status, stdout, stderr = invoke_cli(_connected_command('stubs', tmp_path))
    assert status == 0
    assert stdout.startswith('schema: fetched from https://fake.example.com\nwrote ')
    assert (tmp_path / 'stubs' / 'sgalchemy' / 'entities' / '__init__.pyi').exists()
    assert stderr.startswith(
        f'warning: could not write the schema cache {cache_file("https://fake.example.com")}: '
    )
    assert stderr.endswith(
        '; using the fetched schema without caching it. '
        'Set SGALCHEMY_SCHEMA_CACHE to use a writable directory\n'
    )
    assert len(stderr.splitlines()) == 1
    assert not caplog.records


def test_every_visible_option_and_command_has_lowercase_help() -> None:
    problems = [
        f'{name}: {text!r}'
        for name, text in _help_entries(_cli._build_parser(), 'sgalchemy')
        if not text or not _starts_in_lowercase_style(text) or text.endswith('.')
    ]
    assert problems == []


def test_help_goes_to_stdout() -> None:
    status, stdout, stderr = invoke_cli(['--help'])
    assert (status, stderr) == (0, '')
    assert stdout.startswith('usage: sgalchemy [-h] [--version] COMMAND ...\n')


def test_help_lists_commands_after_options() -> None:
    status, stdout, stderr = invoke_cli(['--help'])
    assert (status, stderr) == (0, '')
    assert [line for line in stdout.splitlines() if line.endswith(':')] == [
        'options:',
        'commands:',
    ]
    commands = stdout.split('\ncommands:\n', 1)[1]
    first_words = {line.split()[0] for line in commands.splitlines() if line.split()}
    assert {'stubs', 'schema'} <= first_words


@pytest.mark.parametrize(
    'credentials',
    [
        pytest.param(
            ['--script-name', 'script'],
            id='site',
        ),
        pytest.param(
            ['--site', 'https://connected.example.com'],
            id='script-name',
        ),
    ],
)
def test_incomplete_stubs_credentials_are_rejected(
    credentials: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create_engine(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError

    monkeypatch.setattr(_cli, 'create_engine', fail_create_engine)
    monkeypatch.delenv('SGALCHEMY_SESSION_TOKEN', raising=False)

    status, stdout, stderr = invoke_cli(['stubs', '-o', str(tmp_path / 'stubs'), *credentials])

    assert (status, stdout) == (2, '')
    assert stderr.startswith('usage: sgalchemy stubs [-h] ')
    message = (
        'stubs requires --from-cache FILE or --site URL'
        if credentials[0] == '--script-name'
        else 'authenticate with --script-name NAME and an API key, or set SGALCHEMY_SESSION_TOKEN'
    )
    assert stderr.endswith(f'sgalchemy stubs: error: {message}\n')


def test_interrupted_api_key_prompt_exits_quietly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(getpass, 'getpass', interrupt)

    assert _cli.main(_REFRESH) == 130
    assert capsys.readouterr() == ('', '')


@pytest.mark.parametrize(
    ('credentials', 'message'),
    [
        pytest.param(
            ['--site', 'not-a-url', '--script-name', 'script'],
            '--site must be a complete https URL',
            id='incomplete',
        ),
        pytest.param(
            ['--site', 'http://example.com', '--script-name', 'script'],
            '--site must be a complete https URL',
            id='plaintext',
        ),
        pytest.param(
            ['--site', 'https://user@example.com', '--script-name', 'script'],
            '--site must be a complete https URL',
            id='userinfo',
        ),
        pytest.param(
            ['--site', 'https://example.com', '--script-name', ''],
            '--script-name must not be empty',
            id='empty-script-name',
        ),
    ],
)
@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_invalid_site_or_script_name_is_a_setup_error(
    command: str,
    credentials: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    prefix = (
        ['stubs', '-o', str(tmp_path / 'stubs')] if command == 'stubs' else ['schema', 'refresh']
    )

    assert invoke_cli([*prefix, *credentials], stdin=None) == (2, '', f'error: {message}\n')


@pytest.mark.parametrize('stdin', ['', None], ids=['not-a-terminal', 'closed'])
@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_missing_api_key_is_reported_by_the_command_parser(
    command: str,
    stdin: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    status, stdout, stderr = invoke_cli(_connected_command(command, tmp_path), stdin=stdin)
    prog = 'sgalchemy stubs' if command == 'stubs' else 'sgalchemy schema refresh'
    assert (status, stdout) == (2, '')
    assert stderr.startswith(f'usage: {prog} [-h] ')
    assert stderr.endswith(
        f'{prog}: error: set SGALCHEMY_API_KEY or run interactively to enter an API key\n'
    )


def test_missing_command_is_a_usage_error() -> None:
    status, stdout, stderr = invoke_cli([])
    assert (status, stdout) == (2, '')
    assert stderr.endswith('sgalchemy: error: the following arguments are required: COMMAND\n')


def test_missing_metadata_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, 'version', missing)
    status, stdout, stderr = invoke_cli(['--version'])
    assert (status, stdout) == (2, '')
    assert stderr == 'error: installed distribution metadata for sgalchemy is unavailable\n'


@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_missing_provider_extra_is_a_setup_error(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _use_site(monkeypatch, tmp_path)
    monkeypatch.setitem(cast('dict[str, ModuleType | None]', sys.modules), 'shotgun_api3', None)

    assert invoke_cli(_connected_command(command, tmp_path)) == (2, '', _MISSING_PROVIDER)


def test_missing_schema_command_is_a_usage_error() -> None:
    status, stdout, stderr = invoke_cli(['schema'])
    assert (status, stdout) == (2, '')
    assert stderr == (
        'usage: sgalchemy schema [-h] COMMAND ...\n'
        'sgalchemy schema: error: the following arguments are required: COMMAND\n'
    )


def test_normal_runs_do_not_read_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(name: str) -> str:
        raise AssertionError(name)

    monkeypatch.setattr(importlib.metadata, 'version', forbidden)
    assert invoke_cli(['--help'])[0] == 0


@pytest.mark.parametrize(
    ('outcome', 'detail'),
    [
        pytest.param(
            _named_error(
                'AuthenticationFault', _named_error('Fault', _named_error('ShotgunError'))
            )("Can't authenticate script 'script'"),
            "Can't authenticate script 'script'",
            id='authentication',
        ),
        pytest.param(
            _named_error('ProtocolError')('<ProtocolError for example.com/api3/json: 503 busy>'),
            '<ProtocolError for example.com/api3/json: 503 busy>',
            id='http-status',
        ),
        pytest.param(
            _named_error('ServerNotFoundError', _named_error('HttpLib2Error'))(
                'Unable to find the server at example.com'
            ),
            'Unable to find the server at example.com',
            id='unknown-host',
        ),
        pytest.param(
            ConnectionRefusedError(61, 'Connection refused'),
            '[Errno 61] Connection refused',
            id='connection-refused',
        ),
        pytest.param(TimeoutError(), 'TimeoutError', id='timeout-without-message'),
        pytest.param(
            _named_error('TransportError', _named_error('ShotgunError'))('first\nsecond'),
            'first\\x0asecond',
            id='multi-line-message',
        ),
        pytest.param(
            {'Shot': 'malformed'},
            'provider returned malformed schema; cache was not updated',
            id='malformed-schema',
        ),
    ],
)
@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_provider_and_network_failures_exit_one(
    command: str,
    outcome: object,
    detail: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _use_site(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, 'shotgun_api3', _fake_provider(outcome))

    assert invoke_cli(_connected_command(command, tmp_path)) == (
        1,
        '',
        f'error: could not load the site schema: {detail}\n',
    )


def test_refresh_help_lists_its_options() -> None:
    status, stdout, stderr = invoke_cli(['schema', 'refresh', '--help'])
    assert (status, stderr) == (0, '')
    assert ' '.join(stdout.split('\n\n', 1)[0].split()) == _REFRESH_USAGE.strip()
    assert 'SGALCHEMY_API_KEY' in stdout


def test_schema_help_lists_its_commands() -> None:
    status, stdout, stderr = invoke_cli(['schema', '--help'])
    assert (status, stderr) == (0, '')
    assert ' '.join(stdout.split('\n\n', 1)[0].split()) == (
        'usage: sgalchemy schema [-h] COMMAND ...'
    )
    assert [line for line in stdout.splitlines() if line.endswith(':')] == [
        'options:',
        'commands:',
    ]
    commands = stdout.split('\ncommands:\n', 1)[1]
    assert 'refresh' in {line.split()[0] for line in commands.splitlines() if line.split()}


def test_schema_refresh_fails_when_the_cache_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_sg: FakeShotgun,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use_site(monkeypatch, tmp_path)
    (tmp_path / 'cache').write_text('blocked', encoding='utf-8')
    _record_engine(monkeypatch, fake_sg)
    status, stdout, stderr = invoke_cli(_REFRESH)
    assert (status, stdout) == (1, '')
    assert stderr.startswith(
        f'error: could not write the schema cache {cache_file("https://fake.example.com")}: '
    )
    assert stderr.endswith('; set SGALCHEMY_SCHEMA_CACHE to a writable directory\n')
    assert len(stderr.splitlines()) == 1
    assert not caplog.records


def test_schema_refresh_forces_refetch_and_prints_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    fake_sg: FakeShotgun,
    tmp_path: Path,
) -> None:
    _use_site(monkeypatch, tmp_path)
    _record_engine(monkeypatch, fake_sg)
    expected = cache_file('https://fake.example.com')
    _cache.store_cache(expected, fetch_schema(fake_sg))
    fake_sg.calls.clear()
    assert invoke_cli(_REFRESH) == (0, f'refreshed schema cache: {expected}\n', '')
    assert sum(name == 'schema_read' for name, _, _ in fake_sg.calls) == 1


@pytest.mark.parametrize(
    ('error', 'detail'),
    [
        pytest.param(
            PermissionError(13, 'Permission denied', 'stubs'),
            "[Errno 13] Permission denied: 'stubs'",
            id='unwritable',
        ),
        pytest.param(
            SgalchemyError("entity types 'Ab' and 'AB' map to the same generated stub path: ab"),
            "entity types 'Ab' and 'AB' map to the same generated stub path: ab",
            id='path-collision',
        ),
    ],
)
def test_stub_write_failure_exits_one(
    error: Exception,
    detail: str,
    fake_sg: FakeShotgun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> Never:
        raise error

    cache_path = _cache_file(fake_sg, tmp_path)
    monkeypatch.setattr(_cli._stubgen, 'generate_stubs', fail)

    assert invoke_cli(
        ['stubs', '-o', str(tmp_path / 'stubs'), '--from-cache', str(cache_path)]
    ) == (
        1,
        '',
        f'error: could not write stubs: {detail}\n',
    )


@pytest.mark.parametrize('cache_state', ['missing', 'corrupt'])
def test_stubs_from_cache_reports_invalid_cache(
    tmp_path: Path,
    cache_state: str,
) -> None:
    cache_path = tmp_path / 'invalid-schema.json'
    if cache_state == 'corrupt':
        cache_path.write_text('{not-json', encoding='utf-8')
    out = tmp_path / 'stubs'

    status, stdout, stderr = invoke_cli(['stubs', '--from-cache', str(cache_path), '-o', str(out)])

    assert (status, stdout) == (2, '')
    assert stderr.startswith('usage: sgalchemy stubs [-h] ')
    assert stderr.endswith(f'sgalchemy stubs: error: could not load schema cache: {cache_path}\n')
    assert not out.exists()


@pytest.mark.parametrize('command', ['stubs', 'refresh'])
@pytest.mark.parametrize('mode', ['environment', 'prompt', 'missing', 'empty', 'unsafe'])
def test_connected_commands_read_secret_safely(
    command: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    credential = 'test-credential-value'
    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: mode != 'missing')
    prompts: list[str] = []

    def prompt(label: str) -> str:
        prompts.append(label)
        if mode == 'environment':
            raise AssertionError('environment credentials must not prompt')
        if mode == 'unsafe':
            warnings.warn('Cannot hide input', getpass.GetPassWarning, stacklevel=2)
            raise AssertionError('must not fall back to visible input')
        return '' if mode == 'empty' else credential

    def create(site: str, *, script_name: str, api_key: str) -> Never:
        assert mode in {'environment', 'prompt'}
        assert (site, script_name, api_key) == ('https://example.com', 'script', credential)
        raise RuntimeError('engine boundary reached')

    monkeypatch.setattr(getpass, 'getpass', prompt)
    monkeypatch.setattr(_cli, 'create_engine', create)
    if mode == 'environment':
        monkeypatch.setenv('SGALCHEMY_API_KEY', credential)
    args = ['stubs', '-o', str(tmp_path)] if command == 'stubs' else ['schema', 'refresh']
    args += ['--site', 'https://example.com', '--script-name', 'script']
    if mode in {'environment', 'prompt'}:
        with pytest.raises(RuntimeError, match='engine boundary reached'):
            _cli.main(args)
    else:
        with pytest.raises(SystemExit) as caught:
            _cli.main(args)
        assert caught.value.code == 2
    output = capsys.readouterr()
    assert credential not in output.out + output.err
    if mode in {'missing', 'empty', 'unsafe'}:
        assert 'SGALCHEMY_API_KEY' in output.err
    assert bool(prompts) == (mode in {'prompt', 'empty', 'unsafe'})


def test_stubs_generate_and_check_roundtrip(fake_sg: FakeShotgun, tmp_path: Path) -> None:
    cache_path = _cache_file(fake_sg, tmp_path)
    out = tmp_path / 'stubs'
    code = _cli.main(['stubs', '-o', str(out), '--from-cache', str(cache_path)])
    assert code == 0
    assert (out / 'sgalchemy' / 'entities' / '__init__.pyi').exists()
    assert _cli.main(['stubs', '--check', '-o', str(out), '--from-cache', str(cache_path)]) == 0


def test_stubs_help_groups_options_by_role() -> None:
    status, stdout, stderr = invoke_cli(['stubs', '--help'])
    headings = [line for line in stdout.splitlines() if line.endswith(':') and line[0] != ' ']
    assert (status, stderr) == (0, '')
    assert stdout.startswith('usage: sgalchemy stubs [-h] -o DIR [--check] [--from-cache FILE]')
    assert headings == ['options:', 'output:', 'schema source:', 'stub content:']
    assert ' '.join(stdout.split()).endswith(
        'With --script-name, the API key comes from SGALCHEMY_API_KEY or a hidden '
        'terminal prompt. '
        'Without it, the session token comes from SGALCHEMY_SESSION_TOKEN.'
    )


def test_stubs_without_a_source_is_a_usage_error() -> None:
    status, stdout, stderr = invoke_cli(['stubs', '-o', 'stubs'])
    assert (status, stdout) == (2, '')
    assert stderr.endswith(
        'sgalchemy stubs: error: stubs requires --from-cache FILE or --site URL\n'
    )


def test_unavailable_stdout_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    # Restore stdout before capsys finalizes, so its capture stream is not reinstated.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, 'stdout', None)
        status = _cli.main(['--help'])

    assert (status, capsys.readouterr().err) == (2, 'error: standard output is unavailable\n')


@pytest.mark.parametrize(
    'error',
    [
        pytest.param(RuntimeError('unexpected'), id='runtime'),
        pytest.param(
            ModuleNotFoundError("No module named 'yaml'", name='yaml'), id='other-module'
        ),
    ],
)
def test_unexpected_schema_load_failures_propagate(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _use_site(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, 'shotgun_api3', _fake_provider(error))

    with pytest.raises(type(error)):
        invoke_cli(_REFRESH)


@pytest.mark.parametrize(
    ('entry', 'message'),
    [
        pytest.param(
            '',
            'the entered API key is empty; set SGALCHEMY_API_KEY or enter a key',
            id='empty',
        ),
        pytest.param(
            EOFError(),
            'could not read a hidden API key; set SGALCHEMY_API_KEY',
            id='end-of-input',
        ),
    ],
)
def test_unusable_api_key_entry_is_reported_by_the_refresh_parser(
    entry: str | Exception,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def prompt(_label: str) -> str:
        if isinstance(entry, Exception):
            raise entry
        return entry

    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(getpass, 'getpass', prompt)

    with pytest.raises(SystemExit) as caught:
        _cli.main(_REFRESH)

    assert caught.value.code == 2
    assert capsys.readouterr().err == (
        f'{_REFRESH_USAGE}sgalchemy schema refresh: error: {message}\n'
    )


@pytest.mark.parametrize('options', [['--site', 'https://example.com'], ['--script-name', 's']])
def test_from_cache_cannot_be_combined_with_site_options(options: list[str]) -> None:
    status, stdout, stderr = invoke_cli(
        ['stubs', '-o', 'stubs', '--from-cache', 'schema.json', *options]
    )
    assert (status, stdout) == (2, '')
    assert stderr.endswith(
        'sgalchemy stubs: error: choose one schema source: --from-cache FILE, '
        'or --site URL with credentials\n'
    )


@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_session_token_authenticates_without_script_name(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_sg: FakeShotgun
) -> None:
    _use_site(monkeypatch, tmp_path)
    monkeypatch.delenv('SGALCHEMY_API_KEY')
    monkeypatch.setenv('SGALCHEMY_SESSION_TOKEN', 'tok')
    creations = _record_engine(monkeypatch, fake_sg)
    status, stdout, stderr = invoke_cli(_connected_command(command, tmp_path)[:-2])
    assert (status, stderr) == (0, '')
    assert stdout
    assert creations == [('https://example.com', {'session_token': 'tok'})]


@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_explicit_script_name_wins_over_an_ambient_token(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_sg: FakeShotgun
) -> None:
    _use_site(monkeypatch, tmp_path)
    monkeypatch.setenv('SGALCHEMY_SESSION_TOKEN', 'tok')
    creations = _record_engine(monkeypatch, fake_sg)
    assert invoke_cli(_connected_command(command, tmp_path))[0] == 0
    assert creations == [('https://example.com', {'script_name': 'script', 'api_key': 'key'})]


@pytest.mark.parametrize('token', [None, ''])
@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_missing_credentials_is_a_usage_error(
    command: str, token: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    monkeypatch.delenv('SGALCHEMY_SESSION_TOKEN', raising=False)
    if token is not None:
        monkeypatch.setenv('SGALCHEMY_SESSION_TOKEN', token)
    status, stdout, stderr = invoke_cli(_connected_command(command, tmp_path)[:-2], stdin=None)
    assert (status, stdout) == (2, '')
    assert stderr.endswith(
        'error: authenticate with --script-name NAME and an API key, '
        'or set SGALCHEMY_SESSION_TOKEN\n'
    )


@pytest.mark.parametrize('source', ['file', 'fetched', 'reused'])
def test_stubs_reports_the_schema_source(
    source: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_sg: FakeShotgun
) -> None:
    _use_site(monkeypatch, tmp_path)
    _record_engine(monkeypatch, fake_sg)
    cache = fetch_schema(fake_sg)
    path = cache_file('https://fake.example.com')
    fetched = cache.fetched_at.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')
    if source != 'fetched':
        _cache.store_cache(path, cache)
    command = _connected_command('stubs', tmp_path)
    if source == 'file':
        command = ['stubs', '-o', str(tmp_path / 'stubs'), '--from-cache', str(path)]
        line = f'schema: {path} (fetched {fetched})\n'
    elif source == 'fetched':
        line = 'schema: fetched from https://fake.example.com\n'
    else:
        line = (
            f'schema: cached copy fetched {fetched}; '
            "run 'sgalchemy schema refresh' to fetch the current schema\n"
        )
    status, stdout, stderr = invoke_cli(command)
    assert (status, stderr) == (0, '')
    assert stdout.startswith(line + 'wrote ')
    assert stdout.endswith(
        "aliases skipped: Asset: field alias 'code' for sg_code collides; skipped\n"
    )
    if source == 'fetched':
        stored = _cache.load_cache(path)
        assert stored is not None
        fetched = stored.fetched_at.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')
        line = (
            f'schema: cached copy fetched {fetched}; '
            "run 'sgalchemy schema refresh' to fetch the current schema\n"
        )
    assert invoke_cli([*command, '--check']) == (0, line + 'stubs are fresh\n', '')


def test_version_ignores_other_valid_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, 'version', _fixed_version)
    assert invoke_cli(['--version', *_VALID_COMMAND_ARGUMENTS]) == (0, 'sgalchemy 9.8.7\n', '')


def test_version_prints_the_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, 'version', _fixed_version)
    assert invoke_cli(['--version']) == (0, 'sgalchemy 9.8.7\n', '')
