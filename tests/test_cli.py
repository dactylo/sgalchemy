import argparse
import getpass
import importlib.metadata
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, Never, cast

import pytest
from _fakes import _FakeShotgun
from _support import fetch_schema
from cli_helpers import invoke_cli

from sgalchemy import _cli
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema import _cache

VALID_COMMAND_ARGUMENTS = ['stubs', '-o', 'stubs', '--from-cache', 'schema.json']

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
_REFRESH_USAGE = 'usage: sgalchemy schema refresh [-h] --site URL --script-name NAME\n'


def _cache_file(fake_sg: _FakeShotgun, tmp_path: Path) -> Path:
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


def _starts_in_lowercase_style(text: str) -> bool:
    first = text.split(maxsplit=1)[0]
    is_acronym = first.upper() == first and any(character.isalpha() for character in first)
    return text[0].islower() or is_acronym


def _use_site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('SGALCHEMY_API_KEY', 'key')
    monkeypatch.setenv('SGALCHEMY_SCHEMA_CACHE', str(tmp_path / 'cache'))


def test_check_reports_drift_on_stdout(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
    cache_path = _cache_file(fake_sg, tmp_path)
    command = ['stubs', '--check', '-o', str(tmp_path / 'stubs'), '--from-cache', str(cache_path)]

    assert invoke_cli(command) == (1, 'stub drift detected\n', '')


@pytest.mark.parametrize('flags', _NONDEFAULT_FLAG_PROFILES)
def test_check_uses_generation_flag_profile(
    fake_sg: _FakeShotgun,
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
    fake_sg: _FakeShotgun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = _cache_file(fake_sg, tmp_path)
    out = tmp_path / 'connected-stubs'
    loads: list[bool] = []
    creations: list[tuple[str, str, str]] = []

    class FakeEngine:
        site = 'https://connected.example.com'

        def load_schema(self, *, force: bool = False) -> dict[str, Any]:
            loads.append(force)
            return {}

    def fake_create_engine(
        site: str,
        *,
        script_name: str,
        api_key: str,
    ) -> FakeEngine:
        creations.append((site, script_name, api_key))
        return FakeEngine()

    def fake_cache_file(site: str) -> Path:
        assert site == FakeEngine.site
        return cache_path

    monkeypatch.setattr(_cli, 'create_engine', fake_create_engine)
    monkeypatch.setattr(_cli, 'cache_file', fake_cache_file)
    monkeypatch.setenv('SGALCHEMY_API_KEY', 'key')
    command = [
        'stubs',
        '-o',
        str(out),
        '--site',
        'https://connected.example.com',
        '--script-name',
        'script',
    ]

    assert _cli.main(command) == 0
    assert creations == [('https://connected.example.com', 'script', 'key')]
    assert loads == [False]
    assert _cli.main([*command, '--check']) == 0
    assert creations == [
        ('https://connected.example.com', 'script', 'key'),
        ('https://connected.example.com', 'script', 'key'),
    ]
    assert loads == [False, False]


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

    status, stdout, stderr = invoke_cli(['stubs', '-o', str(tmp_path / 'stubs'), *credentials])

    assert (status, stdout) == (2, '')
    assert stderr.startswith('usage: sgalchemy stubs [-h] ')
    assert stderr.endswith(
        'sgalchemy stubs: error: stubs requires --from-cache or both --site and --script-name\n'
    )


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
            'url must be a complete http or https site URL',
            id='incomplete',
        ),
        pytest.param(
            ['--site', 'http://example.com', '--script-name', 'script'],
            'static credentials require an https site URL',
            id='plaintext',
        ),
        pytest.param(
            ['--site', 'https://user@example.com', '--script-name', 'script'],
            'url must not embed userinfo credentials',
            id='userinfo',
        ),
        pytest.param(
            ['--site', 'https://example.com', '--script-name', ''],
            'script_name must not be empty',
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
    _use_site(monkeypatch, tmp_path)
    prefix = (
        ['stubs', '-o', str(tmp_path / 'stubs')] if command == 'stubs' else ['schema', 'refresh']
    )

    assert invoke_cli([*prefix, *credentials]) == (2, '', f'error: {message}\n')


@pytest.mark.parametrize('command', ['stubs', 'refresh'])
def test_missing_api_key_is_reported_by_the_command_parser(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('SGALCHEMY_API_KEY', raising=False)
    status, stdout, stderr = invoke_cli(_connected_command(command, tmp_path))
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


def test_schema_refresh_forces_refetch_and_prints_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    expected = tmp_path / 'refreshed.json'
    calls: list[bool] = []

    class FakeEngine:
        site = 'https://refresh.example.com'

        def load_schema(self, *, force: bool = False) -> dict[str, Any]:
            calls.append(force)
            return {}

    def fake_create_engine(
        site: str,
        *,
        script_name: str,
        api_key: str,
    ) -> FakeEngine:
        assert (site, script_name, api_key) == (
            'https://refresh.example.com',
            'script',
            'key',
        )
        return FakeEngine()

    def fake_cache_file(_site: str) -> Path:
        return expected

    monkeypatch.setattr(_cli, 'create_engine', fake_create_engine)
    monkeypatch.setattr(_cli, 'cache_file', fake_cache_file)
    monkeypatch.setenv('SGALCHEMY_API_KEY', 'key')
    code = _cli.main(
        [
            'schema',
            'refresh',
            '--site',
            'https://refresh.example.com',
            '--script-name',
            'script',
        ]
    )
    assert code == 0
    assert calls == [True]
    assert str(expected) in capsys.readouterr().out


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
    fake_sg: _FakeShotgun,
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


def test_stubs_generate_and_check_roundtrip(fake_sg: _FakeShotgun, tmp_path: Path) -> None:
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
    assert stdout.endswith(
        'With --site, the API key comes from SGALCHEMY_API_KEY or a hidden terminal\nprompt.\n'
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


def test_version_ignores_other_valid_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, 'version', _fixed_version)
    assert invoke_cli(['--version', *VALID_COMMAND_ARGUMENTS]) == (0, 'sgalchemy 9.8.7\n', '')


def test_version_prints_the_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, 'version', _fixed_version)
    assert invoke_cli(['--version']) == (0, 'sgalchemy 9.8.7\n', '')
