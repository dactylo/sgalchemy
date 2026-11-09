"""Command-line interface for schema caches and generated stubs."""

import argparse
import getpass
import os
import re
import sys
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from sgalchemy import _stubgen
from sgalchemy._core._construction import require_static_credential_https, validate_site
from sgalchemy._core._engine import Engine, SchemaLoad, create_engine, load_schema_with_source
from sgalchemy._downstream import write_stdout
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema._cache import SchemaCache, load_cache

_COMMAND_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        'commands',
        (
            ('stubs', 'generate or check entity type stubs'),
            ('schema', 'manage the cached schema'),
        ),
    ),
)
_COMMAND_HELP = {name: text for _title, commands in _COMMAND_GROUPS for name, text in commands}
_CONTROL_CHARACTERS = re.compile(r'[\x00-\x1f\x7f-\x9f]')

_MISSING_PROVIDER = (
    'no SG API provider is installed; install sgalchemy with the official, api, or async extra'
)
_PROVIDER_FAILURE_NAMES = frozenset(
    {'HttpLib2Error', 'ProtocolError', 'ResponseError', 'ShotgunError'}
)
_PROVIDER_MODULE = 'shotgun_api3'


class _CommandError(Exception):
    """Carry one anticipated failure message and its exit status."""

    def __init__(self, message: str, status: int) -> None:
        """Record the message and the exit status it maps to."""
        super().__init__(message)
        self.status = status


class _RootParser(argparse.ArgumentParser):
    """Render options first and top-level commands under titled groups."""

    def format_help(self, formatter: argparse.HelpFormatter | None = None) -> str:
        """Return help with options first and commands in titled groups."""
        if formatter is None:
            formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        formatter.start_section('options')
        formatter.add_arguments(self._get_optional_actions())
        formatter.end_section()
        for title, commands in _COMMAND_GROUPS:
            formatter.start_section(title)
            formatter.add_arguments(
                [argparse.Action([], name, help=text) for name, text in commands]
            )
            formatter.end_section()
        formatter.add_text(self.epilog)
        return formatter.format_help()


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = _RootParser(
        prog='sgalchemy',
        description='Generate typed entity stubs and manage cached schemas for SG sites.',
    )
    parser.add_argument('--version', action='store_true', help='show the version and exit')
    commands = parser.add_subparsers(
        dest='command', metavar='COMMAND', parser_class=argparse.ArgumentParser
    )

    stubs = commands.add_parser(
        'stubs',
        help=_COMMAND_HELP['stubs'],
        description=(
            'Generate or check type stubs for the entity classes in a site schema. '
            'Read the schema from --from-cache FILE, or from --site URL with credentials.'
        ),
        epilog=(
            'With --script-name, the API key comes from SGALCHEMY_API_KEY or a hidden '
            'terminal prompt. Without it, the session token comes from SGALCHEMY_SESSION_TOKEN.'
        ),
    )
    stubs.set_defaults(command_parser=stubs)
    output = stubs.add_argument_group('output')
    output.add_argument(
        '-o',
        '--out',
        type=Path,
        required=True,
        metavar='DIR',
        help='root directory for the generated stubs',
    )
    output.add_argument(
        '--check',
        action='store_true',
        help='compare stubs with the schema without writing files',
    )
    source = stubs.add_argument_group('schema source')
    source.add_argument(
        '--from-cache',
        type=Path,
        metavar='FILE',
        help='read the schema from a saved cache file',
    )
    source.add_argument('--site', metavar='URL', help='site URL to read the schema from')
    source.add_argument(
        '--script-name', metavar='NAME', help='script name to authenticate with an API key'
    )
    content = stubs.add_argument_group('stub content')
    content.add_argument(
        '--no-entity-aliases',
        action='store_false',
        dest='entity_aliases',
        help='omit display-name entity aliases',
    )
    content.add_argument(
        '--no-field-aliases',
        action='store_false',
        dest='field_aliases',
        help='omit field aliases with the sg_ prefix removed',
    )
    content.add_argument(
        '--permissive',
        action='store_true',
        help='allow unknown attributes in generated classes',
    )

    schema = commands.add_parser(
        'schema',
        help=_COMMAND_HELP['schema'],
        description='Manage the cached schema of an SG site.',
    )
    schema_commands = schema.add_subparsers(
        title='commands', dest='schema_command', metavar='COMMAND', required=True
    )
    refresh = schema_commands.add_parser(
        'refresh',
        help='fetch the schema now and rewrite its cache file',
        description='Fetch the site schema now and rewrite its cache file.',
        epilog=(
            'With --script-name, the API key comes from SGALCHEMY_API_KEY or a hidden '
            'terminal prompt. Without it, the session token comes from SGALCHEMY_SESSION_TOKEN.'
        ),
    )
    refresh.set_defaults(command_parser=refresh)
    refresh.add_argument(
        '--site', required=True, metavar='URL', help='site URL whose schema to fetch'
    )
    refresh.add_argument(
        '--script-name', metavar='NAME', help='script name to authenticate with an API key'
    )
    return parser


def _describe_failure(error: Exception) -> str:
    """Return an exception's message, or its type name when the message is empty."""
    return str(error) or type(error).__name__


def _format_fetched(at: datetime) -> str:
    """Format a schema snapshot timestamp in UTC at minute precision."""
    return at.astimezone(UTC).strftime('%Y-%m-%d %H:%M UTC')


def _is_provider_failure(error: Exception) -> bool:
    """Report whether an exception is a provider, network, or schema-cache failure."""
    return isinstance(error, (OSError, SgalchemyError)) or any(
        cls.__name__ in _PROVIDER_FAILURE_NAMES for cls in type(error).__mro__
    )


def _load_site_schema(engine: Engine, *, force: bool) -> SchemaLoad:
    """Load the site schema, mapping a missing provider and provider failures."""
    try:
        return load_schema_with_source(engine, force=force)
    except ModuleNotFoundError as error:
        if error.name != _PROVIDER_MODULE:
            raise
        raise _CommandError(_MISSING_PROVIDER, 2) from error
    except Exception as error:
        if not _is_provider_failure(error):
            raise
        raise _CommandError(
            f'could not load the site schema: {_describe_failure(error)}', 1
        ) from error


def _read_api_key(parser: argparse.ArgumentParser) -> str:
    """Read an environment secret or require a hidden terminal prompt."""
    key = os.environ.get('SGALCHEMY_API_KEY')
    if key:
        return key
    if sys.stdin is None or not sys.stdin.isatty():
        parser.error('set SGALCHEMY_API_KEY or run interactively to enter an API key')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            key = getpass.getpass('API key: ')
    except (EOFError, getpass.GetPassWarning):
        parser.error('could not read a hidden API key; set SGALCHEMY_API_KEY')
    if not key:
        parser.error('the entered API key is empty; set SGALCHEMY_API_KEY or enter a key')
    return key


def _render_version() -> tuple[int, str]:
    """Return the installed distribution version without other work."""
    import importlib.metadata  # noqa: PLC0415

    try:
        version = importlib.metadata.version('sgalchemy')
    except importlib.metadata.PackageNotFoundError:
        print(
            'error: installed distribution metadata for sgalchemy is unavailable',
            file=sys.stderr,
        )
        return 2, ''
    return 0, f'sgalchemy {version}\n'


def _report_error(message: str) -> None:
    """Print one terminal-safe error line to standard error."""
    visible = _CONTROL_CHARACTERS.sub(lambda match: f'\\x{ord(match.group()):02x}', message)
    print(f'error: {visible}', file=sys.stderr)


def _report_warning(message: str) -> None:
    """Print one terminal-safe warning line to standard error."""
    visible = _CONTROL_CHARACTERS.sub(lambda match: f'\\x{ord(match.group()):02x}', message)
    print(f'warning: {visible}', file=sys.stderr)


def _site_engine(site: str, script_name: str | None, parser: argparse.ArgumentParser) -> Engine:
    """Validate site options before selecting script or session-token credentials."""
    try:
        site = validate_site(site)
        require_static_credential_https(site)
    except ValueError as error:
        raise _CommandError('--site must be a complete https URL', 2) from error
    if script_name is not None and not script_name.strip():
        raise _CommandError('--script-name must not be empty', 2)
    try:
        if script_name is not None:
            return create_engine(site, script_name=script_name, api_key=_read_api_key(parser))
        token = os.environ.get('SGALCHEMY_SESSION_TOKEN')
        if not token:
            parser.error(
                'authenticate with --script-name NAME and an API key, '
                'or set SGALCHEMY_SESSION_TOKEN'
            )
        return create_engine(site, session_token=token)
    except ValueError as error:
        raise _CommandError(str(error), 2) from error


def _load_schema_cache(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[SchemaCache, str]:
    """Load the chosen schema snapshot and describe its source."""
    if args.from_cache is not None:
        cache = load_cache(args.from_cache)
        if cache is None:
            parser.error(f'could not load schema cache: {args.from_cache}')
        return cache, f'schema: {args.from_cache} (fetched {_format_fetched(cache.fetched_at)})'
    engine = _site_engine(args.site, args.script_name, parser)
    load = _load_site_schema(engine, force=False)
    if load.store_error is not None:
        _report_warning(
            f'could not write the schema cache {load.path}: '
            f'{_describe_failure(load.store_error)}; using the fetched schema without caching it. '
            'Set SGALCHEMY_SCHEMA_CACHE to use a writable directory'
        )
    if load.fetched:
        source = f'schema: fetched from {engine.site}'
    else:
        source = (
            f'schema: cached copy fetched {_format_fetched(load.cache.fetched_at)}; '
            "run 'sgalchemy schema refresh' to fetch the current schema"
        )
    return load.cache, source


def _refresh_schema(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[int, str]:
    """Force a schema refetch and return its cache path."""
    engine = _site_engine(args.site, args.script_name, parser)
    load = _load_site_schema(engine, force=True)
    if load.store_error is not None:
        raise _CommandError(
            f'could not write the schema cache {load.path}: '
            f'{_describe_failure(load.store_error)}; '
            'set SGALCHEMY_SCHEMA_CACHE to a writable directory',
            1,
        )
    return 0, f'refreshed schema cache: {load.path}\n'


def _run_stubs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[int, str]:
    """Generate or check stubs against the selected schema cache."""
    if args.from_cache is not None and (args.site is not None or args.script_name is not None):
        parser.error('choose one schema source: --from-cache FILE, or --site URL with credentials')
    if args.from_cache is None and args.site is None:
        parser.error('stubs requires --from-cache FILE or --site URL')
    cache, source = _load_schema_cache(args, parser)
    if args.check:
        fresh = _stubgen.check_stubs(
            cache,
            args.out,
            entity_aliases=args.entity_aliases,
            field_aliases=args.field_aliases,
            permissive=args.permissive,
        )
        message = 'stubs are fresh' if fresh else 'stub drift detected'
        return (0 if fresh else 1), f'{source}\n{message}\n'

    try:
        report = _stubgen.generate_stubs(
            cache,
            args.out,
            entity_aliases=args.entity_aliases,
            field_aliases=args.field_aliases,
            permissive=args.permissive,
        )
    except (OSError, SgalchemyError) as error:
        raise _CommandError(f'could not write stubs: {_describe_failure(error)}', 1) from error
    lines = [source, *(f'wrote {path}' for path in report.written)]
    if report.skipped_aliases:
        lines.append(f'aliases skipped: {", ".join(report.skipped_aliases)}')
    else:
        lines.append('aliases skipped: none')
    return 0, '\n'.join(lines) + '\n'


def _run(arguments: list[str]) -> tuple[int, str]:
    """Parse arguments, run the selected command, and report anticipated failures."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    if args.version:
        return _render_version()
    if args.command is None:
        parser.error('the following arguments are required: COMMAND')
    handler = _run_stubs if args.command == 'stubs' else _refresh_schema
    try:
        result = handler(args, args.command_parser)
    except _CommandError as failure:
        _report_error(str(failure))
        return failure.status, ''
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `sgalchemy` command-line interface.

    Each command finishes its work before writing output. If the reader closes
    standard output, unwritten output is discarded and the exit status still
    reports the command's result.

    Args:
        argv: Arguments to parse, or `None` to use `sys.argv`.

    Returns:
        The process exit status.

    Raises:
        SystemExit: With status 2 for invalid arguments or missing input, or
            status 0 after printing `--help`.
        OSError: If writing or flushing command output fails for a reason other
            than downstream closure.
    """
    if sys.stdout is None:
        # Started with standard output closed: nothing can be delivered, and
        # no text stream exists to write to.
        _report_error('standard output is unavailable')
        return 2
    output = ''
    try:
        status, output = _run(sys.argv[1:] if argv is None else list(argv))
    except KeyboardInterrupt:
        status = 130
    finally:
        # Also flushes help or usage text that argparse wrote before raising SystemExit.
        write_stdout(output)
    return status
