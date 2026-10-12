"""Command-line interface for schema caches and generated stubs."""

import argparse
import getpass
import os
import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from sgalchemy import _stubgen
from sgalchemy._core._engine import Engine, create_engine
from sgalchemy._downstream import write_stdout
from sgalchemy._errors import SgalchemyError
from sgalchemy.schema._cache import SchemaCache, load_cache
from sgalchemy.schema._cachedir import cache_file

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
        self._status = status


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
        description='Generate or check type stubs for the entity classes in a site schema.',
        epilog=(
            'With --site, the API key comes from SGALCHEMY_API_KEY or a hidden terminal prompt.'
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
    source.add_argument('--script-name', metavar='NAME', help='script name to authenticate with')
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
        epilog='The API key comes from SGALCHEMY_API_KEY or a hidden terminal prompt.',
    )
    refresh.set_defaults(command_parser=refresh)
    refresh.add_argument(
        '--site', required=True, metavar='URL', help='site URL whose schema to fetch'
    )
    refresh.add_argument(
        '--script-name', required=True, metavar='NAME', help='script name to authenticate with'
    )
    return parser


def _describe_failure(error: Exception) -> str:
    """Return an exception's message, or its type name when the message is empty."""
    return str(error) or type(error).__name__


def _is_provider_failure(error: Exception) -> bool:
    """Report whether an exception is a provider, network, or schema-cache failure."""
    return isinstance(error, (OSError, SgalchemyError)) or any(
        cls.__name__ in _PROVIDER_FAILURE_NAMES for cls in type(error).__mro__
    )


def _load_site_schema(engine: Engine, *, force: bool) -> None:
    """Load the site schema, mapping a missing provider and provider failures."""
    try:
        engine.load_schema(force=force)
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
    if not sys.stdin.isatty():
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


def _site_engine(site: str, script_name: str, parser: argparse.ArgumentParser) -> Engine:
    """Create an engine for one site, reporting an invalid site URL or script name."""
    api_key = _read_api_key(parser)
    try:
        return create_engine(site, script_name=script_name, api_key=api_key)
    except ValueError as error:
        raise _CommandError(str(error), 2) from error


def _connected_engine(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> Engine:
    """Create an engine from validated CLI credentials."""
    if args.site is None or args.script_name is None:
        parser.error('stubs requires --from-cache or both --site and --script-name')
    return _site_engine(args.site, args.script_name, parser)


def _load_schema_cache(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> SchemaCache:
    """Load the explicit or connection-populated schema cache."""
    path = args.from_cache
    if path is None:
        engine = _connected_engine(args, parser)
        _load_site_schema(engine, force=False)
        path = cache_file(engine.site)
    cache = load_cache(path)
    if cache is None:
        parser.error(f'could not load schema cache: {path}')
    return cache


def _refresh_schema(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[int, str]:
    """Force a schema refetch and return its cache path."""
    engine = _site_engine(args.site, args.script_name, parser)
    _load_site_schema(engine, force=True)
    return 0, f'refreshed schema cache: {cache_file(engine.site)}\n'


def _run_stubs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[int, str]:
    """Generate or check stubs against the selected schema cache."""
    cache = _load_schema_cache(args, parser)
    if args.check:
        fresh = _stubgen.check_stubs(
            cache,
            args.out,
            entity_aliases=args.entity_aliases,
            field_aliases=args.field_aliases,
            permissive=args.permissive,
        )
        return (0, 'stubs are fresh\n') if fresh else (1, 'stub drift detected\n')

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
    lines = [f'wrote {path}' for path in report.written]
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
        return failure._status, ''  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
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
    """
    if sys.stdout is None:
        # Started with standard output closed: nothing can be delivered, and the
        # no text stream is available to write to.
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
