from pathlib import Path

from _fakes import _FakeShotgun
from _support import fetch_schema
from cli_helpers import process_transcript, run_cli, run_with_closed_output

from sgalchemy.schema import _cache


def test_closed_stdout_before_help_is_quiet() -> None:
    result = run_with_closed_output(['--help'])
    assert (result.returncode, result.stderr) == (0, b''), process_transcript(result)


def test_closed_stdout_during_stub_check_keeps_drift_status(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / 'schema.json'
    _cache.store_cache(cache_path, fetch_schema(fake_sg))

    result = run_with_closed_output(
        ['stubs', '-o', str(tmp_path / 'absent'), '--from-cache', str(cache_path), '--check']
    )

    assert (result.returncode, result.stderr) == (1, b''), process_transcript(result)


def test_closed_stdout_during_stubs_keeps_success(
    fake_sg: _FakeShotgun,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / 'schema.json'
    _cache.store_cache(cache_path, fetch_schema(fake_sg))
    out = tmp_path / 'stubs'

    result = run_with_closed_output(['stubs', '-o', str(out), '--from-cache', str(cache_path)])

    assert (result.returncode, result.stderr) == (0, b''), process_transcript(result)
    assert (out / 'sgalchemy' / 'entities' / '__init__.pyi').is_file()


def test_module_entry_point_runs_main() -> None:
    result = run_cli(['--help'], expect=0)
    assert result.stdout.startswith('usage: sgalchemy '), process_transcript(result)
