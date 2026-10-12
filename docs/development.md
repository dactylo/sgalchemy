# Development

## Local environment

Install these tools before you set up a checkout:

- Python 3.11 or newer, as `python` on `PATH`. To use another interpreter, run
  `make setup PYTHON=/path/to/python`; setup never downloads Python.
- uv 0.12.23. Setup and the lock targets refuse any other version, so dependency locks resolve the
  same way locally and in CI.
- Node.js 24.21.0 for Pyright in quality checks and generated-consumer tests. CI installs it.
- Taplo 0.10.0 and shfmt 3.14.1. The Markdown formatter uses them for TOML and shell code blocks in
  the docs.

Then, from the repository root:

```sh
make setup
make check
```

`make setup` creates `.venv` if it does not exist, installs the exact hashed pins from
`requirements-ci.txt`, and installs the project in editable mode. Run it again after changing
`pyproject.toml` or the lock files, including when `make format` rewrites `pyproject.toml`. The
other targets compare `.venv` with those files and refuse to run when it is missing or outdated, so
a check never installs or updates packages.

`make quality` runs the read-only checks that CI's quality job runs: formatting and lint,
dependency locks, Pyright type checks for Linux, macOS, and Windows, and trailing whitespace.
`make check` adds the test suite, and `make test` runs the suite alone. `make format` applies the
formatters; review its changes before committing. `make install` installs the base `sgalchemy` CLI
with `uv tool`, and `make reinstall` replaces an existing base installation. To install a checkout
with Autodesk's provider, use `uv tool install '.[official]'`, or
`uv tool install --reinstall '.[official]'` to replace that installation. As with the
[README installation choices](../README.md#installation), replace `official` with `api` or `async`
for the corresponding provider.

`requirements-ci.txt` pins the development tools, the `official` provider extra, and the build
backend. `requirements-test.txt` selects the smaller set that test jobs install, constrained by the
same pins. Neither file ships with the package; published dependency ranges stay in
`pyproject.toml`.

CI downloads checksum-verified Linux builds of Taplo and shfmt. The lock leaves out the Python
Taplo package that `mdformat-config` would otherwise pull in, because that package lacks some ARM
wheels and CI uses the native binary instead. When Taplo or shfmt is unavailable, mdformat warns
that it could not format a code block but still succeeds, so read its warnings.

## Continuous integration

The [CI workflow](../.github/workflows/ci.yml) runs `quality` first. The `test` jobs then run in
parallel. `package` starts only after those jobs pass; `hosted-validation` reports the outcome of a
hosted run, including failures.

CI uses GitHub-hosted runners unless a run explicitly requests local runners. In a private
repository, the Actions variables `CI_LINUX_RUNNER`, `CI_MACOS_RUNNER`, and `CI_WINDOWS_RUNNER`
select local runners for runs started from Actions > CI > Run workflow with the hosted option
disabled. If a variable is absent, that operating system uses a hosted runner. Pushes, pull
requests, public repositories, and runs with the hosted option enabled use hosted runners for every
job and run every test.

A staging repository sets the Actions variable `CI_RUN_ON_PUSH` to `false`. Its pushes then appear
as `CI / push skipped` and run no jobs; CI runs only on request. A newer requested run on a branch
cancels an older one. Push runs are never cancelled, because the Release workflow needs the tagged
commit's push run.

| Requested run                                                 | Actions > CI > Run workflow | Title               |
| ------------------------------------------------------------- | --------------------------- | ------------------- |
| Quick: local runners, slow tests omitted on macOS and Windows | hosted off, slow tests off  | `CI / local`        |
| Full: local runners, every test                               | hosted off, slow tests on   | `CI / local (full)` |
| Hosted: GitHub-hosted runners, every test                     | hosted on                   | `CI / hosted`       |

The `test` matrix covers Python 3.11 to 3.15 on Linux and Python 3.13 on macOS and Windows. All
test jobs have a one-hour hang guard, on local and hosted runners, in quick and full runs.

CI runs tests in parallel with pytest-xdist. Set `PYTEST_XDIST_AUTO_NUM_WORKERS` when a runner's
visible CPU count exceeds its CPU quota. For the same scheduling locally, run:

```sh
make test TEST_FLAGS='-n auto --dist worksteal --durations=15'
```

The manual [Release workflow](../.github/workflows/release.yml) publishes the package to PyPI after
checking its version tag and a successful push CI run for the tagged commit.

## Dependency updates

CI selects the official Autodesk SDK extra; alternative provider extras remain optional. Use the uv
version pinned in CI. After changing dependencies, run `make lock-update`. For an intentional
upgrade, run `make lock-update LOCK_FLAGS=--upgrade` and review both pin diffs. Run
`make lock-check` to verify them; `make check` includes this check.

## Project notes

`hosted-validation` is skipped for a local requested run; it is not a manual gate.

No tests are marked `slow`, so quick and full local runs select the same tests. `make test` runs
the suite serially.

### Live tests

The opt-in live contract test reads `SGALCHEMY_LIVE_SITE`, `SGALCHEMY_LIVE_SCRIPT_NAME`, and
`SGALCHEMY_LIVE_API_KEY` for a site and script credential. Set `SGALCHEMY_LIVE_ENTITY_TYPE` to a
schema entity type, `SGALCHEMY_LIVE_NAME_FIELD` to a writable text field on that type, and
`SGALCHEMY_LIVE_CREATE_DATA` to a JSON object containing any other fields required to create it.
The test supplies a unique name, creates a record, updates it, reads it in a fresh session, and
deletes it from the configured site. Missing configuration skips the test by default; set
`SGALCHEMY_REQUIRE_LIVE=1` to make missing configuration or an unavailable provider fail instead.
