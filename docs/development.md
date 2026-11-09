# Development

Use this guide to set up a development checkout, run checks, and update dependencies. To test your
own pipeline code with a fake provider, see [Testing](tooling.md#test-your-pipeline-code).

## Set up a development checkout

Install these tools before running setup:

| Tool    | Required version | Purpose                                                            |
| ------- | ---------------- | ------------------------------------------------------------------ |
| Python  | 3.11 or newer    | Project interpreter, available as `python` on `PATH`               |
| uv      | 0.13.6           | Environment setup and dependency locks; other versions are refused |
| Node.js | 26.12.0          | Pyright quality checks and generated-consumer tests                |
| Taplo   | 0.10.0           | TOML code blocks in Markdown                                       |
| shfmt   | 3.14.1           | Shell code blocks in Markdown                                      |

From the repository root, run:

```sh
make setup
make check
```

`make setup` creates `.venv` if needed, installs the exact hashed pins from `requirements-ci.txt`,
and installs the project in editable mode. To select another interpreter, use
`make setup PYTHON=/path/to/python`; setup never downloads Python.

Run setup again after changing `pyproject.toml` or the lock files, including when `make format`
rewrites `pyproject.toml`. Checks compare `.venv` with those files and refuse to run if the
environment is missing or outdated. A check never installs or updates packages.

## Run checks and formatting

| Command                 | What it does                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `make check`            | Run quality checks and the test suite                                                    |
| `make quality`          | Run formatting, lint, lock, type, whitespace, and workflow checks without changing files |
| `make test`             | Run the test suite serially                                                              |
| `make format`           | Apply formatters; inspect the changes before committing                                  |
| `make format-check`     | Check formatting and lint without changing files                                         |
| `make whitespace-check` | Check trailing whitespace                                                                |

Quality checks include Pyright for Linux, macOS, and Windows, and an offline zizmor audit of GitHub
Actions workflows.

Keep Taplo and shfmt available when formatting Markdown. If either is missing, mdformat warns that
it could not format a code block but still succeeds, so read its warnings. CI downloads
checksum-verified native binaries. The lock excludes the Python Taplo package otherwise pulled in
by `mdformat-config`, because that package lacks some ARM wheels.

For the parallel scheduling used in CI, run:

```sh
make test TEST_FLAGS='-n auto --dist worksteal --durations=15'
```

Set `PYTEST_XDIST_AUTO_NUM_WORKERS` when a runner's visible CPU count exceeds its CPU quota.

## Install the command-line tool

`make install` installs the base `sgalchemy` CLI with `uv tool`; `make reinstall` replaces an
existing base installation. To include Autodesk's provider, run:

```sh
uv tool install '.[official]'
```

Add `--reinstall` to replace that installation. Replace `official` with `api` or `async` for the
other [provider choices](../README.md#installation).

## Update dependencies

`requirements-ci.txt` pins development tools, the `official` provider extra, and the build backend.
`requirements-test.txt` selects the smaller set installed by test jobs, constrained by the same
pins. Neither file ships with the package; published dependency ranges remain in `pyproject.toml`.
Alternative provider extras are optional.

Use the pinned uv version. After changing dependencies, run:

```sh
make lock-update
make setup
make lock-check
```

For an intentional upgrade, use `make lock-update LOCK_FLAGS=--upgrade` and review both pin diffs.
`make check` includes the lock check.

## Run provider tests

Provider contract tests skip when their provider is not installed. Require one provider environment
with `SGALCHEMY_REQUIRE_PROVIDER`:

| Value      | Required distribution |
| ---------- | --------------------- |
| `official` | `shotgun-api3`        |
| `api`      | `sg-python-api`       |

For example:

```sh
SGALCHEMY_REQUIRE_PROVIDER=official make test
```

Set `SGALCHEMY_REQUIRE_NATIVE_ASYNC=1` to require the native async provider, which needs the `api`
provider. In strict provider mode, the run fails if another provider distribution is installed, no
matching test is collected, or a matching test skips.

## Run live tests

The opt-in live contract test creates, updates, reads, and deletes a real record. Use a site where
those writes are safe, and configure a writable entity type and its required fields:

| Variable                     | Value                                                                  |
| ---------------------------- | ---------------------------------------------------------------------- |
| `SGALCHEMY_LIVE_SITE`        | Site URL                                                               |
| `SGALCHEMY_LIVE_SCRIPT_NAME` | Script identity                                                        |
| `SGALCHEMY_LIVE_API_KEY`     | Script API key                                                         |
| `SGALCHEMY_LIVE_ENTITY_TYPE` | Schema entity type to create                                           |
| `SGALCHEMY_LIVE_NAME_FIELD`  | Writable text field for the test's unique name                         |
| `SGALCHEMY_LIVE_CREATE_DATA` | JSON object with any other required create fields                      |
| `SGALCHEMY_REQUIRE_LIVE`     | Set to `1` to fail on missing configuration or an unavailable provider |

With these variables configured, run `make test`. The live test supplies a unique name and reads
the updated record in a fresh session before deleting it. Missing configuration skips the test by
default.

## Continuous integration

The [CI workflow](../.github/workflows/ci.yml) runs `quality` first, then the `test` jobs in
parallel. `package` starts after those jobs pass. For hosted runs, `hosted-validation` reports the
overall outcome, including failures. Pushes run every job on GitHub-hosted runners, with every
test, unless the repository variable `CI_RUN_ON_PUSH` is `false`.

The `test` matrix covers Python 3.11 to 3.15 on Linux and Python 3.13 on macOS and Windows. All
test jobs have a one-hour hang guard. CI installs Node.js for Pyright and runs tests with
pytest-xdist.
