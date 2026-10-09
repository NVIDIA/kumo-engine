# Public connector tests

Run selected reader, SDK, and graph tests without cloud accounts, Kumo
credentials, or a GPU. They use generated data, bundled files, fake cloud
responses, and local MinIO storage. The [test matrix](../connector-test-matrix.md)
lists all 15 groups. `all` means these groups, not every repository test.

## Run the tests

Start from the repository root with Python 3, Docker with Buildx and Compose,
and OpenSSL available. The full setup requires Linux x86_64 containers.
Builds need internet access and space for Rust and Python dependencies.
The command builds DiskGraph from this checkout before it runs native tests:

```bash
python3 scripts/test_public.py --group all
```

To run one group, replace `all` with its name from the matrix.
Use `connectors` for the nine reader groups, which also run on ARM64.
Docker keeps build layers for later runs. The command does not install packages
or change configuration on your host.

For one group, use `--test-file` to select a file or `-k` to filter test names.
You can combine them:

```bash
python3 scripts/test_public.py --group s3 \
  --test-file packages/kumo-s3/tests/test_session.py -k 503
```

File paths start at the repository root. Files outside the group or in its
excluded directories are rejected. These options require one group and cannot
be combined with `--build-wheel`. No matching tests fails the command.
The direct reader command below accepts the same options with one `--suite`.
Without these options, local commands and CI still run the full selected groups.

Tests have no external network access. MinIO uses a private Docker network
with fake accounts and temporary TLS. Other groups use `--network none`.
S3 endpoint tests start a server inside their own container. The runner passes no
host credentials into containers and removes test containers after each group.

Results are in `public-reports/`: one log and XML file per group, plus
`results.json`. The summary records the source revision, uncommitted-change
status, commands, image IDs, Python version, architecture, and installed packages.
Use `--reports PATH` for another directory. It must be writable only by its owner.
Containers write to separate temporary folders and cannot edit host reports or
build files. The runner rejects symbolic links as test results.
Before a test run, it removes the previous summary and selected groups' reports.
A setup failure cannot leave those earlier results in place.
`--build-wheel` only builds the wheel and keeps existing test reports.

A test or setup failure fails the command. Later groups still run after a test,
image build, or report failure. If an image cannot build, its groups fail while
groups that use other images continue. Missing Docker or a failed native wheel
build can stop the command before tests start.

## Add a local test

Put each new `test_*.py` file in its group's directory from the
[test matrix](../connector-test-matrix.md). Pytest discovers files and nested
folders there. Docker copies those test trees. New test files need no runner,
Docker, or CI registration. Put local fixtures in that directory's `conftest.py`.

The `mixed-files` group discovers `test/connector/local` and
`test/io_utils/fsutil/local`. Keep real provider tests outside these folders.
Local connector tests share cache cleanup and data fixtures without loading the
hosted test setup. The shared filesystem contract runs locally here and against
real providers through `test/io_utils/fsutil/test_all.py`. Its two existing
large-file skips remain visible in the results.

The public runner excludes each test directory's `live/` folder before import.
Live provider tests keep their explicit opt-in checks when run directly.
Reader groups also exclude `native/`; the `shared-sql` group runs Snowflake's
native type tests with DiskGraph. Keep new tests in the environment they need.
Changes that introduce another component or environment still need a group.

## Choose the version to test

Use code and tests from the same checkout. These tests cover selected Python
and dependency versions, not every version that Kumo supports.

SDK and native images install dependencies recorded in `uv.lock`.
The native image installs `kumo-engine` with CPU PyTorch and GPU Faiss.
These tests use the CPU. The backend's CPU Faiss selection stays unchanged.
The reader image uses the smaller `tests/public/requirements-all.txt` profile.
Its main runtime pins match the repository where possible. It uses Arrow 23
for Snowflake compatibility, a driver with Python 3.14 ARM64 wheels, and a
newer ARM64-compatible pyinstrument build.
The native image uses Arrow 24. Reader requirements do not lock every indirect
dependency. Use the recorded package versions to identify the tested environment.

CI reuses a native wheel only when its source and build settings match.
`--wheel-dir PATH` accepts a wheel built from the same checkout.

## Run the public connector suites

On Linux x86_64 or ARM64, you can also run reader tests directly in a separate
Python environment. This recipe matches the reader image's Python version:

```bash
python3.14 -m venv .venv-public-tests
.venv-public-tests/bin/python -I -m pip --isolated install \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.0'
.venv-public-tests/bin/python -I -m pip --isolated install \
  --index-url https://pypi.org/simple -r tests/public/requirements-all.txt
.venv-public-tests/bin/python -I scripts/test_public_connectors.py --suite all
```

The Python socket guard blocks ordinary connections during collection and test
setup. The S3 server tests allow loopback connections. The plugin restores
sockets before fixture cleanup, and it cannot block compiled code or child
processes. Use Docker when you need network isolation for the entire run.
The direct command excludes the six SDK, graph, SQL, and MinIO groups.

## Validation evidence

Read the [workflow results](https://github.com/kumo-ai/kumo/actions/workflows/public-connectors.yml)
for the revision under test. Download `public-tests-*` artifacts for commands,
environments, per-test results, and skip reasons. Results apply only to the
recorded source and environment.

The [test matrix](../connector-test-matrix.md#ci-behavior) describes CI triggers,
test coverage, and limits.
