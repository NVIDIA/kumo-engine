# Snowflake notebook platform checks

The trial workflow runs only through manual dispatch on
`egu/mlops-platform-integration` in `NVIDIA/kumo-engine`. It has no push, PR,
schedule, or comment trigger. Choose `graph-ui` for notebook rendering or
`engine` for training and lifecycle execution.

GitHub's repository permissions control who can request a manual run. The
protected environment requires approval from `NVIDIA/kumo-engineers`, the
repository's maintainer team, before Snowflake credentials become available.
Each cloud job rechecks the environment restrictions with its built-in
`GITHUB_TOKEN`, including reruns. No separate team-read credential is required.

The separate engine job covers notebook preflight and the four engine lifecycle
phases: fit/register, fresh-process prediction and alias moves, monitoring and
late-label evaluation, and permission denial. It uses the identified installed
wheel. UI verification is excluded. Full epoch-history export remains a separate
unimplemented requirement; a successful engine check is not full qualification.
Both results are retained in `summary.json` and `ci-verdict.json`.

## GitHub setup

In repository **Settings → Environments**, create `snowflake-mlops-test`.
Under **Deployment branches and tags**, select **Selected branches and tags**
and allow only the branch `egu/mlops-platform-integration`.
The workflow checks this restriction before opening a deployment job; a missing
or differently restricted environment fails before any Snowflake connection.
Require `NVIDIA/kumo-engineers` as the only environment reviewer and disable
administrator bypass. A maintainer can approve their own manually requested run.
The workflow verifies these settings before execution and rechecks them inside
each cloud job.

Add these four secrets to this environment itself:

- `SNOWFLAKE_TEST_CONFIG`: the JSON returned by `resources.expected()` from
  `packages/kumo-engine/test/snowflake_notebook/resources.py`. It authorizes the
  dedicated account, users, role, database, schema, warehouse, pool, service,
  stages, and exact candidate/asset manifests. Update it after an intentional
  change to either manifest.
- `SNOWFLAKE_PRIVATE_KEY`: the complete encrypted PEM private key, including
  its BEGIN/END lines, for `KUMO_SNOWFLAKE_CI`.
- `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE`: its passphrase.
- `SNOWFLAKE_UI_CREDENTIALS`: the browser credential JSON described below.

The local memory cache is not transferred to GitHub. The connector decrypts
these CI secrets in process; neither the key nor passphrase enters the notebook
payload or evidence. Only the graph UI job reads the browser credential values.
Configure all four secrets in this environment, rather than at repository or
organization scope. GitHub runs also reject
local RSA-cache and browser-credential-file fallback.

Resource names remain visible as test-only constraints and in retained evidence.
The protection restricts access to those resources; it does not make their
identifiers confidential. Before local live execution, explicitly supply the
same `SNOWFLAKE_TEST_CONFIG` JSON through the environment. Planning remains
credential-free.

## Candidate and payload

[candidate.json](../../packages/kumo-engine/test/snowflake_notebook/candidate.json)
identifies the existing development wheel and dependency archive by SHA-256,
size, stage location, and observed object metadata. This trial tests that supplied
wheel; it does not build a wheel from the workflow commit. A future PR candidate
build must provide a corresponding manifest rather than reuse this candidate.

The runner submits the current checkout's test helpers. It checks that staged
assets retain their pinned metadata without downloading or uploading the large
archive. Inside the notebook, every referenced file is downloaded and checked
against its SHA-256 before installation or import. Offline pip resolves the
candidate's `snowflake` extra from the recorded dependency archive.

Changed or missing pinned assets fail rather than silently selecting another
wheel. Upload caching for the small helper files also uses content hashes.
Preflight runs before training. Attempts use unique model/experiment, stage
prefixes, and Code Bundles. Existing test resources are required; the runner
neither provisions infrastructure nor deletes retained evidence.

## Local validation and execution

Prepare and inspect without credentials or service calls:

```sh
python packages/kumo-engine/test/snowflake_notebook/prepare.py \
  --output-dir /tmp/snowflake-payload-001
python packages/kumo-engine/test/snowflake_notebook/launch.py \
  --payload-manifest /tmp/snowflake-payload-001/manifest.json \
  --preflight-payload /tmp/snowflake-payload-001/preflight \
  --lifecycle-payload /tmp/snowflake-payload-001/lifecycle \
  --output-dir /tmp/snowflake-result-001
```

For local RSA execution, unlock the key once with
`python packages/kumo-engine/test/snowflake_notebook/key_cache.py --key-file KEY`.
Use a new output directory, add `--key-file KEY --execute` to the launch command,
and keep the encrypted key outside Git. The Linux user-keyring cache expires
after eight hours and writes no decrypted key file. An expired/missing cache
fails without another implicit passphrase prompt.

The GitHub runner uses Python 3.12 and installs the Snowflake connector and
cryptography submission dependencies. Notebook dependencies are installed in a
fresh virtual environment. Each notebook has a 30-minute client deadline and
session SQL timeout with cancellation grace. Runs serialize; they do not cancel
an older run or automatically retry training. A runner interruption attempts
remote query cancellation, but a forced runner termination can still require
operator inspection using the saved execution query ID.

The Actions artifact retains receipts, hashes, query IDs, phase logs, the exact
payload manifest, and full qualification gaps. Review the reported wheel identity
and both engine/full-qualification verdicts. Missing secrets or access are
prerequisite failures, never passes.

## Local Snowsight UI verification

Use `--ui` to select only browser verification. It returns before notebook
preparation, RSA authentication, or training. Without `--execute` it only
validates saved receipts and writes the UI expectations. With `--execute`, it
signs in through Snowsight using password and TOTP, with one MFA submission and
no automatic login retries.

```sh
python packages/kumo-engine/test/snowflake_notebook/launch.py \
  --execute --ui --ui-run /path/to/retained-snowflake-run
```

`--ui-run` selects the lifecycle directory containing
`lifecycle/download/lifecycle-result.json`; omitting it selects the newest valid
local `/tmp/egu-snowflake-e2e-*` lifecycle. The selected path and candidate hash
are retained in the UI report. Assertions use exact checksummed run references,
not the latest visible Snowsight row. `--ui true` and `--ui=true` also work.
UI mode cannot be combined with negative-only, history-readback, or saved-result
modes.

Provide an owner-only regular JSON file at
`~/.config/kumo/snowflake-ui/credentials.json`, or use `--ui-credentials FILE`.
Alternatively, supply the same JSON through `SNOWFLAKE_UI_CREDENTIALS`.
Its fields are `account_url`, `user`, `password`, and `totp_secret`. The account
URL must identify the configured test account. `totp_secret` is the Base32
manual-entry secret saved during enrollment, not a current six-digit code.
The browser identity needs visibility into the same test experiment and models.
SQL/RSA authentication is not used for browser login.

Node.js, Playwright, Chromium, and its shared libraries must be available.
Install the locked browser package with
`npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser`, then
run its `node_modules/playwright/cli.js install chromium` with Node.js. On a
provisioned runner, use `install --with-deps chromium` to include browser libraries. `KUMO_PLAYWRIGHT_MODULE` can select an existing Playwright
installation; `PLAYWRIGHT_BROWSERS_PATH` selects its browser cache. Missing
browser prerequisites or incomplete authentication produce a blocked result.

The browser verifies finished status for the ten retained runs, all recorded
metric values at Snowsight's five-decimal display precision, bundle/version/kind
tracking parameters, both epochs of train/loss and validation AUROC for both fit
runs, and V1/V2 under the exact model/database/schema. Registry bundle metadata,
alias controls, notebook graph/explanation widgets, and epoch-history exports
have separate checks; this browser result does not qualify those surfaces.

Each UI attempt retains `expectations.json`, `summary.json`, screenshots, rendered
accessibility snapshots, and an interaction journal. Credentials are delivered
to the browser worker through stdin. Authentication-page screenshots, browser
storage state, and network traces containing session credentials are not saved.
The result always records that no training was submitted.

## Notebook graph UI

The graph scenario uses the same checked-in command locally and on a runner:

```sh
python packages/kumo-engine/test/snowflake_notebook/graph_notebook.py \
  --execute --key-file /path/to/encrypted-key.p8 \
  --ui-credentials /path/to/browser-credentials.json \
  --output-dir /path/to/new-result-directory
```

Omit `--execute` to inspect the plan without credentials or cloud access. On a
runner, supply the protected environment secrets and omit local file arguments.
`candidate.json` selects the wheel. `--source-ref` defaults to HEAD and selects
seven graph/table/widget source hashes that must match that installed wheel;
it does not build a new wheel. Select another revision only when deliberately
validating that revision, and retain its exact commit in the report.

The notebook uses the unchanged adtech graph from An's checksum-pinned PR172
fixture: 9 tables, 52 columns, and 10 relationships. `graph-assets.json` pins
already staged fixture and PDF test-tool assets. Normal runs need no AWS access
or large upload. The notebook verifies and installs the staged wheel, shuts down its kernel,
reconnects to the same dedicated service, and verifies setup variables were
cleared before executing graph assertions. Installed packages survive this reset. Before executing cells, the browser
checks `KUMO_TEST_POOL`, `KUMO_TEST_WH`, and the dedicated role. Only a known
service-startup timeout permits one connection retry after read-only polling
of the same test pool; no notebook cell or training job is retried.

Kernel checks verify that default display performs no source reads, explicit
previews use actual fixture rows, and SVG/PDF/Markdown exports exclude sample
values. Browser checks cover both graphs' schema, relationships, key and sample
tooltips, zoom/fit/layout controls, and visible export commands. The browser
stops only the dedicated notebook kernel and opens a fresh page to verify the
same saved execution's diagrams and retained data without rerunning cells.
Inactive cached widgets are checked without requiring live interactions.
Finally, RSA collection downloads and verifies all six small evidence files.

For a read-only recheck, replace execution preparation with
`--verify-notebook /path/to/previous-result/notebook.json`. Keep that file and
its `.ipynb` together when relocating artifacts. This mode covers saved outputs
and exports, not live controls or kernel shutdown. Explanations unavailable on
main are outside this scenario.

Dispatch the registered workflow with `mode: graph-ui` or `mode: engine`,
then approve the protected environment as a member of `NVIDIA/kumo-engineers`.
No push starts a test. GitHub-hosted execution requires all environment secrets,
maintainer environment approval, and the verified environment restrictions.
Local results do not qualify a different GitHub-hosted revision.

Saved outputs can render as cached widgets or static SVG fallback images. Both
paths verify all tables, columns, keys, and relationships. Cached widgets also
retain the explicitly requested sample metadata. Static fallback diagrams omit
samples and live controls by design; the report labels that narrower coverage.

The graph UI scenario requires the dedicated browser user’s existing notebook
service `EGU_GRAPH_UI_SERVICE`, with CPU Python 3.12 runtime and
`KUMO_TEST_POOL (GEN_X64_G2_8)`. The runner selects that exact service in the UI
and verifies its pool before connection. It does not create a notebook service
or use Snowsight’s remembered default service.

A high-memory warning on the dedicated service blocks the run immediately; the
runner does not choose Continue anyway. Failed executing attempts retain UI
evidence before stopping their own connected test kernel. Read-only saved-output
checks never stop kernels, including when their assertions fail.

After successful installation, the notebook removes its temporary dependency
archive and extracted wheelhouse. The candidate wheel, pinned fixture, helper
files, and evidence remain available for verification. Staged assets are retained.
