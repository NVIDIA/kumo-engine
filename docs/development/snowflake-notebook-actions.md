# Snowflake notebook tests

Run these Snowflake scenarios locally or through GitHub Actions. Both use the same test scripts and installed
candidate wheel.

## What it tests

| Scenario | Checks |
| --- | --- |
| Engine lifecycle | Train/register two versions; reload in fresh processes; predict by version and alias; move the alias; verify tracking, lineage, prediction IDs/schema, drift and evaluation after labels arrive. Includes real denied artifact access; incomplete-artifact checks currently run only in the local profile. |
| Lifecycle UI | Open the retained lifecycle's runs and model versions in Snowsight; verify status, metrics and epoch history. Reuses existing results without training again. |
| Notebook graph UI | Render the shared PR172 adtech fixture in a Snowflake notebook; verify schema, relationships, tooltips and controls; reopen saved outputs after kernel shutdown; verify SVG/PDF/Markdown exports. Default display must read no source data, and exports must exclude sample values. |

These resource names are deliberate test-only allowlist entries, not production
resources. Keep the browser bot limited to this role and test resources because
its credential JSON contains both the password and TOTP enrollment secret.

The lifecycle uses 60 users, 1,500 events and two fits of two epochs each. See
[local lifecycle testing](mlops-validation.md) for the shared fixture and runner.
The graph scenario runs without training.

Current gaps: full epoch history in model exports, independently verified
nonzero drift sensitivity, and interruption of an active remote writer.
Explanations unavailable on main are outside the graph scenario. A successful
engine or graph check alone does not establish full qualification.

## Setup

Use Python 3.12. Submission needs `snowflake-connector-python` and
`cryptography`. UI checks also need Node.js 22, Playwright and Chromium:

```sh
python -m pip --isolated install --require-hashes -r \
  packages/kumo-engine/test/snowflake_notebook/client-requirements.txt
npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser
node packages/kumo-engine/test/snowflake_notebook/browser/node_modules/playwright/cli.js install chromium
```

Chromium's system libraries must already be available locally. GitHub Actions
installs them with Playwright's `install --with-deps chromium` command.

Local runs and Actions use the same generated client lock. To update its versions,
edit [client-requirements.in](../../packages/kumo-engine/test/snowflake_notebook/client-requirements.in)
and run `python packages/kumo-engine/test/snowflake_notebook/compile_client.py`.
The generator preserves the SPDX header. Add `--upgrade-package NAME` to refresh
a transitive dependency.

The staged notebook inventory is generated from the prepared dependency archive
and the authored `dependency-plan.json`. Keep the candidate wheel outside that
archive. Regenerate or verify the hashes with:

```sh
python packages/kumo-engine/test/snowflake_notebook/dependency_inventory.py \
  --archive /path/to/wheelhouse.tar
python packages/kumo-engine/test/snowflake_notebook/dependency_inventory.py \
  --archive /path/to/wheelhouse.tar --check
```

The inventory generator does not resolve, install, upload or execute wheels.
Version updates are manual; the hashes are generated automatically.

The tests use existing dedicated resources in the test Snowflake account:

- `KUMO_TEST_DB.KUMO_TEST`, `KUMO_TEST_WH`, and `KUMO_TEST_POOL`.
- A writable internal stage and a separate read-only
  [denied stage](../../examples/mlops/snowflake/setup-denied-stage.sql).
- `KUMO_SNOWFLAKE_CI` with key-pair authentication and
  `KUMO_SNOWFLAKE_TEST_ROLE`, including access to the resources above and
  privileges to create test Code Bundles, experiments and models.
- `KUMO_SNOWFLAKE_UI_TEST` with password/TOTP login and access to the test
  runs, models and notebook service.

The graph runner selects the `notebook_service` from `SNOWFLAKE_TEST_CONFIG`
(default example: `KUMO_TEST_NOTEBOOK_SERVICE`)
on `KUMO_TEST_POOL (GEN_X64_G2_8)`, using CPU Python 3.12. The configured name selects a dedicated test notebook service. Selecting it explicitly prevents use of a shared service.

[resources.py](../../packages/kumo-engine/test/snowflake_notebook/resources.py)
defines the exact resource allowlist. Set `SNOWFLAKE_TEST_CONFIG` to the JSON
returned by its `expected()` function, then replace the placeholder `account`,
`account_locator`, and `account_url` with your dedicated test account before live
execution:

```sh
export SNOWFLAKE_TEST_CONFIG="$(python -c 'import json, sys; sys.path.insert(0, "packages/kumo-engine/test/snowflake_notebook"); import resources; print(json.dumps(resources.expected()))')"
```

For local runs, print the template, replace the account placeholders, and paste
the completed JSON on one line. Export it in the shell running the commands below:

```sh
printf '%s\n' "$SNOWFLAKE_TEST_CONFIG"
read -r SNOWFLAKE_TEST_CONFIG
export SNOWFLAKE_TEST_CONFIG
```

For Actions, store the completed JSON in the protected environment secret.

It includes hashes of the dependency inventory, staged dependency archive, and
[graph assets](../../packages/kumo-engine/test/snowflake_notebook/graph-assets.json)
manifests. Wheel changes do not require updating this secret. Actions reuses the shared Engine wheel workflow to build the dispatched commit,
stages the wheel under
a content-addressed cache, and supplies the resulting candidate manifest to both
scenarios. Unchanged dependency archives and fixture uploads are reused.

Before staging or submitting, the SQL connection must report the exact configured
`account_locator` through `CURRENT_ACCOUNT()`; a mismatch or failed check closes
the connection before any test writes.

For local authentication, keep the encrypted key outside Git and unlock it once:

```sh
python packages/kumo-engine/test/snowflake_notebook/key_cache.py --key-file /path/to/key.p8
```

The key cache lasts eight hours. Browser credentials go in an owner-only JSON
file, supplied with `--ui-credentials`, containing `account_url`, `user`,
`password` and `totp_secret`. The TOTP secret is the Base32 enrollment secret.
Alternatively, supply this JSON through `SNOWFLAKE_UI_CREDENTIALS`.

## Run locally

Run commands from the repository root. Use a new output directory each time.
Build and stage the current checkout first (native build prerequisites are in the
[setup guide](setup.md)):

```sh
packages/kumo-engine/scripts/build_wheel.sh --output-dir /tmp/snowflake-wheel-001
export SNOWFLAKE_TEST_CANDIDATE=/tmp/snowflake-candidate-001.json
python packages/kumo-engine/test/snowflake_notebook/staging.py \
  --wheel /tmp/snowflake-wheel-001/*.whl \
  --source-ref "$(git rev-parse HEAD)" --output "$SNOWFLAKE_TEST_CANDIDATE" \
  --key-file /path/to/key.p8
```

The staging command authenticates using the same environment/key cache as
submission, verifies the dependency cache, and uploads only a changed wheel.
The candidate manifest stays outside the repository. Both runners reject missing
provenance or a wheel built from different product sources.

**Engine lifecycle:** prepare the payload, then execute it inside Snowflake notebooks.

```sh
python packages/kumo-engine/test/snowflake_notebook/prepare.py \
  --output-dir /tmp/snowflake-payload-001
python packages/kumo-engine/test/snowflake_notebook/launch.py \
  --payload-manifest /tmp/snowflake-payload-001/manifest.json \
  --preflight-payload /tmp/snowflake-payload-001/preflight \
  --lifecycle-payload /tmp/snowflake-payload-001/lifecycle \
  --key-file /path/to/key.p8 --execute \
  --output-dir /tmp/snowflake-result-001
```

**Lifecycle UI only:** verify an existing lifecycle's rendered runs and versions.

```sh
python packages/kumo-engine/test/snowflake_notebook/launch.py \
  --execute --ui --ui-run /tmp/snowflake-result-001 \
  --ui-credentials /path/to/browser-credentials.json
```

**Notebook graph UI:** create and execute a graph notebook, then verify its saved output.

```sh
python packages/kumo-engine/test/snowflake_notebook/graph_notebook.py \
  --execute --key-file /path/to/key.p8 \
  --ui-credentials /path/to/browser-credentials.json \
  --output-dir /tmp/snowflake-graph-001
```

| Option | Purpose |
| --- | --- |
| Omit `--execute` | Write a plan without cloud execution. Lifecycle UI mode validates retained receipts. |
| `--output-dir DIR` | Save results to a new directory. |
| `--key-file FILE` | Select the locally cached encrypted RSA key. |
| `--ui-credentials FILE` | Select the browser credential file. |
| `launch.py --timeout SECONDS` | Notebook deadline; default 1,800 seconds. |
| `launch.py --ui --ui-run DIR` | Verify only the selected retained lifecycle in Snowsight. |
| `launch.py --ui-timeout SECONDS` | Browser wait setting; default 60 seconds. |
| `launch.py --show-result DIR` | Explain a saved result without authentication or execution. |
| `graph_notebook.py --verify-notebook FILE` | With `--execute`, recheck a retained `notebook.json` without running cells. Keep its `.ipynb` beside it. |
| `graph_notebook.py --source-ref REF` | Source revision whose graph modules must match the installed wheel; default `HEAD`. |

Use each script's `--help` for diagnostic and asset-preparation options.
Saved-output rechecks cover retained rendering and exports, not live controls.

## Run through GitHub Actions

In repository **Settings → Environments**, configure `snowflake-mlops-test`:

1. Allow `main` under deployment branch rules. During PR validation, also allow
   the exact trial branch; remove that rule after merging. Do not allow
   wildcard branches or tags.
2. Require `NVIDIA/kumo-engineers` as the only reviewer; disable admin bypass.
3. Add these environment secrets:

| Secret | Value |
| --- | --- |
| `SNOWFLAKE_TEST_CONFIG` | The resource/configuration JSON described above. |
| `SNOWFLAKE_PRIVATE_KEY` | Complete encrypted PEM private key. |
| `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE` | Key passphrase. |
| `SNOWFLAKE_UI_CREDENTIALS` | Browser credential JSON. |

Credentials are supplied through the protected environment, never workflow
inputs. GitHub runs reject local credential-file and key-cache fallback.
Browser artifacts require the current authenticated test page; session expiry
suppresses captures. Raw browser errors and authentication URLs are not retained.

Open **Actions → Snowflake notebook platform checks → Run workflow**, select
`main` (or the explicitly allowed trial branch during PR validation),
and choose `engine` or `graph-ui`. A maintainer
must approve the environment before credentials are available. The workflow is
manual-only. Lifecycle UI verification is currently a separate local command.

## Results

Read `summary.json` and the retained phase/UI evidence. Engine Actions runs also
write `ci-verdict.json`, separating the engine result from full qualification.
Missing prerequisites are blocked; assertion failures are failed. Evidence
includes wheel/helper hashes, execution IDs, logs, receipts, screenshots and
exports. UI rechecks reuse retained results instead of retraining.

GitHub uploads only an allowlisted aggregate result summary. Raw notebook outputs,
screenshots, accessibility snapshots, identities, and API responses remain private
to the runner. Candidate admission rejects missing complete-wheel build provenance
or product-source differences before notebook execution. Local testing uses the same
staging command and runtime manifest as Actions.

After collecting outputs, lifecycle submission removes its own Code Bundle and
submission-stage prefix. Graph UI execution stops its kernel, deletes the notebook
it created, collects exports, then removes its attempt output prefix. Read-only
inspection preserves existing notebooks. Unconfirmed final cleanup fails the
check. Lifecycle registered models, experiments, and model artifact roots remain
available for at least 24 hours for separate native MLOps UI inspection.
Before each dispatch, a bounded retention sweep expires at most 100 ledger-bound
attempts with matching terminal query history, object owners and creation times.
It removes only their models, experiments, Code Bundles and exact attempt paths.
Unknown or active queries and changed ownership preserve the resources. Attempts
outside the six-day proof window, historical runs without a ledger, and setup
interrupted before its ledger was written require private manual inspection. Shared wheel/fixture caches and compute stay in place.

The graph worker has a 90-minute outer deadline within a 140-minute Actions job.
An `always()` step opens a fresh authenticated browser to stop/delete the exact
prepared attempt if the first browser could not confirm cleanup. It never deletes
other notebooks or treats an unloaded file tree as proof of absence.

After the original browser closes, owned notebook deletion gets one fresh-browser
retry. It can recover only a cleanup failure; verification, export collection,
and browser shutdown failures remain failed. Failed collection preserves stage
evidence, and the original browser result remains available beside the retry proof.

To inspect retention without deleting anything:

```sh
python packages/kumo-engine/test/snowflake_notebook/retention.py --key-file /path/to/key.p8
```

Add `--execute` to apply the verified cleanup. Retention never removes the shared
wheel, dependency or fixture caches, or the configured warehouse/compute pool.

The dependency descriptor currently preserves an existing private-stage archive.
Its exact location is protected from retention even though it is under a historical
attempt prefix. No developer `/tmp` search is used to discover or trust old assets.

Keep notebook credentials separate from ordinary pull-request Snowflake CI.
The existing workflow uses `SNOWFLAKE_CI_USER`/`SNOWFLAKE_CI_PRIVATE_KEY` in a
different environment; do not give that user the notebook role or grants.

Before merging, enable `prevent_self_review` in every platform test environment.
The shared guard enforces independent approval on `main`; explicitly allowlisted
trial branches temporarily permit maintainer self-approval.
After an interrupted engine step, Actions attempts bounded recovery of the owned
query, Code Bundle and submission prefix. Unknown query state, missing ownership
proof, or provider failures preserve the attempt for the retention sweep.
