# Snowflake notebook tests

Run the Snowflake scenarios from the
[product testing plan](https://docs.google.com/document/d/18Jt8Y_rvfpuCrWbwArtEyPdpQu6AiDLn3I-_HqNLyHA/edit)
locally or through GitHub Actions. Both use the same test scripts and installed
candidate wheel.

## What it tests

| Scenario | Checks |
| --- | --- |
| Engine lifecycle | Train/register two versions; reload in fresh processes; predict by version and alias; move the alias; verify tracking, lineage, prediction IDs/schema, drift and evaluation after labels arrive. Includes real denied artifact access; incomplete-artifact checks currently run only in the local profile. |
| Lifecycle UI | Open the retained lifecycle's runs and model versions in Snowsight; verify status, metrics and epoch history. Reuses existing results without training again. |
| Notebook graph UI | Render An's PR172 adtech fixture in a Snowflake notebook; verify schema, relationships, tooltips and controls; reopen saved outputs after kernel shutdown; verify SVG/PDF/Markdown exports. Default display must read no source data, and exports must exclude sample values. |

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
python -m pip --isolated install snowflake-connector-python cryptography
npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser
node packages/kumo-engine/test/snowflake_notebook/browser/node_modules/playwright/cli.js install chromium
```

Chromium's system libraries must already be available locally. GitHub Actions
installs them with Playwright's `install --with-deps chromium` command.

The tests use existing dedicated resources in the test Snowflake account:

- `KUMO_TEST_DB.KUMO_TEST`, `KUMO_TEST_WH`, and `KUMO_TEST_POOL`.
- A writable internal stage and a separate read-only
  [denied stage](../../examples/mlops/snowflake/setup-denied-stage.sql).
- `KUMO_SNOWFLAKE_CI` with key-pair authentication and
  `KUMO_SNOWFLAKE_TEST_ROLE`, including access to the resources above and
  privileges to create test Code Bundles, experiments and models.
- `KUMO_SNOWFLAKE_UI_TEST` with password/TOTP login and access to the test
  runs, models and notebook service.

The graph runner currently selects the existing service `EGU_GRAPH_UI_SERVICE`
on `KUMO_TEST_POOL (GEN_X64_G2_8)`, using CPU Python 3.12. This name is a fixed
test-resource setting in the current implementation, not a Snowflake requirement
or a CLI option. Selecting it explicitly prevents use of a shared service.

[resources.py](../../packages/kumo-engine/test/snowflake_notebook/resources.py)
defines the exact resource allowlist. Set `SNOWFLAKE_TEST_CONFIG` to the JSON
returned by its `expected()` function before live execution:

```sh
export SNOWFLAKE_TEST_CONFIG="$(python -c 'import json, sys; sys.path.insert(0, "packages/kumo-engine/test/snowflake_notebook"); import resources; print(json.dumps(resources.expected()))')"
```

It includes hashes of the [candidate](../../packages/kumo-engine/test/snowflake_notebook/candidate.json)
and [graph assets](../../packages/kumo-engine/test/snowflake_notebook/graph-assets.json)
manifests. The tests use that staged wheel and reuse unchanged uploads; they do
not build a wheel from the current checkout.

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

1. Allow only `egu/mlops-platform-integration` under deployment branch rules.
2. Require `NVIDIA/kumo-engineers` as the only reviewer; disable admin bypass.
3. Add these environment secrets:

| Secret | Value |
| --- | --- |
| `SNOWFLAKE_TEST_CONFIG` | The resource/configuration JSON described above. |
| `SNOWFLAKE_PRIVATE_KEY` | Complete encrypted PEM private key. |
| `SNOWFLAKE_PRIVATE_KEY_PASSPHRASE` | Key passphrase. |
| `SNOWFLAKE_UI_CREDENTIALS` | Browser credential JSON. |

Open **Actions → Snowflake notebook platform checks → Run workflow**, select
`egu/mlops-platform-integration`, and choose `engine` or `graph-ui`. A maintainer
must approve the environment before credentials are available. The workflow is
manual-only. Lifecycle UI verification is currently a separate local command.

## Results

Read `summary.json` and the retained phase/UI evidence. Engine Actions runs also
write `ci-verdict.json`, separating the engine result from full qualification.
Missing prerequisites are blocked; assertion failures are failed. Evidence
includes wheel/helper hashes, execution IDs, logs, receipts, screenshots and
exports. UI rechecks reuse retained results instead of retraining.
