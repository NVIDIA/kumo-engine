# Databricks notebook platform tests

These tests run the same graph scenario as the
[Snowflake notebook tests](snowflake-notebook-actions.md), using An's unchanged
PR172 adtech fixture and an identified candidate wheel. Graph checks do not train
a model; the separate lifecycle scenario performs two small fits.

The runner stages checksum-addressed assets in a dedicated test volume, imports a
new notebook, submits it through the Jobs API, and polls for completion. The notebook
installs the wheel with its Databricks extra, restarts Python, verifies the installed
code, and calls the shared `graph_fixture.exercise()` with `IPython.display.display`.

The kernel checks all nine tables, 52 columns and ten relationships. Default display
must read no source data; explicit previews must use bounded real fixture samples.
SVG, PDF and Markdown exports must exclude sample values. The runner verifies each
output against the receipt printed by the successful notebook execution. After job
success, it polls for persisted cell outputs with a 120-second budget; individual
API requests have their own network timeout.

Playwright opens the actual Jobs HTML export without a Databricks browser login.
It reuses Snowflake's assertions for graph geometry, PK/FK tooltips, preview values,
zoom, layout and export commands. A fresh browser context reopens the same export
and checks retained schema and previews. Screenshots, receipts, outputs and assertions
are saved in the output directory. This does not check live workspace-editor
reopening, native experiment/registry pages, or complete dependency
compatibility. Pip logs remain evidence even when they report dependency conflicts.

## Authentication and test resources

Only `KUMO_DATABRICKS_TEST_PAT` is a secret. The workspace URL and dedicated test
resource names are fixed in
[client.py](../../packages/kumo-engine/test/databricks_notebook/client.py);
no configuration JSON or CONFIG secret is required. The runner does not read shell
startup files, use the generic developer PAT, or pass credentials into the notebook.
The notebook accesses its test volume through the Databricks runtime identity.

The existing test catalog contains a schema and volume named like the Snowflake resources:

| Resource | Name |
| --- | --- |
| Unity Catalog catalog | `kumo_test_catalogue` |
| Schema | `KUMO_TEST` |
| Managed volume | `TEST_STAGE` |
| Asset/output prefix | `/Volumes/kumo_test_catalogue/kumo_test/test_stage/mlops-validation/notebook-ui` |
| Notebook folder | `/Shared/kumo_test_catalogue/KUMO_TEST/notebook-ui` |

Provision this dedicated volume and grant the test identity access before execution.
Previous local trial artifacts remain in their original volume; moving to these
names requires seeding the pinned wheel and fixture into the new volume once.
The runner never silently falls back to the old namespace or customer resources.

Use the GitHub environment `databricks-mlops-test`, require approval from
`NVIDIA/kumo-engineers`, disable administrator bypass, and allow only branch
`egu/databricks-ui-validation`. Its sole required secret is
`KUMO_DATABRICKS_TEST_PAT`, containing the dedicated test identity's PAT.
The PAT identity needs Jobs submission/export, workspace notebook import and
read/write access to the dedicated Unity Catalog volume. GitHub's built-in token
checks environment protections before approval and again before consuming the PAT.
The cloud jobs accept manual dispatch only on the feature branch. A branch-scoped
push event registers the workflow in GitHub; both jobs are skipped for pushes.

By default, the runner uses serverless environment 2. Serverless can select ARM64;
the notebook rejects that architecture before installation. Databricks may
automatically retry on x86, but success is not guaranteed. For predictable local
execution, `--cluster-id ID` selects a running dedicated single-node `KUMO_TEST_`
cluster using `i3.xlarge` or `i3.2xlarge`, with attach and volume access. The runner
never creates compute, attaches to customer/shared clusters, or stops a supplied cluster.

## Run locally or in GitHub Actions

Use Python 3.12 and Node.js 22. The submission client uses the Python standard library.
Install the same locked browser tools used by Snowflake:

```sh
npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser
node packages/kumo-engine/test/snowflake_notebook/browser/node_modules/playwright/cli.js install chromium
```

Supply `KUMO_DATABRICKS_TEST_PAT` securely in the local process environment, then run:

```sh
python packages/kumo-engine/test/databricks_notebook/run.py \
  --execute --output-dir /tmp/databricks-graph-001
```

Omit `--execute` to write a plan without credentials or cloud access. Use a new output
directory for each run. `--timeout` controls polling (default 2,400 seconds); the
notebook task has a 1,800-second deadline. The runner cancels its submitted run if
polling fails or times out.

The candidate wheel and fixture must already be in the configured volume's
`payload-cache/<sha256>/<filename>` paths. To seed missing assets, pass `--wheel`
and `--fixture` pointing to the exact files pinned by
[the Databricks candidate manifest](../../packages/kumo-engine/test/databricks_notebook/candidate.json)
and [An's input manifest](../../packages/kumo-engine/test/pr172/inputs.json).
Existing assets are hash-verified and reused without upload. The test helpers are
uploaded only when their contents change. Asset bytes are checked before installation
and again inside the notebook; installed graph modules must match the test revision.

GitHub Actions runs this same command after environment approval. It requires the
PAT secret and the published feature branch; resource settings come from the code.


## Lifecycle and Snowflake parity

The lifecycle notebook reuses `wheel_mlops_lifecycle.py`,
`wheel_mlops_platform.py`, and An's `wheel_gpu_smoke.py` recipe. Its fixture is
exactly the shared Snowflake lifecycle fixture: 60 users, 1,500 seeded events,
and two two-epoch binary fits. The graph scenario separately uses An's PR172 data.

Each lifecycle attempt uses a new experiment, fully qualified Unity Catalog model,
and Volume artifact prefix. The notebook creates a clean virtual environment,
uses `venv --without-pip` and pip’s [target-interpreter option](https://pip.pypa.io/en/stable/topics/python-option/)
to install the pinned candidate with its Databricks and MLflow extras, and requires
`pip check` to pass. The Databricks extra includes MLflow’s Unity Catalog artifact
storage dependencies. Each phase runs in a fresh process using runtime-derived
credentials, verifies its effective Databricks identity, and checks its installed wheel.
The runner PAT is never copied into the notebook. Producer scratch and source data
are removed before recovery phases. The same process-group deadline handling as the
shared runner stops timed-out workers and their descendants.

```sh
python packages/kumo-engine/test/databricks_notebook/lifecycle_notebook.py \
  --execute --cluster-id DEDICATED_TEST_CLUSTER_ID \
  --output-dir /tmp/databricks-lifecycle-001
```

| Scenario | Databricks coverage |
| --- | --- |
| Graph schema, tooltips, controls, samples and exports | Passed in GitHub Actions [42198152316](https://github.com/NVIDIA/kumo-engine/actions/runs/42198152316). |
| Exported notebook reopened in a fresh browser context | Passed in the same run; does not establish native editor reopening. |
| Fit/register two versions; reload, predict, aliases, lineage, drift and delayed labels | Passed in live notebook run `7236256940191` with the pinned candidate; fit, score and evaluate each exited zero. |
| Real provider-denied artifact access | Requires a deliberately read-only test root; never inferred from a missing path. |
| Native notebook editor, kernel shutdown/reopening, experiment and registry UI | Not qualified by PAT/API or exported HTML evidence. |
| Full exported epoch history, independently nonzero drift, interrupted remote writer | Existing shared qualification gaps remain explicit. |

Native Databricks web pages require a separate browser session; the workspace
redirects PAT-authenticated page requests to login. Under PAT-only authentication,
these native UI checks remain blocked. Do not replace them with a custom test dashboard
or label an API readback as native UI evidence. The lifecycle summary separates
positive API checks, all four engine phases, and full qualification.

GitHub Actions offers `graph-ui`, `engine`, and `all` modes. Lifecycle modes require
the ID of a running dedicated `KUMO_TEST_` x86 cluster. The engine job requires all
four API phases, including real denied access; it preserves native-UI and export
qualification gaps even when those selected engine checks pass. No email login is
part of the PAT notebook runner.


The Databricks candidate includes the committed Volume directory-path fix from
`20110186505aac909f385088c9bd2a4e39823e1c`. The original Snowflake candidate failed
training-table export on Databricks with `Bad Request`; that failure remains retained.
The candidate manifest records the base wheel hash, committed Engine/provider sources,
and changed modules. The normal component and merged-wheel builders produce a new
version/hash with current Engine and Databricks Python code; native payload bytes
remain unchanged.
Snowflake's candidate pin is separate. Training-table recovery also uses the Files
API and shared Parquet reader for immutable Volume exports. Graph evidence from
an earlier candidate does not qualify a new candidate until rerun.
