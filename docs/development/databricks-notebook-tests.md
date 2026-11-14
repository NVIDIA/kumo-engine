# Databricks notebook platform tests

These tests run the same graph scenario as the
[Snowflake notebook tests](snowflake-notebook-actions.md), using the unchanged shared
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

`KUMO_DATABRICKS_TEST_PAT` and `KUMO_DATABRICKS_TEST_HOST` come from protected
environment secrets. The host is the HTTPS test workspace origin. Dedicated test
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
names requires seeding the wheel and fixture into the new volume once.
The runner never silently falls back to the old namespace or customer resources.

Use the GitHub environment `databricks-mlops-test`, require approval from
`NVIDIA/kumo-engineers`, disable administrator bypass, and allow `main`. During PR validation, also allow
`egu/databricks-ui-validation`; remove that rule after merging. Wildcard branch
rules and tags are rejected. Its required secrets are
`KUMO_DATABRICKS_TEST_PAT`, containing the dedicated test identity's PAT, and
`KUMO_DATABRICKS_TEST_HOST`, containing its HTTPS workspace origin.
Both jobs require `KUMO_DATABRICKS_TEST_USER`, identifying the expected PAT
principal. Retention checks the authenticated identity before writing or recovering
control records; the lifecycle checks it again inside every fresh worker.
The PAT identity needs Jobs submission/export, workspace notebook import and
read/write access to the dedicated Unity Catalog volume. GitHub's built-in token
checks environment protections before approval and again before consuming the PAT.
The cloud jobs accept manual dispatch on `main` or the platform feature branch
during PR validation. Push and pull-request events do not launch cloud jobs.

Every scenario requires `--cluster-id ID` for an existing dedicated single-node
`KUMO_TEST_` x86 cluster using `i3.xlarge` or `i3.2xlarge`, with attach and Volume
access. This matches the complete wheel’s architecture; serverless can select ARM.
The runner validates the cluster ID, name, worker count, node type and effective
driver type before assets and again before submission. It accepts running,
starting, restarting or terminated test compute. [Databricks Jobs automatically
starts terminated assigned compute](https://docs.databricks.com/aws/en/compute/clusters-manage#compute-autostart-for-jobs-and-jdbcodbc-queries),
so the cluster’s idle shutdown does not require a manual restart after approval.
The existing bounded Job/task deadlines cover startup and execution. The runner
never creates, resizes or stops supplied compute and calls no cluster-start API.
Actions allows 110 minutes for graph checks and 130 minutes for the lifecycle,
including submission, execution, collection and cleanup.

## Run locally or in GitHub Actions

Use Python 3.12 and Node.js 22. The submission client uses the Python standard library.
Install the same locked browser tools used by Snowflake:

```sh
npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser
node packages/kumo-engine/test/snowflake_notebook/browser/node_modules/playwright/cli.js install chromium
```

Supply `KUMO_DATABRICKS_TEST_PAT` and `KUMO_DATABRICKS_TEST_HOST` securely in
the local process environment. Build the complete wheel from this checkout and create
a runtime manifest outside the repository:

```sh
packages/kumo-engine/scripts/build_wheel.sh --output-dir /tmp/databricks-wheel
python packages/kumo-engine/test/platform_candidate.py \
  --wheel /tmp/databricks-wheel/*.whl --source-ref "$(git rev-parse HEAD)" \
  --output /tmp/databricks-candidate.json
export DATABRICKS_TEST_CANDIDATE=/tmp/databricks-candidate.json
```

Then run:

```sh
python packages/kumo-engine/test/databricks_notebook/run.py \
  --execute --cluster-id DEDICATED_TEST_CLUSTER_ID \
  --wheel /tmp/databricks-wheel/*.whl \
  --output-dir /tmp/databricks-graph-001
```

Omit `--execute` to write a plan without credentials or cloud access. Use a new output
directory for each run. `--timeout` controls polling (default 2,400 seconds); the
notebook task has a 1,800-second deadline. The runner cancels its submitted run if
polling fails or times out.

GitHub builds the complete wheel from the dispatched `github.sha`, downloads that
same run's artifact, and generates the same runtime manifest. Both cloud jobs
receive that manifest and wheel; no committed historical pin needs refreshing.
Admission checks the build source against the current product files before cloud
access. The graph fixture must already be in its checksum-addressed cache, or local
runs can seed it with `--fixture` using
[the shared input manifest](../../packages/kumo-engine/test/pr172/inputs.json).
Existing assets are hash-verified and reused without upload. The test helpers are
uploaded only when their contents change. Asset bytes are checked before installation
and again inside the notebook; installed graph modules must match the test revision.

GitHub Actions runs this same command after environment approval. It requires the
protected PAT and host secrets. Select `main` after merging, or the platform
feature branch during PR validation; resource names come from the code.


## Lifecycle and Snowflake parity

The lifecycle notebook reuses `wheel_mlops_lifecycle.py`,
`wheel_mlops_platform.py`, and the shared `wheel_gpu_smoke.py` recipe. Its fixture is
exactly the shared Snowflake lifecycle fixture: 60 users, 1,500 seeded events,
and two two-epoch binary fits. The graph scenario separately uses the shared PR172 data.

Each lifecycle attempt uses a new experiment, fully qualified Unity Catalog model,
and Volume artifact prefix. Durable snapshots, training tables, models and
predictions use the existing `FileConnector` through the notebook's mounted
`/Volumes` filesystem. Tracking and version/alias loading still use native
Databricks MLflow and Unity Catalog. Direct Volume Parquet ingestion through
`DatabricksConnector` is outside this scenario.
The notebook creates a clean virtual environment,
uses `venv --without-pip` and pip’s [target-interpreter option](https://pip.pypa.io/en/stable/topics/python-option/)
to install the candidate with its Databricks and MLflow extras, and requires
`pip check` to pass. The explicit `mlflow,s3` extras include MLflow’s Unity Catalog artifact
storage dependencies. Each phase runs in a fresh process using runtime-derived
credentials, verifies its effective Databricks identity, and checks its installed wheel.
The runner PAT is never copied into the notebook. Producer scratch and source data
are removed before recovery phases. The same process-group deadline handling as the
shared runner stops timed-out workers and their descendants.

```sh
python packages/kumo-engine/test/databricks_notebook/lifecycle_notebook.py \
  --execute --engine-checks-only --cluster-id DEDICATED_TEST_CLUSTER_ID \
  --wheel /tmp/databricks-wheel/*.whl \
  --output-dir /tmp/databricks-lifecycle-001
```

| Scenario | What is checked |
| --- | --- |
| Graph | Schema, PK/FK tooltips, controls, bounded samples and exports in the actual Jobs HTML output. |
| Saved output | The same exported notebook reopened in a fresh browser context. |
| Lifecycle | Two registered versions; reload, predict, aliases, lineage, drift and delayed labels in four isolated phases. |
| Permission failure | Actual provider write denial on the isolated read-only test Volume. |
| Remaining gaps | Native notebook editor/kernel reopening and experiment/registry UI; complete epoch exports, independently nonzero drift and interrupted remote writer. |

`--engine-checks-only` requires all four engine phases and returns zero when they
pass. The summary's `verdict.engine_checks` records that result; its top-level
status retains the separate full-qualification gaps listed above.

The permission-negative target is fixed at
`/Volumes/kumo_test_denied/kumo_test/test_denied/mlops-validation/lifecycle`.
The runner reads the `TEST_DENIED` Volume metadata and lists its root before
submitting the notebook. Missing or unreadable resources are recorded as setup
blocks, while the positive lifecycle still runs. An empty readable Volume is
valid; the notebook appends its own attempt prefix and must receive an actual
provider write denial. A writable target fails the negative test.

Use a separate `kumo_test_denied` catalog for this target. The training catalog
`kumo_test_catalogue` is shared and grants `WRITE VOLUME` to `account users`;
its inherited write access would defeat a read-only Volume in that catalog.
The runner keeps the existing training resources and grants separate.

In the workspace SQL editor, use an administrator identity other than
`TEST_PRINCIPAL`, with `CREATE CATALOG` on the metastore. Run once:

```sql
CREATE CATALOG kumo_test_denied;
CREATE SCHEMA kumo_test_denied.kumo_test;
CREATE VOLUME kumo_test_denied.kumo_test.test_denied;

GRANT USE CATALOG ON CATALOG kumo_test_denied
TO `TEST_PRINCIPAL`;
GRANT USE SCHEMA ON SCHEMA kumo_test_denied.kumo_test
TO `TEST_PRINCIPAL`;
GRANT READ VOLUME ON VOLUME kumo_test_denied.kumo_test.test_denied
TO `TEST_PRINCIPAL`;

SHOW GRANTS ON CATALOG kumo_test_denied;
SHOW GRANTS ON SCHEMA kumo_test_denied.kumo_test;
SHOW GRANTS ON VOLUME kumo_test_denied.kumo_test.test_denied;
```

The administrator retains ownership; the test identity receives usage/read access
without effective `WRITE VOLUME` or Volume ownership. The metastore has default
managed storage, so these objects require no new storage location. Existing
`kumo_test_catalogue` permissions stay in place. No additional secret or runner
configuration argument is needed after provisioning.

Native Databricks web pages require a separate browser session; the workspace
redirects PAT-authenticated page requests to login. Under PAT-only authentication,
these native UI checks remain blocked. Do not replace them with a custom test dashboard
or label an API readback as native UI evidence. The lifecycle summary separates
positive API checks, all four engine phases, and full qualification.

GitHub Actions offers `graph-ui`, `engine`, and `all` modes. All modes require
the ID of a dedicated single-node `KUMO_TEST_` x86 cluster that can be assigned to notebook Jobs. The engine job requires all
four API phases, including real denied access; it preserves native-UI and export
qualification gaps even when those selected engine checks pass. No email login is
part of the PAT notebook runner.


The candidate manifest records the selected wheel filename, SHA-256, expected
version and source commit. Each dispatch builds that commit; graph modules are
also compared with the committed sources. Earlier candidate evidence does not
qualify a new wheel until rerun.

Candidate wheels live below the admitted attempt prefix and are removed with
that attempt. Small shared fixtures and helper files remain checksum-addressed.
Both jobs run bounded recovery even after cancellation: recovery validates the
ledger scope and run owner/path before cancelling and waiting for termination.
An unknown submission result or unprovable ownership preserves resources.
Definitive rejected submissions release owned resources and retain the API error.

Lifecycle execution uses the shared phase runner (negative first), clean-venv
admission and 600-second phase bounds. The 5,400-second Jobs limit includes
installation, execution and evidence-publication margin; the controller allows
6,000 seconds and the Actions job reserves further recovery time.

GitHub uploads only allowlisted aggregate summaries. Raw notebook output, browser
captures, identities, and API responses stay private to the runner. The candidate
check requires complete-wheel `build.source_commit` provenance matching the
selected product sources before credentials are loaded.

After evidence and UI collection, the runner removes its generated workspace
notebook, exact attempt Volume prefixes, registered model and versions, and
soft-deletes its tagged experiment and runs. Shared caches, supplied compute,
parent folders, and job history remain. Unknown submission/import outcomes fail
cleanup and preserve ambiguous resources for investigation.


## Recovery after a lost runner

After checking that an attempt's paths do not exist, the runner creates a private
control record at `notebook-ui/retention/<attempt>/resources.json` in the test
Volume. It checkpoints the exact scope, creation/removal flags and submitted Job
receipt before continuing. Successful cleanup removes the record. The same
configuration enables this for local execution and Actions; the PAT must match
`KUMO_DATABRICKS_TEST_USER`.

Each approved manual dispatch sweeps old records before notebook execution. In
`all` mode the graph job performs the single sweep. There is no scheduled cloud
job. Records must be at least 24 hours old; discovery is limited to 200 records
and ten pages, cleanup to 20 attempts and 10,000 files per invocation. An inventory
above the discovery limit requires inspection rather than an unbounded sweep.
Recovery reconstructs local state from the Volume and reuses the exact run-owner,
notebook-path, experiment-tag and model/version checks. An owned active Job is
cancelled and must terminate before resources are removed. Unknown submission or
creation results, foreign identities, missing Job history and changed records
are preserved for inspection. Shared payload-cache files and supplied compute
are outside the cleanup scope. Control records and raw API receipts are not
uploaded as public Actions artifacts.

To inspect the plan locally, then execute the same bounded recovery:

```sh
python packages/kumo-engine/test/databricks_notebook/retention_ledger.py \
  --output-dir /tmp/databricks-retention-plan
python packages/kumo-engine/test/databricks_notebook/retention_ledger.py \
  --execute --output-dir /tmp/databricks-retention-execution
```

Control records carry an HMAC-SHA256 signature derived from the runner's PAT;
the PAT is never stored in the record or sent to the notebook. A different Volume
writer cannot forge ownership records. Rotating the PAT makes existing signatures
unverifiable: those attempts are preserved for operator inspection, rather than
adopting the new key. A submission whose response and durable receipt were both
lost cannot be safely identified by name alone and also requires inspection.

If resource cleanup succeeds but deleting the empty control directory fails,
`cleanup.json` records `control_directory_cleanup: pending`. The next sweep
removes only an empty directory with no pagination token; it does not count
that as proof of attempt-resource cleanup. Nonempty unsigned directories remain.
