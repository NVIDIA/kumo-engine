# Snowflake notebook E2E acceptance plan

This is the Snowflake execution specification for the
[v1.1 product-validation assignment](https://docs.google.com/document/d/18Jt8Y_rvfpuCrWbwArtEyPdpQu6AiDLn3I-_HqNLyHA/edit).
The [Engine testing plan](https://docs.google.com/document/d/1YaUXDsA2mZgTrOEyhUh0c2q8V-A_WyVE7JPSFHwrbnU/edit)
remains the overall testing plan. Review the coverage inventory with An and
Vedaant before expanding the workload. This specification does not activate a
workflow, recurring schedule, or required PR check.

## Scope and execution contract

Qualify one identified bundled candidate wheel through a small MLOps lifecycle
executed in Snowflake Container Runtime notebooks. The external runner submits
`.ipynb` jobs; training, prediction and monitoring execute inside those notebooks.
Client-side lifecycle execution, standalone ML Jobs and generic SPCS scripts do
not qualify this profile.

Use the existing binary-task recipe and shared lifecycle assertions described in
[local validation](mlops-validation.md). Train exactly two small versions with
at least two observed metric epochs each. Reuse those versions and saved
predictions for recovery, UI and negative checks. Do not repeat task-quality,
performance or numerical reload campaigns already covered by An's task tests.
CPU is the initial profile; GPU qualification requires a separately identified
runtime and observed CUDA work. Record measured startup, fit, scoring, UI and
collection time before proposing deadlines or resource budgets.

Use the same `graph()` fixture and `fit()` implementation in
[wheel_mlops_lifecycle.py](../../packages/kumo-engine/test/wheel_mlops_lifecycle.py)
for local and Snowflake execution:

| Setting | Local and Snowflake CPU profile |
| --- | --- |
| Population | 60 users; every ordinary prediction covers all 60 IDs. |
| Source events | 1,500 deterministic generated events, filtered by each snapshot cutoff. |
| Snapshot cutoffs | Training: 2024-03-01; scoring: 2024-03-31; labels: 2024-05-01. |
| Task | Binary prediction of whether the next seven days contain an event. |
| Training budget | Two sequential fits, seeds 0 and 1; two epochs, two steps per epoch, batch size 32. |
| Recipe | Shared `wheel_gpu_smoke.py::make_plan('binary')` with the lifecycle overrides; no refits or parameter search. |

The Parquet fixture is generated inside the notebook. Keep the helper/fixture
hashes in the attempt manifest so a larger or different dataset cannot silently
reuse this resource profile. Snowflake internal-stage artifacts, Experiment
Tracking and Model Registry are live services. Snowflake table ingestion and notebook graph/explanation widgets
are separate coverage; neither is claimed by this scenario.

## Prerequisites and admission

Record these before submitting work:

- Candidate source revision, wheel filename/version/SHA-256, platform extra,
  Python/Container Runtime version, dependencies and successful `pip check`.
  Verify imported Engine/Core/DiskGraph/native files against the same wheel.
  Use a fresh notebook runtime; managed base packages and any dependency changes
  must be recorded. Source-tree imports and shared developer environments do not
  qualify the candidate.
- Approved harness revision and hashes, fixture/config hashes and unique attempt
  ID. Use the same wheel and harness bytes in every phase. Credentialed execution
  needs an approved dispatch path and a role scoped to the test resources.
- Existing test database, schema, query warehouse, compute pool, notebook project
  or Code Bundle, and `SNOWFLAKE_SSE` internal stage. Record the exact resources,
  grants and remote prefixes. No provisioning or automatic resource deletion.
- Notebook submission identity and observed notebook/worker account, user and
  role. Workers must reconstruct supported identity without copying an active
  Python session or embedding credentials in payloads. Reject identity mismatch
  before MLOps writes. The mounted-token path and headless notebook-context
  detection require a real preflight probe before lifecycle qualification.
- A separate authenticated Snowsight browser session able to view the same
  experiments and models. SQL authentication is not browser-login evidence.
- Writable evidence prefix and a separately existing denied stage. Stage grants
  apply to the entire stage; another prefix in the writable stage is not a
  permission-denial fixture. Give the CI role `READ` without `WRITE` on the
  admin-owned denied stage using
  [the fixture setup SQL](../../examples/mlops/snowflake/setup-denied-stage.sql).
  Missing credentials, grants, browser access or supported runtime are blockers.

Set `denied_artifact_root` to a prefix in that stage. Full platform runs check
permission denial before training. To check the fixture separately inside the
notebook, pass `phase='negative'` to `run_notebook()` with a new output directory.
This does not repeat fits or qualify the remaining lifecycle and UI requirements.

Install the candidate and pinned dependencies before executing notebook test
cells. An unattended test must not depend on an interactive pip/restart step.
The notebook must use explicit namespaces or fully qualified object names.

## Authentication contract

The runner uses the existing dedicated Snowflake CI service user with key-pair
authentication through the Python connector (`authenticator="SNOWFLAKE_JWT"`).
Local headless execution and worker identity checks have run on the test account.
The [Actions trial](snowflake-notebook-actions.md) requires separately configured
GitHub environment secrets; the local key cache is not transferred.
See [connector key-pair authentication](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#using-key-pair-authentication-and-key-pair-rotation).

The account owner registers the public key on that user and grants its test
role the required notebook/Code Bundle, compute pool, warehouse, stage,
experiment and registry permissions. The runner receives the encrypted private
key and passphrase from protected CI environment secrets. Account, username,
role, warehouse and test namespaces are explicit configuration. The trial decrypts the CI key in process and passes DER bytes to the connector;
it writes no decrypted key file. Never put it in the
notebook payload, logs, reports, browser artifacts or a checked-in connection
file. Record the connector version and its supported passphrase parameter.

| Connection | Authentication and purpose | Current state |
| --- | --- | --- |
| External runner → Snowflake | CI service user's key pair; stage the payload, submit/poll/cancel notebook jobs, and download evidence. Record `CURRENT_ACCOUNT()`, `CURRENT_USER()` and `CURRENT_ROLE()`. | Verified locally; GitHub credentials must be configured for the Actions trial. |
| Notebook/phase workers → Snowflake | Snowflake runtime-provided authentication reconstructed in each process; the CI private key is not forwarded. Verify account/user/role against the submitting identity and active notebook before writes. | Verified in local headless notebook runs, including fresh phase workers. |
| Browser → Snowsight | Separately authenticated browser test identity with visibility into the same test namespace; an approved login/session setup is required. | Browser identity/login mechanism and assertions are not implemented. SQL key-pair access does not establish this session. |

A missing runner identity/key/grant blocks submission. A notebook identity
mismatch blocks lifecycle writes. Missing browser login blocks UI qualification
even if every notebook/API assertion passes.

## Runner and notebook sequence

Here, automatic launch means that the runner submits the notebook and waits
without a person clicking **Run all**, installing packages or restarting a
kernel. Result collection means that each notebook saves checksummed receipts,
assertion results and logs to its evidence stage prefix, and the runner downloads
them to evaluate the check and retain CI artifacts. Collection also runs after
an unsuccessful notebook execution; it must not depend on a success-only cell.
The runner checks both the Snowflake execution status and the test report.

1. Stage an immutable payload containing the wheel, notebook, shared helpers,
   fixture configuration, dependency specification and checksums. Record a
   manifest binding the payload to the attempt and candidate revision.
2. Submit the notebook through Snowflake's headless notebook API and retain its
   query/execution identifier. The current API is
   [EXECUTE CODE BUNDLE](https://docs.snowflake.com/en/sql-reference/sql/execute-code-bundle)
   with a `.ipynb` entrypoint on a compute pool. The compatible
   [EXECUTE NOTEBOOK PROJECT](https://docs.snowflake.com/en/sql-reference/sql/execute-notebook-project)
   API also supports runtime, warehouse, dependency and argument configuration.
   Select and record the API supported by the target account.
3. Run the fit phase, persist its artifacts and receipts to the attempt's stage
   prefix, verify uploaded checksums, then end that notebook execution.
4. Start a fresh notebook execution for scoring, with an empty local work area.
   Reconstruct state only from staged configuration/receipts and durable model,
   snapshot and training-table exports. Do not carry `/tmp`, a Python object or
   the training kernel into scoring. Persist and verify its evidence.
5. Start fresh notebook executions for evaluation and negative cases using the
   same attempt and retained models. Each phase records its own execution ID,
   environment and model/run identities. No additional fit is permitted.
6. Collect all available evidence, including failed-phase logs. Run the separate
   browser assertions against the exact recorded experiment/run/model IDs.
7. Produce one requirement-level report. SQL execution success alone cannot
   produce an E2E pass. Upload phase receipts before returning a failed or
   blocked notebook result. Retain failed attempts separately from retries.

Use unique experiment/model names and artifact prefixes per attempt. Alias
movement is confined to that attempt's model. A presentation or readback retry
reuses matching retained artifacts and does not silently retrain. A failed
producer stays failed; a later attempt has a new identifier.

The external launcher, phase arguments, cross-job receipt restoration and
remote result collection are implementation work. The current notebook is an
interactive scaffold and is not an unattended execution contract.

## Required assertions and coverage inventory

Owner for the Snowflake/MLOps product behavior: Vedaant. Coordinate shared task
coverage with An. Every row needs a command or notebook phase, prerequisites,
execution identifiers, evidence location and remaining limitations in the final
report. The phases below are the planned interface, not implemented CLI flags.

| Case / phase | Required assertion | Reuse and current qualification |
| --- | --- | --- |
| Admission | Exact candidate bytes, compatible runtime/dependencies, supported notebook placement and matching identity before writes. | Candidate-file guards exist; live headless admission is blocked pending resources and probe. |
| Two versions / fit | Two real multi-epoch fits register distinct versions/bundles. Tracking run status, step/value history and registry bindings match each fit. | Shared lifecycle and Snowflake API readback exist; live notebook execution is blocked. |
| Exported history / fit, score | Exported bundles preserve the complete observed epoch history; fresh execution can read it and match the producing run. | Product export capability is not implemented in the current base; no metadata-only substitute. |
| Recovery / score | Fresh notebook restores snapshots, training table and both models after producer-local files are unavailable. | Local fresh-process assertions exist; separate-job transport/restoration is not implemented. |
| Resolution / score | Explicit A/B predictions record A/B. `champion` initially resolves A; after promotion the next prediction records B. Registry versions bind the expected bundle. | Shared identity assertions and Snowflake registry adapter exist; live execution is blocked. |
| Lineage / fit, score | Source fixture → training snapshot → training table → model version associations are preserved. Scoring snapshot is distinct and correctly recorded for predictions. | Shared lineage and training-table fingerprint assertions exist; live execution is blocked. |
| Population/schema / score | Complete expected IDs, finite valid outputs and explicit unknown-ID accounting. Prediction without MLOps preserves the ordinary batch schema. | Shared fixture/schema assertions exist; live execution is blocked. |
| Drift / evaluate | Saved prediction identity, training/scoring snapshots and recorded metric/run agree. Self-comparison is zero; a controlled changed-score distribution produces the independently expected nonzero drift. | Identity/self-comparison checks and platform readback exist; shifted-distribution sensitivity coverage is not implemented. Live execution is blocked. |
| Labels / evaluate | Later label snapshot evaluates the original saved predictions after alias movement. Label window/coverage and recomputed AUROC agree with recorded results. | Shared independent label join and metric calculation exist; live execution is blocked. |
| Runs/metrics UI / browser | Snowsight renders the expected training, prediction and monitoring runs and their recorded metrics, including epoch history. | API checks exist; browser assertions are not implemented. |
| Model versions UI / browser | Snowsight renders both recorded model versions in the correct test namespace. | Registry API checks exist; browser assertions are not implemented. |
| Failure cases / negative | Unknown IDs are accounted for; genuine provider denial is distinguished from transport/missing-resource errors; incomplete/interrupted exports are rejected with useful diagnostics. | Unknown-ID and provider-denial checks exist. Live denial is blocked; remote incomplete/interrupted-operation cases are not implemented. |

The drift case must admit a controlled shifted scoring fixture whose exported
score distribution actually differs from the reference. Reuse the retained
model; extra predictions add no fits. Recompute expected statistics from the
saved scores and documented bins, using the existing monitoring tests as the
starting point. Compare the calculated and tracked results. A constant-score
fixture cannot establish sensitivity and must not qualify this row. Do not
introduce a task-quality threshold or assume every input change moves scores.

Local headless runs exercised the installed candidate wheel in Snowflake
notebooks and retained phase receipts. Browser verification and full epoch-history
export remain incomplete, so these runs do not qualify the full plan.
Existing test sources and local validation commands are inventoried in the
[shared lifecycle guide](mlops-validation.md); the
[platform guide](mlops-platform-validation.md) contains the plan-only command.
The [Actions trial guide](snowflake-notebook-actions.md) documents the headless
launcher and the remaining CI prerequisites.

## UI verification

Headless notebook execution and browser verification are two stages of the same
attempt. Discover Snowsight navigation and stable selectors against the target
account, then bind assertions to receipt IDs, not the latest visible run.

Verify displayed run status/names, metric names and values at displayed precision,
multiple epoch points and both registry versions. Inspect prediction, drift and
evaluation runs separately when the backend emits separate runs. Preserve page
URLs, screenshots and browser traces with the assertions. Account for bounded UI
refresh delay; a missing/wrong element must fail once that deadline expires.
Do not invent routes or treat SQL/API readback as a rendered UI assertion.

Confirm the actual UI surfaces with Vedaant during the first live probe. Required
runs, metrics or versions missing from the product remain a qualification gap.
Lineage, alias resolution and exported history have explicit API/artifact checks;
additional UI assertions apply only to UI surfaces the product exposes. Broader
notebook graph/explanation visualization remains a separate assignment.

## Failure and result rules

- `passed`: the stated assertion ran on the identified wheel and target notebook
  runtime, with retained evidence. UI rows require actual browser evidence.
- `failed`: an executed assertion found incorrect behavior, including wrong
  versions, missing IDs/history, incorrect lineage/metrics or missing UI content.
- `blocked`: prerequisites or access prevent execution or collection. Missing
  evidence cannot become a pass; preserve diagnostics and the responsible phase.
- `not implemented`: required test coverage or product behavior is absent.

Exercise intentional faults only within the test attempt/prefix. Verify the
negative assertion detects the fault and emits the expected diagnostic. A known
missing feature is `not implemented`; an unexpected missing export during an
executed supported scenario is `failed`. Fault recovery must not consume a prior
success marker. Bound polling and cancellation, record the terminal state, and
collect partial evidence without starting replacement fits automatically.

Overall Snowflake qualification passes only when every required row passes.
Blocked and not-implemented rows keep the overall result non-passing. A future
PR check must propagate that result rather than only the notebook SQL status;
its required-check/trigger configuration is a separate explicit decision.

Retain the payload manifest, submitted SQL/arguments, execution IDs, wheel/test
hashes, observed identities/runtime/resources, phase receipts, raw logs,
checksummed artifact manifest, UI evidence and final coverage report. Exclude
credential values and browser authentication state from published artifacts.

## Implementation order

1. Review the inventory and exact initial runtime/access requirements. Resolve
   how the target account exposes headless notebook identity and UI login.
2. Implement and probe dependency preinstallation, submission, per-phase durable
   state, deadlines/cancellation and result retrieval, without training.
3. Execute the two-fit lifecycle once; retain and verify its artifacts.
4. Add browser assertions and safe remote fault cases; retain each failure.
5. Complete the same candidate attempt using the retained models/predictions,
   then publish its coverage evidence. Start a new fit attempt only if changed
   candidate bytes, fixture or producer configuration invalidate the retained
   state; readback/UI changes do not authorize replacement fits. Exported-history
   and other unresolved product gaps remain visible.

## Account and CI setup

These are operator setup instructions, not actions already performed. Reuse an
existing approved CI identity if one is available. Substitute your actual
account, repository, test database/schema, warehouse and compute pool.

1. Generate an encrypted RSA key locally. Both commands prompt for its
   passphrase; keep that passphrase out of shell arguments/history.

   ```sh
   umask 077
   openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 \
     -aes-256-cbc -out snowflake_ci_key.p8
   openssl pkey -in snowflake_ci_key.p8 -pubout -out snowflake_ci_key.pub
   ```

2. Have an authorized Snowflake administrator create the CI user/role, create
   any missing test resources, then grant access. Register the public key body
   without PEM header/footer or line breaks. Never paste the private key into Snowflake SQL.
   Run each statement separately and check its result before continuing.

   ```sql
   USE ROLE ACCOUNTADMIN;
   SELECT CURRENT_ROLE();

   CREATE ROLE IF NOT EXISTS KUMO_SNOWFLAKE_TEST_ROLE;
   CREATE USER IF NOT EXISTS KUMO_SNOWFLAKE_CI
     TYPE = SERVICE
     DEFAULT_ROLE = KUMO_SNOWFLAKE_TEST_ROLE;
   SHOW USERS LIKE 'KUMO_SNOWFLAKE_CI';

   ALTER USER KUMO_SNOWFLAKE_CI SET RSA_PUBLIC_KEY = '<public-key-body>';
   GRANT ROLE KUMO_SNOWFLAKE_TEST_ROLE TO USER KUMO_SNOWFLAKE_CI;
   ```

   Create missing resources before granting usage. The names below are
   placeholders; use the same names in creation, grants and runner configuration.
   `XSMALL` is the initial query warehouse size for this small fixture's SQL,
   tracking and registry operations. Training and prediction run on the compute
   pool. The example uses `GEN_X64_G2_8` (6 available vCPUs, 28 GiB memory on
   AWS/Azure) to leave room for the notebook kernel and fresh worker processes.
   Verify availability in the instance-family results first. On GCP, or where
   that family is unavailable, select an available x86 CPU family with at least
   those resources, such as `CPU_X64_M` when listed. See the
   [AWS](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/instance-families-aws),
   [Azure](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/instance-families-azure)
   and [GCP](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/instance-families-gcp)
   family specifications. Both compute resources start suspended and automatically
   resume when used.

   This is a conservative initial sizing proposal for the exact CPU fixture,
   not a measured minimum or completed Snowflake capacity qualification. Local
   process measurements do not include Snowflake notebook/kernel overhead. Verify
   kernel-plus-worker memory, scratch space and phase completion on the first
   live notebook attempt; retain the resource evidence before qualifying it.

   ```sql
   SHOW COMPUTE POOL INSTANCE FAMILIES;

   CREATE DATABASE IF NOT EXISTS KUMO_TEST_DB;
   CREATE SCHEMA IF NOT EXISTS KUMO_TEST_DB.KUMO_TEST;

   CREATE WAREHOUSE IF NOT EXISTS KUMO_TEST_WH
     WAREHOUSE_SIZE = 'XSMALL'
     AUTO_SUSPEND = 60
     AUTO_RESUME = TRUE
     INITIALLY_SUSPENDED = TRUE;

   CREATE COMPUTE POOL IF NOT EXISTS KUMO_TEST_POOL
     MIN_NODES = 1
     MAX_NODES = 1
     INSTANCE_FAMILY = GEN_X64_G2_8
     AUTO_SUSPEND_SECS = 300
     AUTO_RESUME = TRUE
     INITIALLY_SUSPENDED = TRUE;

   CREATE STAGE IF NOT EXISTS KUMO_TEST_DB.KUMO_TEST.TEST_STAGE
     ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
   ```

   `IF NOT EXISTS` leaves existing resources unchanged. If `KUMO_TEST_POOL` already
   exists, inspect it with `DESCRIBE COMPUTE POOL KUMO_TEST_POOL`; rerunning creation
   does not change its family. Verify existing settings, including stage encryption. See [warehouse creation](https://docs.snowflake.com/en/sql-reference/sql/create-warehouse),
   [compute-pool creation](https://docs.snowflake.com/en/sql-reference/sql/create-compute-pool)
   and [stage creation](https://docs.snowflake.com/en/sql-reference/sql/create-stage).
   Once the resources exist, apply these grants:

   ```sql
   GRANT USAGE ON DATABASE KUMO_TEST_DB TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   GRANT USAGE ON SCHEMA KUMO_TEST_DB.KUMO_TEST TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   GRANT USAGE, MONITOR ON WAREHOUSE KUMO_TEST_WH TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   GRANT USAGE ON COMPUTE POOL KUMO_TEST_POOL TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   GRANT READ, WRITE ON STAGE KUMO_TEST_DB.KUMO_TEST.TEST_STAGE
     TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   GRANT CREATE EXPERIMENT, CREATE MODEL ON SCHEMA KUMO_TEST_DB.KUMO_TEST
     TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   ```

   After the deployment owner has created the notebook Code Bundle, grant its
   execution privilege separately. Do not run this grant before that object
   exists:

   ```sql
   GRANT USAGE ON CODE BUNDLE KUMO_TEST_DB.KUMO_TEST.MLOPS_VALIDATION
     TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   ```

   Deployment/version updates need a separate owner role; the execution grant
   does not authorize changing the bundle. Dependency preinstallation may also
   reference an external access integration or artifact
   repository: grant `USAGE` on each referenced object. If the runtime references
   Snowflake secrets, grant `READ` on those secrets (`USAGE` is insufficient).
   Include the parent database/schema access for schema-level dependencies. See
   [execution privileges](https://docs.snowflake.com/en/sql-reference/sql/execute-code-bundle#access-control-requirements).
   For example, when the payload uses an existing PyPI access integration:

   ```sql
   GRANT USAGE ON INTEGRATION TEST_PYPI_EAI TO ROLE KUMO_SNOWFLAKE_TEST_ROLE;
   ```

   For a target account using the earlier Notebook Project grammar, use its
   corresponding object/grant syntax. See the
   [Code Bundle grants](https://docs.snowflake.com/en/developer-guide/code-bundles/code-bundles#access-control)
   and [experiment creation privilege](https://docs.snowflake.com/en/sql-reference/sql/create-experiment).

3. In the repository that will run the check, create a GitHub Actions environment
   named `snowflake-e2e` and limit it to the approved execution path. Add the two
   secrets below using GitHub Settings or the CLI. These commands assume the
   environment already exists; substitute `OWNER/REPO`.

   ```sh
   gh secret set SNOWFLAKE_PRIVATE_KEY --repo OWNER/REPO \
     --env snowflake-e2e < snowflake_ci_key.p8
   gh secret set SNOWFLAKE_PRIVATE_KEY_PASSPHRASE --repo OWNER/REPO \
     --env snowflake-e2e
   ```

   The second command prompts for the passphrase. Add environment variables for
   `SNOWFLAKE_ACCOUNT` (connector account identifier), `SNOWFLAKE_USER`
   (`KUMO_SNOWFLAKE_CI`), `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`, database, schema,
   compute pool, approved Container Runtime version, bundle name, writable stage
   prefix and a prefix in the separately denied stage. These names describe runner
   configuration; creating variables does not wire an Actions job automatically.

4. The launcher job must select `environment: snowflake-e2e`, receive the two
   secrets, create a temporary mode-0600 key file and connect with the pinned
   Snowflake Python connector using `SNOWFLAKE_JWT`. Start with a read-only
   `SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE()` probe, then verify
   stage and bundle visibility. Retain the observed identifiers. This connection
   submits notebook SQL and retrieves evidence; it does not execute the Engine
   lifecycle on the external runner. Remove the key file in job cleanup.

5. Arrange a separate approved Snowsight test-user/browser-session login with
   access to the test runs and models. This remains required for the UI phase;
   the SQL service user's key pair supplies no browser session. The browser
   identity's visibility grants must be verified against the produced objects.

The checked-in interactive notebook's manual installation/restart step must be
replaced by an unattended, pinned installation before using it for CI dispatch.
