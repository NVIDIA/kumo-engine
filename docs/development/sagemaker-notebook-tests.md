# SageMaker notebook platform tests

The runner submits a SageMaker `NotebookJobStep` in a Pipeline. SageMaker runs
the notebook as a managed training job; the runner collects its executed
notebook, evidence files, and browser results. Local and GitHub execution use
the same entrypoint and resource configuration.

## Notebook execution

The initialization script creates a clean virtual environment, installs the
pinned wheel with `[sagemaker]`, checks its dependencies, and registers the
notebook kernel. Kernel tools are installed separately from the product.
The runner caches inputs in S3 by SHA256 and checks S3 checksums before reuse.

The notebook verifies its actual execution role, training-job identity, image,
kernel, mounted input notebook, architecture, Python, and glibc. Each lifecycle
subprocess repeats the execution-identity check. A local process with copied
environment variables cannot qualify as a managed notebook run.

The controller publishes the job ARN from the Pipeline service response to an
immutable, attempt-specific S3 receipt. The kernel checks that receipt and
independently describes the job and its tags. The execution role does not need
permission to list other training jobs.

The `engine` scenario reuses `wheel_mlops_lifecycle.py` and An's small training
recipe: 60 users, 1,500 seeded events, and two fits of two epochs each. Four
fresh processes run the permission negative, training, scoring, and evaluation
phases. They check two registered versions, reloads, predictions by explicit
version and moved alias, lineage, tracking, artifacts, and monitoring. The
scoring process runs after the producer's scratch and source-data directories
have been removed.

The `graph` scenario reuses the unchanged PR172 adtech fixture and the
Snowflake graph helper. It checks the schema and relationships, instruments
default rendering for sample/backend reads, exercises explicit sample previews,
and validates sample-free exports. `all` runs both scenarios in one notebook.

## Evidence and UI

The executed notebook prints a receipt that binds its attempt, source commit,
wheel hash, source hashes, output prefix, and evidence hashes. The collector
checks the submitted code against the executed cells before accepting the
receipt. Modified files, duplicate receipts, cell errors, missing phases, and
nonzero worker exits cannot qualify.

The browser verifier reuses the graph assertions on the exported notebook,
closes the page, and opens it again. Managed MLflow UI checks use a one-use
presigned URL supplied over stdin to Playwright. They inspect the two fit runs,
four prediction runs, and two model versions. Drift and evaluation update the
two alias prediction runs; the checker compares their final updated metrics.
Training history checks open each metric in a fresh page, compare its UI CSV
download with recorded epochs, and inspect visible hover values and positions
against rendered axis labels. This works with managed MLflow's WebGL charts.
Authentication URLs, browser sessions, and traces are
not retained. This covers the managed MLflow registry; automatic synchronization
to SageMaker Model Registry is disabled on the existing server and is not
claimed by this test.

Native Studio notebook reopening remains a separate requirement. Exported HTML
does not establish live graph controls or Studio coverage. Full qualification
also requires nonzero drift sensitivity, active remote-writer interruption, and
full epoch-history exports; the current harness records these as outstanding.
`--engine-checks-only` explicitly selects a
smaller execution check and does not establish full qualification.

Each attempt retains the submitted notebook, initialization script, Pipeline
definition and execution, training-job description, executed notebook, installed
wheel metadata, worker logs, phase JSON, graph artifacts, browser screenshots,
and summary. Failed jobs retain available job and notebook evidence. Test
models and cloud resources are retained for inspection.

## Test resources and authentication

Configuration is in
[`config.py`](../../packages/kumo-engine/test/sagemaker_notebook/config.py).
The suite uses account `926922431314`, region `us-west-2`, the existing
`kumo-test-dataplane-bucket/mlops-validation/sagemaker/` prefix,
`kumo-mlops-validation` managed MLflow server, and
`kumo-mlops-validation-sagemaker` execution role. Each job has a unique
`kumo-test-notebook-<attempt>` name and writes to its own prefix.

GitHub authenticates through OIDC using the existing
`gha-kumo-workflow-tests` role. No AWS access-key secret is needed. The manual
workflow requires the `sagemaker-mlops-test` environment, maintainer team
approval, disabled administrator bypass, and the exact feature branch. The
environment is checked before and after approval. Pushes register the workflow
without launching cloud jobs.

The existing roles need additional permissions for notebook jobs, passing the
execution role, the selected image, runtime verification, and browser access.
`prepare_access.py` generates a portable ZIP without contacting AWS. Its
application script uses AWS CLI and jq to fetch and preserve the current trust
policy, refusing intervening changes. An AWS admin can review and apply the
bundle in CloudShell without Python or a repository checkout. Those additions
enable GitHub Actions; they do not grant
the local Engineer SSO role permission to pass the execution role.

The permission negative uses an existing readable sentinel at
`mlops-validation/sagemaker-denied/README.txt` and requires an actual provider
write denial. This dedicated sibling prefix is outside the execution role's
writable prefix. Launcher readability proves the object exists; the notebook's
write attempt proves denial. A missing object or local permission failure is
not a pass.

## Running

Install host tools into a dedicated virtual environment:

```sh
python3.11 -m venv /tmp/kumo-sagemaker-runner
/tmp/kumo-sagemaker-runner/bin/pip install \
  -r packages/kumo-engine/test/sagemaker_notebook/requirements.txt
npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser
node packages/kumo-engine/test/snowflake_notebook/browser/node_modules/playwright/cli.js install chromium
```

Commit the exact test inputs before execution. The candidate manifest identifies
the wheel filename, SHA256, version, and main build artifact. The fixture
manifest is `packages/kumo-engine/test/pr172/inputs.json`.

```sh
AWS_PROFILE=<authorized-test-profile> \
  /tmp/kumo-sagemaker-runner/bin/python \
  packages/kumo-engine/test/sagemaker_notebook/run.py \
  --execute --scenario all \
  --wheel /path/to/pinned.whl \
  --fixture /path/to/pr172-inputs-v1.tar.gz \
  --output-dir /tmp/kumo-sagemaker-attempt-unique
```

Omit `--execute` to inspect the configuration without contacting AWS. The output
directory must be new. Omit wheel or fixture paths only after their verified
contents have been cached. GitHub's **SageMaker notebook platform checks**
workflow invokes this same command with OIDC credentials and retains the
attempt directory as an artifact.

## Validation status

Offline evidence/launch regressions verify the actual SDK-generated training
request and reject substituted notebook evidence. Passing them does not verify
AWS permissions, managed MLflow compatibility, native Studio behavior, or a live
notebook execution. Live qualification must be established from the retained
attempt artifacts; it has not yet been established for this branch.
