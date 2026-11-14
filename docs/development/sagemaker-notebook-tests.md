# SageMaker notebook platform tests

The runner submits a SageMaker `NotebookJobStep` in a Pipeline. SageMaker runs
the notebook as a managed training job; the runner collects its executed
notebook, evidence files, and browser results. Local and GitHub execution use
the same entrypoint and resource configuration.

## Notebook execution

The initialization script creates a clean virtual environment, installs the
selected complete wheel with `[sagemaker]`, checks its dependencies, and registers the
notebook kernel. Kernel tools are installed separately from the product.
The runner caches inputs in S3 by SHA256 and checks S3 checksums before reuse.
The container is the unchanged AWS SageMaker Distribution 3.6.1 CPU image,
mirrored into the dedicated `kumo-mlops-validation` ECR repository and pinned
by its original digest. The execution role only needs access to that repository.

The public source is
`public.ecr.aws/sagemaker/sagemaker-distribution@sha256:c67c9ea12dcf5faa14496248553f6b441567137a75d533d0733a7671f00cf986`.
AWS publishes its [3.6.1 build inputs](https://github.com/aws/sagemaker-distribution/tree/main/build_artifacts/v3/v3.6/v3.6.1).
The private mirror is an account setup dependency, not a private image build.
An authorized administrator with Skopeo can populate the existing test repository
without changing its manifest or layers:

```sh
test_registry="$(PYTHONPATH=packages/kumo-engine/test/sagemaker_notebook \
  python -c 'from config import ACCOUNT, REGION; print(f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com")')"
aws ecr get-login-password --region us-west-2 | skopeo login \
  --username AWS --password-stdin "$test_registry"
skopeo copy --all --preserve-digests \
  docker://public.ecr.aws/sagemaker/sagemaker-distribution@sha256:c67c9ea12dcf5faa14496248553f6b441567137a75d533d0733a7671f00cf986 \
  "docker://$test_registry/kumo-mlops-validation:3.6.1-cpu"
```

Other accounts must create their own repository and update the account, IAM and
test-resource configuration together. The workflow does not create the mirror.

The notebook verifies its actual execution role, training-job identity, image,
kernel, mounted input notebook, architecture, Python, and glibc. Each lifecycle
subprocess repeats the execution-identity check. A local process with copied
environment variables cannot qualify as a managed notebook run.

The controller publishes the job ARN from the Pipeline service response to an
immutable, attempt-specific S3 receipt. The kernel checks that receipt and
independently describes the job and its tags. The execution role does not need
permission to list other training jobs.

The `engine` scenario reuses `wheel_mlops_lifecycle.py` and the shared small training
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
presigned URL supplied over stdin to Playwright. The runner also needs the
scoped MLflow read/list policy; `AccessUI` alone does not authorize metric-page
metadata requests. They inspect the two fit runs,
four prediction runs, and two model versions. Drift and evaluation update the
two alias prediction runs; the checker compares their final updated metrics.
Training history checks open each metric in a fresh page, compare its UI CSV
download with recorded epochs, and inspect visible hover values and positions
against rendered axis labels.
Authentication URLs, browser sessions, and traces are
not retained. This covers the managed MLflow registry; automatic synchronization
to SageMaker Model Registry is disabled on the existing server and is not
claimed by this test.

Native Studio checks open the dedicated private JupyterLab space through a
short-lived presigned URL. Playwright uploads the validated executed notebook,
trusts its retained outputs, compares its printed receipt with the producer,
and exercises the graph's zoom, fit, layout, schema, tooltip, sample-preview,
and export controls. The producer reserves a markdown cell; Studio replaces its
text with a unique invocation marker before saving. It reopens the notebook in a
fresh browser context, then verifies the receipt and saved graph outputs again.
Studio does not run cells; this checks interactive rendering of retained outputs.
The runner creates a CPU app only when needed and stops any suite-tagged app
in this exact dedicated private space, including one left by an interrupted run.
The dedicated profile also enforces a 60-minute idle shutdown as a cancellation
fallback. A conditional S3 lease prevents overlapping GitHub/local invocations from
stopping each other's app; expired leases allow recovery after interruption.
The fixed 100 GiB private EBS space and uploaded validation notebooks persist
for inspection; stopping the app stops compute, without deleting that storage. Authentication state
passes between browser contexts in memory and is never saved to disk.

Full qualification also requires nonzero drift sensitivity, active remote-writer interruption, and
full epoch-history exports; the harness records these as outstanding. Successful
selected checks can pass while `full_qualification` remains `incomplete`.
`--engine-checks-only` selects execution and saved-export checks without native
MLflow or native Studio UI verification.

UI replay requires both private local evidence and the corresponding cloud
state. Normal completed attempts clean that state; a replay after cleanup cannot
qualify managed MLflow UI. If the producer stopped before cleanup, use:

```sh
SAGEMAKER_TEST_DOMAIN_ID=d-YOUR_TEST_DOMAIN \
python packages/kumo-engine/test/sagemaker_notebook/run.py --execute \
  --ui-run /path/to/retained-evidence --producer-commit PRODUCER_SHA \
  --candidate /path/to/original-candidate.json \
  --output-dir /path/to/new-ui-result
```

UI replay does not submit a job or upload training inputs. Browser failures retain
safe post-authentication screenshots, accessibility snapshots, frontend error
categories, and completed execution results. Authentication URLs and arbitrary
frontend error messages are not retained. For older producer notebooks without
the markdown fixture, replay adds only that non-code cell to a private upload
copy, validates the original code and receipt, and records both notebook hashes.

Private local evidence includes the submitted and executed notebooks, job and
Pipeline responses, phase receipts, logs, graph outputs, and browser evidence.
GitHub uploads only an allowlisted aggregate summary; raw evidence stays on its
ephemeral runner. After collection and UI verification, cleanup removes the
attempt Pipeline, model registration, S3 job/output/model prefixes, and MLflow
artifact prefix, and soft-deletes its MLflow experiment metadata. Shared inputs
remain cached. Cleanup refuses preexisting attempt resources, foreign tags,
and active Pipelines, records failures, and fails an otherwise passing run.
If a runner is cancelled before cleanup, its attempt-specific resources may
remain; the idle shutdown bounds unattended Studio app lifetime.

## Test resources and authentication

Configuration is in
[`config.py`](../../packages/kumo-engine/test/sagemaker_notebook/config.py).
The suite uses account `926922431314`, region `us-west-2`, the existing
`kumo-test-dataplane-bucket/mlops-validation/sagemaker/` prefix,
`kumo-mlops-validation` managed MLflow server, and
`kumo-engine-sagemaker-test-execution` runtime role. Each job has a unique
`kumo-test-notebook-<attempt>` name and writes to its own prefix.

GitHub authenticates with OIDC as `kumo-engine-sagemaker-test-runner`. Its trust
accepts only this repository's `sagemaker-mlops-test` environment subject and
`sts.amazonaws.com` audience. There is no AWS access-key secret. That environment
must require maintainer-team approval, disable admin bypass, and allow only
explicitly selected branches. Self-approval is permitted for debugging in this
dedicated test environment. These server-side settings
and the IAM trust enforce access; the branch-owned preflight is configuration lint.
Only manual dispatch starts cloud work.

The runner can pass only `kumo-engine-sagemaker-test-pipeline`; that role alone
can create training jobs and pass `kumo-engine-sagemaker-test-execution`.
IAM restricts job instance types to `ml.m5.2xlarge` and runtime to 3,600 seconds.
The compiled NotebookJobStep also requires one instance and no retries.
IAM likewise limits Studio app creation to the same CPU instance type. These
bounds do not cap the number of permitted jobs or replace account quotas.
The notebook/Studio execution role cannot create jobs or pass any role.
The [AWS MLflow action reference](https://docs.aws.amazon.com/service-authorization/latest/reference/list_sagemaker-mlflow.html)
lists `LogModel`, `LogLoggedModelParams` and `SetLoggedModelTags`; it does not
list a `CreateLoggedModel` IAM action. The execution policy includes those named
model-write permissions on the dedicated tracking server.

`prepare_access.py` generates one self-contained
`update-sagemaker-test-permissions.sh` for an admin to review and run with AWS CLI
and jq. It validates all existing state before changes, creates the three dedicated
roles and a fresh private Studio profile/space, and revokes only recognized legacy
suite grants from the shared runner and old combined role, including its recognized
`pipeline-steps` self-launch grant. It preserves unrelated
permissions and trust, rejects drift, and leaves existing domain defaults unchanged.

The script discovers the suite-tagged `kumo-test-sagemaker` domain and creates the
`kumo-test-validation` profile and `kumo-test-validation-space` private space,
using 100 GiB EBS and `ml.m5.2xlarge`. It disables EFS auto-mounting and output sharing, and configures
60-minute idle shutdown. It never starts an app or job. Put its returned domain ID
in the GitHub environment variable `SAGEMAKER_TEST_DOMAIN_ID`; local full-UI runs
use the same variable. No domain identifier is committed to source.

```sh
python packages/kumo-engine/test/sagemaker_notebook/prepare_access.py \
  --output-dir /tmp/kumo-sagemaker-permissions
sh /tmp/kumo-sagemaker-permissions/update-sagemaker-test-permissions.sh
```

The admin runs the second command after reviewing the generated script.
Local credentials must independently have the equivalent scoped permissions;
the environment-only OIDC role does not grant SSO identities additional access.

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

Commit the exact test inputs before execution. GitHub builds the complete wheel
from the dispatched `github.sha`, downloads that same run’s artifact, and generates
a private runtime manifest with its source, filename, SHA256 and version. It does
not select a historical build or require a committed pin update. The fixture
manifest is `packages/kumo-engine/test/pr172/inputs.json`.

Locally, build the wheel from the current checkout and generate the same manifest:

```sh
packages/kumo-engine/scripts/build_wheel.sh --output-dir /tmp/kumo-wheel
python packages/kumo-engine/test/sagemaker_notebook/candidate.py \
  --wheel /tmp/kumo-wheel/*.whl --source-ref "$(git rev-parse HEAD)" \
  --output /tmp/kumo-sagemaker-candidate.json
```

```sh
AWS_PROFILE=<authorized-test-profile> SAGEMAKER_TEST_DOMAIN_ID=<test-domain-id> \
  /tmp/kumo-sagemaker-runner/bin/python \
  packages/kumo-engine/test/sagemaker_notebook/run.py \
  --execute --scenario all \
  --candidate /tmp/kumo-sagemaker-candidate.json \
  --wheel /path/to/built.whl \
  --fixture /path/to/pr172-inputs-v1.tar.gz \
  --output-dir /tmp/kumo-sagemaker-attempt-unique
```

Omit `--execute` to inspect the configuration without contacting AWS. The output
directory must be new. Omit wheel or fixture paths only after their verified
contents have been cached. GitHub's **SageMaker notebook platform checks**
workflow invokes this same command with OIDC credentials and uploads only an allowlisted aggregate summary. Raw notebook outputs, browser
evidence, identities, and cloud API responses remain local to the ephemeral job.
Use a local run to inspect private evidence.

## Validation status

Offline evidence/launch regressions verify the actual SDK-generated training
request and reject substituted notebook evidence. Passing them does not verify
AWS permissions, managed MLflow compatibility, native Studio behavior, or a live
notebook execution. Live qualification must be established from private local evidence or a successful
current-head Actions run; it has not yet been established for this branch.

Actions validates the built wheel’s metadata before assuming AWS credentials.
Execution rejects a runtime manifest whose `build.source_commit` does not match
the committed product sources. Historical or mixed-source candidates cannot qualify. The workflow admits manual dispatch only when the protected environment allows
the selected branch and a maintainer approves the deployment; the person who
started the run may approve it in this dedicated test environment.
The 150-minute job budget covers the bounded notebook, browser, startup,
shutdown, and 10-minute terminal cleanup wait. Offline harness regressions can
run locally without cloud credentials; automatic platform-regression CI remains
a separate follow-up.

The runner and kernel-tool requirements are generated hash locks. Edit their
`.in` files and run `python packages/kumo-engine/test/sagemaker_notebook/compile_requirements.py`
with uv 0.13.7. Actions and bootstrap install these tools with `--require-hashes`.
The candidate product wheel remains selected separately by its verified hash.
Its `[sagemaker]` dependency closure is resolved from PyPI at installation time;
those product dependencies are not covered by the runner or kernel-tool hash locks.
`pip check` validates the installed closure, not its hashes.

Notebook Jobs retain the installed 3,600-second limit. Bootstrap records one
3,000-second work deadline shared by installation, kernel graph work and all
lifecycle phases. Each installation retains its 1,200-second cap and each
lifecycle phase its 900-second cap, reduced when less work time remains.
Subprocess limits are capped by the remaining budget; a kernel
timer also interrupts work at that deadline. The timer is removed before the
final evidence upload, reserving ten minutes of the job limit for publication
and framework overhead. The controller allows 4,500 seconds; Actions allows 150
minutes for managed UI checks and cleanup.

The admin migration accepts only exact recorded previous dedicated policies
when replacing the MLflow wildcard with named actions. The existing runtime cap
is unchanged, so applying that tightening is not a prerequisite for rerunning
the notebook. Preparing or testing the script does not apply IAM changes.

Native Studio save verification adds a unique markdown edit absent from the
uploaded notebook, observes the dirty state, saves, and requires that edit after
reopening in a fresh browser context.

Before merging, enable `prevent_self_review` on the protected platform test
environments. Trial branches permit maintainer self-approval; the shared guard
requires independent approval for `main`.
