# Nightly learning cases

The nightly runner covers seven workload families at 100M and 1B. It has
twelve case variants at each scale. Each variant has a normal case and a
scrambled-control case. The complete roster contains forty-eight independent
cases. Each case builds its own snapshot and training table. Cases share the
selected wheel and immutable source inputs.

Each VM prepares only its assigned family and scale. Legacy archives contain
the declared frozen payloads. Source-only metadata packages instead bind raw
S3 objects and generation definitions. Reference measurements produce proposals.
Candidate qualification requires approved reference, policy, and resource records
with their bound evidence archive.

Temporal controls use the public `allow_new_target_entities=True` import option.
The selected reference wheel and candidate package must support that option to
retain valid future-window labels.

## Tasks and records

The roster below runs at both 100M and 1B. Every variant uses real answers
and deterministic scrambled answers in separate cases. Case identity includes
scale, variant, answer condition, and seed.

| Workload family | Case variants | Cases per scale |
| --- | --- | ---: |
| Adtech | Binary conversion prediction | 2 |
| Delivery | Multiclass, multilabel, multilabel ranking, order-count regression, and twelve-week forecasting | 10 |
| Commerce | Static merchant recommendation and future merchant recommendation | 4 |
| Streaming | Recommend fifty future titles over seven days | 2 |
| Community | Predict the thirty-day message count for recently active accounts | 2 |
| Payments | Predict an account flag within fourteen days for previously unflagged accounts | 2 |
| Fraud | Predict an account flag within twenty-eight days for recently active, previously unflagged accounts | 2 |

All forty-eight cases belong to the required roster.
Automatic rotation is separate scheduler work. It cannot
replace a missing case in the current collection.

Each scale archive pins the complete source graph and table roles. Payments
uses its full source graph. Each variant pins its public-generation definitions.
Reference runs generate fresh tables, answers, and prediction populations.
Candidate runs require approved expectations, tolerances, and measured requirements.

The runner reads four compact records in this directory:

| File | Purpose |
| --- | --- |
| `inputs.json` | Complete and workload input archives, frozen manifest hashes, reference-wheel identity, and record checksums |
| `policy.json` | Learning limits, prediction tolerances, and reference comparison bands |
| `reference.json` | Reviewed score and baseline for each case |
| `requirements.json` | CPU, RAM, scratch, GPU count, GPU memory, and time minima for each case |

Full provenance belongs in the evidence archive. Keep saved models,
predictions, measurements, failures, and calibration records there. Bind the
archive and compact records with separate checksums. The observed GPU model
does not impose an exact-model minimum.

## Case behavior

The wheel under test generates the training table through the public API.
Every reference creation and refresh builds a fresh graph and generates fresh
tables, answers, and prediction populations. It validates current structure,
row identifiers, query semantics, split separation, and control rules. It does
not compare tables, answers, stage counts, or hashes with older references.
Training and prediction continue in that same measured case.

Source-only bundles can omit old staging exports and the manifest paths
`normal_rows`, `control_rows`, and `prediction_input`. Keep the descriptor's
`query_generation_plan`, raw-source hashes, graph schema, and graph metadata pinned.
A reference control requires an explicit `control_recipe` in its manifest case:

```json
{"control_recipe": {"method": "stage-permutation-v1", "seed": 0}}
```

Use `entity-trajectory-permutation-v1` for static-link controls instead. Only
seed zero is supported. Admission rejects missing or unsupported recipes before
graph construction. Controls use canonical rows, SQLite, and disk-backed
`numpy.memmap` arrays to bound memory. Control import compares the result with
that case's generated payload.

Candidate mode compares generated stage identities with approved expectations.
It still supports legacy frozen payloads. Source-only candidate controls use
the pinned recipe and require approved normal and control stage hashes.

Snapshot checks require exact row counts and primary-key roles. Empty foreign
keys are permitted only when the frozen input manifest declares their exact
raw null counts. Malformed non-null keys, unknown keys, duplicate primary keys,
and unexplained counts fail. Each result retains the raw native counters and
the nullable-key proof alongside normalized invalid-key counters.

Training uses the capped nightly plan. Initialize it from public suggested
settings, then apply every cap below before fitting. Unchanged FAST is outside
this recipe.

```text
training_job.num_experiments = 2
optimization.max_epochs = 3
optimization.max_steps_per_epoch = 200
optimization.max_val_steps = 100
optimization.max_test_steps = 200
```

Payments and fraud require four to six sampling hops. A hop crosses one graph
relationship. Their effective plans and execution evidence must record that
depth in the selected model's resolved sampling plan.

The other suggested settings and product recovery stay active. The saved
execution must demonstrate multiple model trials and multiple epochs.
Retries do not count as distinct trials. Insufficient coverage cannot qualify
a reference, even when its score passes.

One full prediction pass supplies quality scores. Repeat and reload checks
use the same fixed sample of up to 1,000 entities. A separate prediction
checks a nonempty proper subset when the consistency sample is the full
population. Exact identities, timestamps, scores, decisions, and ranked-cutoff
evidence remain checked. A missing proper-subset check is insufficient coverage.

Commerce temporal recommendation scores recall@10 and requests up to twenty-one
candidates. Streaming scores recall@50 and requests up to fifty-one candidates.
Recall measures the fraction of actual target items recovered. Each ranked
reader preserves evidence at its own top-K cutoff. The commerce output limit
must not truncate the streaming variant.

Normal quality must improve over its baseline and beat its paired control.
The forecasting control uses a one-sided bound against a training-derived
constant for mean absolute error. Other controls use their configured baseline
limits. Nightly reference-relative quality bands are report-only. New numerical
limits require calibration and review.

The shared runner also defines the weekly replay contract. A reference-model
replay failure remains a failure, while the independent fit, prediction, and
reload still run. Weekly retains its reviewed reference-lift requirement.
Weekly replay configuration and the 10B–25B workloads are separate work.

## Resources and preparation

Run from the repository root in the selected wheel's installed environment.
Use Linux x86_64, Python 3.12, and one available CUDA GPU. The candidate uses
its own resolved dependencies. Set fresh output and scratch paths for every
execution. Keep scratch outside inputs and evidence. The host guard resolves the
scratch mount through any logical volumes to its physical disks. Each physical
disk must be local EC2 NVMe instance storage. EBS-backed, mixed, or unresolved
storage fails before Engine work. The guard records the mount and available bytes.
This check does not establish that every case fits one VM size.

The host owner supplies a delegated cgroup parent with the memory controller
enabled. A cgroup is a Linux accounting group for a process tree. The caller
creates a fresh child group and places the worker there before Engine import
and graph construction. The supervisor stays outside it. RAM comes from that
group's `memory.peak`, including children. Missing delegation or accounting
fails measurement. The parent also enforces the case deadline.

The repository observer records CPU, RAM, scratch, GPU memory, and complete
case duration. Reference measurement produces proposed minima with headroom.
It does not approve them. The new measurements must include snapshot creation,
training-table generation, training, prediction, and saved-model reload.

Input preparation requires a case name and scale. The runner resolves the case
to its workload family and downloads that family's archive. For example,
`binary-normal` selects adtech. Delivery includes its five query variants.
Legacy family archives include the declared query and answer payloads for
that family, with the complete parent manifest. A source-only metadata package
instead pins source data and generation definitions without precomputed answers.
Its manifest binds a `source_objects` inventory for the selected family.
The native `prepare` command downloads that family's raw files. It computes SHA256
while streaming each download and publishes only complete matching files.

The transport checksum identifies the smaller archive. The parent manifest
checksum identifies the frozen data. Preparation checks archive membership and
metadata. Download verification needs no extra disk read. Measured case admission
then reads each selected raw file once to make sure that its checksum matches
the manifest. A missing workload binding fails without a full-bundle fallback.

For both tiers, the host owner freezes the input directory under a separate
staging owner and mounts it read-only before starting the unprivileged case user.
Files use mode 0444 and directories use mode 0555. The case user must not own
these paths or be able to replace their ancestors through a writable alias.
Startup rejects a writable mount or unprotected input paths. Admission checks
file contents against the pinned manifest once before ingest. Neither tier
repeats the full raw-data checksum scan at the end. This freeze applies to the
selected input version while it is in use. Later changes require a new version.

Set `CONFIGURATION` to this directory's `inputs.json`. Set `RUNNER_TEMP` to an
existing SSD directory outside case scratch and evidence. Set `GITHUB_ENV` to
a fresh environment file. The `create` command writes owned directory paths
there. Load those paths before preparation:

```bash
python packages/kumo-engine/scripts/nightly_job.py create \
  --configuration "$CONFIGURATION" --scale 100M --case binary-normal
set -a
. "$GITHUB_ENV"
set +a
python packages/kumo-engine/scripts/nightly_job.py prepare \
  --configuration "$CONFIGURATION" --scale 100M --case binary-normal
export INPUT_ROOT="$NIGHTLY_JOB_ROOT/inputs"
```

Reuse `INPUT_ROOT` for later cases in the same
family and scale. Each execution still needs its own snapshot, process,
accounting group, scratch, and evidence directories. A different family or scale
needs its matching archive in a new preparation directory. Include each download
in staging measurements. `create` also requires `--case` for this configuration.

## Run one case

Set `CONFIGURATION` to the reviewed input descriptor. Set `INPUT_ROOT` to the
prepared archive for the selected family and scale. Set `CASE_OUTPUT` and `CASE_SCRATCH`
to fresh paths. Set `CGROUP_PARENT` to the delegated parent described above.

For reference measurement, install the configured reference wheel. Set
`REFERENCE_WHEEL` to that archive and `CASE_BUDGET_SECONDS` to a positive
execution budget. Set `REFERENCE_OUTPUT_S3` to the S3 destination for proposed
reference data. The CLI requires `--reference-output-s3` in reference mode.
For a reference wheel built from source, set `REFERENCE_BUILD_RECEIPT` to its
authentic `build.json`. Reference mode verifies and retains this receipt.
Reference measurement can collect proposals before a reference exists.
It cannot return a qualified pass.

Reference mode needs no separate answer-preparation job. Source bundles still
need preparation that pins and stages the raw sources. A source-only manifest
must not declare missing answer payloads in its `files` or case paths.
Preparation checks exact declared metadata and stages any raw-source inventory.

Before model training, reference mode saves table exports under
`case/reference-data/` and writes `case/reference-data/reference-data.json` with
status `PREPARED`. It publishes the prepared data to `REFERENCE_OUTPUT_S3` and
makes sure that the uploaded objects match their checksums before fitting.
Failed publication stops fitting. The S3 data survives local SSD loss or model failure. Publication does
not approve answers, models, scores, or resource requirements.

The case also writes `training-table-identity.json` before fitting. Its
`answer_policy` is `record_reference` in reference mode and `compare_approved`
in candidate mode. New reference answers and scores remain proposals until
reviewed. Candidate comparison fails before fitting when approved stage identities differ.

```bash
python packages/kumo-engine/scripts/nightly_job.py run \
  --configuration "$CONFIGURATION" --scale 100M --case binary-normal \
  --mode reference --wheel "$REFERENCE_WHEEL" \
  --build-receipt "$REFERENCE_BUILD_RECEIPT" --input-root "$INPUT_ROOT" \
  --reference-output-s3 "$REFERENCE_OUTPUT_S3" \
  --output-dir "$CASE_OUTPUT" --scratch "$CASE_SCRATCH" \
  --cgroup-root "$CGROUP_PARENT" --time-budget-seconds "$CASE_BUDGET_SECONDS"
```

For candidate execution, build the selected product once. Set `PRODUCT_SOURCE`
to a clean checkout and `WHEEL_OUTPUT` outside that checkout. The build writes
the wheel and `build.json`, which binds the source and wheel identities.

```bash
python packages/kumo-engine/scripts/nightly_job.py build \
  --source "$PRODUCT_SOURCE" --output "$WHEEL_OUTPUT"
```

Install the built wheel with its own dependencies. Set `CANDIDATE_WHEEL` to
the resulting wheel path. Candidate mode requires approved reference, policy,
and resource records. Its default deadline comes from the case's requirements.
Set `REFERENCE_ROOT` to the extracted, checksum-bound evidence archive.

```bash
python packages/kumo-engine/scripts/nightly_job.py run \
  --configuration "$CONFIGURATION" --scale 100M --case binary-normal \
  --mode candidate --wheel "$CANDIDATE_WHEEL" \
  --build-receipt "$WHEEL_OUTPUT/build.json" --input-root "$INPUT_ROOT" \
  --reference-root "$REFERENCE_ROOT" \
  --output-dir "$CASE_OUTPUT" --scratch "$CASE_SCRATCH" \
  --cgroup-root "$CGROUP_PARENT"
```

Use `--scale 1B` with its matching family archive for the next-size cases. Select
each variant and answer condition separately. Variant case names include
`streaming-normal`, `community-normal`, `payments-normal`, and `fraud-normal`,
with matching `-control` names. Both scales accept every case name.
Instance scheduling and automatic rotation remain separate work.

## Combine the complete roster

Place each execution's `result.json` beneath its scale and case directory in
`CAMPAIGN_OUTPUT`. Set `SUMMARY_OUTPUT` to a fresh summary directory. Collect
the forty-eight initial reference executions with the reference wheel:

```bash
python packages/kumo-engine/scripts/nightly_job.py combine \
  --configuration "$CONFIGURATION" --mode reference --wheel "$REFERENCE_WHEEL" \
  --results "$CAMPAIGN_OUTPUT"/100M/*/result.json \
            "$CAMPAIGN_OUTPUT"/1B/*/result.json \
  --output-dir "$SUMMARY_OUTPUT"
```

Candidate collection uses `--mode candidate` and the candidate wheel instead.
The collector checks the roster, input and execution identities, prediction
results, individual learning checks, and paired normal/control checks. It
writes failed evidence before rejecting an incomplete or invalid collection.
Reference collection also writes `reference-proposal.json`,
`requirements-proposal.json`, and `archive-proposal.json`. These files remain
unapproved. Resource proposals use the largest minimum across complete
canonical and repeated executions, with 25 percent headroom per execution.

The archive proposal maps each case's retained files beneath its scale and
case label. Copy each execution's `case/` contents to
`<scale>/<variant-or-task>-<arm>/` in the archive. Retain repeated execution
files in the separate paths declared by the manifest. The manifest also retains the
complete campaign report, including resources, failed checks, and repeats.

For an S3 archive binding, place the bound archive manifest at
`REFERENCE_ROOT/manifest.json`. For a local binding, its path is relative to
the configuration directory. Bind its checksum through `inputs.json`. Keep
the compact scores, policy, and resource minima beside that configuration.

Each selected thin case needs three complete executions under the new recipe:
the initial execution and two repeats. Keep repeats separate from the core
roster. If repeated executions enter collection, supply their preselected
execution IDs through `--canonical-executions`. Select that mapping before
examining reserved test scores. Never choose the best test-scoring execution.

Approve proposed references, calibration, and requirements only after review
of their complete measurements and required stability runs. Publication is not
approval. Scheduling and instance placement remain separate from these commands.
Source-only bundles pin raw sources, generation definitions, and control recipes
at both scales. Keep source graphs and split definitions consistent with the
reviewed stage identities.
