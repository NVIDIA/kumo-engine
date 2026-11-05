# Weekly Engine learning and model compatibility

The weekly checks reuse the [shared case runner](../../packages/kumo-engine/test/nightly/README.md)
at 10B and 25B. Twelve query variants cover seven workload families. Each variant
has normal and scrambled-control cases at each size, giving forty-eight cases.
Each case builds a fresh snapshot and training table. Cases share only the
selected wheel and immutable source data.

| Workload | Variants | Cases per size |
| --- | --- | ---: |
| Adtech | Binary conversion prediction | 2 |
| Delivery | Multiclass, multilabel, ranking, regression, and forecasting | 10 |
| Commerce | Static and temporal merchant recommendations | 4 |
| Streaming | Seven-day future title recommendations, recall@50 | 2 |
| Community | Thirty-day message counts for recently active accounts | 2 |
| Payments | Fourteen-day flags for previously unflagged accounts | 2 |
| Fraud | Twenty-eight-day flags for active, previously unflagged accounts | 2 |

Nightly owns the matching 100M and 1B cases. Automatic rotation, instance-pool
scheduling, and credentialed workflows remain separate. P1 and B9 remain separate
tests. This change supplies only a CPU workflow and explicit case commands.

## Shared execution and weekly gates

`weekly_check.py` delegates to the shared runner. It fixes the weekly tier and
uses `test/weekly/inputs.json` by default. A weekly configuration declares both
10B and 25B. It cannot use the weaker nightly comparison policy.

Training uses the shared capped plan: two experiments, at most three epochs,
200 training steps per epoch, 100 validation steps, and 200 test steps. Product
losses, numerical behavior, early stopping, and memory recovery stay active.
Payments and fraud require four-to-six-hop sampling. Streaming retains top-fifty
output and cutoff evidence. Commerce temporal recommendations retain top ten.

The installed wheel generates training tables through the public API.
Every reference creation and refresh generates fresh tables, answers, controls,
and prediction populations, then trains and predicts in the same measured case.
It validates current structure, keys, query semantics, split separation, and
control rules without comparing older reference tables, answers, counts, or hashes.
Reference controls require an explicit seed-zero `control_recipe` in the manifest.
See the [recipe example](../../packages/kumo-engine/test/nightly/README.md#case-behavior).
Admission rejects unsupported recipes before graph construction. Control generation
uses canonical rows, SQLite, and disk-backed `numpy.memmap` arrays to bound memory.
Its import check compares the result with that case's generated payload.

Before training, reference mode saves table exports and a `PREPARED` receipt at
`case/reference-data/reference-data.json`. The required `--reference-output-s3` destination receives the proposed data, population, and provenance. Full S3 readbacks must pass before fitting starts. The publication manifest is written last. A failed upload prevents fitting. Publication does not approve the reference.
New references remain proposals until reviewed. Candidate mode compares logical
stage counts, labels, keys, and timestamps with approved expectations. Legacy
frozen payloads remain supported. Source-only candidate controls generate from
the pinned recipe and require approved normal and control stage hashes.

Normal and scrambled cases keep separate evidence. Their paired learning check
cannot mix workload variants or scales.

Full label checking uses the existing canonical row format and stage hashes.
Reference mode records fresh hashes. Candidate mode compares approved hashes.
An encoder converts rows into bytes for hashing.
Exports with at least one million rows use up to eight CPU encoders. Each encoder
runs in a fresh process inside the measured case. Encoded batches contain at most
65,536 rows and 64 MiB of Arrow data. At most twice the worker count stays queued.
The parent combines bytes in input order and rejects duplicate keys across batches.
Small exports use the serial path. Temporary encoding files stay in case scratch
and are removed after use or failure.

Control generation sorts each normal and control export once. The case reuses
those ordered files for full label hashes and control construction. This cache
belongs to one fresh case. Approved reference stage hashes remain reusable, while
candidate-generated rows still receive a complete check. No label sampling replaces
that check. Complete 10B and 25B measurements must establish any runtime claim.

Normal quality must beat its baseline and retain at least 75 percent of the
reference improvement over that baseline. The direction follows the metric.
For MAE, lower error is better. Forecasting controls use the shared one-sided
comparison against a constant selected from training answers. Numerical margins
and prediction tolerances require calibration on the selected inputs.

Every candidate case must load its approved reference model and predict the
stored population without fitting that model. Stored canonical predictions
provide the comparison. A replay failure remains a failure while independent
candidate fitting, scoring, and reload continue. Missing or skipped replay fails
candidate collection.

Score the full population once. Repeat and reload checks use a fixed sample of
up to 1,000 entities. Preserve a nonempty proper-subset check. Each case also
reloads its own newly trained model. These checks use the shared bounded Arrow
reader and tolerance-aware decision comparisons.

## Frozen records and evidence

Keep four compact runtime files in [test/weekly](../../packages/kumo-engine/test/weekly/):

| File | Contents |
| --- | --- |
| `inputs.json` | Scale and variant definitions, reference-wheel identity, and approved checksum bindings |
| `policy.json` | Per-variant learning bounds and prediction tolerances |
| `reference.json` | Scores and baselines for each of the forty-eight cases |
| `requirements.json` | Measured CPU, RAM, scratch, GPU count/memory, and time minima |

Leave numerical records pending until the reference campaign supplies evidence.
Reference measurement accepts pending records and explicit temporary budgets.
It produces proposals, never an approved pass. Candidate comparison requires
reviewed records and their committed checksum bindings.

Produce references by running the configured approved wheel. Use its version
and checksum from `inputs.json`. The shared builder supplies authentic receipts
for candidate source builds. A published reference wheel needs no source-build
receipt. The weekly caller has no separate reference importer or worker.

Archive models, selected checkpoints, validation histories, populations,
predictions, resource measurements, and failures. Bind the archive manifest
through `inputs.json`. Give the compact reference and full archive separate
checksums. Do not put executable code in an input or reference archive.

## Host and input preparation

Use the shared Linux x86_64, Python 3.12, CUDA, and delegated-cgroup preparation.
Resolve each candidate wheel's own dependencies and verify the installed files.
The host owner mounts the frozen input directory read-only before starting the
unprivileged worker. Weekly admission rejects a writable input filesystem.
The backing files use mode 0444 and directories use mode 0555 under a separate
staging owner. Startup rejects worker-owned paths, write permissions, replaceable
ancestors, and privileged workers. The trusted host must not modify this input
version while it is in use. Later updates use a new version.
Both tiers hash the frozen source files once during admission before ingest.
Neither tier repeats a complete raw-data checksum scan after model work.

Keep scratch outside inputs, outputs, and the evidence archive. The runner checks that its backing disks are local NVMe instance storage and rejects EBS. Resource observation
starts before Engine import and graph construction. The case builds
its own snapshot there. Complete-case measurements include graph construction,
table generation, replay when enabled, training, readers, reload, and artifacts.
Derive numeric minima with 25 percent headroom after measurement. Observed GPU
model names stay in provenance. They do not impose an exact-model requirement.

`min_free_bytes` controls the shared staging-directory check. It does not approve
case capacity. Reference runs need an explicit positive execution budget.
Candidate runs need the reviewed per-case requirements. The supervisor enforces
the deadline and preserves available evidence after failure or timeout.

Pin source graphs, variant queries, descriptors, and generation plans before execution.
New reference runs need no separate answer-preparation job. Source bundles still
need preparation that pins and stages raw sources. Source-only bundles can omit
old staging exports, `normal_rows`, `control_rows`, and `prediction_input`.
Keep descriptor `query_generation_plan` and source-graph metadata pinned.
Do not declare missing answer payloads in manifest files or case paths.
`nightly_job.py prepare` verifies exact archive membership and downloads the selected raw source inventory. Downloads verify SHA256 while writing, without a separate raw disk scan. The measured case hashes the staged inputs once during admission. Existing raw scale-suite
data and validation records remain sources for this preparation.

Large weekly inputs can use separate graph archives. In that form,
`scales[scale].inputs.manifest` binds the common manifest object, and
`scales[scale].inputs.by_graph` binds all seven graph archives. The common
manifest checksum must match `input_manifest_sha256`. Each graph archive contains shared metadata and either its source files or a checksum-bound source inventory. Preparation fetches the inventory files from their pinned S3 locations before returning. Legacy answer payloads remain optional.
It must not contain a replacement `manifest.json`.

Pass `--case` to `prepare` to select the required graph archive. Stage the
common manifest beside the extracted payload. The manifest still declares
all twenty-four cases at that scale. Admission checks the full roster and
hashes the selected graph, reference dependencies, catalog, and required descriptors.
Candidate admission also checks its declared approved expectations. A case does
not need the other six graphs on its machine.
Single-archive inputs keep their existing complete-file validation.

The host owner starts the unprivileged supervisor inside a separate child
of the delegated parent. Worker groups are siblings of that supervisor
group. This permits process movement under the delegated common ancestor.
The supervisor remains outside each measured worker group.

## Explicit case commands

Set `CONFIGURATION` to the reviewed weekly descriptor. Install the selected
reference wheel. Set `INPUT_ROOT` to the matching frozen, read-only scale
archive. Set fresh `CASE_OUTPUT` and `CASE_SCRATCH` paths. Set `CGROUP_PARENT` to
the delegated parent and `CASE_BUDGET_SECONDS` to the initial execution budget.

```bash
python packages/kumo-engine/scripts/weekly_check.py run \
  --configuration "$CONFIGURATION" --scale 10B --case streaming-normal \
  --mode reference --wheel "$REFERENCE_WHEEL" --input-root "$INPUT_ROOT" \
  --output-dir "$CASE_OUTPUT" --scratch "$CASE_SCRATCH" \
  --cgroup-root "$CGROUP_PARENT" --time-budget-seconds "$CASE_BUDGET_SECONDS" \
  --reference-output-s3 "$REFERENCE_PROPOSALS_S3"
```

Build the candidate once from a clean product checkout. Put `WHEEL_OUTPUT`
outside that checkout. Install the resulting wheel with its own dependencies.

```bash
python packages/kumo-engine/scripts/weekly_check.py build \
  --source "$PRODUCT_SOURCE" --output "$WHEEL_OUTPUT"
```

After reference approval, set `REFERENCE_ROOT` to the bound evidence archive.
Candidate replay is mandatory. It needs no optional enable flag.

```bash
python packages/kumo-engine/scripts/weekly_check.py run \
  --configuration "$CONFIGURATION" --scale 25B --case fraud-control \
  --mode candidate --wheel "$CANDIDATE_WHEEL" \
  --build-receipt "$WHEEL_OUTPUT/build.json" --input-root "$INPUT_ROOT" \
  --reference-root "$REFERENCE_ROOT" \
  --output-dir "$CASE_OUTPUT" --scratch "$CASE_SCRATCH" \
  --cgroup-root "$CGROUP_PARENT"
```

Run each declared case separately. Collect the complete forty-eight-case result
set with `combine`. Reference collection uses the reference wheel and
`--mode reference`. Candidate collection uses its wheel and `--mode candidate`.

```bash
python packages/kumo-engine/scripts/weekly_check.py combine \
  --configuration "$CONFIGURATION" --mode candidate \
  --wheel "$CANDIDATE_WHEEL" \
  --results "$CAMPAIGN_OUTPUT"/10B/*/result.json \
            "$CAMPAIGN_OUTPUT"/25B/*/result.json \
  --output-dir "$SUMMARY_OUTPUT"
```

The collector writes the available per-case summary before reporting failure.
Missing, duplicate, failed, mixed-scale, mixed-variant, or mixed-identity results
cannot pass. Reference collection also writes reference, requirements, and
archive proposals. Review calibrated values before candidate qualification.
