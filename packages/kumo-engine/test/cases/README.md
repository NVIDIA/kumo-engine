# Engine end-to-end cases

The twelve Python cases cover seven graph families. Each case defines its graph, PQL, training limits, and prediction checks. Nightly uses 100M and 1B. Weekly uses the same files at 10B and 25B. Scrambled controls are optional in the review and are omitted here. These cases do not import supplied answer tables.

Engine reads the raw S3 tables, builds a fresh snapshot, and generates the training table. The helper compares node counts per table, edge counts per relation, and every exported training row. It selects a bounded held-out population before prediction. Each case repeats prediction and reloads its own new model. Only adtech at 100M also loads one older model on an approved fixture of at most twenty rows.

Training uses two experiments, three epochs, and 200/100/200 train/validation/test steps. Fraud and payments use six hops. Normal Engine kernels and out-of-memory recovery remain in use. These limits and the declared free-SSD budgets need real-run qualification. Prediction caps and comparison tolerances are explicit inputs, not measured defaults.

## Prepare a fixed test runner

Use the [standard Linux installation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/linux-machine-deployment) on Ubuntu 24.04 x86_64, Python 3.12, and an installed NVIDIA driver. Install the selected wheel with its S3 extra. No DLAMI is required.

On a deliberately prepared fresh machine, the existing setup command is `sudo packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh -y`. It wipes and formats eligible nonboot NVMe disks. It stripes multiple disks and mounts them at `DISKGRAPH_DATA_ROOT`, which defaults to `/mnt/localssd`. Do not run it on a machine with retained data.

Run the candidate without root, sudo, host credentials, or instance metadata access. Reserve the runner for its case so resource measurements cover that workload. Host setup, shutdown, and pool scheduling are outside the case scripts. No workflow creates or starts an instance.

## Bind source data and references

Each small input JSON binds one family and scale. Graph schemas and queries remain in the Python case. Use this structure with real values:

```json
{
  "scale": "100M",
  "source": "s3://dataset-bucket/immutable-revision/adtech/100M/",
  "region": "us-west-2",
  "revision": "reviewed-source-revision",
  "immutable_prefix_policy": "reviewed-write-protection-policy",
  "read_only_role": "arn:aws:iam::ACCOUNT:role/test-source-reader",
  "objects": [{"key": "immutable-revision/adtech/100M/users/part-0.parquet", "version_id": "NON_NULL_VERSION", "size": 123}]
}
```

List every source object. The public S3 connector reads current objects, not historical VersionIds. The inventory records provenance. It does not select historical bytes. The source prefix must have externally enforced write protection for every writer, and the test role must be read-only. Verify those policies before enabling execution. The Engine also checks object consistency during reads. There is no harness downloader or raw-file hash scan.

Write a reference explicitly with `--write-reference`. Ordinary runs only read it. Review an intentional graph or table-generation change together with its replacement reference. Use versioned S3 for reference JSON files. Do not reuse older full-population metrics as references for these capped runs.

The logical table fingerprint includes schema, row count, entities, anchors, labels, split, and forecast history. It combines four vectorized noncryptographic row fingerprints as a multiset, then hashes that record with SHA-256. It preserves duplicate rows and ignores file order, row order, and Parquet encoding. Link answers ignore target order while pairing frequency weights. Forecast horizons retain order. This is a regression fingerprint, not a cryptographic proof of row equality. Pandas versions are recorded for diagnosis. A changed hashing implementation can require a reviewed reference refresh. Large-table cost remains unqualified.

Metric bands are two-sided: `abs(current - reference) <= abs(reference) * tolerance`. Performance bands reject regressions only: `current - reference <= abs(reference) * tolerance`. A zero reference permits no positive difference. Hardware identity must match. Prediction checks use an absolute score tolerance and allow ranking changes only at a tolerated boundary or tie.

## Run a case

`--requirements` prints only free SSD scratch and exits before Engine or Torch imports:

```bash
python packages/kumo-engine/test/cases/adtech_binary.py --scale 100M --requirements
```

Run from an environment that contains the wheel built from the recorded source commit:

```bash
python packages/kumo-engine/test/cases/adtech_binary.py \
  --scale 100M --input input.json --reference reference.json --output result.json \
  --source-sha FULL_PRODUCT_COMMIT_SHA --scratch /mnt/localssd \
  --prediction-rows APPROVED_CAP --metric-tolerance APPROVED_RELATIVE_BAND \
  --performance-tolerance APPROVED_RELATIVE_BAND --prediction-tolerance APPROVED_ABSOLUTE_BAND
```

For the complete nightly suite, adtech at 100M also requires `--compatibility --older-model MODEL_DIRECTORY --older-predictions EXPECTED.parquet`. The fixture contains the older model's public binary prediction output, including `ENTITY`, `ANCHOR_TIMESTAMP`, `True_PROB`, and `TARGET_PRED`. Its entities and timestamps define the compatibility population. Keep the model and fixture in an immutable, reviewed location. This check compares the loaded older model with its stored predictions.

A case writes one result JSON on success or failure. It records the real source SHA, input binding, training metrics, phase times, sampled peak GPU/process-tree RAM, and scratch growth. Results are saved before owned temporary scratch is removed. The shared filesystem scratch-growth measurement assumes the runner is reserved. Independent matrix cases continue after another case fails.

## Configure the workflows

The callers are manual-only and require `main` plus `ENGINE_E2E_ENABLED=true`. Keep activation off during a test-fleet stop. The shared workflow builds the real product commit once with the existing builder and installs its wheel in a clean virtual environment. It does not change the tracked source or version files.

Configure `ENGINE_E2E_RUNNER_GROUP`, `ENGINE_E2E_RUNNER_LABEL`, and `ENGINE_E2E_MAX_PARALLEL` for approved fixed runners. Protect `gha-kumo-workflow-tests` so only `main` can use its scoped roles. The runner must block instance metadata and host/provisioning credentials externally. SDK environment settings alone do not provide that isolation.

Configure `ENGINE_E2E_READ_ROLE`, `ENGINE_E2E_WRITE_ROLE`, `ENGINE_E2E_DATA_S3`, `ENGINE_E2E_INPUTS_S3`, `ENGINE_E2E_REFERENCES_S3`, and `ENGINE_E2E_RESULTS_S3`. Inputs and references use `<scale>/<case>.json`. Results use `<run>/<attempt>/<scale>/<case>.json`. The candidate receives only source/reference read credentials. The later upload step receives separate write credentials. Source immutability, bucket versioning, role policies, and runner isolation need live qualification before activation.

`ENGINE_E2E_CASE_SETTINGS_JSON` maps each case and scale to `prediction_rows`, `metric_tolerance`, `performance_tolerance`, and `prediction_tolerance`. Set `ENGINE_E2E_COMPAT_MODEL_S3` to the older model directory prefix and `ENGINE_E2E_COMPAT_PREDICTIONS_S3` to its Parquet fixture. Missing bindings fail the affected case. Resource sizing, GPU launch, full completion, and runtime targets remain pending real-run evidence.
