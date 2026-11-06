# Engine end-to-end cases

Twelve cases cover seven graph families. Nightly uses 100M and 1B. Weekly uses the same cases at 10B and 25B.
Each case defines its graph, query, split, plan overrides, scratch budget, prediction cap and comparison tolerances.
`common.run` builds a fresh snapshot, generates training rows, trains, predicts twice and reloads that run's model.
The older-model compatibility test is separate. Cases use Engine-generated answers.
`--control` exports those answers, shuffles them with seed zero and imports the supplied rows through Engine.
Controls are off by default. They keep entity/time coordinates and splits, with paired labels and frequency weights.
Forecast history stays with its original input. Static links move complete answer trajectories between compatible entities.

## Prepare a runner

Use the [standard Linux installation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/linux-machine-deployment), Python 3.12 and an NVIDIA driver.
Install the selected wheel with its S3 extra. Use the documented setup instead of DLAMI.
On a deliberately prepared fresh test host, run `sudo packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh -y`.
This command wipes eligible nonboot NVMe disks and mounts the combined storage at `$DISKGRAPH_DATA_ROOT`, default `/mnt/localssd`.
Do not run it on a host that holds retained data. Put case scratch under that mount.

## Run one case

Set `ENGINE_E2E_DATA_S3` to the raw dataset root. A case reads `<root>/<family>/<scale>/` through `kumo.S3Connector`.
Keep that source prefix fixed for reference and candidate runs through the bucket policy.
`--requirements` prints only free SSD scratch as JSON and exits before Engine or Torch imports:

```bash
python packages/kumo-engine/test/cases/adtech_binary.py --scale 100M --requirements
```

Run with the installed wheel and its full source commit:

```bash
python packages/kumo-engine/test/cases/adtech_binary.py \
  --scale 100M --reference reference.json --output result.json \
  --source-sha FULL_PRODUCT_COMMIT_SHA --scratch /mnt/localssd
```

Use `--write-reference` explicitly to create a reference after review. Ordinary runs only read it.
For a control, also pass `--control` and use a separate control reference file.
The result records `control=true`; a normal reference cannot qualify a control run.
Temporal-link controls require a wheel with `allow_new_target_entities` import support.
Controls load all exported answer rows into RAM. Their large-scale memory and runtime remain unqualified.
Static-link controls use manual split import, which does not retain the generated table's prior-stage target exclusions.
Controls compare with their own references. This change adds no paired normal/control quality gate.
References compare node and edge counts, training-table checks and metric bands.
At 100M and 1B, an order-independent pandas hash covers all exported training rows.
At 10B and 25B, native metadata supplies total rows, per-split rows and binary positive counts without a full export.
Repeat and reload predictions need identical coordinates and score differences within the case tolerance.
Metric bands use `abs(current - reference) <= abs(reference) * tolerance`.
At 10B and 25B, payments and fraud use one experiment, one epoch and balanced training sampling at batch 64.
`majority_sampling_ratio=[1.0]` and `weight_mode=[WeightMode.SAMPLE]` affect training only.
Every case and scale uses fixed validation and test budgets of 1,000 steps each.
Payments and fraud remain at every scale. Non-finite metrics fail the run. Evaluation budgets stay fixed.
Correct inadequate label support in the dataset, then regenerate its reference after review.
Results and references retain full-split positive counts under `training_table.splits.<split>.positive_rows`.
These counts describe the whole split, not measured positives in the capped evaluation prefix.
Results also record hardware, elapsed time and sampled peak RAM, GPU memory and scratch growth. Hardware and timings do not gate results.

## Configure workflows

Set `ENGINE_E2E_RUNNER_GROUP`, `ENGINE_E2E_RUNNER_LABEL`, `ENGINE_E2E_MAX_PARALLEL`, `ENGINE_E2E_DATA_S3`, `ENGINE_E2E_REFERENCES_S3` and `ENGINE_E2E_RESULTS_S3`.
References use `<scale>/<case>.json`. Results use `<run>/<attempt>/<scale>/<case>.json`.
Set `run_controls=true` when dispatching a workflow to add control jobs. The default runs only normal cases.
Control jobs use `<case>-control` in reference, result and artifact paths, with a separate VM for each job.
The workflow reuses the existing wheel builder and `gha-kumo-workflow-tests` AWS role. The role needs source/reference reads and result writes.
Both callers require `main` and `ENGINE_E2E_ENABLED=true`. They use prepared runners and run only by manual dispatch.
Configure the runner and reference locations before enabling a run. These workflows do not provision or schedule VMs.
