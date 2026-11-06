# Engine end-to-end cases

Twelve cases cover seven graph families. Nightly uses 100M and 1B. Weekly uses the same cases at 10B and 25B.
Each case defines its graph, query, split, plan overrides and scratch budget. `common.LIMITS` supplies the prediction cap, prediction tolerance and relative metric tolerance.
`common.run` builds a fresh snapshot, generates training rows, trains, predicts twice and reloads that run's model.
Cases use Engine-generated answers. Older-model compatibility runs separately.
`--control` exports those answers, shuffles them with seed zero and imports the supplied rows through Engine.
Controls default to off and retain coordinates, splits, forecast history and paired label/frequency values. Static links move complete trajectories between compatible entities.

## Prepare a runner

Use the [standard Linux installation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/installation/linux-machine-deployment), Python 3.12 and an NVIDIA driver.
Install the selected wheel with its S3 extra.
On a deliberately prepared fresh test host, run `sudo packages/diskgraph/scripts/localssd-raid0-xfs-tui.sh -y`.
This command wipes eligible nonboot NVMe disks and mounts the combined storage at `$DISKGRAPH_DATA_ROOT`, default `/mnt/localssd`.
Do not run it on a host that holds retained data. Put case scratch under that mount.

## Run one case

Set `ENGINE_E2E_DATA_S3` to the raw dataset root. A case reads `<root>/<family>/<scale>/` through `kumo.S3Connector`.
Keep that source prefix fixed for reference and candidate runs through the bucket policy.
`--requirements` prints required free SSD scratch as JSON without Engine imports. Run a case with the installed wheel and its full source commit:

```bash
python packages/kumo-engine/test/cases/adtech_binary.py \
  --scale 100M --reference reference.json --output result.json \
  --source-sha FULL_PRODUCT_COMMIT_SHA --scratch /mnt/localssd
```

Use `--write-reference` explicitly to create a reference after review. Ordinary runs only read it.
For a control, add `--control` and use a separate reference. Results record `control=true` and reject normal references.
Temporal-link controls require a wheel with `allow_new_target_entities` import support.
Controls load all exported answer rows into RAM. Their large-scale memory and runtime remain unqualified.
Static-link imports omit native prior-stage target exclusions. Controls have no paired normal/control quality gate.
References compare node and edge counts, training-table checks and metric bands.
At 100M and 1B, an order-independent pandas hash covers all exported training rows.
At 10B and 25B, native metadata supplies total rows, per-split rows and binary positive counts without a full export.
Repeat and reload predictions need identical coordinates and score differences within the case tolerance.
AUROC and AP, including their variants, use an absolute band of 0.03.
Other metrics, including MAE and RMSE, use `abs(current - reference) <= abs(reference) * tolerance`, with a relative tolerance of 25%.
At 10B and 25B, payments and fraud use one experiment, one epoch and balanced training sampling at batch 64.
`majority_sampling_ratio=[1.0]` and `weight_mode=[WeightMode.SAMPLE]` affect training only.
Every case and scale uses fixed validation and test budgets of 1,000 steps each.
Payments and fraud remain at every scale. Non-finite metrics fail. Correct inadequate label support in the dataset before regenerating its reference after review.
`training_table.splits.<split>.positive_rows` records whole-split positives, not measured positives in the capped evaluation prefix.
Results also record hardware, elapsed time and sampled peak RAM, GPU memory and scratch growth. Hardware and timings do not gate results.

## Configure workflows

Set `ENGINE_E2E_RUNNER_GROUP`, `ENGINE_E2E_RUNNER_LABEL`, `ENGINE_E2E_MAX_PARALLEL`, `ENGINE_E2E_DATA_S3`, `ENGINE_E2E_REFERENCES_S3` and `ENGINE_E2E_RESULTS_S3`.
References use `<scale>/<case>.json`. Results use `<run>/<attempt>/<scale>/<case>.json`.
Set `run_controls=true` to add control jobs. They use `<case>-control` in reference, result and artifact paths.
The workflow reuses the existing wheel builder and `gha-kumo-workflow-tests` AWS role. The role needs source/reference reads and result writes.
Both callers require `main` and `ENGINE_E2E_ENABLED=true`. They use prepared runners and run only by manual dispatch.
Nightly and weekly define no schedule yet. They do not run automatically until scheduling lands.
Configure the runner and reference locations before enabling a run. These workflows do not provision or schedule VMs.

## Run older-model compatibility manually

Use the installed candidate wheel to load saved older models and compare predictions with approved fixtures.
This test performs no training. Full manual coverage is twelve cases at four scales: 48 case/scale pairs.
The cases already cover all seven graph families.
The nightly scales, 100M and 1B, contain 24 pairs. The weekly scales, 10B and 25B, contain 24 pairs.

Prepare a GPU host, Python 3.12, pytest, the candidate wheel's S3 extra and AWS read access.
Size the host for the full selected source. Each check builds a fresh graph snapshot despite predicting at most twenty inputs.
The existing case `--requirements` output gives scratch guidance.
Measured compatibility resource requirements and GPU execution across all scales remain unqualified.

Choose approved source, model and fixture bindings for every pair before running.
Use a trusted S3 model root or an absolute local directory.
Put each saved model under `<model-root>/<scale>/<case>/`.
Put `population.parquet` and `expected.parquet` under `<fixture-root>/<scale>/<case>/` on the host.
Keep models and fixtures outside the disposable work directory. Record their producer wheel and checksums before approval.
These paths are a directory convention, not a supplied or approved set of 48 models.

Run this block in Bash from the repository root, after replacing the three example roots:

```bash
(
  set -euo pipefail
  compat_source_root='s3://APPROVED_DATA_ROOT'
  compat_model_root='s3://APPROVED_MODELS'
  compat_fixture_root='/absolute/path/approved-fixtures'
  compat_scales=(100M 1B 10B 25B)
  compat_cases=(
    adtech_binary delivery_multiclass delivery_multilabel
    delivery_multilabel_ranking delivery_regression delivery_forecasting
    commerce_static_link commerce_temporal_link streaming_temporal_link
    community_regression payments_binary fraud_binary
  )
  compat_work=$(mktemp -d "${DISKGRAPH_DATA_ROOT:-/mnt/localssd}/model-compat.XXXXXXXX")
  cp -R packages/kumo-engine/test/cases "$compat_work/cases"
  cd "$compat_work"
  compat_status=0
  for compat_scale in "${compat_scales[@]}"; do
    mkdir -p "reports/$compat_scale" "scratch/$compat_scale"
    for compat_case in "${compat_cases[@]}"; do
      export ENGINE_COMPAT_CASE="$compat_case"
      export ENGINE_COMPAT_SOURCE="${compat_source_root%/}/${compat_case%%_*}/$compat_scale/"
      export ENGINE_COMPAT_MODEL="${compat_model_root%/}/$compat_scale/$compat_case/"
      export ENGINE_COMPAT_FIXTURE="${compat_fixture_root%/}/$compat_scale/$compat_case"
      if python -E -s -m pytest -c /dev/null -p no:cacheprovider \
          --confcutdir=cases cases/model_compatibility.py \
          --basetemp "$compat_work/scratch/$compat_scale/$compat_case" \
          --junitxml "$compat_work/reports/$compat_scale/$compat_case.xml" \
          2>&1 | tee "$compat_work/reports/$compat_scale/$compat_case.log"; then
        printf 'PASS %s/%s\n' "$compat_scale" "$compat_case"
      else
        compat_status=1
        printf 'FAIL %s/%s\n' "$compat_scale" "$compat_case"
      fi
      rm -rf -- "$compat_work/scratch/$compat_scale/$compat_case"
    done
  done
  printf 'Reports: %s/reports\n' "$compat_work"
  exit "$compat_status"
)
```

The loop runs pairs sequentially and continues after failures. Any failed pair makes the final exit status nonzero.
It removes only each pair's generated scratch after execution. Logs and JUnit reports remain outside scratch.
For smaller hosts, select `compat_scales=(100M 1B)` or `compat_scales=(10B 25B)` and retain both groups' reports.
Count full coverage only after all 48 approved pairs pass. Model selection, reference preparation and approval remain separate work.
This command runs manually and adds no nightly, weekly, scheduler or cron trigger.
