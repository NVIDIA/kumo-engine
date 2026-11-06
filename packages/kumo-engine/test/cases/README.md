# Engine end-to-end cases

Twelve cases cover seven graph families. Nightly uses 100M and 1B. Weekly uses the same cases at 10B and 25B.
Each case defines its graph, query, split, plan overrides, scratch budget and instance type. `common.LIMITS` supplies shared prediction caps and comparison tolerances.
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
`--requirements` prints `scratch_gib` and `instance` as JSON without Engine imports. Run a case with the installed wheel and its full source commit:

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
Results also record hardware, elapsed time and sampled peak RAM, GPU memory and scratch growth. CI requires the declared instance type and matching reference hardware. Timings do not gate results.
Outside CI, the declared instance type does not restrict the host. Local comparisons without instance metadata retain their existing behavior.

## Configure workflows

The workflow reuses the wheel builder and `gha-kumo-workflow-tests` for source/reference reads, admission object reads and result writes.
Pool and lifecycle controllers use the `engine-e2e-pool` environment, restricted to `main`, with the cleanup App key stored there.
`run_controls=true` adds control jobs with `<case>-control` reference, result and artifact keys. Normal cases remain the default.
The callers support manual dispatch and schedules. Scheduled jobs require the enable flags below.
A scheduled nightly skips pool creation if a newer scheduled nightly exists on `main`. Manual runs bypass this check.
Before the first candidate run at each scale, create references with `write_reference=true` on the declared instance type.
Older references without `hardware.instance_type` fail the hardware check.
If controls are enabled, create their separate references with `run_controls=true`.
After reviewing the references, keep `write_reference=false` for candidate runs on matching hardware.

| Configuration | Value or scope |
| --- | --- |
| `ENGINE_E2E_DATA_S3`, `ENGINE_E2E_REFERENCES_S3`, `ENGINE_E2E_RESULTS_S3` | Raw sources, `<scale>/<case>.json` references and `<run>/<attempt>/<scale>/<case>.json` results. |
| `ENGINE_E2E_ENABLED`, `ENGINE_E2E_SCHEDULER_ENABLED` | Both `true`, on `main` only. |
| `ENGINE_E2E_SCHEDULE_ENABLED` | Scheduling also requires `true`. Nightly runs at 00:17 daily and weekly at 00:47 Sunday, in `America/Los_Angeles`. The timezone follows daylight saving time. |
| `ENGINE_E2E_MAX_PARALLEL` | Total VM limit, default `2`, with at least one worker per selected type. |
| `ENGINE_E2E_TEMPLATES_JSON` | Pinned template/version and subnets per type: `{"g7e.24xlarge":{"template":"lt-example","version":"1","subnets":["subnet-example"]}}`. Keep mappings fixed until cleanup finishes. |
| `ENGINE_E2E_POOL_ROLE` | Main-only OIDC access to owned groups, derived template versions and schedules, and `s3:GetObject` for admission objects under `ENGINE_E2E_RESULTS_S3`. |
| `ENGINE_E2E_AWS_REGION` | Default `us-west-2`. |
| `ENGINE_E2E_RUNNER_APP_ID`, `ENGINE_E2E_RUNNER_APP_KEY` | Cleanup App credentials. |
| Lambda `RUNNER_SECRET` | Secrets Manager secret containing `app_id`, `installation_id` and `private_key`. |
| `ENGINE_E2E_CAPACITY_FUNCTION_ARN`, `ENGINE_E2E_CAPACITY_ROLE` | Registration/report Lambda and its Scheduler invocation role. |
| Lambda `RESULTS_S3` | Set to `ENGINE_E2E_RESULTS_S3`. The Lambda role needs secret reads and result-object reads/writes. |
| `ENGINE_E2E_EXPIRY_ROLE` | Scheduler access to suspend/delete only owned groups. Limit launcher PassRole to approved roles. |

Package `.github/scripts/engine-e2e-capacity.cjs` and `engine-e2e-pool.cjs` as a Node.js 22 Lambda with handler `engine-e2e-capacity.handler`.
The App needs Administration write and Actions write for `NVIDIA/kumo-engine` only. Lambda keeps its keys and issues fresh registration tokens at boot.
The external Lambda role needs Auto Scaling `DescribeAutoScalingGroups` and `SuspendProcesses`, with suspension restricted to owned groups.
The VM profile invokes only this Lambda. Repository workflow permissions control runner access. Labels select hardware without restricting workflow access.

AWS maintains one pool per declared instance type and retries that type across approved subnets outside the CPU job.
Capacity waits are four hours nightly and 24 hours weekly. Case execution limits are 180 and 360 minutes, respectively.
Weekly replaces its VM after each case. Nightly can reuse VMs. Re-run all jobs to create a new pool for an attempt.
The Lambda treats a pool as drained only after all planned jobs are visible and none remains queued or waiting.
It suspends only new launches for that pool. Existing workers continue, and the unused replacement exits cleanly.
Before allocation, AWS schedules suspend launches at the capacity deadline.
Deletion is scheduled at the capacity deadline plus the case job timeout and a 30-minute cleanup buffer.
These safeguards use minute precision and are best effort.
Conditional S3 claims protect admitted cases. From the wait limit, EventBridge checks unstarted cases every ten minutes. Runs completed before that limit are skipped.
Local prerequisite failures send their check and exit status to the registration Lambda. Registration and configuration failures shut down only that VM.
Network reads retry up to five times. Registration makes up to three attempts, with backoff, before the capacity deadline.
Retries never repeat the SSD formatter. Hardware checks still fail immediately.
The Lambda validates the live run and owned group, suspends launches, and reports unadmitted cases through lifecycle jobs with `phase=bootstrap`.
Stored claims retain the failure reason and protect admitted cases.
Capacity failures appear in a separate lifecycle run and preserve requested hardware.
After a four-hour capacity failure, the original nightly job can remain queued until GitHub's 24-hour queue expiry.
The source run and its cleanup can remain pending until those queued jobs expire.
Saved failure claims remain reportable when a run finishes at or after the deadline.
Cleanup first deletes owned groups and waits for their deletion under one shared limit of ten minutes. It then removes owned template versions and runner registrations.

Templates use the prepared GPU runner above, Python 3.12 with venv, AWS CLI, curl, `script`, XFS/RAID tools and IPv4/IPv6 iptables.
Install the pinned, unconfigured Actions runner at `/opt/actions-runner`. Attach only disposable nonboot NVMe disks for bootstrap formatting.
Bootstrap checks the actual type and free SSD, enables linger for shared memory and blocks worker access to instance metadata.
Verify image, IAM, registration, deadline reports and cleanup before schedule activation. Full-campaign quality and runtime remain separate from launch qualification.
CPU checks run without AWS in [OSS Readiness](../../../../.github/workflows/oss-readiness.yml).
