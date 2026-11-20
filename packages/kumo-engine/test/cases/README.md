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
Temporal-link imports use the query's label-window end to validate target creation times, matching generated answers.
Controls load all exported answer rows into RAM. Their large-scale memory and runtime remain unqualified.
Static-link imports omit native prior-stage target exclusions. Controls have no paired normal/control quality gate.
References compare node and edge counts, training-table checks and bands for each configured `val_<metric>` and `test_<metric>`. Training metrics and losses are recorded but do not gate.
At 100M and 1B, an order-independent pandas hash covers all exported training rows.
At 10B and 25B, native metadata supplies total rows, per-split rows and binary positive counts without a full export.
Repeat and reload predictions need identical coordinates and score differences within the case tolerance.
If all three outputs agree, every prediction must still cover the exact sampled entity/anchor population.
Results record requested, scored and skipped counts for materialization and for each prediction output. These held-out cases require zero skipped rows. A population error names the failing prediction and its counts.
Forecast coordinates include the entity, original anchor and horizon column resolved from the scoring snapshot's graph metadata. Every sampled anchor must return each `FORECAST` horizon, so forecast output counts are horizon rows.
AUROC and AP, including their variants, use an absolute band of 0.03.
Other metrics, including MAE and RMSE, use `abs(current - reference) <= abs(reference) * tolerance`, with a relative tolerance of 25%.
Payments and fraud use balanced training sampling at batch 64 at every scale. At 10B and 25B, they also use one experiment and one epoch.
`majority_sampling_ratio=[1.0]` and `weight_mode=[WeightMode.SAMPLE]` affect training only.
Static link trains with `overlap_ratio=[0.5]`, so half of the supervision links are absent from message passing.
Every case and scale uses fixed validation and test budgets of 1,000 steps each.
Payments and fraud remain at every scale. References and fitted results require finite `val_<metric>` and `test_<metric>` for every configured metric. A reference without them fails before ingest.
Missing required metrics and any non-finite returned metric fail. Correct inadequate label support in the dataset before regenerating its reference after review.
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
| `ENGINE_E2E_TEMPLATES_JSON` | Regions in sweep order, each with a pinned regional template/version, one subnet per zone and an optional VM limit per type: `[{"region":"us-west-2","templates":{"g7e.24xlarge":{"template":"lt-example","version":"1","subnets":["subnet-example"],"limit":8}}}]`. Every selected type needs at least one region. Keep the list fixed until cleanup finishes. |
| `ENGINE_E2E_POOL_ROLE` | Main-only OIDC access to owned groups, derived template versions and schedules in every configured region, and `s3:GetObject` for admission objects under `ENGINE_E2E_RESULTS_S3`. |
| `ENGINE_E2E_AWS_REGION` | Home region of the Lambda, the sweep and report schedules and the results. Default `us-west-2`. |
| `ENGINE_E2E_RUNNER_APP_ID`, `ENGINE_E2E_RUNNER_APP_KEY` | Cleanup App credentials. |
| Lambda `RUNNER_SECRET` | Secrets Manager secret containing `app_id`, `installation_id` and `private_key`. |
| `ENGINE_E2E_CAPACITY_FUNCTION_ARN`, `ENGINE_E2E_CAPACITY_ROLE` | Registration, sweep and report Lambda and its Scheduler invocation role. |
| Lambda `RESULTS_S3` | Set to `ENGINE_E2E_RESULTS_S3`. The Lambda role needs secret reads and result-object reads/writes. |
| `ENGINE_E2E_EXPIRY_ROLE` | Scheduler access to delete only owned groups in every configured region. Its trust policy must allow schedules in every configured region. Limit launcher PassRole to approved roles. |

Package `.github/scripts/engine-e2e-capacity.cjs` and `engine-e2e-pool.cjs` as a Node.js 22 Lambda with handler `engine-e2e-capacity.handler`. Bundle `@aws-sdk/client-auto-scaling` and `@aws-sdk/client-s3` from one SDK release, 3.934.0 or later. The Lambda does not load with older versions. `@aws-sdk/client-auto-scaling` needs 3.934.0 for `LaunchInstances`.
Set the Lambda function timeout to about 100 seconds. It must be longer than the 45-second launch walk, one hung call of about 30 seconds, and the S3 and GitHub calls together. It must be shorter than the 120-second sweep lease. The asynchronous retry count can be 0.
If the Lambda runs in a VPC, it needs a network path to S3, Secrets Manager and the GitHub API. It also needs a path to the Auto Scaling endpoint of every configured region.
Deploy a new Lambda or change `ENGINE_E2E_TEMPLATES_JSON` only while no Engine run and no lifecycle cleanup is in progress. Set `ENGINE_E2E_SCHEDULER_ENABLED` to `false` first, so that no manual or scheduled run starts a pool. Set it to `true` again after the Lambda, its roles and `ENGINE_E2E_TEMPLATES_JSON` all match the code on main. A run keeps the event, record and runner name formats of the code that started it. New code cannot serve or clean up a run that old code started.
The App needs Administration write and Actions write for `NVIDIA/kumo-engine` only. Lambda keeps its keys and issues fresh registration tokens at boot.
In every configured region, the external Lambda role needs Auto Scaling `DescribeAutoScalingGroups` on `"Resource": "*"`, because Auto Scaling describe actions have no resource-level permissions. It needs `LaunchInstances`, `SetDesiredCapacity` and `TerminateInstanceInAutoScalingGroup` on owned groups only. `LaunchInstances` uses the launch permissions of the caller: `ec2:CreateFleet`, `ec2:DescribeLaunchTemplateVersions`, and `ec2:RunInstances` and `ec2:CreateTags` on the regional launch template, image, subnets, security group, instance, volume and network interface. The role also needs `iam:PassRole` for the VM role.
If the launch templates encrypt EBS volumes with a customer managed KMS key, the key policy in each region must allow these launches.
It also needs `s3:GetObject` and `s3:PutObject` on the `<run>/<attempt>/pools/` objects: sweep state, closed markers, pool failures and sweep records. It needs `s3:ListBucket`, restricted to the result prefix. The Lambda lists at most one matching key before reading a record. Missing records are absent. Permission and network errors remain failures.
The pool role's tag allowlist must include `EngineE2ESha`, `EngineE2EDeadline`, `EngineE2ECount` and `EngineE2EKey` alongside the existing ownership tags. `EngineE2ECount` holds the pool size. `EngineE2EKey` holds the SHA-256 hash of a random key that only the sweep schedule of the pool carries. Both are on the group only. VMs do not get them.
Before the Lambda changes a group, all owner tags of the group must match the sweep, including that hash. The subnets of the group must include the subnets of the sweep.
The VM profile invokes only this Lambda. VMs in every configured region call it in the home region. Repository workflow permissions control runner access. Labels select hardware without restricting workflow access.

Each declared instance type has one pool, with one group in each configured region. Every group starts with no VM and with `Launch` suspended.
Its VMs have scale-in protection, so a lower desired capacity never terminates a VM.
Every minute until the capacity deadline, the Lambda counts the VMs and waiting cases of a pool. In the launch walk, it asks one zone after the other for the missing VMs with `LaunchInstances`.
The walk starts no new launch call after 45 seconds. A call that started can still run for up to about 30 seconds. The sweep of the next minute continues at the next zone.
An error for one zone, such as no capacity, moves to the next zone. Some answers skip the rest of that region in that sweep.
These are a definite rejection (HTTP 4xx in one attempt) and a quota, permission or request-rate code. The codes are `VcpuLimitExceeded`, `InstanceLimitExceeded`, `UnauthorizedOperation` and `RequestLimitExceeded`.
A call with an unknown outcome (a timeout, a server error, or an error after SDK retries) stops the walk. Its requested VMs count as launched until `DescribeAutoScalingGroups` shows them, for up to ten minutes. In a region that the Lambda cannot read, they count until a sweep reads that region again.
A launch that returns VMs clears the saved errors of its zone and its region. When describe shows all VMs that unknown-outcome calls hold in a region, the Lambda clears the saved error of that region. The end of the ten minutes does not clear it. If no VM is missing and no call with an unknown outcome holds VMs, the sweep clears all saved zone errors.
If the Lambda cannot read the group of a region, it still counts the known VMs of that region. It changes nothing in that region in that sweep. If it can read no group, the sweep writes nothing.
Each sweep writes a record under `<run>/<attempt>/pools/` in `ENGINE_E2E_RESULTS_S3`.
Capacity waits are four hours nightly and 24 hours weekly. Case execution limits are 180 and 360 minutes, respectively.
Weekly replaces its VM after each case. Nightly can reuse VMs. Re-run all jobs to create a new pool for an attempt.
The Lambda treats a pool as drained only after all planned jobs are visible and none remains queued or waiting.
It then gives no runner token, and an unused VM exits cleanly. Existing workers continue. The sweep schedule ends at the capacity deadline.
Deletion is scheduled in the region of each group at the capacity deadline plus the case job timeout and a 30-minute cleanup buffer.
These safeguards use minute precision and are best effort.
Conditional S3 claims protect admitted cases. From the wait limit, EventBridge checks unstarted cases every ten minutes. Runs completed before that limit are skipped.
Boot failures, including runner registration and configuration, close the region of the VM for that pool. The closed marker saves the check and exit status.
The boot report does nothing else, and it needs no GitHub App access.
The sweep terminates a VM that has no runner 15 minutes after its launch. If cases wait, it closes its region. If cases wait, a VM that ends before its runner registers also closes its region.
While a region stays open, queued cases stay available to healthy workers. If the pool failed early, unstarted cases report the saved pool failure at the deadline.
Otherwise they report the last error code of every zone, and the code of every region whose last read failed. They also report the closed regions with their reasons, in at most 1000 characters.
A longer text first loses the region reasons and then the end of the zone list, so the closed region names stay.
Network reads retry up to five times. Registration and failure reports make up to nine attempts.
The waits between the attempts are 5, 10, 20 and 30 seconds, then 60 seconds four times. That is about five minutes of waits. The attempts stop before the capacity deadline.
A runner registration failure closes its region only after these retries.
Retries never repeat the SSD formatter. Hardware checks still fail immediately.
The Lambda checks pool ownership, pinned commit and deadline before it saves a boot failure.
A sweep fails the waiting cases at once only in this state:

1. Every region of the pool is closed.
2. At least one region closed for a reported boot failure that is not runner registration or configuration.
3. No VM of the pool is left.

The sweep then saves the pool failure, validates the live run and reports unadmitted cases immediately through lifecycle jobs with `phase=bootstrap`.
The case error is the first such boot failure in region order. If this step fails, the next sweep tries again.
The other close reasons never fail the waiting cases at once by themselves. These are runner registration or configuration failures, a VM without a runner after 15 minutes and a VM that ended without a runner.
The cases then wait for a healthy runner or for the deadline.
A reported boot failure that is not runner registration or configuration replaces one of these other close reasons in its region. A later close reason never replaces such a boot failure.
Stored claims retain the failure reason and protect admitted cases.
Capacity failures appear in a separate lifecycle run and preserve requested hardware.
After a four-hour capacity failure, the original nightly job can remain queued until GitHub's 24-hour queue expiry.
The source run and its cleanup can remain pending until those queued jobs expire.
Saved failure claims remain reportable when a run finishes at or after the deadline.
Cleanup first deletes owned groups in every configured region and waits for their deletion under one shared limit of ten minutes. If one group deletion fails, cleanup still deletes the other groups. It then removes owned template versions in each region and runner registrations.

Templates use the prepared GPU runner above, Python 3.12 with venv, AWS CLI, curl, `script`, XFS/RAID tools and IPv4/IPv6 iptables.
Install the pinned, unconfigured Actions runner at `/opt/actions-runner`. Attach only disposable nonboot NVMe disks for bootstrap formatting.
Bootstrap checks the actual type and free SSD, enables linger for shared memory and blocks worker access to instance metadata.
Verify image, IAM, registration, deadline reports and cleanup before schedule activation. Full-campaign quality and runtime remain separate from launch qualification.
CPU checks run without AWS in [OSS Readiness](../../../../.github/workflows/oss-readiness.yml).
