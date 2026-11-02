# P1 CPU sampling check

P1 runs five sampling operations on the full commerce/100M graph. Each operation
consumes the frozen 300000 seed evaluations. It fetches features for every sampled
node. It does not construct, fit, or load a model.
The seed map has 100000 customers. Each operation repeats it three times.

The sampler and feature reader come from
`packages/diskgraph/scripts/benchmarks/commerce/dg_bench.py` in this repository.
The input bundle contains data only. `p1.json` uses format
`kumo-engine-weekly-p1-v1`. It declares `dataset`, `index`, `seedmap`, `population`,
`row_counts`, `files`, `work`, and `time_tolerance`. The `files` map binds every native index file,
seed map, and population file to its SHA-256 checksum.

The seed map contains `customer_id`, `anchor_s`, and `gid`. The population contains
`customer_id` and `anchor`. Workers make sure that business IDs and timestamps
match. They also make sure that the native global IDs resolve to those business
IDs. `work` fixes `n=300000`, `offset`, `batch`, `procs`, `threads`, and `hops`.
The approved reference and candidate use the same manifest checksum.

Each operation retains every worker's completion count, sampled-node and edge counts,
feature-row count, elapsed time, and membership checksum. Membership includes the
seed position, global node ID, node type, and hop for every sampled node. The
candidate must complete five operations with the same counts and membership
checksums as the approved reference. Each operation records elapsed seconds,
including child setup, sampling, feature fetch, and shutdown.

`time_tolerance` declares finite nonnegative `atol_seconds` and `rtol` values.
The candidate median must stay at or below
`reference_median * (1 + rtol) + atol_seconds`. Missing or invalid timing fails
the check. Reference review must approve the timing budget for its workload and
CPU capacity. Cache state remains uncontrolled, so the budget must account for
its variation.
This check does not gate memory use.

The [calibration record](engine-weekly-p1-calibration.json) contains three native
blocks with the published 1.0.0 wheel. All fifteen operations completed with the
same counts and membership. The three block medians range from 89.67 to 89.97
seconds. The proposed budget is ten percent plus five seconds above the middle
block's median, or 103.67 seconds. That allowance is operational headroom, not a
measured confidence interval. It still needs reference review. These direct
runs do not qualify the full GitHub workflow or a different candidate wheel.

Run the script with the selected wheel's isolated Python environment. Supply
`--input-root`, `--manifest-sha256`, `--output`, `--host-lock`, `--wheel`, and
`--wheel-sha256`. Reference preparation requires `--mode reference`. Candidate
work defaults to `--mode candidate` and also requires `--reference` and
`--reference-sha256`. The reference argument selects a separate reviewed
`reference.json` with format `kumo-engine-weekly-p1-reference-v1` and status
`CALIBRATED`. A raw `p1-result.json` cannot serve as an approved reference.

Preserve each raw result before choosing a timing budget. The reference copies
one complete block's `operations` and records `physical_fits=0`, the final
`manifest_sha256`, and `measurement_results_sha256` for the raw calibration blocks.
Its `workload_sha256` covers the input manifest except `time_tolerance` and
`calibration_only`. Thus a reviewer can approve a timing budget after measurement
without changing the measured data or work. Keep the original measurement manifest
and its checksum in the calibration evidence. Never rewrite a raw result to apply
a new budget. Review the counts, membership, CPU allocation, runtime, and observed
timing range before approving the reference checksum. A pass does not approve it.

The host lock covers input hashing, all five operations, and the final input audit.
Keep the evidence directory outside the immutable input tree. A failed worker causes
the runner to stop the other workers and retain the failure.

The weekly workflow adds a separate `p1` job for manual candidate runs. It waits
for the eleven-case GPU jobs to finish. It downloads their shared candidate wheel,
makes sure that the build receipt matches, and installs it in a fresh environment.
CUDA is hidden and the job sets eight threads per CPU worker. Its timeout is
120 minutes. Reference-preparation mode skips this job. There is no cron trigger.

Add `p1` to `KUMO_ENGINE_WEEKLY_CONFIG` after reference review. It requires
`runner` with the existing `group` and `labels`, absolute `input_root`, `reference`,
`host_lock`, and `work_root`, plus `manifest_sha256` and `reference_sha256`.
Use the host's shared lock and the CPU allocation used for calibration. The job
does not change these repository variables or approve a reference automatically.
Missing bindings stop configuration before native work starts.

The job uploads raw results and worker logs as a separate `weekly-p1` artifact.
It then removes its fresh environment and work directory. Immutable inputs and
the reviewed reference remain on the runner. A failed P1 job fails the workflow.

The focused CPU guards run with:

```bash
python packages/kumo-engine/test/test_weekly_p1.py
```
