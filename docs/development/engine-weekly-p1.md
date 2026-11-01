# P1 CPU sampling check

P1 runs five sampling operations on the full commerce/100M graph. Each operation
consumes the frozen 300000 seed evaluations. It fetches features for every sampled
node. It does not construct, fit, or load a model.

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
The reference and candidate use the same manifest checksum.

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
its variation. This code change does not establish a calibrated timing budget.
Memory remains a report.

Run the script with the selected wheel's isolated Python environment. Supply
`--input-root`, `--manifest-sha256`, `--output`, `--host-lock`, `--wheel`, and
`--wheel-sha256`. Reference preparation requires `--mode reference`. Candidate
work defaults to `--mode candidate` and also requires `--reference` and
`--reference-sha256`. The reference argument selects the reviewed `p1-result.json`.
Reference approval is a separate review, not an automatic consequence of a pass.

The host lock covers input hashing, all five operations, and the final input audit.
Keep the evidence directory outside the immutable input tree. A failed worker causes the runner to
stop the other workers and retain the failure. This check has no scheduled job.
The native workload and input conversion still need validation on an admitted
host before reference approval or workflow activation.

The focused CPU guards run with:

```bash
python packages/kumo-engine/test/test_weekly_p1.py
```
