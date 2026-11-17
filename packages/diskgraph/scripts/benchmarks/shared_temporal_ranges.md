# Shared temporal candidate-range benchmark

This benchmark compares the shared row-query implementation from PR #310 with
candidate-range seeking applied to that same implementation. It does not compare
different timestamp policies or establish FM model-quality, GPU, cold-cache or
end-to-end training gains.

Build both revisions with the same Rust compiler, release flags and Python
environment. Put each revision's `diskgraph` Python package and its corresponding
`_c.so` in separate directories. Both builds must use the same ingestion code.

```sh
python shared_temporal_ranges.py /tmp/shared-range-results \
  --baseline /path/to/baseline-package \
  --candidate /path/to/candidate-package \
  --degrees 256 131072 --cycles 4 --repeats 7
```

The output directory must be new. The runner validates both native libraries
before creating it. It records their hashes and each fixture's Parquet hash.
Four independent anchors each have the requested degree. Second-clock fixtures
use shifted minute sort-key buckets; nanosecond fixtures use full-width keys.
Historical queries select the oldest 1% of each history, without a sequence cap.
The sum case also returns all selected IDs to exercise a shared row frame rather
than the separate specialized numeric executor. Length queries are uncapped.
Recent latest-128, unbounded, and mismatched-clock queries are controls.

For every case, four adjacent pairs alternate baseline/candidate order. Each
worker is a fresh process pinned to the same available CPU, with one native
thread, two warmups, and seven measured calls. Timings include native execution
and Python result conversion; fixture preparation, independent source-ID/sum/
length assertions, and hashing are excluded. Both workers open the same index.
Returned values must match the source-derived oracle and cross-build hashes.
The output retains individual times, work counters, ratios, and paired medians.
Small controls may be dominated by process/cache variability; do not generalize
a selective high-degree speedup to every query.

## Validation of the reworked implementation

Baseline: PR #310 at `5e1448bf84f5d640b1aaa846cc28080f6eb86e4e`.
Candidate implementation: `2e9784749f8b81c3c76333a4975332cb2d30a310`.
The candidate preserves `frame_bounds` and `row_clock_time` unchanged.

- Native Python clock, aggregation, temporal-query and sampling suites: 481 passed.
- Kumo-pytorch clock, anchor and materialized-batch suites: 246 passed, with
  required imports enabled.
- The new regression file contains 45 cases. On the baseline, its 33 semantic
  cases pass and its 12 work-reduction checks fail as expected. All 45 pass on
  the candidate.
- Rust core: 142 passed; engine: 58 passed; ingestion library: 488 passed,
  one failed, two ignored. Shared-query library and the remaining ingestion/
  train-table integration binaries: 782 passed, eight ignored. An additional
  engine integration case is ignored by the repository.
- The failure is `disk_manager::tests::test_eviction_under_real_disk_pressure`
  (`InsufficientSpace`). It also fails when rerun alone using temporary memory
  storage. The disk-manager module is unchanged from the baseline; this PR
  does not fix that separate failure. The full Rust validation is not green.
- New Python files pass Ruff's undefined-name/error checks; the diff passes
  whitespace checks. Rust tests use release builds with installed Rust 1.99.0;
  the repository pins 1.99.1. Rustfmt and Clippy are unavailable on this host.

The initial Cargo invocation built the four affected crates' libraries and
integration tests. After the ingestion-library failure, the already-built
remaining test executables were run directly from their crate directories.
No assertions were weakened or tests changed to hide the disk-manager failure.

## Measured incremental performance (2026-11-13)

Linux x86-64, Intel Xeon 6975P-C, CPU 0, release builds with identical compiler
and flags, Python 3.11.15 environment. Four paired AB/BA cycles, seven measured
calls per worker, two warmups. Compilation and validation finished before the
benchmark. All 24 cases match independent source answers and cross-build hashes.

The large fixture has four anchors with 131,072 events each. Historical windows
return every eligible row: 1,310 per anchor, with no sequence cap.

| Clock / query | Baseline ms | Candidate ms | Paired speedup (range) |
| --- | ---: | ---: | ---: |
| seconds / historical_rows | 116.663 | 1.547 | 77.07× (71.47–93.74) |
| seconds / historical_lengths | 112.548 | 1.023 | 110.01× (90.39–111.89) |
| seconds / historical_sum | 117.902 | 1.761 | 67.32× (66.68–67.80) |
| seconds / recent_latest128 | 155.146 | 154.781 | 1.00× (0.99–1.01) |
| seconds / unbounded | 183.832 | 184.609 | 1.00× (0.97–1.02) |
| seconds / mismatched | 116.901 | 116.827 | 1.00× (0.99–1.01) |
| nanoseconds / historical_rows | 7.944 | 1.540 | 5.16× (5.10–5.21) |
| nanoseconds / historical_lengths | 29.951 | 0.236 | 127.09× (116.97–128.51) |
| nanoseconds / historical_sum | 8.038 | 1.603 | 5.02× (4.99–5.07) |
| nanoseconds / recent_latest128 | 38.522 | 36.053 | 1.07× (1.06–1.10) |
| nanoseconds / unbounded | 166.360 | 165.640 | 1.00× (0.99–1.01) |
| nanoseconds / mismatched | 117.533 | 118.138 | 0.99× (0.99–1.00) |

For historical rows, lengths and sums, unpacked candidates fall from 524,288 to
5,284 with compressed minute keys, and to 5,240 with full-width nanosecond keys.
Coarse-key candidate rows still undergo stored-row refinement. Large unbounded
and mismatched controls remain within 1% of parity by median paired ratio.

At degree 256, selective seconds-clock queries measure 3.18–3.65× and
nanosecond-clock queries 1.54–7.95×. These are submillisecond measurements.
Small controls range from 0.968× to 1.030× by median paired ratio; the small
nanosecond mismatched-clock control shows about 3.3% overhead, with pair ratios
0.88–0.98. There is no universal small-query speedup claim.

These gains depend on selective windows and index compatibility. Large clipped
sentinel regions still require refinement, and broad queries retain their scan
cost. Separately aggregated timing medians need not divide to the paired ratio.

Receipt identity (SHA-256):

- baseline native library: `8bf79b064372619bdead6babf529972e3b3773ede7147ee2f148bb22d8d519ae`.
- candidate native library: `4e4058384e5b2fe4bc208af179a32e2945e51d9bfe2806634ac4b0def0a7113c`.
- Raw benchmark summary: `2595042417d9ad89f2e835711caa772a76d61dc0a75e4014c4854f979cb1c048`.
