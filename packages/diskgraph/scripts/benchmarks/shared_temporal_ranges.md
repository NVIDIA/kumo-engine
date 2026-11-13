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
