# Arrow offset performance assessment

Status: **not cleared for readiness**. The overflow regression passes, but
these measurements do not establish absence of performance regressions.

## Method

Compared main `3d199ee657d88c927c83356a5cf3e5f5899153d8` with the
large-offset implementation in `958b33d10093ef8a1c43a98be7d94d762a68d86e`.
Both used the same Linux host, compiler, locked dependencies, release profile,
synthetic values, and CPU affinity. Each case had nine paired process runs
with randomized baseline/candidate order, three warm-up decodes, and identical
iteration counts. Runs were serial. One in seventeen rows was null.

The host had substantial competing activity. Observed ranges overlapped
widely, including the unchanged numeric output. Ratios below are descriptive
medians, **not statistically established regressions or improvements**.

| Case | Rows | Iterations per trial | Candidate/baseline time | Baseline/candidate allocated Arrow buffers (MiB) | Baseline/candidate median peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Integer | 100,000 | 600 | 1.103 | 0.77 / 0.77 | 4.01 / 4.04 |
| 8-byte string | 100,000 | 600 | 1.008 | 3.45 / 3.83 | 4.01 / 4.04 |
| 8-byte binary | 100,000 | 600 | 1.067 | 3.45 / 3.83 | 4.04 / 4.01 |
| Four short strings per list | 100,000 | 300 | 0.870 | 24.93 / 37.64 | 6.01 / 8.04 |
| 1,024-byte string | 30,000 | 200 | 0.974 | 29.41 / 29.53 | 30.22 / 30.30 |

Allocated Arrow buffer capacity includes reserved capacity; it is not the same
as resident memory. The list case increases allocated capacity by 51% and
median peak RSS by approximately 34%. This is an actionable memory cost, even
though the timing result is inconclusive. Wider offsets and existing generous
child-buffer reservations both matter. The change must not be described as
performance-neutral on this evidence.

An earlier comparison accidentally reused the baseline binary through a shared
Cargo target cache. It was discarded. The reported comparison explicitly
rebuilt the candidate and verified distinct executable hashes. Use separate
build directories when reproducing.

## Reproduce

The example is standalone and needs no external data or service. Build in
release mode, then time the executable rather than `cargo run`:

```sh
cd packages/diskgraph
CARGO_TARGET_DIR=/tmp/dg-offset-candidate cargo build --release --locked \
  -p diskgraph-rowpack --example arrow_offset_bench
/usr/bin/time -v /tmp/dg-offset-candidate/release/examples/arrow_offset_bench \
  list 100000 300
```

Other case names are `numeric`, `short`, `binary`, and `long`. Arguments are
case, row count, and timed iterations. The example reports elapsed seconds
and allocated output buffer capacity.

For the baseline, copy the same example into a separate checkout of the
pinned main commit and replace its two `decode_batch_to_arrow_large` calls
with `decode_batch_to_arrow`. Build with a different `CARGO_TARGET_DIR`.
Alternate the executables on one fixed CPU, keep inputs and settings identical,
and retain all trials rather than only the fastest result.

Before readiness: investigate the list memory cost, repeat timing on a quiet
host, and complete current-base Python integration and end-to-end temporal-query
measurements with identical output values. The real >2 GiB correctness test is
separate: the baseline cannot successfully finish that workload, so it cannot
provide a throughput comparison there.
