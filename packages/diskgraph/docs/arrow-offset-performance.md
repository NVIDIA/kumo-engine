# Arrow offset performance assessment

These measurements describe the earlier always-wide temporal-query implementation
at `07c92802715e3a2e1e5baf8b5355bef98f529847`, before the explicit opt-in API.
The current API defaults to 32-bit offsets and selects the wide path only with
`arrow_offset_width=64`. No new performance benchmark was run for that API
revision. The measurements below are historical evidence, not a fresh comparison
against the current base or a claim that wide offsets are free. The generic
decoder inlining improvement is retained.

## Setup

The final comparison ran on an Apple Silicon Mac with 48 GiB RAM and 18 CPU
cores, on AC power, using Rust 1.96.1 and release builds. Baseline:
`3d199ee657d88c927c83356a5cf3e5f5899153d8`. The adjusted RowPack source SHA-256 is
`9b7696ed75a856c282a2dabc75ab508f60edac3c7eb9e869c5e25559da0acd04`.

Each case used nine paired process runs with randomized baseline/candidate
order and identical inputs/settings. Builds and tests from this comparison
ran separately from timing trials. macOS scheduled the processes; CPU affinity
was not pinned. Separate build directories and distinct executable hashes
prevented accidental reuse of the baseline executable. The candidate wheel
reused unchanged third-party dependency artifacts only after all 15 workspace
crates' artifacts were removed and the wheel's workspace dependencies rebuilt.

## Decoder results

One in seventeen rows was null. Each process warmed up with three decodes,
then timed the indicated number of iterations, including allocation and drop.

| Case | Rows / iterations | Candidate/baseline median time | Allocated Arrow buffers, baseline / candidate (MiB) | Peak process RSS, baseline / candidate (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Integer | 100,000 / 600 | 0.940 | 0.77 / 0.77 | 8.81 / 8.78 |
| 8-byte string | 100,000 / 600 | 0.973 | 3.45 / 3.83 | 9.19 / 9.53 |
| 8-byte binary | 100,000 / 600 | 0.997 | 3.45 / 3.83 | 9.17 / 9.52 |
| Four short strings per list | 100,000 / 300 | 0.973 | 24.93 / 37.64 | 11.53 / 12.78 |
| 1,024-byte string | 30,000 / 200 | 0.993 | 29.41 / 29.53 | 1802.44 / 1802.52 |

The original refactor was 9.8% slower for integers and 15.5% slower for binary
values. Its 32-bit path reproduced essentially the same slowdowns, isolating
the shared refactor rather than offset width alone. Inlining the generic row
decoder into the batch loop removed those measured slowdowns. The adjusted
cases show no median decoder throughput slowdown; overlapping trial ranges
mean small differences should not be presented as speedup claims. A follow-up on the original 32-bit path measured time ratios
of 0.941 for integers and 1.005 for binary values, with overlapping binary
ranges; the earlier large regression is removed on that path too.

Allocated buffers include reserved capacity and are not equivalent to RSS.
For lists, allocated capacity increases by 51%, while measured Mac peak RSS
increases by about 11%. Wider offsets cost four additional bytes per offset;
the existing generous child-buffer reservations amplify allocated capacity.
The large long-string process RSS is present in both versions and is distinct
from the approximately 29.5 MiB output-column buffer capacity.

## Full temporal queries

Both wheels read the same generated index: 100 users and 10,000 events, with
numeric, string, binary, and string-list features and periodic nulls. Each query
used 256 root-node requests and a sequence limit of 32, returning 8,192 rows. Each process performed an initial
output check, five warm-ups, and 100 timed queries. Timing counters were
excluded from the output digest; all returned values and deterministic
statistics matched across baseline and candidate in all trials.

| Output format | Median time for 100 queries, baseline / candidate (s) | Median per-process p95 query latency, baseline / candidate (ms) | Median peak RSS, baseline / candidate (MiB) |
| --- | ---: | ---: | ---: |
| `model` | 0.27925 / 0.28196 | 2.915 / 2.951 | 68.66 / 69.50 |
| `inspectable` | 0.28904 / 0.29178 | 2.981 / 3.026 | 68.94 / 69.97 |

Median total time is approximately 1% higher for both formats. The ranges
overlap, but `inspectable` was slower in eight of nine paired trials: the small
cost must not be dismissed or described as proven performance-neutral. The
`model` pairs were mixed. These are synthetic Mac measurements, not a guarantee
for other workloads or architectures. The revised Linux throughput has not
been measured; the earlier busy-host Linux comparison was inconclusive.

## Correctness checks

- 321 ordinary RowPack tests passed after the inlining adjustment.
- Six focused Rust temporal-query unit tests passed, including the new string-aggregate case.
- 340 affected Python integration tests passed against the candidate Mac wheel.
- The explicit >2 GiB synthetic regression passed on Linux before the
  performance-only inlining adjustment. It confirms failure of the narrow
  decoder and exact values from the wide decoder. It was not repeated on the Mac.

## Reproduce

From `packages/diskgraph`, build the standalone decoder example in release
mode and time the executable rather than `cargo run`:

```sh
CARGO_TARGET_DIR=/tmp/dg-offset-candidate cargo build --release --locked \
  -p diskgraph-rowpack --example arrow_offset_bench
/usr/bin/time -l /tmp/dg-offset-candidate/release/examples/arrow_offset_bench \
  list 100000 300
```

Use `/usr/bin/time -v` on Linux. Other case names are `numeric`, `short`,
`binary`, and `long`; arguments are case, row count, and timed iterations.
For the baseline, copy the same example into a separate checkout of the pinned
main commit and replace its two `decode_batch_to_arrow_large` calls with
`decode_batch_to_arrow`. Use a different `CARGO_TARGET_DIR`.

For the query benchmark, use baseline and candidate wheels in separate Python
environments. Run the same example with each environment's Python, reusing the
same generated index root; the label identifies the result but does not select
the installed wheel:

```sh
python diskgraph-pybind/python/examples/arrow_offset_query_bench.py \
  /tmp/dg-offset-query baseline model
python diskgraph-pybind/python/examples/arrow_offset_query_bench.py \
  /tmp/dg-offset-query candidate model 64
```

The `64` argument selects the current opt-in path. Omit it only when replaying
the historical always-wide candidate wheel; old wheels do not accept the flag.

Repeat with `inspectable`. Alternate the environments across trials, retain
all results, and require matching value/statistics digests. The example creates
its own small input tables; it needs no external data or service.
