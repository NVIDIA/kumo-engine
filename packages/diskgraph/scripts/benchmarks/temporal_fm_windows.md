# Coarse-clock admission benchmark

This benchmark evaluates conservative indexed-clock admission followed by exact
row filtering. It does not evaluate range seeking, GNN-versus-RDL API performance,
FM payload assembly, or model accuracy. The native query still scans adjacency.

## Workload

[temporal_fm_windows.py](temporal_fm_windows.py) generates 32 roots and 74,880 events
over two years. Degrees are 16, 128, 1,024, and 8,192. At each degree, two roots
have each activity pattern: steady, 75% in the last 90 days, dormant for 180 days,
and 75% in the last seven days. Dense roots dominate the total event count.
The distributions are chosen scenarios, not fitted production data.

The 55 cases per clock cover recent/historical cutoffs, upper-only and
7/30/90/365-day windows, latest-24/latest-64 and uncapped sequences, narrow/wide
projections, shared sums, lengths, degree/activity cohorts, and unbounded and
mismatched-clock controls. Sort-key bucket steps are second, minute, hour, and day on seconds-based
indexes; all use 32-bit encodings. These bucket steps are distinct from the
index-wide storage unit. Correctness tests separately exercise clipping, nulls,
and exact row-decoding fallback for differently scaled index clocks.
The worker checks selected source IDs and sequence lengths independently of the
native implementation. Complete result hashes must match across builds.

## Reproduce

Use one Python environment with NumPy and PyArrow plus two separately built
DiskGraph packages. Each package-parent directory contains a `diskgraph/` package
and its extension. Use a separate `CARGO_TARGET_DIR` for each source revision
when building them, so switching trees cannot reuse the other build's extension.
From `packages/diskgraph`:

```bash
BENCH_DATA=$(mktemp -d)
BASE_PACKAGE=/path/to/baseline-package-parent
OPT_PACKAGE=/path/to/optimized-package-parent
for unit in s minute hour day; do
  PYTHONPATH="$BASE_PACKAGE" python scripts/benchmarks/temporal_fm_windows.py \
    "$BENCH_DATA/fm-$unit" --prepare --unit "$unit"
done
python scripts/benchmarks/temporal_fm_windows_compare.py "$BENCH_DATA" \
  --baseline "$BASE_PACKAGE" --candidate "$OPT_PACKAGE" \
  --output "$BENCH_DATA/paired" --cycles 8 --repeats 11
python scripts/benchmarks/temporal_fm_windows_compare.py "$BENCH_DATA" \
  --baseline "$BASE_PACKAGE" --candidate "$OPT_PACKAGE" \
  --output "$BENCH_DATA/boundary" --units minute hour day \
  --cutoff-offset-seconds 1379 --cycles 4 --repeats 11 \
  --case age1_upper_only_narrow_k24 --case age1_w7_narrow_k24 \
  --case age1_w30_narrow_k24
```

The [paired runner](temporal_fm_windows_compare.py) rejects an existing output
directory, rejects identical baseline/candidate native libraries, records their
hashes, checks that clock indexes use identical source data, and records raw
samples/hashes/counters. It runs adjacent baseline/candidate batches with
balanced AB/BA order. Linux workers share one CPU affinity. Each cycle starts
fresh worker processes; each batch has two warmups. Queries never overlap.
Use `--baseline-revision` and `--candidate-revision` to record build provenance.
The default confirmation set includes small-degree and uncapped cases. Use
repeatable `--case` flags to select cases; omitting `--case` on the worker itself
runs all 55 cases. Avoid simultaneous compilation or other benchmark campaigns.

Timings cover the public native API and Python result conversion. Preparation,
source checks, and hashing are excluded. Report median paired ratios together
with absolute times, variability, edges visited, and rows decoded. Do not infer
an improvement from a noisy unchanged control.

## Rebased validation (2026-11-03 UTC)

Current main `3d199ee657d88c927c83356a5cf3e5f5899153d8` was compared with the
revised native source at `a9efb9d050b6a50036963e02652d71593bc301d7`.
The release build, 411 Python tests and 11 focused Rust tests passed. The six
index-unit regressions also ran against unmodified main: seconds passed, while
all five non-second units produced wrong selected rows. The revised unit guard
uses exact row decoding for those encodings.

All **68 aligned cases and nine shifted-cutoff cases** matched between builds,
with independent source-ID/length checks. Aligned cases used eight pairs;
shifted cases used four; both used two warmups and eleven measured calls per
batch on one CPU. Native extension SHA-256 values:

- Main: `ac19a03cf15624ae293741ba92128c984e92298d8c5bb1de361779e8f9fcfe93`
- Revised: `68375881b49d40e4475745b2f29e1390d300fbab068aa8daa472c91e84c46a56`

| Query | Median paired speedup across coarse bucket steps |
| --- | ---: |
| Recent upper-only, latest 24 | Approximately unchanged |
| 7-day window, latest 24 | 3.96–4.48× |
| 30-day window, latest 24 | 3.12–3.45× |
| 90-day window, latest 24 | 2.23–2.27× |
| 365-day window, latest 24 | 1.50–1.52× |
| Historical upper-only, latest 24 | 2.35–2.41× |
| 30-day window, every eligible row | 3.22–3.28× |
| Degree 16, 30-day window | 1.84–1.93× |
| Degree 128, 30-day window | 2.85× |

Ranges include shifted cutoffs where measured. The aligned hour-bucket 30-day
query measured 23.804 ms versus 7.082 ms; rows decoded fell from 74,880 to
17,667 while edges visited remained 74,880. Separately aggregated times need
not divide to the median paired ratio.

Small upper-only queries were near parity. The earlier noisy 10% slowdown in
a degree-16 exact-second extraction comparison did not reproduce: this run's
paired ratio was 1.014×. There are still overhead cases: the shifted day-bucket
upper-only query was **1.5% slower**, and the exact-second 30-day length query
was **3.3% slower** (3.418 ms versus 3.528 ms). Unbounded/mismatched controls
were within about 1.1% of parity by median paired ratio. These measurements do
not establish zero regression on every ordinary workload.

## Earlier evaluation (2026-11-02)

The following measurements compare PR82 at
`7e5669f043a871e3a8ec433f1fd353dceb324feb` with the original coarse-admission patch,
before the rebase and shared-module extraction. They are not measurements of the
latest PR head. Baseline extension SHA-256:
`b93652471fff21c610b66680173c4d460051ef358a255b8e25776c4ceb9e8864`;
optimized extension SHA-256:
`93e6f3483670e8f2b8bf8fff16152bbc0dfb579530463a410d35f42b750bf5b5`.

The host had four vCPUs; measurements used one native thread and warm caches.
All 220 matrix cases and nine shifted-cutoff cases passed. Noise in the broad
matrix prompted confirmation of 48 key cases with eight adjacent pairs each,
two warmups and eleven measured calls/batch, pinned to one CPU. Shifted-boundary
cases used four pairs. Ratios below are medians of paired ratios, including
shifted cases where tested; exact reproduction of timings is not expected.

| Latest-24 narrow query | Speedup across minute/hour/day clocks |
| --- | ---: |
| Recent cutoff, upper bound only | Essentially unchanged |
| 7-day lookback | 3.9–4.4× |
| 30-day lookback | 3.2–3.4× |
| 90-day lookback | 2.2–2.3× |
| 365-day lookback | About 1.5× |
| Historical cutoff, upper bound only | 2.3–2.4× |

For the aligned hour-clock 30-day query, median batch times were 23.701 ms and
6.912 ms. Their ratio need not equal the median paired ratio. Rows decoded fell
from 74,880 to 17,667; edges visited remained 74,880.

With a cutoff 1,379 seconds inside a day bucket, the recent upper-only query
retained every candidate and was about **5% slower**. The coarse check adds work
without reducing decoding. This remains an explicit ordinary-case limitation;
no universal no-regression claim is made. In the same shifted 30-day case,
19,862 candidates refined to 17,705 eligible rows, with a 3.15× speedup.

Exact-second and unbounded/mismatched-clock controls had median paired ratios
within about 1.3% of 1×, though individual pairs had occasional outliers. Exact
second indexes remained faster for limited sequences. The 30-day lower bound
changes the latest-24 result for 16 of 32 roots, so choosing that window requires
model-quality evaluation. The original 14× narrow-window microbenchmark is a
favorable mechanism probe, not an expected FM improvement.
