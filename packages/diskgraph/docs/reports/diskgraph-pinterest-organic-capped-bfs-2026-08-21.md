# Organic Pinterest graph: capped-neighborhood mmap versus cached O_DIRECT

This experiment replaces the earlier synthetic Pinterest hotspots with a
deterministic broken-power-law generator. It starts from real pin primary keys,
returns real primary keys for every sampled node, and compares mmap with cached
O_DIRECT on a breadth-first neighborhood capped at 80 distinct neighbors per
seed. The report includes the original 515.132 GiB index and a follow-up
1,217.057 GiB index built specifically to control for the original hotspot
index's 1,177.447 GiB size.

On the size-matched index, cached O_DIRECT reaches 9.087M nodes/s for uniformly
selected roots and 9.382M nodes/s for Pinterest-style skewed roots. The
matching mmap results are 1.702M and 1.879M nodes/s, so Direct is 5.340 and
4.994 times faster. The Direct cache hit rates are 50.39% and 49.61%: request
skew reduces physical reads per returned node, but it does not increase the 4
KiB page-hit rate at this scale.

For reference, the initial 515.132 GiB index produced 19.383M and 20.876M
Direct nodes/s versus 9.490M and 10.422M for mmap. Its much higher 81--83% hit
rate primarily reflected the fact that a 320 GiB cache covered most of the
smaller index.

## Size-matched rerun

The first organic index was only 1.024 times host RAM, so its 81--83% Direct
cache hit rate could not answer whether the organic distribution improved
reuse at the original hotspot index's scale. The size-matched configuration
scales every entity and relationship table by 2.285719 times. Scaling the
whole graph, rather than adding edges to the original node population,
preserves node-type ratios, mean degree, the configured broken-power-law
curves, and the popularity permutations used across edge families.

The configuration contains 34.766B entity rows and 64.594B source edge rows,
or 99.360B source rows in total. Generation used 192 workers and completed in
2m31s, producing 8,128 Parquet files and 992.146 GiB of Zstd-compressed input.
The ingest completed in 33m03s under a 500 GiB no-swap limit and produced:

- 34.766B nodes across four entity tables;
- 129.188B directed adjacency entries across all 16 relation directions;
- zero primary-key duplicates, dropped timestamps, parse failures, lookup
  failures, or membership failures; and
- a 1,217.057 GiB index, 2.420 times the host's 502.973 GiB RAM and 3.36%
  larger than the 1,177.447 GiB hotspot index.

Peak anonymous RSS during ingest was 312.8 GiB. The source-to-index counts in
`ingest_stats.json` exactly match the configuration. The size-matched generator
configuration is
`diskgraph-datagenerator/configs/pinterest_omnisage_pruned_organic_size_matched.json`
(SHA-256 `7535fdc990c57562a7399d6b12ffb1c40838d3f279a3e17ac3b2421a3dbcd570`).

### Size-matched protocol

The sampling contract is unchanged: real `pin_id` inputs from all 512 source
Parquet files, returned real primary keys, three hops over all 16 relation
directions, and at most 80 distinct neighbors per root. Both retained input
streams span effectively the entire 9.966B-key pin table. Uniform roots use
10M warmup roots and 10M measured roots. Pinterest-style skewed roots use 20M
warmup roots and 10M measured roots. Every backend starts in a fresh 500 GiB
no-swap process after file-scoped eviction.

mmap uses the agreed 256 sampling contexts. Direct uses the highest-throughput
point from a 192/256/320/384-context sweep: 384 contexts, 32 io_uring owners at
QD32 each, adjacent-read coalescing up to 16 pages, and a 320 GiB fixed cache.
The Direct sweep rose from 8.647M nodes/s at 192 contexts to 8.916M at 384 in
short diagnostic windows; most of the gain was already present by 256. The
retained windows below are longer and begin after the cache is full.

### Size-matched results

| root distribution | backend | contexts | nodes/s | edges/s | roots/s | physical IOPS | fio utilization | avg QD | reads/node | Direct hit | avg cores | CPU-s/M nodes | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | mmap | 256 | 1.702M | 1.978M | 21.008k | 1.014M | 15.4% | 64.1 | 0.596 | kernel | 91.6 | 53.830 | 1.000x |
| uniform | cached O_DIRECT | 384 | **9.087M** | **10.558M** | **112.189k** | **5.993M** | **91.1%** | **715.1** | 0.660 | 50.39% | **59.1** | **6.504** | **5.340x** |
| Pinterest request skew | mmap | 256 | 1.879M | 2.117M | 23.193k | 1.087M | 16.5% | 69.6 | 0.579 | kernel | 93.7 | 49.876 | 1.000x |
| Pinterest request skew | cached O_DIRECT | 384 | **9.382M** | **10.574M** | **115.823k** | **5.992M** | **91.1%** | **713.9** | 0.639 | 49.61% | **58.1** | **6.193** | **4.994x** |

For uniform roots, Direct returns 5.340 times as many nodes per second while
using 35.5% fewer average CPU cores. For skewed roots, it is 4.994 times faster
while using 38.0% fewer cores. Normalized CPU time falls by 87.9% and 87.6%,
respectively. Both Direct runs sustain about 5.99M physical reads/s, or 91.1%
of the 6.577M fio reference, while mmap reaches only 1.01--1.09M IOPS. The
Direct path is storage-bound without saturating the CPUs; mmap leaves both
resources underused because blocking faults expose only QD64--70.

All four measured caches are stable. mmap stays near 459 GiB for uniform roots
and 456--457 GiB for skewed roots. Direct stays at 319.7--319.9 GiB in both
windows. Every retained measurement is 86 seconds or longer, neither process
swaps, verification passes, and the sampler reports zero tracked heap
allocations.

### What happened to the hit rate

The size-controlled result does not show a higher 4 KiB page-hit rate. Uniform
roots perform 1.330 cache lookups per returned node and hit 50.39%, yielding
0.660 physical reads per node. The Pinterest-style root skew performs only
1.271 lookups per returned node but hits 49.61%, yielding 0.640 misses per
node. The accounting agrees with the independent device counters:

```text
uniform: 1.330 lookups/node x (1 - 0.5039) = 0.660 misses/node
skewed:  1.271 lookups/node x (1 - 0.4961) = 0.640 misses/node
```

The skewed stream is still useful: it reduces physical reads per returned node
by 3.2% and raises Direct throughput by 3.2%. It does so by reducing total page
demand, not by raising the page-hit fraction. This is consistent with the
capped traversal filling its neighborhood earlier around popular roots, while
the second and third hops still fan out across a broad set of neighbors.

The original 515.132 GiB organic index reported 81.58% uniform and 82.59%
skewed hit rates because a 320 GiB cache covered a much larger fraction of that
index. At 1,217.057 GiB, the corresponding rates fall to 50.39% and 49.61%.
The original hotspot graph reported 55.16% at a similar index size, but it used
a different two-hop direction-reversing plan. The size control therefore rules
out index size as the explanation for the organic graph's earlier high hit
rate; it does not isolate data distribution from traversal behavior. Under the
actual three-hop capped workload, the new generator and request skew provide
no evidence of a more favorable 4 KiB cache hit rate.

## Initial 515 GiB dataset generation

The generator maps each edge-table row to a destination key without storing a
degree array:

1. An affine permutation of the source-row ordinal makes the assignment
   independent of Parquet shard boundaries.
2. `minimum_degree` reserves exactly that many rows for every destination key.
3. Remaining row ordinals are evenly spaced quantiles of a continuous,
   piecewise power-law CDF. Inverting the CDF yields a popularity rank.
4. A second affine permutation maps popularity rank to primary-key ID. Edge
   families can share this permutation, so the same entities are popular
   across relations without placing all hubs in one numeric ID range.

The work per row is bounded by the number of configured curve segments (four
here). Workers share no degree counters, locks, or generated-key state, so all
8,128 Parquet shards can be generated independently and deterministically.

The corrected dataset was generated with 192 workers in 66.5 seconds. It
contains 43.47B source rows in 396.302 GiB of Zstd-compressed Parquet:

| entity | nodes | conceptual outbound rows | mean outbound degree |
|---|---:|---:|---:|
| pin | 4.36B | 9.592B | 2.2 |
| board | 4.70B | 10.340B | 2.2 |
| item | 2.82B | 3.666B | 1.3 |
| query | 3.33B | 4.662B | 1.4 |

The final index contains exactly 15.21B nodes and 56.52B directed adjacency
entries across 16 relation directions. It occupies 515.132 GiB, or 1.024 times
the host's 502.973 GiB RAM. Ingest completed in 11m12s under a 500 GiB no-swap
limit, peaked at 302.5 GiB RSS, and reported zero duplicate primary keys and
zero dropped timestamps.

## Initial degree validation

The following values are from one million uniformly selected nodes of each
type in the completed index. `max sample` is not the whole-population maximum,
so it is expected to miss rare hubs.

| direction | type | p50 | p90 | p99 | p99.9 | max sample | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| outbound | pin | 1 | 4 | 14 | 36 | 211 | 2.1994 |
| outbound | board | 1 | 4 | 14 | 35 | 4,597 | 2.2079 |
| outbound | item | 1 | 2 | 5 | 13 | 62 | 1.3001 |
| outbound | query | 1 | 2 | 7 | 40 | 739 | 1.4003 |
| inbound | pin | 1 | 8 | 92 | 524 | 5,799 | 5.2928 |
| inbound | item | 1 | 3 | 9 | 37 | 828 | 1.7016 |

The supplied pre-pruned pin reference is 1/12/110/575 with a maximum of
41,777. The configured whole-population pin curve is approximately
1/10/93/528 with a maximum of 38,801; the measured sample is consistent with
it. The outbound targets are also close to the supplied percentiles and means.

The pre-pruned inbound means cannot coexist with these pruned node and edge
counts: the reference pin mean alone would require 34.0B incoming pin edges,
more than the pruned graph's 28.26B source edges of every type combined. This
dataset therefore prioritizes the requested p50=1 behavior, the pin tail shape,
and the pruned outbound means. Item indegree remains less skewed than the
pre-pruned reference and is a documented limitation rather than an exact
Pinterest replica.

## Initial 515 GiB sampling protocol

- Host: AMD Ryzen Threadripper PRO 9995WX, 96 physical / 192 logical CPUs,
  502.973 GiB RAM.
- Storage: four-NVMe md/XFS RAID0. The retained fio reference is 6.577M 4 KiB
  random-read IOPS at aggregate QD1024.
- Memory: each backend runs in a fresh `MemoryMax=500G`, `MemorySwapMax=0`
  scope after file-scoped `POSIX_FADV_DONTNEED` eviction. Offsets are
  prefaulted before warmup.
- Input and output: real `pin_id` values are read from all 512 original pin
  Parquet shards. Timed work includes primary-key-map lookup, sampling, and
  returned-primary-key materialization. The harness requires exactly one
  returned primary key per sampled node.
- Traversal: three breadth-first hops over all 16 relation directions. A root
  accepts at most 80 distinct neighbors; once full, it does not expand on a
  later hop. Edges whose endpoints are already admitted remain valid. Equal-hop
  completion order is nondeterministic, so verification enforces the cap and
  node cardinality while reporting small edge-count drift.
- Root streams: `uniform` draws every source-table row with equal probability.
  `Pinterest request skew` sends 60% of requests to the hottest 1.06% of pins,
  80% to the hottest 3.89%, and 20% across the remainder. The rank-to-ID
  permutation is the same one used by the generator.
- Concurrency: mmap peaks at 160 sampling contexts. Direct peaks at 192 with
  32 io_uring owners, QD32 per owner, a 320 GiB fixed cache, and adjacent reads
  coalesced up to 16 pages.
- Stable windows: uniform uses 10M warmup roots and 10M measured roots. The
  skewed stream requires 20M warmup roots before the same 10M-root measurement.
  All retained measurement windows exceed 30 seconds.

Plan digest: `125e41ecd63aaebf`.

## Initial 515 GiB results

| root distribution | backend | threads | nodes/s | edges/s | roots/s | physical IOPS | fio utilization | avg QD | reads/node | Direct hit | avg cores | CPU-s/M nodes | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | mmap | 160 | 9.490M | 11.041M | 117.2k | 0.277M | 4.2% | 11.3 | 0.029 | kernel | 111.4 | 11.739 | 1.000x |
| uniform | cached O_DIRECT | 192 | **19.383M** | **22.550M** | **239.3k** | **4.736M** | **72.0%** | **496.1** | 0.244 | 81.58% | **97.9** | **5.051** | **2.042x** |
| Pinterest request skew | mmap | 160 | 10.422M | 11.743M | 128.7k | 0.345M | 5.2% | 14.3 | 0.033 | kernel | 111.9 | 10.736 | 1.000x |
| Pinterest request skew | cached O_DIRECT | 192 | **20.876M** | **23.509M** | **257.7k** | **4.593M** | **69.8%** | **478.6** | 0.220 | 82.59% | **98.0** | **4.694** | **2.003x** |

Direct's measured cache contains about 295 GiB at the end of both retained
windows. Its high hit rate is expected: the data cache is 57% of the complete
index, adjacency access is not uniform, and many node records share one 4 KiB
page. The hit counter measures explicit page lookups, while nodes/s counts all
returned graph nodes; there is no requirement for one physical read per output
node.

The kernel path keeps about 451 GiB of index pages and consequently performs
fewer physical reads per node. It is still slower because its misses arrive as
blocking page faults and the hot page-fault/page-cache path consumes more CPU.
Direct groups misses explicitly and sustains roughly QD480--500, reaching
4.59--4.74M workload IOPS while all four drives are busy. The remaining gap to
the 6.577M fio ceiling reflects a less regular multi-file graph workload and a
lower average queue depth than the synthetic QD1024 fio point; the benchmark
does not issue extra reads merely to raise IOPS.

Both retained Direct measurements report zero tracked heap allocations during
sampling. Cached state, frontier storage, relation plans, and capped-neighbor
counters are reused from the sampling-state pool.

## Reproduction

```bash
# Uniform roots
DG_BENCH_THREADS=192 DG_BENCH_BACKENDS=direct \
DG_BENCH_WARMUP_SEEDS=10000000 DG_BENCH_NUM_SEEDS=10000000 \
scripts/bench_pinterest_organic_pkey.sh INDEX SOURCE/diskgraph_config.json

# Pinterest request skew
DG_BENCH_THREADS=192 DG_BENCH_BACKENDS=direct \
DG_BENCH_WARMUP_SEEDS=20000000 DG_BENCH_NUM_SEEDS=10000000 \
DG_BENCH_SEED_RANK_COVERAGE='0.0106:0.60,0.0389:0.80,1:1' \
DG_BENCH_SEED_RANK_PERMUTATION_SEED=1102 \
scripts/bench_pinterest_organic_pkey.sh INDEX SOURCE/diskgraph_config.json
```

Use `DG_BENCH_THREADS=160 DG_BENCH_BACKENDS=mmap` for the corresponding mmap
controls. The preset contains the complete three-hop relation plan and the
80-neighbor cap.

For the size-matched index, use 256 mmap contexts or 384 Direct contexts. The
uniform retained runs use 10M warmup and 10M measured roots. Add the rank
coverage settings shown above and increase warmup to 20M roots for the skewed
pair.
