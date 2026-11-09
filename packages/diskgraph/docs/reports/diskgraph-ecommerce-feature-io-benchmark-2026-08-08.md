# Feature-heavy e-commerce sampler I/O benchmark (2026-08-08)

Status: complete. Every scale row is a paired fresh-process mmap and Direct-I/O
comparison that passed the output-digest and resource-saturation gates.

## Workload

- Four node tables: `users` (10%), `items` (10%), `orders` (78%), and
  `merchants` (2%).
- User and item rows contain 768- and 256-element float arrays. RowPack infers
  float16 storage, producing approximately 2 KiB of embedding data per node.
  Order and merchant rows remain below 200 bytes.
- Every order links to one user, one item, and one merchant. Item selection is
  hotspot distributed (1% of items receive 80% of orders); merchant selection
  is also skewed (5% receive 70%).
- Sampling starts at `users` and traverses all available relations for four
  hops with fanout 10. Canonical plan: `default:4x10:Uniform`, digest
  `5d2e1672de275d8d`.
- `--fetch-features` materializes features for every sampled node type after
  topology sampling. Reported nodes/s is therefore end-to-end topology plus
  feature I/O, RowPack decode, dedup expansion, and Arrow output construction.

The scale configs use 280M / 1.4B / 7B / 14B / 28B total source rows for
1% / 5% / 25% / 50% / 100%. The benchmark host has 502.973 GiB of physical
RAM; the index/RAM ratio is reported explicitly because it is the primary
predictor of the resident-to-storage-bound transition.

## Correctness and implementation gates

- A 1M-row smoke ingest produced 1M nodes, 4.68M directed relation entries,
  a 62 MiB compressed-Parquet source, and a 442 MiB final index. User and item
  `edges.bin` files were 209.5 MB and 212.8 MB respectively.
- Identical Direct-I/O and mmap runs produced topology digest
  `db23f164b7af0778`, 418,810 nodes, and 564,906 edges for the same 1,000
  seeds.
- Consecutive physical misses are coalesced with `Readv` into at most 16
  adjacent 4 KiB buffers. Sparse/single-page batches retain the original
  one-SQE-per-page path.
- The final partial page is accepted using its exact EOF-valid byte count; an
  O_DIRECT integration test covers a two-page vectored read ending in a short
  tail.
- Parsed header, relation-plan, selected-index, shard-group, frontier, raw-edge,
  and offset-range buffers are retained in pooled sampling state.

## Allocation and hot-path optimization gate

Topology-only, cache-hot Direct-I/O diagnostic, 500k seeds, 32 seeds/request,
16 request threads, exact measurement-seed prewarm:

| Revision stage | Nodes/s | Allocations/node | Allocated bytes/node |
|---|---:|---:|---:|
| Before feature-graph allocation work | 70.17M | 0.0180 | 613.4 B |
| Pooled frontier/shard groups/raw edges | 84.63M | 0.0133 | 363.5 B |
| Flattened selected indices | 88.93M | 0.0143 | 254.1 B |
| Pooled relation plans/ranges | 86.64M | 0.0114 | 121.2 B |
| Pooled parsed headers/header records | **92.11M** | **0.0107** | **42.6 B** |

This is a 31.3% throughput increase and a 93.1% reduction in allocated bytes
per sampled node. The remaining allocations are batch/wave and cache-lease
containers; there is no per-node or per-hit heap allocation in the steady
topology path.

Feature output necessarily allocates Arrow buffers. In a paired hot diagnostic,
Direct-I/O allocated 12,630.8 B/node versus mmap's 12,143.0 B/node, while
Direct-I/O achieved 4.63M nodes/s versus mmap's 3.36M. Total process CPU was
about 15.1 CPU-seconds for Direct versus 20.2 CPU-seconds for mmap over the
same 10k seeds. A CPU profile attributes the feature path primarily to RowPack
float decode, Arrow `take`, result copies, and large Arrow-buffer reclamation;
cache lookup and explicit page planning were each below 1% flat CPU.

## Multi-page read coalescing gate

Paired cacheless Direct-I/O smoke runs used 64 request/engine threads, eight
rings at QD128, 20k measured seeds, and identical feature sampling:

| Coalesce cap | Nodes/s | Logical pages | SQEs | Device IOPS | User CPU-s | System CPU-s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 page | 2.317M | 10.338M | 10.338M | 2.242M | 29.4 | 26.7 |
| 16 pages | **2.569M** | 10.338M | **8.047M** | **2.464M** | 30.0 | **23.4** |

Coalescing improved end-to-end nodes/s by 10.9%, reduced SQEs by 22.2%, and
reduced total CPU from 56.1 to 53.4 CPU-seconds. The higher observed device
IOPS is the result of completing the same logical work sooner, not issuing
extra reads.

## Scale matrix

| Scale | Rows | Index size | Index / 502.97 GiB RAM | Backend | Nodes/s | Device IOPS | Logical 4K pages/s | CPU cores | Cache hit | Notes |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1% | 280M | 122.84 GiB | 0.244× | mmap | 8.082M | 0.031M | n/a | 157.7 | kernel | 192 threads; file cache 98.3→102.7 GiB |
| 1% | 280M | 122.84 GiB | 0.244× | Direct | **9.297M** | 0.068M | 0.068M | **105.8** | 99.60% | 96 threads; cache 96.0→100.1/256 GiB |
| 5% | 1.4B | 616.24 GiB | 1.225× | mmap | 5.877M | 0.184M | n/a | 164.8 | kernel | 256 threads; file cache 315.0→345.3 GiB |
| 5% | 1.4B | 616.24 GiB | 1.225× | Direct | **8.188M** | 1.499M | 1.500M | **133.2** | 89.85% | 128 threads; cache 225.9→233.5/256 GiB |
| 25% | 7B | 3107.34 GiB | 6.178× | mmap | 1.651M | 0.990M | n/a | 172.2 | kernel | 768 threads; file cache 449.0→448.7 GiB |
| 25% | 7B | 3107.34 GiB | 6.178× | Direct | **5.976M** | **4.250M** | 4.268M | **117.5** | 61.33% | 256 threads; cache 319.6→319.9/320 GiB |
| 50% | 14B | 6220.25 GiB | 12.367× | mmap | 1.302M | 1.172M | n/a | 174.2 | kernel | 768 threads; file cache 434.3→432.2 GiB |
| 50% | 14B | 6220.25 GiB | 12.367× | Direct | **5.554M** | **4.228M** | 5.366M | **131.6** | 48.10% | 512 threads; cache 318.6→320.0/320 GiB; eager ≤3 pages |
| 100% | 28B | 12449.49 GiB | 24.752× | mmap | 0.845M | 0.970M | n/a | 164.4 | kernel | 768 threads; file cache 405.0→401.2 GiB |
| 100% | 28B | 12449.49 GiB | 24.752× | Direct | **4.897M** | **4.296M** | 5.682M | **97.9** | 37.68% | 384 threads; cache 319.7→320.0/320 GiB; eager ≤3 pages |

### Retained 1% result

Direct is 1.150x faster than mmap while using 32.9% fewer average cores and
41.7% fewer CPU-seconds per sampled node. Both retained measured windows used
700,000 independent roots after a 2,000,000-root warmup, lasted longer than 30
seconds, and grew their respective caches by less than 5%. The topology output
digest was `d9263324cc0632fc` for both backends; the 20,000-root determinism gate
also passed with digest `fb51bcdfd9085a42`.

The 1% Direct point is deliberately compute/hot-data bound. Its 99.60% hit rate
is not caused by replaying the measurement keys: warmup and measurement use
independent RNG streams. The complete index is only 122.84 GiB, the converged
Direct working set is about 100 GiB, and the configured item/merchant hotspots
create substantial natural reuse. Accordingly, low physical IOPS at this scale
is expected and increasing misses would only reduce end-to-end throughput.

The retained geometry is batch 32. Direct peaks at 96 request/engine threads;
128 threads slightly reduce hot-path nodes/s. Mmap continues scaling through
all 192 hardware threads. Direct uses 32 rings at QD32/ring and coalesces at
most 16 adjacent pages; sparse misses remain one-page requests.

After the retained result, a cumulative allocation call-site audit identified
redundant per-shard ID and position vectors in explicit feature fetch. Replacing
them with spans over the already-sorted unique-ID array reduces the full
feature-path allocation diagnostic from 1.415 to 1.149 calls/node (18.8%) with
13.9 KiB/node still dominated by the required expanded Arrow feature buffers.
The current-code steady Direct revalidation produced 9.297M nodes/s and the
same output digest. A more aggressive type-wide decode cut calls further but
regressed nodes/s by 39%; it was rejected and bounded RowPack/Arrow chunks were
retained.

### Retained 5% result

The 5% source contains 1.4B nodes in 435 Zstd Parquet files (94.784
GiB). Ingest produced a 616.240-GiB index, including 607.756 GiB of
`edges.bin`, in 11m56s with 117.4 GiB peak RSS. Direct is 1.393x faster than
mmap while using 19.2% fewer average cores and about 42.0% fewer CPU-seconds
per sampled node. Both measured 600,000 roots after an independent
1,500,000-root warmup, use batch 32, and pass the same deterministic output
gate (`2307b379f2a99d42` for the verified adjacent pair).

Direct peaks at 128 threads; 160 regresses. The retained run averages 95.3%
drive busy at aggregate QD194.8. Mmap peaks at 256 request/engine contexts;
320 regresses. Even with oversubscription, blocking mmap faults expose only
QD10.4 in the retained window. The kernel cache is larger (315.0→345.3 GiB)
than Direct's fixed data cache, so Direct's speedup is not the result of a
larger cache budget.

### Offset-guided eager complete-record experiment

`DG_IO_EAGER_NODE_PAGES` / `--io-eager-node-pages` now accepts `0`, `2`, `3`,
or `4` (and is internally bounded). When a record's offset-index span is at
most that many pages, the header wave requests the whole record. The later
topology/feature wave reuses those pages, and adjacent physical misses can be
issued by the existing `Readv` coalescer. `0` preserves the legacy two-wave
planner and remains the default.

| policy | per-ring QD | nodes/s | device IOPS | aggregate QD | cores | result |
|---|---:|---:|---:|---:|---:|---|
| eager off | 32 | **8.188M** | 1.499M | 194.8 | 133.2 | retained |
| eager ≤2 pages | 32 | 4.853M | 0.790M | 335.2 | 123.0 | rejected: effective page depth is too high |
| eager ≤2 pages | 16 | 7.961M | 1.295M | 156.3 | 134.5 | rejected: 2.8% fewer nodes/s |

Short 500k-warmup runs for eager limits 2, 3, and 4 reached 8.21–8.25M
nodes/s and reduced SQEs by about 13%, but the 1.5M-warmup steady runs expose
the long-run penalty. Limits 3 and 4 have essentially the same logical-page
and SQE counts as limit 2, showing that this schema contributes negligible
additional three-/four-page records. An exact-key hot-cache gate showed no
resident regression (8.91M eager-2 versus 8.82M off), but the mixed steady
workload is the governing result. The optimization therefore remains opt-in:
fewer I/O operations alone is not a win when queueing and CPU overhead reduce
end-to-end sample rate.

### Retained 25% result

The 25% generator produced exactly 7.0B source rows in 2,176 Zstd Parquet
files (476.039 GiB) in 31m27s, using 123.4 GiB peak RSS and no swap. Ingest
validated 700M users, 700M items, 5.46B orders, and 140M merchants, then wrote
32.76B directed relation entries. It completed in 66m49s with 214.2 GiB peak
RSS and no swap; stage times were 34m04s encode, 1m06s PK map, 2m53s edge
map, and 28m26s index assembly. The final index is 3107.34 GiB, including
3065.16 GiB in 2,004 `edges.bin` files. Temporary ingest data was removed.

Direct peaks at 256 request/engine threads; 320 threads produces the same
nodes/s within 0.01% while using 5.8 more cores. Its retained 500k-root window
followed a 1M-root warmup, lasted 35.45s, and held the cache at
319.6→319.9/320 GiB. It reached 5.976M nodes/s, 4.250M physical IOPS,
4.268M logical pages/s, aggregate QD572, and 400% device busy. The 61.33%
cache hit rate is natural workload reuse across independent warmup and
measurement RNG streams; the 3.1-TiB index is much larger than the cache.

mmap needs 768 request/engine threads to peak at 1.651M nodes/s. Scaling from
256 through 320, 384, 512, 640, and 768 threads raises nodes/s monotonically,
but 1,024 threads regresses 10.0% as CPU reaches 178.4 cores. The retained
768-thread point uses 172.2 cores, a stable 449.0→448.7-GiB kernel cache,
0.990M IOPS, aggregate QD65.9, and 383% device busy. Thus Direct is 3.620x
faster, uses 31.8% fewer average cores, consumes 81.2% less CPU time per
sampled node, and drives 4.292x the physical IOPS despite using about 129 GiB
less cache. Both retained paths produced topology digest `8a4fe3aa4f50af12`
and passed the same 20k-root verification digest `7d624f0406d62fcb`.

### Retained 50% result

The 50% generator produced exactly 14.0B source rows in 4,352 Zstd Parquet
files (953.19 GiB) in 1h03m46s with 123.5 GiB peak RSS. Ingest validated
1.4B users, 1.4B items, 10.92B orders, and 280M merchants, then wrote six
10.92B-edge directed relations (65.52B directed entries). It completed in
1h26m48s with 351.5 GiB peak RSS and no swap; stage times were 1h04m07s
encode, 2m34s PK map, 2m44s edge map, and 17m18s index assembly. The final
index is 6220.25 GiB, including 6135.89 GiB in 1,524 nonempty `edges.bin`
files and 27.71 GiB of offset indexes. Temporary ingest data was removed.

Direct peaks at 512 request/engine threads. Raising concurrency to 640 and
768 reduced nodes/s, so the retained point uses 32 rings at QD64/ring and an
offset-guided eager-record cap of three pages. Its 500k-root measured window
followed an independent 500k-root warmup, lasted 38.19s, and brought an
already-filled cache from 318.6 to 320.0 GiB. It reached 5.554M nodes/s,
4.228M physical IOPS, 5.366M logical pages/s, aggregate QD1211, and 399%
device busy. The 48.10% userspace-cache hit rate comes from independent RNG
streams over a 6.22-TiB index. The measured sampler hot path reported no heap
allocations.

At the same 512 threads and QD64, controlled tuning points produced 5.300M
nodes/s with eager reads disabled, 5.355M with a two-page cap, 5.500M with a
three-page cap, and 5.450M with a four-page cap. Eager three-page reads reduce
dependent header/body operations without overfetching enough data to displace
useful cache entries; the four-page regression establishes the bound. Raising
QD from 32 to 64 improved the non-eager point by 1.1%, while QD128 gained only
0.3% more nodes/s at the cost of 11 additional average cores and substantially
higher latency, so QD128 was rejected.

mmap uses its established 768-thread peak. Its identical 500k-root measured
window followed the same-size independent warmup and lasted 162.90s. It
reached 1.302M nodes/s, 1.172M IOPS, aggregate QD82.3, 383% device busy, and
used 174.2 average cores with a 434.3→432.2-GiB kernel cache. Direct is
4.265x faster, uses 24.5% fewer average cores, consumes 82.3% less CPU time
per sampled node, and drives 3.606x the physical command rate while using
less cache memory. Both retained backends produced topology digest
`96306bdc2611094b` and passed the same 20k-root verification digest
`e3ea59a6b31e5cb8`.

### Retained 100% result

The 100% generator produced exactly 28.0B source rows in 8,704 Zstd Parquet
files occupying 1,912.292 GiB. Ingest validated 2.8B users, 2.8B items,
21.84B orders, and 560M merchants, then wrote six 21.84B-edge directed
relations (131.04B directed entries). The byte-exact final index is
12,449.492 GiB, or **24.752x the host's 502.973 GiB RAM**. It includes
12,280.782 GiB in 2,884 `edges.bin` files and 55.414 GiB of offset indexes.
Temporary ingest data was removed, returning the array from a 2.4-TiB
low-water mark to 15 TiB free.

The durable ingest phases took 97m32s encode, 4m42s PK map, 5m27s edge map,
and 35m52s index assembly (2h23m33s summed phase time). Encode includes a
deliberate approximately seven-minute allocator diagnostic pause. The first
edge-map attempt exposed a pre-existing memory-retention bug: all shard tasks
completed before the unbounded result channel was drained, so hundreds of
whole-destination-table degree bitsets accumulated and displaced the PK maps.
The process reached 473.0 GiB RSS and major-faulted instead of making useful
progress. Commit `e0200a03` drains and merges results concurrently with the
unchanged task execution. The resumed edge map then completed in 5m27s and
the index in 35m52s, with 361.0 GiB peak RSS and no swap. Across the durable
phases, the governing peak remains the earlier 411-GiB PK-map phase.

Direct peaks at 384 request/engine contexts, 32 rings, QD64/ring, a 320-GiB
data cache, and an eager complete-record cap of three pages. Its 400k-root
measured window followed 500k independent warmup roots, lasted 34.617s, and
held the cache at 319.7→320.0 GiB. It reached **4.897M nodes/s**, 4.296M
physical IOPS, 5.682M planned logical pages/s, aggregate QD1434, 399% device
busy, and 21.675 GiB/s of device reads using 97.9 average cores. The natural
userspace-cache hit rate was 37.68%. The measured sampler hot path reported
zero heap allocations.

The workload is bandwidth-bound rather than 4K-command-bound at this scale.
With eager reads disabled, Direct issues 5.536M physical IOPS but reaches only
4.854M nodes/s at 21.48 GiB/s. Eager ≤3 coalesces adjacent pages into larger
commands, lowers physical IOPS to 4.296M, slightly raises bandwidth to 21.675
GiB/s, and improves complete nodes/s by 0.88%. The fourth-page cap regresses
to 4.879M nodes/s, establishing that further overfetch is not useful.

| eager cap | nodes/s | device IOPS | aggregate QD | average cores | result |
|---:|---:|---:|---:|---:|---|
| off | 4.854M | 5.536M | 1235 | 108.2 | rejected: more commands, fewer nodes/s |
| ≤3 pages | **4.897M** | 4.296M | 1434 | **97.9** | retained |
| ≤4 pages | 4.879M | 4.283M | 1424 | 101.2 | rejected: slight overfetch regression |

The retained concurrency is also the complete-throughput peak rather than an
IOPS selection. All points saturated the four drives; more contexts added
scheduler/feature-decode contention without more useful storage throughput.

| Direct contexts | nodes/s | device IOPS | average cores |
|---:|---:|---:|---:|
| 256 | 4.873M | 4.272M | 92.1 |
| 320 | 4.894M | 4.292M | 95.7 |
| 384 | **4.897M** | **4.296M** | 97.9 |
| 512 | 4.896M | 4.298M | 102.0 |
| 640 | 4.876M | 4.284M | 103.1 |

mmap retains its 768-context peak. Its identical 400k-root measured window
followed the same 500k-root warmup, lasted 200.546s, and used a larger
405.0→401.2-GiB kernel file cache. It reached 0.845M nodes/s, 0.970M IOPS,
aggregate QD69.1, 379% device busy, and 3.701 GiB/s using 164.4 average cores.
Raising mmap to 1,024 contexts regressed to 0.682M nodes/s while increasing
CPU to 175.0 cores, so it was rejected.

Direct is **5.793x faster**, drives 4.428x the physical command rate and
5.857x the read bandwidth, uses 40.5% fewer average cores, and consumes 89.7%
less CPU time per sampled node despite mmap retaining about 81 GiB more data
cache. Both retained windows produced digest `b60f465769e2244c`; the adjacent
20k-root mmap/Direct gate also matched exactly at `a4448a19c6247f02`.
