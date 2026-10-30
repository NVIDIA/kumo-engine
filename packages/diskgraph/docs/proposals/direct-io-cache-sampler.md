# O_DIRECT io_uring sampler and graph-page cache

Status: reviewed; cacheless and simple sharded-cache baselines implemented and benchmarked

## Decision

Add an `O_DIRECT` io_uring backend behind the sampling-aware parser/page-access
boundary in the buffered proposal. First establish cacheless, bounded transient
I/O against a captured production page trace. Then add a process-local,
partitioned, fixed-size 4K cache with one pin/reference acquired per unique page
per prepared microbatch. Do not implement epochs, TinyLFU, or a node-record L1
until measurements justify them.

The goal is not to reproduce Linux's general page cache. The graph is immutable,
read-only, fixed-page addressable, has no writeback/coherency problem, and has a
dense resident node-to-byte directory. The userspace cache may be less
hit-efficient and still win if asynchronous miss overlap raises total sample
rate, but it must be judged under equal total memory and CPU.

## Baseline outcome

At 16 rings x QD48, 160 sampler threads, batch 50, and a 96 GiB/no-swap cgroup,
cacheless O_DIRECT sustained **8.640M nodes/s**, 929k physical IOPS, and
aggregate QD123 over 39.1 seconds. It reached 14.1% of the corrected
6.573M-IOPS multi-file XFS ceiling, showing that request preparation and
consumption—not raw storage—now limit the engine path. The best valid long
point observed during tuning reached 9.000M nodes/s with a longer warmup seed
stream, but 8.640M is the retained apples-to-apples result against the canonical
400k-warmup mmap and buffered runs.

A 16 GiB, 256-partition sharded HashMap/FIFO cache recovered a 17.6% hit rate
but fell to **7.215M nodes/s**. The cache's partition lock, hash lookup, Arc
lease, allocation footprint, and replacement work outweighed the saved reads.
It also displaced 19.3 GiB of file-backed offset/cache pages inside the fixed
96 GiB budget and incurred 3,609 major faults. The kernel mmap control reached
**9.244M nodes/s** with an estimated 26.6% hit fraction. This evidence rejects
the simple userspace page-cache baseline as
a production direction; it does not reject a packed fixed-set cache or a future
topology-specific layout, but either needs a trace-backed case before more code.

The implemented cache is deliberately a correctness/concurrency baseline, not
the lock-free four-way Stage-2 design below. It coalesces 200 concurrent misses
to one owner and uses 256 independent partitions, but every hit still takes a
partition mutex and an Arc reference. The measured loss is the gate for whether
the more complex design is worth pursuing.

## Performance budget

The historical mmap workload issued 0.089 physical reads per sampled output
node at 11.456M nodes/s, or 1.021M IOPS. At that throughput, a 2.544M XFS ceiling
permits 0.222 reads/node and the 6.56M raw-md ceiling permits 0.573. At 20M
nodes/s those budgets fall to 0.127 and 0.328.

These are bounds, not acceptance ceilings. The relevant result is a 4K O_DIRECT
multi-file trace replay across the graph's actual XFS files. The first fio
point used one 441 MiB inode; the largest 32/128/1,024 graph files span roughly
13.15/52.60/420.66 GiB. Raw per-drive fio and the multi-file XFS knee must use
enough workers and aggregate QD (including 32 workers and QD1024) rather than
repeating the earlier four-worker ceiling artifact.

The resulting 128-file measurement is 6.573M IOPS at QD1024 (25.1 GiB/s,
156 us mean latency, p99 594 us); QD2048 adds only about 0.8% IOPS while roughly
doubling latency. Raw member-device runs remain permission-blocked on this host.

## Shared layout and sampling contract

The direct path uses the reviewed `ParsedNodeHeader` and `PageAccessor` design:

```text
offset resolution
→ boundary-safe header pages
→ relation byte ranges
→ exact sampling-index plan
→ only sampled-entry pages (+ requested rowpack pages)
```

It does not materialize an entire topology span for a small fanout. Uniform
`k < degree` reads pages for selected indices; `k >= degree` reads the relation;
temporal-most-recent scans progressively; fused access adds the exact rowpack
range. Header I/O can be deduplicated by node, while logical-node visitation and
RNG use remain in production order. The exact legacy mmap path stays the
throughput control.

A prefix can straddle a page. A final O_DIRECT page normally completes short on
this index: all 6,285 non-empty edge files have unaligned tails. For page offset
`p`, accept exactly `min(4096, file_len - p)`, zero-fill the suffix, retain
`valid_len`, and publish it. Any other short result fails. Empty files submit no
I/O.

## Stage 1: cacheless direct executor

Before cache policy can obscure the result, replay real prepared page groups
through:

- 1/2/4/8/16/32 independent single-owner rings;
- aggregate depths 64/128/256/512/1024/2048;
- fixed-capacity aligned transient buffers and waiter/operation slabs;
- batch-local page deduplication and owner-local in-flight coalescing;
- one `PartitionReady` completion per `(batch, owner)`, not one shared atomic
  decrement per page;
- no cache admission, replacement, epoch, registered file table, or node L1.

Each SQE carries a stable operation-slab index plus wide generation. Shutdown
accounts for cancellation and the original CQE in either order; no buffer is
reused before the original completion arrives. Validate `IORING_FEAT_NODROP`,
size CQ for cancellation bursts, and treat dropped completions as fatal.

This stage answers whether the application can generate enough asynchronous
page traffic and whether ring-owner service capacity reaches at least 1.25x its
target share using an immediately-completing fake device. It also establishes
the real multi-file filesystem ceiling before cache claims.

## Stage 2: simple durable-lease page cache

### Key and geometry

Assign every edge file a dense page base at graph open:

```text
global_page = file_page_base[file_id] + file_page_number
```

Published tags need one `u64` (`global_page + 1`, with zero empty). The I/O
request separately carries FD, offset, and valid length. Benchmark 4- and 8-way
set associativity. A stable hash chooses partition and set. Replacement is
set-local round-robin/NRU first; a partition-wide CLOCK hand is incompatible
with a fixed target set.

The cache is partitioned to spread hundreds of clients, with read-mostly public
tags/data separated from owner-only replacement and in-flight metadata. One
owner mutates a partition. There is no global map, LRU, cache lock, or metrics
counter.

### V1 page lifetime

Use a durable reference/pin once per unique cached page in a prepared
microbatch—not once per node. This matches the prepared-window lifetime and is
the correctness baseline. Measure one-hot, Zipf, collision, and full parsing
with `perf c2c`; only replace pins if their end-to-end cost is material.

Miss I/O is independent of cache-slot availability:

```text
miss
→ owner-local LOADING/coalesce entry
→ read into aligned transient frame
→ satisfy waiting prepared windows
→ copy/admit into an unpinned target-set way, or bypass admission
```

If every way is pinned, I/O still completes through the transient page and the
cache simply misses. Admission/reclamation can lower hit rate but cannot stall
device progress. Fixed slabs bound transient buffers, waiters, and in-flight
entries. Set-local replacement never overwrites a pinned page.

An Arc/refcount implementation is a safe oracle; a tighter slot pin can combine
wide generation and pin state while keeping exact-key validation. Views borrow
table layouts so the existing per-node layout-Arc clone is removed from the new
path.

### Epoch follow-up, not V1

If batched pins are proven hot, use atomic frame handles and decouple logical
ways from physical frames:

```text
public set: atomic frame handles
frame: FREE → FILLING → PUBLISHED → RETIRED → FREE
```

A reader acquire-loads one handle containing frame index + generation and
validates the immutable full key. An owner fills an unpublished frame,
release-swaps a target-set handle, and retires the old frame separately. Spare
frames cover aggregate QD plus grace backlog; exhaustion causes transient
no-admit bypass, never deadlock.

No epoch/RCU guard may cover I/O, a queue wait, Rayon parking, continuation
migration, or a stored prepared batch. Once all pages are ready, a worker enters
a short guard, revalidates handles, parses/copies promptly, and exits. Use a
proven reclamation library or a fully model-checked nested/panic-safe protocol
with cache-line-padded worker announcements and one amortized grace coordinator.

## Concurrency path

Workers deduplicate page keys per microbatch and send one group per target
owner. The owner rechecks READY/LOADING, coalesces misses in a fixed-capacity
table, submits them, and sends one ready message for its whole group. A batch
coordinator therefore observes at most the number of participating owners,
avoiding a page-completion cache line bounced across 16-32 cores.

Single-writer does not make MPSC tails free. Benchmark realistic frontend hash
and dedup, 1-32 sends per graph batch, owner lookup, coalescing, and completion
handoff with 192 producers. If tails show HITM traffic, use fixed per-worker SPSC
mailboxes or hierarchical dispatch.

Owner threads count against the host's 192 logical / 96 physical core budget.
Headline runs hold total runnable workers and affinity constant across mmap,
buffered, and direct; also report CPU-seconds per million nodes. Ring-owner count
and cache partition count need not be the same.

## Direct-I/O requirements

- Open immutable edge FDs with `O_RDONLY|O_DIRECT|O_CLOEXEC` at graph load.
- Query `STATX_DIOALIGN`; 4K is safe on the current 512-byte-aligned XFS but is
  not assumed elsewhere. Alignment failure is a load error, never silent
  buffered fallback.
- Backend selection is immutable for the process. Never mix dynamic mmap and
  O_DIRECT access to an edge file.
- Keep buffers aligned, alive, and immutable through the original CQE.
- Mark private direct buffers `MADV_DONTFORK` or prohibit fork while I/O is in
  flight.
- Retain/check enough FDs at startup, skip empty edge files, and fail explicitly
  if limits are inadequate.
- Keep offsets buffered/mmaped, prefault them, and verify their residency.
- Benchmark registered QD-sized bounce buffers versus unregistered final frames.
  Do not register a 32-256 GiB cache arena; current `RLIMIT_MEMLOCK` is 8 MiB.

## Memory and multi-process constraints

The primary experiment fixes total cgroup memory and disables swap. Direct cache
capacity is computed as total budget minus measured non-cache peak, metadata,
transients, and safety margin. Record anonymous residency/prefault time,
`memory.current/peak`, anon/file/kernel/slab, reclaim/refault, major/minor faults,
PSI, and cache/transient/waiter/spill bytes. A nominal virtual arena is not
resident capacity.

The cache is process-local in this experiment. Linux page cache shares pages
across processes; eight private sampler caches can duplicate hot pages and
misses. Production must either centralize sampling, build a true shared-memory
cache/ownership protocol, deliberately bound aggregate private caches, or keep
mmap/buffered mode. Benchmark 1/2/4/8 processes under fixed aggregate CPU and
memory before claiming multiprocess suitability.

## Node-ID cache decision

Do not build a node-record L1 initially. The physical read is still 4K, sorted
IDs already improve page sharing, per-worker locality is uncertain under Rayon,
variable records require more indexing/allocation, and hubs are better retained
as selected pages than whole records.

Capture post-hub-dedup actual sampled-page traces and compare equal total bytes:

- exact proposed 4-/8-way placement and NRU;
- fully associative LRU upper bound;
- CLOCK/TinyLFU admission;
- packed exact-node records;
- hybrid node L1/page L2;
- topology-only and genuine feature plans;
- 1/2/4/8 process splits.

Use one seed trace for policy selection and a holdout trace for validation.
Count READY hits, LOADING coalesces, batch dedup, conflict misses, and physical
reads separately. A cache policy earns complexity only from holdout evidence.
Separating topology and rowpack in a future index is likely a higher-value
feature-heavy optimization than a universal node cache.

## Tests and microbenchmarks

Correctness/model tests include:

1. zero-length files and EOF remainders 1, 511/512, 1,373, and 4,095;
2. page-straddling record headers and selected entries;
3. exact sampled-entry plan/digest versus mmap for empty, low-degree, hub,
   topology-only, and feature-heavy records;
4. 200 concurrent misses for one page issue one device read, separately from
   within-batch dedup;
5. five or more keys colliding in one four-way set complete via bypass;
6. a paused pinned reader cannot block unrelated misses or exhaust progress;
7. failure wakes every waiter and never publishes partial data;
8. stale generations, op-slot reuse, cancellation order, CQ overflow, and
   shutdown do not cause ABA, leak, or premature buffer reuse;
9. cache/queue/pool exhaustion remains bounded and deadlock-free;
10. benchmark errors, unknown flags, fallbacks, incomplete seeds, and digest
    mismatch invalidate a point.

Concurrency microbenchmarks cover 192-reader one-hot/Zipf/uniform hits, cold
misses at QD64-2048, same-page thundering herd, set collisions, long pins,
frontend MPSC traffic, and fake-device owner capacity. Report throughput,
p50/p95/p99/p99.9, CPU, LLC/TLB misses, atomic stalls/HITM, queue/service
latencies, memory, and metrics-on overhead.

## Storage and production benchmark sequence

1. Repair strict harness, canonical plan/digest, fatal errors, JSON output,
   fixed total memory/CPU, swap=0, and three independent repetitions.
2. Run raw md and each raw NVMe with enough workers/depth; report actual depth
   distribution and latency, not just requested QD.
3. Run O_DIRECT XFS over one, 32, 128, and 1,024 real files, then exact trace
   replay. Test fixed buffers/files and submit/completion batch sizes.
4. Validate cacheless executor throughput and app-to-device amplification.
5. Capture two production traces and simulate cache geometry/policy.
6. Implement simple pinned set cache; verify trace prediction within 10% of
   submitted reads/node and compare against holdout fully associative LRU.
7. Run canonical mmap/buffered/direct points in randomized order under equal
   total memory and CPU, after cache/reclaim/refault/throughput stabilization.
8. Only then consider epochs, CLOCK/TinyLFU, node L1, fused expansion, or shared
   multiprocess cache.

Suggested success thresholds are zero errors/fallbacks/digest mismatch; three
60s stable repetitions with CV <=5%; cold executor >=70% of exact multi-file
XFS ceiling at the latency-efficient knee; proposed cache within 15% reads/node
of equal-byte fully associative LRU; conflict misses <5%; mixed direct >=1.10x
mmap and miss-heavy direct >=1.25x mmap with intervals excluding parity; warm
direct cache >=90% of PTE-hot mmap; and sustained demand below 80% of the exact
filesystem ceiling.

## Rejected first implementations

- Whole-topology reads for small fanout: invalid I/O amplification.
- Epoch guards held by prepared batches: unsafe under waits/migration.
- Cache admission required before miss I/O: can deadlock on pinned ways.
- Global map/LRU or per-hit exact recency writes: centralized contention.
- Per-thread full caches: duplicate capacity and physical reads.
- One slot/residency byte per 19.6B node: prohibitive before payload.
- Dynamic mmap fallback on the same files: coherence/performance ambiguity.
- Node L1, TinyLFU, registered full arena, or custom EBR before evidence.
