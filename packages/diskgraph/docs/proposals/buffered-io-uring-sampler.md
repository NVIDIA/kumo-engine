# Buffered io_uring sampler backend

Status: reviewed, baseline implemented and benchmarked on 2026-08-06

## Decision

Add an explicit buffered `io_uring` path for immutable `edges.bin` files while
keeping the compact `offsets.bin` directory resident. It reads only header and
sampled-entry pages into bounded prepared windows and parses those buffers
directly. It never warms an mmap and then faults through it. The current mmap
code remains untouched as the throughput control.

This is the lowest-risk way to test whether controlled outstanding I/O breaks
the present coupling between one sampler thread and one blocking major fault
while retaining Linux page-cache hits and cross-process sharing. It necessarily
copies page-cache bytes into userspace, so it is expected to lose to mmap for a
fully PTE-hot working set.

## Baseline outcome

The optimized implementation validates the mechanism but does not beat mmap.
On the canonical 4M-seed run (144 sampler threads, batch 50, eight rings at
QD128, 96 GiB cgroup, swap disabled), buffered io_uring sustained **7.933M
nodes/s**, 647k physical IOPS, and aggregate QD292. The unchanged mmap control
sustained **9.244M nodes/s**, 731k physical IOPS, and QD43.2. Buffered reads
preserved an estimated 24.2% kernel-cache hit fraction versus mmap's 26.6%, but
queue handoff, page objects, buffered-page copying, and completion handling
still cost more than asynchronous overlap saved.

This is substantially better than the first 4.877M implementation. The main
gain came from batching all shard/file requests in each header or selected-data
wave into at most one producer envelope per owner, rather than allocating and
sending one job per tiny shard group. Owner batching by eight ring depths and
bounded aligned-frame recycling removed another sustained allocator problem.
The retained run submitted 36.37M logical pages through 31.91M logical shard
requests but only 1.94M producer envelopes; 2,742 frames were allocated and
36.37M reused, with zero pool drops.

This answers the Phase-1 question precisely: the explicit path does not incur
one blocking mmap page fault per worker. It submits all header-page requests for
a shard wave, waits for that wave, plans exact sampled entries, submits the data
wave, and parses returned buffers. The current bounded implementation still has
wave barriers, however; it is not a continuation scheduler. More scheduling
machinery is unjustified until the explicit-buffer CPU cost is reduced.

## Measured context

The retained 25% graph has 19,645,490,124 nodes, 76,475,264,044 stored directed
edges, 866.874 GiB of edge blobs, and 38.880 GiB of offsets. The historical mmap
point reported 11.456M sampled nodes/s, 1.021M physical IOPS, 0.089 reads/node,
and aggregate member QD 59.5. Raw md RAID0 fio reached 6.56M 4K random-read IOPS
with 32 workers and aggregate QD1024; QD2048 reached 6.62M with twice the
latency. One XFS edge file reached 2.544M O_DIRECT IOPS at 32xQD32. The
corrected 128-file XFS run reaches **6.573M IOPS** at the same aggregate QD1024,
matching the 6.56M raw-md result and ruling out meaningful md/XFS overhead for
this test.

That sampler point is historical context, not the new comparison baseline: its
script passed an unsupported `--relations` flag which the benchmark silently
ignored. The repaired harness must use an explicit per-layer `--plan`, reject
unknown flags, print a plan digest, fail on any sample error, and rerun mmap.

## On-disk layout and resident directory

A local node ID maps through delta-packed offsets to `[record_start, record_end)`
in `edges.bin`:

```text
[compact count for each source relation]
[2/3/4-byte node_data_len when rowpack is present]
[packed relation 0 entries]
...
[packed relation N entries]
[rowpack PK/features]
```

The relation layouts provide fixed entry widths, so counts determine every
relation span, topology end, and feature start. A record is not page aligned:
even a two-byte header can straddle pages. Header planning therefore reads all
pages intersecting:

```text
[record_start,
 min(record_end,
     record_start + 4 * relation_count + node_data_len_bytes))
```

and then parses the actual variable-length counts.

All current offset files use canonical 64-entry chunks and 2-byte deltas. The
reader's prior 24-byte heap descriptor per chunk costs about 6.86 GiB on this
graph. Removing that duplicate vector and computing canonical chunk positions
arithmetically is an independently tested prerequisite, not an I/O-format
change. The benchmark must prefault/verify offset residency and record offset
faults; a 38.88 GiB mmap is not resident merely because it exists.

Tables containing only PK/FKs still store the PK rowpack when
`has_node_data=true`; a no-feature topology walk must avoid that tail. On
feature-heavy tables the same projection saves much more. A later format could
separate topology and features, but it is not required to test explicit I/O.

## Parser and page-access boundary

Topology-only bytes cannot be wrapped in the existing strict `EdgeListView`:
that parser requires the declared node-data tail and exact record end. The new
paths share lower-level parsing and access objects:

```rust,ignore
struct ParsedNodeHeader {
    record_start: u64,
    record_end: u64,
    relations: Vec<RelationRange>,
    topology_end: u64,
    node_data: Option<Range<u64>>,
}

trait PageAccessor {
    fn bytes(&self, absolute_range: Range<u64>) -> Result<PageSpan<'_>>;
}

struct PreparedShardWindow {
    headers: Vec<ParsedNodeHeader>,
    pages: PreparedPages,
    stats: PrepareStats,
}
```

Header parsing does not require feature bytes. Relation sampling consumes
relation ranges through `PageAccessor`; requesting unavailable node data returns
`NotLoaded`, distinct from a node with no data. Page-backed paths copy only a
selected entry that crosses a boundary or an explicitly requested rowpack—not
an entire hub record. Views borrow layouts from their reader/window rather than
Arc-cloning layouts for every node.

`ShardReader` retains an `Arc<EdgeFile>` with an owned FD, file length, and
stable file ID for all in-flight work. Graph open checks `RLIMIT_NOFILE` before
retaining thousands of descriptors. V1 does not register all 7,506 files in
every ring.

## Sampling-aware three-stage pipeline

Explicit I/O must not read an entire adjacency merely because the mmap record
span is large. For uniform sampling with `k < degree`, the existing code skips
unselected entries by pointer arithmetic, so mmap faults only header pages and
pages containing selected entries.

The async path preserves that behavior:

1. Batch-decode start/end offsets for sorted unique local IDs.
2. Deduplicate and read bounded header pages.
3. Parse relation counts and absolute spans.
4. Generate exactly the sample indices the current sampler will use.
5. Deduplicate and read only pages containing those entries.
6. Consume results in original logical visitation/RNG order.

When `k >= degree`, the relation is read completely. Temporal-most-recent reads
progressive page windows from the relation start until `k` eligible edges are
found. `NodeDataOnly` jumps to the rowpack range. Fused sampling groups I/O by
shard but retains an original-position map; non-fused topology keeps its
existing shard sort/hub-header dedup; standalone feature fetch reuses its
sorted-unique chunks.

## io_uring model

Use independent single-owner rings assigned initially by shard/file. Sweep 1,
2, 4, 8, 16, and 32 rings at aggregate SQ depths 64, 128, 256, 512, and 1024.
The direct-fio result does not imply buffered XFS needs 32 rings: one ring can
hold many callback-driven reads. Split a proven-hot shard across executors only
if file-level assignment shows imbalance.

V1 submits a bounded prepared shard window and waits on one batch ticket. Each
executor completes one sub-batch, avoiding a shared per-page decrement. Existing
outer request and inner shard concurrency provides I/O overlap. This blocking
adapter is sufficient for provider falsification; final production evidence
requires a completion handle/continuation if profiles show parked Rayon workers,
nested stealing, or stack growth limiting throughput.

Ring setup arms to measure include `SINGLE_ISSUER`, `COOP_TASKRUN`, and
`TASKRUN_FLAG`. `DEFER_TASKRUN` is used only with a dedicated owner loop that
regularly enters the kernel. Registered files/buffers and shared io-wq are
evidence-driven follow-ups.

## Buffered-kernel behavior

Open the retained buffered FD normally and apply `POSIX_FADV_RANDOM`;
`MADV_RANDOM` affects only the mmap VMA. On current XFS, ordinary io_uring
buffered reads can start readahead, arm a folio-unlock callback, and retry via
task work rather than blocking one io-wq thread per miss. The owner must keep
entering the kernel so task work and completions run.

Explicit `RWF_NOWAIT` is a diagnostic only. It may start readahead and then
return `EAGAIN`, forcing userspace to submit a second SQE. `IOSQE_ASYNC` is also
diagnostic because it skips the inline attempt and offloads immediately.

For page offset `p`, the expected completion is:

```text
min(4096, file_len - p)
```

The common unaligned EOF page is valid; it carries `valid_len` and bytes beyond
that length are never exposed. Only fewer bytes than this expected length are an
unexpected short read.

## Bounds, lifetime, and failure

- Prepared windows have a hard page/byte cap and reserve continuation capacity
  before admitting headers.
- Completed pages are consumed/released incrementally; ready buffers cannot
  deadlock continuation reads.
- Batch-owned arenas or owned completed pages make lifetime explicit.
- Output order and RNG use never depend on CQE order.
- Any unexpected short read, CQE error, executor failure, or premature close
  cancels the batch and fails the benchmark point.
- CQ overflow is monitored; actual completion drops are fatal.
- Backend selection is immutable for a graph/process. There is no per-node
  `mincore` probe or backend switch.

## Measurement contract

The primary comparison fixes total cgroup memory, sets `MemorySwapMax=0`, and
records `memory.current/peak`, anon/file/kernel/slab, reclaim/refault, PSI, and
arena high-water. A secondary diagnostic comparison fixes usable cache-data
bytes. Each backend runs in a fresh process/cache state, at least three times in
randomized order, on identical fixed work with the exact legacy mmap path.

Required metrics include:

- nodes, edges, seeds, records, unique planned pages, and logical bytes/s;
- physical reads/bytes per seed, record, planned page, and output node;
- application SQEs outstanding, executor queue time, submit time, SQE-to-CQE,
  CQE-to-consumer, and window p50/p95/p99/p99.9;
- md and member device IOPS, bandwidth, average request size, utilization, and
  physical QD (not inferred from SQEs);
- page-cache and offset residency; major/minor and cgroup major faults;
- executor max/mean imbalance, live/in-flight/spill arena bytes;
- user/system CPU, cycles, context switches, migrations, CPU/memory/I/O PSI;
- io-wq punts via `io_uring_queue_async_work` tracing when permissions allow;
- output/page-plan digest and complete/requested batch counts.

Depth is increased until doubling it changes IOPS or nodes/s by less than about
5%, rather than requiring an arbitrary physical QD in a cache-hit workload.
Invalid ramping, reclaim-changing, digest-mismatched, errored, or thermally
drifting points do not count.

## Tests and microbenchmarks

Unit/integration coverage:

1. Batched offset decode equals scalar decode for random IDs, chunk boundaries,
   repeat offsets, and 2/3/4-byte deltas.
2. Header starts at page offsets 4093-4095, including four-byte counts.
3. Natural EOF-short pages expose exactly `valid_len`.
4. Topology projection with nonzero node-data length reports `NotLoaded` for
   data while relation access succeeds.
5. Uniform small-fanout planning touches only selected-entry pages; `k >= degree`
   touches all relation pages.
6. Empty, one-page, cross-page, feature-heavy, and large-hub records match mmap.
7. Completion reordering preserves canonical sampled global-edge output.
8. Within-batch and 192-cross-batch duplicate requests are measured separately.
9. A batch larger than the arena cap progresses; exhaustion and shutdown with
   in-flight FDs do not deadlock or leak.
10. The harness propagates its first sampling error and rejects unknown flags.

Microbenchmarks use captured 25% page traces plus synthetic key-only and
feature-heavy records. They cover batch sizes 64/256/1000/4096, one-page and
coalesced/random pages, warm working sets larger than LLC, PTE-hot mmap versus
page-cache-hot mmap, normal/fixed-buffer/forced-async reads, 1-192 clients, the
ring/depth sweep, `FADV_RANDOM` on/off, and metrics on/off.

## Staged rollout and gates

1. Repair the benchmark contract and rerun exact mmap.
2. Land boundary-safe header parsing, sampling-aware page plans, and trace
   capture; prove their requested-entry digest against mmap.
3. Replay the real trace against edge files to find the smallest saturated
   buffered ring/depth configuration.
4. Integrate one canonical non-fused topology point using bounded synchronous
   windows. It succeeds only with statistically higher end-to-end nodes/s than
   mmap under fixed total memory.
5. If provider replay wins but the engine does not, profile and add
   completion-driven scheduling before calling results final.
6. Sweep narrowly around the knee, then test fused/node-data and diagnostic
   NOWAIT arms.
7. Keep mmap as production default until correctness, memory, and repeatable
   throughput gates pass.

The matching ceiling is a buffered XFS multi-file replay over the real page
distribution. Neither raw-md O_DIRECT nor a one-inode O_DIRECT fio point is a
valid buffered acceptance ceiling.

## Rejected first steps

- `MADV_WILLNEED` or `mincore` in the hot path: retains mmap/PTE locking and
  measured worse previously.
- Async read followed by mmap parsing: hides whether explicit buffers help.
- Whole-record topology reads: amplifies large-hub I/O relative to mmap.
- One read per node: loses page deduplication.
- One globally locked ring: centralized submission contention.
- A continuation/work-stealing scheduler before the bounded provider test.
- Replacing the exact mmap control with a newly allocated prepared-mmap path.
