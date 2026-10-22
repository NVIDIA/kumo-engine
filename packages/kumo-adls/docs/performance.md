# Object-store transfer performance methodology

Measure complete, verified workloads before attributing performance to a
connector. This guide describes engineering tradeoffs and reproducible controls;
it contains no deployment-specific results or resource identifiers.

## Implementation and tradeoffs

- Bound queued tasks as well as active threads. A worker limit alone can still
  retain one future per chunk of a large object. Lazy submission bounds that
  overhead independently of object size.
- Budget concurrency across files. Larger chunks or fewer workers per file
  do not necessarily reduce aggregate memory when the request ceiling stays
  unchanged. Capture CPU and RSS alongside elapsed time.
- Avoid redundant buffering when the SDK already holds a response. Bulk ADLS
  transfers use one serial SDK downloader per file, with files copied in
  parallel. Matching response and transport-block sizes reduces fragment
  assembly. This is not zero-copy or direct network-to-disk streaming.
- Cache immutable bytes under a version identity with an explicit lifetime.
  The small-range cache avoids repeated footer reads within one reader; it
  does not cache across separate reader/source-access instances or replace fresh
  metadata checks.
- Separate experiments. Do not pool measurements from different revisions,
  layouts, client placements, or request policies into a cumulative speedup.

The [filesystem adapter](../src/kumo_adls/sdk_filesystem.py),
[session setup](../src/kumo_adls/session.py), and
[transfer limits](../src/kumo_adls/transfer.py) own these behaviors.

## Resource controls

Bulk ADLS downloads hold one transfer slot per file and use serial conditional
16 MiB responses, with no per-file executor or chunk-future backlog. The SDK
retains its initial response while processing later chunks. Transport blocks
are also 16 MiB; prepared uploads retain up to four 4 MiB chunks. The SDK still
buffers responses; a 64 MiB shared copy reservation is not a cap on total RSS.
Arrow buffers, cached bytes, metadata, loaded libraries, and Python overhead
consume additional memory.
Budget roughly 48–64 MiB of bulk payload buffers per active file, plus SDK-copy
headroom; 128 active files can therefore imply 6–8 GiB of payload alone. These
are planning estimates, not measured peaks or strict allocation bounds.

Projected reads use `readall()` and materialize the complete requested range.
A 64 MiB range uses up to three SDK workers after its initial 16 MiB response;
ranges above 64 MiB can use four. The initial response plus four 16 MiB chunks
can account for about 80 MiB before transport blocks, SDK copies and the output
range itself. Parallel SDK range reads can also queue futures for the requested
range. Neither the 80 MiB payload estimate nor the 64 MiB bulk-copy reservation
is a memory bound for projected reads.

`KUMO_SOURCE_COPY_WORKERS` lowers the shared copy ceiling. Set the positive
startup value `KUMO_ADLS_TRANSFER_WORKERS` before importing the provider to
lower only ADLS transfer slots and HTTP pooling, capped by the shared CPU/copy
ceiling. Bulk downloads use the same serial SDK path at every ceiling.
Metadata requests bypass transfer slots but share the HTTP pool. These controls bound requests,
not total RSS; a reduced ceiling alone is not a fixed-RAM-quota benchmark.
Both settings belong in the launching environment; restart a process after
changing the ADLS limit, which is fixed at import. Invalid ADLS values fail at
first provider import. Catchable allocation failures inside ADLS SDK operations
retain their original exception/traceback and add a note suggesting both controls
if the failure occurred during transfer; they may not help with listing or
session setup. An OS OOM kill cannot show this guidance. For lower memory across whole reads,
reduce shared file concurrency as well as ADLS transfer slots.

The process-wide small-range LRU caps payload at 64 MiB and 1,024 entries, with
Python key/container overhead additional. It admits complete successful ranges
up to 128 KiB, keyed by filesystem namespace, full path, raw ETag, and exact
range. Each source access creates a new namespace and cleanup releases it.
Opens and final source checks remain fresh. Prepared exports do not use this
reader cache.

## Live comparison recipe

Use identical complete tables and physical files on both stores. Prepare
trusted manifests containing every relative path, size, SHA256, footer row
count, and pinned provider ETag. Fixture copying and equivalence validation
must finish before timing. Do not substitute unrelated objects of equal
compressed size for equivalent work.

The [live bulk-reader harness](../../../scripts/benchmarks/object_store/compare_live.py) only reads cloud
objects and writes local results/scratch. Use authenticated, existing sources
and a Linux environment with provider dependencies and Arrow S3 support.
For native deployment comparisons, run S3 on the AWS client and ADLS on the
Azure client in the respective storage regions. Record actual CPU/RAM, OS,
network, disk and storage-class differences; matching vCPU counts does not
make the hosts identical.

Example single-provider phases, executed on their respective hosts:

```bash
# AWS host
python scripts/benchmarks/object_store/compare_live.py --providers s3 \
  --manifest /data/manifests/table.json --placement /data/aws-placement.json \
  --output /data/transfer-scratch --results /data/s3-results \
  --workers 32 --trials 3 --verification-workers 8 --s3-region us-west-2

# Azure host, using its configured ambient identity
python scripts/benchmarks/object_store/compare_live.py --providers adls \
  --manifest /data/manifests/table.json --placement /data/azure-placement.json \
  --output /data/transfer-scratch --results /data/adls-results \
  --workers 32 --trials 3 --verification-workers 8
```

Results directories must be new and scratch must fit the complete table plus
headroom. Run one timed workload per host. Preserve every individual trial,
rejection, manifest/harness/imported-source hash, dependency version and
placement record. Cloud-side cache state remains unknown in a fresh process.

## Timing, acceptance, and limits

Setup, authentication preflight, provenance and initial source identity checks
occur outside timing. The preparation timer covers the production reader's
`prepare(transform=False)` metadata guards and raw copying. A nested download
phase includes fetch's own metadata/checks/retries; it is not pure wire time,
and the remainder is not pure metadata latency. This workload does not measure
Graph construction, typed decoding, projection, conversion, training, or export.

Capture process CPU and RSS/HWM before untimed output verification. Loaded
provider stacks and setup allocations can affect the high-water mark; kernel
page cache and remote service memory are excluded. Local buffered writes,
close and atomic promotion do not establish durable-disk throughput without
an explicit durability requirement.

Require unchanged exact remote membership/identities and every output file's
SHA256, size and footer rows before accepting a trial. Verification is bounded
and untimed; its scheduling/cache advice must remain in provenance. Request,
retry, token-refresh counts and effective worker overlap are uninstrumented
in the live harness, so do not infer them from configured ceilings.

The [matched-request harness](../../../scripts/benchmarks/object_store/matched-transfer.md) separately isolates stream
client stacks under loopback HTTP. It bypasses production bulk paths and does
not replace native cloud measurements. Its fixed 8 MiB windows, 4 MiB SDK
chunks and 1 MiB transport blocks model different geometry from production
ADLS downloads (16 MiB for all three). Historical raw artifacts are audit
provenance, not current deployment results. Projections/conversions, CSV,
fixed-RAM limits, sustained refresh/throttling, DataFrame writes, model bundles,
and exports require their own representative evidence.
