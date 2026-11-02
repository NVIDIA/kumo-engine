# Frozen-geometry object-store stream diagnostic

The [standalone harness](compare_streams.py) compares the actual ADLS
SDK/fsspec adapter and native Arrow S3 stream under a fixed diagnostic request
geometry and local write loop. The ADLS client is constructed with explicit
diagnostic settings; this does not measure production transfer defaults. It creates only a loopback server and local output; it does not
provision or write cloud resources. This page documents the method, not
historical timing results or cloud-service rankings.

## Fixed diagnostic geometry

These constants deliberately stay independent of production tuning:

| Control | Diagnostic value |
| --- | ---: |
| Common sequential GET window | 8 MiB |
| ADLS SDK initial response | 8 MiB |
| ADLS SDK subsequent chunk setting | 4 MiB |
| HTTP transport / local write block | 1 MiB |

The 8 MiB stream windows fit within the diagnostic SDK initial response.
These values are not the current production bulk-transfer geometry. Results
and arguments record `geometry.derived_from_production=false` with the exact
controls. Keep earlier frozen harness snapshots and their hashes with historical
evidence; relocating this maintained script does not change those snapshots.

## Controls and acceptance

Prepare one complete immutable Parquet table and a per-file manifest of SHA256,
byte lengths and footer rows. Both clients must read identical files and
encodings. Equal stored-byte totals from unrelated data do not establish equal
work.

Each accepted trial must satisfy:

- Exactly two HEADs per file: opening identity and a fresh identity after
  reading. Discovery/schema sampling and provider listing shortcuts are excluded.
- Identical explicit 8 MiB GET windows, including partial final windows, and
  exact response-body byte totals derived from the manifest. Retries, extra
  requests, changed identities or failed responses reject the trial.
- One in-flight read per file and the same file-worker ceiling. Both clients
  use the same buffered local writes, short-read/write checks, temporary files,
  version recheck, close and atomic promotion.
- Every output's SHA256, size and footer rows match the source. Verification
  runs after timing, CPU and RSS capture.

ADLS disables fsspec read-ahead. S3 uses native Arrow
`open_input_file`/`read_at`, without replacing that path with boto3. ADLS sends
its opening ETag as a GET condition; the native S3 interface exposes no explicit
If-Match option here. Record observed headers and use a stable fixture with
fresh before/after identities. This does not establish identical mutation
semantics.

ADLS slots and HTTP pooling match the requested worker count; Arrow I/O threads
use that count. Native S3 exposes no public HTTP-pool sizing option. Traces
record realized overlap and reject counts above the common ceiling. Equal
ceilings do not guarantee equal overlap. Both clients disable retries.

A common HTTP/1.1 server handles both URL forms with identical file responses,
TCP_NODELAY and a common listen backlog. Include zero injected delay as well
as a latency condition. Server and clients share the host and local disk;
record page-cache state and server resource pressure rather than interpreting
server saturation as a client result.

## Reproduction

Use Linux with provider dependencies and an Arrow build supporting S3. Prepare
the complete local table separately; do not overlap hashing, tests, fixture
preparation or another transfer workload with timing.

```bash
export PYTHONPATH=packages/kumo-adls/src:packages/kumo-core/src:packages/kumo-api
python scripts/benchmarks/object_store/compare_streams.py \
  --source /data/table --output /data/transfer-scratch \
  --results /data/protocol-check --check-protocol
python scripts/benchmarks/object_store/compare_streams.py \
  --source /data/table --output /data/transfer-scratch \
  --results /data/matched-transfers --workers 4 32 --latencies-ms 0 10 --trials 3
```

Each result directory must be new. The protocol check selects the largest file
to exercise multiple ranges and emits no timing fields. The harness defaults
to 580 expected files; use `--expected-files` for another complete table.
`--max-files` is an explicitly marked subset diagnostic, not a complete-table
result.

Trials alternate provider order in fresh child processes. Imports and client
construction are outside timing; source hashing warms the OS page cache.
Results retain source manifests, imported-code/dependency versions, harness
hash, identities, per-file traces, output fingerprints, requested and observed
concurrency, and individual metrics. Preserve rejections and explain exclusions.
RSS includes loaded dependencies and excludes server memory; it is not merely
range-buffer allocation.

## Interpretation

This common sequential range loop deliberately bypasses both providers' bulk
fetch callbacks and backend discovery/versioning
policies. It can diagnose stream-stack overhead under matched geometry; it
cannot establish complete connector ingestion, native deployment parity,
object-store durability, TLS/authentication cost, cloud networking or service
throttling. A smaller worker ceiling is not a fixed-RAM-quota benchmark.

Use the [live bulk-reader harness](compare_live.py)
for production bulk-reader measurements on the appropriate native cloud hosts. Keep results from
these different scopes separate; historical raw traces/CSV remain provenance,
not current ADLS native-host measurements.
