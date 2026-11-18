# FM payload comparison

`runner.py` compares the installed `kumo-fm` wheel with the pinned
fm-dataloader DuckDB reference on the same frozen bundle, workload, and
selected instances. Install the reference checkout only in the benchmark
environment; it is not a `kumo-fm` runtime dependency.

Run correctness first, then warmups and interleaved measured draws:

```bash
python make_fixture.py /tmp/fm-case
python runner.py --bundle /tmp/fm-case/bundle \
  --workload /tmp/fm-case/workload.json --index /tmp/fm-case/index \
  --draws 5 --output /tmp/fm-report.json
```

For an existing bundle and workload:

```bash
python runner.py --bundle /data/bundle --workload /data/workload.json \
  --index /data/fm-index --draws 7 --output /tmp/fm-report.json
```

The workload must be eligible for the public `kumo.pytorch` recent-sampling
path. Uniform/timeless and empty-layer workloads are reported as excluded,
not silently rewritten. Reports include source revisions, dependency
versions, machine metadata, row counts, logical Arrow bytes, body bytes, and
raw per-phase timings. Retain the bundle, workload, wheel hashes, index
provenance, and report together when publishing a comparison.

## Omnistation smoke comparison

[Raw report](reports/omnistation-smoke.json) records five interleaved draws
after two warmups on a four-event, one-hop synthetic graph. Both encodings
decoded identically to the DuckDB reference and produced the same body byte
counts. Median phase times in milliseconds:

| Encoding | DuckDB reference full payload | DiskGraph load and assembly | Encoding retained batch |
| --- | ---: | ---: | ---: |
| Arrays | 40.691 | 3.643 | 0.731 |
| Arrow IPC | 40.765 | 3.457 | 0.446 |

The backend paths and measured phase boundaries differ. This small smoke
case validates parity and timing instrumentation, not general throughput.
