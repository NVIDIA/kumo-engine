# Deferred FM benchmark tooling

This draft is a sibling follow-up based on #391. It is outside the required
#404 → #403 → #448 → #391 → #490 release sequence. Nothing here is a workspace
member or bundled into the engine wheel. DuckDB is required only in this
optional benchmark environment.

`kumo_fm_benchmark` owns frozen bundle/workload loading, dataset generation and
validation, and the DuckDB comparison runner removed from the product package.
The production `kumo.fm` path accepts connector snapshots and Arrow/Parquet
fixtures without this package. The benchmark currently uses private FM Arrow
assembly helpers; review that coupling before marking this draft ready.

## Development setup

Install matching FM, PyTorch, settings and DiskGraph source wheels from #391 in
a separate environment, then install this package and its test dependencies:

```bash
python -m pip install -e './benchmark/fm-payloads[test]'
python -m pytest benchmark/fm-payloads/tests
```

Do not install standalone first-party wheels over a merged engine install. A
benchmark install path targeting the merged wheel is deferred until #490's
packaging lands. The comparison runner also needs the separate fm-dataloader
reference checkout at the source baseline recorded in the retained smoke report.
That reference is not a dependency of the product FM package.

## Frozen cases

```bash
cd benchmark/fm-payloads
python make_fixture.py /tmp/fm-case
python runner.py --bundle /tmp/fm-case/bundle \
  --workload /tmp/fm-case/workload.json --index /tmp/fm-case/work \
  --scratch-dir /tmp/fm-case/scratch --draws 5 --output /tmp/fm-report.json
```

`fixtures/smoke/` retains the old tiny frozen fixture. `reports/omnistation-smoke.json`
is historical evidence, not a measurement of this draft. New comparisons must
record matching source/wheel identities and decoded arrays/IPC parity. SQL
validation and compatibility checks run before timed retrieval. Each frozen
loader opening creates an independent managed base; repeated requests reuse
that loader. Base indexes/caches persist without automatic garbage collection.
