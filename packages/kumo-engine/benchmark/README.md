# Native I/O training and prediction benchmarks

`native_io_tasks.py` prepares deterministic local graphs and runs real Engine
training, held-out prediction, repeat prediction and bundle reload prediction.
The three workloads use uniform, hotspot and rotating community relationships
for binary classification, regression and temporal link prediction respectively.
These are synthetic workloads, not representative customer performance claims.

Use an installed candidate DiskGraph wheel built from this tree, together with
the matching Engine/Core/API source roots and their prepared runtime dependencies.
The benchmark uses CPU on Linux. It requires io_uring for buffered/direct runs;
failure to create it is an error, never a fallback to mmap.

Preparation is separate from the measurement and writes an immutable manifest:

```bash
python packages/kumo-engine/benchmark/native_io_tasks.py --prepare \
  --task binary --data-root /local/benchmark-source \
  --prepared-dir /local/prepared/binary --scale 1 --feature-width 256
```

Repeat for `regression` and `temporal_link`, each with its own prepared directory.
Each scale unit adds 128 users, 256 items and approximately 12,288 events across
72 weeks. Every table has a dense sequence feature used by `NumericalList` in
the fitted model. Scale grows the number of sampled entities and feature rows;
it does not add unreferenced padding. Actual index bytes are in the manifest.
Preparation records source hashes, generator configuration, index files, physical
training rows and stage timings. It must finish before resource-constrained runs.

Run a cell in a fresh interpreter with a new output filename:

```bash
PYTHONHASHSEED=0 DG_DIRECT_CACHE_GB=1 DG_IO_RINGS=2 DG_IO_QD=32 \
python packages/kumo-engine/benchmark/native_io_tasks.py \
  --task binary --backend mmap --prepared-dir /local/prepared/binary \
  --output /local/results/binary-mmap-0.json --steps 128 \
  --batch-size 64 --workers 0 --threads 4 --rayon-threads 4 --seed 0
```

Repeat with `buffered` and `direct`. The primary comparison fixes loader workers
to zero so native counters cover the actual process. `--loader-mode standalone`
(the default) holds the same loader algorithm across all three backends. The
automatic settings and effective overrides are recorded separately. The `auto`
option measures the product's automatic policy (mmap fused, new backends
standalone); it is a separate diagnostic because those paths can select
different training rows. More workers require process aggregation for
native counters and are not the primary benchmark configuration.

One seeded model trains a fixed number of training batches without search,
early stopping or refits. Completed fit/batch counts are checked. Prediction
uses up to 2,048 evenly spaced held-out entities (configurable at preparation);
values and decisions must match repeat and reload
predictions within absolute tolerance `1e-6`. Raw outputs allow cross-backend
comparison. Link output is top ten items per entity, with exact item identities
checked as well as scores and ranks. The result separates fit, first prediction, repeat prediction and
reload prediction. The fit-to-first-prediction timer includes public Engine
bundle creation, checkpoint work and prediction-table preparation; immutable
source generation, snapshot creation and training-table generation are excluded
and reported separately. No per-batch CUDA synchronization is installed.
Use `--build-provenance` to attach release/compiler/wheel/source hashes; the
runner rejects a native module that differs from that record before timing.

An uncontrolled run does not establish disk-bound performance. Use the matrix
driver to control CPU/memory limits, verify cache residency before each cell,
alternate backend order and collect child-inclusive cgroup resource counters.
The dataset must exceed the constrained memory budget by at least two times.
Keep raw results, source/native provenance and all resource/correctness gates
with any reported median and paired comparison.
