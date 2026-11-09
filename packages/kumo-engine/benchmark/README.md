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

Every cell requires a new private copy. Public `Trainer.fit` ingests a task
extension beneath its snapshot, so using the shared preparation directly would
mutate it. Copy only the prepared manifest's file whitelist, using independent
inodes; hardlinks would share page cache. Clone setup runs outside fit timers
and outside the constrained scope, and its duration is recorded separately:

```bash
python packages/kumo-engine/benchmark/native_io_isolation.py \
  --prepared-dir /local/prepared/binary --destination /local/trials/binary-mmap-0
```

Run a calibration cell in a fresh interpreter with a new output filename:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
DG_DIRECT_CACHE_GB=1 DG_IO_RINGS=2 DG_IO_QD=32 DG_IO_BATCH_MULT=1 \
python packages/kumo-engine/benchmark/native_io_tasks.py \
  --task binary --backend mmap --prepared-dir /local/trials/binary-mmap-0 \
  --output /local/results/binary-mmap-0.json --steps 128 \
  --batch-size 64 --workers 0 --threads 1 --rayon-threads 4 --seed 0
```

Repeat with `buffered` and `direct`. The primary comparison fixes loader workers
to zero so native counters cover the actual process. `--loader-mode standalone`
(the default) holds the same loader algorithm across all three backends. The
automatic settings and effective overrides are recorded separately. The `auto`
option measures the product's automatic policy (mmap fused, new backends
standalone); it is a separate diagnostic because those paths can select
different training rows. More workers require process aggregation for
native counters and are not the primary benchmark configuration.

The matrix's experimental `--io-batch-mult` sets the existing native
`DG_IO_BATCH_MULT` control (default 1, accepted range 1..32). Both the raw
environment value and parsed `requested_io_config.io_batch_mult` are recorded
separately from the model configuration hash. A value of 8 increases the owner
batch page target to eight times queue depth and also increases its bounded
page recycler reserve. This changes both aggregation and retained anonymous
memory, so comparisons cannot attribute an effect solely to larger I/O batches.
Keep the native wheel fixed when comparing these settings and retain its build
provenance; changing Python benchmark metadata does not rebuild that wheel.

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
The experiment pins the release native wheel to `b9aa15d9`. Python-only
benchmark changes can reuse it after verifying no native source difference
with `git diff --exit-code b9aa15d9 -- packages/diskgraph`. Record the
benchmark revision separately from the wheel's build revision and binary hashes.

The benchmark fixes distinct task namespaces for fit, first prediction, repeat
and reload. Production currently includes each fit's random UUID in its seeded
row-shuffle key; repeated fits with identical model seeds selected different
logical rows. The benchmark overrides only task materialization namespaces,
preserving public run IDs and bundle IDs. It performs four fresh ingests in
the private index and verifies their published manifests. This control is
recorded in the result; it is not a production reproducibility fix.

Each of those four native task ingests explicitly uses one Parquet source
loader (`context_threads=1`) and one encoder (`encode_threads=1`). Production
defaults remain unchanged. Repeated parallel task ingests assigned the same
physical seed row IDs to different logical entities and labels. The benchmark
still performs each complete ingest inside the public fit/prediction timer;
its serialized ingest cost is included. Sampler threads retain their separate
requested budget.
Every completed training batch contributes logical seed IDs, timestamps and
available targets to `work.logical_seed_sha256`. Cross-backend comparisons
require that hash to match as well as work counts and predictions.

The primary CPU comparison uses one Torch/OMP/MKL/OpenBLAS thread and four
sampler threads. A 128-batch binary diagnostic diverged with four Torch threads
despite identical batch tensors; one Torch thread matched model states,
gradients and predictions across mmap and buffered. The exact differing Torch
kernel is unproven. Each task requires its own correctness and pressure
validation before reporting comparisons.

Results retain the original manifest hash, source file hashes and sizes,
private copy mode and cloning time. After each run, baseline inventory, inode
identities, modification times and small metadata hashes must remain unchanged.
The matrix pins pristine file hashes across trials and deletes successful trial
copies only after saving results/resources and verifying the baseline. Failed
trial copies remain for investigation. `--cleanup-work` removes only that
runner's own successful model-work directory after all measurements and checks.

An uncontrolled run does not establish disk-bound performance. Use the matrix
driver to control CPU/memory limits, verify cache residency before each cell,
alternate backend order and collect child-inclusive cgroup resource counters.
The dataset must exceed the constrained memory budget by at least two times.
Calibrate each task's fixed step budget so every backend trains for at least
30 seconds, then collect at least five matched repetitions per task/regime.
Constrained training must show physical reads plus OS reclaim or native-cache
eviction; report OS pressure and native-cache pressure separately. A short
calibration or a large index alone does not establish those conditions.
Fewer than five pairs are labeled exploratory and have no confidence interval;
pilot runs remain exploratory. `PASS` records completed comparison checks.
Keep raw results, source/native provenance and all resource/correctness gates
with any reported median and paired comparison.
