# TT-NG Weighted Seed Sampling

This document describes the weighted replacement seed-sampling design added for
TT-NG task datasets. The implementation supports the DiskGraph side of Kumo's
`WeightMode.SAMPLE` parity work for binary classification:

- `majority_sampling_ratio` with sample-mode class rebalancing.
- Training-table row weights through the reserved `WEIGHT` column.
- Combined row weights and class weights, matching the legacy effective-weight
  model.

The companion Kumo PR is responsible for choosing the training policy,
computing epoch length, and passing a compact `weighted_seed_sampling` intent to
the wheel. DiskGraph is responsible for producing row-aligned sampling
artifacts and for drawing replacement seed node ids at runtime.

## Goals

The main goal is to preserve legacy `WeightMode.SAMPLE` semantics without
moving large train-table scans or sampled-id materialization into Kumo Python.

The desired effective sampling weight is:

```text
effective_weight(row) =
  row_weight(row) if a WEIGHT column exists else 1.0
  *
  class_weight(label) if majority_sampling_ratio applies else 1.0
```

From that contract, the supported variants are:

1. Majority-only sample mode:
   sample uniformly within false/true class buckets, with class bucket mass
   determined by Kumo's class weights.
2. Row-weight-only sample mode:
   sample directly from all train rows according to the row-level `WEIGHT`
   distribution.
3. Row-weight plus majority sample mode:
   sample from false/true row-weight CDF buckets, with each bucket's total mass
   multiplied by the class weight from Kumo.

## Non-Goals

This implementation intentionally does not change the normal unweighted runtime
iterator path. Weighted replacement sampling is only enabled when Kumo passes
`weighted_seed_sampling`.

It also does not implement lazy per-batch weighted drawing yet. The runtime
currently draws the epoch's worker-local replacement seed ids up front, then
serves batches from that vector. This is simpler and preserves deterministic
multi-worker behavior, but it can be optimized later.

Finally, this implementation does not make every lower-level train-table ingest
helper automatically request weighted sidecars. The product TT-NG materialize
path goes through the pybind executor stage metadata, where train-stage
sidecars are requested. Generic library helpers that construct `StageSpec`
directly keep the default no-sidecar behavior unless they call the sidecar-aware
stage-ingest helper with explicit `StageSidecarColumns`.

## Artifacts

Weighted sampling uses two families of stage-table sidecars, emitted only for
train-stage seed tables.

Binary label sidecars:

```text
<stage_table>/binary_label_false_node_ids.bin
<stage_table>/binary_label_true_node_ids.bin
```

Each file is a packed little-endian `u64` list of final stage-table node ids.
The files are emitted by `diskgraph-ingest/src/binary_label.rs` and described in
the task-dataset manifest as:

```json
{
  "binary_label_index": {
    "false_node_ids": "train_seeds/binary_label_false_node_ids.bin",
    "false_count": 123,
    "true_node_ids": "train_seeds/binary_label_true_node_ids.bin",
    "true_count": 456
  }
}
```

Sample-weight sidecars:

```text
<stage_table>/sample_weight_node_ids.bin
<stage_table>/sample_weight_cdf_f64.bin
<stage_table>/sample_weight_false_node_ids.bin
<stage_table>/sample_weight_false_cdf_f64.bin
<stage_table>/sample_weight_true_node_ids.bin
<stage_table>/sample_weight_true_cdf_f64.bin
```

The node-id files are packed little-endian `u64` final stage-table node ids.
The CDF files are packed little-endian `f64` cumulative weights aligned with
the corresponding node-id file. The manifest shape is:

```json
{
  "sample_weight_index": {
    "all": {
      "node_ids": "train_seeds/sample_weight_node_ids.bin",
      "cdf": "train_seeds/sample_weight_cdf_f64.bin",
      "count": 123,
      "total_weight": 456.0
    },
    "false_bucket": {
      "node_ids": "train_seeds/sample_weight_false_node_ids.bin",
      "cdf": "train_seeds/sample_weight_false_cdf_f64.bin",
      "count": 50,
      "total_weight": 100.0
    },
    "true_bucket": {
      "node_ids": "train_seeds/sample_weight_true_node_ids.bin",
      "cdf": "train_seeds/sample_weight_true_cdf_f64.bin",
      "count": 73,
      "total_weight": 356.0
    }
  }
}
```

The `all` bucket is used for row-weight-only sampling. The class buckets are
used when row weights and class weights are both present.

## Stage Ingest Capture

Stage sidecars are requested separately from ordinary graph/table config:

- `StageSpec` carries only the stage table name and parquet parts.
- `StageSidecarColumns` carries optional sidecar request columns.
- `ingest_stage_extension_with_sidecars` and
  `ingest_stage_extension_with_resolution_and_sidecars` accept a
  `HashMap<stage_table_name, StageSidecarColumns>`.

This is intentional. `TableSpec` describes a graph table and should not expose
weighted-training-only internals. Plain `ingest_stage_extension` remains the
default no-sidecar path.

The pybind TT executor sets these only for the train stage:

- `binary_label_column = TARGET1`
- `sample_weight_column = WEIGHT`

Validation is intentionally semantic rather than schema-only:

- Binary labels are accepted only when values can be interpreted as false/true.
  Invalid labels cause the binary label index to be omitted.
- Sample weights are accepted only when weights are finite and non-negative.
  Any invalid weight causes the sample weight index to be omitted.
- A total sample-weight mass of zero omits the sample-weight index because a
  replacement sampler cannot draw from an all-zero distribution.

The important alignment rule is that labels and weights are captured in final
node-write order, not source parquet order. During node encoding, DiskGraph
reads the label and weight before writing the row, but pushes the label/weight
spill only after the row survives the normal node-write path. That keeps each
spill record aligned with the final stage-table node id:

```text
spill record k in shard s -> final node id shard_min_id[s] + k
```

This avoids the earlier failure mode where a source-row index could diverge
from the final node-id sequence after drops, oversized-row handling, shard
routing, or rowpack write behavior.

## Sidecar Merge

During ingest finalization, after shard offsets are known and before temp files
are removed, DiskGraph merges per-shard spills into final sidecars:

1. `run_sample_weight_indexes`
2. `run_binary_label_indexes`

The order matters only because sample-weight class buckets optionally read the
binary-label spills while both temporary spill families still exist.

The merge is fail-loud for internal artifact corruption:

- if counts exist for `(table, shard)`, the corresponding spill file must exist;
- if a counted shard exists, the finalized shard offset must exist;
- if sample-weight class buckets need labels, the label spill must exist;
- writer flush errors are propagated.

Semantic invalidity, such as invalid labels or invalid weights, is handled
differently: the corresponding index is omitted. Missing internal artifacts are
not semantic invalidity; they indicate a broken materialization and fail ingest.

## Manifest Propagation

`StageExtensionResult` carries the sidecar indexes produced by ingest:

- `binary_label_indexes`
- `sample_weight_indexes`

The pybind TT executor attaches the relevant stage index metadata to the
task-dataset stage manifest. Kumo then copies those manifest entries into the
flat runtime manifest so the Python runtime wrapper can resolve relative
sidecar paths without scanning the train table.

This keeps the runtime contract explicit:

- If Kumo requests majority-only sample mode, the train stage must have
  `binary_label_index`.
- If Kumo requests row-weight sample mode, the train stage must have
  `sample_weight_index`.
- If Kumo requests row-weight plus class weighting, the train stage must have
  `sample_weight_index.false_bucket` and `sample_weight_index.true_bucket`.

Missing required manifest entries fail before training rather than silently
falling back to the natural seed order.

## Runtime Contract

Kumo passes `weighted_seed_sampling` to the Python wrapper. The wrapper resolves
manifest-relative sidecar paths and emits a compact JSON config consumed by the
Rust runtime iterator:

```json
{
  "num_samples": 8192,
  "buckets": [
    {
      "name": "false_bucket",
      "node_ids_path": ".../sample_weight_false_node_ids.bin",
      "cdf_path": ".../sample_weight_false_cdf_f64.bin",
      "count": 123,
      "total_weight": 456.0,
      "mass": 912.0
    }
  ]
}
```

A bucket without `cdf_path` is a uniform bucket. A bucket with `cdf_path` is a
row-weighted CDF bucket.

The runtime opens sidecar files using mmap-backed readers:

- `U64Sidecar` validates node-id file alignment and count.
- `F64Sidecar` validates CDF file alignment and count.
- `WeightedSeedBucket::open` checks manifest/file counts and verifies that the
  final CDF value matches `total_weight` within tolerance.

For each global draw position in the epoch:

1. Draw a bucket according to bucket `mass`.
2. Draw a node id inside that bucket.
   - Uniform buckets use `rng.gen_range(0..count)`.
   - CDF buckets use binary search over the cumulative f64 sidecar.
3. Keep the draw only if `position % num_workers == worker_id`.
4. Validate that the sampled node id belongs to the stage table.

This preserves deterministic multi-worker behavior because all workers see the
same global draw sequence and take disjoint positions from it.

## Why CDF Sidecars

The row-weight path needs arbitrary per-row probabilities. Class-only manifests
are not enough because two rows in the same class can have different weights.

The CDF sidecar design keeps large arrays out of Python:

- no Python pass over train-stage parquet during training setup;
- no Python list of all row weights;
- no repeated train-table scans for every epoch;
- runtime reads are mmap-backed and binary-searchable.

Alias sampling was considered, but CDFs were chosen first because they are
smaller to implement, easy to validate, stable on disk, and sufficient for this
parity path. Alias tables could reduce per-draw CPU from `O(log n)` to `O(1)`,
but would require two additional table arrays per bucket and more artifact
validation. That is a reasonable future optimization if weighted sampling
becomes hot.

## Tradeoffs

### Up-front Sampled Id Vector

The Rust runtime currently draws the worker-local sampled node ids for the epoch
before serving batches. This adds memory proportional to:

```text
num_worker_samples * sizeof(u64)
```

That is much smaller than Python-side full-sidecar materialization and is
bounded by Kumo's epoch-length/step caps, but it is still eager. A future runtime
iterator could sample per batch and avoid storing the whole worker-local vector.

### Zero-Mass Class Buckets

Rows with zero row weight are accepted and naturally get zero probability within
the all-row CDF. However, when row weights and class weights are combined, a
class bucket whose total row weight is zero is omitted from
`sample_weight_index`. That means Kumo cannot currently request a combined
row-weight/class-weight sample run if one class has rows but zero total row
weight.

This is a valid data edge case. A future improvement should represent zero-mass
class buckets explicitly and teach the runtime to accept zero-total CDF buckets
when their bucket mass is also zero.

### Invalid Labels or Weights

Invalid binary labels omit `binary_label_index`; invalid sample weights omit
`sample_weight_index`. This lets non-binary or malformed stages ingest normally
when weighted sample mode is not requested. If Kumo later requests
`WeightMode.SAMPLE`, the missing manifest entry causes a clear training-time
configuration error.

### Product Path vs Generic Helper

The product TT-NG path requests sidecars in `diskgraph-pybind/src/tt_executor.rs`
for train-stage task datasets by building a sidecar map and calling the
sidecar-aware stage-ingest helper. The lower-level
`diskgraph-train-table/src/tt_stage_ingest.rs` helper continues to call plain
`ingest_stage_extension`, so it does not opt in to weighted sidecars.

If future product materialization moves through that helper, it should pass the
same train-stage columns (`TARGET1`, `WEIGHT`) there as well.

## Failure Modes

Expected configuration failures:

- Kumo requests majority-only sampling but the train stage has no
  `binary_label_index`.
- Kumo requests row-weight sampling but the train stage has no
  `sample_weight_index`.
- The runtime bucket config has zero total sampling mass.
- Sidecar file counts do not match manifest counts.
- CDF count does not match node-id count.
- Final CDF value does not match manifest `total_weight`.

Materialization failures:

- counted sidecar spill missing;
- shard offset missing for a counted sidecar shard;
- label spill missing when sample-weight class buckets depend on it;
- output sidecar writer flush failure.

These fail loudly because a partial sidecar silently changes the training
distribution.

## Tests

The implementation includes focused coverage for:

- binary label sidecar merge behavior and invalid-label omission;
- sample weight sidecar CDF construction;
- hard failures for missing counted sidecar spills;
- train-stage sidecar emission through stage ingest;
- train-only sidecar request behavior;
- runtime weighted bucket validation and sampling;
- Python wrapper manifest-to-runtime-config translation;
- flat runtime manifest preservation for sidecar metadata.

The Kumo companion PR covers:

- epoch-length calculation from effective sampling mass;
- majority-only, row-weight-only, and combined policy construction;
- fallback from `WeightMode.SAMPLE` to weighted loss for refit modes;
- trainer/datamodule wiring for `weighted_seed_sampling`.

## Operational Notes

The sidecars live in the train-table/task-dataset extension next to stage-table
artifacts. They are not source-data artifacts and should be treated as part of
the materialized task dataset contract.

Because the runtime uses mmap, sidecar paths must point to local files in the
runtime manifest. The Kumo materialization path already localizes task-dataset
artifacts before training.

The normal iterator path remains unchanged unless `weighted_seed_sampling` is
provided. This keeps the blast radius limited to explicitly weighted sample-mode
training.
