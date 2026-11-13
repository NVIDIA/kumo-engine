# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Once-per-anchor scalar values, streamed to Parquet before fitting or epochs.

This is the internal scalar-preparation phase of ``Plan.materialize``. The
existing anchor reader carries computed columns with each anchor, so consuming
them needs no Python dictionary keyed by entity/time and preserves duplicate
entity/time pairs with different labels and sample identities.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from kumo.pytorch.anchors import (
    ALL_ANCHORS,
    AllAnchors,
    AnchorTable,
    describe_selection,
    split_selection,
)
from kumo.pytorch.assembly import _fingerprint
from kumo.pytorch.contract.sample import Aggregate, Compare, TargetValue, sample_value_types
from kumo.pytorch.data.records import FieldValue
from kumo.pytorch.dataset import Parquet

_META = b"kumo.pytorch.materialized_values"


@dataclass(frozen=True)
class MaterializedValues:
    """A durable scalar table and the anchor binding used to consume it.

    Pass ``values.anchors`` to the same plan's ``materialize``; the persisted
    binding is discovered automatically. ``values=`` remains accepted.
    Only scalar features and computed scalar targets are cached. Historical
    event tensors, neighborhoods and structured row targets stay on demand.
    """

    anchors: AnchorTable
    features: dict[str, str]
    targets: dict[str, str]
    contract: str
    dataset: str
    rows: int
    feature_stats: dict = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | Path) -> MaterializedValues:
        """Open the persisted scalar table and its original anchor binding."""
        import pyarrow.parquet as pq

        path = Path(path).resolve()
        from kumo.pytorch.dataset import _local_parquet_files

        files = [pq.ParquetFile(file) for file in _local_parquet_files(str(path))]
        file = files[0]
        payload = json.loads(file.schema_arrow.metadata[_META])
        binding = dict(payload["anchors"])
        # Generated labels remain task metadata, not caller-supplied targets.
        anchors = AnchorTable(source=Parquet(str(path)), **binding)
        feature_stats = {}
        metadata_path = path / "feature-stats.json"
        if path.is_dir() and metadata_path.exists():
            feature_stats = json.loads(metadata_path.read_text())
            columns_file = feature_stats.get("columns_file")
            feature_stats["stats"] = (
                json.loads((path / columns_file).read_text())["anchors"] if columns_file else {}
            )
        return cls(
            anchors,
            payload["features"],
            payload["targets"],
            payload["contract"],
            payload["dataset"],
            sum(file.metadata.num_rows for file in files),
            feature_stats,
        )

    def column_statistics(self, feature: str) -> dict:
        """The ordinary ingestion column entry for this selected feature."""
        return self.feature_stats["stats"][self.features[feature]]

    def encoder_columns(self, contract):
        from kumo.pytorch.column_stats import table_column_encoders

        return table_column_encoders(
            contract,
            scalar_columns=self.features,
            target_columns={**dict(self.anchors.targets), **self.targets},
        )

    def selected_column_stats(self, requests):
        from kumo.pytorch.column_stats import ColumnStats

        return ColumnStats(
            {item: self.feature_stats["stats"][item[1]] for item in requests},
            {"value": None, "unit": self.feature_stats.get("clock", "seconds")},
            # FitRecord's file references are durable index/col_stats entries.
            # These task-local values/statistics may disappear when the task closes.
        )

    def _bind(self, contract, source, anchors, dataset):
        if self.contract != contract.fingerprint:
            raise ValueError("materialized values belong to a different contract")
        if self.dataset != dataset:
            raise ValueError("materialized values belong to a different dataset")
        if _fingerprint(anchors) != _fingerprint(self.anchors):
            raise ValueError("use the materialized table's own anchor binding")
        native_binding = getattr(source, "bind_materialized_values", None)
        if native_binding is not None:
            return native_binding(self, contract)
        kind = (
            _ModelValuesSource if callable(getattr(source, "build_model", None)) else _ValuesSource
        )
        return kind(source, self, contract)


def materialize_values(
    plan,
    anchors: AnchorTable,
    destination,
    *,
    batch_size=1024,
    inference=None,
    stats_split=None,
):
    """Stream all requested scalar values once, with bounded anchor batches.

    No fitting occurs. Generated targets use their declared future windows;
    feature values keep their historical windows. No row-count/sample-ID dedupe
    or implicit train-only selection is applied. All-NULL first batches retain
    the declared Arrow types. The destination must not already exist.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    inference = plan.inference if inference is None else inference
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(anchors, AnchorTable):
        raise TypeError("materialize_values requires an AnchorTable")
    if anchors.candidates:
        raise ValueError("candidate sets are not scalar materialization inputs")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    from kumo.pytorch.preparation import check
    from kumo.pytorch.spec import ValidationError

    issues = check(plan.contract, plan.dataset, plan.engine, anchors=anchors, inference=inference)
    if issues:
        raise ValidationError(issues)
    sample = plan.contract.sample
    feature_types = sample_value_types(plan.contract.schema, sample)
    targets = (
        {}
        if inference
        else {
            name: expression
            for name, expression in sample.targets.items()
            if isinstance(expression, Aggregate | Compare)
        }
    )
    scalar = replace(
        sample,
        events={},
        current_events={},
        dated_entity_values={},
        neighborhoods={},
        candidate_sets={},
        targets=targets,
    )
    features = {name: f"_value_feature_{i}" for i, name in enumerate(sample.entity_values)}
    target_columns = {name: f"_value_target_{i}" for i, name in enumerate(targets)}
    reserved = {
        "_anchor_id",
        "_anchor_entity",
        "_anchor_time",
        "_anchor_split",
        *features.values(),
        *target_columns.values(),
    }
    if reserved.intersection(anchors.metadata):
        raise ValueError("anchor metadata uses a materialization-reserved column")

    if isinstance(anchors.source, Parquet):
        from kumo.pytorch.dataset import _local_parquet_files

        original = pq.read_schema(_local_parquet_files(anchors.source.path)[0])
        renamed = dict(anchors.source.columns)
        types = {renamed.get(field.name, field.name): field.type for field in original}
    else:
        types = {
            field.name: field.type for field in pa.Table.from_pylist(anchors.source.rows).schema
        }
    metadata = list(anchors.metadata)
    external = {name: f"_external_target_{i}" for i, name in enumerate(anchors.targets)}
    if set(external.values()).intersection(metadata):
        raise ValueError("anchor metadata uses a materialization-reserved target column")
    fields = [
        pa.field("_anchor_id", types[anchors.sample_id] if anchors.sample_id else pa.int64()),
        pa.field("_anchor_entity", types[anchors.entity_id]),
        pa.field("_anchor_time", types[anchors.as_of_time] if anchors.as_of_time else pa.int64()),
        pa.field("_anchor_split", pa.string()),
    ]
    fields += [pa.field(name, types[name]) for name in metadata]
    fields += [
        pa.field(column, sample.targets[name].dtype.to_arrow()) for name, column in external.items()
    ]
    fields += [
        pa.field(column, feature_types[name].to_arrow()) for name, column in features.items()
    ]
    fields += [
        pa.field(column, targets[name].dtype.to_arrow()) for name, column in target_columns.items()
    ]
    binding = dict(
        entity_id="_anchor_entity",
        as_of_time="_anchor_time" if anchors.as_of_time else None,
        sample_id="_anchor_id",
        split="_anchor_split" if anchors.split else None,
        metadata=[*metadata, *features.values(), *target_columns.values()],
        targets=external,
    )
    payload = dict(
        anchors=binding,
        features=features,
        targets=target_columns,
        contract=plan.contract.fingerprint,
        dataset=plan._dataset_fingerprint,
    )
    selection = split_selection(stats_split)
    stats_specs, stats_kinds = [], []
    if selection is not None:
        from kumo.pytorch.column_stats import table_column_encoders

        requested = {}
        for encoder in table_column_encoders(
            plan.contract, scalar_columns=features, target_columns={**external, **target_columns}
        ).values():
            for table, column, kind in encoder.requests:
                if table == "__materialized_anchors":
                    requested.setdefault(column, set()).add(kind)
        for column, kinds in sorted(requested.items()):
            dtype = next(field for field in fields if field.name == column).type
            from kumo.pytorch.materialization import _diskgraph_dtype

            stats_specs.append(dict(name=column, **_diskgraph_dtype(pa, dtype)))
            stats_kinds.append(sorted(kinds))
    schema = pa.schema(fields, metadata={_META: json.dumps(payload).encode()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".materialize-", dir=destination.parent))
    # The TT writer creates its own dataset directory beneath the owned staging root.
    output = temporary / "parts"
    stats_path = temporary / "stats"
    stats_writer = None
    stats_rows = 0
    present_splits = set()
    try:
        native_batches = getattr(plan._require_open(), "materialize_scalar_batches", None)
        native_writer = native_batches is not None or selection is not None
        if native_writer:
            import diskgraph

            writer = diskgraph.TaskParquetWriter(str(output))
        else:
            output.mkdir()
            writer = pq.ParquetWriter(output / "part-00000.parquet", schema)
        with writer:
            if native_writer:
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        [pa.array([], type=field.type) for field in schema], schema=schema
                    )
                )

            def write_batch(batch):
                nonlocal stats_writer, stats_rows
                writer.write_batch(batch)
                if selection is None:
                    return
                import pyarrow.compute as pc

                splits = batch.column("_anchor_split")
                present_splits.update(pc.unique(splits).to_pylist())
                if not isinstance(selection, AllAnchors):
                    mask = pc.fill_null(
                        pc.is_in(
                            splits,
                            value_set=pa.array(
                                [v for v in selection if v is not None], type=pa.string()
                            ),
                        ),
                        False,
                    )
                    if None in selection:
                        mask = pc.or_(mask, pc.is_null(splits))
                    batch = batch.filter(mask)
                stats_rows += batch.num_rows
                if not stats_specs:
                    return
                projected = pa.record_batch(
                    [batch.column(spec["name"]) for spec in stats_specs],
                    names=[spec["name"] for spec in stats_specs],
                )
                if stats_writer is None:
                    stats_writer = diskgraph.TaskParquetWriter(str(stats_path))
                stats_writer.write_batch(projected)

            if native_batches is not None and isinstance(anchors.source, Parquet):
                import pyarrow.compute as pc

                offset = 0
                for columns, computed_features, computed_targets, cutoffs in native_batches(
                    scalar, anchors, batch_size
                ):
                    n = len(cutoffs)
                    values = {
                        "_anchor_id": columns[anchors.sample_id]
                        if anchors.sample_id
                        else pa.array(range(offset, offset + n), type=pa.int64()),
                        "_anchor_entity": columns[anchors.entity_id],
                        "_anchor_time": columns[anchors.as_of_time]
                        if anchors.as_of_time
                        else cutoffs,
                        "_anchor_split": pc.cast(columns[anchors.split], pa.string())
                        if anchors.split
                        else pa.nulls(n, type=pa.string()),
                        **{name: columns[name] for name in metadata},
                        **{
                            column: columns[anchors.targets[name]]
                            for name, column in external.items()
                        },
                        **{column: computed_features[name] for name, column in features.items()},
                        **{
                            column: computed_targets[name]
                            for name, column in target_columns.items()
                        },
                    }
                    arrays = [
                        pc.cast(values[field.name], field.type, safe=True) for field in schema
                    ]
                    write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
                    offset += n
            else:
                for requests in _anchor_chunks(plan, anchors, batch_size):
                    computed = plan._require_open().build(
                        scalar, [replace(r, targets=()) for r in requests]
                    )
                    if len(computed) != len(requests):
                        raise ValueError("source returned a different number of scalar rows")
                    rows = []
                    for request, built in zip(requests, computed, strict=True):
                        if built.sample_id != request.sample_id:
                            raise ValueError("source changed scalar row identity/order")
                        values = {field.key: field.value for field in built.history.entity.values}
                        generated = {target.name: target.value for target in built.targets}
                        known = {target.name: target.value for target in request.targets}
                        rows.append(
                            {
                                "_anchor_id": request.sample_id,
                                "_anchor_entity": request.entity_id,
                                "_anchor_time": request.as_of_time if anchors.as_of_time else 0,
                                "_anchor_split": request.split,
                                **request.metadata,
                                **{column: known.get(name) for name, column in external.items()},
                                **{column: values.get(name) for name, column in features.items()},
                                **{
                                    column: generated.get(name)
                                    for name, column in target_columns.items()
                                },
                            }
                        )
                    for batch in pa.Table.from_pylist(rows, schema=schema).to_batches():
                        write_batch(batch)
            if selection is not None:
                if not isinstance(selection, AllAnchors) and selection - present_splits:
                    raise ValueError(f"no anchors in split(s) {selection - present_splits}")
                if stats_writer is not None:
                    stats_writer.close()
                    stats_writer = None
                if stats_specs and stats_path.exists():
                    from kumo.pytorch.column_stats import QUANTILE_GRID

                    clock = getattr(plan._require_open(), "_clock", "seconds")
                    directory = output / "encoder-statistics"
                    directory.mkdir()
                    diskgraph.write_column_stats_for_parquet(
                        [str(path) for path in sorted(stats_path.glob("part-*.parquet"))],
                        json.dumps(stats_specs),
                        json.dumps(stats_kinds),
                        str(directory),
                        quantiles=QUANTILE_GRID,
                        timestamp_unit=clock,
                    )
                    columns_file = "encoder-statistics/col_stats.json"
                else:
                    columns_file, clock = None, "seconds"
                (output / "feature-stats.json").write_text(
                    json.dumps(
                        dict(
                            format="column_stats_v1",
                            fit_split=describe_selection(selection),
                            rows=stats_rows,
                            features=features,
                            clock=clock,
                            columns_file=columns_file,
                        )
                    )
                )
        # A caller never sees a partial materialization. Do not overwrite an
        # existing table if another writer won the same destination.
        os.rename(output, destination)
    finally:
        import shutil

        if stats_writer is not None:
            stats_writer.close()
        shutil.rmtree(temporary)
    return MaterializedValues.open(destination)


def _anchor_chunks(plan, anchors, batch_size):
    """Reuse native ownership when present; otherwise scan bounded Parquet batches.

    The scan reuses ParquetAnchors.fetch's existing request conversion rules,
    but never calls its full-population constructor/to_table path.
    """
    native = getattr(plan._require_open(), "materialize_anchors", None)
    if native is not None or not isinstance(anchors.source, Parquet):
        task = (
            native(anchors, plan.contract.sample)
            if native is not None
            else plan.engine.materialize_anchors(anchors)
        )
        try:
            from itertools import islice

            positions = iter(
                task.order(ALL_ANCHORS, seed=0, epoch=0, shuffle=False, rank=0, world_size=1)
            )
            while chunk := list(islice(positions, batch_size)):
                yield task.fetch(ALL_ANCHORS, chunk)
        finally:
            task.close()
        return
    import numpy as np
    import pyarrow.dataset as pads

    from kumo.pytorch.engines.parquet_anchors import ParquetAnchors, _files, _physical

    physical = _physical(anchors)
    scanner = pads.dataset(_files(anchors.source.path), format="parquet").scanner(
        columns=sorted(set(physical.values())),
        batch_size=batch_size,
        batch_readahead=0,
        fragment_readahead=0,
    )
    offset = 0
    for batch in scanner.to_batches():
        reader = object.__new__(ParquetAnchors)
        reader._table = anchors
        reader._n = batch.num_rows
        reader._all_rows = np.arange(batch.num_rows, dtype=np.int64)
        reader._columns = {name: batch.column(column) for name, column in physical.items()}
        requests = reader.fetch(ALL_ANCHORS, range(batch.num_rows))
        if anchors.sample_id is None:
            requests = [
                replace(request, sample_id=request.sample_id + offset) for request in requests
            ]
        yield requests
        offset += batch.num_rows


class _ValuesSource:
    def __init__(self, source, values, contract):
        self.source, self.values = source, values
        self._materialized_values = values
        self.types = sample_value_types(contract.schema, contract.sample)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def column_stats(self, requests, cutoff):
        from kumo.pytorch.column_stats import ColumnStats

        task = [item for item in requests if item[0] == "__materialized_anchors"]
        stored = [item for item in requests if item[0] != "__materialized_anchors"]
        selected = self.values.selected_column_stats(task)
        base = self.source.column_stats(stored, cutoff) if stored else None
        return ColumnStats(
            {**(base.columns if base else {}), **selected.columns},
            base.cutoff if base else selected.cutoff,
            (*(base.files if base else ()), *selected.files),
        )

    def _online(self, spec):
        return replace(
            spec,
            entity_values={},
            targets={
                name: value
                for name, value in spec.targets.items()
                if name not in self.values.targets
            },
        )

    def _clean(self, request):
        internal = {*self.values.features.values(), *self.values.targets.values()}
        return replace(
            request,
            metadata={key: value for key, value in request.metadata.items() if key not in internal},
        )

    def _raw_requests(self, requests):
        return tuple(
            replace(r, targets=tuple(t for t in r.targets if t.name not in self.values.targets))
            for r in requests
        )

    def build(self, spec, requests):
        computed = self.source.build(self._online(spec), self._raw_requests(requests))
        result = []
        for request, built in zip(requests, computed, strict=True):
            fields = tuple(
                FieldValue(name, request.metadata[self.values.features[name]], self.types[name])
                for name, expression in spec.entity_values.items()
            )
            targets = (
                *built.targets,
                *(
                    TargetValue(name, request.metadata[column])
                    for name, column in self.values.targets.items()
                    if name in spec.targets and request.metadata[column] is not None
                ),
            )
            result.append(
                replace(
                    built,
                    history=replace(
                        built.history, entity=replace(built.history.entity, values=fields)
                    ),
                    targets=targets,
                    metadata=self._clean(request).metadata,
                )
            )
        return result


class _ModelValuesSource(_ValuesSource):
    def build_model(self, spec, requests):
        return self.build_model_ordered(spec, requests)

    def build_model_ordered(self, spec, requests, **kwargs):
        import pyarrow as pa

        builder = getattr(self.source, "build_model_ordered", self.source.build_model)
        built = builder(self._online(spec), self._raw_requests(requests), **kwargs)
        features = {
            name: pa.array(
                [r.metadata[self.values.features[name]] for r in requests],
                type=self.types[name].to_arrow(),
            )
            for name in spec.entity_values
        }
        targets = {
            **built.targets,
            **{
                name: pa.array(
                    [r.metadata[column] for r in requests], type=spec.targets[name].dtype.to_arrow()
                )
                for name, column in self.values.targets.items()
                if name in spec.targets
            },
        }
        return replace(
            built,
            entity_values=features,
            targets=targets,
            request_metadata=tuple({"split": r.split, **self._clean(r).metadata} for r in requests),
        )
