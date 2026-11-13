# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-owned seeds served by the same native iterator as Engine GNN tasks.

Only anchor rows are ingested. Task extensions live under the base index;
cleanup removes task data, never base graph files.
Python keeps bindings and stage counts, not a second full anchor column store.
"""

from __future__ import annotations

import heapq
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kumo.pytorch.anchors import ALL_ANCHORS, AllAnchors, AnchorTable
from kumo.pytorch.contract.sample import SampleRequest, TargetValue
from kumo.pytorch.data.records import cutoff_to_epoch
from kumo.pytorch.engines.parquet_anchors import _files, _physical, check_parquet_anchors
from kumo.pytorch.materialization import _diskgraph_dtype


class NativeAnchors:
    """An AnchorTask backed by a TT-NG seed extension, with bounded reads.

    Stage order is fixed at materialization. The native stage permutation owns
    epoch shuffling; ranks partition that concatenation before marked padding.
    Original row ordinals, rather than entity/time pairs, supply default IDs.
    """

    def __init__(self, source: Any, sample: Any, table: AnchorTable, *, directory=None):
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if issues := check_parquet_anchors(table):
            raise ValueError(issues[0].describe())
        self._source, self._sample, self._table = source, sample, table
        self._pa, self._pc = pa, pc
        parent = Path(source._index_path)
        (parent / "tasks").mkdir(exist_ok=True)
        (parent / "train_tables").mkdir(exist_ok=True)
        self._temporary = directory or tempfile.TemporaryDirectory(
            prefix="task-", dir=parent / "tasks"
        )
        self._extension = tempfile.TemporaryDirectory(
            prefix=Path(self._temporary.name).name + "-", dir=parent / "train_tables"
        )
        self.path = Path(self._temporary.name)
        self._closed = False
        self._length_caches = {}
        self._last_length_planning_stats = (0, 0)
        try:
            self._materialize(pq)
        except BaseException:
            self.close()
            raise

    def _materialize(self, pq: Any) -> None:
        pa, pc = self._pa, self._pc
        table = self._table
        physical = _physical(table)
        files = _files(table.source.path)
        schema = pq.read_schema(files[0])
        self._types = {name: schema.field(column).type for name, column in physical.items()}
        self._aliases = {name: f"anchor_col_{i}" for i, name in enumerate(table.columns)}
        # RowPack has no struct/map/list scalar. Carry those metadata values in
        # standard one-row Arrow IPC, not a second Python-owned table or pickle.
        self._ipc = set()
        for name, dtype in self._types.items():
            try:
                _diskgraph_dtype(pa, dtype)
            except TypeError:
                self._ipc.add(name)
        self._stages: list[dict[str, Any]] = []
        by_split: dict[str | None, dict[str, Any]] = {}
        writers = {}
        position = 0
        view = Path(self._source._index_path)
        extension = Path(self._extension.name)
        try:
            for file in files:
                for batch in pq.ParquetFile(file).iter_batches(
                    columns=sorted(set(physical.values())), batch_size=32768
                ):
                    columns = {name: batch.column(column) for name, column in physical.items()}
                    entity = self._source._tables[self._sample.entity.table]
                    dtype = entity.columns[entity.primary_key]
                    entities = pa.array(
                        [
                            dtype.normalize(value, path="anchors.entity_id")
                            for value in columns[table.entity_id].to_pylist()
                        ],
                        type=dtype.to_arrow(),
                    )
                    dense = self._source._sampler.lookup_node_ids(
                        self._sample.entity.table, entities
                    )
                    if any(value is None for value in dense):
                        raise ValueError("anchor table contains an unknown graph entity")
                    split = (
                        columns[table.split]
                        if table.split is not None
                        else pa.nulls(batch.num_rows, type=pa.string())
                    )
                    if split.type != pa.string():
                        split = pc.cast(split, pa.string(), safe=True)
                    payload = {}
                    for name, values in columns.items():
                        if name in self._ipc:
                            encoded = []
                            for i in range(batch.num_rows):
                                sink = pa.BufferOutputStream()
                                with pa.ipc.new_stream(
                                    sink, pa.schema([("value", values.type)])
                                ) as writer:
                                    writer.write_batch(
                                        pa.record_batch([values.slice(i, 1)], names=["value"])
                                    )
                                encoded.append(sink.getvalue().to_pybytes())
                            values = pa.array(encoded, type=pa.binary())
                        elif pa.types.is_timestamp(values.type):
                            # The payload is not an index clock. Keep subsecond
                            # precision here; only the reserved cutoff is seconds.
                            values = pc.cast(values, pa.int64())
                        payload[self._aliases[name]] = values
                    payload["anchor_position"] = pa.array(
                        range(position, position + batch.num_rows), type=pa.int64()
                    )
                    payload["ENTITY1"] = entities
                    payload["entity_node_id"] = pa.array(dense, type=pa.uint64())
                    for index, column in enumerate(table.targets.values(), 1):
                        # Named RDL labels also occupy the existing native TT target slots.
                        target = columns[column]
                        if target.type in (pa.float32(), pa.float64(), pa.int64(), pa.bool_()):
                            payload[f"TARGET{index}"] = target
                    if table.as_of_time is not None:
                        times = columns[table.as_of_time].to_pylist()
                        if any(value is None for value in times):
                            raise ValueError("anchor prediction times cannot be NULL")
                        payload["TIMESTAMP"] = pa.array(
                            [cutoff_to_epoch(value) for value in times], type=pa.timestamp("s")
                        )
                    data = pa.table(payload)
                    for split_name in pc.unique(split).to_pylist():
                        if split_name not in by_split:
                            stage = {
                                "split": split_name,
                                "name": f"stage_{len(by_split)}",
                                "count": 0,
                            }
                            stage["path"] = self.path / f"{stage['name']}.parquet"
                            self._stages.append(stage)
                            by_split[split_name] = stage
                            writers[split_name] = self._source._diskgraph.TaskParquetWriter(
                                str(stage["path"])
                            )
                        stage = by_split[split_name]
                        selected = data.filter(
                            pc.is_null(split) if split_name is None else pc.equal(split, split_name)
                        )
                        writers[split_name].write_batch(selected.to_batches()[0])
                        stage["count"] += len(selected)
                    position += batch.num_rows
        finally:
            for writer in writers.values():
                writer.close()
        manifest = {
            "kind": "diskgraph_task_dataset",
            "graph": {"index_path": str(view)},
            "task": {"is_temporal": table.as_of_time is not None},
            "stages": {},
        }
        if self._stages:
            specs = []
            for stage in self._stages:
                specs.append(
                    {
                        "name": stage["name"],
                        "columns": [
                            {"name": field.name, **_diskgraph_dtype(pa, field.type)}
                            for field in pq.read_schema(stage["path"] / "part-00000.parquet")
                        ],
                        "primary_key": None,
                        "create_time": "TIMESTAMP" if table.as_of_time is not None else None,
                        "fkeys": [],
                        "source": {
                            "type": "parquet",
                            "paths": [str(p) for p in sorted(stage["path"].glob("part-*.parquet"))],
                        },
                    }
                )
                manifest["stages"][stage["name"]] = {
                    "table_name": stage["name"],
                    "index_path": str(extension),
                    "row_count": stage["count"],
                }
            threads = self._source._num_threads
            options = (
                {}
                if threads is None
                else {
                    key: threads for key in ("context_threads", "encode_threads", "index_threads")
                }
            )
            self._source._diskgraph.ingest(
                json.dumps(
                    {
                        "dataset_name": "anchors",
                        "timestamp_unit": "seconds",
                        "num_shards": 1,
                        "tables": specs,
                    }
                ),
                str(extension),
                parent_index_path=str(view),
                col_stats=False,
                **options,
            )
        self._manifest = manifest
        path = self.path / "runtime.json"
        path.write_text(json.dumps(manifest))
        self._runtime = self._source._diskgraph.RuntimeIndex(
            str(path), rnd_seed=0, num_threads=self._source._num_threads
        )
        if self._stages:
            # Preparation may use a base reader. The loading loop uses only the
            # combined runtime context for anchors, histories and fallback reads.
            self._history_engine = self._runtime.sampler(self._stages[0]["name"])
            self._source._bind_sampler(self._history_engine)

    def _selection(self, selection: Any) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("anchor task is closed")
        if not isinstance(selection, AllAnchors):
            missing = selection - {stage["split"] for stage in self._stages} - {None}
            if missing:
                raise ValueError(
                    f"no anchors in split(s) {sorted(missing)}; splits present: {self.splits()}"
                )
        return [
            stage
            for stage in self._stages
            if isinstance(selection, AllAnchors) or stage["split"] in selection
        ]

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(stage["split"] for stage in self._stages if stage["split"] is not None))

    def count(self, selection: Any = ALL_ANCHORS) -> int:
        return sum(stage["count"] for stage in self._selection(selection))

    def _iter(
        self,
        stage: Any,
        *,
        ids=None,
        batch_size=1024,
        seed=0,
        epoch=0,
        shuffle=False,
        rank=0,
        world_size=1,
        plan=None,
        projections=None,
        history=None,
        neighborhoods=None,
        length_planning=None,
    ):
        iterator, _ = self._runtime.iter(
            stage["name"],
            plan or {"layers": []},
            world_size,
            rank,
            shuffle,
            epoch,
            batch_size,
            rnd_seed=seed,
            entity_node_type=self._sample.entity.table,
            fused=False,
            swap_edge_direction=False,
            selected_stage_node_ids=ids,
            projections=projections,
            disable_pseudo_time=True,
        )
        if neighborhoods:
            iterator.with_neighborhoods(json.dumps(neighborhoods))
        if history is not None:
            plan_json, merges_json = history
            iterator.with_history(
                self._history_engine, self._sample.entity.table, plan_json, merges_json
            )
        if length_planning is not None:
            length_contract, token_lengths = length_planning
            payload = json.loads(length_contract)
            costs = {
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "max_samples",
                    "max_tokens",
                    "packed",
                    "nested_padded",
                    "bucket",
                    "drop_last",
                }
            }
            key = (stage["name"], json.dumps(costs, sort_keys=True))
            if key not in self._length_caches:
                import hashlib

                filename = hashlib.sha256(key[1].encode()).hexdigest()[:24]
                fields = (
                    1
                    if payload["history_only"] and payload["max_tokens"] is None
                    else 3
                    if payload["nested_padded"]
                    else 2
                )
                self._length_caches[key] = self._source._diskgraph.RuntimeLengthCache(
                    stage["count"],
                    str(self.path / f"lengths-{stage['name']}-{filename}.bin"),
                    fields,
                )
            cache = self._length_caches[key]
            iterator.with_length_planning(cache, length_contract, token_lengths)
        return iterator

    def _batch_values(self, stage, runtime):
        """Retain typed label/value columns and adapt only public identity metadata."""
        from kumo.pytorch.data.model import ModelSourceBatch

        pa = self._pa
        key = next(k for k in runtime["raw_features"] if k.endswith("/" + stage["name"]))
        data = pa.Table.from_batches(runtime["raw_features"][key]["record_batches"])
        columns = {}
        for name, alias in self._aliases.items():
            values = data[alias]
            if name in self._ipc:
                values = pa.array(
                    [pa.ipc.open_stream(v.as_py()).read_all()["value"][0].as_py() for v in values],
                    type=self._types[name],
                )
            else:
                values = values.cast(self._types[name])
            columns[name] = values
        table = self._table
        n = data.num_rows
        bound = getattr(self._source, "_materialized_values", None)
        internal = {
            *getattr(bound, "features", {}).values(),
            *getattr(bound, "targets", {}).values(),
        }
        metadata = {
            name: columns[name].to_pylist() for name in table.metadata if name not in internal
        }
        return ModelSourceBatch(
            sample_ids=tuple(columns[table.sample_id].to_pylist())
            if table.sample_id
            else tuple(data["anchor_position"].to_pylist()),
            entity_ids=tuple(columns[table.entity_id].to_pylist()),
            evaluation_times=tuple(runtime["task"]["seed_timestamps"].to_pylist())
            if table.as_of_time
            else (0,) * n,
            entity_type=self._sample.entity.entity_type,
            entity_values={
                name: columns[column].combine_chunks()
                for name, column in getattr(bound, "features", {}).items()
            },
            events={},
            row_targets={},
            targets={
                name: columns[column].combine_chunks()
                for name, column in {**table.targets, **getattr(bound, "targets", {})}.items()
            },
            splits=(stage["split"],) * n,
            request_metadata=tuple(
                {"split": stage["split"], **{k: v[i] for k, v in metadata.items()}}
                for i in range(n)
            ),
        )

    def bind_token_lengths(self, assembler, contract, statistics):
        """Reuse #307's constant costs and optional recipe-bound per-node files."""
        from kumo.pytorch.column_stats import (
            _stored_column,
            read_token_lengths,
            token_length_columns,
            token_lengths_file,
        )
        from kumo.pytorch.contract.batch.validation import resolved_fields

        self._event_costs = {}
        self._token_maps = []
        name = assembler.spec.batching.length_input
        if name is None:
            return
        selected = {}
        for use in contract.batch.inputs[name].representation.fields:
            for variant, _, _ in resolved_fields(contract, name, use.field):
                stored = _stored_column(contract.sample, contract.schema, variant)
                if stored is not None and variant[0] in {"events", "current_events"}:
                    selected.setdefault(stored[0], set()).add(stored[1])
        cutoff = statistics.get("cutoff")
        ticks = (
            None
            if cutoff is None
            else self._source._diskgraph.convert_ticks(
                cutoff, "nanoseconds", self._source._clock, rounding="floor"
            )
        )
        for table, (constant, columns, required) in token_length_columns(
            contract, assembler
        ).items():
            entry = statistics.get("token_lengths", {}).get(table)
            expected = (
                token_lengths_file(table, columns, required, ticks, self._source._clock)
                if columns or required
                else None
            )
            if entry != {"constant": constant, "file": expected}:
                raise ValueError(
                    "event token costs do not match the designated input's fitted recipe"
                )
            fixed, values = read_token_lengths(self._source._index_path, entry)
            if values is not None:
                self._token_maps.append(values)
            self._event_costs[table] = dict(
                columns=selected.get(table, set()),
                text_columns={column: tokenizer for column, tokenizer, _ in columns},
                constant=fixed,
                values=None if values is None else self._pa.array(values),
            )

    def _length_contract(self, assembler, compiled, merges, selections, maximum, drop_last):
        """Lower frozen encoder costs to DG row metadata, never invoke an encoder.

        #307 supplies recipe-bound per-event uint16 costs indexed by DG node ID.
        They are mapped once and reused for selected/capped histories. Native
        planning adds derived/missing tokens and applies layout/truncation;
        the metadata producer does not decide sample or batch composition.
        """
        from copy import deepcopy

        from kumo.pytorch.contract.batch import (
            Columns,
            Composite,
            EventSource,
            Flatten,
            Nested,
            Packed,
            Tokens,
        )
        from kumo.pytorch.contract.batch.encodings import Text
        from kumo.pytorch.contract.sample import Constant, Derived, EventTimeDelta, Field

        policy = assembler.spec.batching
        prepared = assembler._prepared_inputs[policy.length_input]
        spec = prepared.spec
        if not isinstance(spec.source, EventSource) or compiled is None:
            raise ValueError(
                "native length planning currently requires an EventSource length input"
            )
        if (
            not isinstance(spec.layout, (Packed, Nested))
            and spec.layout.__class__.__name__ != "Padded"
        ):
            raise ValueError("native length planning requires Packed/Padded or Nested layout")
        event_source = spec.source
        selection_key = (
            tuple(event_source.events),
            event_source.max_events,
            event_source.keep,
            event_source.order,
        )
        merge = merges[list(selections).index(selection_key)]
        query = deepcopy(compiled.plan)
        query["aggregates"] = []
        sequence_names = set(merge["sequences"])
        query["sequences"] = [item for item in query["sequences"] if item["name"] in sequence_names]
        frame_names = {item["frame"] for item in query["sequences"]}
        query["frames"] = [item for item in query["frames"] if item["name"] in frame_names]
        columns = {}
        fixed = {}
        event_costs = getattr(self, "_event_costs", {})
        token_lengths = {}
        for sequence in query["sequences"]:
            logical = compiled.sequences[sequence["name"]].logical_name
            event = assembler._model_sample.events[logical]
            costs = []
            fixed[sequence["name"]] = 0
            from kumo.pytorch.contract.sample import destination_table

            table_name = destination_table(
                self._source.schema, self._sample.entity.table, event.rows.path
            )
            stored_costs = event_costs.get(table_name)
            if isinstance(spec.representation.row, (Columns, Composite)):
                # Row/compound inputs cost one event, but counting metadata still
                # decides whether Composite.required_fields drops that event.
                if (
                    isinstance(spec.representation.row, Composite)
                    and stored_costs
                    and stored_costs["values"] is not None
                ):
                    key = f"stored-event:{table_name}"
                    token_lengths[key] = [stored_costs["values"]]
                    costs.append(
                        dict(
                            column="",
                            metadata=key,
                            missing_token=False,
                            required=False,
                            validity_only=True,
                        )
                    )
                columns[sequence["name"]] = costs
                sequence["columns"] = [compiled.sequences[sequence["name"]].primary_key]
                continue
            if stored_costs and isinstance(spec.representation.row, Tokens):
                declared_columns = [
                    event.fields[item[0][-1]].column
                    for use in spec.representation.fields
                    for item in prepared.declarations[use.field]
                    if item[0][0] == "events"
                    and item[0][1] == logical
                    and isinstance(event.fields[item[0][-1]], Field)
                    and not event.fields[item[0][-1]].path
                ]
                if set(declared_columns) != set(stored_costs["columns"]) or len(
                    declared_columns
                ) != len(set(declared_columns)):
                    raise ValueError(
                        "event token metadata does not match this component's stored fields"
                    )
                if spec.representation.row.missing == "emit":
                    fixed[sequence["name"]] += stored_costs["constant"]
                if stored_costs["values"] is not None:
                    key = f"stored-event:{table_name}"
                    token_lengths[key] = [stored_costs["values"]]
                    costs.append(
                        dict(
                            column="",
                            metadata=key,
                            missing_token=False,
                            required=False,
                            validity_only=False,
                        )
                    )

            def always_present(value):
                if isinstance(value, Constant):
                    return value.value is not None
                if isinstance(value, EventTimeDelta):
                    return event.rows.time is not None
                if isinstance(value, Field):
                    return (
                        not value.path
                        and event.rows.time is not None
                        and value.column == event.rows.time.column
                    )
                if (
                    isinstance(value, Derived)
                    and value.operation.__class__.__name__ == "CalendarPart"
                ):
                    return all(always_present(event.fields[name]) for name in value.inputs)
                return False

            if not isinstance(spec.representation.row, Columns):
                for use in spec.representation.fields:
                    declarations = [
                        item
                        for item in prepared.declarations[use.field]
                        if item[0][0] == "events" and item[0][1] == logical
                    ]
                    if not declarations:
                        continue
                    local = declarations[0][0][-1]
                    value = event.fields[local]
                    row = spec.representation.row
                    composite = isinstance(row, Composite)
                    required = (
                        use.field in row.required_fields
                        if composite
                        else row.missing == "skip_event"
                    )
                    if composite and not required:
                        continue
                    binding = prepared.state[use.field]
                    supplied = (
                        isinstance(binding.encoder.config, Text) or declarations[0][2] == "repeated"
                    )
                    direct = isinstance(value, Field) and not value.path
                    if (
                        not composite
                        and direct
                        and stored_costs
                        and value.column in stored_costs["columns"]
                        and (value.column in stored_costs["text_columns"] or row.missing == "emit")
                    ):
                        if value.column in stored_costs["text_columns"] and (
                            required or row.missing == "emit"
                        ):
                            costs.append(
                                dict(
                                    column=value.column,
                                    metadata=None,
                                    missing_token=row.missing == "emit",
                                    required=required,
                                    validity_only=row.missing != "emit",
                                    missing_only=True,
                                )
                            )
                        continue
                    if (
                        not composite
                        and not supplied
                        and (row.missing == "emit" or always_present(value))
                    ):
                        fixed[sequence["name"]] += 1
                        continue
                    if not supplied and (
                        not isinstance(value, Field) or value.path or value.column is None
                    ):
                        if (
                            isinstance(value, Constant)
                            and value.value is None
                            and row.missing == "omit"
                        ):
                            continue
                        raise ValueError(
                            "native length planning needs validity metadata for this derived field"
                        )
                    metadata = None
                    column = None if supplied and not composite else value.column
                    if column is None and metadata is None:
                        raise ValueError(
                            f"post-fit text token lengths missing for {logical}.{local}"
                        )
                    costs.append(
                        dict(
                            column=column or "",
                            metadata=metadata,
                            missing_token=not composite and row.missing == "emit",
                            required=required,
                            validity_only=composite,
                        )
                    )
            columns[sequence["name"]] = costs
            primary_key = compiled.sequences[sequence["name"]].primary_key
            sequence["columns"] = list(
                dict.fromkeys(
                    [primary_key, *(item["column"] for item in costs if item["metadata"] is None)]
                )
            )
        for frame in query["frames"]:
            frame["columns"] = sorted(
                {
                    column
                    for sequence in query["sequences"]
                    if sequence["frame"] == frame["name"]
                    for column in sequence["columns"]
                }
            )
        row = spec.representation.row
        flatten = spec.representation.history
        contract = json.dumps(
            dict(
                query=json.dumps(query),
                merge=merge["name"],
                columns=columns,
                fixed=fixed,
                history_only=isinstance(row, (Columns, Composite)),
                row_maximum=row.max_tokens if isinstance(row, Tokens) else None,
                flatten=dict(
                    maximum=flatten.max_tokens,
                    keep_last=flatten.keep == "last",
                    event_boundary=flatten.prefer_event_boundary,
                    separator=flatten.separator is not None,
                    between=flatten.separator_placement == "between",
                    before=flatten.before is not None,
                    after=flatten.after is not None,
                )
                if isinstance(flatten, Flatten)
                else None,
                max_samples=maximum,
                max_tokens=policy.max_tokens,
                packed=isinstance(spec.layout, Packed)
                or isinstance(spec.layout, Nested)
                and isinstance(spec.layout.fields, Packed),
                nested_padded=isinstance(spec.layout, Nested)
                and not isinstance(spec.layout.fields, Packed),
                bucket=policy.bucket_by_length,
                drop_last=drop_last,
            ),
            sort_keys=True,
        )
        return contract, token_lengths

    def _model_batches(
        self,
        assembler,
        stage,
        *,
        plan,
        projections,
        size,
        neighborhoods,
        maximum,
        seed,
        epoch,
        shuffle,
        rank,
        world_size,
        drop_last,
    ):
        """One native iterator call per batch; no request objects or re-batching."""
        from dataclasses import replace

        from kumo.pytorch.anchors import PADDING_KEY
        from kumo.pytorch.contract.batch import EventSource
        from kumo.pytorch.engines.diskgraph import _TemporalPlanCompiler
        from kumo.pytorch.task import _padding_sample_ids, _surface_padding

        assembler._validate_model_contract()
        source = self._source
        sample = assembler._model_sample
        orders = {}
        selections = {}
        for prepared in assembler._prepared_inputs.values():
            event_source = prepared.spec.source
            if isinstance(event_source, EventSource):
                key = (
                    tuple(event_source.events),
                    event_source.max_events,
                    event_source.keep,
                    event_source.order,
                )
                selections[key] = event_source
                for name in event_source.events:
                    orders.setdefault(name, set()).add(event_source.order)
        compiled = (
            _TemporalPlanCompiler(source).compile(
                sample, event_orders={k: tuple(sorted(v)) for k, v in orders.items()}
            )
            if source._needs_temporal_query(sample)
            else None
        )
        merges = []
        merge_keys = []
        if compiled is not None:
            for key, selection in selections.items():
                names = [
                    next(
                        native
                        for native, output in compiled.sequences.items()
                        if output.logical_name == name and output.order == selection.order
                    )
                    for name in selection.events
                ]
                merges.append(
                    dict(
                        name=str(len(merges)),
                        sequences=names,
                        ascending=[selection.order == "ascending"] * len(names),
                        maximum=selection.max_events,
                        keep_last=selection.keep == "last",
                        descending=selection.order == "descending",
                    )
                )
                merge_keys.append(key)
        history = None if compiled is None else (json.dumps(compiled.plan), json.dumps(merges))
        policy = assembler.spec.batching
        length_planning = None
        if policy.max_tokens is not None or policy.bucket_by_length:
            length_planning = self._length_contract(
                assembler, compiled, merges, selections, maximum, drop_last
            )
        options = dict(
            length_planning=length_planning,
            plan=plan,
            projections=projections,
            neighborhoods=neighborhoods,
            history=history,
            batch_size=size,
            seed=seed,
            epoch=epoch,
        )
        owned = max(0, (stage["count"] - rank + world_size - 1) // world_size)
        missing = -(-stage["count"] // world_size) - owned

        def runtime_batches():
            remaining = missing
            first = None
            emitted = 0
            iterator = self._iter(
                stage, **options, shuffle=shuffle, rank=rank, world_size=world_size
            )
            for runtime in iterator:
                ids = runtime["task"]["tt_stage_node_ids"]
                n = len(ids)
                if first is None:
                    first = ids[0].as_py()
                emitted += n
                if remaining and emitted == owned and n < size:
                    # Reuse native selected-row iteration for the bounded final
                    # batch, including repeated padding seeds and their histories.
                    chosen = ids.to_pylist() + [first] * remaining
                    runtime = next(self._iter(stage, **options, ids=chosen))
                    yield runtime, remaining
                    remaining = 0
                else:
                    yield runtime, 0
            if length_planning is not None:
                self._last_length_planning_stats = iterator.length_planning_stats()
            if remaining:
                yield (
                    next(
                        self._iter(
                            stage, **options, ids=[0 if first is None else first] * remaining
                        )
                    ),
                    remaining,
                )

        for runtime, padded in runtime_batches():
            anchors = self._batch_values(stage, runtime)
            if length_planning is None and anchors.batch_size < size and drop_last:
                continue
            if padded:
                flags = [False] * (anchors.batch_size - padded) + [True] * padded
                anchors = replace(
                    anchors,
                    sample_ids=_padding_sample_ids(anchors.sample_ids, flags),
                    request_metadata=tuple(
                        {**meta, PADDING_KEY: True} if flag else meta
                        for meta, flag in zip(anchors.request_metadata, flags, strict=True)
                    ),
                )
            with runtime_features(source, runtime):
                values = source.build_model_batch(sample, runtime, anchors, compiled)
                values = replace(
                    values,
                    merged_events={
                        key: runtime["history_merges"][str(i)] for i, key in enumerate(merge_keys)
                    },
                )
                yield _surface_padding(
                    assembler._collate(values, epoch=epoch, seed=seed, model=True)
                )

    def _requests(self, stage: Any, runtime: Any) -> list[SampleRequest]:
        pa = self._pa
        key = next(name for name in runtime["raw_features"] if name.endswith("/" + stage["name"]))
        data = pa.Table.from_batches(runtime["raw_features"][key]["record_batches"])
        columns = {}
        for name, alias in self._aliases.items():
            values = data[alias]
            if name in self._ipc:
                columns[name] = [
                    pa.ipc.open_stream(value.as_py()).read_all()["value"][0].as_py()
                    for value in values
                ]
            else:
                columns[name] = values.cast(self._types[name]).to_pylist()
        original_positions = data["anchor_position"].to_pylist()
        table = self._table
        return [
            SampleRequest(
                sample_id=columns[table.sample_id][i] if table.sample_id else original_positions[i],
                entity_id=columns[table.entity_id][i],
                as_of_time=columns[table.as_of_time][i] if table.as_of_time else 0,
                targets=tuple(
                    TargetValue(name, columns[column][i])
                    for name, column in sorted(table.targets.items())
                ),
                split=stage["split"],
                metadata={name: columns[name][i] for name in table.metadata},
            )
            for i in range(len(original_positions))
        ]

    def order(
        self, selection: Any, *, seed: int, epoch: int, shuffle: bool, rank: int, world_size: int
    ) -> Iterator[int]:
        stages = self._selection(selection)
        if not shuffle and len(stages) > 1:
            for ordinal, (_, position, _) in enumerate(self._merged_rows(stages)):
                if ordinal % world_size == rank:
                    yield position
            return
        offset = 0
        for stage in stages:
            for batch in self._iter(
                stage,
                seed=seed,
                epoch=epoch,
                shuffle=shuffle,
                rank=(rank - offset) % world_size,
                world_size=world_size,
            ):
                yield from (
                    offset + value for value in batch["task"]["tt_stage_node_ids"].to_pylist()
                )
            offset += stage["count"]

    def _stage_rows(self, stage: Any, offset: int):
        for runtime in self._iter(stage, projections={name: [] for name in self._source._tables}):
            key = next(
                name for name in runtime["raw_features"] if name.endswith("/" + stage["name"])
            )
            data = self._pa.Table.from_batches(runtime["raw_features"][key]["record_batches"])
            rows = self._requests(stage, runtime)
            yield from zip(
                data["anchor_position"].to_pylist(),
                (offset + i for i in runtime["task"]["tt_stage_node_ids"].to_pylist()),
                rows,
                strict=True,
            )

    def _merged_rows(self, stages: Any):
        # A row-sized heap plus one bounded native batch per stage, not an
        # all-anchor position array. GNN fusion remains the single-stage path.
        streams = []
        offset = 0
        for stage in stages:
            streams.append(self._stage_rows(stage, offset))
            offset += stage["count"]
        return heapq.merge(*streams, key=lambda row: row[0])

    def fitting_requests(self, selection: Any) -> Sequence[SampleRequest]:
        # The current encoder fitter consumes a sequence. Preserve source order
        # without first creating an all-anchor Python position list and then
        # reading every native row a second time through positional fetch.
        return [request for _, _, request in self._merged_rows(self._selection(selection))]

    def iteration_anchors(self, selection: Any, *, shuffle: bool):
        # Refit selects native stage batches, never a Python union of anchors.
        return self

    def fetch(self, selection: Any, positions: Sequence[int]) -> Sequence[SampleRequest]:
        stages = self._selection(selection)
        count = sum(stage["count"] for stage in stages)
        grouped = defaultdict(list)
        for output, position in enumerate(positions):
            if not 0 <= position < count:
                raise IndexError(position)
            offset = 0
            for i, stage in enumerate(stages):
                if position < offset + stage["count"]:
                    grouped[i].append((output, position - offset))
                    break
                offset += stage["count"]
        result: list[Any] = [None] * len(positions)
        for i, entries in grouped.items():
            for start in range(0, len(entries), 32768):
                part = entries[start : start + 32768]
                values = []
                for batch in self._iter(stages[i], ids=[position for _, position in part]):
                    values.extend(self._requests(stages[i], batch))
                for (output, _), value in zip(part, values, strict=True):
                    result[output] = value
        return result

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._closed = True
            sampler = getattr(self, "_history_engine", None)
            self._source._release_sampler(sampler)
            self._runtime = None
            self._history_engine = None
            del sampler
            self._extension.cleanup()
            self._length_caches.clear()
            getattr(self, "_token_length_arrays", {}).clear()
            self._token_maps = []
            self._temporary.cleanup()

    def batches(
        self,
        assembler: Any,
        selection: Any,
        *,
        chunk: int,
        batch_size: int | None,
        seed: int,
        epoch: int,
        shuffle: bool,
        rank: int,
        world_size: int,
        drop_last: bool | None,
        output_format: str,
    ):
        """Read native anchor batches and retain the existing fitted assembler.

        Fixed-size GNN batches reuse TT-NG topology/features directly. Boundary
        batches spanning two split stages use the normal sampler, since their
        topology cannot be concatenated without changing local node indices.
        Token-budget and mixed-layout assembly retain the existing batching
        policy; only anchor storage/order/reading changes.
        """
        from kumo.pytorch.contract.batch import Graph
        from kumo.pytorch.contract.sample import Field
        from kumo.pytorch.task import _mark_padding, _surface_padding

        source = self._source
        policy = assembler.spec.batching
        fixed = policy.max_tokens is None and not policy.bucket_by_length
        maximum = batch_size if batch_size is not None else policy.max_samples
        size = maximum if fixed else chunk
        neighborhoods = {
            prepared.spec.source.name
            for prepared in assembler._prepared_inputs.values()
            if isinstance(prepared.spec.representation, Graph)
        }
        name = next(iter(neighborhoods)) if fixed and len(neighborhoods) == 1 else None
        plan = None
        projections = {table: [] for table in source._tables}
        if name is not None:
            plan, _ = source._sampling_plan(self._sample.neighborhoods[name])
            for prepared in assembler._prepared_inputs.values():
                if isinstance(prepared.spec.representation, Graph):
                    for table, fields in prepared.spec.representation.node_fields.items():
                        projections[table].extend(field.field for field in fields.fields)
            # Identity is required by the existing graph assembler, even for
            # graph node types without a declared model feature.
            for table, definition in source._tables.items():
                if definition.primary_key is not None:
                    projections[table].insert(0, definition.primary_key)
            for value in self._sample.entity_values.values():
                if isinstance(value, Field) and not value.path:
                    projections[self._sample.entity.table].append(value.column)
            projections = {
                table: list(dict.fromkeys(fields)) for table, fields in projections.items()
            }

        def assemble(requests: Sequence[SampleRequest], runtime: Any, final: bool):
            with runtime_features(source, runtime):
                for batch in assembler.batches(
                    requests,
                    batch_size=batch_size,
                    epoch=epoch,
                    seed=seed,
                    drop_last=drop_last if final else False,
                    output_format=output_format,
                ):
                    yield _surface_padding(batch)

        named = {}
        if len(neighborhoods) > 1:
            for name in sorted(neighborhoods):
                fields = {table: [] for table in source._tables}
                for prepared in assembler._prepared_inputs.values():
                    if (
                        isinstance(prepared.spec.representation, Graph)
                        and prepared.spec.source.name == name
                    ):
                        for table, group in prepared.spec.representation.node_fields.items():
                            fields[table].extend(use.field for use in group.fields)
                for table in fields:
                    fields[table] = list(
                        dict.fromkeys(
                            [
                                *(
                                    ()
                                    if source._tables[table].primary_key is None
                                    else (source._tables[table].primary_key,)
                                ),
                                *fields[table],
                            ]
                        )
                    )
                named[name] = {
                    "plan": source._sampling_plan(self._sample.neighborhoods[name])[0],
                    "projections": fields,
                }
        stages = self._selection(selection)
        if (
            output_format == "model"
            and (fixed or world_size == 1 and not neighborhoods)
            and assembler._source is self._source
        ):
            for value in self._sample.entity_values.values():
                if isinstance(value, Field) and not value.path:
                    projections[self._sample.entity.table].append(value.column)
            for stage in stages:
                yield from self._model_batches(
                    assembler,
                    stage,
                    plan=plan,
                    projections=projections,
                    size=size,
                    maximum=maximum,
                    neighborhoods=named,
                    seed=seed,
                    epoch=epoch,
                    shuffle=shuffle,
                    rank=rank,
                    world_size=world_size,
                    drop_last=policy.drop_last if drop_last is None else drop_last,
                )
            return

        pending: list[SampleRequest] = []
        first = None
        emitted = 0
        stages = self._selection(selection)

        def streams():
            if not shuffle and len(stages) > 1:
                requests = []
                for ordinal, (_, _, request) in enumerate(self._merged_rows(stages)):
                    if ordinal % world_size == rank:
                        requests.append(request)
                        if len(requests) == size:
                            yield None, requests
                            requests = []
                if requests:
                    yield None, requests
                return
            offset = 0
            for stage in stages:
                for runtime in self._iter(
                    stage,
                    batch_size=size,
                    seed=seed,
                    epoch=epoch,
                    shuffle=shuffle,
                    rank=(rank - offset) % world_size,
                    world_size=world_size,
                    plan=plan,
                    projections=projections,
                ):
                    yield runtime, self._requests(stage, runtime)
                offset += stage["count"]

        for runtime, requests in streams():
            emitted += len(requests)
            if requests and first is None:
                first = requests[0]
            if runtime is not None:
                runtime["_anchor_roots"] = tuple(
                    (request.sample_id, request.entity_id, cutoff_to_epoch(request.as_of_time))
                    for request in requests
                )
                runtime["_anchor_neighborhood"] = name
            if not pending and len(requests) == size:
                yield from assemble(requests, runtime, False)
            else:
                pending.extend(requests)
                while len(pending) >= size:
                    yield from assemble(pending[:size], None, False)
                    pending = pending[size:]
        target = -(-self.count(selection) // world_size)
        if emitted < target:
            if first is None:
                first = self.fetch(selection, [0])[0]
            flags = [False] * len(pending) + [True] * (target - emitted)
            pending.extend([first] * (target - emitted))
            pending = _mark_padding(pending, flags)
        if pending:
            yield from assemble(pending, None, True)


@contextmanager
def runtime_features(source: Any, runtime: Any):
    """Scope a prefetched native batch to its one assembler call."""
    saved = getattr(source, "_runtime_batch", None)
    source._runtime_batch = runtime
    try:
        yield
    finally:
        source._runtime_batch = saved
