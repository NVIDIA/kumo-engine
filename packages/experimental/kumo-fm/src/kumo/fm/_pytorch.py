# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public kumo.pytorch columnar adapter backed by DiskGraph."""
import hashlib
import importlib.metadata
import json
from contextlib import ExitStack
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._assembly import GraphAssembler
from ._contract import schema_for
from ._phase import phase
from ._validation import validate_contract
from .bundle import TYPES, quote
from .native_selection import inclusive_second


def compatibility_issues(workload):
    """Report semantic gaps without importing optional native dependencies."""
    issues = []
    if workload.selection != "native_multihop":
        issues.append("kumo-pytorch currently requires native_multihop selection")
    for number, layer in enumerate(workload.layers):
        active = [step for step in layer if step["fanout"] != 0]
        if not active:
            issues.append(f"layer {number}: kumo-pytorch cannot express an empty graph layer")
        for step in active:
            _, dest = workload.bundle.endpoint(step)
            if not workload.bundle.tables[dest].get("time_column"):
                issues.append(f"{step['relation']} {step['direction']}: uniform sampling into "
                              f"non-temporal table {dest} is unsupported by kumo-pytorch")
    return list(dict.fromkeys(issues))


def contract_for(workload):
    issues = compatibility_issues(workload)
    if issues:
        raise ValueError("Cannot preserve FM sampling with kumo-pytorch:\n" + "\n".join(issues))
    from kumo.pytorch import api
    bundle = workload.bundle
    schema = api.DataSchema(
        tables=tuple(api.TableSpec(name=name, primary_key=table["primary_key"],
                                   columns={c: api.DType.from_arrow(TYPES[s["dtype"]])
                                            for c, s in table["columns"].items()},
                                   time_column=table.get("time_column") or api.NON_TEMPORAL)
                     for name, table in bundle.tables.items() if name in workload.table_names),
        relations=tuple(api.RelationSpec(name=name, left_table=rel["child"],
                        left_key=rel["foreign_key"], right_table=rel["parent"],
                        right_key=bundle.tables[rel["parent"]]["primary_key"])
                        for name, rel in bundle.relations.items() if name in workload.relation_names))
    layers = []
    for layer in workload.layers:
        samples = []
        for step in layer:
            if step["fanout"] == 0:
                continue
            _, dest = bundle.endpoint(step)
            total = bundle.tables[dest]["rows"]
            count = total if step["fanout"] is None else min(total, step["fanout"])
            if not count:
                raise ValueError("kumo-pytorch migration cannot yet preserve empty destinations")
            samples.append(api.GraphSample(relation=step["relation"],
                           direction=step["direction"], fanout=min(count, 1)
                           if step["direction"] == "forward" else count, strategy="recent"))
        layers.append(api.GraphLayer(samples=tuple(samples)))
    sample = api.SampleSpec(entity=api.EntitySpec(table=workload.entity),
                            neighborhoods={"fm": api.NeighborhoodSpec(layers=tuple(layers))})
    fields = {name: api.Fields(fields=tuple(api.Encode(field=column, encoding=api.Raw())
              for column in bundle.tables[name]["columns"])) for name in workload.table_names}
    batch = api.BatchSpec(inputs={"graph": api.Input(source=api.NeighborhoodSource(name="fm"),
                                                     representation=api.Graph(node_fields=fields), layout=None)},
                          batching=api.Batching(max_samples=workload.budgets["max_instances"]))
    return api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)


class KumoPytorchBackend:
    """Use prepare → materialize(AnchorTable) → Task.batches, without RDL.

    Inherit only FM's retained columnar graph-to-payload assembly. No direct
    SamplerEngine calls or private kumo.pytorch implementation hooks are used.
    """

    merged_edge_output = True

    def __init__(self, workload, index, *, threads=2):
        self.output_format = "columnar"
        contract = contract_for(workload)
        from kumo.pytorch import api
        validate_contract(workload)
        self.bundle, self.index = workload.bundle, Path(index).resolve()
        if self.index == self.bundle.root or self.bundle.root in self.index.parents:
            raise ValueError("Native index must be outside the immutable bundle")
        self.node_types = list(self.bundle.tables)
        self.assembler = GraphAssembler(self.bundle)
        self.metrics = {"kumo_pytorch_batches": 0, "multihop_selected_nodes": 0,
                        "multihop_selected_edges": 0, "native_candidate_rows": 0,
                        "api": "kumo.pytorch.Task.batches", "output_format": "inspectable"}
        self.derived_source_bytes = 0
        self._stack, self._tasks = ExitStack(), {}
        self._workload = workload
        with duckdb.connect(config={"threads": threads}) as con:
            for name in workload.table_names:
                table = self.bundle.tables[name]
                con.read_parquet(self.bundle.paths(name)).create_view("source")
                for column, spec in table["columns"].items():
                    if spec["dtype"] == "timestamp[us]":
                        fractional = con.execute(f"SELECT count(*) FROM source WHERE "
                                                 f"epoch_us({quote(column)}) % 1000000 != 0").fetchall()[0][0]
                        if fractional:
                            raise ValueError(f"kumo-pytorch requires whole-second timestamp features: {name}.{column}")
                if table.get("time_column"):
                    nulls = con.execute(f"SELECT count(*) FROM source WHERE "
                                        f"{quote(table['time_column'])} IS NULL").fetchall()[0][0]
                    if nulls:
                        raise ValueError(f"kumo-pytorch would drop NULL-clock source rows: {name}")
        marker = self.index / "fm_snapshot.json"
        if self.index.exists() and any(self.index.iterdir()) and not marker.exists():
            raise FileExistsError("Refusing an incomplete or unrelated kumo-pytorch index")
        self.index.mkdir(parents=True, exist_ok=True)
        provenance = {"snapshot_id": self.bundle.snapshot_id, "adapter_version": 1,
                      "contract": contract.fingerprint}
        provenance["dependencies"] = {
            name: {"version": (dist := importlib.metadata.distribution(name)).version,
                   "source": json.loads(dist.read_text("direct_url.json") or "null")}
            for name in ("diskgraph", "kumo-pytorch")}
        if marker.exists() and json.loads(marker.read_text()) != provenance:
            raise ValueError("kumo-pytorch cache belongs to another snapshot/contract")
        self.preparation_mode = "cached_index_open" if marker.exists() else "index_build"
        try:
            # Exact file inventories: do not glob sibling files into immutable data.
            source_root = self.index / "sources"
            sources = {}
            for name in workload.table_names:
                directory = source_root / name
                directory.mkdir(parents=True, exist_ok=True)
                for i, path in enumerate(self.bundle.paths(name)):
                    link = directory / f"{i:08d}.parquet"
                    if not link.exists():
                        link.symlink_to(path)
                sources[name] = api.Parquet(path=str(directory))
            with phase("kumo_pytorch_prepare", "preparation"):
                self.plan = self._stack.enter_context(api.prepare(
                    contract=contract, dataset=api.Dataset(tables=sources),
                    engine=api.DiskGraphEngine(index=self.index / "engine", num_threads=threads)))
            metadata = json.loads((self.index / "engine" / "index_metadata.json").read_text())
            counts = {t["name"]: t["max_id"] + 1 for t in metadata["node_types"]}
            for name in workload.table_names:
                if counts.get(name, 0) != self.bundle.tables[name]["rows"]:
                    raise ValueError(f"kumo-pytorch ingestion lost source rows: {name}")
            marker.write_text(json.dumps(provenance, indent=2) + "\n")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _key(seeds):
        return tuple((s["entity_key"], s["anchor_time"]) for s in seeds)

    def prepare_batch(self, seeds):
        """Materialize this exact ordered request outside request timing."""
        from kumo.pytorch import api
        key = self._key(seeds)
        if key in self._tasks:
            return
        token = hashlib.sha256(repr(key).encode()).hexdigest()
        path = self.index / f"anchors-{token}.parquet"
        pk = self.bundle.tables[self._workload.entity]["primary_key"]
        dtype = TYPES[self.bundle.tables[self._workload.entity]["columns"][pk]["dtype"]]
        # Graph neighborhoods use the native inclusive whole-second cutoff.
        anchors = pa.table({"entity": pa.array([s["entity_key"] for s in seeds], dtype),
                            "sample": pa.array(range(len(seeds)), pa.int64()),
                            "at": pa.array([inclusive_second(s["anchor_time"]) for s in seeds],
                                           pa.timestamp("s", tz="UTC"))})
        pq.write_table(anchors, path)
        with phase("kumo_pytorch_materialize", "preparation"):
            task = self._stack.enter_context(self.plan.materialize(api.AnchorTable(
                source=api.Parquet(path=str(path)), entity_id="entity", sample_id="sample",
                as_of_time="at")))
        if task.count() != len(seeds):
            raise ValueError("kumo-pytorch materialization lost anchors")
        self._tasks[key] = task

    def extract(self, workload, seeds):
        if not seeds:
            return {name: pa.table({column: pa.array([], type=TYPES[spec['dtype']])
                    for column, spec in table['columns'].items()})
                    for name, table in schema_for(workload)['related_tables'].items()}
        if workload.id != self._workload.id:
            raise ValueError("kumo-pytorch backend is bound to its prepared workload")
        key = self._key(seeds)
        if key not in self._tasks:
            raise ValueError("Call prepare_batch(seeds) before measuring a new request")
        if (len(seeds) > workload.budgets["max_instances"] or
                len(seeds) > workload.budgets["max_rows_per_table"]):
            raise ValueError("Native-multihop root budget exceeded")
        with phase("kumo_pytorch_batches", "retrieval_selection"):
            batches = iter(self._tasks[key].batches(batch_size=len(seeds), shuffle=False,
                                                    output_format=self.output_format))
            batch = next(batches)
            if next(batches, None) is not None:
                raise ValueError("Expected exactly one native anchor batch")
        self.metrics["kumo_pytorch_batches"] += 1
        with phase("kumo_pytorch_graph_adaptation", "graph_assembly"):
            return self._adapt_columnar_batch(workload, seeds, batch)

    def _adapt_columnar_batch(self, workload, seeds, batch):
        """Consume public Arrow columns; only reorder when anchor order requires it."""
        from kumo.pytorch.api import ColumnarGraph
        graph = batch.inputs["graph"]
        if not isinstance(graph, ColumnarGraph) or graph.root_type != workload.entity:
            raise ValueError("Expected a columnar graph rooted at the workload entity")
        permutation = np.asarray(batch.metadata["sample_ids"], dtype=np.int64)
        if not np.array_equal(np.sort(permutation), np.arange(len(seeds))):
            raise ValueError("kumo-pytorch anchor IDs are not a permutation of the request")
        same_order = np.array_equal(permutation, np.arange(len(seeds)))
        nodes, features, edges, directions, inverse = {}, {}, {}, {}, {}
        for name, node in graph.nodes.items():
            type_id = self.node_types.index(name)
            positions = node.owners.to_numpy(zero_copy_only=False)
            if (node.owners.null_count or np.any(positions >= len(seeds)) or
                    np.any(positions < 0)):
                raise ValueError("kumo-pytorch node owner is out of bounds")
            table = node.features
            if len(table) != len(node.ids) or len(table) != len(positions):
                raise ValueError("kumo-pytorch columnar node/feature alignment mismatch")
            owners = node.owners.cast(pa.uint64()) if same_order else pa.array(permutation[positions], pa.uint64())
            order = None
            if not same_order and name == workload.entity:
                order = np.arange(len(positions))
                order[:len(seeds)] = np.argsort(permutation[positions[:len(seeds)]])
                undo = np.empty(len(order), dtype=np.int64)
                undo[order] = np.arange(len(order))
                inverse[name] = undo
                owners, table = owners.take(pa.array(order)), table.take(pa.array(order))
            nodes[type_id] = [pa.record_batch(
                [owners, pa.array(np.arange(len(positions)), pa.uint64())], names=["id_a", "id_b"])]
            features[type_id] = table.to_batches() or [pa.RecordBatch.from_arrays(
                [table[c].combine_chunks() for c in table.column_names], schema=table.schema)]
        for rel_id, (name, edge) in enumerate(graph.edges.items()):
            rel = self.bundle.relations[name]
            orientation = (edge.source_type, edge.destination_type)
            if orientation not in ((rel["child"], rel["parent"]),
                                   (rel["parent"], rel["child"])):
                raise ValueError("kumo-pytorch relation orientation differs from the schema")
            native_types = orientation
            columns = []
            for column, table_name in zip(("source", "destination"), native_types, strict=True):
                values = edge.index[column]
                if table_name in inverse:
                    indices = values.to_numpy(zero_copy_only=False)
                    if values.null_count or np.any(indices < 0) or np.any(indices >= len(inverse[table_name])):
                        raise ValueError("kumo-pytorch edge endpoint is out of bounds")
                    values = pa.chunked_array([pa.array(inverse[table_name][indices], pa.uint64())])
                columns.append(values.cast(pa.uint64()))
            if orientation != (rel["child"], rel["parent"]):
                columns.reverse()
            index = pa.Table.from_arrays(columns, names=["id_a", "id_b"])
            directions[rel_id] = name, 'forward', rel["child"], rel["parent"]
            edges[rel_id] = index.to_batches() or [pa.RecordBatch.from_arrays(
                [index[c].combine_chunks() for c in index.column_names], schema=index.schema)]
        count = sum(len(b) for group in (*nodes.values(), *edges.values()) for b in group)
        if count > workload.budgets["max_candidate_rows"]:
            raise ValueError("Native-multihop node-plus-edge budget exceeded")
        self.metrics["native_candidate_rows"] += count
        return self.assembler._adapt(workload, seeds, np.arange(len(seeds), dtype=np.uint64),
                                     nodes, features, edges, directions)

    def close(self):
        self._stack.close()
