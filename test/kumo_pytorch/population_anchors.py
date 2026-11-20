# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project declared anchor columns into native tables without sampling full runs."""

import hashlib
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import kumo.pytorch.api as r

from .aml_inputs import SECOND, SPLITS
from .bindings import profile as binding_profile
from .bindings import task_definition
from .targets import target_rules, target_value


def add_identity(digest, sample_id):
    digest.update(json.dumps(sample_id, ensure_ascii=False).encode() + b"\n")


def project_population(
    path,
    destination,
    identities,
    *,
    graph,
    native,
    limit=None,
    splits=SPLITS,
    workload="account-involvement-next24h",
    profile=None,
    rules=None,
):
    """Keep original order, labels and splits; v2 shifts only graph execution clocks."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive or None for all eligible anchors")
    if not splits or len(set(splits)) != len(splits) or not set(splits) <= set(SPLITS):
        raise ValueError("Select distinct known anchor splits")
    profile = profile or {
        "sample_id": "sample_id",
        "entity_id": "entity_id",
        "time": "as_of_time",
        "split": "split" if native else "forecast_split",
        "time_unit": "ns",
        "cadence_seconds": 86400,
        "execution_offset_seconds": -1 if graph else 0,
        **({"execution_time": "graph_as_of_time"} if graph else {}),
    }
    column = profile["split"]
    rules = target_rules(workload) if rules is None else rules
    sample_column, entity_column, time_column = (
        profile[k] for k in ("sample_id", "entity_id", "time")
    )
    execution_column = profile.get("execution_time", time_column)
    offset_seconds = profile.get("execution_offset_seconds", 0)
    clocks = tuple(dict.fromkeys((time_column, execution_column)))
    columns = [sample_column, entity_column, *clocks, column]
    columns.extend(rule["column"] for rule in rules.values())
    columns.extend(
        c for rule in rules.values() for c in rule.get("validity_columns", ())
    )
    columns = list(dict.fromkeys(columns))
    source = pq.ParquetFile(path)
    for name in clocks:
        if source.schema_arrow.field(name).type != pa.timestamp(profile["time_unit"]):
            raise ValueError(
                f"Expected declared anchor timestamp unit {profile['time_unit']}: {name}"
            )
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("entity_id", pa.int64()),
            ("as_of_time", pa.int64()),
            ("split", pa.string()),
            *(
                (name, pa.string() if rule["kind"] == "category" else pa.float32())
                for name, rule in rules.items()
            ),
            *(
                ("original_" + name, pa.int64())
                for name, rule in rules.items()
                if rule["kind"] == "count"
            ),
            ("prediction_time_ns", pa.int64()),
            ("registered_entity_id", pa.string()),
            ("anchor_row_position", pa.int64()),
        ]
    )
    counts = dict.fromkeys(splits, 0)
    source_counts = dict.fromkeys(splits, 0)
    hashes = {split: hashlib.sha256() for split in splits}
    seen, offset, train_cutoff = set(), 0, None
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in source.iter_batches(columns=columns, batch_size=65536):
            for item in pc.value_counts(batch.column(column)).to_pylist():
                if item["values"] in source_counts:
                    source_counts[item["values"]] += item["counts"]
            for name in clocks:
                batch = batch.set_column(
                    batch.schema.get_field_index(name),
                    name,
                    pc.cast(
                        pc.cast(batch.column(name), pa.timestamp("ns")), pa.int64()
                    ),
                )
            records = []
            for position, row in enumerate(batch.to_pylist(), offset):
                split = row[column]
                if split is None:
                    continue
                if split not in SPLITS:
                    raise ValueError(f"Unknown anchor split: {split}")
                if split not in counts:
                    continue
                if limit is not None and counts[split] >= limit:
                    continue
                sample_id, entity = row[sample_column], row[entity_column]
                if not sample_id or sample_id in seen:
                    raise ValueError("Missing or duplicate selected sample identity")
                if not entity or entity not in identities:
                    raise ValueError("Selected entity is absent from mapped source")
                stamp = row[time_column]
                cadence = profile.get("cadence_seconds")
                if stamp is None or (
                    cadence is not None and stamp % (cadence * SECOND)
                ):
                    raise ValueError(
                        "Prediction time violates the declared anchor cadence"
                    )
                if stamp % SECOND:
                    raise ValueError(
                        "Native projection requires whole-second prediction clocks"
                    )
                clock = stamp + offset_seconds * SECOND
                if execution_column != time_column and row[execution_column] != clock:
                    raise ValueError(
                        "Graph cutoff must equal prediction time plus the declared execution offset"
                    )
                for rule in rules.values():
                    if any(
                        type(row[flag]) is not bool
                        for flag in rule.get("validity_columns", ())
                    ):
                        raise ValueError(
                            f"Target validity must be non-null boolean: {rule['column']}"
                        )
                records.append(
                    {
                        "sample_id": sample_id,
                        "entity_id": identities[entity],
                        "as_of_time": clock // SECOND,
                        "split": split,
                        **{
                            name: target_value(row[rule["column"]], rule)
                            if all(
                                row[c] is True for c in rule.get("validity_columns", ())
                            )
                            else None
                            for name, rule in rules.items()
                        },
                        **{
                            "original_" + name: row[rule["column"]]
                            for name, rule in rules.items()
                            if rule["kind"] == "count"
                        },
                        "prediction_time_ns": stamp,
                        "registered_entity_id": str(entity),
                        "anchor_row_position": position,
                    }
                )
                seen.add(sample_id)
                counts[split] += 1
                add_identity(hashes[split], sample_id)
                if split == "train":
                    cutoff = stamp // SECOND - 1
                    train_cutoff = (
                        cutoff if train_cutoff is None else max(train_cutoff, cutoff)
                    )
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
            offset += batch.num_rows
            if limit is not None and all(n == limit for n in counts.values()):
                break
    if any(n == 0 or (limit is not None and n != limit) for n in counts.values()):
        raise ValueError("Insufficient required anchors in selected splits")
    if limit is None and (
        offset != source.metadata.num_rows or counts != source_counts
    ):
        raise ValueError("Full projection did not retain every eligible source anchor")
    manifest = {
        "mode": "full" if limit is None else "smoke",
        "split_column": column,
        "counts": counts,
        "ordered_sample_id_sha256": {s: h.hexdigest() for s, h in hashes.items()},
        "encoder_cutoff_seconds": train_cutoff,
        "graph": graph,
        "native": native,
        "source_rows": source.metadata.num_rows,
        "scanned_rows": offset,
        "scanned_source_split_counts": source_counts,
    }
    return r.AnchorTable(
        r.Parquet(str(destination)),
        "entity_id",
        "as_of_time",
        sample_id="sample_id",
        split="split",
        targets={} if native else {name: name for name in rules},
        metadata=("prediction_time_ns", "registered_entity_id", "anchor_row_position"),
    ), manifest


def projected_rows(path, split):
    """Bounded-memory, independent delivery reference for one original split."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        mask = pc.equal(batch.column("split"), split)
        yield from batch.filter(mask).to_pylist()


def validate_binding(binding):
    family = binding["representation"]
    graph = family in {"graphsage", "combined"}
    anchor, inputs = (
        binding_profile(binding, "anchor"),
        binding_profile(binding, "input"),
    )
    task = task_definition(binding)
    required_family = {"masked_values": "pragma", "next_token": "nuformer"}
    if task["kind"] not in {"stored", *required_family} or (
        task["kind"] in required_family and family != required_family[task["kind"]]
    ):
        raise ValueError(f"Task {task['kind']} is incompatible with model {family}")
    native = task["kind"] != "stored"
    rules = task["targets"]
    if anchor.get("execution_offset_seconds", 0) != (-1 if graph else 0):
        raise ValueError(
            "Strict history profiles require graph cutoff T-1s and sequence cutoff T"
        )
    if (
        binding["selector"]["anchor_version"] != anchor["version"]
        or binding["split_column"] != anchor["split"]
        or binding["input_variant"] != inputs["variant"]
    ):
        raise ValueError("Binding has incompatible anchors, splits or input variant")
    if set(binding["targets"]) != set(rules):
        raise ValueError("Binding targets differ from its registered task")
    for name, rule in rules.items():
        expected = binding["targets"][name]
        if (
            expected["column"] != rule["column"]
            or expected["dtype"] != rule["dtype"]
            or expected["all_true_columns"] != rule.get("validity_columns", [])
            or not expected["non_null_required"]
        ):
            raise ValueError(f"Target meaning changed: {name}")
    return family, graph, native, rules


def project_binding(path, destination, identities, binding, **options):
    """Reuse the native Parquet projection; never build a request population."""
    _, graph, native, rules = validate_binding(binding)
    schema = pq.ParquetFile(path).schema_arrow
    types = {
        "BOOLEAN": pa.bool_(),
        "INTEGER": pa.int32(),
        "BIGINT": pa.int64(),
        "DECIMAL(38,10)": pa.decimal128(38, 10),
        "DOUBLE": pa.float64(),
        "VARCHAR": pa.string(),
    }
    for rule in rules.values():
        column, expected = rule["column"], types[rule["dtype"]]
        if column not in schema.names or schema.field(column).type != expected:
            raise ValueError(
                f"Expected registered target dtype {expected} for {column}"
            )
        for flag in rule.get("validity_columns", ()):
            if flag not in schema.names or schema.field(flag).type != pa.bool_():
                raise ValueError(f"Expected boolean target validity: {flag}")
    anchors, report = project_population(
        path,
        destination,
        identities,
        graph=graph,
        native=native,
        workload=binding["selector"]["workload_id"],
        profile=binding_profile(binding, "anchor"),
        rules=rules,
        **options,
    )
    return anchors, {
        **report,
        "binding_id": binding["binding_id"],
        "ref": binding["ref"],
    }
