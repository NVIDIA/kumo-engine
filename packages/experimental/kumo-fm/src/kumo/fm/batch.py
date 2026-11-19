# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained FM Arrow tables and explicit HTTP encoding."""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import pyarrow as pa

from ._encoding import encode_arrow_table
from ._contract import metadata_for
from .bundle import TYPES
from . import native_json


def _typed_table(table, columns, max_columns):
    if len(columns) > max_columns:
        raise ValueError("Payload table exceeds column budget")
    schema = pa.schema([pa.field(name, TYPES[spec["dtype"]], nullable=spec["nullable"])
                        for name, spec in columns.items()])
    table = table.select(list(columns)).cast(schema)
    for field in schema:
        if not field.nullable and table[field.name].null_count:
            raise ValueError(f"Null in required Arrow field: {field.name}")
    return table


class PayloadBatch:
    """One sampled request; Arrow tables stay valid after loader closure."""

    def __init__(self, workload, splits):
        metadata = metadata_for(workload)
        self._metadata = deepcopy(metadata)
        self._budgets = dict(workload.budgets)
        self._splits = {}
        logical_bytes = 0
        for split, (seeds, related) in splits.items():
            columns = dict(metadata["schema"]["instance_table"]["columns"])
            if split == "predict":
                columns.pop("__fm_target")
            instance = pa.table({
                name: pa.array([str(seed["target"]) if name == "__fm_target"
                                and workload.kind != "regression" else seed[{
                                    "instance_id": "instance_id",
                                    "__fm_entity_key": "entity_key",
                                    "__fm_anchor": "anchor_time",
                                    "__fm_target": "target",
                                }[name]] for seed in seeds], type=TYPES[spec["dtype"]])
                for name, spec in columns.items()
            })
            instance = _typed_table(instance, columns, self._budgets["max_columns"])
            if len(instance) > self._budgets["max_rows_per_table"]:
                raise ValueError("Instance table exceeds row budget")
            normalized = {}
            for name, spec in metadata["schema"]["related_tables"].items():
                table = _typed_table(related[name], spec["columns"], self._budgets["max_columns"])
                if len(table) > self._budgets["max_rows_per_table"]:
                    raise ValueError("Related table exceeds row budget")
                normalized[name] = table
                logical_bytes += table.nbytes
            logical_bytes += instance.nbytes
            self._splits[split] = (instance, MappingProxyType(normalized))
        if logical_bytes > self._budgets["max_decoded_arrow_bytes"]:
            raise ValueError("Decoded Arrow budget exceeded")
        self._logical_bytes = logical_bytes

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def logical_arrow_bytes(self):
        return self._logical_bytes

    @property
    def splits(self):
        return tuple(self._splits)

    def tables(self, split):
        instance, related = self._splits[split]
        return MappingProxyType({"instance_table": instance, "related_tables": related})

    def as_arrow_request(self):
        result = self.metadata
        for split, (instance, related) in self._splits.items():
            result[split] = {"instance_table": instance, "related_tables": dict(related)}
        return result

    def encode(self, encoding="arrays"):
        if encoding not in ("arrays", "arrow_ipc"):
            raise ValueError("Unknown encoding")
        payload = self.metadata
        for split, (instance, related) in self._splits.items():
            instance_columns = dict(payload["schema"]["instance_table"]["columns"])
            if split == "predict":
                instance_columns.pop("__fm_target")
            encoded_instance, _ = encode_arrow_table(
                instance, instance_columns, encoding, self._budgets, native_arrays=encoding == "arrays")
            encoded_related = {}
            for name, table in related.items():
                encoded_related[name], _ = encode_arrow_table(
                    table, payload["schema"]["related_tables"][name]["columns"], encoding,
                    self._budgets, native_arrays=encoding == "arrays")
            payload[split] = {"instance_table": encoded_instance, "related_tables": encoded_related}
        if encoding == "arrays":
            return native_json.dumps(payload, self._budgets["max_payload_bytes"])
        import json
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        if len(body) > self._budgets["max_payload_bytes"]:
            raise ValueError("Payload exceeds byte budget; reduce the workload batch")
        return body
