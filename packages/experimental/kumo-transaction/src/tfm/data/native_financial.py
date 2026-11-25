# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project fixed financial scalar features once per source row, before retrieval."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.table_source import TableSource
from tfm.features.tokenizers.financial import (
    TransactionFieldTokenizer,
    _stable_hash,
    financial_event_values,
)


def financial_source(
    source: TableSource, fields: tuple[str, ...], directory: Path
) -> tuple[TableSource, dict[str, tuple[str, ...]]]:
    """Reuse the fixed feature definitions; native encoders assign delivered IDs.

    This is a bounded scalar projection, never a prepared prediction window or
    model minibatch. Fixed vocabularies do not learn anything from held-out rows.
    """
    vocabulary = TransactionFieldTokenizer(fields=fields)
    tables = {table.name: table for table in source.tables}
    if "events" not in tables or "users" not in tables:
        raise UnsupportedCombinationError(
            "Native financial fields require the transaction source schema"
        )
    events = tables["events"]
    aliases = {field: f"flat_{field}" for field in fields}
    fixed = {
        aliases[field]: tuple(vocabulary.id_to_token[i] for i in range(*bounds))
        for field, bounds in zip(fields, vocabulary.field_vocabulary().field_ranges, strict=True)
    }
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "events.parquet"
    original = pq.ParquetFile(source.files["events"])
    schema = original.schema_arrow
    for column in aliases.values():
        schema = schema.append(pa.field(column, pa.string()))
    with pq.ParquetWriter(output, schema) as writer:
        for block in original.iter_batches(batch_size=4096):
            selected, encoded = [], {column: [] for column in aliases.values()}
            for i, row in enumerate(block.to_pylist()):
                cats = {
                    key: str(row[column])
                    for key, column in events.categorical.items()
                    if row[column] is not None
                }
                if cats.get("source", "transaction") != "transaction":
                    continue
                # Existing flat transaction aliases are scalar feature rules,
                # independent of fitting, model construction and labels.
                from tfm.data.adapters.pragmatiq import (
                    _FLAT_CATEGORICAL_ALIASES,
                    _FLAT_NUMERIC_ALIASES,
                )

                for old, new in _FLAT_CATEGORICAL_ALIASES.items():
                    if old in cats and new not in cats:
                        cats[new] = cats[old]
                    if old in events.text and new not in cats and row[events.text[old]] is not None:
                        cats[new] = str(row[events.text[old]])
                if "channel" in cats and "use_chip" not in cats:
                    cats["use_chip"] = cats["channel"]
                numeric = {
                    key: float(row[column])
                    if row[column] is not None and math.isfinite(float(row[column]))
                    else 0.0
                    for key, column in events.numeric.items()
                }
                for old, new in _FLAT_NUMERIC_ALIASES.items():
                    if old in numeric and new not in numeric:
                        numeric[new] = numeric[old]
                if "user" not in numeric:
                    numeric["user"] = float(
                        int.from_bytes(str(row["owner"]).encode(), "little") % 3000
                    )
                values = financial_event_values(
                    row[events.event_time or "at"],
                    numeric,
                    cats,
                    fields=fields,
                    merchant_bucket=lambda merchant: _stable_hash(merchant, 2000),
                )
                for column, value in zip(aliases.values(), values, strict=True):
                    encoded[column].append(value)
                selected.append(i)
            table = pa.Table.from_batches([block]).take(pa.array(selected, type=pa.int64()))
            for column, values in encoded.items():
                table = table.append_column(column, pa.array(values, type=pa.string()))
            writer.write_table(table)
    projected = replace(
        events, categorical={**events.categorical, **{alias: alias for alias in aliases.values()}}
    )
    return replace(
        source,
        tables=tuple(projected if table.name == "events" else table for table in source.tables),
        files={**source.files, "events": output},
    ), fixed
