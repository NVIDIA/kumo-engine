# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream complete registered Synthetic and MBD populations through the package projections."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from e2e.datasets import bind_tables, mbd, synthetic


def _blocks(paths, columns=None):
    offset = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=262144):
            yield pa.Table.from_batches([batch]), np.arange(offset, offset + len(batch))
            offset += len(batch)


def _write(path, tables):
    with ExitStack() as stack:
        writer = None
        for table in tables:
            if writer is None:
                writer = stack.enter_context(
                    pq.ParquetWriter(path, table.schema, compression="zstd")
                )
            writer.write_table(table)
        if writer is None:
            raise ValueError(f"No source batches for {path.name}")


def _paths(files, table):
    return [path for name, path in sorted(files.items()) if name.startswith(table + "/")]


def _synthetic(work, files, kind):
    owner = "account" if kind == "account" else "customer"
    root_key = owner + "_id"
    accounts = pq.read_table(
        _paths(files, "accounts"), columns=["account_id", "customer_id", "opened_at"]
    )
    tables = {"accounts": accounts}
    if kind == "account":
        tables["accounts"], tables["fanout"] = synthetic._rank_accounts(accounts)
        entities, clock = accounts, "opened_at"
    else:
        clock = "onboarded_at"
        entities = pq.read_table(_paths(files, "customers"), columns=[root_key, clock])
    roots = entities[root_key].combine_chunks()
    if roots.null_count or len(roots.unique()) != len(roots):
        raise ValueError(f"Invalid Synthetic {root_key} identity")
    pq.write_table(
        pa.table({"root_id": roots, "available_at": entities[clock].cast(pa.timestamp("s"))}),
        work / "roots.parquet",
    )

    def anchors():
        for rows, positions in _blocks([files["anchors"]], synthetic._anchor_columns(kind)):
            if not rows[root_key].cast(pa.string()).equals(rows["entity_id"]):
                raise ValueError("Registered Synthetic root differs from its anchor")
            if pc.index_in(rows[root_key], value_set=roots).null_count:
                raise ValueError("Unknown Synthetic anchor root")
            if kind == "account":
                indices = pc.index_in(rows[root_key], value_set=accounts["account_id"])
                if not accounts["customer_id"].take(indices).equals(rows["customer_id"]):
                    raise ValueError("Registered Synthetic account has the wrong owner")
            yield synthetic._project_anchors(
                rows.append_column("anchor_row_position", pa.array(positions)), kind
            )

    _write(work / "anchors.parquet", anchors())
    capacity = 0
    for family, (key, features) in synthetic._FAMILIES.items():
        owners = ["src_account_id", "dst_account_id"] if family == "transfers" else ["customer_id"]
        if family in ("card_txns", "disputes"):
            owners.append("account_id")
        counts = np.zeros(len(roots), dtype=np.int64)

        def events():
            for raw, _ in _blocks(_paths(files, family), [key, "event_at", *owners, *features]):
                rows = synthetic._events(family, raw, tables, None, kind)
                frequencies = pc.value_counts(rows["root_id"])
                idx = pc.index_in(frequencies.field("values"), value_set=roots)
                if idx.null_count:
                    raise ValueError(f"Unknown Synthetic root in {family}")
                counts[idx.to_numpy()] += frequencies.field("counts").to_numpy()
                yield rows

        _write(work / f"{family}.parquet", events())
        capacity += max(1, counts.max())
    targets = ("label",) if kind == "ltv" else ()
    return synthetic._contract(int(capacity), kind), root_key, targets


def _mbd(work, files, binding):
    categories = binding == "M03"
    members = []
    for name, path in sorted(files.items()):
        if name.startswith("client_split/"):
            clients = pq.ParquetFile(path).read(columns=["client_id"])
            fold = int(Path(name).parent.name.removeprefix("fold="))
            if fold not in range(5):
                raise ValueError(f"Unexpected membership fold in {name}")
            members.append(clients.append_column("fold", pa.repeat(fold, len(clients))))
    membership = pa.concat_tables(members).sort_by("client_id")
    names = membership["client_id"].combine_chunks()
    if names.null_count or len(names.unique()) != len(names):
        raise ValueError("Invalid MBD client membership")
    ranks = np.arange(len(names), dtype=np.int64)
    pq.write_table(pa.table({"root_id": ranks}), work / "roots.parquet")
    counts = np.zeros_like(ranks)

    def root_ids(rows, column, fold_column):
        indices = pc.index_in(rows[column], value_set=names)
        if (
            indices.null_count
            or not pc.all(pc.equal(membership["fold"].take(indices), rows[fold_column])).as_py()
        ):
            raise ValueError(f"MBD {column} disagrees with its source fold")
        return indices.cast(pa.int64())

    def events():
        for rows, positions in _blocks(
            _paths(files, "detail/trx"),
            ["client_id", "fold", "event_time", "amount", *mbd.CATEGORIES],
        ):
            ids = root_ids(rows, "client_id", "fold")
            frequencies = pc.value_counts(ids)
            counts[frequencies.field("values").to_numpy()] += frequencies.field("counts").to_numpy()
            yield mbd._project_events(
                rows.append_column("event_id", pa.array(positions)), ids, categories
            )

    def anchors():
        for rows, positions in _blocks([files["anchors"]]):
            ids = root_ids(rows, "entity_id", "source_fold")
            expected = pc.take(
                pa.array(["train", "train", "train", "valid", "test"]), rows["source_fold"]
            )
            if not pc.all(pc.fill_null(pc.equal(expected, rows["split"]), False)).as_py():
                raise ValueError("MBD source fold disagrees with registered split")
            yield mbd._project_anchors(
                rows.append_column("anchor_row_position", pa.array(positions)), ids, categories
            )

    _write(work / "transactions.parquet", events())
    _write(work / "anchors.parquet", anchors())
    return (
        mbd._contract(int(max(1, counts.max())), categories),
        "root_id",
        mbd._HEADS if categories else ("count",),
    )


def prepare_full(binding, files, directory):
    """Bind full-source tables after every verified shard and anchor has been projected."""
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent) as tmp:
        work = Path(tmp)
        if binding in ("M03", "M04"):
            spec, root, targets = _mbd(work, files, binding)
        else:
            kind = {"S01": "account", "S02": "customer", "S04": "ltv"}[binding]
            spec, root, targets = _synthetic(work, files, kind)
        work.rename(directory)
    return bind_tables(directory, spec, root, targets)
