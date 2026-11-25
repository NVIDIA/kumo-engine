# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert official interaction archives into stable chronological actor/item/time arrays.

Usage: python -m tests.e2e.omnisage_event_sources movielens ml-100k.zip movielens.npz
Retail needs openpyxl. The other readers use NumPy and the Python standard library.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import numpy as np

SOURCES = {
    "movielens": {
        "url": "https://grouplens.org/datasets/movielens/100k/",
        "archive_url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "citation": "Harper & Konstan (2015). The MovieLens Datasets: History and Context. "
        "https://doi.org/10.1145/2827872",
        "policy": "One user/movie interaction per published u.data rating, including low ratings. "
        "No rating threshold or deduplication. Supplied Unix seconds.",
    },
    "collegemsg": {
        "url": "https://snap.stanford.edu/data/CollegeMsg.html",
        "archive_url": "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
        "citation": "Panzarasa, Opsahl & Carley (2009). Patterns and dynamics of users' behavior "
        "and interaction: Network analysis of an online community. https://doi.org/10.1002/asi.21015",
        "policy": "One directed sender/recipient event for every message. Repeat messages and "
        "equal timestamps are retained. Supplied Unix seconds.",
    },
    "retail": {
        "url": "https://doi.org/10.24432/C5BW33",
        "archive_url": "https://archive.ics.uci.edu/static/public/352/online+retail.zip",
        "citation": "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5BW33",
        "policy": "One event per paid, positive-quantity, uncancelled invoice line with an "
        "identified customer. Exclusions are counted once in the reported order. Retain all "
        "paid stock codes, including paid services, and repetitions. Quantity does not weight "
        "the event. Interpret naive invoice times as Europe/London, taking the earlier "
        "occurrence of an ambiguous local time.",
    },
}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _retail(path):
    from openpyxl import load_workbook

    consumer_ids, item_ids = {}, {}
    excluded = dict.fromkeys(
        ("missing_customer", "cancelled_invoice", "nonpositive_quantity", "nonpositive_price"), 0
    )
    service_codes = Counter()
    source_rows = 0
    actors, items, ticks = [], [], []
    ambiguous = 0
    tz = ZoneInfo("Europe/London")
    with ZipFile(path) as archive:
        workbook = load_workbook(
            io.BytesIO(archive.read("Online Retail.xlsx")), read_only=True, data_only=True
        )
    try:
        rows = workbook.active.values
        header = next(rows)
        indexes = [
            header.index(name)
            for name in (
                "InvoiceNo",
                "CustomerID",
                "StockCode",
                "InvoiceDate",
                "Quantity",
                "UnitPrice",
            )
        ]
        for row in rows:
            invoice, customer, stock, at, qty, price = [row[i] for i in indexes]
            source_rows += 1
            if invoice is None or stock is None or not isinstance(at, datetime):
                raise ValueError(
                    f"missing invoice, item or timestamp at source row {source_rows + 1}"
                )
            if customer is None:
                excluded["missing_customer"] += 1
                continue
            if str(invoice).upper().startswith("C"):
                excluded["cancelled_invoice"] += 1
                continue
            if qty <= 0:
                excluded["nonpositive_quantity"] += 1
                continue
            if price <= 0:
                excluded["nonpositive_price"] += 1
                continue
            if not re.fullmatch(r"[0-9]{5}[a-z]*", str(stock), re.IGNORECASE):
                service_codes[str(stock)] += 1
            actors.append(consumer_ids.setdefault(str(customer), len(consumer_ids)))
            items.append(item_ids.setdefault(str(stock), len(item_ids)))
            local = at.replace(tzinfo=tz)
            ticks.append(int(local.timestamp()))
            ambiguous += local.utcoffset() != local.replace(fold=1).utcoffset()
    finally:
        workbook.close()
    return (
        actors,
        items,
        ticks,
        {
            "source_rows": source_rows,
            "exclusions_in_order": excluded,
            "retained_nonstandard_stock_codes": dict(sorted(service_codes.items())),
            "ambiguous_local_times": ambiguous,
        },
    )


def normalize(dataset, archive, output):
    if dataset == "movielens":
        with ZipFile(archive) as source, source.open("ml-100k/u.data") as data:
            users, items, ratings, ticks = np.loadtxt(data, dtype=np.int64, unpack=True)
        extra = {"rating_counts": np.bincount(ratings, minlength=6)[1:].tolist()}
    elif dataset == "collegemsg":
        with gzip.open(archive, "rt") as source:
            users, items, ticks = np.loadtxt(source, dtype=np.int64, unpack=True)
        extra = {"unique_people_across_roles": int(len(np.union1d(users, items)))}
    elif dataset == "retail":
        users, items, ticks, extra = _retail(archive)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    # The benchmark names actor and item roles independently.
    actor_ids, actor = np.unique(users, return_inverse=True)
    item_ids, item = np.unique(items, return_inverse=True)
    time = np.asarray(ticks, dtype=np.int64)
    order = np.argsort(time, kind="stable")
    with output.open("wb") as file:
        np.savez_compressed(
            file,
            actor=actor[order].astype(np.int64),
            item=item[order].astype(np.int64),
            time=time[order],
        )
    metadata = {
        **SOURCES[dataset],
        "id": dataset,
        "source_filename": archive.name,
        "source_sha256": sha256(archive),
        "npz_sha256": sha256(output),
        "events": len(time),
        "actors": len(actor_ids),
        "items": len(item_ids),
        "ordering": "timestamp ascending, original file order within a tie",
        "id_mapping": "independent contiguous integer IDs for actor and item roles",
        "start_utc": datetime.fromtimestamp(int(time.min()), UTC).isoformat(),
        "end_utc": datetime.fromtimestamp(int(time.max()), UTC).isoformat(),
        **extra,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(SOURCES))
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(normalize(args.dataset, args.archive, args.output), indent=2))
