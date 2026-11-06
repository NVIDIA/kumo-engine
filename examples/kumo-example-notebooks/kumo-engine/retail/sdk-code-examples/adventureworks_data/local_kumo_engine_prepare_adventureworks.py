"""Validate and prepare the AdventureWorks tables used by the Linux examples."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path


SPECS = {
    "Customer": {
        "source_width": 7,
        "columns": [("CustomerID", 0), ("StoreID", 2), ("TerritoryID", 3)],
    },
    "Product": {
        "source_width": 25,
        "columns": [
            ("ProductID", 0), ("Name", 1), ("Color", 5),
            ("ProductLine", 15), ("Class", 16),
            ("StandardCost", 8), ("ListPrice", 9),
        ],
    },
    "SalesOrderHeader": {
        "source_width": 26,
        "columns": [
            ("SalesOrderID", 0), ("CustomerID", 10), ("OrderDate", 2),
            ("OnlineOrderFlag", 6), ("TerritoryID", 12),
            ("SubTotal", 19), ("TotalDue", 22),
        ],
    },
    "SalesOrderDetail": {
        "source_width": 11,
        "columns": [
            ("SalesOrderDetailID", 1), ("SalesOrderID", 0),
            ("ProductID", 4), ("ModifiedDate", 10), ("OrderQty", 3),
            ("UnitPrice", 6), ("LineTotal", 8),
        ],
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(path, spec):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for number, source in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(source) != spec["source_width"]:
                raise ValueError(
                    f"{path.name}:{number}: expected {spec['source_width']} "
                    f"fields, got {len(source)}"
                )
            rows.append({name: source[index] for name, index in spec["columns"]})
    if not rows:
        raise ValueError(f"{path.name}: source table is empty")
    return rows


def unique_key(rows, table, column):
    values = [row[column] for row in rows]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{table}.{column}: primary key must be nonempty and unique")
    return set(values)


def prepare(source, destination):
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directories")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination already exists: {destination}")
    tables = {
        name: read_source(source / f"{name}.csv", spec)
        for name, spec in SPECS.items()
    }
    customer_ids = unique_key(tables["Customer"], "Customer", "CustomerID")
    product_ids = unique_key(tables["Product"], "Product", "ProductID")
    order_ids = unique_key(
        tables["SalesOrderHeader"], "SalesOrderHeader", "SalesOrderID"
    )
    unique_key(
        tables["SalesOrderDetail"], "SalesOrderDetail", "SalesOrderDetailID"
    )
    relationships = [
        ("SalesOrderHeader", "CustomerID", customer_ids, "Customer"),
        ("SalesOrderDetail", "SalesOrderID", order_ids, "SalesOrderHeader"),
        ("SalesOrderDetail", "ProductID", product_ids, "Product"),
    ]
    for table, column, target_keys, target in relationships:
        missing = {row[column] for row in tables[table]} - target_keys
        if missing:
            raise ValueError(
                f"{table}.{column}: {len(missing)} values are absent from {target}"
            )
    customer_by_order = {
        row["SalesOrderID"]: row["CustomerID"]
        for row in tables["SalesOrderHeader"]
    }
    for row in tables["SalesOrderDetail"]:
        customer_id = customer_by_order.get(row["SalesOrderID"])
        if customer_id is None:
            raise ValueError(
                "SalesOrderDetail.SalesOrderID must resolve to exactly one "
                "SalesOrderHeader row"
            )
        if customer_id not in customer_ids:
            raise ValueError(
                "Derived SalesOrderDetail.CustomerID is absent from Customer"
            )
        row["CustomerID"] = customer_id
    relationships.append(
        ("SalesOrderDetail", "CustomerID", customer_ids, "Customer")
    )
    timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    order_times = [
        datetime.strptime(row["OrderDate"], timestamp_format)
        for row in tables["SalesOrderHeader"]
    ]
    detail_times = [
        datetime.strptime(row["ModifiedDate"], timestamp_format)
        for row in tables["SalesOrderDetail"]
    ]
    destination.mkdir(parents=True)
    metadata = {}
    for name, rows in tables.items():
        path = destination / f"{name}.csv"
        columns = [column for column, _ in SPECS[name]["columns"]]
        if name == "SalesOrderDetail":
            columns.insert(2, "CustomerID")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        metadata[name] = {
            "file": path.name,
            "rows": len(rows),
            "columns": columns,
            "sha256": digest(path),
        }
    return {
        "tables": metadata,
        "relationships": [
            {"src_table": table, "fkey": column, "dst_table": target}
            for table, column, _, target in relationships
        ],
        "order_date_min": min(order_times).isoformat(),
        "order_date_max": max(order_times).isoformat(),
        "detail_modified_min": min(detail_times).isoformat(),
        "detail_modified_max": max(detail_times).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
