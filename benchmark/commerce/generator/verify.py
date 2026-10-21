"""Verify generated Parquet hashes, schemas and row counts, without distribution tests."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from scalebench import schema


def verify(root, rung):
    root = root.resolve()
    manifest = json.loads((root / "commerce" / rung / "manifest.json").read_text())
    if manifest["graph"] != "commerce" or manifest["rung"] != rung:
        raise ValueError("manifest graph or rung does not match the requested dataset")
    if set(manifest["tables"]) != set(schema.COMMERCE):
        raise ValueError("manifest must contain all seven commerce tables")
    count = rows = total_bytes = 0
    for table, info in manifest["tables"].items():
        table_rows = table_bytes = 0
        expected_schema = schema.arrow_schema("commerce", table)
        for record in info["files"]:
            path = (root / record["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("manifest path escapes the dataset root")
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"size mismatch: {record['path']}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {record['path']}")
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != record["rows"]:
                raise ValueError(f"row-count mismatch: {record['path']}")
            if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
                raise ValueError(f"schema mismatch: {record['path']}")
            table_rows += record["rows"]
            table_bytes += record["bytes"]
            count += 1
        if table_rows != info["rows"] or table_bytes != info["bytes"]:
            raise ValueError(f"manifest totals disagree for {table}")
        rows += table_rows
        total_bytes += table_bytes
    if rows != manifest["total_rows"] or total_bytes != manifest["total_bytes"]:
        raise ValueError("manifest dataset totals disagree")
    return {"files": count, "rows": rows, "bytes": total_bytes, "checks": "sha256, schema, row counts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--rung", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, args.rung)))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Verification failed: {error}\n")


if __name__ == "__main__":
    main()
