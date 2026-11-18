"""Validated Parquet bundle metadata, shared by every backend.

Full validation uses streaming file hashes and spillable DuckDB constraints;
only task rows and sampled graph rows need to become Python objects.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

TYPES = {"int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_(),
         "string": pa.string(), "date": pa.date32(), "timestamp[us]": pa.timestamp("us", tz="UTC")}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def check_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name.startswith("__") or name == "instance_id":
        raise ValueError(f"Unsupported or reserved schema name: {name!r}")


class Bundle:
    def __init__(self, root: Path, *, verify_files: bool = True):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        self.dataset_manifest = m if m.get("format") == "fm-dataset-v1" else None
        if self.dataset_manifest is None and m.get("format_version") != 2:
            raise ValueError("Expected dataset-suite bundle format_version=2")
        complete = json.loads((self.root / "_COMPLETE.json").read_text())
        if complete.get("manifest_sha256") != digest(self.root / "manifest.json"):
            raise ValueError("Incomplete or changed bundle manifest")
        validation = json.loads((self.root / "validation.json").read_text())
        if (validation.get("status") != ("pass" if self.dataset_manifest else "passed") or
                complete.get("validation_sha256") != digest(self.root / "validation.json")):
            raise ValueError("Bundle has no matching completed validation")
        if self.dataset_manifest is not None:
            from .datasets import manifest_view
            self.manifest = m = manifest_view(self.dataset_manifest, self.root)
        self.tables = m["tables"]
        self.relations = m["relations"]
        self.logical_id = m["logical_id"]
        self.snapshot_id = self.dataset_manifest["snapshot_id"] if self.dataset_manifest else identity(m)
        if not self.tables:
            raise ValueError("Bundle must declare tables")
        for name, table in self.tables.items():
            check_name(name)
            columns = table["columns"]
            for column, spec in columns.items():
                check_name(column)
                if spec["dtype"] not in TYPES or type(spec.get("nullable")) is not bool:
                    raise ValueError(f"Unsupported column specification: {name}.{column}")
            pk = table["primary_key"]
            if pk not in columns or columns[pk]["dtype"] not in ("int64", "string") or columns[pk]["nullable"]:
                raise ValueError(f"Table {name} needs a non-null int64/string primary key")
            tc = table.get("time_column")
            if tc is not None and (tc not in columns or columns[tc]["dtype"] not in ("timestamp[us]", "date")):
                raise ValueError(f"Invalid time column: {name}")
            if tc is None and table.get("availability") != "timeless":
                raise ValueError(f"Declare {name} temporal or explicitly timeless")
            if type(table["rows"]) is not int or table["rows"] < 0 or not table["files"]:
                raise ValueError(f"Invalid table inventory: {name}")
            if table["files"] != sorted(set(table["files"])):
                raise ValueError(f"Table file inventory must be sorted and unique: {name}")
            if any(p not in m["files"] for p in table["files"]):
                raise ValueError(f"Uninventoried table file: {name}")
        for name, rel in self.relations.items():
            check_name(name)
            child, parent = self.tables[rel["child"]], self.tables[rel["parent"]]
            fk = child["columns"][rel["foreign_key"]]
            pk = parent["columns"][parent["primary_key"]]
            if fk["dtype"] != pk["dtype"]:
                raise ValueError(f"Incompatible relation key types: {name}")
        if verify_files:
            for name, entry in m["files"].items():
                if digest(self.path(name)) != entry["sha256"]:
                    raise ValueError(f"Bundle checksum mismatch: {name}")

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("Bundle file path escapes the dataset directory")
        return path

    def paths(self, name: str) -> list[str]:
        return [str(self.path(p)) for p in self.tables[name]["files"]]

    def schema(self, name: str) -> pa.Schema:
        return pa.schema([pa.field(c, TYPES[s["dtype"]], s["nullable"])
                          for c, s in self.tables[name]["columns"].items()])

    def endpoint(self, step: dict) -> tuple[str, str]:
        rel = self.relations[step["relation"]]
        return (rel["child"], rel["parent"]) if step["direction"] == "forward" else (rel["parent"], rel["child"])

    def register(self, con, *, native: bool = False) -> None:
        for name in self.tables:
            con.read_parquet(self.paths(name)).create_view("__source_" + name)
            kind = "TABLE" if native else "VIEW"
            con.execute(f"CREATE {kind} {quote(name)} AS SELECT * FROM {quote('__source_' + name)}").fetchall()
        if native:
            con.execute("ANALYZE").fetchall()

    def validate(self, *, memory_limit: str = "1GB", threads: int = 2) -> dict:
        """Full source validation; preparation cost, never timed as retrieval."""
        if getattr(self, "dataset_manifest", None) is not None:
            from ._dataset_support.validate import validate
            validate(self.root)
        result = {"tables": {}, "relations": {}}
        with duckdb.connect(config={"memory_limit": memory_limit, "threads": threads}) as con:
            self.register(con)
            for name, table in self.tables.items():
                expected = self.schema(name)
                for path in self.paths(name):
                    parquet = pq.ParquetFile(path)
                    if not parquet.schema_arrow.equals(expected):
                        raise ValueError(f"Parquet schema differs from manifest: {path}")
                    entry = self.manifest["files"][str(Path(path).relative_to(self.root))]
                    if parquet.metadata.num_rows != entry["rows"]:
                        raise ValueError(f"File row count differs from inventory: {path}")
                n = con.execute(f"SELECT count(*) FROM {quote(name)}").fetchall()[0][0]
                if n != table["rows"]:
                    raise ValueError(f"Table row count mismatch: {name}")
                for col, spec in table["columns"].items():
                    if not spec["nullable"] and con.execute(
                        f"SELECT count(*) FROM {quote(name)} WHERE {quote(col)} IS NULL"
                    ).fetchall()[0][0]:
                        raise ValueError(f"Null required column: {name}.{col}")
                    if spec["dtype"] == "float64" and con.execute(
                        f"SELECT count(*) FROM {quote(name)} WHERE NOT isfinite({quote(col)})"
                    ).fetchall()[0][0]:
                        raise ValueError(f"Nonfinite feature: {name}.{col}")
                pk = quote(table["primary_key"])
                if con.execute(f"SELECT count(*) FROM (SELECT {pk} FROM {quote(name)} GROUP BY {pk} HAVING count(*) > 1)").fetchall()[0][0]:
                    raise ValueError(f"Duplicate primary key: {name}")
                result["tables"][name] = {"rows": n}
            for name, rel in self.relations.items():
                pk = self.tables[rel["parent"]]["primary_key"]
                child, parent, fk = quote(rel["child"]), quote(rel["parent"]), quote(rel["foreign_key"])
                bad = con.execute(f"SELECT count(*) FROM {child} c LEFT JOIN {parent} p ON c.{fk}=p.{quote(pk)} WHERE c.{fk} IS NOT NULL AND p.{quote(pk)} IS NULL").fetchall()[0][0]
                if bad:
                    raise ValueError(f"Dangling foreign keys: {name} ({bad})")
                nulls = con.execute(f"SELECT count(*) FROM {child} WHERE {fk} IS NULL").fetchall()[0][0]
                result["relations"][name] = {"null_links": nulls, "edges": self.tables[rel["child"]]["rows"] - nulls}
        self.validation = result
        return result
