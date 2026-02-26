"""Convert Kumo GraphConfig to DiskGraph ingest config.

The DiskgraphConfigConverter accepts a Kumo GraphConfig dict (loaded from
a local JSON file or fetched from S3) and produces the diskgraph ingest
config dict expected by the ``ingest`` binary.

Usage::

    from diskgraph.config_converter import DiskgraphConfigConverter

    converter = DiskgraphConfigConverter(
        kumo_graph_config,         # dict from graph_config.json
        data_persist_dir="/data",  # where parquet files live
        table_data_versions={"USERS": "1771621683"},
    )
    ingest_config = converter.convert()
"""

import os
from typing import Dict, List, Optional

# Mapping from kumo dtype strings to diskgraph ingested_dtype
_DTYPE_MAP = {
    "int": "int64",
    "float": "float64",
    "string": "string",
}

# Mapping from kumo stype strings to diskgraph semantic_type (lowercase)
_STYPE_MAP = {
    "ID": "id",
    "numerical": "numerical",
    "categorical": "categorical",
    "timestamp": "timestamp",
    "text": "text",
    "multicategorical": "multicategorical",
    "sequence": "sequence",
}


def _list_parquet_files(dir_path: str) -> List[str]:
    """List .parquet files in the given directory."""
    if not os.path.isdir(dir_path):
        return []
    return sorted([
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if f.endswith(".parquet")
    ])


class DiskgraphConfigConverter:
    """Converts a Kumo GraphConfig dict to a DiskGraph ingest config dict.

    Args:
        kumo_graph_config: Kumo-format GraphConfig dict (as loaded from JSON).
            Can come from a local file path or S3.
        data_persist_dir: Base directory where parquet files are stored
            (includes the 'data-persist-dir' suffix).
        table_data_versions: Per-table data version dict
            (e.g., {"ACCOUNT": "1771621683", "LOAN": "1771621683"}).
    """

    def __init__(
        self,
        kumo_graph_config: dict,
        data_persist_dir: str,
        table_data_versions: Dict[str, str],
    ):
        self.kumo_config = kumo_graph_config
        self.data_persist_dir = data_persist_dir
        self.table_data_versions = table_data_versions

    @classmethod
    def from_file(
        cls,
        config_path: str,
        data_persist_dir: str,
        table_data_versions: Dict[str, str],
    ) -> "DiskgraphConfigConverter":
        """Create a converter from a local graph_config.json file path."""
        import json
        with open(config_path) as f:
            kumo_graph_config = json.load(f)
        return cls(kumo_graph_config, data_persist_dir, table_data_versions)

    def convert(self) -> dict:
        """Convert to diskgraph ingest config dict.

        Returns:
            Dict ready to be serialized to JSON and passed to the ingest binary.

        Raises:
            ValueError: If table_data_versions is missing entries or no
                parquet files are found for a table.
        """
        kumo_config = self.kumo_config

        # Dataset name: strip _graph suffix if present
        raw_name = kumo_config.get("name", "")
        if raw_name.endswith("_graph"):
            dataset_name = raw_name[: -len("_graph")]
        else:
            dataset_name = raw_name

        # Build time_info map: table_name -> create_time column name.
        time_info: Dict[str, str] = {}
        for table in kumo_config.get("tables", []):
            tc = table.get("time_col")
            if tc:
                time_info[table["table_name"]] = tc.get("name") if isinstance(tc, dict) else tc

        tables = []
        missing_table_versions: List[str] = []
        missing_parquet_sources: List[str] = []

        for table in kumo_config.get("tables", []):
            table_name = table["table_name"]
            connector_id = table.get("data_source") or table.get("connector_id") or "default"
            source_name = table.get("source_table_name", table_name)

            # Convert columns.  GraphConfig stores pkey and time_col
            # separately from ``cols``, but the Rust ingest binary
            # expects them in the columns array as well.
            columns = []
            col_names_seen: set = set()

            def _append_col(col: dict) -> None:
                name = col.get("name", "")
                if not name or name in col_names_seen:
                    return
                col_names_seen.add(name)
                stype_raw = col.get("stype", "")
                dtype_raw = col.get("dtype", "")
                semantic_type = _STYPE_MAP.get(stype_raw, stype_raw.lower())
                if semantic_type == "timestamp":
                    ingested_dtype = "timestamp"
                else:
                    ingested_dtype = _DTYPE_MAP.get(dtype_raw, dtype_raw)
                columns.append({
                    "name": name,
                    "ingested_dtype": ingested_dtype,
                    "semantic_type": semantic_type,
                })

            # Add pkey and time_col first so they appear in the columns list.
            raw_pkey_col = table.get("pkey")
            if isinstance(raw_pkey_col, dict):
                _append_col(raw_pkey_col)
            raw_time_col_col = table.get("time_col")
            if isinstance(raw_time_col_col, dict):
                _append_col(raw_time_col_col)

            # Add foreign key columns so src_column references resolve.
            for fk in table.get("fkeys", []):
                if isinstance(fk, dict):
                    _append_col(fk)

            for col in table.get("cols", []):
                _append_col(col)

            raw_src_time = table.get("time_col")
            src_create_time = (raw_src_time.get("name")
                               if isinstance(raw_src_time, dict)
                               else raw_src_time)

            # Convert foreign keys with forward/reverse direction specs
            fkeys = []
            for fk in table.get("fkeys", []):
                raw_dst = fk["dst_table"]
                # GraphConfig dst_table is "connector.TABLE"; strip the
                # connector prefix so it matches the diskgraph table name.
                dst_table_name = raw_dst.split(".", 1)[-1] if "." in raw_dst else raw_dst
                fk_spec: dict = {
                    "src_column": fk["name"],
                    "dst_table": dst_table_name,
                }

                # Forward direction: destination table's create_time
                dst_create_time = time_info.get(dst_table_name)
                if dst_create_time:
                    fk_spec["forward"] = {
                        "time": {
                            "table": dst_table_name,
                            "column": dst_create_time,
                            "unit": "s",
                        },
                    }

                # Reverse direction: source (current) table's create_time
                if src_create_time:
                    fk_spec["reverse"] = {
                        "time": {
                            "table": table_name,
                            "column": src_create_time,
                            "unit": "s",
                        },
                    }

                fkeys.append(fk_spec)

            # Build source paths from filesystem
            table_version = self.table_data_versions.get(table_name, "")
            if not str(table_version):
                missing_table_versions.append(table_name)
            parquet_dir = os.path.join(
                self.data_persist_dir,
                "ingest",
                f"fpv3.{connector_id}.{source_name}",
                str(table_version),
            )
            paths = _list_parquet_files(parquet_dir)
            if not paths:
                missing_parquet_sources.append(
                    f"{table_name} (connector={connector_id}, "
                    f"source={source_name}, version={table_version}, "
                    f"dir={parquet_dir})"
                )

            # Extract just the column name string from pkey/time_col,
            # which may be serialized as dicts with a "name" field.
            raw_pkey = table.get("pkey")
            if isinstance(raw_pkey, dict):
                raw_pkey = raw_pkey.get("name")

            raw_time_col = table.get("time_col")
            if isinstance(raw_time_col, dict):
                raw_time_col = raw_time_col.get("name")

            tables.append({
                "name": table_name,
                "columns": columns,
                "primary_key": raw_pkey,
                "create_time": raw_time_col,
                "source": {
                    "type": "parquet",
                    "paths": paths,
                },
                "fkeys": fkeys,
            })

        if missing_table_versions:
            raise ValueError(
                "Missing table_data_versions for tables: "
                + ", ".join(sorted(missing_table_versions))
            )

        if missing_parquet_sources:
            raise ValueError(
                "No parquet inputs found for diskgraph ingest tables:\n- "
                + "\n- ".join(missing_parquet_sources)
            )

        return {
            "dataset_name": dataset_name,
            "tables": tables,
        }
