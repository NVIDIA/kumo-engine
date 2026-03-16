"""Convert Kumo GraphConfig to DiskGraph ingest config.

The DiskgraphConfigConverter accepts a Kumo GraphConfig dict (loaded from
a local JSON file or fetched from S3) and produces the diskgraph ingest
config dict expected by the ``ingest`` binary.

This is a thin wrapper around the Rust ``convert_kumo_config`` function
exposed via pybind, which handles dtype mapping, edge table detection,
sort key construction, and timestamp format detection.

Usage::

    from diskgraph.config_converter import DiskgraphConfigConverter

    converter = DiskgraphConfigConverter(
        kumo_graph_config,         # dict from graph_config.json
        data_persist_dir="/data",  # where parquet files live
        table_data_versions={"USERS": "1771621683"},
    )
    ingest_config = converter.convert()
"""

import json
import os
from typing import Dict, List


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
        source_paths_base: str | None = None,
        source_paths: Dict[str, str] | None = None,
    ):
        self.kumo_config = kumo_graph_config
        self.data_persist_dir = data_persist_dir
        self.table_data_versions = table_data_versions
        self.source_paths_base = source_paths_base
        self.source_paths = source_paths

    @classmethod
    def from_file(
        cls,
        config_path: str,
        data_persist_dir: str,
        table_data_versions: Dict[str, str],
        source_paths_base: str | None = None,
        source_paths: Dict[str, str] | None = None,
    ) -> "DiskgraphConfigConverter":
        """Create a converter from a local graph_config.json file path."""
        with open(config_path) as f:
            kumo_graph_config = json.load(f)
        return cls(kumo_graph_config, data_persist_dir, table_data_versions,
                   source_paths_base, source_paths)

    def convert(self) -> dict:
        """Convert to diskgraph ingest config dict.

        The Rust converter handles everything: schema mapping, timestamp
        format detection, and num_shards estimation.

        Returns:
            Dict ready to be serialized to JSON and passed to the ingest binary.

        Raises:
            RuntimeError: If the Rust conversion fails.
            ValueError: If no parquet files are found for a table.
        """
        from . import convert_kumo_config as _convert_rust

        # Compute dataset_name: strip _graph suffix if present.
        raw_name = self.kumo_config.get("name", "")
        if raw_name and raw_name.endswith("_graph"):
            dataset_name = raw_name[: -len("_graph")]
        else:
            dataset_name = raw_name or ""

        # Call the Rust converter (handles timestamps + num_shards internally).
        result_json = _convert_rust(
            json.dumps(self.kumo_config),
            self.data_persist_dir,
            dataset_name,
            json.dumps(self.table_data_versions),
            self.source_paths_base,
            json.dumps(self.source_paths) if self.source_paths else None,
        )
        config = json.loads(result_json)

        # The Rust converter outputs fpv3 directory paths.  For local paths,
        # expand them to individual parquet file lists (the ingest binary
        # needs individual file paths).
        missing_parquet: List[str] = []
        for table in config.get("tables", []):
            source = table.get("source", {})
            if source.get("type") != "parquet":
                continue
            paths = source.get("paths", [])
            expanded: List[str] = []
            for path in paths:
                if path.startswith("s3://") or path.startswith("gs://") or path.startswith("az://"):
                    # Cloud paths are expanded in Rust by expand_cloud_paths_in_config;
                    # pass through as-is here.
                    expanded.append(path)
                else:
                    files = _list_parquet_files(path.rstrip("/"))
                    if files:
                        expanded.extend(files)
                    else:
                        missing_parquet.append(
                            f"{table['name']} (path={path})"
                        )
            source["paths"] = expanded

        if missing_parquet:
            raise ValueError(
                "No parquet inputs found for diskgraph ingest tables:\n- "
                + "\n- ".join(missing_parquet)
            )

        return config
