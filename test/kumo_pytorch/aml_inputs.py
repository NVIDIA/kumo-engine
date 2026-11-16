"""Pinned AML inputs and bounded, unchanged anchor selections."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import load_manifest, safe_path, verify_file

SPLITS = ("train", "valid", "test")
SECOND = 1_000_000_000


def verify_inputs(data_dir, variant="hi-small"):
    """Verify complete downloaded files before selecting any source or anchor."""
    entry = load_manifest()["datasets"][variant]
    files = {item["key"]: item for item in entry["files"]}
    for key, item in files.items():
        verify_file(safe_path(Path(data_dir), key), item)
    for version in ("v1", "v2"):
        prefix = f"anchors/aml-data/v8/{variant}/account-daily/{version}"
        manifest = json.loads(
            safe_path(Path(data_dir), f"{prefix}/manifest.json").read_text()
        )
        identity = manifest["definition"]
        expected = dict(
            dataset_id=f"aml-data/{variant}",
            dataset_version="v8",
            task_id="account-daily",
            anchor_version=version,
        )
        if any(identity[name] != value for name, value in expected.items()):
            raise ValueError("Anchor manifest identity differs from the selection")
        item = files[f"{prefix}/anchors.parquet"]
        if manifest["files"] != [
            {
                "path": "anchors.parquet",
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        ]:
            raise ValueError("Anchor payload differs from the pinned manifest")
        if (
            pq.ParquetFile(safe_path(Path(data_dir), item["key"])).metadata.num_rows
            != manifest["statistics"]["rows"]
        ):
            raise ValueError("Anchor row count differs from its manifest")
    return files


def select_anchors(path, *, split_columns, limit, graph=False):
    """First N physical rows per explicit split; never select by label/content.

    This scans in Arrow batches and retains only the bounded selection. The
    caller must verify the complete source payload before calling this reader.
    """
    if (
        type(limit) is not int
        or limit < 1
        or not set(split_columns) <= {"split", "forecast_split"}
    ):
        raise ValueError("Expected a positive limit and AML split columns")
    selected = {column: {split: [] for split in SPLITS} for column in split_columns}
    columns = [
        "sample_id",
        "entity_id",
        "as_of_time",
        "split",
        "forecast_split",
        "involvement",
    ]
    if graph:
        columns.append("graph_as_of_time")
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(
        columns=columns, batch_size=65536, use_threads=False
    ):
        for column, groups in selected.items():
            for split, rows in groups.items():
                if len(rows) == limit:
                    continue
                mask = pc.fill_null(pc.equal(batch.column(column), split), False)
                positions = pc.indices_nonzero(mask).slice(0, limit - len(rows))
                chosen = batch.take(positions)
                for name in (
                    ("as_of_time", "graph_as_of_time") if graph else ("as_of_time",)
                ):
                    if chosen.schema.field(name).type != pa.timestamp("ns"):
                        raise ValueError(
                            f"Expected original nanosecond anchor clock: {name}"
                        )
                    chosen = chosen.set_column(
                        chosen.schema.get_field_index(name),
                        name,
                        pc.cast(chosen.column(name), pa.int64()),
                    )
                for position, row in zip(
                    positions.to_pylist(), chosen.to_pylist(), strict=True
                ):
                    if not row["sample_id"] or not row["entity_id"]:
                        raise ValueError("Missing anchor identity")
                    stamp = row["as_of_time"]
                    if stamp is None or stamp % (86400 * SECOND):
                        raise ValueError(
                            "AML anchors require daily UTC midnight prediction times"
                        )
                    if graph and row["graph_as_of_time"] != stamp - SECOND:
                        raise ValueError(
                            "v2 graph cutoff must equal original prediction time minus one second"
                        )
                    if row["involvement"] not in (None, 0, 1):
                        raise ValueError("Involvement label must be binary or null")
                    rows.append({**row, "physical_row_position": offset + position})
        offset += batch.num_rows
        if all(
            len(rows) == limit
            for groups in selected.values()
            for rows in groups.values()
        ):
            break
    for groups in selected.values():
        for split, rows in groups.items():
            if len(rows) != limit or len({row["sample_id"] for row in rows}) != limit:
                raise ValueError(f"Insufficient or duplicate {split} anchor rows")
    return selected


def load_selections(data_dir, *, limit):
    prefix = Path(data_dir) / "anchors/aml-data/v8/hi-small/account-daily"
    v1 = select_anchors(
        prefix / "v1/anchors.parquet",
        split_columns=("split", "forecast_split"),
        limit=limit,
    )
    v2 = select_anchors(
        prefix / "v2/anchors.parquet",
        split_columns=("forecast_split",),
        limit=limit,
        graph=True,
    )
    for split in SPLITS:
        for left, right in zip(
            v1["forecast_split"][split], v2["forecast_split"][split], strict=True
        ):
            if any(left[key] != right[key] for key in left):
                raise ValueError(
                    "Selected v2 anchors changed original values or physical order"
                )
    return {"v1": v1, "v2": v2}


def requests_for(rows, identity_map, *, split, graph, native):
    from kumo.pytorch.api import SampleRequest, TargetValue

    result = []
    for row in rows:
        column = "split" if native else "forecast_split"
        if split not in SPLITS or row[column] != split:
            raise ValueError("Cannot relabel an anchor split")
        prediction = row["as_of_time"]
        cutoff = row["graph_as_of_time"] if graph else prediction
        if graph and cutoff != prediction - SECOND:
            raise ValueError("Invalid graph execution clock")
        metadata = {
            "registered_entity_id": row["entity_id"],
            "anchor_row_position": row["physical_row_position"],
        }
        if graph:
            metadata["prediction_time_ns"] = prediction
        result.append(
            SampleRequest(
                row["sample_id"],
                identity_map[row["entity_id"]],
                cutoff // SECOND,
                targets=() if native else (TargetValue("label", row["involvement"]),),
                split=split,
                metadata=metadata,
            )
        )
    return tuple(result)
