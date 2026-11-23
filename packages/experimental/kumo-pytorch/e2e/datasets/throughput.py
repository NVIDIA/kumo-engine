# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a single pinned dataset and bind its shared Parquet anchors to a workload."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

_DATASETS = {"M04": "mbd", "S02": "synthetic-medium", "L01": "aml-li-medium"}


def pin_path(workload):
    return Path(__file__).parent / "pins" / "throughput" / f"{_DATASETS[workload]}.json"


def _check(path, pin):
    if path.stat().st_size != pin["size"]:
        raise ValueError(f"Size mismatch: {pin['key']}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {pin['key']}")


def _files(root, manifest, *, download=False):
    files = {}
    root = Path(root).resolve()
    for name, pin in manifest["objects"].items():
        key = pin["key"]
        path = root / key
        if Path(key).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Object key outside dataset root: {key}")
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=path.parent) as tmp:
                pending = Path(tmp) / "download"
                print(
                    f"Downloading s3://{manifest['bucket']}/{key} ({pin['size']:,} bytes)",
                    flush=True,
                )
                subprocess.run(
                    [
                        "aws",
                        "s3api",
                        "get-object",
                        "--region",
                        manifest["region"],
                        "--expected-bucket-owner",
                        manifest["expected_bucket_owner"],
                        "--bucket",
                        manifest["bucket"],
                        "--key",
                        key,
                        str(pending),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                _check(pending, pin)
                pending.replace(path)
        else:
            _check(path, pin)
        files[name] = path
    return files


def load_fixture(workload, root):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from e2e.datasets.throughput_contracts import aml_li_medium, mbd, synthetic_medium
    from kumo.pytorch import api

    manifest = json.loads(pin_path(workload).read_text())
    files = _files(root, manifest)
    factory, root_key, targets = {
        "M04": (mbd, "root_id", ("count",)),
        "S02": (synthetic_medium, "customer_id", ()),
        "L01": (aml_li_medium, "entity_id", ("amount", "currency")),
    }[workload]
    contract = factory(manifest["capacity"])
    path = files["anchors"]
    rows = pq.read_table(path, columns=["sample_id", "prediction_time_ns", "as_of_time", "split"])
    counts = {v["values"]: v["counts"] for v in rows["split"].value_counts().to_pylist()}
    if (
        counts != manifest["split_counts"]
        or rows["sample_id"].null_count
        or pc.count_distinct(rows["sample_id"]).as_py() != len(rows)
    ):
        raise ValueError(f"{workload}: shared anchor population differs from its pin")
    prediction_ns = rows["prediction_time_ns"]
    # The prepared file already projects the graph cutoff from the shared prediction time.
    cutoff_ns = (
        pc.subtract(prediction_ns, 1_000_000_000) if workload in {"M04", "L01"} else prediction_ns
    )
    if prediction_ns.null_count or not rows["as_of_time"].cast(pa.int64()).equals(cutoff_ns):
        raise ValueError(f"{workload}: sampling cutoff disagrees with the shared prediction time")
    latest_train = pc.max(prediction_ns.filter(pc.equal(rows["split"], "train"))).as_py()
    fit_cutoff = latest_train // 1_000_000_000 - 1
    dataset = api.Dataset({t.name: api.Parquet(str(files[t.name])) for t in contract.schema.tables})
    anchors = api.AnchorTable(
        api.Parquet(str(path)),
        root_key,
        "as_of_time",
        sample_id="sample_id",
        split="split",
        targets={name: name for name in targets},
        metadata=("registered_entity_id", "prediction_time_ns", "anchor_row_position"),
    )
    return contract.resolve(dataset=dataset), dataset, anchors, fit_cutoff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=tuple(_DATASETS))
    parser.add_argument("--stage", required=True, type=Path, metavar="DIR")
    args = parser.parse_args()
    _files(args.stage, json.loads(pin_path(args.workload).read_text()), download=True)
