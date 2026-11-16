# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse bound contracts in a fresh process: ``python -m conformance.transfer <dir>...``.

The directory holds ``contract.json`` (a bound contract), ``inputs.json`` (the
Parquet paths and the requests) and receives ``batches.json``. The process
ingests its own DiskGraph index, so nothing but those files is shared, and it
materializes only a few anchors: a contract that lost its pinned state would refit.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

from kumo.pytorch.api import (
    Batch,
    DataLoaderSpec,
    Dataset,
    DiskGraphEngine,
    Parquet,
    SampleRequest,
    prepare,
)

Json = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
TRANSFER_ARGUMENTS = {"batch_size": 8}
#: The fresh process materializes fewer anchors, so a refit there would change the state.
FRESH_FIT = 4


def jsonable(value: object) -> Json:
    """Plain JSON: tensors become nested lists, with their dtype."""

    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "values": value.tolist()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)


#: Identify the contract, which binding changes on purpose; everything else must match.
FINGERPRINTS = ("artifact_fingerprint", "contract_fingerprint")


def batches_json(batches: Iterable[Batch]) -> Json:
    return [
        jsonable(
            {
                "inputs": batch.inputs,
                "targets": batch.targets,
                "metadata": {k: v for k, v in batch.metadata.items() if k not in FINGERPRINTS},
            }
        )
        for batch in batches
    ]


def transfer_command(*directories: Path) -> list[str]:
    return [sys.executable, "-m", "conformance.transfer", *map(str, directories)]


def main(directory: Path) -> None:
    contract = DataLoaderSpec.from_json((directory / "contract.json").read_text())
    inputs = json.loads((directory / "inputs.json").read_text())
    dataset = Dataset({name: Parquet(path) for name, path in inputs["tables"].items()})
    requests = [SampleRequest(**request) for request in inputs["requests"]]
    engine = DiskGraphEngine(index=directory / "index", num_threads=1, num_shards=1)
    with prepare(contract, dataset, engine) as plan:
        task = plan.materialize(requests[:FRESH_FIT])
        output = batches_json(task.batches(requests, **TRANSFER_ARGUMENTS))
    (directory / "batches.json").write_text(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        main(Path(argument))
