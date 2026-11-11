# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote feeder prototype: one script yields the same bytes locally and through a feeder."""

from __future__ import annotations

import itertools
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

pytest.importorskip("diskgraph")
pytest.importorskip("pyarrow.flight")
pytest.importorskip("torch_geometric")

import pyarrow as pa  # noqa: E402
import torch  # noqa: E402

from e2e.flows import hm_gnn, hm_pragma  # noqa: E402
from e2e.flows.common import TABLES, anchors, dataset  # noqa: E402
from kumo.pytorch.api import (  # noqa: E402
    AnchorTable,
    Batch,
    DataLoaderSpec,
    Dataset,
    DiskGraphEngine,
    Encode,
    Fields,
    InMemoryRows,
    Input,
    ValidationError,
    ZScore,
    prepare,
)
from kumo.pytorch.artifact import FittedState  # noqa: E402
from kumo.pytorch.assembly import PreparedBatchPlan  # noqa: E402
from kumo.pytorch.datasets import hm  # noqa: E402
from kumo.pytorch.engine import Engine  # noqa: E402
from kumo.pytorch.engines.distributed import Cluster, DistributedEngine  # noqa: E402
from kumo.pytorch.engines.distributed.wire import LISTENING  # noqa: E402

BATCH_SIZE = 128
CONTRACTS: Mapping[str, Callable[[], DataLoaderSpec]] = {
    "gnn": hm_gnn.contract,
    "sequence": hm_pragma.pretraining_contract,
}


@dataclass(frozen=True)
class Workload:
    data: Path
    index: Path

    def inner(self) -> DiskGraphEngine:
        return DiskGraphEngine(index=self.index, num_threads=4)


@dataclass
class Run:
    fitted_state: FittedState
    fingerprints: dict[str, str]
    batches: list[Batch]


@pytest.fixture(scope="module")
def workload(tmp_path_factory: pytest.TempPathFactory) -> Workload:
    root = tmp_path_factory.mktemp("hm-xs")
    hm(root / "data", "xs")
    return Workload(root / "data", root / "index")


def run(
    engine: Engine,
    contract: DataLoaderSpec,
    data: Path,
    *,
    world_size: int,
    output_format: Literal["inspectable", "model"],
) -> Run:
    """The training script: two shuffled epochs on every rank, then a mid-epoch resume."""

    def arguments(epoch: int, rank: int) -> dict[str, object]:
        return {
            "split": "train",
            "batch_size": BATCH_SIZE,
            "epoch": epoch,
            "seed": 11,
            "shuffle": True,
            "rank": rank,
            "world_size": world_size,
            "output_format": output_format,
        }

    batches: list[Batch] = []
    with prepare(contract, dataset(data), engine) as plan:
        task = plan.materialize(anchors(data), fit_split="train")
        for epoch in (0, 1):
            for rank in range(world_size):
                batches.extend(task.batches(**arguments(epoch, rank)))
        for rank in range(world_size):
            first = task.batches(**arguments(1, rank))
            batches.extend([next(first), next(first)])
            resumed = task.batches(**arguments(1, rank))
            resumed.load_state_dict(first.state_dict())
            batches.extend(resumed)
        return Run(task.fitted_state, dict(task.fingerprints), batches)


def assert_identical(actual: object, expected: object, path: str = "batch") -> None:
    assert type(actual) is type(expected), f"{path}: {type(actual)} != {type(expected)}"
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert actual.contiguous().numpy().tobytes() == expected.contiguous().numpy().tobytes(), (
            path
        )
    elif isinstance(expected, pa.Array | pa.ChunkedArray):
        assert actual.equals(expected), path  # type: ignore[union-attr]
    elif isinstance(expected, Batch):
        assert isinstance(actual, Batch)
        for name in ("inputs", "targets", "metadata"):
            assert_identical(getattr(actual, name), getattr(expected, name), f"{path}.{name}")
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert list(actual) == list(expected), path
        for key in expected:
            assert_identical(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, Sequence)
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_identical(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path


@pytest.mark.parametrize(
    ("contract_name", "world_size", "output_format"),
    [
        ("gnn", 1, "model"),
        ("gnn", 2, "model"),
        ("gnn", 1, "inspectable"),
        ("sequence", 1, "model"),
        ("sequence", 2, "model"),
        ("sequence", 2, "inspectable"),
    ],
)
def test_same_script_gives_identical_batches_locally_and_through_a_feeder(
    workload: Workload,
    contract_name: str,
    world_size: int,
    output_format: Literal["inspectable", "model"],
) -> None:
    contract = CONTRACTS[contract_name]()

    def script(engine: Engine) -> Run:
        return run(
            engine, contract, workload.data, world_size=world_size, output_format=output_format
        )

    local = script(workload.inner())
    with DistributedEngine(workload.inner(), cluster=Cluster.subprocess(feeders=1)) as engine:
        remote = script(engine)
    assert remote.fingerprints == local.fingerprints
    assert remote.fitted_state == local.fitted_state
    assert len(remote.batches) == len(local.batches) > 4 * world_size
    for index, (actual, expected) in enumerate(zip(remote.batches, local.batches, strict=True)):
        assert_identical(actual, expected, f"batches[{index}]")


_LOCAL_RUNS: dict[tuple[str, int], Run] = {}


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(("feeders", "stripes"), [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("contract_name", ["gnn", "sequence"])
def test_striped_feeders_give_identical_batches(
    workload: Workload, contract_name: str, feeders: int, stripes: int, world_size: int
) -> None:
    contract = CONTRACTS[contract_name]()

    def script(engine: Engine) -> Run:
        return run(engine, contract, workload.data, world_size=world_size, output_format="model")

    key = (contract_name, world_size)
    if key not in _LOCAL_RUNS:
        _LOCAL_RUNS[key] = script(workload.inner())
    local = _LOCAL_RUNS[key]
    with DistributedEngine(
        workload.inner(), cluster=Cluster.subprocess(feeders=feeders), stripes=stripes
    ) as engine:
        remote = script(engine)
        assert [len(pids) for pids in engine.producer_pids] == [stripes] * feeders
    assert remote.fingerprints == local.fingerprints
    assert len(remote.batches) == len(local.batches)
    for index, (actual, expected) in enumerate(zip(remote.batches, local.batches, strict=True)):
        assert_identical(actual, expected, f"batches[{index}]")


@pytest.mark.parametrize("contract_name", ["gnn", "sequence"])
def test_strided_passes_build_only_their_batches(
    workload: Workload, contract_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    original = PreparedBatchPlan.batches

    def counting(self: PreparedBatchPlan, values: object, **options: object) -> object:
        for batch in original(self, values, **options):
            built.append(1)
            yield batch

    with prepare(CONTRACTS[contract_name](), dataset(workload.data), workload.inner()) as plan:
        task = plan.materialize(anchors(workload.data), fit_split="train")
        for world_size, drop_last, (first, step) in itertools.product(
            (1, 7), (False, True), ((0, 1), (3, 1), (1, 4), (5, 3), (2, 2))
        ):
            arguments: dict[str, object] = {
                "split": "train",
                "batch_size": 12,
                "epoch": 1,
                "seed": 5,
                "shuffle": True,
                "rank": world_size - 1,
                "world_size": world_size,
                "drop_last": drop_last,
                "chunk_batches": 4,
                "output_format": "model",
            }
            expected = list(itertools.islice(task.batches(**arguments), first, None, step))
            monkeypatch.setattr(PreparedBatchPlan, "batches", counting)
            built.clear()
            actual = list(task.batches(**arguments)._strided(first, step))
            monkeypatch.setattr(PreparedBatchPlan, "batches", original)
            assert len(built) == len(actual) == len(expected) > 0
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                assert_identical(left, right, f"{world_size}/{drop_last}/{first}/{step}[{index}]")


def _start_feeder() -> tuple[subprocess.Popen[str], str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "kumo.pytorch.feeder", "--listen", "127.0.0.1:0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    assert line.startswith(LISTENING + " "), line
    return process, line.split(" ", 1)[1]


def _first_batch(engine: Engine, workload: Workload) -> Batch:
    with prepare(hm_gnn.contract(), dataset(workload.data), engine) as plan:
        task = plan.materialize(anchors(workload.data), fit_split="train")
        return next(task.batches(split="train", batch_size=BATCH_SIZE, output_format="model"))


def test_endpoints_attach_to_a_running_feeder_and_close_leaves_it_running(
    workload: Workload,
) -> None:
    expected = _first_batch(workload.inner(), workload)
    process, url = _start_feeder()
    try:
        for _ in range(2):
            with DistributedEngine(workload.inner(), cluster=Cluster.endpoints([url])) as engine:
                assert engine.feeder_pids == ()
                assert_identical(_first_batch(engine, workload), expected)
            time.sleep(0.5)
            assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


def _gone(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return True
    return "\nState:\tZ" in status


def _wait_gone(pid: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.2)
    return False


def test_close_stops_the_feeders_it_launched(workload: Workload) -> None:
    engine = DistributedEngine(workload.inner(), cluster=Cluster.subprocess(feeders=1))
    engine.start()
    (pid,) = engine.feeder_pids
    assert not _gone(pid)
    engine.close()
    assert _wait_gone(pid)


def test_killing_the_driver_does_not_orphan_its_feeder(workload: Workload) -> None:
    driver = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from kumo.pytorch.api import DiskGraphEngine
                from kumo.pytorch.engines.distributed import Cluster, DistributedEngine
                engine = DistributedEngine(
                    DiskGraphEngine(index={str(workload.index)!r}), cluster=Cluster.subprocess(1)
                )
                engine.start()
                print(engine.feeder_pids[0], flush=True)
                time.sleep(600)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert driver.stdout is not None
    pid = int(driver.stdout.readline())
    assert not _gone(pid)
    os.kill(driver.pid, signal.SIGKILL)
    driver.wait()
    assert _wait_gone(pid)


def test_in_memory_rows_are_refused_before_a_feeder_starts(workload: Workload) -> None:
    engine = DistributedEngine(workload.inner(), cluster=Cluster.subprocess(feeders=1))
    in_memory = Dataset({name: InMemoryRows(()) for name in TABLES})
    with pytest.raises(ValidationError) as refused:
        prepare(hm_gnn.contract(), in_memory, engine)
    assert "DISTRIBUTED-IN-MEMORY" in {issue.code for issue in refused.value.issues}
    assert engine.feeder_pids == ()
    engine.close()


def test_in_memory_anchors_are_refused_before_they_are_shipped(workload: Workload) -> None:
    rows = [{"customer_id": 1, "cutoff": 0, "sample_id": 0, "split": "train"}]
    table = replace(anchors(workload.data), source=InMemoryRows(rows))
    assert isinstance(table, AnchorTable)
    with DistributedEngine(workload.inner(), cluster=Cluster.subprocess(feeders=1)) as engine:
        with prepare(hm_gnn.contract(), dataset(workload.data), engine) as plan:
            with pytest.raises(ValidationError) as refused:
                plan.materialize(table, fit_split="train")
    assert "DISTRIBUTED-IN-MEMORY" in {issue.code for issue in refused.value.issues}


@dataclass(frozen=True)
class ClippedZScore(ZScore):
    """A user's encoder: code the feeder doesn't have."""

    limit: float = 3.0


def test_custom_encoders_are_refused_before_a_feeder_starts(workload: Workload) -> None:
    contract = hm_gnn.contract()
    customer = contract.batch.inputs["customer"]
    assert isinstance(customer.representation, Fields)
    custom = replace(
        customer.representation,
        fields=(Encode("age", ClippedZScore()), *customer.representation.fields[1:]),
    )
    inputs = {**contract.batch.inputs, "customer": Input(customer.source, custom, customer.layout)}
    contract = replace(contract, batch=replace(contract.batch, inputs=inputs))
    engine = DistributedEngine(workload.inner(), cluster=Cluster.subprocess(feeders=1))
    with pytest.raises(ValidationError) as refused:
        prepare(contract, dataset(workload.data), engine)
    codes = {issue.code for issue in refused.value.issues}
    assert "DISTRIBUTED-CUSTOM-CODE" in codes
    paths = [issue.path for issue in refused.value.issues if issue.code in codes]
    assert any("ClippedZScore" in issue.message for issue in refused.value.issues), paths
    assert engine.feeder_pids == ()
    engine.close()
