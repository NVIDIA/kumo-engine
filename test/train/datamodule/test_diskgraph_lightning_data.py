# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``DiskgraphLightningDataModule``."""
from __future__ import annotations

import json
import os
import pickle
import select
import signal
import subprocess
import sys
import threading
import time
import types
import warnings
from typing import Any

import kumo_core.diskgraph.lightning_data as dldm
import pandas as pd
import psutil
import pyarrow as pa
import pytest
import pytorch_lightning as pl
import torch
from kumo_core.diskgraph.lightning_data import (
    DiskgraphLightningDataModule,
    DiskgraphTaskBatchIterable,
    DiskgraphTaskManifest,
    DiskgraphValidRHS,
    _ShuffleStreamState,
)
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumoapi.encoder import Stype
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import HeteroData


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_nonlinux_pidfd_skips_libc(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    monkeypatch.setattr(dldm, 'sys', types.SimpleNamespace(platform=platform))

    def reject_load(name: str | None, *, use_errno: bool) -> None:
        pytest.fail('Non-Linux pidfd initialization must not load libc')

    monkeypatch.setattr(dldm.ctypes, 'CDLL', reject_load)
    assert dldm._load_pidfd_libc() is None
    assert dldm._pidfd_open(os.getpid()) is None


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='pidfd syscalls require Linux')
def test_pidfd_syscall_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(dldm.os, 'pidfd_open', raising=False)
    monkeypatch.delattr(dldm.signal, 'pidfd_send_signal', raising=False)
    pidfd = dldm._pidfd_open(os.getpid())
    assert pidfd is not None
    try:
        dldm._pidfd_send_signal(pidfd, 0)
    finally:
        os.close(pidfd)


def _numerical_mapper(values: list[float]) -> TensorMapper:
    return TensorMapper.from_stype(Stype.numerical, pd.Series(values))


def _mapper_bank() -> dict[str, dict[str, TensorMapper]]:
    return {"users_v1": {"age": _numerical_mapper([18.0, 25.0, 30.0, 50.0])}}


def test_stage_batch_allocation_is_proportional_and_covers_stages() -> None:
    assert dldm._allocate_stage_batches(
        {
            "train": 100,
            "val": 10,
        },
        10,
        require_each_stage=True,
    ) == {
        "train": 9,
        "val": 1,
    }
    with pytest.raises(ValueError, match="1 batches cannot cover 2 stages"):
        dldm._allocate_stage_batches(
            {
                "train": 100,
                "val": 10,
            },
            1,
            require_each_stage=True,
        )


@pytest.mark.parametrize("weight", [float("nan"), float("inf")])
def test_stage_batch_allocation_rejects_nonfinite_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="non-finite stages.*val"):
        dldm._allocate_stage_batches(
            {
                "train": 100,
                "val": weight
            },
            10,
            require_each_stage=True,
        )


def test_stage_batch_target_limits_runtime_worker_shards(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_stage={
            "train": [[0], [1]],
            "val": [[100], [101]],
        },
    )
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=4, id=0),
    )
    iterable = DiskgraphTaskBatchIterable(
        manifest_path="/tmp/runtime_manifest.json",
        stage=("train", "val"),
        sampling_plan={"layers": []},
        batch_size=4,
        shuffle=True,
        drop_last=False,
        base_seed=0,
        shuffle_state=_ShuffleStreamState(),
        dataloader_num_workers=4,
        stage_batch_targets={
            "train": 3,
            "val": 1,
        },
        stage_num_seeds={
            "train": 12,
            "val": 4,
        },
    )

    assert len(list(iterable)) == 2
    workers_by_stage = {
        call["kwargs"]["stage"]: call["kwargs"]["num_workers"]
        for call in calls["history"]
    }
    assert workers_by_stage == {
        "train": 3,
        "val": 1,
    }


def test_stage_batch_target_limits_workers_to_full_drop_last_shards(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=4, id=0),
    )
    iterable = DiskgraphTaskBatchIterable(
        manifest_path="/tmp/runtime_manifest.json",
        stage=("train", ),
        sampling_plan={"layers": []},
        batch_size=8,
        shuffle=True,
        drop_last=True,
        base_seed=0,
        shuffle_state=_ShuffleStreamState(),
        dataloader_num_workers=4,
        stage_batch_targets={"train": 4},
        stage_num_seeds={"train": 10},
    )

    list(iterable)

    assert {call["kwargs"]["num_workers"] for call in calls["history"]} == {1}


def test_runtime_sharding_rejects_capped_iteration() -> None:
    with pytest.raises(ValueError, match='requires uncapped iteration'):
        DiskgraphTaskBatchIterable(
            manifest_path="/tmp/runtime_manifest.json",
            stage=("train", "val"),
            sampling_plan={"layers": []},
            batch_size=4,
            shuffle=True,
            drop_last=False,
            base_seed=0,
            shuffle_state=_ShuffleStreamState(),
            runtime_num_workers=2,
            stage_batch_targets={"train": 3, "val": 1},
        )


class _WorkerProbeIterable(IterableDataset):
    """Infinite worker-local stream with a finite advertised phase length."""
    def __init__(self, advertised_batches: int) -> None:
        self.advertised_batches = advertised_batches

    def __len__(self) -> int:
        return self.advertised_batches

    def __iter__(self):
        while True:
            yield os.getpid()


class _FailingWorkerIterable(IterableDataset):
    def __len__(self) -> int:
        return 10

    def __iter__(self):
        raise RuntimeError("worker fetch failed")
        yield  # pragma: no cover


def _feature_schema() -> dict[str, Any]:
    return {
        "node_types": {
            "User": {
                "feature_columns": ["age"],
                "time_column": None,
                "pkey_column": None,
                "mapper_key": "users_v1",
            }
        },
        "edge_types": {},
        "raw_feature_format": "arrow_columnar_v1",
        "provenance_mode": "node_id_time_keyed_v1",
        "metadata": {},
    }


def _write_grouped_manifests(
    tmp_path: Any,
    *,
    buckets_by_stage: dict[str, list[int]],
) -> str:
    """Write a ``runtime_manifest.json`` + sibling ``diskgraph_manifest.json``
    carrying per-stage timestamp ``bucket_index`` lists, and return the
    runtime-manifest path. ``buckets_by_stage`` maps each stage to its list of
    per-bucket row counts; ``_grouped_global_batches`` reads these counts (one
    ``bucket_index`` entry per stage table) to derive the grouped batch count
    WITHOUT opening the wheel iterator. The stage table is named ``<stage>_seeds``
    (mirrors DG's stage_table_name)."""
    ext_root = tmp_path / "tt_ext"
    ext_root.mkdir(exist_ok=True)
    runtime_stages: dict[str, Any] = {}
    dg_stages: dict[str, Any] = {}
    for stage, counts in buckets_by_stage.items():
        table_name = f"{stage}_seeds"
        runtime_stages[stage] = {
            "table_name": table_name,
            "index_path": str(ext_root),
            "row_count": sum(counts),
        }
        offset = 0
        bucket_list = []
        for i, c in enumerate(counts):
            bucket_list.append({"ts": i, "offset": offset, "count": c})
            offset += c
        dg_stages[stage] = {
            "table_name": table_name,
            "bucket_index": {
                "path": f"{table_name}/bucket_index.bin",
                "buckets": bucket_list,
            },
        }
    runtime_path = ext_root / "runtime_manifest.json"
    runtime_path.write_text(json.dumps({"stages": runtime_stages}))
    (ext_root / "diskgraph_manifest.json").write_text(
        json.dumps({"task_dataset": {
            "stages": dg_stages
        }}))
    return str(runtime_path)


def _manifest(
    train_table_name: str = "train_table",
    *,
    tmp_path: Any | None = None,
    stage_counts: dict[str, int] | None = None,
    stage_seed_times: dict[str, tuple[int, int]] | None = None,
    runtime_manifest_path: str = "/tmp/runtime_manifest.json",
) -> DiskgraphTaskManifest:
    manifest_path = "/tmp/manifest.json"
    manifest: dict[str, Any] = {}
    if stage_counts is not None or stage_seed_times is not None:
        assert tmp_path is not None
        split_info_path = tmp_path / "split_info.json"
        split_info_path.write_text(
            json.dumps({
                "stage_counts": stage_counts or {},
                "stage_seed_times": stage_seed_times or {},
            }))
        manifest_path = str(tmp_path / "manifest.json")
        manifest = {
            "artifact_root": str(tmp_path),
            "split_info_path": "split_info.json",
        }
    return DiskgraphTaskManifest(
        manifest_path=manifest_path,
        runtime_manifest_path=runtime_manifest_path,
        manifest=manifest,
        metadata={
            "train_table_name": train_table_name,
            "seed_node_type": "User",
            "task_type": "binary_classification",
            "train_table_columns": {
                "seed_id_column": "ENTITY",
                "target_column": "TARGET",
                "time_column": "TIMESTAMP",
                "weight_column": "WEIGHT",
            },
        },
        feature_schema=_feature_schema(),
        graph_runtime_metadata={
            "node_types": ["User"],
            "edge_types": []
        },
    )


def _runtime_batch(input_ids: list[int] | None = None) -> dict[str, Any]:
    input_ids = input_ids or [0, 1, 2]
    num_rows = len(input_ids)
    return {
        "task": {
            "input_ids":
            pa.array(input_ids, type=pa.uint64()),
            "tt_stage_node_ids":
            pa.array([10 + i for i in input_ids], type=pa.uint64()),
            "seed_node_ids":
            pa.array([100 + i for i in input_ids], type=pa.uint64()),
            "target":
            pa.array([float((i + 1) % 2) for i in input_ids],
                     type=pa.float64()),
            "weight":
            pa.array([1.0] * num_rows, type=pa.float64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array([100 + i for i in input_ids],
                                     type=pa.uint64())
                },
                "batch_dict": {
                    "User": pa.array(list(range(num_rows)), type=pa.uint32())
                },
                "edge_index_dict": [],
                "num_sampled_nodes_dict": {
                    "User": pa.array([num_rows], type=pa.uint64())
                },
                "num_sampled_edges_dict": [],
            }
        },
        "raw_features": {},
    }


class _FakeTaskIterator:
    def __init__(
        self,
        num_batches: int = 1,
        batch_input_ids: list[list[int]] | None = None,
    ) -> None:
        self.num_batches = num_batches
        self.batch_input_ids = batch_input_ids

    def __iter__(self):
        if self.batch_input_ids is not None:
            for input_ids in self.batch_input_ids:
                yield _runtime_batch(input_ids)
            return
        for _ in range(self.num_batches):
            yield _runtime_batch()


def _backpressure_runtime_iter(*, worker_id: int, **_kwargs: Any):
    """Yield large prediction batches until the parent stops requesting them."""
    payload = "x" * (1024 * 1024)

    def _batches():
        while True:
            batch = _runtime_batch([worker_id])
            batch["raw_features"] = {
                "User": {
                    "record_batches": [
                        pa.RecordBatch.from_pydict({
                            "pk": [payload],
                            "age": [18.0],
                        })
                    ],
                    "node_ids":
                    pa.array([100 + worker_id], type=pa.uint64()),
                }
            }
            yield batch

    return _batches(), 0


class _BackpressureTaskBatchIterable(DiskgraphTaskBatchIterable):
    """Feed enough non-tensor data to block the worker result pipe."""
    def _runtime_index(self) -> Any:
        """Return a process-local backpressure runtime stub."""
        return types.SimpleNamespace(iter=_backpressure_runtime_iter)


def _run_shutdown_probe_consumer(ready_fd: int, release_fd: int) -> None:
    """Start workers, fill their result queue, then exit without draining it."""
    try:
        dldm.DiskgraphTaskBatchIterable = _BackpressureTaskBatchIterable
        dm = DiskgraphLightningDataModule(
            _manifest(),
            _mapper_bank(),
            sampling_plan={"layers": []},
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=2,
            prefetch_factor=2,
        )
        dm._pk_col_dict = {"User": "pk"}
        dm._time_col_dict = {}
        probe_batch = next(_backpressure_runtime_iter(worker_id=0)[0])
        probe_result = dm._to_hetero(probe_batch)
        assert len(pickle.dumps(probe_result)) > 1024 * 1024
        iterator = iter(dm.predict_dataloader())
        torch_iterator = getattr(iterator, "_iterator", iterator)
        assert dm.prefetch_factor is not None
        expected_results = dm.num_workers * dm.prefetch_factor
        deadline = time.monotonic() + 10
        while torch_iterator._worker_result_queue.qsize() < expected_results:
            if time.monotonic() >= deadline:
                raise TimeoutError("workers did not fill the result queue")
            time.sleep(0.01)
        pids = [worker.pid for worker in torch_iterator._workers]
        os.write(ready_fd, json.dumps({"pids": pids}).encode())
        os.read(release_fd, 1)
    except BaseException as exc:
        os.write(ready_fd, json.dumps({"error": repr(exc)}).encode())
        os._exit(2)
    os._exit(0)


def _wait_for_processes(
    pids: list[int],
    timeout_seconds: float,
) -> list[psutil.Process]:
    """Return worker processes that remain alive after the bounded wait."""
    processes = []
    for pid in pids:
        try:
            processes.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + timeout_seconds
    while processes:
        alive = []
        for process in processes:
            try:
                if process.status() != psutil.STATUS_ZOMBIE:
                    alive.append(process)
            except psutil.NoSuchProcess:
                pass
        if not alive:
            return []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return alive
        psutil.wait_procs(alive, timeout=min(0.1, remaining))
        processes = alive
    return []


def _install_fake_diskgraph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batches_per_iterator: int = 1,
    batch_input_ids_by_epoch: dict[int, list[list[int]]] | None = None,
    batch_input_ids_by_stage: dict[str, list[list[int]]] | None = None,
) -> dict[str, Any]:
    calls: dict[str, Any] = {"history": []}

    def make_runtime_batch_iterator(*args: Any, **kwargs: Any):
        calls["args"] = args
        calls["kwargs"] = kwargs
        calls["history"].append({"args": args, "kwargs": kwargs})
        batch_input_ids = None
        if batch_input_ids_by_stage is not None:
            batch_input_ids = batch_input_ids_by_stage[kwargs["stage"]]
        elif batch_input_ids_by_epoch is not None:
            batch_input_ids = batch_input_ids_by_epoch[kwargs["epoch"]]
        return _FakeTaskIterator(batches_per_iterator, batch_input_ids), 3

    stub = types.ModuleType("diskgraph")
    stub.make_runtime_batch_iterator = make_runtime_batch_iterator  # type: ignore[attr-defined]

    class _StubRuntimeIndex:
        """Stands in for ``diskgraph.RuntimeIndex``.

        Delegates to the fake factory above so each test's expectations about
        Remembers what moved to open-time and puts it back.
        """
        def __init__(self, dataset_ref, rnd_seed=None, num_threads=None):
            self._ref = dataset_ref
            self._rnd_seed = rnd_seed
            self._num_threads = num_threads

        def iter(self, **kwargs: Any):
            kwargs.setdefault("rnd_seed", self._rnd_seed)
            if self._num_threads is not None:
                kwargs.setdefault("num_threads", self._num_threads)
            return make_runtime_batch_iterator(self._ref, **kwargs)

    stub.RuntimeIndex = _StubRuntimeIndex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diskgraph", stub)
    return calls


class _SmokeModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.step_epochs: list[int] = []
        self.step_batch_indices: list[int] = []
        self.seen_input_ids: list[int] = []

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        self.step_epochs.append(int(self.current_epoch))
        self.step_batch_indices.append(batch_idx)
        self.seen_input_ids.extend(batch["train_table"].input_id.tolist())
        return self.weight * 0.0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _fit_smoke_model(
    dm: DiskgraphLightningDataModule,
    *,
    max_epochs: int = 2,
) -> _SmokeModel:
    model = _SmokeModel()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[DiskgraphLightningDataModule.shuffle_stream_callback()],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    return model


def test_shuffle_stream_callback_is_picklable() -> None:
    """Regression test for the Python 3.14 forkserver crash: the dataloader
    spawns workers by pickling trainer/callback state, and a ``<locals>`` class
    cannot be pickled. The callback returned by ``shuffle_stream_callback`` must
    therefore be a module-level (top-level) class so ``pickle`` can resolve it,
    and it must round-trip and still drive the per-epoch behavior afterwards.
    """
    cb = DiskgraphLightningDataModule.shuffle_stream_callback()
    # A ``<locals>`` class raises PicklingError here pre-fix.
    restored = pickle.loads(pickle.dumps(cb))
    assert type(restored).__qualname__ == type(cb).__qualname__
    assert "<locals>" not in type(cb).__qualname__

    # The unpickled callback must still update the datamodule each epoch.
    seen: list[int] = []

    # Subclass so the callback's isinstance(dm, DiskgraphLightningDataModule)
    # gate (diskgraph_lightning_data.py:551) fires. Instantiate via __new__ to
    # skip the heavy real constructor without defining a base-init-skipping
    # __init__ (avoids the super-init-not-called lint).
    class _FakeDm(DiskgraphLightningDataModule):
        def set_shuffle_stream_id(self, shuffle_stream_id: int) -> None:
            seen.append(int(shuffle_stream_id))

    trainer = types.SimpleNamespace(datamodule=_FakeDm.__new__(_FakeDm),
                                    current_epoch=5)
    restored.on_train_epoch_start(trainer, None)
    assert seen == [5]


def test_shuffle_stream_callback_sets_epoch_on_datamodule() -> None:
    """The callback propagates ``trainer.current_epoch`` to the datamodule's
    shuffle stream on each ``on_train_epoch_start`` -- proving the module-scope
    hoist preserves the original closure behavior exactly."""
    cb = DiskgraphLightningDataModule.shuffle_stream_callback()
    seen: list[int] = []

    # Subclass for the isinstance gate; __new__ skips the heavy constructor
    # without a base-init-skipping __init__ (see the picklable test above).
    class _FakeDm(DiskgraphLightningDataModule):
        def set_shuffle_stream_id(self, shuffle_stream_id: int) -> None:
            seen.append(int(shuffle_stream_id))

    for epoch in (0, 1, 2):
        cb.on_train_epoch_start(
            types.SimpleNamespace(datamodule=_FakeDm.__new__(_FakeDm),
                                  current_epoch=epoch),
            None,
        )
    assert seen == [0, 1, 2]

    # Non-DG datamodules are ignored (no AttributeError, no-op).
    cb.on_train_epoch_start(
        types.SimpleNamespace(datamodule=object(), current_epoch=9), None)
    assert seen == [0, 1, 2]


def test_constructor_reads_train_table_name_and_schema() -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(train_table_name="my_train"),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    assert dm._train_table_name == "my_train"
    assert dm._feature_schema["node_types"]["User"]["mapper_key"] == "users_v1"


def test_split_info_bounds_drive_native_stage_times(tmp_path) -> None:
    from kumoapi.train import TrainingStage

    bounds = {
        'train': (10, 19),
        'val': (20, 29),
        'test': (30, 39),
    }
    dm = DiskgraphLightningDataModule(
        _manifest(
            tmp_path=tmp_path,
            stage_counts={stage: 5
                          for stage in bounds},
            stage_seed_times=bounds,
        ),
        _mapper_bank(),
        sampling_plan={'layers': []},
        train_batch_size=2,
        eval_batch_size=1,
    )

    assert dm._stage_time_dict == {
        TrainingStage.TRAIN: bounds['train'],
        TrainingStage.VAL: bounds['val'],
        TrainingStage.TEST: bounds['test'],
    }


def test_to_hetero_threads_target_encoder(
        monkeypatch: pytest.MonkeyPatch) -> None:
    target_encoder = object()
    captured: dict[str, Any] = {}

    def _fake_runtime_batch_to_hetero(*_args: Any, **kwargs: Any):
        captured.update(kwargs)
        return HeteroData(), {}, {}

    monkeypatch.setattr(dldm, "runtime_batch_to_hetero",
                        _fake_runtime_batch_to_hetero)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        target_encoder=target_encoder,
    )

    dm._to_hetero(_runtime_batch())

    assert captured["target_encoder"] is target_encoder


def test_constructor_rejects_missing_train_table_name() -> None:
    with pytest.raises(ValueError, match="train_table_name"):
        DiskgraphLightningDataModule(
            _manifest(train_table_name=""),
            _mapper_bank(),
            sampling_plan={"layers": []},
            train_batch_size=2,
            eval_batch_size=1,
        )


def test_set_shuffle_stream_id_propagates_to_task_iterator(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    """Per diskgraph PR #266, ``epoch`` is a first-class kwarg to
    ``make_runtime_batch_iterator``. The wheel mixes
    ``(rnd_seed, epoch, stage_table_name)`` into the shuffle RNG
    internally, so ``rnd_seed`` is passed through unmodified."""
    calls = _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        base_seed=10,
    )
    dm.set_shuffle_stream_id(7)
    list(dm.train_dataloader())
    assert calls["kwargs"]["stage"] == "train"
    # rnd_seed is the unmodified base_seed; epoch is passed separately.
    assert calls["kwargs"]["rnd_seed"] == 10
    assert calls["kwargs"]["epoch"] == 7


def test_runtime_worker_shard_composes_with_torch_worker(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=2, id=1),
    )
    iterable = DiskgraphTaskBatchIterable(
        manifest_path="/tmp/runtime_manifest.json",
        stage="predict",
        sampling_plan={"layers": []},
        batch_size=4,
        shuffle=False,
        drop_last=False,
        base_seed=0,
        shuffle_state=types.SimpleNamespace(stream_id=0),
        num_seeds=20,
        runtime_num_workers=3,
        runtime_worker_id=1,
        dataloader_num_workers=2,
    )

    list(iterable)

    assert calls["kwargs"]["num_workers"] == 6
    assert calls["kwargs"]["worker_id"] == 4
    assert len(iterable) == 2


def test_train_dataloader_len_honors_max_steps_per_epoch(tmp_path, ) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        max_steps_per_epoch=3,
    )
    assert len(dm.train_dataloader()) == 3


def test_train_dataloader_iteration_stops_at_max_steps_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch, batches_per_iterator=10)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        max_steps_per_epoch=3,
    )
    assert len(list(dm.train_dataloader())) == 3
    assert len(calls["history"]) == 1


def test_lightning_fit_reopens_train_iterator_with_new_shuffle_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_epoch={
            0: [[0, 1]],
            1: [[10, 11]],
        },
    )
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 2}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        max_steps_per_epoch=1,
    )

    model = _fit_smoke_model(dm)

    assert [call["kwargs"]["epoch"] for call in calls["history"]] == [0, 1]
    assert model.seen_input_ids == [0, 1, 10, 11]


def test_lightning_epoch_ends_after_configured_train_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch, batches_per_iterator=10)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        max_steps_per_epoch=3,
    )

    model = _fit_smoke_model(dm)

    assert [call["kwargs"]["epoch"] for call in calls["history"]] == [0, 1]
    assert model.step_epochs == [0, 0, 0, 1, 1, 1]
    assert model.step_batch_indices == [0, 1, 2, 0, 1, 2]


def test_epoch_cap_does_not_change_wheel_training_population(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch, batches_per_iterator=10)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        max_steps_per_epoch=3,
    )

    model = _fit_smoke_model(dm, max_epochs=1)

    assert len(model.step_epochs) == 3
    assert calls["history"][0]["kwargs"]["stage"] == "train"
    assert calls["history"][0]["kwargs"]["batch_size"] == 10
    assert "max_batches" not in calls["history"][0]["kwargs"]
    assert "num_samples" not in calls["history"][0]["kwargs"]
    assert "max_num_samples" not in calls["history"][0]["kwargs"]


def test_train_dataloader_len_honors_min_steps_per_epoch(tmp_path, ) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 2}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        min_steps_per_epoch=3,
    )
    assert len(dm.train_dataloader()) == 3


def test_train_iter_replays_wheel_when_natural_lt_max_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Regression: ``__iter__`` must deliver ``__len__``-many batches.

    ``_max_train_batches`` returns ``max(natural, min_steps_per_epoch)``,
    so ``__len__`` can exceed the wheel iterator's natural per-worker
    yield. PyTorch Lightning trusts ``__len__`` to schedule end-of-epoch
    validation (``val_check_batch = __len__ * val_check_interval``); if
    ``__iter__`` yields fewer batches than advertised, the train loop
    StopIterates before the val trigger fires, validation is silently
    skipped, and ``EarlyStopping(monitor='val_auroc')`` crashes at
    ``on_train_epoch_end`` because no ``val_*`` metric was ever logged.

    The fix replays the wheel iterator (with a perturbed shuffle epoch
    each time) until the per-worker share of ``max_batches`` has been
    delivered. This mirrors what legacy ``LightningNodeData`` +
    sampler-with-replacement does for small datasets.
    """
    # 20 seeds, batch_size=10 → natural=2. min_steps_per_epoch=5 forces
    # __len__=5. The wheel yields 2 batches per open (matching natural),
    # so __iter__ must reopen 3 times to deliver 5 batches. num_workers=0
    # → DataLoader main-process path → per-worker target = __len__ = 5.
    calls = _install_fake_diskgraph(monkeypatch, batches_per_iterator=2)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        min_steps_per_epoch=5,
    )
    # Advertised length.
    assert len(dm.train_dataloader()) == 5
    # Iterator actually delivers the advertised count, not the natural 2.
    batches = list(dm.train_dataloader())
    assert len(batches) == 5
    # Reopens used to satisfy the contract: ceil(5 / batches_per_open=2) = 3.
    assert len(calls["history"]) == 3
    # Each reopen uses a distinct shuffle ``epoch`` so the iteration
    # order changes rather than repeating verbatim.
    epochs = [call["kwargs"]["epoch"] for call in calls["history"]]
    assert epochs == [0, 1, 2]


def test_val_dataloader_does_not_replay_wheel_iterator(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    """Replay is train-only.

    ``val``/``test``/``predict`` pass ``max_batches=None`` (the
    iterable yields the natural shard once and stops). The fix for
    ``__len__`` ↔ ``__iter__`` consistency must not over-iterate the
    eval stages, which is what allows PL's eval loop to terminate
    cleanly. This regression test pins that contract.
    """
    calls = _install_fake_diskgraph(monkeypatch, batches_per_iterator=2)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    batches = list(dm.val_dataloader())
    assert len(batches) == 2
    assert len(calls["history"]) == 1


def test_val_dataloader_honors_max_val_steps_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch, batches_per_iterator=10)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"val": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
        max_val_steps=3,
    )

    assert len(dm.val_dataloader()) == 3
    batches = list(dm.val_dataloader())

    assert len(batches) == 3
    assert len(calls["history"]) == 1


def test_predict_dataloader_len_counts_local_worker_shards(tmp_path) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"predict": 199444}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=1024,
        eval_batch_size=1024,
        num_workers=16,
    )

    assert len(dm.predict_dataloader()) == 208


def test_train_dataloader_chains_refit_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_stage={
            "train": [[0]],
            "val": [[10]],
        },
    )
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={
            "train": 1,
            "val": 1
        }),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=1,
        eval_batch_size=1,
        train_stages=("train", "val"),
    )

    batches = list(dm.train_dataloader())

    assert [call["kwargs"]["stage"] for call in calls["history"]] == [
        "train",
        "val",
    ]
    assert [b["train_table"].input_id.item() for b in batches] == [0, 10]


def test_capped_refit_interleaves_all_training_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_stage={
            "train": [[i] for i in range(20)],
            "val": [[100 + i] for i in range(20)],
        },
    )
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={
            "train": 100,
            "val": 10,
        }),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=1,
        eval_batch_size=1,
        max_steps_per_epoch=10,
        train_stages=("train", "val"),
    )

    batches = list(dm.train_dataloader().dataset)
    seen = [batch["task"]["input_ids"][0].as_py() for batch in batches]

    assert len(batches) == 10
    assert sum(value < 100 for value in seen) == 9
    assert sum(value >= 100 for value in seen) == 1
    assert seen.index(100) < len(seen) - 1
    assert {call["kwargs"]["stage"]
            for call in calls["history"]} == {
                "train",
                "val",
            }


def test_runtime_options_are_passed_to_wheel_iterator(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        runtime_options={
            "replace": True,
            "disjoint": False,
            "adaptive": True
        },
    )

    list(dm.train_dataloader())

    assert calls["kwargs"]["replace"] is True
    assert calls["kwargs"]["disjoint"] is False
    assert calls["kwargs"]["adaptive"] is True


def test_binary_weighted_seed_sampling_sets_train_epoch_length_and_runtime_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "kind": "diskgraph_task_dataset",
            "graph": {
                "index_path": "/idx"
            },
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 100,
                    "binary_label_index": {
                        "false_node_ids": "train_seeds/false_node_ids.bin",
                        "false_count": 90,
                        "true_node_ids": "train_seeds/true_node_ids.bin",
                        "true_count": 10,
                    },
                },
            },
        }))
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100},
                  runtime_manifest_path=str(runtime_path)),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=16,
        eval_batch_size=16,
        max_steps_per_epoch=4,
        min_steps_per_epoch=1,
        weighted_seed_sampling={
            "false_weight": 1.0,
            "true_weight": 9.0,
        },
    )

    loader = dm.train_dataloader()
    assert len(loader) == 4

    list(loader)

    assert calls["kwargs"]["weighted_seed_sampling"] == {
        "false_weight": 1.0,
        "true_weight": 9.0,
        "num_samples": 64,
    }


def test_weighted_seed_sampling_allocates_across_refit_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_stage={
            "train": [[i] for i in range(3)],
            "val": [[100]],
        },
    )
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 100,
                    "binary_label_index": {
                        "false_node_ids": "train_seeds/false.bin",
                        "false_count": 90,
                        "true_node_ids": "train_seeds/true.bin",
                        "true_count": 10,
                    },
                },
                "val": {
                    "table_name": "val_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 10,
                    "binary_label_index": {
                        "false_node_ids": "val_seeds/false.bin",
                        "false_count": 9,
                        "true_node_ids": "val_seeds/true.bin",
                        "true_count": 1,
                    },
                },
            },
        }))
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={
            "train": 100,
            "val": 10,
        }, runtime_manifest_path=str(runtime_path)),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=16,
        eval_batch_size=16,
        max_steps_per_epoch=4,
        train_stages=("train", "val"),
        weighted_seed_sampling={
            "false_weight": 1.0,
            "true_weight": 9.0,
        },
    )

    batches = list(dm.train_dataloader().dataset)
    options_by_stage = {
        call["kwargs"]["stage"]: call["kwargs"]["weighted_seed_sampling"]
        for call in calls["history"]
    }

    assert len(batches) == 4
    assert options_by_stage["train"]["num_samples"] == 48
    assert options_by_stage["val"]["num_samples"] == 16
    assert options_by_stage["train"]["false_weight"] == 1.0
    assert options_by_stage["val"]["true_weight"] == 9.0


def test_weighted_seed_sampling_validates_every_refit_stage(tmp_path, ) -> None:
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "stages": {
                "train": {
                    "binary_label_index": {
                        "false_count": 90,
                        "true_count": 10,
                    },
                },
                "val": {},
            },
        }))

    with pytest.raises(ValueError, match="binary_label_index for stage 'val'"):
        DiskgraphLightningDataModule(
            _manifest(tmp_path=tmp_path, stage_counts={
                "train": 100,
                "val": 10,
            }, runtime_manifest_path=str(runtime_path)),
            _mapper_bank(),
            sampling_plan={"layers": []},
            train_batch_size=16,
            eval_batch_size=16,
            train_stages=("train", "val"),
            weighted_seed_sampling={
                "false_weight": 1.0,
                "true_weight": 9.0,
            },
        )


def test_binary_weighted_seed_sampling_rejects_drop_last_train(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_fake_diskgraph(monkeypatch)
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "kind": "diskgraph_task_dataset",
            "graph": {
                "index_path": "/idx"
            },
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 100,
                    "binary_label_index": {
                        "false_node_ids": "train_seeds/false_node_ids.bin",
                        "false_count": 90,
                        "true_node_ids": "train_seeds/true_node_ids.bin",
                        "true_count": 10,
                    },
                },
            },
        }))
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100},
                  runtime_manifest_path=str(runtime_path)),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=16,
        eval_batch_size=16,
        drop_last_train=True,
        weighted_seed_sampling={
            "false_weight": 1.0,
            "true_weight": 9.0,
        },
    )

    with pytest.raises(ValueError, match="drop_last_train=True"):
        dm.train_dataloader()


def test_positive_row_weight_mass_covers_every_refit_stage(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    monkeypatch.setattr(dldm, "_sample_weight_index_for_stage",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(dldm, "_bucket_total_weight",
                        lambda *_args, **_kwargs: 0.1)
    dm = object.__new__(DiskgraphLightningDataModule)
    dm.task_manifest = types.SimpleNamespace(runtime_manifest_path="unused")

    assert dm._weighted_seed_mass(stage="val",
                                  config={"use_sample_weights": True}) == 1.0


@pytest.mark.parametrize("weight", [float("nan"), float("inf")])
def test_nonfinite_row_weight_mass_is_rejected(monkeypatch: pytest.MonkeyPatch,
                                               weight: float) -> None:
    monkeypatch.setattr(dldm, "_sample_weight_index_for_stage",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(dldm, "_bucket_total_weight",
                        lambda *_args, **_kwargs: weight)
    dm = object.__new__(DiskgraphLightningDataModule)
    dm.task_manifest = types.SimpleNamespace(runtime_manifest_path="unused")

    with pytest.raises(ValueError, match="non-finite weighted seed mass"):
        dm._weighted_seed_mass(stage="val", config={"use_sample_weights": True})


def test_nonfinite_class_weight_mass_is_rejected(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    monkeypatch.setattr(
        dldm,
        "_binary_label_index_for_stage",
        lambda *_args, **_kwargs: {
            "false_count": 90,
            "true_count": 10
        },
    )
    dm = object.__new__(DiskgraphLightningDataModule)
    dm.task_manifest = types.SimpleNamespace(runtime_manifest_path="unused")

    with pytest.raises(ValueError, match="non-finite weighted seed mass"):
        dm._weighted_seed_mass(stage="val",
                               config={"true_weight": float("nan")})


def test_weighted_seed_sampling_uses_row_weight_mass_for_epoch_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "kind": "diskgraph_task_dataset",
            "graph": {
                "index_path": "/idx"
            },
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 3,
                    "sample_weight_index": {
                        "all": {
                            "node_ids":
                            "train_seeds/sample_weight_node_ids.bin",
                            "cdf": "train_seeds/sample_weight_cdf_f64.bin",
                            "count": 3,
                            "total_weight": 6.5,
                        },
                    },
                },
            },
        }))
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 3},
                  runtime_manifest_path=str(runtime_path)),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=4,
        eval_batch_size=4,
        max_steps_per_epoch=4,
        min_steps_per_epoch=1,
        weighted_seed_sampling={"use_sample_weights": True},
    )

    list(dm.train_dataloader())

    assert calls["kwargs"]["weighted_seed_sampling"] == {
        "use_sample_weights": True,
        "num_samples": 8,
    }


def test_weighted_seed_sampling_combines_row_and_class_weight_mass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "kind": "diskgraph_task_dataset",
            "graph": {
                "index_path": "/idx"
            },
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 3,
                    "sample_weight_index": {
                        "all": {
                            "node_ids": "train_seeds/all.bin",
                            "cdf": "train_seeds/all.cdf",
                            "count": 3,
                            "total_weight": 7.0,
                        },
                        "false_bucket": {
                            "node_ids": "train_seeds/false.bin",
                            "cdf": "train_seeds/false.cdf",
                            "count": 2,
                            "total_weight": 3.0,
                        },
                        "true_bucket": {
                            "node_ids": "train_seeds/true.bin",
                            "cdf": "train_seeds/true.cdf",
                            "count": 1,
                            "total_weight": 4.0,
                        },
                    },
                },
            },
        }))
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 3},
                  runtime_manifest_path=str(runtime_path)),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=4,
        eval_batch_size=4,
        max_steps_per_epoch=10,
        min_steps_per_epoch=1,
        weighted_seed_sampling={
            "use_sample_weights": True,
            "false_weight": 1.0,
            "true_weight": 2.0,
        },
    )

    list(dm.train_dataloader())

    assert calls["kwargs"]["weighted_seed_sampling"] == {
        "use_sample_weights": True,
        "false_weight": 1.0,
        "true_weight": 2.0,
        "num_samples": 12,
    }


@pytest.mark.parametrize("stage,method_name", [
    ("train", "train_dataloader"),
    ("val", "val_dataloader"),
    ("test", "test_dataloader"),
    ("predict", "predict_dataloader"),
])
def test_each_stage_method_passes_correct_stage(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    method_name: str,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    _ = list(getattr(dm, method_name)())
    assert calls["kwargs"]["stage"] == stage


def test_dataloader_yields_hetero_from_task_batch_iterator(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    batches = list(dm.train_dataloader())
    assert len(batches) == 1
    hetero = batches[0]
    assert isinstance(hetero, HeteroData)
    assert torch.equal(hetero["User"].n_id, torch.tensor([100, 101, 102]))
    seed = hetero["train_table"]
    assert torch.equal(seed.input_id, torch.tensor([0, 1, 2]))
    assert torch.equal(seed.y, torch.tensor([[1], [0], [1]]))
    assert hasattr(hetero["User"], "tf")


def test_explain_selection_capture_returns_stage_node_ids() -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
    )

    dm.begin_explain_selection_capture("test")
    dm._to_hetero(_runtime_batch([0, 1]))
    dm._to_hetero(_runtime_batch([2]))

    input_type, input_nodes, input_time, input_id = dm.stage_inputs("test")

    assert input_type == "User"
    assert input_time is None
    assert torch.equal(input_nodes, torch.tensor([100, 101, 102]))
    assert torch.equal(input_id, torch.tensor([10, 11, 12]))


def test_stage_inputs_requires_explain_selection_capture() -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
    )

    with pytest.raises(RuntimeError, match="explain selection capture"):
        dm.stage_inputs("test")


def test_explain_dataloader_uses_selected_stage_node_ids(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=3,
        persistent_workers=True,
    )

    loader = dm.explain_dataloader("test", torch.tensor([10, 12]))
    batches = list(loader)

    assert loader.num_workers == 0
    assert batches and all(batch is not None for batch in batches)
    assert calls["kwargs"]["selected_stage_node_ids"] == [10, 12]
    assert calls["kwargs"]["shuffle"] is False
    assert calls["kwargs"]["drop_last"] is False
    assert calls["kwargs"]["num_workers"] == 1
    assert calls["kwargs"]["worker_id"] == 0


def test_explain_dataloader_maps_pred_stage_to_predict_runtime_stage(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
    )

    list(dm.explain_dataloader("pred", torch.tensor([10, 12])))

    assert calls["kwargs"]["stage"] == "predict"


def test_explain_dataloader_skips_lp_target_resolution_for_pred(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
        resolve_entity_node_type="User",
        resolve_target_node_type="Article",
    )

    list(dm.explain_dataloader("test", torch.tensor([10, 12])))
    assert calls["kwargs"]["entity_node_type"] == "User"
    assert calls["kwargs"]["target_node_type"] == "Article"

    list(dm.explain_dataloader("pred", torch.tensor([10, 12])))
    assert "entity_node_type" not in calls["kwargs"]
    assert "target_node_type" not in calls["kwargs"]


def test_explain_dataloader_link_prediction_pred_uses_predict_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls = _install_fake_diskgraph(monkeypatch)
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=4, predict_time_ns=0),
        eval_batch_size=2,
    )

    list(dm.explain_dataloader("pred", torch.tensor([10, 12])))

    assert calls["kwargs"]["stage"] == "predict"
    assert "selected_stage_node_ids" not in calls["kwargs"]


def test_explain_selection_capture_forces_single_worker_eval_loader() -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=4,
        persistent_workers=True,
    )

    dm.begin_explain_selection_capture("test")
    loader = dm.test_dataloader()

    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.dataset.dataloader_num_workers == 1


def test_pred_explain_selection_capture_forces_single_worker_predict_loader(
) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=4,
        persistent_workers=True,
    )

    dm.begin_explain_selection_capture("pred")
    loader = dm.predict_dataloader()

    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.dataset.dataloader_num_workers == 1


def test_dataloader_passes_through_loader_kwargs(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    loader = dm.train_dataloader()
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.persistent_workers is True


def test_dataloader_passes_through_prefetch_factor(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        num_workers=2,
        prefetch_factor=4,
    )
    assert dm.train_dataloader().prefetch_factor == 4


def _worker_pool_loader(
    registry: dldm._WorkerPoolRegistry,
    *,
    phase: str,
    advertised_batches: int,
    persistent_workers: bool = False,
) -> DataLoader:
    return dldm._WorkerPoolDataLoader(
        _WorkerProbeIterable(advertised_batches),
        phase=phase,
        registry=registry,
        batch_size=None,
        num_workers=2,
        multiprocessing_context='fork',
        persistent_workers=persistent_workers,
        prefetch_factor=1,
    )


def _assert_processes_gone(pids: tuple[int, ...]) -> None:
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


class _LifecycleProbeLoader:
    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator


class _LifecycleProbeIterator:
    def __init__(self, shutdown: Any = None) -> None:
        self._workers = (types.SimpleNamespace(pid=101), )
        self._batches = iter(['final-batch'])
        self._shutdown = shutdown or (lambda: None)

    def __next__(self) -> Any:
        return next(self._batches)

    def _shutdown_workers(self) -> None:
        self._shutdown()


class _LifecycleProbeHandle:
    def __init__(self) -> None:
        self.pid = 101
        self.process = types.SimpleNamespace(join=lambda timeout: None)
        self.alive = True
        self.close_calls = 0
        self.signals = []

    def exited(self) -> bool:
        return not self.alive

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.alive = False

    def close(self) -> None:
        self.close_calls += 1

    @property
    def closed(self) -> bool:
        return self.close_calls > 0


class _EmergencyWorker:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.sentinel = pid
        self.alive = True
        self.signals = []
        self.join_calls = []

    def terminate(self) -> None:
        self.signals.append(signal.SIGTERM)

    def kill(self) -> None:
        self.signals.append(signal.SIGKILL)
        self.alive = False

    def join(self, timeout: float) -> None:
        self.join_calls.append(timeout)


def _lifecycle_probe_worker_pool(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shutdown: Any = None,
) -> tuple[dldm._WorkerPoolRegistry, dldm._WorkerPoolIterator,
           _LifecycleProbeHandle]:
    registry = dldm._WorkerPoolRegistry()
    handle = _LifecycleProbeHandle()
    private_iterator = _LifecycleProbeIterator(
        shutdown or (lambda: setattr(handle, 'alive', False)))
    loader = _LifecycleProbeLoader(private_iterator)
    monkeypatch.setattr(dldm._WorkerHandle, 'open', lambda _process: handle)
    iterator = dldm._WorkerPoolIterator(
        private_iterator,
        loader=loader,
        registry=registry,
        phase='predict',
    )
    return registry, iterator, handle


def test_worker_handle_open_closes_duplicated_sentinel_when_pidfd_fails(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_read, sentinel_write = os.pipe()
    duplicates = []
    original_dup = os.dup

    def _record_dup(fd: int) -> int:
        duplicate = original_dup(fd)
        duplicates.append(duplicate)
        return duplicate

    monkeypatch.setattr(os, 'dup', _record_dup)
    monkeypatch.setattr(
        dldm, '_pidfd_open', lambda _pid:
        (_ for _ in ()).throw(OSError('synthetic pidfd failure')))
    process = types.SimpleNamespace(pid=os.getpid(), sentinel=sentinel_read)
    try:
        with pytest.raises(OSError, match='synthetic pidfd failure'):
            dldm._WorkerHandle.open(process)
        assert len(duplicates) == 1
        with pytest.raises(OSError):
            os.fstat(duplicates[0])
    finally:
        os.close(sentinel_read)
        os.close(sentinel_write)


@pytest.mark.parametrize('fail_at', [0, 1])
def test_worker_handle_capture_failure_keeps_registered_poison(
    monkeypatch: pytest.MonkeyPatch,
    fail_at: int,
) -> None:
    registry = dldm._WorkerPoolRegistry()
    private_iterator = _LifecycleProbeIterator()
    private_iterator._workers = (
        types.SimpleNamespace(pid=101),
        types.SimpleNamespace(pid=102),
    )
    loader = _LifecycleProbeLoader(private_iterator)
    handles = [_LifecycleProbeHandle(), _LifecycleProbeHandle()]
    calls = 0
    emergency_calls = []

    def _open(_process: Any) -> _LifecycleProbeHandle:
        nonlocal calls
        index = calls
        calls += 1
        if index == fail_at:
            raise OSError(f'synthetic capture failure {fail_at}')
        return handles[index]

    monkeypatch.setattr(dldm._WorkerHandle, 'open', _open)
    monkeypatch.setattr(
        dldm._WorkerPoolIterator,
        '_emergency_cleanup_after_initialization_failure',
        lambda _self: emergency_calls.append(True),
    )

    with pytest.raises(OSError, match=f'synthetic capture failure {fail_at}'):
        dldm._WorkerPoolIterator(
            private_iterator,
            loader=loader,
            registry=registry,
            phase='predict',
        )

    owner, = registry._matching(None)
    assert owner.state is dldm._WorkerPoolState.FAILED
    assert registry.active_phases == ('predict', )
    assert emergency_calls == [True]
    assert handles[0].close_calls == (1 if fail_at == 1 else 0)
    registry.discard(owner)


def test_worker_handle_capture_failure_runs_bounded_emergency_cleanup(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    workers = (_EmergencyWorker(101), _EmergencyWorker(102))
    workers_by_sentinel = {worker.sentinel: worker for worker in workers}
    private_iterator = _LifecycleProbeIterator()
    private_iterator._workers = workers
    loader = _LifecycleProbeLoader(private_iterator)
    captured_handle = _LifecycleProbeHandle()

    def _open(process: _EmergencyWorker) -> _LifecycleProbeHandle:
        if process is workers[1]:
            raise OSError('synthetic second-handle capture failure')
        return captured_handle

    monkeypatch.setattr(dldm._WorkerHandle, 'open', _open)
    monkeypatch.setattr(
        dldm._WorkerHandle, '_fd_ready',
        lambda sentinel: not workers_by_sentinel[sentinel].alive)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.001)
    monkeypatch.setattr(dldm, '_WORKER_SHUTDOWN_DEADLINE_SECONDS', 1.0)

    with pytest.raises(OSError,
                       match='synthetic second-handle capture failure'):
        dldm._WorkerPoolIterator(
            private_iterator,
            loader=loader,
            registry=registry,
            phase='predict',
        )

    owner, = registry._matching(None)
    assert owner.state is dldm._WorkerPoolState.FAILED
    assert registry.active_phases == ('predict', )
    assert [worker.signals for worker in workers] == [
        [signal.SIGTERM, signal.SIGKILL],
        [signal.SIGTERM, signal.SIGKILL],
    ]
    assert [worker.join_calls for worker in workers] == [[0], [0]]
    assert captured_handle.close_calls == 1
    registry.discard(owner)


def test_concurrent_close_during_handle_capture_publishes_atomically(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    handles = [_LifecycleProbeHandle(), _LifecycleProbeHandle()]
    handles[1].pid = 102
    private_iterator = _LifecycleProbeIterator(
        lambda: [setattr(handle, 'alive', False) for handle in handles])
    private_iterator._workers = (
        types.SimpleNamespace(pid=101),
        types.SimpleNamespace(pid=102),
    )
    loader = _LifecycleProbeLoader(private_iterator)
    second_capture_started = threading.Event()
    release_capture = threading.Event()
    calls = 0

    def _open(_process: Any) -> _LifecycleProbeHandle:
        nonlocal calls
        index = calls
        calls += 1
        if index == 1:
            second_capture_started.set()
            assert release_capture.wait(timeout=2)
        return handles[index]

    monkeypatch.setattr(dldm._WorkerHandle, 'open', _open)
    constructor_errors = []
    close_errors = []

    def _construct() -> None:
        try:
            dldm._WorkerPoolIterator(
                private_iterator,
                loader=loader,
                registry=registry,
                phase='predict',
            )
        except BaseException as error:
            constructor_errors.append(error)

    def _close() -> None:
        try:
            registry.shutdown('predict')
        except BaseException as error:
            close_errors.append(error)

    constructor = threading.Thread(target=_construct)
    constructor.start()
    assert second_capture_started.wait(timeout=2)
    owner, = registry._matching(None)
    assert owner.state is dldm._WorkerPoolState.INITIALIZING
    closer = threading.Thread(target=_close)
    closer.start()
    deadline = time.monotonic() + 2
    while not owner._close_started and time.monotonic() < deadline:
        time.sleep(0.001)
    assert owner._close_started
    release_capture.set()
    constructor.join(timeout=2)
    closer.join(timeout=2)

    assert not constructor.is_alive()
    assert not closer.is_alive()
    assert close_errors == []
    assert len(constructor_errors) == 1
    assert 'closed during initialization' in str(constructor_errors[0])
    assert owner.state is dldm._WorkerPoolState.QUIESCENT
    assert registry.active_phases == ()
    assert [handle.close_calls for handle in handles] == [1, 1]


def test_initialization_cannot_reactivate_after_close_deadline(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    handle = _LifecycleProbeHandle()
    worker = _EmergencyWorker(101)
    private_iterator = _LifecycleProbeIterator()
    private_iterator._workers = (worker, )
    loader = _LifecycleProbeLoader(private_iterator)
    capture_started = threading.Event()
    release_capture = threading.Event()

    def _open(_process: Any) -> _LifecycleProbeHandle:
        capture_started.set()
        assert release_capture.wait(timeout=2)
        return handle

    monkeypatch.setattr(dldm._WorkerHandle, 'open', _open)
    monkeypatch.setattr(dldm._WorkerHandle, '_fd_ready',
                        lambda _sentinel: not worker.alive)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.001)
    monkeypatch.setattr(dldm, '_WORKER_SHUTDOWN_DEADLINE_SECONDS', 0.05)
    constructor_errors = []

    def _construct() -> None:
        try:
            dldm._WorkerPoolIterator(
                private_iterator,
                loader=loader,
                registry=registry,
                phase='predict',
            )
        except BaseException as error:
            constructor_errors.append(error)

    constructor = threading.Thread(target=_construct)
    constructor.start()
    assert capture_started.wait(timeout=2)
    owner, = registry._matching(None)
    with pytest.raises(dldm.DataLoaderWorkerShutdownError,
                       match='initialization did not finish'):
        registry.shutdown('predict')
    release_capture.set()
    constructor.join(timeout=2)

    assert not constructor.is_alive()
    assert owner.state is dldm._WorkerPoolState.FAILED
    assert registry.active_phases == ('predict', )
    assert handle.close_calls == 1
    assert worker.signals == [signal.SIGTERM, signal.SIGKILL]
    assert worker.join_calls == [0]
    assert not worker.alive
    assert len(constructor_errors) == 1
    registry.discard(owner)


def test_super_iter_partial_failure_preserves_primary_and_reaps_owner(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    loader = _worker_pool_loader(registry, phase='predict',
                                 advertised_batches=1)
    handle = _LifecycleProbeHandle()
    private_iterator = _LifecycleProbeIterator(
        lambda: setattr(handle, 'alive', False))
    monkeypatch.setattr(dldm._WorkerHandle, 'open', lambda _process: handle)

    def _fail_super(data_loader: DataLoader) -> Any:
        data_loader._iterator = private_iterator
        raise ValueError('synthetic super iterator failure')

    monkeypatch.setattr(DataLoader, '__iter__', _fail_super)

    with pytest.raises(ValueError, match='synthetic super iterator failure'):
        iter(loader)

    assert registry.active_phases == ()
    assert loader._iterator is None
    assert handle.close_calls == 1


def test_final_batch_survives_private_shutdown_thread_construction_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry, iterator, handle = _lifecycle_probe_worker_pool(monkeypatch)
    original_thread = threading.Thread

    def _fail_private_construction(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get('name', '').endswith('pytorch-shutdown'):
            raise RuntimeError('synthetic private construction failure')
        return original_thread(*args, **kwargs)

    monkeypatch.setattr(threading, 'Thread', _fail_private_construction)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.0)

    assert next(iterator) == 'final-batch'
    assert iterator.state is dldm._WorkerPoolState.ACTIVE
    with pytest.raises(dldm.DataLoaderWorkerShutdownError,
                       match='synthetic private construction failure'):
        iterator.close()

    assert iterator.state is dldm._WorkerPoolState.FAILED
    assert registry.active_phases == ('predict', )
    registry.discard(iterator)
    assert handle.closed
    assert handle.close_calls == 1


def test_final_batch_survives_private_shutdown_thread_start_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry, iterator, handle = _lifecycle_probe_worker_pool(monkeypatch)
    original_start = threading.Thread.start

    def _fail_private_start(thread: threading.Thread) -> None:
        if thread.name.endswith('pytorch-shutdown'):
            raise RuntimeError('synthetic private start failure')
        original_start(thread)

    monkeypatch.setattr(threading.Thread, 'start', _fail_private_start)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.0)

    assert next(iterator) == 'final-batch'
    assert iterator.state is dldm._WorkerPoolState.ACTIVE
    with pytest.raises(dldm.DataLoaderWorkerShutdownError,
                       match='synthetic private start failure'):
        iterator.close()

    assert iterator.state is dldm._WorkerPoolState.FAILED
    assert registry.active_phases == ('predict', )
    assert handle.closed
    assert handle.close_calls == 1


def test_final_batch_survives_cleanup_polling_exception(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry, iterator, handle = _lifecycle_probe_worker_pool(monkeypatch)

    def _fail_poll() -> bool:
        raise OSError('synthetic pidfd polling failure')

    monkeypatch.setattr(handle, 'exited', _fail_poll)

    assert next(iterator) == 'final-batch'
    with pytest.raises(dldm.DataLoaderWorkerShutdownError,
                       match='synthetic pidfd polling failure'):
        iterator.close()

    assert iterator.state is dldm._WorkerPoolState.FAILED
    assert iterator.worker_pids == (handle.pid, )
    assert registry.active_phases == ('predict', )
    assert handle.closed
    assert handle.close_calls == 1


def test_final_batch_survives_cleanup_signaling_exception(
        monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    registry, iterator, handle = _lifecycle_probe_worker_pool(
        monkeypatch, shutdown=release.wait)

    def _fail_signal(_sig: int) -> None:
        raise RuntimeError('synthetic pidfd signaling failure')

    monkeypatch.setattr(handle, 'send_signal', _fail_signal)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_SHUTDOWN_DEADLINE_SECONDS', 1.0)

    try:
        assert next(iterator) == 'final-batch'
        with pytest.raises(dldm.DataLoaderWorkerShutdownError,
                           match='synthetic pidfd signaling failure'):
            iterator.close()
    finally:
        release.set()

    assert iterator.state is dldm._WorkerPoolState.FAILED
    assert registry.active_phases == ('predict', )
    assert handle.closed
    assert handle.close_calls == 1


def test_worker_escalation_signals_all_live_handles_in_one_pass(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    handles = [_LifecycleProbeHandle(), _LifecycleProbeHandle()]
    handles[1].pid = 102
    all_signaled = threading.Event()
    signal_calls = []

    def _shutdown() -> None:
        assert all_signaled.wait(timeout=1)

    private_iterator = _LifecycleProbeIterator(_shutdown)
    private_iterator._workers = (
        types.SimpleNamespace(pid=101),
        types.SimpleNamespace(pid=102),
    )
    loader = _LifecycleProbeLoader(private_iterator)
    opened = iter(handles)
    monkeypatch.setattr(dldm._WorkerHandle, 'open',
                        lambda _process: next(opened))

    for handle in handles:

        def _signal(sig: int, *,
                    target: _LifecycleProbeHandle = handle) -> None:
            signal_calls.append((target.pid, sig))
            target.alive = False
            if len(signal_calls) == len(handles):
                all_signaled.set()

        monkeypatch.setattr(handle, 'send_signal', _signal)

    iterator = dldm._WorkerPoolIterator(
        private_iterator,
        loader=loader,
        registry=registry,
        phase='predict',
    )
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 1.0)
    iterator.close()

    assert signal_calls == [(101, signal.SIGTERM), (102, signal.SIGTERM)]
    assert iterator.state is dldm._WorkerPoolState.QUIESCENT
    assert [handle.close_calls for handle in handles] == [1, 1]


@pytest.mark.parametrize('post_kill_polls', [0, 3])
def test_worker_escalation_kills_term_survivor_and_waits_for_exit(
    monkeypatch: pytest.MonkeyPatch,
    post_kill_polls: int,
) -> None:
    registry, iterator, handle = _lifecycle_probe_worker_pool(
        monkeypatch, shutdown=lambda: None)
    remaining_polls = post_kill_polls
    kill_sent = False

    def _exited() -> bool:
        nonlocal remaining_polls
        if kill_sent and handle.alive:
            if remaining_polls:
                remaining_polls -= 1
            else:
                handle.alive = False
        return not handle.alive

    def _signal(sig: int) -> None:
        nonlocal kill_sent
        handle.signals.append(sig)
        if sig == signal.SIGKILL:
            kill_sent = True

    monkeypatch.setattr(handle, 'exited', _exited)
    monkeypatch.setattr(handle, 'send_signal', _signal)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.0)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.001)
    monkeypatch.setattr(dldm, '_WORKER_SHUTDOWN_DEADLINE_SECONDS', 1.0)

    iterator.close()

    assert handle.signals == [signal.SIGTERM, signal.SIGKILL]
    assert remaining_polls == 0
    assert iterator.state is dldm._WorkerPoolState.QUIESCENT
    assert registry.active_phases == ()
    assert handle.close_calls == 1


def test_successful_final_batch_does_not_start_cleanup(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry, iterator, handle = _lifecycle_probe_worker_pool(monkeypatch)
    thread_names = []
    original_start = threading.Thread.start

    def _record_start(thread: threading.Thread) -> None:
        thread_names.append(thread.name)
        original_start(thread)

    monkeypatch.setattr(threading.Thread, 'start', _record_start)

    assert next(iterator) == 'final-batch'

    assert iterator.state is dldm._WorkerPoolState.ACTIVE
    assert handle.alive
    assert not handle.closed
    assert not any(name.endswith('pytorch-shutdown') for name in thread_names)
    iterator.close()
    assert iterator.state is dldm._WorkerPoolState.QUIESCENT
    assert registry.active_phases == ()
    assert handle.close_calls == 1


def test_worker_pool_shutdown_waits_for_phase_barrier() -> None:
    registry = dldm._WorkerPoolRegistry()
    loader = _worker_pool_loader(registry, phase='train', advertised_batches=2)
    iterator = iter(loader)
    assert isinstance(iterator, dldm._WorkerPoolIterator)

    next(iterator)
    worker_pids = registry.worker_pids
    assert len(worker_pids) == 2

    final_batch = next(iterator)

    assert final_batch is not None
    assert iterator.state is dldm._WorkerPoolState.ACTIVE
    assert registry.worker_pids == worker_pids
    iterator.close('test-phase-barrier')
    assert registry.worker_pids == ()
    _assert_processes_gone(worker_pids)


def test_worker_pool_escalates_when_private_shutdown_blocks(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    loader = _worker_pool_loader(registry, phase='train', advertised_batches=2)
    iterator = iter(loader)
    assert isinstance(iterator, dldm._WorkerPoolIterator)
    next(iterator)
    worker_pids = registry.worker_pids
    release = threading.Event()
    private_iterator = iterator._iterator
    assert private_iterator is not None
    monkeypatch.setattr(private_iterator, '_shutdown_workers',
                        lambda: release.wait(timeout=5))
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.01)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.01)
    monkeypatch.setattr(dldm, '_WORKER_SHUTDOWN_DEADLINE_SECONDS', 2.0)

    release_timer = threading.Timer(0.1, release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        assert next(iterator) is not None
        assert time.monotonic() - started < 1
        iterator.close()
    finally:
        release.set()
        release_timer.cancel()

    assert registry.worker_pids == ()
    _assert_processes_gone(worker_pids)


def test_worker_pool_uses_one_absolute_deadline_when_private_hook_never_returns(
        monkeypatch: pytest.MonkeyPatch) -> None:
    registry = dldm._WorkerPoolRegistry()
    loader = _worker_pool_loader(registry, phase='predict',
                                 advertised_batches=1)
    iterator = iter(loader)
    assert isinstance(iterator, dldm._WorkerPoolIterator)
    private_iterator = iterator._iterator
    assert private_iterator is not None
    worker_pids = registry.worker_pids
    assert len(worker_pids) == 2
    expected_signal_error = RuntimeError(
        f'DataLoader worker (pid {worker_pids[0]}) is killed by signal: '
        'Terminated. ')
    assert iterator._is_expected_worker_signal_error(expected_signal_error)
    assert not iterator._is_expected_worker_signal_error(
        RuntimeError('unexpected shutdown error'))
    assert not iterator._is_expected_worker_signal_error(
        RuntimeError(
            'DataLoader worker (pid 999999) is killed by signal: Terminated. '))
    release = threading.Event()
    monkeypatch.setattr(private_iterator, '_shutdown_workers', release.wait)
    monkeypatch.setattr(dldm, '_WORKER_COOPERATIVE_GRACE_SECONDS', 0.01)
    monkeypatch.setattr(dldm, '_WORKER_TERM_GRACE_SECONDS', 0.01)
    monkeypatch.setattr(dldm, '_WORKER_SHUTDOWN_DEADLINE_SECONDS', 0.2)
    original_signal = iterator._signal_live_workers
    signal_error_injected = False

    def _signal_and_report_worker_exit(sig: int) -> None:
        nonlocal signal_error_injected
        original_signal(sig)
        if not signal_error_injected:
            signal_error_injected = True
            raise expected_signal_error

    monkeypatch.setattr(iterator, '_signal_live_workers',
                        _signal_and_report_worker_exit)

    started = time.monotonic()
    try:
        final_batch = next(iterator)
        assert final_batch is not None
        with pytest.raises(dldm.DataLoaderWorkerShutdownError,
                           match='absolute deadline'):
            iterator.close()
    finally:
        release.set()

    assert time.monotonic() - started < 1
    assert signal_error_injected
    assert iterator.state is dldm._WorkerPoolState.FAILED


def test_train_pool_is_gone_before_validation_pool_starts() -> None:
    registry = dldm._WorkerPoolRegistry()
    train = _worker_pool_loader(registry, phase='train', advertised_batches=1)
    val = _worker_pool_loader(registry, phase='val', advertised_batches=2)

    train_iterator = iter(train)
    next(train_iterator)
    train_iterator.close()
    assert registry.worker_pids == ()

    val_iterator = iter(val)
    next(val_iterator)
    val_pids = registry.worker_pids
    assert len(val_pids) == 2
    next(val_iterator)
    val_iterator.close()

    assert registry.worker_pids == ()
    _assert_processes_gone(val_pids)


def test_new_phase_is_rejected_before_spawning_overlapping_workers() -> None:
    registry = dldm._WorkerPoolRegistry()
    train = _worker_pool_loader(registry, phase="train", advertised_batches=2)
    val = _worker_pool_loader(registry, phase="val", advertised_batches=1)
    train_iterator = iter(train)
    next(train_iterator)
    train_pids = registry.worker_pids

    with pytest.raises(RuntimeError, match="epoch-boundary validation"):
        iter(val)

    assert registry.worker_pids == train_pids
    registry.shutdown()
    _assert_processes_gone(train_pids)


def test_persistent_worker_loader_can_restart_after_explicit_close() -> None:
    registry = dldm._WorkerPoolRegistry()
    loader = _worker_pool_loader(
        registry,
        phase='train',
        advertised_batches=2,
        persistent_workers=True,
    )

    first_iterator = iter(loader)
    next(first_iterator)
    first_pids = registry.worker_pids
    assert len(first_pids) == 2
    next(first_iterator)
    first_iterator.close()
    _assert_processes_gone(first_pids)
    assert loader._iterator is None

    iterator = iter(loader)
    second_pids = registry.worker_pids
    assert len(second_pids) == 2
    next(iterator)
    next(iterator)
    iterator.close()

    assert registry.worker_pids == ()
    _assert_processes_gone(second_pids)


def test_worker_pool_callback_closes_truncated_and_failed_phases(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    phases = []
    monkeypatch.setattr(dm, 'shutdown_dataloader_workers', phases.append)
    callback = dm.worker_pool_lifecycle_callback()
    trainer = types.SimpleNamespace(datamodule=dm, is_last_batch=False)

    callback.on_train_batch_end(trainer, None, None, None, 0)
    trainer.is_last_batch = True
    callback.on_train_batch_end(trainer, None, None, None, 1)
    callback.on_validation_end(trainer, None)
    callback.on_exception(trainer, None, RuntimeError('failed'))

    assert phases == ['train', 'val', None]


def test_predict_callback_deferral_does_not_skip_writer_owned_barrier(
        monkeypatch: pytest.MonkeyPatch) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    events = []
    monkeypatch.setattr(
        dm._worker_pool_registry, 'shutdown',
        lambda phase, reason='phase-barrier': events.append(
            ('shutdown', phase)))
    callback = dm.worker_pool_lifecycle_callback()
    trainer = types.SimpleNamespace(datamodule=dm)

    with dm.defer_dataloader_worker_shutdown('predict'):
        callback.on_predict_end(trainer, None)
        assert events == []
        dm.shutdown_dataloader_workers('predict')

    assert events == [('shutdown', 'predict')]


def test_predict_worker_shutdown_deferral_is_nested_and_base_exception_safe(
) -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )

    with pytest.raises(KeyboardInterrupt, match='synthetic cancellation'):
        with dm.defer_dataloader_worker_shutdown('predict'):
            with dm.defer_dataloader_worker_shutdown('predict'):
                assert dm.is_dataloader_worker_shutdown_deferred('predict')
                raise KeyboardInterrupt('synthetic cancellation')

    assert not dm.is_dataloader_worker_shutdown_deferred('predict')


def test_worker_pool_callback_reaps_processes_on_exception() -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    loader = _worker_pool_loader(
        dm._worker_pool_registry,
        phase='train',
        advertised_batches=10,
    )
    iterator = iter(loader)
    next(iterator)
    worker_pids = dm.active_dataloader_worker_pids
    assert len(worker_pids) == 2

    trainer = types.SimpleNamespace(datamodule=dm)
    dm.worker_pool_lifecycle_callback().on_exception(trainer, None,
                                                     RuntimeError('failed'))

    assert dm.active_dataloader_worker_pids == ()
    _assert_processes_gone(worker_pids)


def test_worker_fetch_failure_reaps_processes() -> None:
    registry = dldm._WorkerPoolRegistry()
    loader = dldm._WorkerPoolDataLoader(
        _FailingWorkerIterable(),
        phase="train",
        registry=registry,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="fork",
        prefetch_factor=1,
    )
    iterator = iter(loader)
    worker_pids = iterator.worker_pids
    assert len(worker_pids) == 2

    with pytest.raises(RuntimeError, match="worker fetch failed"):
        next(iterator)

    assert registry.worker_pids == ()
    _assert_processes_gone(worker_pids)


def test_direct_lightning_fit_installs_callback_and_never_overlaps_pools(
) -> None:
    class ProbeDataModule(DiskgraphLightningDataModule):
        def __init__(self) -> None:
            super().__init__(
                _manifest(),
                _mapper_bank(),
                sampling_plan={"layers": []},
                train_batch_size=2,
                eval_batch_size=1,
            )
            self.train_loader = _worker_pool_loader(
                self._worker_pool_registry,
                phase='train',
                advertised_batches=2,
            )
            self.val_loader = _worker_pool_loader(
                self._worker_pool_registry,
                phase='val',
                advertised_batches=1,
            )

        def train_dataloader(self) -> DataLoader:
            return self.train_loader

        def val_dataloader(self) -> DataLoader:
            return self.val_loader

    class ProbeModel(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.first_batch_worker_counts = []
            self.training_batches = 0
            self.val_start_worker_counts = []

        def training_step(self, batch, batch_idx):
            self.training_batches += 1
            if batch_idx == 0:
                pids = self.trainer.datamodule.active_dataloader_worker_pids
                self.first_batch_worker_counts.append(len(pids))
            return self.weight * 0

        def validation_step(self, batch, batch_idx):
            return None

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def on_validation_start(self) -> None:
            pids = self.trainer.datamodule.active_dataloader_worker_pids
            self.val_start_worker_counts.append(len(pids))

    dm = ProbeDataModule()
    model = ProbeModel()
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, datamodule=dm)

    assert model.training_batches == 4
    assert model.first_batch_worker_counts == [2, 2]
    assert model.val_start_worker_counts == [2, 2, 2]
    assert dm.active_dataloader_worker_pids == ()
    assert sum(
        isinstance(callback, dldm._ShutdownWorkerPoolsCallback)
        for callback in trainer.callbacks) == 1


def test_setup_does_not_duplicate_explicit_worker_lifecycle_callback() -> None:
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
    )
    callback = dm.worker_pool_lifecycle_callback()
    dm.trainer = types.SimpleNamespace(callbacks=[callback])

    dm.setup('fit')

    assert dm.trainer.callbacks == [callback]


def test_prefetch_factor_is_not_set_without_workers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # PyTorch rejects prefetch_factor at num_workers == 0, so it must be
    # dropped rather than forwarded.
    _install_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=1,
        num_workers=0,
        prefetch_factor=4,
    )
    assert dm.train_dataloader().prefetch_factor is None


@pytest.mark.parametrize("bad", [0, -1])
def test_prefetch_factor_below_one_is_rejected_early(bad: int) -> None:
    # Caught at construction rather than when the DataLoader is built, so
    # the error names the config that is wrong.
    with pytest.raises(ValueError, match="prefetch_factor must be >= 1"):
        DiskgraphLightningDataModule(
            _manifest(),
            _mapper_bank(),
            sampling_plan={"layers": []},
            train_batch_size=2,
            eval_batch_size=1,
            num_workers=2,
            prefetch_factor=bad,
        )


def test_task_manifest_loads_sidecar_bundle_from_writer(tmp_path) -> None:
    """Round-trip: writer's sidecar bundle -> loader.

    Locks the cross-module contract between
    ``kumo_core.task_dataset`` (producer) and
    ``DiskgraphTaskManifest`` (consumer); drift on either side breaks
    this test before it can silently break training.
    """
    from kumo_core.task_dataset import (
        TaskDatasetMaterializeSpec,
        write_flat_manifest,
        write_thin_manifest,
    )

    spec = TaskDatasetMaterializeSpec(
        tt_manifest_path="/tmp/tt/manifest.json",
        job_id="job-test",
        artifact_root=str(tmp_path / "bundle"),
        task_family="node_prediction",
        task_type="binary_classification",
        is_temporal=False,
        seed_node_type="User",
        train_table_name="train_table",
        entity_table_name="User",
        feature_schema={
            "node_types": {
                "User": {
                    "feature_columns": ["AGE"],
                    "mapper_key":
                    "User",
                    "time_column":
                    None,
                    "column_specs": [{
                        "column_name": "AGE",
                        "dtype": "int64",
                        "stype": "numerical",
                        "encoder": {
                            "name": "Numerical"
                        },
                        "stats": {
                            "MEAN": 0.0
                        },
                    }],
                },
            },
            "target_encoder_input": None,
        },
        graph_node_types=("User", ),
        graph_edge_types=(),
    )
    manifest_path = write_thin_manifest(
        spec,
        graph_index_id="g/v1",
        graph_index_path="/idx",
        stats={},
        stage_counts={"train": 3},
        stage_table_refs={"train": {
            "table_name": "train_seeds"
        }},
        has_time=False,
        has_weight=False,
    )
    runtime_manifest_path = write_flat_manifest(
        spec,
        bundle_dir=str(tmp_path / "bundle"),
        graph_index_path="/idx",
        stage_index_path="/idx/train_tables/tt-test",
        stage_counts={"train": 3},
        stage_table_refs={
            "train": {
                "table_name": "train_seeds",
                "binary_label_index": {
                    "false_node_ids": "train_seeds/false_node_ids.bin",
                    "false_count": 2,
                    "true_node_ids": "train_seeds/true_node_ids.bin",
                    "true_count": 1,
                },
            }
        },
        has_time=False,
        has_weight=False,
    )

    manifest = DiskgraphTaskManifest.load(manifest_path)

    assert manifest.manifest_path == manifest_path
    assert manifest.runtime_manifest_path == runtime_manifest_path
    assert manifest.train_table_name == "train_table"
    assert manifest.seed_node_type == "User"
    assert manifest.task_type == "binary_classification"
    runtime_manifest = json.loads(open(runtime_manifest_path).read())
    assert runtime_manifest["stages"]["train"]["binary_label_index"] == {
        "false_node_ids": "train_seeds/false_node_ids.bin",
        "false_count": 2,
        "true_node_ids": "train_seeds/true_node_ids.bin",
        "true_count": 1,
    }
    # Sidecar carries post-bridge ENTITY / TARGET (legacy graphstore shape).
    assert manifest.train_table_columns["seed_id_column"] == "ENTITY"
    assert manifest.train_table_columns["target_column"] == "TARGET"
    assert manifest.graph_runtime_metadata["graph_index_path"] == "/idx"


# RHS MIPS dataloaders (two-tower deep LP eval) ###########################


def _rhs_lp_manifest(
    tmp_path: Any,
    *,
    num_rhs: int,
    val_time_ns: int | None = None,
    test_time_ns: int | None = None,
    predict_time_ns: int | None = None,
    target_node_type: str | None = None,
) -> DiskgraphTaskManifest:
    """LP manifest wired so ``rhs_*_dataloader`` resolves ``num_rhs`` (from a
    real ``index_metadata.json``) and per-stage seed times (from a real
    ``split_info.json``). ``target_node_type`` defaults to ``seed_node_type``
    (== ``"User"``); pass it explicitly to model the LHS != RHS topology where
    the RHS seed block is a distinct node type."""
    rhs_type = target_node_type or "User"
    index_dir = tmp_path / "graph_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index_metadata.json").write_text(
        json.dumps({
            "node_types": [{
                "name": rhs_type,
                "max_id": num_rhs - 1,
            }],
        }))

    manifest_obj: dict[str, Any] = {
        "artifact_root": str(tmp_path),
        "split_info_path": "split_info.json",
    }
    stage_seed_times: dict[str, list[int]] = {}
    if val_time_ns is not None:
        stage_seed_times["val"] = [val_time_ns, val_time_ns]
    if test_time_ns is not None:
        stage_seed_times["test"] = [test_time_ns, test_time_ns]
    if predict_time_ns is not None:
        stage_seed_times["predict"] = [predict_time_ns, predict_time_ns]
    (tmp_path / "split_info.json").write_text(
        json.dumps({
            "stage_counts": {
                "val": 5,
                "test": 5
            },
            "stage_seed_times": stage_seed_times,
        }))

    return DiskgraphTaskManifest(
        manifest_path=str(tmp_path / "manifest.json"),
        runtime_manifest_path="/tmp/runtime_manifest.json",
        manifest=manifest_obj,
        metadata={
            "train_table_name": "train_table",
            "seed_node_type": "User",
            "target_node_type": rhs_type,
            "task_type": "temporal_link_prediction",
            "train_table_columns": {
                "seed_id_column": "ENTITY",
                "target_column": "TARGET",
                "time_column": "TIMESTAMP",
            },
        },
        feature_schema=_feature_schema(),
        graph_runtime_metadata={
            "node_types": list({"User", rhs_type}),
            "edge_types": [],
            "graph_index_path": str(index_dir),
        },
    )


def _rhs_dm(
    manifest: DiskgraphTaskManifest,
    *,
    eval_batch_size: int = 2,
) -> DiskgraphLightningDataModule:
    return DiskgraphLightningDataModule(
        manifest,
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=2,
        eval_batch_size=eval_batch_size,
        task_type="temporal_link_prediction",
    )


def test_rhs_val_dataloader_noderange_kwargs_and_ordering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """rhs_val_dataloader seeds the target type over [0, num_rhs) in
    NodeRange mode at the val seed time (converted ns -> index seconds),
    single-worker, in-order, no drop_last / no replay."""
    calls = _install_fake_diskgraph(monkeypatch)
    # 2_000_000_000 ns == 2 s.
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=5, val_time_ns=2_000_000_000))

    loader = dm.rhs_val_dataloader()

    # __len__ covers all RHS nodes: ceil(5 / 2) == 3 batches, no drop_last.
    assert len(loader) == 3
    # MIPS-ordering contract enforced on the loader / iterable.
    assert loader.num_workers == 0
    assert loader.dataset.shuffle is False
    assert loader.dataset.drop_last is False
    assert loader.dataset.replay is False
    assert loader.dataset.num_seeds == 5

    # Drive one batch to capture the NodeRange kwargs threaded to the wheel.
    list(loader)
    kwargs = calls["kwargs"]
    assert kwargs["shuffle"] is False
    assert kwargs["drop_last"] is False
    assert kwargs["rhs_seed_node_type"] == "User"
    assert kwargs["rhs_seed_time"] == 2  # 2_000_000_000 ns -> 2 s
    # Single shard (num_workers=0 -> the iterable runs as 1 worker).
    assert kwargs["num_workers"] == 1
    assert kwargs["worker_id"] == 0


def test_rhs_val_dataloader_static_seed_time_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Static LP has no val/test seed-time range; NodeRange still requires a
    fixed seed time, so the RHS loader uses the static 0 sentinel."""
    dm = _rhs_dm(_rhs_lp_manifest(tmp_path, num_rhs=4))

    calls = _install_fake_diskgraph(monkeypatch)
    list(dm.rhs_val_dataloader())
    assert calls["kwargs"]["rhs_seed_time"] == 0


def test_rhs_test_and_predict_dataloader_seed_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """rhs_test uses test_time; rhs_predict falls back to test_time (no
    dedicated predict seed time is emitted)."""
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=4, val_time_ns=1_000_000_000,
                         test_time_ns=7_000_000_000))

    calls = _install_fake_diskgraph(monkeypatch)
    list(dm.rhs_test_dataloader())
    assert calls["kwargs"]["rhs_seed_time"] == 7

    calls = _install_fake_diskgraph(monkeypatch)
    list(dm.rhs_predict_dataloader())
    # predict falls back to test_time when no dedicated predict range exists.
    assert calls["kwargs"]["rhs_seed_time"] == 7


def test_predict_time_uses_predict_stage_seed_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """When the bundle emits a dedicated ``predict`` seed-time range,
    ``predict_time`` (and the RHS predict loader) use it over ``test_time``.
    This is the temporal-LP prediction-anchor path."""
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=4, val_time_ns=1_000_000_000,
                         test_time_ns=7_000_000_000,
                         predict_time_ns=9_000_000_000))

    # predict_time resolves to the predict-stage anchor (ns), not test_time.
    assert int(dm.predict_time.reshape(-1)[0].item()) == 9_000_000_000

    calls = _install_fake_diskgraph(monkeypatch)
    list(dm.rhs_predict_dataloader())
    # 9_000_000_000 ns -> 9 s; the RHS NodeRange loader seeds at the anchor.
    assert calls["kwargs"]["rhs_seed_time"] == 9


def test_predict_only_bundle_predict_time_non_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A PREDICT-only bundle carries ONLY a ``predict`` seed-time range (no
    train/val/test). ``predict_time`` must still resolve to the anchor so the
    RHS NodeRange loader gets a non-None ``rhs_seed_time`` (the bug: it used to
    fall back to ``test_time`` == None and the engine raised)."""
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=4, predict_time_ns=5_000_000_000))

    # No test_time in a predict-only bundle; predict_time still non-None.
    assert dm.test_time is None
    assert dm.predict_time is not None
    assert int(dm.predict_time.reshape(-1)[0].item()) == 5_000_000_000

    calls = _install_fake_diskgraph(monkeypatch)
    list(dm.rhs_predict_dataloader())
    assert calls["kwargs"]["rhs_seed_time"] == 5


def test_predict_only_static_bundle_zero_sentinel_seed_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Static LP emits a ``0`` sentinel as the predict seed time. The RHS
    NodeRange loader must receive ``rhs_seed_time=0`` (a valid non-None int
    satisfying NodeRange mode), matching the static LHS predict seeds."""
    dm = _rhs_dm(_rhs_lp_manifest(tmp_path, num_rhs=4, predict_time_ns=0))

    assert dm.predict_time is not None
    assert int(dm.predict_time.reshape(-1)[0].item()) == 0

    calls = _install_fake_diskgraph(monkeypatch)
    list(dm.rhs_predict_dataloader())
    assert calls["kwargs"]["rhs_seed_time"] == 0


def test_rhs_dataloader_covers_node_ids_in_order_with_seed_time_and_bs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The RHS batches cover seed-table ids exactly once IN ORDER, and each
    batch exposes ``seed_time`` + ``batch_size`` on the target type so
    ``init_mips_structure`` can read them."""
    # Emit the full [0,5) range as three in-order batches on the "test" stage.
    _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_stage={"test": [[0, 1], [2, 3], [4]]},
    )
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=5, test_time_ns=3_000_000_000))

    # Patch the collator to inject the seed_ref/seed_time the engine would
    # emit in NodeRange mode (the fake iterator yields the base batch shape).
    orig_to_hetero = dm._to_hetero

    def _collate(batch: Any) -> Any:
        ids = batch["task"]["input_ids"].to_pylist()
        batch["task"]["seed_timestamps"] = pa.array([3] * len(ids),
                                                    type=pa.int64())
        return orig_to_hetero(batch)

    dm._to_hetero = _collate  # type: ignore[method-assign]

    seen_stage_ids: list[int] = []
    for hetero in dm.rhs_test_dataloader():
        # batch_size == #RHS seeds in this batch.
        assert int(hetero["User"].batch_size) == hetero["User"].input_id.numel()
        # seed_time present (temporal) on the target type, fixed value.
        # Engine seconds normalized to ns canonical (default unit 's' -> *1e9).
        assert hetero["User"].seed_time is not None
        assert torch.all(hetero["User"].seed_time == 3 * 1_000_000_000)
        seen_stage_ids.extend(hetero["User"].input_id.tolist())

    assert seen_stage_ids == [10, 11, 12, 13, 14]


def test_rhs_dataloader_seed_time_on_distinct_target_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Regression: when the RHS target type differs from the LHS seed type
    (the real two-tower topology, e.g. amazon User->Item), the temporal RHS
    eval batch must carry ``seed_time`` on the TARGET block so
    ``init_mips_structure`` (link_gnn_module.py:170 reads
    ``batch[rhs_name].seed_time``) does not crash. The shared adapter only
    mirrors ``seed_time`` onto the LHS seed type, which masked this when
    seed==target."""
    _install_fake_diskgraph(
        monkeypatch,
        batch_input_ids_by_stage={"test": [[0, 1], [2]]},
    )
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=3, test_time_ns=3_000_000_000,
                         target_node_type="Item"))

    # Mimic the engine's NodeRange RHS batch: the seed block is the TARGET type
    # ("Item"), at a fixed seed time. The fake iterator yields a "User" block,
    # so rewrite the graph channel + seed_timestamps to the target type.
    orig_to_hetero = dm._to_hetero

    def _collate(batch: Any) -> Any:
        ids = batch["task"]["input_ids"].to_pylist()
        batch["task"]["seed_timestamps"] = pa.array([3] * len(ids),
                                                    type=pa.int64())
        main = batch["graphs"]["main"]
        main["node_ids"] = {"Item": main["node_ids"].pop("User")}
        main["batch_dict"] = {"Item": main["batch_dict"].pop("User")}
        main["num_sampled_nodes_dict"] = {
            "Item": main["num_sampled_nodes_dict"].pop("User")
        }
        return orig_to_hetero(batch)

    dm._to_hetero = _collate  # type: ignore[method-assign]

    for hetero in dm.rhs_test_dataloader():
        assert "Item" in hetero.node_types
        # seed_time lands on the TARGET block, not only the LHS seed type.
        # Engine seconds normalized to ns canonical (default unit 's' -> *1e9).
        assert hetero["Item"].seed_time is not None
        assert torch.all(hetero["Item"].seed_time == 3 * 1_000_000_000)
        assert int(
            hetero["Item"].seed_time.numel()) == hetero["Item"].batch_size


def test_rhs_loader_ordering_guard_fires_on_shuffle(tmp_path) -> None:
    """The MIPS-ordering guard rejects a misconfigured RHS loader (shuffle)."""
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=4, test_time_ns=1_000_000_000))
    bad = DataLoader(
        DiskgraphTaskBatchIterable(
            manifest_path="/tmp/runtime_manifest.json",
            stage="test",
            sampling_plan={"layers": []},
            batch_size=2,
            shuffle=True,  # violates the contract
            drop_last=False,
            base_seed=0,
            shuffle_state=dm._shuffle_state,
            num_seeds=4,
            replay=False,
        ),
        batch_size=None,
        num_workers=0,
    )
    with pytest.raises(AssertionError, match="must not shuffle"):
        dm._assert_rhs_loader_ordering(bad)


def test_rhs_loader_ordering_guard_fires_on_num_workers(tmp_path) -> None:
    """The guard rejects a multi-worker RHS loader (breaks in-order cat)."""
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=4, test_time_ns=1_000_000_000))
    bad = DataLoader(
        DiskgraphTaskBatchIterable(
            manifest_path="/tmp/runtime_manifest.json",
            stage="test",
            sampling_plan={"layers": []},
            batch_size=2,
            shuffle=False,
            drop_last=False,
            base_seed=0,
            shuffle_state=dm._shuffle_state,
            num_seeds=4,
            replay=False,
        ),
        batch_size=None,
        num_workers=2,
    )
    with pytest.raises(AssertionError, match="num_workers=0"):
        dm._assert_rhs_loader_ordering(bad)


def test_rhs_dataloader_raises_when_num_rhs_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A missing / empty RHS node count is a hard error (an empty MIPS index
    would silently break retrieval)."""
    _install_fake_diskgraph(monkeypatch)
    dm = _rhs_dm(
        _rhs_lp_manifest(tmp_path, num_rhs=0, test_time_ns=1_000_000_000))
    with pytest.raises(RuntimeError, match="positive RHS node count"):
        dm.rhs_test_dataloader()


def test_max_eval_batches_returns_none_when_cap_does_not_bite():
    """Regression: eval truncation bug.

    ``_max_eval_batches`` must return ``None`` (→ the exhaustion dataloader
    path, whose ``__len__`` correctly sums per-worker shards) whenever the
    configured ``max_*_steps`` cap is >= the natural batch count. Returning the
    natural (global) batch count instead forced the bounded path, where a
    global ``max_batches`` used as a stop-point under N sharded DataLoader
    workers truncated eval to ~1/N of the seeds (e.g. 214 val rows over
    num_workers=16 -> 14, single-class -> AUROC/AP nan -> "No valid trials").
    A genuine cap (max_steps < natural) is still honored.
    """
    inst = DiskgraphLightningDataModule.__new__(DiskgraphLightningDataModule)
    # Cap (2000) far larger than natural=ceil(214/1024)=1 -> must NOT cap.
    assert inst._max_eval_batches(214, 1024, 2000) is None
    # No cap configured.
    assert inst._max_eval_batches(214, 1024, None) is None
    # num_seeds unknown -> fall back to the cap value.
    assert inst._max_eval_batches(None, 1024, 7) == 7
    # Genuine cap that bites: natural=ceil(100000/16)=6250 > 5 -> honor 5.
    assert inst._max_eval_batches(100000, 16, 5) == 5
    # Boundary: cap exactly == natural -> no real cap -> None.
    assert inst._max_eval_batches(214, 16, inst._natural_batches(214,
                                                                 16)) is None


# Single-timestamp grouped batches (temporal-LP train loader) ################


def _install_grouped_fake_diskgraph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grouped_num_samples: int,
    flat_num_samples: int = 3,
    flat_batches_per_iterator: int = 1,
) -> dict[str, Any]:
    """Fake ``make_runtime_batch_iterator`` that, in ``group_by_timestamp``
    mode, reports ``grouped_num_samples`` as the (BATCH-count) ``num_samples``
    and yields exactly that many batches; otherwise behaves like the flat fake.
    """
    calls: dict[str, Any] = {"history": []}

    def make_runtime_batch_iterator(*args: Any, **kwargs: Any):
        calls["args"] = args
        calls["kwargs"] = kwargs
        calls["history"].append({"args": args, "kwargs": kwargs})
        if kwargs.get("group_by_timestamp"):
            return _FakeTaskIterator(grouped_num_samples), grouped_num_samples
        return _FakeTaskIterator(flat_batches_per_iterator), flat_num_samples

    stub = types.ModuleType("diskgraph")
    stub.make_runtime_batch_iterator = make_runtime_batch_iterator  # type: ignore[attr-defined]

    class _StubRuntimeIndex:
        """Stands in for ``diskgraph.RuntimeIndex``.

        Delegates to the fake factory above so each test's expectations about
        Remembers what moved to open-time and puts it back.
        """
        def __init__(self, dataset_ref, rnd_seed=None, num_threads=None):
            self._ref = dataset_ref
            self._rnd_seed = rnd_seed
            self._num_threads = num_threads

        def iter(self, **kwargs: Any):
            kwargs.setdefault("rnd_seed", self._rnd_seed)
            if self._num_threads is not None:
                kwargs.setdefault("num_threads", self._num_threads)
            return make_runtime_batch_iterator(self._ref, **kwargs)

    stub.RuntimeIndex = _StubRuntimeIndex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diskgraph", stub)
    # Grouping is gated on a TEMPORAL RHS (see ``_rhs_is_temporal``). These
    # grouping tests use a fake manifest with no index metadata, so force the
    # RHS-temporal signal True to isolate the grouping behaviour under test.
    monkeypatch.setattr(DiskgraphLightningDataModule, "_rhs_is_temporal",
                        property(lambda self: True))
    return calls


def _install_grouped_fake_diskgraph_per_stage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grouped_num_samples_by_stage: dict[str, int],
    flat_num_samples: int = 3,
) -> dict[str, Any]:
    """Fake ``make_runtime_batch_iterator`` for the GLOBAL-count grouped path.

    In ``group_by_timestamp`` mode it reports the per-stage grouped BATCH count
    (``grouped_num_samples_by_stage[stage]``) and yields exactly that many
    batches, each tagged with its stage so iteration can be checked to span all
    stages. The global (unsharded) count helper opens with
    ``num_workers=1, worker_id=0``; ``__iter__`` opens with the real worker
    fan-out. Both opens of a stage report the SAME grouped count (the wheel
    derives it from the stage ``bucket_index``, independent of worker fan-out in
    this fake), which is what lets the global-count path reuse the flat replay
    machinery.
    """
    calls: dict[str, Any] = {"history": []}
    _STAGE_ID_BASE = {"train": 0, "val": 1000, "test": 2000}

    def make_runtime_batch_iterator(*args: Any, **kwargs: Any):
        calls["args"] = args
        calls["kwargs"] = kwargs
        calls["history"].append({"args": args, "kwargs": kwargs})
        stage = kwargs["stage"]
        if kwargs.get("group_by_timestamp"):
            count = grouped_num_samples_by_stage[stage]
            # Tag each batch with a stage-specific input_id range so iteration
            # coverage across stages is checkable (train->0.., val->1000..).
            base = _STAGE_ID_BASE.get(stage, 0)
            batch_input_ids = [[base + i] for i in range(count)]
            return _FakeTaskIterator(count, batch_input_ids), count
        return _FakeTaskIterator(1), flat_num_samples

    stub = types.ModuleType("diskgraph")
    stub.make_runtime_batch_iterator = make_runtime_batch_iterator  # type: ignore[attr-defined]

    class _StubRuntimeIndex:
        """Stands in for ``diskgraph.RuntimeIndex``.

        Delegates to the fake factory above so each test's expectations about
        Remembers what moved to open-time and puts it back.
        """
        def __init__(self, dataset_ref, rnd_seed=None, num_threads=None):
            self._ref = dataset_ref
            self._rnd_seed = rnd_seed
            self._num_threads = num_threads

        def iter(self, **kwargs: Any):
            kwargs.setdefault("rnd_seed", self._rnd_seed)
            if self._num_threads is not None:
                kwargs.setdefault("num_threads", self._num_threads)
            return make_runtime_batch_iterator(self._ref, **kwargs)

    stub.RuntimeIndex = _StubRuntimeIndex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diskgraph", stub)
    # See _install_grouped_fake_diskgraph: force the RHS-temporal gate True.
    monkeypatch.setattr(DiskgraphLightningDataModule, "_rhs_is_temporal",
                        property(lambda self: True))
    return calls


_TEMPORAL_LP = "temporal_link_prediction"


def test_temporal_lp_train_loader_enables_grouping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """W1: ``group_by_timestamp=True`` is passed for the temporal-LP TRAIN
    loader and ONLY that loader (val/test/predict stay flat)."""
    calls = _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=2)
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [10, 10]})
    dm = DiskgraphLightningDataModule(
        _manifest(
            tmp_path=tmp_path, stage_counts={
                "train": 20,
                "val": 10,
                "test": 10,
                "predict": 10
            }, runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )

    list(dm.train_dataloader())
    assert calls["history"], "train loader never opened the iterator"
    assert all(c["kwargs"].get("group_by_timestamp") is True
               for c in calls["history"])

    for method in ("val_dataloader", "test_dataloader", "predict_dataloader"):
        calls["history"].clear()
        list(getattr(dm, method)())
        assert calls["history"], f"{method} never opened the iterator"
        assert all(c["kwargs"].get("group_by_timestamp") is False
                   for c in calls["history"]), f"{method} must not group"


def test_static_lp_with_lhs_loader_and_temporal_rhs_groups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Gate (truth table): a STATIC-LP task ALSO groups when it uses an LHS node
    loader and its RHS is temporal. The gate mirrors legacy ``is_link_pred``
    (static OR temporal), NOT just ``temporal_link_prediction``."""
    calls = _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=2)
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [10, 10]})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type="static_link_prediction",
        use_lhs_node_loader=True,
    )
    list(dm.train_dataloader())
    assert calls["history"]
    assert all(c["kwargs"].get("group_by_timestamp") is True
               for c in calls["history"])


def test_non_lp_task_never_groups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Gate (truth table): a non-LP task (e.g. regression / node task) never
    groups even with a temporal RHS, because ``is_link_pred`` is False."""
    calls = _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=2)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type="regression",
        use_lhs_node_loader=True,
    )
    list(dm.train_dataloader())
    assert calls["history"]
    assert all(c["kwargs"].get("group_by_timestamp") is False
               for c in calls["history"])


def test_two_tower_no_lhs_loader_does_not_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Gate (truth table): a temporal-LP task whose module is two_tower (no LHS
    node loader) must take the FLAT path, mirroring legacy which only selects
    ``TimestampSampler`` under ``module.use_lhs_node_loader``."""
    calls = _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=2)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        rhs_node_type="products",
        use_lhs_node_loader=False,
    )
    list(dm.train_dataloader())
    assert calls["history"]
    assert all(c["kwargs"].get("group_by_timestamp") is False
               for c in calls["history"])


def _install_plain_fake_diskgraph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grouped_num_samples: int = 2,
    flat_num_samples: int = 3,
    flat_batches_per_iterator: int = 1,
) -> dict[str, Any]:
    """Like ``_install_grouped_fake_diskgraph`` but WITHOUT forcing the
    ``_rhs_is_temporal`` gate, so the real RHS-temporality gate is exercised."""
    calls: dict[str, Any] = {"history": []}

    def make_runtime_batch_iterator(*args: Any, **kwargs: Any):
        calls["args"] = args
        calls["kwargs"] = kwargs
        calls["history"].append({"args": args, "kwargs": kwargs})
        if kwargs.get("group_by_timestamp"):
            return _FakeTaskIterator(grouped_num_samples), grouped_num_samples
        return _FakeTaskIterator(flat_batches_per_iterator), flat_num_samples

    stub = types.ModuleType("diskgraph")
    stub.make_runtime_batch_iterator = make_runtime_batch_iterator  # type: ignore[attr-defined]

    class _StubRuntimeIndex:
        """Stands in for ``diskgraph.RuntimeIndex``.

        Delegates to the fake factory above so each test's expectations about
        Remembers what moved to open-time and puts it back.
        """
        def __init__(self, dataset_ref, rnd_seed=None, num_threads=None):
            self._ref = dataset_ref
            self._rnd_seed = rnd_seed
            self._num_threads = num_threads

        def iter(self, **kwargs: Any):
            kwargs.setdefault("rnd_seed", self._rnd_seed)
            if self._num_threads is not None:
                kwargs.setdefault("num_threads", self._num_threads)
            return make_runtime_batch_iterator(self._ref, **kwargs)

    stub.RuntimeIndex = _StubRuntimeIndex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diskgraph", stub)
    return calls


def test_atemporal_rhs_temporal_lp_train_loader_does_not_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Gate: a TEMPORAL-LP task whose RHS table is ATEMPORAL (no time/end_time
    column, e.g. amazon ``products``) must take the FLAT path, mirroring legacy
    ``TimestampSampler`` (selected iff the RHS has a time/end_time column). This
    is the regression fix: task_type==temporal_link_prediction alone must NOT
    force single-timestamp grouping on an atemporal RHS."""
    calls = _install_plain_fake_diskgraph(monkeypatch)
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        rhs_node_type="products",
    )
    # Fake manifest has no index metadata -> time_col_dict empty ->
    # _rhs_is_temporal is False.
    assert dm._rhs_is_temporal is False
    list(dm.train_dataloader())
    assert calls["history"]
    assert all(c["kwargs"].get("group_by_timestamp") is False
               for c in calls["history"])


def test_temporal_rhs_temporal_lp_train_loader_groups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Gate: a TEMPORAL-LP task whose RHS table IS temporal (has a create-time
    column) DOES use single-timestamp grouping. This is the case the feature is
    for (per-batch RHS temporal-validity correctness)."""
    calls = _install_plain_fake_diskgraph(monkeypatch)
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [10, 10]})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        rhs_node_type="products",
        use_lhs_node_loader=True,
    )
    # Simulate a temporal RHS: products has a create-time column.
    monkeypatch.setattr(type(dm), "time_col_dict",
                        property(lambda self: {"products": "create_ts"}))
    assert dm._rhs_is_temporal is True
    list(dm.train_dataloader())
    assert calls["history"]
    assert all(c["kwargs"].get("group_by_timestamp") is True
               for c in calls["history"])


def test_rhs_is_temporal_end_time_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Gate signal: an RHS with only an END-time column (no create-time) still
    counts as temporal (matches legacy ``rhs_end_time is not None``)."""
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        rhs_node_type="products",
    )
    # No create-time column, but an end-time column is present.
    monkeypatch.setattr(type(dm), "time_col_dict", property(lambda self: {}))
    monkeypatch.setattr(type(dm), "_end_time_col_for",
                        lambda self, nt: "expire_ts")
    assert dm._rhs_is_temporal is True


def test_rhs_is_temporal_false_without_rhs_node_type(tmp_path, ) -> None:
    """Gate signal: no RHS node type -> not temporal (flat path)."""
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
    )
    assert dm._rhs_is_temporal is False


def test_grouped_train_len_uses_true_batch_count_not_ceil(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """W2: grouped epoch length == the iterator's true grouped batch count,
    which exceeds ``ceil(num_seeds/batch_size)`` (each timestamp bucket has its
    own partial tail). The flat path keeps ``ceil(num_seeds/batch_size)``."""
    # 20 seeds, batch_size=10 -> flat ceil = 2. Five single-ts buckets (each with
    # its own partial tail, count=3) give Σ ceil(3/10) = 5 grouped batches.
    _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=5)
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [3, 3, 3, 3, 3]})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )
    loader = dm.train_dataloader()
    # __len__ is the true grouped batch count (5), strictly greater than the
    # flat ceil(20/10) = 2.
    assert len(loader) == 5
    assert len(loader) > (20 + 10 - 1) // 10

    # Control: same shape, flat (non-temporal) -> ceil(20/10) = 2.
    flat_dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type="static_link_prediction",
    )
    assert len(flat_dm.train_dataloader()) == 2


def test_grouped_train_iter_does_not_truncate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """W2 (the crux): the replay/iter target equals the advertised grouped
    ``__len__``, so iteration delivers every grouped batch (no bucket dropped).
    """
    _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=5)
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [3, 3, 3, 3, 3]})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )
    loader = dm.train_dataloader()
    advertised = len(loader)
    delivered = len(list(loader))
    assert advertised == 5
    assert delivered == advertised  # would be 2 (truncated) under the old ceil


def test_grouped_train_len_honors_step_caps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """W2: per-epoch min/max-step caps still apply on top of the grouped count.
    """
    # Grouped natural = 4 (four single-ts buckets); max cap 3 bites; min cap
    # floor 2 (does not bite).
    _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=4)
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [3, 3, 3, 3]})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 20},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        max_steps_per_epoch=3,
        min_steps_per_epoch=2,
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )
    loader = dm.train_dataloader()
    assert len(loader) == 3
    assert len(list(loader)) == 3


def test_grouped_num_samples_not_divided_by_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """W2: in grouped mode ``num_samples`` is already a BATCH count and must NOT
    be divided by ``batch_size`` (the semantic flip vs the flat row-count)."""
    _install_grouped_fake_diskgraph(monkeypatch, grouped_num_samples=7)
    # 7 single-ts buckets, each count=5 (< batch_size): Σ ceil(5/128) = 7. If the
    # bucket counts were summed (35) then //128 it would collapse to 0/1.
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [5, 5, 5, 5, 5, 5, 5]})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 1000},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=128,
        eval_batch_size=128,
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )
    # If counts were treated as a single row count it would be //128 -> 0/1.
    assert len(dm.train_dataloader()) == 7


def test_grouped_global_batches_from_manifest_buckets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """FIX 1: ``_grouped_global_batches`` derives the count from the manifest
    bucket lists (Σ_buckets ceil(count/batch_size) over all stages) WITHOUT
    opening the wheel iterator."""

    # Make make_runtime_batch_iterator explode so the test proves the count is
    # manifest-derived (never opens the sampler).
    def _explode(*_a: Any, **_k: Any):
        raise AssertionError(
            "_grouped_global_batches must not open the iterator")

    stub = types.ModuleType("diskgraph")
    stub.make_runtime_batch_iterator = _explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diskgraph", stub)

    # train: buckets [3, 25] @bs10 -> ceil(3/10)+ceil(25/10) = 1+3 = 4
    # val:   buckets [10, 10]      -> 1+1 = 2  => global = 6.
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={
            "train": [3, 25],
            "val": [10, 10]
        })
    iterable = DiskgraphTaskBatchIterable(
        manifest_path=runtime_path,
        stage=("train", "val"),
        sampling_plan={"layers": []},
        batch_size=10,
        shuffle=True,
        drop_last=False,
        base_seed=0,
        shuffle_state=_ShuffleStreamState(),
        group_by_timestamp=True,
    )
    assert iterable._grouped_global_batches() == 6
    # Memoized: a second call returns the same value (still no iterator open).
    assert iterable._grouped_global_batches() == 6


def test_grouped_global_batches_fails_loud_without_bucket_index(
    tmp_path, ) -> None:
    """FIX 1: a grouped stage with no ``bucket_index`` in the DG manifest fails
    loud (ingest must have emitted grouping), never a silent flat fallback."""
    ext_root = tmp_path / "tt_ext"
    ext_root.mkdir()
    runtime_path = ext_root / "runtime_manifest.json"
    runtime_path.write_text(
        json.dumps({
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": str(ext_root),
                    "row_count": 20,
                }
            }
        }))
    (ext_root / "diskgraph_manifest.json").write_text(
        json.dumps({
            "task_dataset": {
                "stages": {
                    "train": {
                        "table_name": "train_seeds"
                    }  # no bucket_index
                }
            }
        }))
    iterable = DiskgraphTaskBatchIterable(
        manifest_path=str(runtime_path),
        stage="train",
        sampling_plan={"layers": []},
        batch_size=10,
        shuffle=False,
        drop_last=False,
        base_seed=0,
        shuffle_state=_ShuffleStreamState(),
        group_by_timestamp=True,
    )
    with pytest.raises(ValueError, match="no bucket_index"):
        iterable._grouped_global_batches()


def test_grouped_multistage_refit_spans_all_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """BUG 1 regression: grouped multi-stage train (refit_trainval /
    refit_full) must source ``__len__`` AND the replay target from the GLOBAL
    grouped count summed over ALL stages — not ``stages[0]`` only — so val/test
    are opened and never dropped mid-epoch.

    Pre-fix: ``__len__`` summed all stages (-> 8) but ``__iter__`` set the
    per-worker target from ``stages[0]`` (train=3), so iteration RETURNED after
    the train stage and the val stage was never sampled.
    """
    calls = _install_grouped_fake_diskgraph_per_stage(
        monkeypatch,
        grouped_num_samples_by_stage={
            "train": 3,
            "val": 5
        },
    )
    runtime_path = _write_grouped_manifests(
        tmp_path,
        buckets_by_stage={
            "train": [10, 10, 10],
            "val": [10] * 5
        },
    )
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={
            "train": 30,
            "val": 50
        }, runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        train_stages=("train", "val"),
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )
    loader = dm.train_dataloader()
    # __len__ == capped(global train+val grouped count) = 3 + 5 = 8.
    assert len(loader) == 8

    batches = list(loader)
    # delivered == advertised (single worker -> no ceil rounding).
    assert len(batches) == 8
    seen_ids = [b["train_table"].input_id.item() for b in batches]
    # Both stages contributed (train ids < 1000, val ids >= 1000); val NOT
    # dropped. Pre-fix this list would be the 3 train ids only.
    assert any(i < 1000 for i in seen_ids), "train stage missing"
    assert any(i >= 1000 for i in seen_ids), "val stage dropped (BUG 1)"
    assert sum(1 for i in seen_ids if i < 1000) == 3
    assert sum(1 for i in seen_ids if i >= 1000) == 5
    # The two stages were actually opened (group mode) for iteration.
    grouped_stages = {
        c["kwargs"]["stage"]
        for c in calls["history"] if c["kwargs"].get("group_by_timestamp")
    }
    assert grouped_stages == {"train", "val"}


def test_grouped_global_cap_distributed_across_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """BUG 2 regression: with ``dataloader_num_workers>1`` and a biting cap, the
    cap is GLOBAL per-epoch (mirrors the flat path). ``__len__`` must apply the
    cap to the global grouped count, then distribute by the flat per-worker
    formula -- i.e. ``caps(Sigma) == Sigma caps`` no longer diverges.

    Global grouped count = 10. With max_steps_per_epoch=6 the capped global is 6.
    Across 2 workers the flat distribution is ceil(6/2)=3 per worker -> __len__
    advertises the global capped 6 (matching the flat path for the same cap).
    Pre-fix __len__ summed per-worker un-distributed grouped counts (10+10=20),
    THEN applied the cap once (->6 here, but via caps(Sigma)) while __iter__
    applied the cap PER worker (caps(10)=6 each -> 12 yielded): divergent.
    """
    _install_grouped_fake_diskgraph_per_stage(
        monkeypatch,
        grouped_num_samples_by_stage={"train": 10},
    )
    runtime_path = _write_grouped_manifests(
        tmp_path, buckets_by_stage={"train": [10] * 10})
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100},
                  runtime_manifest_path=runtime_path),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        num_workers=2,
        max_steps_per_epoch=6,
        task_type=_TEMPORAL_LP,
        use_lhs_node_loader=True,
    )
    loader = dm.train_dataloader()
    # __len__ is the GLOBAL capped count (6), exactly as the flat path would
    # report for a global natural of 10 capped at 6.
    assert len(loader) == 6

    # Per-worker target is the flat formula over the capped GLOBAL count:
    # ceil(6 / 2) = 3. Mirror what flat does (per_worker_target).
    iterable = loader.dataset
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=2, id=0),
    )
    worker0 = list(iterable)
    assert len(worker0) == 3
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=2, id=1),
    )
    worker1 = list(iterable)
    assert len(worker1) == 3
    # Sum of per-worker yields == advertised __len__ (no caps(Sigma) != Sigma
    # caps divergence). Pre-fix: 6 (each worker) per worker => 12 total.
    assert len(worker0) + len(worker1) == len(loader)


def test_grouped_global_cap_min_steps_distributed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """BUG 2 regression (min cap): a biting ``min_steps_per_epoch`` is applied to
    the GLOBAL grouped count then distributed across workers, same as flat."""
    _install_grouped_fake_diskgraph_per_stage(
        monkeypatch,
        grouped_num_samples_by_stage={"train": 4},
    )
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 40}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=10,
        num_workers=2,
        min_steps_per_epoch=8,  # bites: floors global 4 -> 8
        task_type=_TEMPORAL_LP,
    )
    loader = dm.train_dataloader()
    assert len(loader) == 8  # global floored to min, then advertised
    iterable = loader.dataset
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=2, id=0),
    )
    w0 = list(iterable)
    monkeypatch.setattr(
        "kumo_core.diskgraph.lightning_data.get_worker_info",
        lambda: types.SimpleNamespace(num_workers=2, id=1),
    )
    w1 = list(iterable)
    # ceil(8/2) = 4 each, sum == advertised.
    assert len(w0) == 4
    assert len(w1) == 4
    assert len(w0) + len(w1) == len(loader)


class _FakeSamplerEngine:
    """Stub for ``diskgraph.SamplerEngine`` returning a fixed
    ``timestamp[s]`` feature column (mirrors the real engine, which returns
    create-time columns as arrow ``timestamp[s]``)."""

    # 2011-08-17, 2013-07-15, 1999-12-04 as unix SECONDS.
    _UNIX_SECONDS = [1313539200, 1373846400, 944265600]

    def __init__(self, index_path: str, **_kwargs) -> None:
        self._index_path = index_path

    def fetch_features(self, request):
        col = request[0]["feature_names"][0]
        n = len(request[0]["node_ids"])
        secs = [
            self._UNIX_SECONDS[i % len(self._UNIX_SECONDS)] for i in range(n)
        ]
        arr = pa.array(secs, type=pa.timestamp("s"))
        batch = pa.RecordBatch.from_arrays([arr], names=[col])
        return [{"type": "feature", "batch": batch}]


def test_fetch_rhs_time_col_returns_nanoseconds_not_seconds(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (ns-canonical): ``_fetch_rhs_time_col`` must return create-time
    in int64 NANOSECONDS regardless of the arrow source resolution. Kumo Python
    is ns-canonical, so this column is compared against an ns ``seed_time`` and ns
    ``unix_*_offset`` in ``RHSTime`` (all-ns RHS validity). A bare
    ``pd.to_datetime(...).astype('int64')`` is resolution-dependent under pandas
    >=2 (a ``timestamp[s]`` source yields SECONDS), so mixing it with ns
    seed/offsets empties the valid-RHS set => zero training signal (the
    temporal-RHS LP metric floored to ~0)."""
    import diskgraph
    monkeypatch.setattr(diskgraph, "SamplerEngine", _FakeSamplerEngine)

    out = dldm.DiskgraphLightningDataModule._fetch_rhs_time_col(
        index_path="/unused",
        node_type="products",
        num_nodes=3,
        col_name="product_time",
    )
    assert out is not None
    expected = torch.tensor([1313539200, 1373846400, 944265600],
                            dtype=torch.long) * 1_000_000_000
    assert torch.equal(out.long(), expected), (
        f"expected nanoseconds {expected.tolist()}, got {out.tolist()} "
        "(seconds => mixed-unit empty-valid-RHS floor bug)")


def test_valid_rhs_filter_nonempty_with_ns_create_time() -> None:
    """End-to-end of the validity arithmetic with EVERYTHING in nanoseconds:
    create_time, seed_time, and (crucially) NONZERO ns offsets. The prior tests
    passed offsets=0, which masked the mixed-unit bug; here a real ns offset must
    not blow up the comparison (seconds offsets would be ~1e9x too small)."""
    NS = 1_000_000_000
    create_time = torch.tensor([1313539200, 1373846400, 944265600],
                               dtype=torch.long) * NS
    # 1-day offsets in NANOSECONDS (compute_unix_step convention).
    one_day_ns = 24 * 60 * 60 * NS
    provider = DiskgraphValidRHS(
        num_rhs_nodes=3,
        unix_start_offset=-one_day_ns,
        unix_end_offset=one_day_ns,
        create_time=create_time,
        end_time=None,
    )
    # seed_time (ns) just before product 0's create time: still valid because the
    # +1day unix_end_offset widens the window (create_time <= seed + end_offset).
    seed_ns = (1313539200 - 60) * NS  # 1 minute before product 0's create
    valid = provider.get_valid_rhs_idx(seed_ns)
    # product 0 (within +1day) and product 2 (created 1999, well before) valid;
    # product 1 (2013) is still in the future.
    assert set(valid.tolist()) == {0, 2}, valid.tolist()

    # And with all-ns, a far-future seed makes ALL non-expiring products valid.
    valid_all = provider.get_valid_rhs_idx(2_000_000_000 * NS)
    assert valid_all.numel() == 3


def test_seed_time_to_index_unit_uses_manifest_unit(tmp_path) -> None:
    """The ONE ns -> engine-unit conversion (for the NodeRange engine query) is
    driven by the manifest's configured ``timestamp_unit``, NOT a hardcoded
    ``//1e9``. Seconds index => ns//1e9; millisecond index => ns//1e6."""
    NS = 1_000_000_000
    seed_ns = torch.tensor([5 * NS], dtype=torch.long)

    dm_s = _rhs_dm(_rhs_lp_manifest(tmp_path / "s", num_rhs=3, test_time_ns=NS))
    # Default index unit is seconds.
    assert dm_s.index_np_time_unit == "s"
    assert dm_s._seed_time_to_index_unit(seed_ns) == 5  # 5e9 ns -> 5 s
    assert dm_s._seed_time_to_index_unit(None) is None

    # Force a millisecond index unit; ns -> ms divides by 1e6.
    dm_ms = _rhs_dm(
        _rhs_lp_manifest(tmp_path / "ms", num_rhs=3, test_time_ns=NS))
    dm_ms._index_np_time_unit = "ms"
    assert dm_ms._seed_time_to_index_unit(seed_ns) == 5_000  # 5e9 ns -> 5e3 ms


def test_train_dataloader_sets_fork_context_when_workers(tmp_path) -> None:
    """Python 3.14 workers retain fork to avoid pickling CUDA state."""
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        num_workers=4,
    )
    loader = dm.train_dataloader()
    context = loader.multiprocessing_context
    context_name = (context if isinstance(context, str) else getattr(
        context, "_name", None))
    assert context_name == "fork"
    assert loader.worker_init_fn is dldm._init_diskgraph_worker


def test_train_dataloader_no_context_when_single_process(tmp_path) -> None:
    """num_workers=0 (single-process) must NOT pin a start-method context: there
    are no worker subprocesses, so the hazard does not arise."""
    dm = DiskgraphLightningDataModule(
        _manifest(tmp_path=tmp_path, stage_counts={"train": 100}),
        _mapper_bank(),
        sampling_plan={"layers": []},
        train_batch_size=10,
        eval_batch_size=1,
        num_workers=0,
    )
    loader = dm.train_dataloader()
    assert loader.multiprocessing_context is None


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="fork and queue introspection require Linux")
def test_dataloader_workers_exit_when_parent_stops_draining() -> None:
    """Regression for workers joining a full result queue after parent exit."""
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    consumer = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--shutdown-probe-consumer",
            str(ready_write),
            str(release_read),
        ],
        pass_fds=(ready_write, release_read),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(ready_write)
    os.close(release_read)
    stuck: list[psutil.Process] = []
    try:
        poller = select.poll()
        poller.register(ready_read, select.POLLIN)
        assert poller.poll(60_000), "loader did not fill its result queue"
        raw_payload = os.read(ready_read, 4096)
        assert raw_payload, "loader exited before reporting worker PIDs"
        payload = json.loads(raw_payload)
        assert "error" not in payload, payload.get("error")
        os.write(release_write, b"x")
        consumer.wait(timeout=5)
        stuck = _wait_for_processes(payload["pids"], timeout_seconds=10)
        assert not stuck, f"DataLoader workers did not exit: {stuck}"
    finally:
        for process in stuck:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        os.close(ready_read)
        os.close(release_write)
        if consumer.poll() is None:
            consumer.kill()
            consumer.wait()


def test_suppress_iterable_len_warning_only_scopes_to_the_context() -> None:
    def _warn() -> None:
        warnings.warn("Your `IterableDataset` has `__len__` defined.",
                      UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        with dldm.suppress_iterable_len_warning():
            _warn()
        assert caught == []

        # Outside the context, the same warning is not suppressed.
        _warn()
        assert len(caught) == 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--shutdown-probe-consumer":
        _run_shutdown_probe_consumer(int(sys.argv[2]), int(sys.argv[3]))
