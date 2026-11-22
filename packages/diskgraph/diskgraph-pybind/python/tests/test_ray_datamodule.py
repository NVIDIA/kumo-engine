# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native sampling, Python actor encoding, async delivery, and Lightning fit."""

import json
from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from test_runtime_batch_iterator_e2e import (
    _sampling_plan,
    seed_table_fixture,  # noqa: F401 -- shared native ingest fixture
)

pytest.importorskip("ray")
pytest.importorskip("kumo_core")

import pytorch_lightning as pl
import ray
import torch
from diskgraph import RuntimeIndex, SamplingOptions
from kumo_core.diskgraph.batch_contract import BatchRequest, SamplingContractError
from kumo_core.diskgraph.lightning_data import (
    DiskgraphLightningDataModule,
    DiskgraphTaskBatchIterable,
    RemoteDataLoader,
)
from kumo_core.diskgraph.ray_sampling import (
    ArtifactIdentity,
    RayBackendConfig,
    RayConnection,
    RayRuntimeIndex,
    ReplicaConfig,
    SamplerActor,
    start_sampling_replicas,
    stop_sampling_replicas,
)
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper
from kumo_core.task_dataset import DiskgraphTaskManifest
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData


@pytest.fixture(scope="module")
def remote_module(
    seed_table_fixture: dict[str, str | int],  # noqa: F811 -- imported fixture
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DiskgraphLightningDataModule]:
    """Start actual native actors with the DataModule's fitted Python encoder."""
    root = tmp_path_factory.mktemp("remote_datamodule")
    original = Path(str(seed_table_fixture["manifest_path"]))
    runtime = json.loads(original.read_text())
    runtime["stages"] = {
        stage: runtime["stages"]["train"]
        for stage in ("train", "val", "test", "predict")
    }
    manifest_path = root / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(runtime))
    (root / "split_info.json").write_text(
        json.dumps(
            {
                "stage_counts": {
                    stage: seed_table_fixture["n_seeds"] for stage in runtime["stages"]
                }
            }
        )
    )
    manifest = DiskgraphTaskManifest(
        str(root / "manifest.json"),
        str(manifest_path),
        {"artifact_root": str(root), "split_info_path": "split_info.json"},
        {
            "train_table_name": "train_seeds",
            "seed_node_type": "users",
            "task_type": "regression",
            "train_table_columns": {
                "seed_id_column": "ENTITY1",
                "target_column": "TARGET1",
                "time_column": "TIMESTAMP",
            },
        },
        {
            "node_types": {
                "users": {"mapper_key": "users", "feature_columns": []},
                "orders": {"mapper_key": "orders", "feature_columns": ["order_id"]},
            }
        },
        {"graph_index_path": str(seed_table_fixture["parent_root"])},
    )
    module = DiskgraphLightningDataModule(
        manifest,
        {"users": {}, "orders": {"order_id": NumericalTensorMapper()}},
        sampling_plan=_sampling_plan(),
        train_batch_size=2,
        eval_batch_size=2,
        base_seed=42,
        shuffle_train=False,
        num_workers=4,
        prefetch_factor=2,
        runtime_options={"num_threads": 1},
        resolve_entity_node_type="users",
    )
    snapshot = Path(str(seed_table_fixture["parent_root"])) / "index_metadata.json"
    encoder = module.remote_batch_encoder()
    artifacts = ArtifactIdentity(
        sha256(snapshot.read_bytes()).hexdigest(),
        sha256(manifest_path.read_bytes()).hexdigest(),
        sha256(ray.cloudpickle.dumps(encoder)).hexdigest(),
    )
    ray.init(
        num_cpus=2,
        namespace="remote-datamodule",
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
        _node_ip_address="127.0.0.1",
    )
    actors: tuple[SamplerActor, ...] = ()
    try:
        placement = NodeAffinitySchedulingStrategy(
            ray.get_runtime_context().get_node_id(), soft=False
        )
        config = ReplicaConfig(
            str(manifest_path), str(snapshot), artifacts, 42, 1, encoder
        )
        actors = start_sampling_replicas([config, config], [placement, placement])
        module.sampling_backend = RayBackendConfig(
            RayConnection("auto", "remote-datamodule"),
            actors,
            artifacts,
            42,
            1,
        )
        yield module
    finally:
        module.teardown()
        stop_sampling_replicas(actors)
        ray.shutdown()


def test_remote_training_batches_match_local_and_evaluation_stays_local(
    remote_module: DiskgraphLightningDataModule,
) -> None:
    """Use the same epoch/options, mappers, labels, graph, and sampled counts."""
    module = remote_module
    backend = module.sampling_backend
    assert backend is not None
    loader = module.train_dataloader()
    assert isinstance(loader, RemoteDataLoader)
    assert loader.num_workers == 0
    source = cast(DiskgraphTaskBatchIterable, loader.dataset)
    assert source is loader._batch_dataset
    requests = list(source.iter_requests())
    assert len(requests) == len(loader) > 0
    index = RuntimeIndex(
        module.task_manifest.runtime_manifest_path, rnd_seed=42, num_threads=1
    )
    context, _ = index.iter(requests[0].options)
    encoder = module.remote_batch_encoder()
    references = {
        tuple(request.seed_ids): encoder(context.sample_request(list(request.seed_ids)))
        for request in requests
    }
    with patch(
        "kumo_core.diskgraph.lightning_data.runtime_batch_to_hetero",
        side_effect=AssertionError("remote batches must not be re-encoded"),
    ) as trainer_encode:
        received = list(loader)
    trainer_encode.assert_not_called()
    assert len(received) == len(references)
    for actual in received:
        key = tuple(actual["users"].input_id.tolist())
        expected = references[key]
        assert set(actual.node_types) == set(expected.node_types)
        assert set(actual.edge_types) == set(expected.edge_types)
        for name in actual.node_types + actual.edge_types:
            assert set(actual[name].keys()) == set(expected[name].keys())
            for attribute, value in expected[name].items():
                received_value = actual[name][attribute]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(received_value, value)
                elif isinstance(value, TensorFrame):
                    assert received_value.col_names_dict == value.col_names_dict
                    assert received_value.num_rows == value.num_rows
                    for stype, tensor in value.feat_dict.items():
                        torch.testing.assert_close(
                            received_value.feat_dict[stype], tensor
                        )
                else:
                    assert received_value == value
    # Evaluation uses its ordinary primary-key-aware local collator.
    for method in (
        module.val_dataloader,
        module.test_dataloader,
        module.predict_dataloader,
    ):
        local = method()
        assert not isinstance(local, RemoteDataLoader)
        assert local.num_workers == 0
        assert local.dataset.sampling_backend is None
        assert isinstance(next(iter(local)), HeteroData)
    module.sampling_backend = replace(backend, engine_seed=43)
    try:
        with pytest.raises(SamplingContractError, match="seed differs"):
            module.train_dataloader()
    finally:
        module.sampling_backend = backend


class _Model(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.validation_batches = 0

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        assert batch["orders"].tf.num_cols > 0
        return (self.weight - batch["train_seeds"].y.float().mean()).square()

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        assert torch.isfinite(batch["train_seeds"].y).all()
        self.validation_batches += 1

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_lightning_fit_restarts_remote_epochs_and_drains_on_limit(
    remote_module: DiskgraphLightningDataModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise optimizer steps, epoch restart, and an early training limit."""
    epochs: list[int] = []
    original_schedule = RayRuntimeIndex.iter_requests

    def record_schedule(
        index: RayRuntimeIndex,
        options: SamplingOptions,
    ) -> tuple[Iterator[BatchRequest], int]:
        epochs.append(options.epoch)
        return original_schedule(index, options)

    monkeypatch.setattr(RayRuntimeIndex, "iter_requests", record_schedule)
    model = _Model()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[DiskgraphLightningDataModule.shuffle_stream_callback()],
    )
    trainer.fit(model, datamodule=remote_module)
    assert epochs == [0, 1]
    assert trainer.global_step == 4
    assert model.weight.item() != 1.0
    assert model.validation_batches == 2
    assert remote_module.active_dataloader_worker_pids == ()
    for loader in remote_module._remote_loaders:
        active = loader._batch_dataset._remote_iterator
        assert active is None or (active.closed and not active._dispatcher.is_alive())
