# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from kumo_core.train.base_module import BaseLightningModule
from torch_geometric.data import HeteroData


class _BatchDumpModule(BaseLightningModule):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def common_step(self, batch):
        raise NotImplementedError

    def train_step(self, batch, batch_idx):
        raise NotImplementedError

    def evaluation_step(self, batch, stage, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError


class _BatchDumpExtraModule(_BatchDumpModule):
    def _batch_dump_extra_payload(self, batch, stage):
        return {"stage": stage, "value": torch.tensor([7])}


def _module_for_batch_dump(tmp_path):
    module = object.__new__(_BatchDumpModule)
    module._batch_dump_dir = str(tmp_path)
    module._trainer = SimpleNamespace(datamodule=SimpleNamespace(
        _train_table_name="train_table"))
    module.entity_name = "customers"
    return module


def test_maybe_dump_batch_is_noop_when_disabled(tmp_path) -> None:
    module = object.__new__(_BatchDumpModule)
    module._batch_dump_dir = ""
    module._trainer = None

    batch = HeteroData()
    batch["customers"].input_id = torch.tensor([1])

    BaseLightningModule._maybe_dump_batch(module, batch, "train")

    assert list(tmp_path.glob("batch_*.pt")) == []


def test_maybe_dump_batch_hashes_seed_identity_independent_of_order(
        tmp_path) -> None:
    first = HeteroData()
    first["customers"].pkey = np.array(["c2", "c1"], dtype=object)
    first["customers"].time = torch.tensor([20, 10])
    first["customers"].batch_size = 2
    first["train_table"].time = torch.tensor([20, 10])
    first["train_table"].y = torch.tensor([1.0, 0.0])

    second = HeteroData()
    second["customers"].pkey = np.array(["c1", "c2"], dtype=object)
    second["customers"].time = torch.tensor([10, 20])
    second["customers"].batch_size = 2
    second["train_table"].time = torch.tensor([10, 20])
    second["train_table"].y = torch.tensor([0.0, 1.0])

    BaseLightningModule._maybe_dump_batch(_module_for_batch_dump(tmp_path),
                                          first, "train")
    BaseLightningModule._maybe_dump_batch(_module_for_batch_dump(tmp_path),
                                          second, "train")

    dumps = list(tmp_path.glob("batch_train_train_table_*.pt"))
    assert len(dumps) == 1


def test_maybe_dump_batch_falls_back_to_input_id(tmp_path) -> None:
    batch = HeteroData()
    batch["customers"].input_id = torch.tensor([42, 43])
    batch["customers"].batch_size = 2

    BaseLightningModule._maybe_dump_batch(_module_for_batch_dump(tmp_path),
                                          batch, "val")

    dumps = list(tmp_path.glob("batch_val_batch_*.pt"))
    assert len(dumps) == 1


def test_maybe_dump_batch_saves_extra_payload_without_mutating_batch(
        tmp_path) -> None:
    module = object.__new__(_BatchDumpExtraModule)
    module._batch_dump_dir = str(tmp_path)
    module._trainer = SimpleNamespace(datamodule=SimpleNamespace(
        _train_table_name="train_table"))
    module.entity_name = "customers"

    batch = HeteroData()
    batch["customers"].input_id = torch.tensor([42])
    batch["customers"].batch_size = 1

    BaseLightningModule._maybe_dump_batch(module, batch, "test")

    dumps = list(tmp_path.glob("batch_test_batch_*.pt"))
    assert len(dumps) == 1
    dumped = torch.load(dumps[0], map_location="cpu", weights_only=False)
    assert dumped.kumo_parity_extra["stage"] == "test"
    assert dumped.kumo_parity_extra["value"].tolist() == [7]
    assert not hasattr(batch, "kumo_parity_extra")
