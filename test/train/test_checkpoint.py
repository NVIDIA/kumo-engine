# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from kumo_core.train.checkpoint import KumoCheckpointIO


class _CheckpointMetadata:
    pass


def test_kumo_checkpoint_io_loads_trusted_metadata(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"value": _CheckpointMetadata()}, checkpoint_path)

    checkpoint = KumoCheckpointIO().load_checkpoint(checkpoint_path)

    assert isinstance(checkpoint["value"], _CheckpointMetadata)
