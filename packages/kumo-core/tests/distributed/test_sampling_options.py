# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ray requests carry DiskGraph-owned options directly across serialization."""

from __future__ import annotations

import pickle
from dataclasses import replace

import pytest
from diskgraph.sampling_options import SamplingOptions
from diskgraph.sampling_plan import LayerSpec, SamplingPlan
from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchRequest,
    ConsumerToken,
    SamplingContractError,
)


def test_request_compares_typed_options_and_rejects_wrong_plan() -> None:
    """Request identity includes structural options instead of a serialized plan digest."""
    options = SamplingOptions(stage="train", batch_size=2)
    request = BatchRequest(
        ArtifactIdentity("snapshot", "task", "mapping"),
        ConsumerToken("job", 0, "train", 0),
        0,
        0,
        1,
        0,
        (1, 2),
        options,
        42,
        1,
    )
    assert request.validate() is options
    assert type(request.options) is SamplingOptions
    restored = pickle.loads(pickle.dumps(request))
    assert restored == request
    assert (
        replace(
            request, options=replace(options, sampling_plan=SamplingPlan([LayerSpec()]))
        )
        != request
    )
    with pytest.raises(SamplingContractError, match="seed schedule"):
        replace(request, options=replace(options, num_workers=2)).validate()
    with pytest.raises(SamplingContractError, match="wire version"):
        replace(request, wire_version=1).validate()
