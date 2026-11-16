# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.diskgraph._diskgraph_legacy import resolve_split_config


@pytest.mark.parametrize("seed", [None, 7])
def test_resolve_split_config_normalizes_random_seed(seed):
    config, family = resolve_split_config(
        {
            "ratio": [0.8, 0.2],
            "seed": seed
        },
        "RandomSplit",
    )

    assert config == {"ratio": [0.8, 0.2], "rnd_seed": seed}
    assert family == "RandomSplit"


def test_resolve_split_config_rejects_conflicting_random_seeds():
    with pytest.raises(RuntimeError, match="seed and rnd_seed conflict"):
        resolve_split_config(
            {
                "seed": 7,
                "rnd_seed": 8
            },
            "RandomSplit",
        )
