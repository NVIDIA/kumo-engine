# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from kumo_core.automl.error import construct_gpu_oom_message
from kumo_core.job_config import (
    ModuleSize,
    SubgraphSize,
    TextEmbeddingSize,
    TrialStatus,
)
from kumo_core.types import TaskType


def test_construct_gpu_oom_message():
    trial = TrialStatus(
        trial=0,
        batch_size=16,
        succeed=False,
        stacktrace='RuntimeError: out of memory',
        module_size=ModuleSize(
            encoder_num_params=13_056,
            model_num_params=363_776,
            head_num_params=16_897,
            total_num_params=393_731,
            encoder_num_bytes=10_306_399_008,
            model_num_bytes=1_455_104,
            head_num_bytes=67_588,
            total_num_bytes=10_307_921_700,
            channels=256,
            num_aggrs=5,
            num_hops=3,
            text_embedding_sizes=[
                TextEmbeddingSize(
                    model_name='glove.6B-50d',
                    num_tokens=60_000,
                    embedding_dim=50,
                ),
                TextEmbeddingSize(
                    model_name='glove.840B-300d',
                    num_tokens=400_000,
                    embedding_dim=300,
                )
            ],
        ),
        subgraph_size=SubgraphSize(
            num_nodes=200_000,
            num_edges=1_000_000,
            num_features=200_000_000,
            num_feature_bytes=800_000_000,
            num_rhs_feature_bytes=100_000_000,
            batch_size=2048,
        ),
    )

    msg = construct_gpu_oom_message(
        1,
        [trial],
        TaskType.TEMPORAL_LINK_PREDICTION,
    )

    assert "Out of 1 experiment(s), 1 experiment(s) failed" in msg

    assert "we were unable to avoid GPU out-of-memory issues" in msg

    assert "require approximately 9830.40MB of GPU memory" in msg
    # The percentage is computed against `TOTAL_GPU_BYTES` which is set at
    # module load to either the hardcoded Tesla T4 value (17 GB) when CUDA
    # is unavailable, OR `torch.cuda.get_device_properties('cuda').total_memory`
    # when CUDA is available. Different runners have different GPUs, so the
    # exact percent varies (e.g. 60.00% on the hardcoded fallback,
    # ~65.92% on a 15.6 GB Tesla T4 with reserved memory). Loosen to a
    # regex match on the message shape rather than the GPU-dependent value.
    assert re.search(r'amounts to roughly \d+\.\d+% of available GPU memory',
                     msg)
    assert "Encoder: 13056 parameters (9828.95MB)" in msg
    assert "GNN: 363776 parameters (1.39MB)" in msg
    assert "Head: 16897 parameters (0.06MB)" in msg
    assert "Total: 393731 parameters (9830.40MB)" in msg
    assert "consider excluding high cardinality columns" in msg
    assert "decrease the number of GNN layers" in msg
    assert "consider decreasing the output embedding dimensionality" in msg

    assert "current model is using 460000 tokens to embed text" in msg
    assert "This consumes about 469.21MB of GPU memory" in msg
    assert "Consider using a smaller text embedding model" in msg
    assert "multiple text embedding models have been used to embed text" in msg

    assert "contains about 200000 nodes and about 1000000 edges" in msg
    assert "this consumes approximately 778.20MB of GPU memory" in msg
    assert "store the target entity features" in msg
    assert "Consider reducing the number of hops" in msg
