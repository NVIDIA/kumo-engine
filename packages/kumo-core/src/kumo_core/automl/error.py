# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

import torch
from kumo_core.job_config import TrialStatus
from kumo_core.types import TaskType

logger = logging.getLogger(__name__)

# Default retry settings for GPU readiness check
_GPU_CHECK_MAX_RETRIES = 5
_GPU_CHECK_DELAY_SECONDS = 5


def check_gpu_ready(
    max_retries: int = _GPU_CHECK_MAX_RETRIES,
    delay_seconds: int = _GPU_CHECK_DELAY_SECONDS,
    require_gpu: bool | None = None,
) -> None:
    """Verify an available GPU is actually usable by PyTorch.

    CUDA visibility alone is insufficient when NVML or the device driver has
    not initialized. MPS availability likewise must be followed by a real
    allocation before training starts.

    Args:
        max_retries: Maximum number of attempts before giving up.
        delay_seconds: Seconds to wait between retry attempts.
        require_gpu: If True (or env KUMO_REQUIRE_GPU is set), raise when no
            CUDA or MPS device is available instead of silently allowing a CPU
            fallback. Defaults to the KUMO_REQUIRE_GPU env var.

    Raises:
        RuntimeError: If GPU fails to initialize after all retries.
    """
    if require_gpu is None:
        require_gpu = os.environ.get('KUMO_REQUIRE_GPU',
                                     '').strip().lower() in ('1', 'true', 'yes')
    device_type = ('cuda' if torch.cuda.is_available() else
                   'mps' if torch.backends.mps.is_available() else None)
    if device_type is None:
        if require_gpu:
            # A GPU deployment that has lost its GPU must FAIL LOUDLY, not
            # silently train on CPU (idle GPU, orders-of-magnitude slower,
            # job still reports success). GPU loss commonly follows a
            # cgroup/daemon event (systemd reload, apt upgrade, docker
            # hiccup) or a driver desync.
            raise RuntimeError(
                "GPU required (KUMO_REQUIRE_GPU) but no CUDA or MPS device is "
                "available — refusing to silently fall back to CPU. Verify "
                "that the GPU image is running, the accelerator resource is "
                "assigned, and its driver/runtime is healthy.")
        logger.info("No GPU device available, skipping GPU readiness check")
        return

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            if device_type == 'cuda':
                # This is the critical check - it requires NVML to work.
                torch.cuda.set_device(0)
                _ = torch.zeros(1, device='cuda:0')
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            else:
                _ = torch.zeros(1, device='mps')
                torch.mps.synchronize()
                torch.mps.empty_cache()
            logger.info("GPU readiness check passed for %s on attempt %d",
                        device_type, attempt + 1)
            return
        except Exception as e:
            last_error = e
            logger.warning(
                f"GPU readiness check attempt {attempt + 1}/{max_retries} "
                f"failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)

    raise RuntimeError(
        f"{device_type.upper()} GPU failed to initialize after "
        f"{max_retries} attempts. Last error: {last_error}") from last_error


# We use a Tesla T4 GPU which has 16 GB of RAM:
TOTAL_GPU_BYTES = 17_179_869_184
if torch.cuda.is_available():
    TOTAL_GPU_BYTES = torch.cuda.get_device_properties('cuda').total_memory


def to_megabytes(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


message = ("Out of {num_experiments} experiment(s), {num_failed_experiments} "
           "experiment(s) failed due to a GPU out-of-memory issue.")

batch_reduction_success_message = (
    'The batch size has been reduced to avoid GPU out-of-memory issues. '
    'Refer to the model plan for the final batch size.')

batch_reduction_failure_message = (
    'Despite re-attempting the training with smaller batch sizes, we were '
    'unable to avoid GPU out-of-memory issues.')

module_size_message = (
    "The current model parameters require approximately {total_memory:.2f}MB "
    "of GPU memory, which amounts to roughly {percent:.2f}% of available GPU "
    "memory. Here are the memory requirements for each component:\n"
    "- Encoder: {encoder_num_params} parameters ({encoder_memory:.2f}MB)\n"
    "- GNN: {model_num_params} parameters ({model_memory:.2f}MB)\n"
    "- Head: {head_num_params} parameters ({head_memory:.2f}MB)\n"
    "- Total: {total_num_params} parameters ({total_memory:.2f}MB)\n"
    "Furthermore, the GNN is using {num_hops} message passing layers, each "
    "with a hidden feature dimensionality of {channels}. Together with "
    "{num_aggrs} aggregation operators, this results in an intermediate "
    "feature size of {total_channels}.")

text_embedding_size_message = (
    "In addition, the current model is using {num_tokens} tokens to "
    "embed text. This consumes about {memory:.2f}MB of GPU memory, "
    "which amounts to roughly {percent:.2f}% of available GPU memory.")

subgraph_size_message = (
    "Lastly, the sampled graph size contains about {num_nodes} nodes and "
    "about {num_edges} edges. Together with its features, this consumes "
    "approximately {memory:.2f}MB of GPU memory to store the input data, "
    "which amounts to roughly {percent:.2f}% of available GPU memory.")

rhs_feature_size_message = (
    "In order to store the target entity features, an additional "
    "{memory:.2f}MB of GPU memory is required, which amounts to roughly "
    "{percent:.2f}% of available GPU memory.")

module_size_action = (
    "In order to decrease GPU memory requirements, consider excluding high "
    "cardinality columns from the graph, or decrease the number of GNN "
    "layers, their hidden feature dimensionality and their number of "
    "aggregation operators.")

module_size_link_pred_action = (
    "For link prediction tasks, consider decreasing the output embedding "
    "dimensionality, or reduce the number of target nodes to predict.")

text_embedding_size_action = (
    "Consider using a smaller text embedding model such as the default "
    "'glove.6B-50d'.")

multiple_text_encoders_action = (
    "Specifically, multiple text embedding models have been used to embed "
    "text across different columns. You can consolidate them by specifying "
    "a single text embedding model via the model plan.")

subgraph_size_action = (
    "Consider reducing the number of hops and the amount of sampled neighbors "
    "via the model plan. Additionally, consider decreasing the batch size. "
    "Alternatively, consider reducing the number of feature columns per table.")

rhs_feature_size_action = (
    "Please consider reducing the number of columns in your target entity "
    "table or the dimensionality of its input embeddings if it contains any.")


def construct_gpu_oom_message(
    num_experiments: int,
    failed_trials: list[TrialStatus],
    task_type: TaskType,
) -> str:
    trial = failed_trials[0]  # Pick the first one to display stats:

    msg = message.format(
        num_experiments=num_experiments,
        num_failed_experiments=len(failed_trials),
    )
    msg += batch_reduction_failure_message

    if trial.module_size is not None:
        msg += '\n\n'
        msg += module_size_message.format(
            encoder_num_params=trial.module_size.encoder_num_params,
            encoder_memory=to_megabytes(trial.module_size.encoder_num_bytes),
            model_num_params=trial.module_size.model_num_params,
            model_memory=to_megabytes(trial.module_size.model_num_bytes),
            head_num_params=trial.module_size.head_num_params,
            head_memory=to_megabytes(trial.module_size.head_num_bytes),
            total_num_params=trial.module_size.total_num_params,
            total_memory=to_megabytes(trial.module_size.total_num_bytes),
            percent=100 * trial.module_size.total_num_bytes / TOTAL_GPU_BYTES,
            channels=trial.module_size.channels,
            num_aggrs=trial.module_size.num_aggrs,
            num_hops=trial.module_size.num_hops,
            total_channels=(trial.module_size.channels *
                            trial.module_size.num_aggrs),
        )
        if (trial.module_size.total_num_bytes > 0.2 * TOTAL_GPU_BYTES
                or trial.module_size.channels > 128
                or trial.module_size.num_aggrs > 3
                or trial.module_size.num_hops > 2):
            msg += ' '
            msg += module_size_action
        if task_type.is_link_pred:
            msg += ' '
            msg += module_size_link_pred_action

        if len(trial.module_size.text_embedding_sizes) > 0:
            num_tokens = sum([
                size.num_tokens
                for size in trial.module_size.text_embedding_sizes
            ])
            num_bytes = sum([
                size.num_tokens * size.embedding_dim * 4
                for size in trial.module_size.text_embedding_sizes
            ])

            msg += '\n\n'
            msg += text_embedding_size_message.format(
                num_tokens=num_tokens,
                memory=to_megabytes(num_bytes),
                percent=100 * num_bytes / TOTAL_GPU_BYTES,
            )

            if any([
                    size.model_name != 'glove.6B-50d'
                    for size in trial.module_size.text_embedding_sizes
            ]):
                msg += ' '
                msg += text_embedding_size_action

            if len(trial.module_size.text_embedding_sizes) > 1:
                msg += ' '
                msg += multiple_text_encoders_action

    if trial.subgraph_size is not None:
        msg += '\n\n'

        num_bytes = trial.subgraph_size.num_feature_bytes
        num_bytes += 2 * 8 * trial.subgraph_size.num_edges

        msg += subgraph_size_message.format(
            num_nodes=trial.subgraph_size.num_nodes,
            num_edges=trial.subgraph_size.num_edges,
            memory=to_megabytes(num_bytes),
            percent=100 * num_bytes / TOTAL_GPU_BYTES,
        )

        if trial.subgraph_size.num_rhs_feature_bytes is not None:
            num_bytes += trial.subgraph_size.num_rhs_feature_bytes
            msg += ' '
            msg += rhs_feature_size_message.format(
                memory=to_megabytes(trial.subgraph_size.num_rhs_feature_bytes),
                percent=100 * trial.subgraph_size.num_rhs_feature_bytes /
                TOTAL_GPU_BYTES,
            )
            if num_bytes > 0.2 * TOTAL_GPU_BYTES:
                msg += ' '
                msg += rhs_feature_size_action

        if (num_bytes > 0.04 * TOTAL_GPU_BYTES
                or trial.subgraph_size.num_nodes > 150_000):
            msg += ' '
            msg += subgraph_size_action

    return msg


def construct_reduce_batch_msg():
    return batch_reduction_success_message
