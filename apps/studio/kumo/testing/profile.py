# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import torch

from io_utils import logging

logger = logging.getLogger(__name__)


@contextmanager
def measure_time_and_memory(num_iters: int = 1):
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  # type: ignore
    yield
    end.record()  # type: ignore
    torch.cuda.synchronize()
    gpu_time = start.elapsed_time(end)
    gpu_time_in_s = gpu_time / 1000
    print(f"total time: {gpu_time_in_s:.3f} s, "
          f"average per-iter time: {gpu_time / num_iters:.3f} ms, "
          f"peakmem: {torch.cuda.max_memory_allocated() / 2**30:.3f} GB")


def _get_filename_prefix() -> str:
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{hostname}_{timestamp}"


def _trace_handler(prof: torch.profiler.profile):
    """Hook to be called when the profiler is done profiling."""
    filename_prefix = _get_filename_prefix()
    prof.export_chrome_trace(f"{filename_prefix}.json")
    logger.warning(
        f"Open '{filename_prefix}.json' in 'https://ui.perfetto.dev'.")


# Alias to PyTorch profiler but with a callback hook saving a trace file.
profile = partial(
    torch.profiler.profile,
    on_trace_ready=_trace_handler,
    with_stack=True,
)


@contextmanager
def profile_memory():
    """Context manager to profile memory usage and dump a snapshot.

    Example:
        from kumo.testing.profile import profile_memory

        with profile_memory():
            run_something()
    """
    # Keep a max of 100,000 alloc/free events in the recorded history
    # leading up to the snapshot.
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    yield
    filename_prefix = _get_filename_prefix()
    torch.cuda.memory._dump_snapshot(f"{filename_prefix}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore
    logger.warning(
        f"Open '{filename_prefix}.pickle' in 'https://pytorch.org/memory_viz'."
    )
