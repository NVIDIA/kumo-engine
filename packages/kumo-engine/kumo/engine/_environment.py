# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A one-call environment report for support threads and bug reports."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List

from kumo_core.utils.torch_build import (
    CUDA_INDEX_HINT,
    UNSUPPORTED_BACKENDS,
    local_label,
    nvidia_driver_present,
    unsupported_backend,
)


def _torch_details() -> Dict[str, Any]:
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        return {"torch": None, "torch_import_error": str(error)}

    details: Dict[str, Any] = {
        "torch": torch.__version__,
        "torch_backend": local_label(torch.__version__) or "default",
        "torch_cuda": torch.version.cuda,
    }
    try:
        details["cuda_available"] = torch.cuda.is_available()
        if details["cuda_available"]:
            details["gpus"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception:  # noqa: BLE001
        details["cuda_available"] = False
    try:
        details["mps_available"] = torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        details["mps_available"] = False
    return details


def _problems(report: Dict[str, Any]) -> List[str]:
    version = report.get("torch")
    if version is None:
        return ["PyTorch is not importable."]

    problems = []
    backend = unsupported_backend(version)
    if backend is not None:
        problems.append(
            f"PyTorch {version} is {UNSUPPORTED_BACKENDS[backend]} build, "
            f"which Kumo cannot use. Reinstall with: {CUDA_INDEX_HINT}")
    elif report.get("torch_cuda") is None and nvidia_driver_present():
        problems.append(
            f"PyTorch {version} has no CUDA support, but this machine has an "
            f"NVIDIA driver, so Kumo will run on CPU. Reinstall with: "
            f"{CUDA_INDEX_HINT}")
    elif report.get(
            "torch_cuda") is not None and not report.get("cuda_available"):
        problems.append(
            "PyTorch has CUDA support but no GPU is visible. Check the NVIDIA "
            "driver and GPU access for this process.")
    return problems


def check_environment(*, verbose: bool = True) -> Dict[str, Any]:
    """Report the installed Kumo, PyTorch and device configuration.

    Prints a readable block and returns the same values as a dictionary. Use
    it when a Kumo run is slower than expected or fails to see a GPU, and
    include the output when reporting an installation problem.

    Args:
        verbose: print the report. Set to ``False`` to only return it.

    Returns:
        The report, including a ``problems`` list that is empty when nothing
        looks wrong.
    """
    from kumo.engine._provenance import environment
    from kumo.engine._scratch import get_scratch_dir

    report: Dict[str, Any] = environment().to_dict()
    report.pop("torch", None)
    report.update(_torch_details())
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["architecture"] = platform.machine()
    report["executable"] = sys.executable
    try:
        report["scratch_dir"] = str(get_scratch_dir())
    except Exception:  # noqa: BLE001
        report["scratch_dir"] = None
    report["problems"] = _problems(report)

    if verbose:
        width = max(len(key) for key in report)
        for key, value in report.items():
            if key == "problems":
                continue
            print(f"{key:<{width}}  {value}")
        problems = report["problems"]
        if problems:
            print()
            for problem in problems:
                print(f"PROBLEM: {problem}")
        else:
            print("\nNo environment problems detected.")
    return report
