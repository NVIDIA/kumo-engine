# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys


def test_actual_auto_trainer_does_not_load_platform_services():
    script = """
import json
import sys

from kumo_core.automl.auto_trainer import AutoTrainer

forbidden = ('rest', 'temporalio_commons', 'service_framework', 'statedb')
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in forbidden)
)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
