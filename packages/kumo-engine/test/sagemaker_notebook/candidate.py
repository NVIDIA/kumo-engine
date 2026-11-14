# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the shared wheel identity for SageMaker's private runtime manifest."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from platform_candidate import main, prepare  # noqa: E402,F401


def candidate_path():
    value = os.environ.get('SAGEMAKER_TEST_CANDIDATE')
    return Path(value) if value else None


if __name__ == '__main__':
    main()
