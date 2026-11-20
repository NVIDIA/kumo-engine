# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify notebook exceptions in the kernel, where their traceback still exists."""

import json
from pathlib import Path

from conformance.refusals import clean_refusal
from tests.native_fixtures import forbid_deprecated_engine


def execute(source, filename, namespace, evidence_path):
    try:
        with forbid_deprecated_engine():
            exec(compile(source, filename, "exec"), namespace)
    except Exception as error:
        evidence = clean_refusal(error)
        if evidence is not None:
            Path(evidence_path).write_text(
                json.dumps(
                    {
                        "evidence": {**evidence, "cell": filename},
                        "ename": type(error).__name__,
                        "evalue": str(error),
                    }
                )
            )
        raise
