# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end lifecycle tooling: datasets, baselines and metrics for the end-to-end flows."""

from pathlib import Path

E01_MANIFEST = Path(__file__).parent / "datasets/pins/aml-e01.json"
L01_MANIFEST = E01_MANIFEST.with_name("aml-l01.json")
