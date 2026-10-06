# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    sources=["**/*.py"],
    dependencies=[
        "//3rdparty:diskgraph",
        "//packages/kumo-pql",
        "//3rdparty:faiss",
        "//3rdparty:filelock",
        "//3rdparty:fsspec",
        "//3rdparty:matplotlib",
        "//3rdparty:optuna",
        "//3rdparty:psutil",
        "//3rdparty:pytorch_lightning",
        "//3rdparty:sqlglot",
        "//3rdparty:tabulate",
        "//3rdparty:torchmetrics",
    ],
    # The extracted modules retain the monolith's existing type-checking debt.
    # Package-specific checks cover this boundary until that debt is resolved.
    skip_pyright=True,
)

resource(
    name="py_typed",
    source="py.typed",
)
