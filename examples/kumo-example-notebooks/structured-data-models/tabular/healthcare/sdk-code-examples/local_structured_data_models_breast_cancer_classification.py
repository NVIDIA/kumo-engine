# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Breast cancer classification and embedding capture

Install: pip install structured-data-models pandas scikit-learn
Documentation: https://nvidia.github.io/structured-data-models/
"""

import runpy
from pathlib import Path

import pandas as pd
import sdm
import torch
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generate local synthetic files with the shared dataset generator.

generate = runpy.run_path(
    str(Path(__file__).resolve().parents[4] /
        "synthetic-data/sdm/generate.py"))["generate"]
df = generate(
    "breast_cancer_classification",
    Path("synthetic_data") / "breast_cancer_classification",
)

table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device=device,
)
model = sdm.models.TabICLv2(device=device)

# Default in-context learning forward pass:
with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model(
        x_context=table[:300].drop_columns("target"),
        y_context=table[:300, "target"],
        x_query=table[300:].drop_columns("target"),
        num_estimators=2,
    )

# Batch predictions
# Cache labeled context, then score every held-out row without target labels.
with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model.fit(
        x=table[:300].drop_columns("target"),
        y=table[:300, "target"],
        num_estimators=2,
    )
    scoring = table[300:].drop_columns("target")
    predictions = model.predict(scoring)

# Probability column names correspond to the dataset's target values (0 and 1).
results = pd.DataFrame(
    predictions.numerical.float().cpu().numpy(),
    columns=predictions.columns[sdm.Stype.numerical],
)
if len(results) != len(df) - 300:
    raise ValueError("Prediction rows do not match scoring rows")
results.insert(0, "source_row", df.index[300:])
print(results.head(10).to_string(index=False))
results.to_csv("breast_cancer_predictions.csv", index=False)
model.clear()

# Capturing embeddings:
embeddings: list[Tensor] = []


def _embedding(module: torch.nn.Module, args: tuple[Tensor, ...]) -> None:
    embeddings.append(args[0])


head = model.models["classification"].icl_block.head
handle = head.register_forward_pre_hook(_embedding)
with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model(
        x_context=table[:300].drop_columns("target"),
        y_context=table[:300, "target"],
        x_query=table[300:].drop_columns("target"),
    )
handle.remove()
