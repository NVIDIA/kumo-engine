"""Synthetic sales entity classification with the RelBench task contract

Metrics are demonstration only.

Install: pip install structured-data-models pandas torchmetrics tqdm relbench sentence-transformers
CPU relational sampling also requires a pyg-lib wheel matching PyTorch.
Follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
for the matching wheel before running this script.
Documentation: https://nvidia.github.io/structured-data-models/
"""

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from typing import Any, cast
from pathlib import Path
from types import SimpleNamespace
import runpy

import pandas as pd
import relbench
import torch
import torchmetrics
from relbench_companions.local_structured_data_models_lagged_targets import (
    add_lag_target,
)
from relbench_companions.local_structured_data_models_text_embeddings import embed_text
from tqdm import tqdm

import sdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["synthetic-sales"], default="synthetic-sales")
parser.add_argument("--task", choices=["purchase"], default="purchase")
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--context_mode", choices=("last", "rnd"), default="last")
parser.add_argument("--batch_size", type=int, default=10_000)
parser.add_argument("--max_test_steps", type=int, default=None)
parser.add_argument("--num_neighbors", type=int, nargs="*", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--num_lags", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--text_dim", type=int, default=0)
args = parser.parse_args()
if args.context_size < 2 or args.batch_size < 1 or args.num_estimators < 1:
    parser.error(
        "context-size must be at least two; batch-size and estimators must be positive"
    )

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collect Relational Data #####################################################
generate = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "synthetic-data/sdm/relational.py")
)["generate"]
data_dir = generate(Path("synthetic_data/sales"), args.seed)
customers = pd.read_csv(data_dir / "customers.csv").drop(
    columns=["__index_level_0__", "ADDRESSID"]
)
sales = pd.concat(
    [
        pd.read_csv(data_dir / f"salesdocuments_{split}.csv")
        for split in ["train", "test"]
    ],
    ignore_index=True,
)
sales["timestamp"] = pd.to_datetime(sales.CREATIONDATE + " " + sales.CREATIONTIME)
sales = sales[["SALESDOCUMENT", "CUSTOMER", "timestamp", "value"]]
db = SimpleNamespace(
    table_dict={
        "customers": SimpleNamespace(
            df=customers, pkey_col="CUSTOMER", fkey_col_to_pkey_table={}, time_col=None
        ),
        "sales": SimpleNamespace(
            df=sales,
            pkey_col="SALESDOCUMENT",
            fkey_col_to_pkey_table={"CUSTOMER": "customers"},
            time_col="timestamp",
        ),
    }
)
task = SimpleNamespace(
    entity_table="customers",
    entity_col="CUSTOMER",
    time_col="timestamp",
    target_col="target",
    timedelta=pd.Timedelta(days=30),
    task_type=relbench.base.TaskType.BINARY_CLASSIFICATION,
)
data = sdm.RelationalData(
    tables={
        name: sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=sdm.infer_stypes(
                table.df.head(10_000),
                overrides={
                    cast(str, table.pkey_col): "id",
                    **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
                },
                text=(
                    "infer"
                    if args.text_dim > 0 and name == task.entity_table
                    else "drop"
                ),
                unsupported="drop",
            ),
        )
        for name, table in db.table_dict.items()
    },
    relationships=[
        {
            "left_table": left_table,
            "left_column": left_column,
            "right_table": right_table,
            "right_column": cast(str, db.table_dict[right_table].pkey_col),
        }
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ],
)

# Collect Task Table ##########################################################
if task.task_type not in {
    relbench.base.TaskType.BINARY_CLASSIFICATION,
    relbench.base.TaskType.MULTICLASS_CLASSIFICATION,
    relbench.base.TaskType.REGRESSION,
}:
    quit(f"Unsupported task {task.task_type.value!r}")

dfs = [
    pd.read_csv(data_dir / f"task_{split}.csv", parse_dates=["timestamp"])
    for split in ["train", "val", "test"]
]
df = pd.concat(dfs, ignore_index=True)
df = add_lag_target(df, task, args.num_lags, len(dfs[0]) + len(dfs[1]))

if task.task_type == relbench.base.TaskType.REGRESSION:
    target_stype = "numerical"
else:
    target_stype = "categorical"

task_table = sdm.TableTensor.from_pandas(
    df=df,
    stypes={
        task.entity_col: "id",
        task.time_col: "datetime",
        task.target_col: target_stype,
        **{
            f"{task.target_col}_lag_{lag}": target_stype
            for lag in range(1, args.num_lags + 1)
        },
    },
)

if args.text_dim > 0:  # Embed entity table text:
    data.tables[task.entity_table] = embed_text(  # type: ignore
        table=data.tables[task.entity_table],
        text_model="sentence-transformers/all-MiniLM-L6-v2",
        text_dim=args.text_dim,
        mask=torch.from_numpy(
            db.table_dict[task.entity_table]
            .df[db.table_dict[task.entity_table].pkey_col]
            .isin(df[task.entity_col])
            .to_numpy()
        ),
        device=device,
    )

context, query = task_table.split([len(dfs[0]) + len(dfs[1]), len(dfs[2])])
num_estimators = args.num_estimators
if len(context) > args.context_size:  # Sample different context per estimator:
    width = args.context_size * num_estimators
    if args.context_mode == "last" and len(context) > width:
        order = context.datetime.reshape(-1).argsort(descending=True)
        context = context[order[:width]]
    repeats = math.ceil(width / len(context))
    perm = torch.cat([torch.randperm(len(context)) for _ in range(repeats)])
    context = context[perm[:width]]
    if num_estimators > 1:
        context = context.unflatten(0, (num_estimators, args.context_size))
        query = query.expand(num_estimators, *query.size())
        num_estimators = None  # Context provides ensemble dimension.


# Create Subgraph Sampler #####################################################
sampler = data.sampler(
    time_columns={
        name: table.time_col
        for name, table in db.table_dict.items()
        if table.time_col is not None
    },
)
kwargs: dict[str, Any] = {
    "task_link": {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(str, db.table_dict[task.entity_table].pkey_col),
    },
    "num_neighbors": args.num_neighbors,
    "task_time_column": task.time_col,
}

# Execute Model ###############################################################
model = sdm.models.KumoRelational(device=device)

context, related_tables = sampler(context, **kwargs).to(device)
with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
    model.fit(
        x=context.drop_columns(task.target_col),
        y=context[task.target_col],
        related_tables=related_tables,
        num_estimators=num_estimators,
    )

if task.task_type == relbench.base.TaskType.REGRESSION:
    metric = torchmetrics.regression.MeanAbsoluteError()
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    metric = torchmetrics.classification.BinaryAUROC()
    positive_class = sorted(dfs[0][task.target_col].dropna().unique().tolist())[-1]
else:
    metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=df[task.target_col].nunique(), average="micro"
    )
metric = metric.to(device)
# Batch predictions: preserve test-table order while sampling label-free query features.
result_parts = []
scored_rows = 0
for batch in tqdm(query.split(args.batch_size, -2)[: args.max_test_steps]):
    x_query = batch.drop_columns(task.target_col)
    y_query = batch[task.target_col].to(device)
    if num_estimators is None:
        y_query = y_query[0]
    with torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
        out = model.predict(*sampler(x_query, **kwargs).to(device))

    if task.task_type == relbench.base.TaskType.REGRESSION:
        pred = out["q500"].numerical.squeeze(-1)  # Median prediction.
        target = y_query.numerical.squeeze(-1)
    elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
        pred, target = sdm.evaluation.to_binary_class(out, y_query, positive_class)
    else:
        pred, target = sdm.evaluation.to_class_indices(out, y_query)
    metric.update(pred, target)
    output_frame = pd.DataFrame(
        out.numerical.detach().float().cpu().numpy(),
        columns=out.columns[sdm.Stype.numerical],
    )
    expected_rows = x_query.size(-2)
    if len(output_frame) != expected_rows:
        raise ValueError("Prediction count does not match the query batch")
    keys = (
        dfs[2]
        .iloc[scored_rows : scored_rows + expected_rows][
            [task.entity_col, task.time_col]
        ]
        .reset_index(drop=True)
    )
    result_parts.append(pd.concat([keys, output_frame], axis=1))
    scored_rows += expected_rows

if task.task_type == relbench.base.TaskType.REGRESSION:
    print(f"MAE: {metric.compute():.4f} (target units)")
elif task.task_type == relbench.base.TaskType.BINARY_CLASSIFICATION:
    print(f"AUROC: {100 * metric.compute():.2f}")
else:
    print(f"ACC: {100 * metric.compute():.2f}")

if not result_parts or scored_rows < 2:
    raise ValueError("Need at least two scoring records; check max_test_steps")
results = pd.concat(result_parts, ignore_index=True)
assert (
    results[task.entity_col].tolist()
    == dfs[2][task.entity_col].iloc[:scored_rows].tolist()
)
# Numerical column names are quantiles for regression, class probabilities otherwise.
print(results.head(10).to_string(index=False))
results.to_csv("relbench_predictions.csv", index=False)
