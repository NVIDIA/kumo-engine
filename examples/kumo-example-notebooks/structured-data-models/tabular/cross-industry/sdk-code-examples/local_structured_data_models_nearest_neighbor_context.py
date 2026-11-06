"""Compare full, random and nearest-neighbor context for synthetic classification.

Install: pip install structured-data-models pandas scikit-learn
Run with --dataset synthetic. Metrics are demonstration only.
Model weights download on first use.
This experiment follows the local-context idea in LoCalPFN (Thomas et al.,
NeurIPS 2024), using TabICLv2 and a random holdout rather than its benchmark.
Documentation: https://nvidia.github.io/structured-data-models/
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import sdm
from sdm.processing.execution import RecipeExecution
from nearest_neighbor_companions.local_structured_data_models_preprocessing_recipe import (
    knn_recipe,
)

DATASETS = {"synthetic": {"target": "class", "n_train": 4000}}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", choices=list(DATASETS), default="synthetic")
parser.add_argument("--k", type=int, default=100)
parser.add_argument("--num-estimators", type=int, default=8)
parser.add_argument(
    "--output", type=Path, default=Path("nearest_neighbor_predictions.csv")
)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
spec = DATASETS[args.dataset]
target_name = str(spec["target"])
# Generate local synthetic files with the shared dataset generator.
from pathlib import Path
import runpy

generate = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "synthetic-data/sdm/generate.py")
)["generate"]
df = generate("nearest_neighbor", Path("synthetic_data") / "nearest_neighbor")
n_train = min(int(spec["n_train"]), int(len(df) * 0.8))
if not 1 <= args.k < n_train < len(df) - 1:
    raise ValueError("Require 1 <= k < training rows and at least two scoring rows")
if args.num_estimators < 1:
    raise ValueError("num-estimators must be positive")
# Build one category vocabulary so row labels retain the same meaning in both splits.
table = sdm.TableTensor.from_pandas(
    df,
    stypes=sdm.infer_stypes(df, overrides={target_name: "categorical"}),
    device=device,
)
train, test = table.split(n_train)
class_values = sorted(df[target_name].dropna().unique().tolist(), key=str)
class_names = [str(v) for v in class_values]
y_true = pd.Categorical(df[target_name].iloc[n_train:], categories=class_values).codes
model = sdm.models.TabICLv2(task="classification", device=device)
recipe = knn_recipe()

# Fit retrieval preprocessing only on context; labels are absent from query features.
execution = RecipeExecution(copy.deepcopy(recipe))
contexts = execution.fit_transform(
    x=train.drop_columns(target_name),
    y=train[target_name],
    related_tables=None,
    num_members=1,
)
queries = execution.transform(x=test.drop_columns(target_name), related_tables=None)
train_features = contexts[0].x.numerical
query_features = queries[0].x.numerical
mean = train_features.mean(0)
scale = train_features.std(0).clamp_min(1e-8)
train_features = (train_features - mean) / scale
query_features = (query_features - mean) / scale


def neighbors(features, exclude_self=False):
    # Chunk the distance matrix; never allocate all query-by-training distances.
    parts = []
    for start in range(0, len(features), 128):
        distances = torch.cdist(features[start : start + 128], train_features)
        if exclude_self:
            rows = torch.arange(len(distances), device=device)
            distances[rows, rows + start] = torch.inf
        parts.append(distances.topk(args.k, largest=False).indices)
    return torch.cat(parts)


query_neighbors = neighbors(query_features)


def predict(context, scoring):
    with (
        torch.inference_mode(),
        torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"),
    ):
        out = model(
            x_context=context.drop_columns(target_name),
            y_context=context[target_name],
            x_query=scoring,
            recipe=recipe,
            num_estimators=args.num_estimators,
        )
    # Reindex by actual class names, including classes absent from a local context.
    frame = pd.DataFrame(
        out.numerical.float().cpu().numpy(), columns=out.columns[sdm.Stype.numerical]
    )
    return frame.reindex(columns=class_names, fill_value=0.0).to_numpy()


def auc(probabilities):
    if len(class_names) == 2:
        return roc_auc_score(y_true, probabilities[:, 1])
    return roc_auc_score(
        y_true, probabilities, labels=np.arange(len(class_names)), multi_class="ovr"
    )


# Batch predictions: the held-out scoring population never supplies target labels.
scoring = test.drop_columns(target_name)
full_predictions = predict(train, scoring)
print(f"Full-context AUROC: {auc(full_predictions):.3f}")
random_scores = []
for draw in range(5):
    generator = torch.Generator(device=device).manual_seed(draw)
    indices = torch.randperm(n_train, device=device, generator=generator)[: args.k]
    random_scores.append(auc(predict(train[indices], scoring)))
print(f"Random-context mean AUROC: {np.mean(random_scores):.3f}")

# Score each held-out row from its own labeled neighborhood, then save one aligned batch.
local_predictions = np.concatenate(
    [
        predict(train[indices], scoring[i : i + 1])
        for i, indices in enumerate(query_neighbors)
    ]
)
if len(local_predictions) != len(df) - n_train:
    raise ValueError("Prediction count does not match scoring population")
results = pd.DataFrame(local_predictions, columns=class_names)
results.insert(0, "source_row", df.index[n_train:])
assert results["source_row"].tolist() == df.index[n_train:].tolist()
print(f"Nearest-neighbor AUROC: {auc(local_predictions):.3f}")
print(results.head(10).to_string(index=False))
# Each class-named column is its probability; source_row identifies the synthetic row.
args.output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(args.output, index=False)
