# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install: pip install structured-data-models pandas scikit-learn sentence-transformers

# # Predict passage readability with text features
#
# Predict synthetic passage easiness from passage text, metadata, and readability indices using Structured Data Models and pretrained TabICLv2.
# This local script uses CPU by default and requires Python 3.11 or later.
# See the [TabICLv2 API](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.TabICLv2) for the prediction interface.
#
# ## Setup
#
# Install the latest package and its optional text encoder, before running this script.
# See the [SDM installation guide](https://nvidia.github.io/structured-data-models/install.html) for package requirements.

# ### Parameters
#
# Use Sentence Transformer embeddings followed by PCA to represent text.
# The seed controls the random holdout, and eight estimators combine predictions from different preprocessing configurations.
# CPU is the default; set `device` to `torch.device("cuda")` on GPU compute.
# The output path is a new writable directory on this machine.

# Modified for the consolidated examples: Databricks setup and batch outputs.

import argparse
import runpy
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import sdm
import sdm.processing as sp
import torch

parser = argparse.ArgumentParser(
    description="Predict synthetic passage readability")
parser.add_argument("--text-processor", choices=("embed", "tfidf", "none"),
                    default="embed")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
seed = 0
device = torch.device("cpu")
num_estimators = 8
output_path = (Path(tempfile.mkdtemp(prefix="clear_readability_")) /
               "predictions.parquet")
if args.output is not None:
    output_path = args.output
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(seed)

# ## Generate synthetic example data
# Passages vary in length and vocabulary; metrics are demonstration only.

generate_readability = runpy.run_path(
    str(
        Path(__file__).resolve().parents[4] /
        "synthetic-data/sdm/readability.py"))["generate"]
data_path = generate_readability(Path("synthetic_data/readability"))
arrow_table = pq.read_table(data_path)
target_name = "BT Easiness"
print(f"Passages: {arrow_table.num_rows}, columns: {arrow_table.num_columns}")
print(arrow_table.select(["ID", "Title", target_name]).slice(0, 5).to_pandas())

# ## Prepare labeled context
#
# Reserve about 80% of shuffled passages as context and hold out the rest for scoring.
# Column type inference and category dictionaries use the full table before splitting; this example retains metadata and other readability indices, so its metrics describe this random holdout rather than a text-only or official benchmark.
# Keep the original `ID` values alongside the same shuffle to identify each output; exclude this identifier from model features.

table = sdm.TableTensor.from_arrow(
    table=arrow_table.drop(["ID"]),
    stypes=sdm.infer_stypes(
        arrow_table.drop(["ID"]),
        text="off" if args.text_processor == "none" else "infer",
    ),
    device=device,
)
row_order = torch.randperm(table.size(0), device=device)
shuffled_ids = arrow_table["ID"].take(row_order.cpu().numpy())
table = table[row_order]
context_rows = int(0.8 * table.size(0))
context, query = table.split(context_rows)
query_ids = shuffled_ids.slice(context_rows)
print(f"Context rows: {context.size(0)}, held-out rows: {query.size(0)}")

# ## Configure pretrained regression
#
# TabICLv2 predicts from labeled context without a separate training loop.
# Apply the [text processor](https://nvidia.github.io/structured-data-models/api/generated/sdm.processing.SentenceTransformer) only to text columns, then compress their embeddings to 64 features with PCA fitted on context rows.
# The default recipe handles the remaining numerical and categorical features.

model = sdm.models.TabICLv2(task="regression", device=device)
recipe = model.default_recipe()
if args.text_processor != "none":
    text_processor = (sp.TFIDF(ngram_range=(
        4, 6), max_features=256) if args.text_processor == "tfidf" else [
            sp.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
            sp.PCA(64),
        ])
    recipe.prepend_features(sp.StypeDispatch(text=text_processor))

# ## Batch predictions
#
# Score every held-out passage using the configured pretrained model and labeled context.
# Remove `BT Easiness` from both context and scoring features; only the context labels go to the model.

x_context = context.drop_columns(target_name)
y_context = context[:, target_name]
x_query = query.drop_columns(target_name)
assert x_query.size(0) == len(query_ids)

# Predict the held-out batch in one model call.
# The recipe fits preprocessing on context rows and transforms query rows with that state.

with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    quantiles = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        recipe=recipe,
        num_estimators=num_estimators,
    ).numerical

# The recipe averages estimators; the remaining axis holds 999 quantiles.
predictions = quantiles.mean(dim=-1)
assert predictions.shape == (x_query.size(0), )

# Each input passage produces one row: `ID` is the synthetic passage identifier, and `predicted_bt_easiness` is the mean of its predicted quantiles.
# Verify the row count and ID order, then inspect a sample.

results = pa.table({
    "ID": query_ids,
    "predicted_bt_easiness": predictions.detach().cpu().numpy(),
})
assert results.num_rows == query.size(0)
assert results["ID"].equals(query_ids)
print(results.slice(0, 10).to_pandas())

# Save all batch predictions as Parquet.
# The printed local path is temporary; copy the file to retained storage before the compute is terminated if you need it later.

pq.write_table(results, output_path)
print(f"Saved {results.num_rows} predictions to {output_path}")

# ## Evaluate the holdout
#
# Compare predictions with the held-out labels only after scoring.
# RMSE and MAE are in BT Easiness units; lower values mean smaller errors, and RMSE gives larger errors more weight.

y_query = query[:, target_name].numerical.squeeze(-1)
rmse = (predictions - y_query).pow(2).mean().sqrt()
mae = (predictions - y_query).abs().mean()
print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f} (BT Easiness units)")
