# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predict santander value prediction challenge with synthetic example data.

Generate local files from the repository root:
    python synthetic-data/sdm/generate.py santander_value_prediction_challenge --output data/santander_value_prediction_challenge
Then pass --data-dir data/santander_value_prediction_challenge to this script.
Metrics are demonstration only; these are not benchmark or clinical results.
Install: pip install structured-data-models pandas scikit-learn sentence-transformers
Documentation: https://nvidia.github.io/structured-data-models/
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

import sdm
import sdm.processing as sp

ID_COLUMN = "ID"
TARGET_COLUMN = "target"


def _build_model(
    name: str,
    device: torch.device,
    local_checkpoint: Path | None = None,
) -> sdm.models.ICLModel:
    if name == "tabiclv2":
        return sdm.models.TabICLv2(task="regression", device=device)
    size = "small" if name == "kumo-small" else "large"
    if local_checkpoint is None:
        return sdm.models.KumoTabular(task="regression", size=size, device=device)
    model = sdm.models.KumoTabular(
        task="regression", size=size, pretrained=False, device=device
    )
    state = torch.load(local_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _wide_column_recipe(model: sdm.models.ICLModel, max_columns: int) -> sp.Recipe:
    """Select a different feature subset per estimator for wide tables."""
    recipe = model.default_recipe()
    for processor in recipe.features.modules():
        if isinstance(processor, sp.SelectColumns):
            processor.method = "round_robin"
            processor.max_columns = max_columns
    return recipe


def _predict(
    model: sdm.models.ICLModel,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    recipe: sp.Recipe,
    num_estimators: int,
    max_context_size: int | None,
    query_batch_size: int,
    generator: torch.Generator,
) -> sdm.TableTensor:
    """Cache labeled context, score label-free rows, then release the cache.

    Optional context subsampling bounds attention cost per estimator."""
    device = x_context.device
    effective_num_estimators: int | None = num_estimators
    if max_context_size is not None and x_context.size(0) > max_context_size:
        num_repeats = math.ceil(num_estimators * max_context_size / x_context.size(0))
        perm = torch.cat(
            [
                torch.randperm(x_context.size(0), generator=generator, device=device)
                for _ in range(num_repeats)
            ]
        )[: num_estimators * max_context_size]
        shape = (num_estimators, max_context_size)
        x_context = x_context[perm].unflatten(0, shape)
        y_context = y_context[perm].unflatten(0, shape)
        x_query = x_query.expand(num_estimators, *x_query.size())
        effective_num_estimators = None

    with (
        torch.inference_mode(),
        torch.amp.autocast(device.type, torch.float16, enabled=x_context.is_cuda),
    ):
        model.fit(
            x=x_context,
            y=y_context,
            recipe=recipe,
            num_estimators=effective_num_estimators,
            generator=generator,
        )
        # Row dim is always -2 (there may be a leading ensemble dim from
        # the max_context_size branch above), unlike the plain dim=0
        # split for an unbatched context.
        out = torch.cat(
            [model.predict(batch) for batch in x_query.split(query_batch_size, dim=-2)],
            dim=-2,
        )
    model.clear()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory written by synthetic-data/sdm/generate.py.",
    )
    parser.add_argument(
        "--model",
        choices=("kumo-small", "kumo-large", "tabiclv2"),
        default="kumo-large",
    )
    parser.add_argument(
        "--local-checkpoint",
        type=Path,
        default=None,
        help=(
            "KumoTabular state_dict saved from the same model size and task; bypasses "
            "Hugging Face Hub download (for --model kumo-small/kumo-large)."
        ),
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=2000,
        help=(
            "Feature column cap (see _wide_column_recipe): raising this "
            "improves RMSLE but costs memory, especially for kumo-large, "
            "which OOMs at the full ~4991 columns with --num-estimators 8."
        ),
    )
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=5000,
        help=(
            "Predict in query-row chunks of this size, reusing the same "
            "fit()-cached context (see _predict), instead of one call over "
            "the full query; keeps memory bounded for the 49342-row "
            "test set at higher --max-columns. Set to a value >= the "
            "query size (or pass a very large number) to disable."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    if args.num_estimators < 1 or args.num_splits < 1:
        parser.error("num-estimators and num-splits must be positive")
    if args.max_context_size is not None and args.max_context_size < 2:
        parser.error("max-context-size must be at least two")
    if args.query_batch_size < 1:
        parser.error("query-batch-size must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "train.csv").drop(columns=[ID_COLUMN])
    train_frame[TARGET_COLUMN] = np.log1p(train_frame[TARGET_COLUMN])
    test_ids = pd.read_csv(args.data_dir / "test.csv")[ID_COLUMN]
    test_frame = pd.read_csv(args.data_dir / "test.csv").drop(columns=[ID_COLUMN])

    stypes = sdm.infer_stypes(train_frame, overrides={TARGET_COLUMN: "numerical"})
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)
    recipe = _wide_column_recipe(
        model, max_columns=min(args.max_columns, len(feature_stypes))
    )

    splitter = ShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    rmsles: list[float] = []
    for split_train_index, split_val_index in splitter.split(train_frame):
        split_train = sdm.TableTensor.from_pandas(
            df=train_frame.iloc[split_train_index],
            stypes=stypes,
            device=device,
        )
        split_val = sdm.TableTensor.from_pandas(
            df=train_frame.iloc[split_val_index],
            stypes=stypes,
            device=device,
        )
        out = _predict(
            model,
            x_context=split_train.drop_columns(TARGET_COLUMN),
            y_context=split_train[:, TARGET_COLUMN],
            x_query=split_val.drop_columns(TARGET_COLUMN),
            recipe=recipe,
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            query_batch_size=args.query_batch_size,
            generator=generator,
        )
        # out.numerical holds 999 quantile columns; their mean is the
        # point estimate (a point estimate).
        pred_log = out.numerical.mean(dim=-1).float().cpu().numpy()
        actual_log = (
            split_val[:, TARGET_COLUMN].numerical.squeeze(-1).float().cpu().numpy()
        )
        rmsles.append(float(np.sqrt(np.mean((pred_log - actual_log) ** 2))))

    rmsle = np.array(rmsles)
    print(
        f"{args.model}  val_rmsle={rmsle.mean():.4f}+-{rmsle.std():.4f}  "
        f"({args.num_splits} splits of {args.val_fraction:.0%})"
    )

    # Batch predictions: score competition rows without target labels.
    full_train = sdm.TableTensor.from_pandas(
        df=train_frame, stypes=stypes, device=device
    )
    query = sdm.TableTensor.from_pandas(
        df=test_frame, stypes=feature_stypes, device=device
    )
    out = _predict(
        model,
        x_context=full_train.drop_columns(TARGET_COLUMN),
        y_context=full_train[:, TARGET_COLUMN],
        x_query=query,
        recipe=recipe,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        query_batch_size=args.query_batch_size,
        generator=generator,
    )
    pred_log = out.numerical.mean(dim=-1).float().cpu().numpy()
    predicted_target = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None)

    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET_COLUMN: predicted_target})
    if len(submission) != len(test_frame) or len(submission) < 2:
        raise ValueError("Expected one prediction for each of at least two test rows")
    if submission[ID_COLUMN].tolist() != test_ids.tolist():
        raise ValueError("Prediction IDs do not match scoring input order")
    # Each row follows the original test-file order; columns hold the task outputs.
    print(submission.head(10).to_string(index=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
