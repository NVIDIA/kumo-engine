# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict icr identify age related conditions with synthetic example data.

Generate local files, starting in the NVIDIA/kumo-engine checkout:
    cd examples/kumo-example-notebooks
    python synthetic-data/sdm/generate.py icr_identify_age_related_conditions --output data/icr_identify_age_related_conditions
Then pass --data-dir data/icr_identify_age_related_conditions to this script.
Metrics are demonstration only; these are not benchmark or clinical results.
Install: pip install structured-data-models pandas scikit-learn sentence-transformers
Documentation: https://nvidia.github.io/structured-data-models/
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sdm
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ID_COLUMN = "Id"
TARGET_COLUMN = "Class"
# to_binary_class compares this against the target's raw (int) category
# values, not their stringified column-name form, so this must be an int.
POSITIVE_CLASS = 1


def _build_model(
    name: str,
    device: torch.device,
    local_checkpoint: Path | None = None,
) -> sdm.models.ICLModel:
    if name == "tabiclv2":
        return sdm.models.TabICLv2(task="classification", device=device)
    size = "small" if name == "kumo-small" else "large"
    if local_checkpoint is None:
        return sdm.models.KumoTabular(task="classification", size=size,
                                      device=device)
    model = sdm.models.KumoTabular(task="classification", size=size,
                                   pretrained=False, device=device)
    state = torch.load(local_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _predict(
    model: sdm.models.ICLModel,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    num_estimators: int,
    max_context_size: int | None,
    generator: torch.Generator,
) -> sdm.TableTensor:
    """Cache labeled context, score label-free rows, then release the cache.

    Optional context subsampling bounds attention cost per estimator."""
    device = x_context.device
    effective_num_estimators: int | None = num_estimators
    if max_context_size is not None and x_context.size(0) > max_context_size:
        num_repeats = math.ceil(num_estimators * max_context_size /
                                x_context.size(0))
        perm = torch.cat([
            torch.randperm(x_context.size(0), generator=generator,
                           device=device) for _ in range(num_repeats)
        ])[:num_estimators * max_context_size]
        shape = (num_estimators, max_context_size)
        x_context = x_context[perm].unflatten(0, shape)
        y_context = y_context[perm].unflatten(0, shape)
        x_query = x_query.expand(num_estimators, *x_query.size())
        effective_num_estimators = None

    with (
            torch.inference_mode(),
            torch.amp.autocast(device.type, torch.float16,
                               enabled=x_context.is_cuda),
    ):
        model.fit(
            x=x_context,
            y=y_context,
            num_estimators=effective_num_estimators,
            generator=generator,
        )
        out = model.predict(x_query)
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
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_estimators < 1 or args.num_splits < 1:
        parser.error("num-estimators and num-splits must be positive")
    if args.max_context_size is not None and args.max_context_size < 2:
        parser.error("max-context-size must be at least two")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "train.csv")
    train_frame.columns = train_frame.columns.str.strip()
    train_frame = train_frame.drop(columns=[ID_COLUMN])

    stypes = sdm.infer_stypes(train_frame,
                              overrides={TARGET_COLUMN: "categorical"})
    model = _build_model(args.model, device, args.local_checkpoint)

    splitter = StratifiedShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    balanced_log_losses: list[float] = []
    aucs: list[float] = []
    split_train = split_val = split_val_index = None
    for split_train_index, split_val_index in splitter.split(
            train_frame, train_frame[TARGET_COLUMN]):
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
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            generator=generator,
        )
        scores, target = sdm.evaluation.to_binary_class(
            out,
            split_val[:, TARGET_COLUMN],
            positive_class=POSITIVE_CLASS,
        )
        scores_np = scores.float().cpu().numpy()
        target_np = target.cpu().numpy()
        class_counts = np.bincount(target_np.astype(np.int64))
        sample_weight = np.where(target_np, 1.0 / class_counts[1],
                                 1.0 / class_counts[0])
        balanced_log_losses.append(
            log_loss(target_np, scores_np, sample_weight=sample_weight))
        aucs.append(roc_auc_score(target_np, scores_np))

    loss = np.array(balanced_log_losses)
    auc = np.array(aucs)
    print(f"{args.model}  val_balanced_log_loss={loss.mean():.4f}"
          f"+-{loss.std():.4f}  val_auc={auc.mean():.4f}+-{auc.std():.4f}  "
          f"({args.num_splits} splits of {args.val_fraction:.0%})")

    # Batch predictions: the final validation fold is an illustrative scoring population.
    # No validation labels enter prediction; class_1 is the positive-class probability.
    assert split_train is not None and split_val is not None
    assert split_val_index is not None
    scoring = split_val.drop_columns(TARGET_COLUMN)
    out = _predict(
        model,
        x_context=split_train.drop_columns(TARGET_COLUMN),
        y_context=split_train[:, TARGET_COLUMN],
        x_query=scoring,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        generator=generator,
    )
    predictions = out[str(POSITIVE_CLASS)].numerical.squeeze(
        -1).float().cpu().numpy()
    ids = train_frame.iloc[split_val_index].index.to_numpy()
    if len(predictions) != len(ids) or len(ids) < 2:
        raise ValueError("Expected one score per held-out row")
    result = pd.DataFrame({"source_row": ids, "class_1": predictions})
    print(result.head(10).to_string(index=False))
    result.to_csv("icr_holdout_predictions.csv", index=False)


if __name__ == "__main__":
    main()
