# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict itk fall 2018 with synthetic example data.

Generate local files, starting in the NVIDIA/kumo-engine checkout:
    cd examples/kumo-example-notebooks
    python synthetic-data/sdm/generate.py itk_fall_2018 --output data/itk_fall_2018
Then pass --data-dir data/itk_fall_2018 to this script.
Metrics are demonstration only; these are not benchmark or clinical results.
Install: pip install structured-data-models pandas scikit-learn sentence-transformers
Documentation: https://nvidia.github.io/structured-data-models/
"""

import argparse
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sdm
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# pandas infers the ARFF "class" column (values "false"/"true") as a
# native bool dtype, so sdm exposes it as a boolean-categorical target
# rather than a string one, keyed by the Python bool below.
TARGET_COLUMN = "class"
POSITIVE_CLASS = True
FALSE_POSITIVE_COST = 10.0
FALSE_NEGATIVE_COST = 500.0
# Bayes-optimal decision threshold minimizing expected cost under this
# asymmetric cost matrix: predict positive when p(positive) exceeds
# FP_cost / (FP_cost + FN_cost), not the usual 0.5.
DECISION_THRESHOLD = FALSE_POSITIVE_COST / (FALSE_POSITIVE_COST +
                                            FALSE_NEGATIVE_COST)


def _load_arff(path: Path) -> pd.DataFrame:
    """Parse a dense ARFF file into a DataFrame, treating "?" as missing."""
    lines = path.read_text().splitlines(keepends=True)
    columns = [
        line.split()[1] for line in lines
        if line.lower().startswith("@attribute")
    ]
    data_start = (next(i for i, line in enumerate(lines)
                       if line.lower().startswith("@data")) + 1)
    data = "".join(lines[data_start:])
    return pd.read_csv(io.StringIO(data), header=None, names=columns,
                       na_values="?")


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


def _cost(predicted_positive: np.ndarray, target_positive: np.ndarray) -> float:
    false_positives = (predicted_positive & ~target_positive).sum()
    false_negatives = (~predicted_positive & target_positive).sum()
    return FALSE_POSITIVE_COST * false_positives + FALSE_NEGATIVE_COST * false_negatives


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
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    if args.num_estimators < 1 or args.num_splits < 1:
        parser.error("num-estimators and num-splits must be positive")
    if args.max_context_size is not None and args.max_context_size < 2:
        parser.error("max-context-size must be at least two")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = _load_arff(args.data_dir / "aps_failure_training_set.arff")
    test_frame = _load_arff(args.data_dir / "aps_failure_test_set.arff").drop(
        columns=[TARGET_COLUMN])

    stypes = sdm.infer_stypes(train_frame,
                              overrides={TARGET_COLUMN: "categorical"})
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)

    splitter = StratifiedShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    aucs: list[float] = []
    kaggle_scores: list[float] = []
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
        predicted_positive = scores_np >= DECISION_THRESHOLD
        aucs.append(roc_auc_score(target_np, scores_np))
        cost = _cost(predicted_positive, target_np)
        # Matches the competition's own per-instance-normalized score (see
        # module docstring); normalizing by the fold size rather than a
        # fixed constant keeps this comparable across --val-fraction values.
        kaggle_scores.append(-cost / len(target_np))

    auc = np.array(aucs)
    score = np.array(kaggle_scores)
    print(f"{args.model}  val_auc={auc.mean():.4f}+-{auc.std():.4f}  "
          f"val_score={score.mean():.4f}+-{score.std():.4f}  "
          f"({args.num_splits} splits of {args.val_fraction:.0%}; "
          "val_score approximates the competition's own metric "
          "(-(10*FP + 500*FN)/N), higher is better; decision threshold="
          f"{DECISION_THRESHOLD:.4f})")

    # Batch predictions: score competition rows without target labels.
    full_train = sdm.TableTensor.from_pandas(
        df=train_frame,
        stypes=stypes,
        device=device,
    )
    query = sdm.TableTensor.from_pandas(
        df=test_frame,
        stypes=feature_stypes,
        device=device,
    )
    out = _predict(
        model,
        x_context=full_train.drop_columns(TARGET_COLUMN),
        y_context=full_train[:, TARGET_COLUMN],
        x_query=query,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        generator=generator,
    )
    positive_scores = (
        out[str(POSITIVE_CLASS)].numerical.squeeze(-1).float().cpu().numpy())
    predicted_class = np.where(positive_scores >= DECISION_THRESHOLD, "true",
                               "false")

    submission = pd.DataFrame({
        "ID": np.arange(1,
                        len(predicted_class) + 1),
        TARGET_COLUMN: predicted_class,
    })
    if len(submission) != len(test_frame) or len(submission) < 2:
        raise ValueError(
            "Expected one prediction for each of at least two test rows")
    # Each row follows the original test-file order; columns hold the task outputs.
    print(submission.head(10).to_string(index=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
