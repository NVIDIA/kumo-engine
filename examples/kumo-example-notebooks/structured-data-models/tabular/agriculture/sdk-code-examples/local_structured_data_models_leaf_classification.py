# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predict leaf classification with synthetic example data.

Generate local files from the repository root:
    python synthetic-data/sdm/generate.py leaf_classification --output data/leaf_classification
Then pass --data-dir data/leaf_classification to this script.
Metrics are demonstration only; these are not benchmark or clinical results.
Install: pip install structured-data-models pandas scikit-learn sentence-transformers
Documentation: https://nvidia.github.io/structured-data-models/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

import sdm

ID_COLUMN = "id"
TARGET_COLUMN = "species"


def _predict(
    model: sdm.models.TabICLv2,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    num_estimators: int,
    generator: torch.Generator,
) -> sdm.TableTensor:
    """Cache labeled context, score label-free rows, then release the cache.

    Optional context subsampling bounds attention cost per estimator."""
    device = x_context.device
    with (
        torch.inference_mode(),
        torch.amp.autocast(device.type, torch.float16, enabled=x_context.is_cuda),
    ):
        model.fit(
            x=x_context,
            y=y_context,
            num_estimators=num_estimators,
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
    parser.add_argument("--model", choices=("tabiclv2",), default="tabiclv2")
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    if args.num_estimators < 1 or args.num_splits < 1:
        parser.error("num-estimators and num-splits must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    train_frame = pd.read_csv(args.data_dir / "train.csv").drop(columns=[ID_COLUMN])
    test_ids = pd.read_csv(args.data_dir / "test.csv")[ID_COLUMN]
    test_frame = pd.read_csv(args.data_dir / "test.csv").drop(columns=[ID_COLUMN])
    classes = tuple(sorted(train_frame[TARGET_COLUMN].unique()))

    stypes = sdm.infer_stypes(train_frame, overrides={TARGET_COLUMN: "categorical"})
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = sdm.models.TabICLv2(task="classification", device=device)

    splitter = StratifiedShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    log_losses: list[float] = []
    accuracies: list[float] = []
    for split_train_index, split_val_index in splitter.split(
        train_frame, train_frame[TARGET_COLUMN]
    ):
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
            generator=generator,
        )
        scores, indices = sdm.evaluation.to_class_indices(
            out, split_val[:, TARGET_COLUMN], missing_score=0.0
        )
        scores_np = scores.float().cpu().numpy()
        indices_np = indices.cpu().numpy()
        log_losses.append(
            log_loss(indices_np, scores_np, labels=list(range(scores_np.shape[-1])))
        )
        accuracies.append(float((scores_np.argmax(-1) == indices_np).mean()))

    loss = np.array(log_losses)
    acc = np.array(accuracies)
    print(
        f"{args.model}  val_log_loss={loss.mean():.4f}+-{loss.std():.4f}  "
        f"val_accuracy={acc.mean():.4f}+-{acc.std():.4f}  "
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
        num_estimators=args.num_estimators,
        generator=generator,
    )
    columns = out.columns[sdm.Stype.numerical]
    indices = [columns.index(c) for c in classes]
    probabilities = out.numerical[..., indices].float().cpu().numpy()

    submission = pd.DataFrame(probabilities, columns=pd.Index(classes))
    submission.insert(0, ID_COLUMN, test_ids.to_numpy())
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
