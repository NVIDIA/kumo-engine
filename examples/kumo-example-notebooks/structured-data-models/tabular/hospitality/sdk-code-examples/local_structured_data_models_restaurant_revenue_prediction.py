# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict restaurant revenue prediction with synthetic example data.

Generate local files, starting in the NVIDIA/kumo-engine checkout:
    cd examples/kumo-example-notebooks
    python synthetic-data/sdm/generate.py restaurant_revenue_prediction --output data/restaurant_revenue_prediction
Then pass --data-dir data/restaurant_revenue_prediction to this script.
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
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit

ID_COLUMN = "Id"
TARGET_COLUMN = "revenue"
DATE_COLUMN = "Open Date"
AGE_COLUMN = "days_since_open"


def _build_model(
    name: str,
    device: torch.device,
    local_checkpoint: Path | None = None,
) -> sdm.models.ICLModel:
    if name == "tabiclv2":
        return sdm.models.TabICLv2(task="regression", device=device)
    size = "small" if name == "kumo-small" else "large"
    if local_checkpoint is None:
        return sdm.models.KumoTabular(task="regression", size=size,
                                      device=device)
    model = sdm.models.KumoTabular(task="regression", size=size,
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
    query_batch_size: int,
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
        # Row dim is always -2 (there may be a leading ensemble dim from
        # the max_context_size branch above).
        out = torch.cat(
            [
                model.predict(batch)
                for batch in x_query.split(query_batch_size, dim=-2)
            ],
            dim=-2,
        )
    model.clear()
    return out


def _with_age_feature(frame: pd.DataFrame,
                      reference_date: pd.Timestamp) -> pd.DataFrame:
    frame = frame.copy()
    open_date = pd.to_datetime(frame[DATE_COLUMN], format="%m/%d/%Y")
    frame[AGE_COLUMN] = (reference_date - open_date).dt.days.astype(float)
    return frame.drop(columns=[DATE_COLUMN])


def _engineer_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    drop_city: bool,
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally remove city and compress numeric restaurant features."""
    if drop_city:
        train_frame = train_frame.drop(columns=["City"])
        test_frame = test_frame.drop(columns=["City"])

    if pca_components > 0:
        p_columns = [c for c in train_frame.columns if c.startswith("P")]
        pca = PCA(n_components=pca_components).fit(
            pd.concat([train_frame[p_columns], test_frame[p_columns]]))
        train_pca = pca.transform(train_frame[p_columns])
        test_pca = pca.transform(test_frame[p_columns])
        pc_columns = pd.Index(f"PC{i}" for i in range(1, pca_components + 1))
        train_frame = pd.concat(
            [
                train_frame.drop(columns=p_columns),
                pd.DataFrame(train_pca, columns=pc_columns,
                             index=train_frame.index),
            ],
            axis=1,
        )
        test_frame = pd.concat(
            [
                test_frame.drop(columns=p_columns),
                pd.DataFrame(test_pca, columns=pc_columns,
                             index=test_frame.index),
            ],
            axis=1,
        )

    return train_frame, test_frame


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
    parser.add_argument("--query-batch-size", type=int, default=20000)
    parser.add_argument(
        "--drop-city",
        action="store_true",
        help=(
            "Drop the sparse, high-cardinality City column (see _engineer_features)."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help=(
            "PCA-reduce the 37 P columns to this many components (0 disables)."
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

    train_frame = pd.read_csv(args.data_dir /
                              "train.csv").drop(columns=[ID_COLUMN])
    test_ids = pd.read_csv(args.data_dir / "test.csv")[ID_COLUMN]
    test_frame = pd.read_csv(args.data_dir /
                             "test.csv").drop(columns=[ID_COLUMN])

    # One reference date (after the latest date in either split) so "age"
    # is on a comparable scale between train and test.
    reference_date = pd.to_datetime(
        pd.concat([train_frame[DATE_COLUMN], test_frame[DATE_COLUMN]]),
        format="%m/%d/%Y",
    ).max() + pd.Timedelta(days=1)
    train_frame = _with_age_feature(train_frame, reference_date)
    test_frame = _with_age_feature(test_frame, reference_date)
    train_frame, test_frame = _engineer_features(
        train_frame,
        test_frame,
        drop_city=args.drop_city,
        pca_components=args.pca_components,
    )
    train_frame[TARGET_COLUMN] = np.log1p(train_frame[TARGET_COLUMN])

    stypes = sdm.infer_stypes(train_frame,
                              overrides={TARGET_COLUMN: "numerical"})
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)

    splitter = ShuffleSplit(
        n_splits=args.num_splits,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    rmses: list[float] = []
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
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            query_batch_size=args.query_batch_size,
            generator=generator,
        )
        # out.numerical holds 999 quantile columns; their mean is the
        # point estimate (a point estimate).
        pred_raw = np.expm1(out.numerical.mean(dim=-1).float().cpu().numpy())
        actual_raw = np.expm1(split_val[:, TARGET_COLUMN].numerical.squeeze(
            -1).float().cpu().numpy())
        rmses.append(float(np.sqrt(np.mean((pred_raw - actual_raw)**2))))

    rmse = np.array(rmses)
    print(f"{args.model}  val_rmse={rmse.mean():.1f}+-{rmse.std():.1f}  "
          f"({args.num_splits} splits of {args.val_fraction:.0%})")

    # Batch predictions: score competition rows without target labels.
    full_train = sdm.TableTensor.from_pandas(df=train_frame, stypes=stypes,
                                             device=device)
    query = sdm.TableTensor.from_pandas(df=test_frame, stypes=feature_stypes,
                                        device=device)
    out = _predict(
        model,
        x_context=full_train.drop_columns(TARGET_COLUMN),
        y_context=full_train[:, TARGET_COLUMN],
        x_query=query,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        query_batch_size=args.query_batch_size,
        generator=generator,
    )
    pred_log = out.numerical.mean(dim=-1).float().cpu().numpy()
    predicted_revenue = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None)

    submission = pd.DataFrame({
        ID_COLUMN: test_ids,
        "Prediction": predicted_revenue
    })
    if len(submission) != len(test_frame) or len(submission) < 2:
        raise ValueError(
            "Expected one prediction for each of at least two test rows")
    if submission[ID_COLUMN].tolist() != test_ids.tolist():
        raise ValueError("Prediction IDs do not match scoring input order")
    # Each row follows the original test-file order; columns hold the task outputs.
    print(submission.head(10).to_string(index=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
