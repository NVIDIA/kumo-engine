# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predict two sigma connect rental listing inquiries with synthetic example data.

Generate local files from the repository root:
    python synthetic-data/sdm/generate.py two_sigma_connect_rental_listing_inquiries --output data/two_sigma_connect_rental_listing_inquiries
Then pass --data-dir data/two_sigma_connect_rental_listing_inquiries to this script.
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
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

import sdm
import sdm.processing as sp

ID_COLUMN = "listing_id"
TARGET_COLUMN = "interest_level"
# Matches sample_submission.csv's column order exactly.
CLASSES = ("high", "medium", "low")
TEXT_COLUMNS = ("description", "features")
DATE_COLUMN = "created"
AGE_COLUMN = "days_since_listed"
DROP_COLUMNS = ("photos", "display_address", "street_address")


def _build_model(
    name: str,
    device: torch.device,
    local_checkpoint: Path | None = None,
) -> sdm.models.ICLModel:
    if name == "tabiclv2":
        return sdm.models.TabICLv2(task="classification", device=device)
    size = "small" if name == "kumo-small" else "large"
    if local_checkpoint is None:
        return sdm.models.KumoTabular(task="classification", size=size, device=device)
    model = sdm.models.KumoTabular(
        task="classification", size=size, pretrained=False, device=device
    )
    state = torch.load(local_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _text_recipe(
    model: sdm.models.ICLModel, text_model: str, embedding_dim: int
) -> sp.Recipe:
    """Embed text and fit PCA on the context before the default numeric recipe."""
    recipe = model.default_recipe()
    recipe.prepend_features(
        sp.StypeDispatch(
            text=[sp.SentenceTransformer(text_model), sp.PCA(embedding_dim)]
        )
    )
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
        # the max_context_size branch above).
        out = torch.cat(
            [model.predict(batch) for batch in x_query.split(query_batch_size, dim=-2)],
            dim=-2,
        )
    model.clear()
    return out


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path)
    frame["features"] = frame["features"].apply(lambda items: " ".join(items))
    return frame.drop(columns=list(DROP_COLUMNS))


def _with_age_feature(
    frame: pd.DataFrame, reference_date: pd.Timestamp
) -> pd.DataFrame:
    frame = frame.copy()
    created = pd.to_datetime(frame[DATE_COLUMN])
    frame[AGE_COLUMN] = (reference_date - created).dt.days.astype(float)
    return frame.drop(columns=[DATE_COLUMN])


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
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--text-embedding-dim", type=int, default=64)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=20000)
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

    train_frame = _load(args.data_dir / "train.json").drop(columns=[ID_COLUMN])
    test_frame_full = _load(args.data_dir / "test.json")
    test_ids = test_frame_full[ID_COLUMN]
    test_frame = test_frame_full.drop(columns=[ID_COLUMN])

    # One reference date (after the latest date in either split) so "age"
    # is on a comparable scale between train and test.
    reference_date = pd.to_datetime(
        pd.concat([train_frame[DATE_COLUMN], test_frame[DATE_COLUMN]])
    ).max() + pd.Timedelta(days=1)
    train_frame = _with_age_feature(train_frame, reference_date)
    test_frame = _with_age_feature(test_frame, reference_date)

    stypes = sdm.infer_stypes(
        train_frame,
        overrides={
            TARGET_COLUMN: "categorical",
            **dict.fromkeys(TEXT_COLUMNS, "text"),
        },
    )
    feature_stypes = {k: v for k, v in stypes.items() if k != TARGET_COLUMN}
    model = _build_model(args.model, device, args.local_checkpoint)
    recipe = _text_recipe(
        model,
        text_model=args.text_model,
        embedding_dim=args.text_embedding_dim,
    )

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
            recipe=recipe,
            num_estimators=args.num_estimators,
            max_context_size=args.max_context_size,
            query_batch_size=args.query_batch_size,
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
        recipe=recipe,
        num_estimators=args.num_estimators,
        max_context_size=args.max_context_size,
        query_batch_size=args.query_batch_size,
        generator=generator,
    )
    columns = out.columns[sdm.Stype.numerical]
    indices = [columns.index(c) for c in CLASSES]
    probabilities = out.numerical[..., indices].float().cpu().numpy()

    submission = pd.DataFrame(probabilities, columns=pd.Index(CLASSES))
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
