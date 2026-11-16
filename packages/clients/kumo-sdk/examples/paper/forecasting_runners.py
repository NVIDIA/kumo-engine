# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import timesfm
import torch
from chronos import Chronos2Pipeline
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
from tabpfn_time_series.features import CalendarFeature, RunningIndexFeature
from timesfm import TimesFM_2p5_200M_torch
from toto2 import Toto2Model
from tqdm import tqdm

from examples.paper.datasets.benchmarking_dataset import ForecastingTables
from examples.paper.metrics import CRPS_QUANTILE_LEVELS, QUANTILE_COLUMNS
from examples.paper.metrics import \
    has_forecast_context as _has_forecast_context
from examples.paper.metrics import \
    predict_context_frame as _predict_context_frame
from examples.paper.paper_utils import extract_temporal_features

# Candidate ARIMA orders for AIC-based per-entity model selection.
# Covers both stationary (d=0) and integrated (d=1) series.
_ARIMA_CANDIDATE_ORDERS: list[tuple[int, int, int]] = [
    (1, 0, 0),
    (2, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (1, 1, 0),
    (2, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
]


def _select_arima_order(
    y_train: np.ndarray,
    exog_train: np.ndarray | None,
    candidates: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Return the candidate order with the lowest AIC on the training data."""
    best_aic, best_order = np.inf, candidates[0]
    for order in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = SARIMAX(
                    y_train,
                    order=order,
                    exog=exog_train,
                    trend="c",
                ).fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except Exception:
            continue
    return best_order


def _limit_per_item(
    df: pd.DataFrame,
    entity_col: str,
    max_rows: int | None,
    keep: str,
) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    rows_per_item = max(1, max_rows // df[entity_col].nunique())
    parts = [
        getattr(item_df, keep)(rows_per_item)
        for _, item_df in df.groupby(entity_col, sort=False)
    ]
    return pd.concat(parts, ignore_index=True)


def run_seasonal_naive(
    tables: ForecastingTables,
    *,
    seasonal_period: int,
) -> pd.DataFrame:
    """Seasonal naive (Hyndman): y_hat_{T+h} = y_{T+h-m(k+1)}, k=(h-1)//m."""
    m = seasonal_period
    entity_col = tables.entity_col
    time_col = tables.time_col
    target_col = tables.target_col
    history = _predict_context_frame(tables)[[
        entity_col,
        time_col,
        target_col,
    ]].sort_values([entity_col, time_col])
    test = tables.test[[entity_col, time_col,
                        target_col]].sort_values([entity_col, time_col])
    hist_groups = {
        entity: df[target_col].to_numpy(dtype=float)
        for entity, df in history.groupby(entity_col)
    }
    parts: list[pd.DataFrame] = []
    for entity, test_group in test.groupby(entity_col):
        history_vals = hist_groups.get(entity)
        if history_vals is None or len(history_vals) < m:
            raise ValueError(f"Insufficient history for entity={entity!r}.")
        hist_len = len(history_vals)
        pred_vals: list[float] = []
        for h in range(1, len(test_group) + 1):
            k = (h - 1) // m
            idx = hist_len - 1 + h - m * (k + 1)
            pred_vals.append(float(history_vals[idx]))
        parts.append(
            pd.DataFrame({
                entity_col: entity,
                time_col: test_group[time_col].values,
                "prediction": np.asarray(pred_vals, dtype=float),
            }))
    return pd.concat(parts, ignore_index=True)


def _quantile_frame(values: np.ndarray) -> dict[str, np.ndarray]:
    """Map a ``(n_obs, 9)`` decile array to named CRPS quantile columns."""
    return {col: values[:, i] for i, col in enumerate(QUANTILE_COLUMNS)}


def save_forecasting_benchmark(
    save_dir: Path,
    *,
    dataset: str,
    model: str,
    tables: ForecastingTables,
    predictions: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    """Write metrics CSV + prediction/history parquet under a run directory."""
    metrics_path = save_dir / "metrics" / dataset / f"{model}.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": dataset,
        "model": model,
        **metrics
    }]).to_csv(metrics_path, index=False)
    print(f"Wrote metrics: {metrics_path}")

    pred_dir = save_dir / "predictions" / dataset
    pred_dir.mkdir(parents=True, exist_ok=True)
    ecol, tcol, ycol = tables.entity_col, tables.time_col, tables.target_col
    entities = set(predictions[ecol].astype(str))

    history_path = pred_dir / "history.parquet"
    if not history_path.exists():
        context = tables.context if _has_forecast_context(tables) else None
        frames = [tables.train, tables.test]
        splits = ["train", "test"]
        if context is not None:
            frames.insert(1, context)
            splits.insert(1, "context")
        parts = []
        for frame, split in zip(frames, splits):
            part = frame[frame[ecol].astype(str).isin(entities)]
            if not part.empty:
                parts.append(
                    pd.DataFrame({
                        "entity": part[ecol].astype(str).values,
                        "time": part[tcol].values,
                        "y_true": part[ycol].astype(float).values,
                        "split": split,
                    }))
        pd.concat(parts, ignore_index=True).to_parquet(history_path,
                                                       index=False)
        print(f"Wrote history: {history_path}")

    pred_path = pred_dir / f"{model}.parquet"
    quantile_cols = [c for c in QUANTILE_COLUMNS if c in predictions.columns]
    out = predictions[[ecol, tcol, "prediction", *quantile_cols]].copy()
    out = out.rename(columns={ecol: "entity", tcol: "time"})
    out["dataset"] = dataset
    out["model"] = model
    out.to_parquet(pred_path, index=False)
    print(f"Wrote predictions: {pred_path}")


def _predict_context_forecast_frame(
    tables: ForecastingTables,
    max_train_rows: int | None,
    *,
    add_temporal_features: bool = False,
) -> pd.DataFrame:
    """Model-ready context: fit history + optional val-year targets."""
    source = _limit_per_item(
        _predict_context_frame(tables),
        tables.entity_col,
        max_train_rows,
        "tail",
    )
    return _forecast_frame(
        source,
        tables,
        include_target=True,
        max_rows=None,
        keep="tail",
        add_temporal_features=add_temporal_features,
    )


def _same_prediction_length(
    df: pd.DataFrame,
    entity_col: str,
    model_name: str,
) -> int:
    horizon_counts = df.groupby(entity_col).size()
    if horizon_counts.nunique() != 1:
        raise ValueError(
            f"{model_name} requires the same forecast horizon for every "
            f"entity; found lengths {horizon_counts.unique().tolist()}.")
    return int(horizon_counts.iloc[0])


def _quantile_mean(quantiles: np.ndarray, axis: int = 0) -> np.ndarray:
    """Approximate the predictive mean by averaging the evenly spaced deciles
    (0.1..0.9) for models with no dedicated mean head.
    """
    return quantiles.mean(axis=axis)


def _forecast_frame(
    df: pd.DataFrame,
    tables: ForecastingTables,
    include_target: bool,
    max_rows: int | None = None,
    keep: str = "head",
    add_temporal_features: bool = False,
) -> pd.DataFrame:
    data = {
        "timestamp": df[tables.time_col],
        "item_id": df[tables.entity_col],
    }
    if include_target:
        data["target"] = df[tables.target_col].astype(float)
    out = pd.DataFrame(data)
    if add_temporal_features:
        features = extract_temporal_features(
            pd.DataFrame({"timestamp": out["timestamp"]}))
        out = pd.concat([out, features], axis=1)
    out = out.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    return _limit_per_item(out, "item_id", max_rows, keep)


def _regularize_to_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Fill time gaps per item so pd.infer_freq succeeds on gapped series."""
    chunks = []
    for item_id, grp in df.groupby("item_id", sort=False):
        extra_cols = [
            c for c in grp.columns if c not in ("item_id", "timestamp")
        ]
        grp = grp.set_index(pd.DatetimeIndex(grp["timestamp"]))[extra_cols]
        grp = grp.resample(freq).asfreq()
        grp.insert(0, "timestamp", grp.index)
        grp.insert(0, "item_id", item_id)
        chunks.append(grp.reset_index(drop=True))
    return pd.concat(chunks, ignore_index=True)


def run_arima(
    tables: ForecastingTables,
    order: tuple[int, int, int] | str = "auto",
    use_exog: bool = True,
) -> pd.DataFrame:
    train_df = tables.train.sort_values(tables.time_col)
    context_df = (tables.context.sort_values(tables.time_col)
                  if _has_forecast_context(tables) else train_df.iloc[:0])
    test_df = tables.test.sort_values(tables.time_col)
    if use_exog:
        exog_frames = [
            extract_temporal_features(df[[tables.time_col]])
            for df in (train_df, context_df, test_df)
        ]
        keep_cols = [
            col for col in exog_frames[0].columns
            if exog_frames[0][col].nunique(dropna=False) > 1
        ]
        if keep_cols:
            exog_train, exog_context, exog_test = (frame[keep_cols].to_numpy(
                dtype=float) for frame in exog_frames)
        else:
            exog_train = exog_context = exog_test = None
    else:
        exog_train = exog_context = exog_test = None

    y_train = train_df[tables.target_col].to_numpy(dtype=float)

    if order == "auto":
        order = _select_arima_order(y_train, exog_train,
                                    _ARIMA_CANDIDATE_ORDERS)

    result = SARIMAX(
        y_train,
        order=order,
        exog=exog_train,
        trend="c",
    ).fit(disp=False)
    if len(context_df) > 0:
        y_context = context_df[tables.target_col].to_numpy(dtype=float)
        result = result.append(
            y_context,
            exog=exog_context,
            refit=False,
        )
    test_pred = result.forecast(steps=len(test_df), exog=exog_test)
    return pd.DataFrame({
        tables.entity_col: test_df[tables.entity_col].values,
        tables.time_col: test_df[tables.time_col].values,
        "prediction": np.asarray(test_pred),
    })


def run_arima_per_entity(
    tables: ForecastingTables,
    order: tuple[int, int, int] | str = "auto",
    use_exog: bool = True,
    n_jobs: int = 1,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
) -> pd.DataFrame:
    train = _limit_per_item(tables.train, tables.entity_col, max_train_rows,
                            "tail")
    context = None
    if _has_forecast_context(tables):
        context = _limit_per_item(tables.context, tables.entity_col,
                                  max_train_rows, "tail")
    test = _limit_per_item(tables.test, tables.entity_col, max_test_rows,
                           "head")
    entities = test[tables.entity_col].unique()
    entity_col = tables.entity_col

    def _run_one(entity):
        train_e = train[train[entity_col] == entity].reset_index(drop=True)
        test_e = test[test[entity_col] == entity].reset_index(drop=True)
        if len(train_e) == 0 or len(test_e) == 0:
            return None
        context_e = None
        if context is not None and len(context) > 0:
            context_e = context[context[entity_col] == entity].reset_index(
                drop=True)
            if len(context_e) == 0:
                context_e = None
        return run_arima(
            ForecastingTables(
                train=train_e,
                test=test_e,
                entity_col=entity_col,
                time_col=tables.time_col,
                target_col=tables.target_col,
                freq=tables.freq,
                seasonal_period=tables.seasonal_period,
                feature_preset=tables.feature_preset,
                context=context_e,
            ),
            order=order,
            use_exog=use_exog,
        )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(entity)
        for entity in tqdm(entities, desc="Entities"))
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No entities with both train and test rows.")
    return pd.concat(results, ignore_index=True)


def run_tabpfn_ts(
    tables: ForecastingTables,
    max_train_rows: int | None,
    max_test_rows: int | None,
    tabpfn_mode: str = "local",
    max_context_length: int = 4096,
    feature_preset: str = "default",
    output_selection: str = "median",
    n_estimators: int = 8,
    model_path: str | None = None,
) -> pd.DataFrame:
    context_df = _predict_context_forecast_frame(tables, max_train_rows)
    future_df = _forecast_frame(tables.test, tables, False, max_test_rows)

    temporal_features: list[Any] | None = None
    if feature_preset in ("calendar", "calendar_running"):
        calendar_feature = CalendarFeature(
            components=[],
            seasonal_features={
                "minute_of_hour": [60],
                "hour_of_day": [24],
                "day_of_week": [7],
            },
        )
        temporal_features = [calendar_feature]
        if feature_preset == "calendar_running":
            temporal_features = [RunningIndexFeature(), calendar_feature]
    elif feature_preset != "default":
        raise ValueError(f"Unknown TabPFN-TS feature preset: {feature_preset}")

    mode = TabPFNMode(tabpfn_mode)
    model_config: dict[str, Any] = {"n_estimators": n_estimators}
    if model_path is not None:
        model_config["model_path"] = model_path
    pipeline_kwargs: dict[str, Any] = {
        "max_context_length": max_context_length,
        "tabpfn_mode": mode,
        "tabpfn_output_selection": output_selection,
        "tabpfn_model_config": model_config,
    }
    # If unset, TabPFN-TS uses its built-in temporal feature generators.
    if temporal_features is not None:
        pipeline_kwargs["temporal_features"] = temporal_features

    print("Running TabPFN-TS with:\n"
          f"\tcontext_df shape {context_df.shape}, \n"
          f"\tfuture_df shape {future_df.shape}, \n"
          f"\tmode={tabpfn_mode}, max_context_length={max_context_length}, "
          f"features={feature_preset}, output={output_selection}, "
          f"n_estimators={n_estimators}, model_path={model_path}")
    pipeline = TabPFNTSPipeline(**pipeline_kwargs)
    # ``target`` is TabPFN-TS's native point (per output_selection); the
    # deciles are kept alongside it for CRPS.
    pred = pipeline.predict_df(context_df, future_df=future_df,
                               quantiles=list(CRPS_QUANTILE_LEVELS))
    pred = pred.reset_index()
    deciles = pred[list(CRPS_QUANTILE_LEVELS)].to_numpy()
    pred_df = pred[["item_id", "timestamp",
                    "target"]].rename(columns={
                        "target": "prediction"
                    }).assign(**_quantile_frame(deciles))

    test_df = _forecast_frame(tables.test, tables, True, max_test_rows)
    if pred_df.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("TabPFN-TS returned duplicate predictions.")
    eval_df = test_df.merge(pred_df, on=["item_id", "timestamp"], how="left")
    if len(eval_df) != len(test_df):
        raise ValueError("TabPFN-TS prediction merge changed row count.")
    if eval_df["prediction"].isna().any():
        missing = int(eval_df["prediction"].isna().sum())
        raise ValueError(f"TabPFN-TS returned {missing} missing predictions.")

    eval_df = eval_df.rename(columns={
        "item_id": tables.entity_col,
        "timestamp": tables.time_col,
    })
    return eval_df[[
        tables.entity_col, tables.time_col, "prediction", *QUANTILE_COLUMNS
    ]]


def run_chronos2(
    tables: ForecastingTables,
    max_train_rows: int | None,
    max_test_rows: int | None,
    model_name_or_path: str = "amazon/chronos-2",
    batch_size: int = 100,
    context_length: int | None = None,
    feature_preset: str = "default",
    output_selection: str = "median",
    cross_learning: bool = False,
    progress_every: int = 1,
    max_output_patches: int | None = None,
) -> pd.DataFrame:
    if feature_preset not in ("default", "calendar"):
        raise ValueError(f"Unknown Chronos-2 feature preset: {feature_preset}")
    if max_output_patches is not None and max_output_patches <= 0:
        raise ValueError("--chronos2-max-output-patches must be positive.")
    add_features = feature_preset == "calendar"
    future_df = _forecast_frame(tables.test, tables, False, max_test_rows,
                                "head", add_features)
    context_df = _predict_context_forecast_frame(tables, max_train_rows)
    context_df = _regularize_to_freq(context_df, tables.freq)
    if add_features:
        feats = extract_temporal_features(
            pd.DataFrame({"timestamp": context_df["timestamp"]}))
        context_df = pd.concat([context_df, feats], axis=1)
    test_df = _forecast_frame(tables.test, tables, True, max_test_rows, "head",
                              add_features)

    prediction_length = _same_prediction_length(future_df, "item_id",
                                                "Chronos-2")

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running Chronos-2 with:\n"
          f"\tcontext_df shape {context_df.shape}, \n"
          f"\tfuture_df shape {future_df.shape}, \n"
          f"\tmodel={model_name_or_path}, device_map={device_map}, "
          f"batch_size={batch_size}, context_length={context_length}, "
          f"features={feature_preset}, output={output_selection}, "
          f"cross_learning={cross_learning}, "
          f"prediction_length={prediction_length}, "
          f"max_output_patches={max_output_patches}")
    pipeline = Chronos2Pipeline.from_pretrained(
        model_name_or_path,
        device_map=device_map,
    )
    batch_progress = {"completed": 0}

    def _after_batch() -> None:
        batch_progress["completed"] += 1
        if progress_every > 0 and batch_progress[
                "completed"] % progress_every == 0:
            print(
                "Chronos-2 batches completed: "
                f"{batch_progress['completed']}", flush=True)

    predict_kwargs = {}
    if max_output_patches is not None:
        predict_kwargs["max_output_patches"] = max_output_patches
    quantile_levels = list(CRPS_QUANTILE_LEVELS)
    pred = pipeline.predict_df(
        context_df,
        future_df=future_df,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        batch_size=batch_size,
        context_length=context_length,
        cross_learning=cross_learning,
        after_batch=_after_batch,
        **predict_kwargs,
    )
    rows = pred[pred["target_name"] == "target"]
    deciles = rows[[str(q) for q in quantile_levels]].to_numpy()
    if output_selection == "median":
        point = rows["0.5"].to_numpy()
    elif output_selection == "mean":
        point = _quantile_mean(deciles, axis=1)
    else:
        raise ValueError(
            f"Unknown Chronos-2 output selection: {output_selection}")
    pred_df = rows[["item_id", "timestamp"]].assign(prediction=point,
                                                    **_quantile_frame(deciles))

    if pred_df.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("Chronos-2 returned duplicate predictions.")
    eval_df = test_df.merge(pred_df, on=["item_id", "timestamp"], how="left")
    if len(eval_df) != len(test_df):
        raise ValueError("Chronos-2 prediction merge changed row count.")
    if eval_df["prediction"].isna().any():
        missing = int(eval_df["prediction"].isna().sum())
        raise ValueError(f"Chronos-2 returned {missing} missing predictions.")

    eval_df = eval_df.rename(columns={
        "item_id": tables.entity_col,
        "timestamp": tables.time_col,
    })
    return eval_df[[
        tables.entity_col, tables.time_col, "prediction", *QUANTILE_COLUMNS
    ]]


def _load_timesfm2_5(model_name_or_path: str):
    try:
        return TimesFM_2p5_200M_torch.from_pretrained(
            model_name_or_path,
            torch_compile=False,
        )
    except TypeError as exc:
        if "proxies" not in str(exc):
            raise
        # Compatibility workaround for TimesFM with newer huggingface_hub:
        # `proxies` is forwarded into the TimesFM constructor.
        return TimesFM_2p5_200M_torch._from_pretrained(
            model_id=model_name_or_path,
            revision=None,
            cache_dir=None,
            local_files_only=False,
            token=None,
            torch_compile=False,
        )


def run_timesfm2_5(
    tables: ForecastingTables,
    max_train_rows: int | None,
    max_test_rows: int | None,
    model_name_or_path: str = "google/timesfm-2.5-200m-pytorch",
    batch_size: int = 32,
    context_length: int = 8064,
    max_horizon: int = 1024,
    feature_preset: str = "default",
    output_selection: str = "mean",
    xreg_mode: str = "xreg + timesfm",
    direct_full_horizon: bool = False,
) -> pd.DataFrame:
    train = _limit_per_item(
        _predict_context_frame(tables),
        tables.entity_col,
        max_train_rows,
        "tail",
    )
    test = tables.test.sort_values([tables.entity_col, tables.time_col])
    test = _limit_per_item(test, tables.entity_col, max_test_rows, "head")
    entities = list(test[tables.entity_col].drop_duplicates())
    test_groups = {
        entity: df
        for entity, df in test.groupby(tables.entity_col)
    }
    train_groups = {
        entity: df
        for entity, df in train.groupby(tables.entity_col)
    }

    prediction_length = _same_prediction_length(test, tables.entity_col,
                                                "TimesFM 2.5")
    if max_horizon <= 0:
        raise ValueError("--timesfm-max-horizon must be positive.")
    if context_length <= 0:
        raise ValueError("--timesfm-context-length must be positive.")
    if output_selection not in ("mean", "median"):
        raise ValueError(f"Unknown TimesFM 2.5 output selection: "
                         f"{output_selection}")
    if direct_full_horizon:
        max_horizon = max(max_horizon, prediction_length)
    if feature_preset not in ("default", "calendar"):
        raise ValueError(f"Unknown TimesFM 2.5 feature preset: "
                         f"{feature_preset}")

    histories: list[np.ndarray] = []
    context_feature_lengths: list[int] = []
    combined_features: list[pd.DataFrame] = []
    for entity in entities:
        train_e = train_groups.get(entity)
        if train_e is None or train_e.empty:
            raise ValueError(f"No training rows for entity {entity!r}.")
        test_e = test_groups[entity]
        history = train_e[tables.target_col].to_numpy(dtype=float)
        histories.append(history[-context_length:])
        if feature_preset == "calendar":
            context_timestamps = train_e[tables.time_col].tail(context_length)
            context_feature_lengths.append(len(context_timestamps))
            timestamps = pd.concat(
                [context_timestamps, test_e[tables.time_col]],
                ignore_index=True)
            combined_features.append(
                extract_temporal_features(
                    pd.DataFrame({tables.time_col: timestamps})))

    torch.set_float32_matmul_precision("high")
    model = _load_timesfm2_5(model_name_or_path)
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_length,
            max_horizon=max_horizon,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=feature_preset == "calendar",
        ))

    print(
        "Running TimesFM 2.5 with:\n"
        f"\tentities={len(entities)}, prediction_length={prediction_length}, "
        f"model={model_name_or_path}, batch_size={batch_size}, "
        f"context_length={context_length}, max_horizon={max_horizon}, "
        f"features={feature_preset}, output={output_selection}, "
        f"xreg_mode={xreg_mode}, direct_full_horizon={direct_full_horizon}")

    predictions: list[np.ndarray] = [
        np.empty(0, dtype=float) for _ in entities
    ]
    decile_parts: list[np.ndarray] = [
        np.empty((0, len(CRPS_QUANTILE_LEVELS)), dtype=float) for _ in entities
    ]
    chunk_start = 0
    while chunk_start < prediction_length:
        chunk_len = prediction_length if direct_full_horizon else min(
            max_horizon, prediction_length - chunk_start)
        inputs = [history[-context_length:] for history in histories]
        if feature_preset == "calendar":
            dynamic_covariates: dict[str, list[list[float]]] = {}
            for name in combined_features[0].columns:
                values = []
                # Histories may include prior predicted chunks. Slice calendar
                # features over the same timeline so they stay aligned.
                for features, context_len, history in zip(
                        combined_features, context_feature_lengths, inputs):
                    context_end = context_len + chunk_start
                    context_start = context_end - len(history)
                    future_start = context_len + chunk_start
                    future_end = future_start + chunk_len
                    values.append(
                        features[name].iloc[context_start:context_end].tolist(
                        ) +
                        features[name].iloc[future_start:future_end].tolist())
                dynamic_covariates[name] = values
            chunk_point, quantile_forecast = model.forecast_with_covariates(
                inputs=[history.tolist() for history in inputs],
                dynamic_numerical_covariates=dynamic_covariates,
                dynamic_categorical_covariates={},
                static_numerical_covariates={},
                static_categorical_covariates={},
                xreg_mode=xreg_mode,
                force_on_cpu=False,
            )
            if output_selection == "mean":
                chunk_pred = np.asarray(chunk_point, dtype=float)
            else:
                quantile_forecast = np.asarray(quantile_forecast, dtype=float)
                chunk_pred = quantile_forecast[:, :chunk_len, 5]
            quantile_forecast = np.asarray(quantile_forecast, dtype=float)
            chunk_deciles = quantile_forecast[:, :chunk_len, 1:10]
        else:
            _, quantile_forecast = model.forecast(horizon=chunk_len,
                                                  inputs=inputs)
            # Channels: 0 = mean, 1..9 = deciles 0.1..0.9 (5 = median).
            channel = 0 if output_selection == "mean" else 5
            chunk_pred = quantile_forecast[:, :chunk_len, channel]
            chunk_deciles = np.asarray(quantile_forecast[:, :chunk_len, 1:10],
                                       dtype=float)

        for i in range(len(entities)):
            pred_i = np.asarray(chunk_pred[i, :chunk_len], dtype=float)
            predictions[i] = np.concatenate([predictions[i], pred_i])
            histories[i] = np.concatenate([histories[i], pred_i])
            decile_parts[i] = np.concatenate(
                [decile_parts[i], chunk_deciles[i]], axis=0)
        chunk_start += chunk_len
        print(
            f"TimesFM 2.5 forecasted {chunk_start}/"
            f"{prediction_length} steps", flush=True)

    parts = []
    for i, entity in enumerate(entities):
        frame = pd.DataFrame({
            tables.entity_col:
            entity,
            tables.time_col:
            test_groups[entity][tables.time_col].values,
            "prediction":
            predictions[i],
        })
        frame = frame.assign(**_quantile_frame(decile_parts[i]))
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def run_toto2(
    tables: ForecastingTables,
    max_train_rows: int | None,
    max_test_rows: int | None,
    model_name_or_path: str = "Datadog/Toto-2.0-313m",
    batch_size: int = 64,
    context_length: int = 4096,
    output_selection: str = "mean",
) -> pd.DataFrame:
    train = _limit_per_item(
        _predict_context_frame(tables),
        tables.entity_col,
        max_train_rows,
        "tail",
    )
    test = tables.test.sort_values([tables.entity_col, tables.time_col])
    test = _limit_per_item(test, tables.entity_col, max_test_rows, "head")
    test_groups = {
        entity: df
        for entity, df in test.groupby(tables.entity_col, sort=False)
    }
    entities = list(test_groups)
    train_groups = {
        entity: df
        for entity, df in train.groupby(tables.entity_col)
    }
    prediction_length = _same_prediction_length(test, tables.entity_col,
                                                "Toto 2.0")

    histories: list[np.ndarray] = []
    for entity in entities:
        train_e = train_groups.get(entity)
        if train_e is None or train_e.empty:
            raise ValueError(f"No training rows for entity {entity!r}.")
        history = train_e[tables.target_col].to_numpy(dtype=float)
        histories.append(history[-context_length:])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "Running Toto 2.0 with:\n"
        f"\tentities={len(entities)}, prediction_length={prediction_length}, "
        f"model={model_name_or_path}, batch_size={batch_size}, "
        f"context_length={context_length}, output={output_selection}, "
        f"device={device}")
    model = Toto2Model.from_pretrained(model_name_or_path).to(device).eval()

    # Toto 2.0 has no exogenous support (calendar preset unused). The patched
    # transformer needs the context length to be a multiple of patch_size, so
    # histories are left-padded and masked.
    patch_size = int(model.config.patch_size)
    predictions: list[np.ndarray] = []
    deciles_list: list[np.ndarray] = []
    for start in range(0, len(entities), batch_size):
        chunk = histories[start:start + batch_size]
        max_len = max(len(h) for h in chunk)
        padded_len = -(-max_len // patch_size) * patch_size
        target = np.zeros((len(chunk), 1, padded_len), dtype=np.float32)
        mask = np.zeros((len(chunk), 1, padded_len), dtype=bool)
        for i, history in enumerate(chunk):
            target[i, 0, padded_len - len(history):] = history
            mask[i, 0, padded_len - len(history):] = True
        has_missing = not bool(mask.all())
        with torch.no_grad():
            quantiles = model.forecast(
                {
                    "target":
                    torch.from_numpy(target).to(device),
                    "target_mask":
                    torch.from_numpy(mask).to(device),
                    "series_ids":
                    torch.zeros(len(chunk), 1, dtype=torch.long,
                                device=device),
                },
                horizon=prediction_length,
                has_missing_values=has_missing,
            )
        # q: (9, batch, horizon) at knots 0.1..0.9, so index 4 is the median.
        q = quantiles[:, :, 0, :].float().cpu().numpy()
        if output_selection == "median":
            point = q[4]
        elif output_selection == "mean":
            point = _quantile_mean(q, axis=0)
        else:
            raise ValueError(
                f"Unknown Toto 2.0 output selection: {output_selection}")
        predictions.extend(point[i] for i in range(len(chunk)))
        # Per entity: (9, horizon) knots -> (horizon, 9) decile columns.
        deciles_list.extend(q[:, i, :].T for i in range(len(chunk)))
        print(
            f"Toto 2.0 forecasted {min(start + batch_size, len(entities))}/"
            f"{len(entities)} entities", flush=True)

    parts = [
        pd.DataFrame({
            tables.entity_col: entity,
            tables.time_col: test_groups[entity][tables.time_col].values,
            "prediction": np.asarray(predictions[i], dtype=float),
        }).assign(**_quantile_frame(deciles_list[i]))
        for i, entity in enumerate(entities)
    ]
    return pd.concat(parts, ignore_index=True)


def run_forecasting_model(
    model_name: str,
    tables: ForecastingTables,
    max_train_rows: int | None,
    max_test_rows: int | None,
    n_jobs: int = 1,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    kwargs = model_kwargs or {}
    if model_name == "arima":
        return run_arima_per_entity(
            tables,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
            n_jobs=n_jobs,
            **kwargs,
        )
    if model_name == "tabpfn_ts":
        return run_tabpfn_ts(
            tables,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
            **kwargs,
        )
    if model_name == "chronos2":
        return run_chronos2(
            tables,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
            **kwargs,
        )
    if model_name == "timesfm2_5":
        return run_timesfm2_5(
            tables,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
            **kwargs,
        )
    if model_name == "toto2":
        return run_toto2(
            tables,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
            **kwargs,
        )
    if model_name == "seasonal_naive":
        period = kwargs.pop("seasonal_period", None)
        if period is None:
            raise ValueError(
                "seasonal_naive requires model_kwargs['seasonal_period']")
        return run_seasonal_naive(tables, seasonal_period=int(period))
    raise ValueError(f"Unsupported forecasting model: {model_name}")
