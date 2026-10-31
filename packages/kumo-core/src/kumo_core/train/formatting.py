# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
from kumo_core.encoders.target.multicategorical import (
    MultiCategoricalTargetEncoder, )
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
)
from kumo_core.train.mapping import from_index_to_raw
from kumo_core.types import TaskType
from torch import Tensor


class NamedColumn(Protocol):
    name: str


class LinkPredictionTable(Protocol):
    fkeys: Sequence[NamedColumn]
    time_col: NamedColumn


_ANCHOR_TIME_COL = "ANCHOR_TIMESTAMP"
_CLASS_TASKS = frozenset({
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
})
_LONG_FORMAT_TASKS = _CLASS_TASKS | {
    TaskType.MULTILABEL_RANKING,
    TaskType.TEMPORAL_LINK_PREDICTION,
    TaskType.STATIC_LINK_PREDICTION,
}


def _get_col_names(
    mapping: pd.Index,
    task_type: TaskType,
    is_wide_format: bool,
) -> list[str]:
    r"""Returns column names to be used in the formatted prediction
    dataframe.

    Args:
        mapping (pd.Index): The mapping containing the categories/classes for
            which predictions are generated.
        task_type (TaskType): The type of task. E.g. Regression, Binary
            Classification etc.
        is_wide_format(bool): Whether the prediction dataframe is in wide or
            long format. If it's in wide format, the column names will contain
            a '_PROB' suffix.
    """
    if task_type == TaskType.BINARY_CLASSIFICATION:
        if is_wide_format:
            return ['False_PROB', 'True_PROB']
        return ['False', 'True']

    result = []
    for col_value in mapping.values:
        if isinstance(col_value, np.generic):
            col_value = col_value.item()
        if isinstance(col_value, bool):
            col_value = str(col_value)
        if is_wide_format:
            col_value = f"{col_value}_PROB"
        result.append(col_value)
    return result


def prediction_output_columns(task_type: TaskType) -> list[str]:
    r"""The prediction columns a scored row carries after its entity column,
    for a target named ``TARGET`` and a time column named ``TIMESTAMP``."""
    if task_type in _CLASS_TASKS:
        return ['PREDICTED', 'CLASS', 'SCORE']
    if task_type.is_ranking:
        return ['CLASS', 'SCORE']
    if task_type == TaskType.FORECASTING:
        return ['TIMESTAMP', 'TARGET_PRED']
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return [
            'TARGET_PRED',
            *_get_col_names(pd.Index([]), task_type, is_wide_format=True)
        ]
    return ['TARGET_PRED']


def _format_prediction_sanity_checks(
    target_encoder: TargetEncoder,
    task_type: TaskType,
    pred: Tensor,
    prob: Tensor | None,
):
    if task_type == TaskType.REGRESSION:
        assert isinstance(target_encoder, NumericalTargetEncoder)
    elif task_type == TaskType.FORECASTING:
        assert isinstance(target_encoder, NumericalTargetEncoder)
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # Binary classification has pred tensor(bool dtype), prob tensor
        # of shape N*2 and an Index target encoder with 2 categories
        assert isinstance(target_encoder, CategoricalTargetEncoder)
        assert prob is not None
        assert prob.shape == pred.shape + (2, )
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        assert isinstance(target_encoder, CategoricalTargetEncoder)
        N = pred.shape[0]
        assert prob is not None
        assert prob.shape == (N, target_encoder.num_categories)
    elif task_type.is_multilabel:
        assert isinstance(target_encoder, MultiCategoricalTargetEncoder)
        assert prob is not None
        if task_type == TaskType.MULTILABEL_RANKING:
            N = pred.shape[0]
            assert prob.shape == (N, target_encoder.num_categories)
        else:
            assert pred.shape == prob.shape
    else:
        raise ValueError("Unsupported task type in format_prediction: "
                         f"{task_type}")


def _add_pred_to_df(
    df: pd.DataFrame,
    pred: Tensor,
    target_col_name: str,
    target_encoder: TargetEncoder,
) -> pd.DataFrame:
    r"""Adds the prediction output to the dataframe by decoding them using
    the target column encoder."""
    # Skip decoding if the pred has already been boolean
    # such as in the binary classification case
    if pred.dtype == torch.bool:
        df[target_col_name] = pred.numpy()
        df[target_col_name] = df[target_col_name].astype(pd.BooleanDtype())
    else:
        df[target_col_name] = target_encoder.decode(pred).values
    return df


def _add_quantiles_to_df(
    df: pd.DataFrame,
    quantiles: Tensor,
    quantiles_names: list[str],
    target_encoder: TargetEncoder,
    task_type: TaskType,
) -> pd.DataFrame:
    r"""Adds the quantiles of predicted distributions to the dataframe."""

    decoded_quantiles = target_encoder.decode_tensor(quantiles)
    assert isinstance(decoded_quantiles, Tensor)
    decoded_quantiles = decoded_quantiles.sort(dim=-1).values
    for i, quantile_name in enumerate(quantiles_names):
        quantile = decoded_quantiles[:, :, i]
        is_multi_step_forecast = (task_type == TaskType.FORECASTING
                                  and quantile.ndim == 2
                                  and quantile.size(1) > 1)
        if is_multi_step_forecast:
            df['q_' + quantile_name] = quantile.cpu().numpy().tolist()
        else:
            df['q_' + quantile_name] = quantile.reshape(-1).cpu().numpy()

    return df


def _add_prob_to_df(
    df: pd.DataFrame,
    prob: Tensor,
    mapping: pd.Index,
    format: str,
    task_type: TaskType,
) -> pd.DataFrame:
    r"""Add the class probabilities to the dataframe.

    Args:
        df (pd.DataFrame) : Dataframe of the prediction table.
        prob (Tensor) : The output probabilities of the model.
        mapping (pd.Index) : Contains the list of class names.
        format (str): Formatting option for predictions - 'long' or 'wide'.
    """
    col_names = _get_col_names(
        mapping,
        task_type,
        is_wide_format=format == 'wide',
    )
    prob_df = pd.DataFrame(prob.numpy(), columns=col_names).set_index(df.index)
    df = pd.concat([df, prob_df], axis=1)
    return df


def _convert_to_list(x: Any) -> Any:
    if isinstance(x, float) and pd.isna(x):
        return []
    if not pd.api.types.is_list_like(x):
        return [x]
    return x


def _wide_to_long_format(
    df: pd.DataFrame,
    target_col_name: str,
    target_encoder: TargetEncoder,
    id_vars: pd.Index,
    task_type: TaskType,
    top_k: int | None = None,
    original_target_col_name: str | None = None,
    array_targets: bool = False,
) -> pd.DataFrame:
    r"""Converts the prediction dataframe from 'wide' format to 'long'
    format.
    """
    df = pd.melt(df, id_vars=id_vars.tolist(), var_name='CLASS',
                 value_name='SCORE', ignore_index=False).sort_index()

    # Handle the data type of the 'var_name' column, since pandas
    # melt does not handle this properly normally:
    if pd.api.types.is_bool_dtype(df[target_col_name].dtype):
        # Boolean round-trip conversion fails, so we need this check:
        df['CLASS'] = (df['CLASS'] == "True")
    else:
        df['CLASS'] = df['CLASS'].astype(
            df[target_col_name].dtype)  # type: ignore

    entity_cols = id_vars.drop(target_col_name).tolist()
    # Filter top k classes
    if top_k is not None:
        df = df.sort_values(["SCORE"], ascending=False)

        # groupby op is hash-based and hence requires column to be hashable
        if original_target_col_name in df:
            df[original_target_col_name] = df[original_target_col_name].apply(
                lambda x: tuple(_convert_to_list(x)))

        df = df.groupby(entity_cols).head(top_k).sort_index()

    # Convert the `target_col_name` column to a one-hot representation:
    if task_type.is_multilabel:
        assert isinstance(target_encoder.tensor_mapper,
                          MultiCategoricalTensorMapper)

        if array_targets:
            df[target_col_name] = df[target_col_name].apply(_convert_to_list)
        else:
            df[target_col_name] = df[target_col_name].fillna('').str.split(
                target_encoder.tensor_mapper.sep)

        df[target_col_name] = df.apply(
            lambda x: 1 if x["CLASS"] in x[target_col_name] else 0, axis=1)
    else:
        df[target_col_name] = df.apply(
            lambda x: 1 if x["CLASS"] == x[target_col_name] else 0, axis=1)

    df.rename(columns={target_col_name: "PREDICTED"}, inplace=True)
    return df


def format_prediction(
    df: pd.DataFrame,
    target_col_name: str,
    target_encoder: TargetEncoder,
    task_type: TaskType,
    pred: Tensor | None = None,
    prob: Tensor | None = None,
    quantiles: Tensor | None = None,
    quantile_names: list[str] | None = None,
    top_k: int | None = None,
    time_col_name: str | None = None,
    out_timestamps: pd.Series | None = None,
    array_targets: bool = False,
) -> pd.DataFrame:
    r"""Formats the trainer's predictions output and returns a
    :class:`pandas.DataFrame`.

    Args:
        df (pd.DataFrame) : dataframe of the training table.
        target_col_name (str) : name of the target column
        target_encoder (TargetEncoder) : Encoder used for the target column.
        task_type (TaskType) : The type of the task.
        pred (Tensor, optional): The output predictions of the model.
            (default: :obj:`None`)
        prob (Tensor, optional): The output probabilities of the model.
            (default: :obj:`None`)
        quantiles (Tensor, optional): Quantiles of predicted distributions from
            distribution heads.
            (default: :obj:`None`)
        quantile_names (list, optional): List of quantiles names.
            (default: :obj:`None`)
        top_k (int, optional): The number of :obj:`top_k` classes to return
            per entity, if not :obj:`None`. (default: :obj:`None`)
        time_col_name (str, optional): Specifies the time column name.
            (default: :obj:`None`)
        out_timestamps (pd.Series, optional): If multiple predictions per
            entity exist (e.g. in forecasting), provides the series of
            timestamps of predictions to format each prediction in its own row.
            (default: :obj:`None`)
    """
    if pred is None:
        return df
    if (task_type == TaskType.FORECASTING
            and time_col_name == _ANCHOR_TIME_COL):
        raise ValueError(
            'Forecast time column ANCHOR_TIMESTAMP conflicts with anchor '
            'metadata. Rename the time column before forecasting.')

    format: Literal['wide', 'long']
    if task_type in _LONG_FORMAT_TASKS:
        format = 'long'
    else:
        assert top_k is None, f"`top_k` is not supported for {task_type}."
        format = 'wide'

    # TODO Investigate why this option is unspecified.
    if (task_type == TaskType.MULTILABEL_CLASSIFICATION and array_targets):
        assert isinstance(target_encoder.tensor_mapper,
                          MultiCategoricalTensorMapper)
        target_encoder.tensor_mapper._is_string_dtype = False

    original_target_col_name = target_col_name
    target_col_name = f'{target_col_name}_PRED'
    assert target_encoder is not None
    _format_prediction_sanity_checks(target_encoder, task_type, pred, prob)

    # Add the predictions to the dataframe:
    df = _add_pred_to_df(df, pred, target_col_name, target_encoder)

    if quantile_names is not None and quantiles is not None:
        df = _add_quantiles_to_df(
            df,
            quantiles,  # type: ignore
            quantile_names,
            target_encoder,
            task_type,
        )
    if task_type == TaskType.FORECASTING:
        assert time_col_name is not None
        assert time_col_name in df.columns
        # The current time_col before explosion are all anchor times
        df[_ANCHOR_TIME_COL] = df[time_col_name].copy()
        explode_cols = [target_col_name]
        if original_target_col_name in df:
            explode_cols += [original_target_col_name]
        if quantile_names is not None and quantiles is not None:
            explode_cols += ['q_' + name for name in quantile_names]
        df = df.explode(explode_cols)
        out_timestamps = out_timestamps.detach().cpu().numpy()  # type: ignore
        df[time_col_name] = out_timestamps.\
            astype('datetime64[ns]')  # type: ignore

    # Regression task type has no implications with class probabilities
    # TODO: Forecasting has none either until we support conf intervals
    if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        return df

    # Prob tensor shouldn't be None for other task types.
    assert prob is not None

    # These are required for long format later - collect before adding
    # class probabilities
    id_vars = df.columns

    assert isinstance(target_encoder.tensor_mapper,
                      (CategoricalTensorMapper, MultiCategoricalTensorMapper))
    mapping = target_encoder.tensor_mapper.mapping

    # Construct the wide format dataframe by adding the class probabilities:
    df = _add_prob_to_df(df, prob, mapping, format, task_type)

    # Convert the wide format to long if required:
    if format == 'long':
        df = _wide_to_long_format(
            df,
            target_col_name,
            target_encoder,
            id_vars,
            task_type,
            top_k,
            original_target_col_name,
            array_targets,
        )
    return df


# TODO(kiriti): Add a TaskType dependency here as well similar to
# format_predictions
def format_embeddings(
    embeddings: np.array,  # type: ignore
    meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    r"""Formats the embeddings output and returns a :class:`pandas.DataFrame`.
    The output format contains a single column for the embedding, as well
    as optionally columns for entity information that is transferred from
    the meta dataframe, such as `entity_id` or `timestamp`.

    Args:
        embeddings (np.array): embeddings to be formatted.
        meta_df (pd.DataFrame, optional): dataframe containing the entity
            information to be transferred to the output dataframe. Including
            columns such as entity_id and timestamp.
    """
    emb_df = pd.DataFrame()
    emb_df['EMBEDDING'] = list(embeddings)
    if meta_df is not None:
        meta_df.index = emb_df.index
        emb_df = pd.concat([meta_df, emb_df], axis=1)
    return emb_df


def create_lp_prediction_output(
    outs: dict[str, Tensor],
    pred_table: LinkPredictionTable,
    top_k: int | None,
    is_temporal: bool,
    lhs_index_map: pd.DataFrame,
    rhs_index_map: pd.DataFrame,
    time_col_data: pd.Series | None = None,
    custom_time_col_name: str | None = None,
) -> pd.DataFrame:
    r"""Creates a standardized, formatted dataframe of predictions associated
    with the links in a link prediction task.
    Args:
        custom_time_col_name (str, optional): The name of the time column to
            use. If :obj:`None`, the time column from the prediction table is
            used. (default :obj:`None`)
    """
    # Convert LHS ids to original ids:
    ids = outs['id']
    ids = np.array(from_index_to_raw(ids, lhs_index_map))

    if top_k is None:
        # Infer top k from outs.
        top_k = outs['target'].shape[-1]
    ids = np.repeat(ids, top_k)
    ids_flat = ids.flatten()

    # Convert RHS ids to original ids:
    rhs_ids = torch.flatten(outs['target'])
    rhs_ids = np.array(from_index_to_raw(rhs_ids, rhs_index_map))
    rhs_ids_flat = rhs_ids.flatten()

    outs_flat = torch.flatten(outs['score']).numpy()

    # Pass flattened numpy arrays to dataframe:
    df = pd.DataFrame({
        pred_table.fkeys[0].name: ids_flat,
        'CLASS': rhs_ids_flat,
        'SCORE': outs_flat,
    })
    if 'label' in outs:
        labels_flat = torch.flatten(outs['label']).numpy()
        df['LABEL'] = labels_flat

    if is_temporal:
        # TODO(manan): duplicate from embeddings:
        time_col_name = custom_time_col_name or pred_table.time_col.name
        if 'time' in outs:
            time = pd.to_datetime(outs['time'].numpy(), unit='ns')
            time = np.repeat(time, top_k)  # type: ignore
            df[time_col_name] = time
        else:
            if time_col_data is None:
                time_col = pred_table.time_col.pandas().values
            else:
                time_col = time_col_data.values
            timestamps = np.unique(time_col)  # type: ignore
            assert len(timestamps) == 1
            df[time_col_name] = timestamps[0]

    # We add an additional column to the prediction table to ensure the
    # same schema as in the prediction table for the multilabel-class case:
    df['PREDICTED'] = None

    return df
