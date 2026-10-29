# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import logging
import math
import os
import os.path as osp
import zipfile
from dataclasses import asdict, dataclass
from typing import Any, Final, Mapping, Protocol, cast

import kumo_core.io.filesystem as filesystem
import numpy as np
import pandas as pd
import pyarrow as pa
from kumo_core.train.artifact_stats import FrameWriterStats
from kumo_core.train.io import (
    StreamingFrameWriter,
    makedirs,
    read_binary,
    write_binary,
    write_frame,
    write_frame_chunks,
)
from kumo_core.train.stats import (
    TargetHistogram,
    TargetHistogramAndStats,
    TargetHistogramAndStatsPair,
    parse_histogram_and_stats,
)
from kumo_core.types import PredictionArtifactType, TaskType
from kumoapi.train import TrainingStage
from typing_extensions import Self


class StageTableStats(Protocol):
    target_distribution: Any


class SplitTableStatsReader(Protocol):
    def get_stage_table_stats(
        self,
        stage: TrainingStage,
    ) -> StageTableStats | None:
        ...


class TaskOutputContext(Protocol):
    task_type: TaskType

    def get_train_table_stats(
        self,
        *,
        after_split: bool,
    ) -> SplitTableStatsReader | None:
        ...


def _json_safe_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _save_json(value: Any, path: str) -> None:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    write_binary(path, json.dumps(payload).encode("utf-8"))


_ONE_GB: Final[int] = 1024 * 1024 * 1024
_HOLDOUT_MAX_BYTES: int = _ONE_GB
_STREAMING_BUFFER_SIZE: Final[int] = 128 * 1024 * 1024
_MAX_NUM_SAMPLES: Final[int] = 10_000
_HISTOGRAM_BIN_PRECISION: Final[int] = 8
_EVAL_OUTPUT_FORMAT_VERSION: Final[int] = 1
_EVAL_OUTPUT_METADATA_FILE: Final[str] = 'metadata.json'
_EVAL_OUTPUT_DF_FILE: Final[str] = 'df.parquet'
_EVAL_OUTPUT_EMBEDDING_FILE: Final[str] = 'embedding.parquet'

logger = logging.getLogger(__name__)


def _normalize_mixed_type_class_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'CLASS' not in df.columns or df['CLASS'].dtype != object:
        return df

    inferred_dtype = pd.api.types.infer_dtype(df['CLASS'], skipna=True)
    if not inferred_dtype.startswith('mixed'):
        return df

    df = df.copy(deep=False)
    df['CLASS'] = df['CLASS'].map(lambda value: None
                                  if pd.isna(value) else str(value))
    return df


def _normalize_prediction_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Use Parquet's interoperable microsecond timestamp representation."""

    datetime_columns = [
        name for name in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[name].dtype)
    ]
    if not datetime_columns:
        return df
    df = df.copy(deep=False)
    for name in datetime_columns:
        df[name] = pd.Series(
            df[name].array.to_pydatetime(),
            index=df.index,
            dtype=object,
        )
    return df


@dataclass(repr=False)
class TargetPredictionHeatMap:
    r"""A dataclass that represents the label value vs prediction value
    heatmap, where target label value resides in the x-axis while the
    corresponding prediction value resides in the y-axis. It is expected that
    most of the target value, prediction value pair appear roughly in the
    straight line represented by `f(x) = x`.

    Args:
        cell_width (int): The width of each cell in the heatmap.
        label_bin_start (float): the left most value the heatmap.
        label_bin_end (float): the right most value of the heatmap.
        prediction_bin_start (float): the bottom most value of the heatmap.
        prediction_bin_end (float): the top most value of the heatmap.
        cells (List[Tuple[float, float]]): A list of cells in the heatmap
            represented by the bottom left corner of the cell, where at
            least one label prediction value pair resides in it.
        counts (List[int]): The number of label prediction pair in each cell.
        home_dir (str): The s3 home directory that stores heatmap artifact.
    """
    cell_width: float
    label_bin_start: float
    label_bin_end: float
    prediction_bin_start: float
    prediction_bin_end: float
    cells: list[tuple[float, float]]
    counts: list[int]
    home_dir: str

    def heatmap_path(self) -> str:
        return osp.join(self.home_dir, 'heatmap.json')

    def sampled_df_path(self) -> str:
        return osp.join(self.home_dir, 'sampled_df.parquet')

    def save(self) -> None:
        _save_json(self, self.heatmap_path())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  cell_width={self.cell_width},\n'
                f'  label_bin_start={self.label_bin_start},\n'
                f'  label_bin_end={self.label_bin_end},\n'
                f'  prediction_bin_start={self.prediction_bin_start},\n'
                f'  prediction_bin_end={self.prediction_bin_end},\n'
                f'  cells={self.cells},\n'
                f'  counts={self.counts},\n'
                f'  home_dir={self.home_dir},\n'
                f')')


@dataclass
class EvalOutput:
    r"""A dataclass that contains information produced from performing
    validation or testing on a trained model."""

    # Dictionary mapping metric name to metric value:
    metrics: dict[str, float]

    # Prediction output dataframe, containing target column, time column (if
    # exists), prediction outputs, and prediction probabilities. See
    # formatting.py/format_prediction for more details:
    df: pd.DataFrame | None = None

    # Embedding dataframe containing embedding meta-information and embedding
    # outputs. See formatting.py/format_embeddings for more details:
    embedding: pd.DataFrame | None = None

    # Exact counts of the test split's scores (`kumo_core.evaluation_counts`):
    counts: Any = None

    def save_all(
        self,
        root_dir: str,
        stage: TrainingStage,
        task: TaskOutputContext,
    ) -> None:
        r"""Saves output files to a given directory.

        Args:
            root_dir: The root directory to save the output files.
            stage: The stage of the output files to save.
            task: TaskOutputContext object to fetch train table stats through.

        Example:
            >>> eval_output.save_all(
            ...    root_dir='./version_0',  # trainer.log_dir
            ...    stage=TrainingStage.TEST,
            ...    task=task,
            ... )
            >>> # Saves the following files:
            >>> # ./version_0/test_df.parquet
            >>> # ./version_0/test_target_prediction_heatmap/heatmap.json
            >>> # ./version_0/test_target_prediction_heatmap/sampled_df.parquet
            >>> # ./version_0/test_prediction_histogram_and_stats.json
            >>> # ./version_0/test_output.pt
        """
        self.save(root_dir=root_dir, stage=stage)
        self.save_prediction_heatmap(
            task=task,
            stage=stage,
            root_dir=root_dir,
        )
        self.save_prediction_histogram_and_stats(
            task=task,
            stage=stage,
            root_dir=root_dir,
        )
        if stage is TrainingStage.TEST:
            self.save_df(root_dir=root_dir, stage=stage)

    def save(self, root_dir: str, stage: TrainingStage) -> None:
        """Save metrics and frames without executable pickle content."""
        path = self.get_output_path(root_dir, stage)
        logger.info(f'Saving {stage.value} output to {path}')
        metadata = {
            'format_version': _EVAL_OUTPUT_FORMAT_VERSION,
            'metrics': self.metrics,
            'has_df': self.df is not None,
            'has_embedding': self.embedding is not None,
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                _EVAL_OUTPUT_METADATA_FILE,
                json.dumps(metadata),
            )
            self._write_frame(archive, _EVAL_OUTPUT_DF_FILE, self.df)
            self._write_frame(
                archive,
                _EVAL_OUTPUT_EMBEDDING_FILE,
                self.embedding,
            )
        write_binary(path, buffer.getvalue())

    @staticmethod
    def _write_frame(
        archive: zipfile.ZipFile,
        name: str,
        frame: pd.DataFrame | None,
    ) -> None:
        if frame is None:
            return
        buffer = io.BytesIO()
        _normalize_mixed_type_class_column(frame).to_parquet(buffer)
        archive.writestr(name, buffer.getvalue())

    def save_df(
        self,
        root_dir: str,
        stage: TrainingStage,
        max_size: int | None = None,
    ) -> None:
        if self.df is None:
            return

        max_size = _HOLDOUT_MAX_BYTES if max_size is None else max_size
        path = self.get_df_path(root_dir, stage)
        total_size = self.df.memory_usage(deep=True).sum()
        lines = len(self.df)
        if total_size > max_size:
            # Down sample the holdout output to fit in memory.
            ratio = float(max_size) / total_size
            lines = int(lines * ratio)
        df = _normalize_mixed_type_class_column(self.df.head(lines))
        write_frame(df, path)
        write_binary(
            self.get_df_meta_path(root_dir, stage),
            json.dumps({
                'total_rows': len(self.df),
                'saved_rows': len(df),
                'truncated': len(df) < len(self.df),
            }).encode(),
        )

    @staticmethod
    def _generate_prediction_histogram_and_stats(
        label_distribution: TargetHistogramAndStats,
        prediction_df: pd.DataFrame,
        prediction_col: str,
    ) -> TargetHistogramAndStatsPair:
        label_histogram = label_distribution.histogram
        prediction_min_value = cast(float, prediction_df[prediction_col].min())
        prediction_max_value = cast(float, prediction_df[prediction_col].max())

        bin_width = label_histogram.bin_width
        if bin_width is None or bin_width <= 0:
            # Issue handled silenlty.
            raise ValueError(
                f"label value histogram `bin_width` should always be a "
                f"positive number for label vs prediction histogram to "
                f"work. However the value of `bin_width` is {bin_width}.")

        # Make sure prediction score histogram bins and label histogram
        # bins are aligned by setting the bin width of the prediction
        # histogram to be the same as label histogram and setting the
        # minimum and maximum bin boundary for prediction histogram to be
        # in the format of `min_value` + k * `bin_width` and
        # `max_value` + l * `bin_width` where `min_value` is the minimum
        # bin boundary for label histogram and `max_value` is the maximum
        # bin boundary for label histogram and `k` and `l` are integer
        # values.
        min_value = label_histogram.bins[0]
        if min_value >= prediction_min_value:
            prediction_min_value_adjust = min_value - math.ceil(
                (min_value - prediction_min_value) / bin_width) * bin_width
        else:
            prediction_min_value_adjust = min_value + math.floor(
                (prediction_min_value - min_value) / bin_width) * bin_width

        max_value = label_histogram.bins[-1]
        if max_value >= prediction_max_value:
            prediction_max_value_adjust = max_value - math.floor(
                (max_value - prediction_max_value) / bin_width) * bin_width
        else:
            prediction_max_value_adjust = max_value + math.ceil(
                (prediction_max_value - max_value) / bin_width) * bin_width

        # Inlined from former ``PandasDataProcessingBackend.histogram``:
        # builds ``np.arange``-spaced bins, runs ``np.histogram`` over the
        # selected column, and rounds bin edges to ``HISTOGRAM_BIN_PRECISION``
        # to avoid float drift across many steps (matching the optimized
        # ``StatsManager.combine_numerical_distribution`` path).
        _hist_max = max(prediction_max_value_adjust,
                        prediction_min_value_adjust + bin_width)
        _hist_bins = np.arange(prediction_min_value_adjust,
                               _hist_max + bin_width, bin_width)
        _hist_counts, _hist_bin_edges = np.histogram(
            prediction_df[[prediction_col]], bins=_hist_bins)
        prediction_bins = [
            round(float(b), _HISTOGRAM_BIN_PRECISION) for b in _hist_bin_edges
        ]
        prediction_counts = _hist_counts.tolist()
        prediction_histogram = TargetHistogram(
            bin_width=_json_safe_float(bin_width),
            num_bins=len(prediction_counts),
            bins=[_json_safe_float(value) for value in prediction_bins],
            counts=[int(c) for c in prediction_counts],
        )

        # Adjust bins in `label_histogram` and `prediction_histogram` to
        # exactly match
        histogram_bin_start = min(label_histogram.bins[0],
                                  prediction_histogram.bins[0])
        histogram_bin_end = max(label_histogram.bins[-1],
                                prediction_histogram.bins[-1])

        # Adjust `label_histogram`
        label_histogram_left_padding = [
            histogram_bin_start + bin_width * x for x in range(
                round((label_histogram.bins[0] - histogram_bin_start) /
                      bin_width))
        ]
        label_histogram_right_padding = [
            label_histogram.bins[-1] + bin_width * (x + 1) for x in range(
                round((histogram_bin_end - label_histogram.bins[-1]) /
                      bin_width))
        ]
        label_histogram.bins = (label_histogram_left_padding +
                                label_histogram.bins +
                                label_histogram_right_padding)
        label_histogram.counts = ([0] * len(label_histogram_left_padding) +
                                  label_histogram.counts +
                                  [0] * len(label_histogram_right_padding))
        label_histogram.num_bins = len(label_histogram.counts)

        # Adjust `prediction_histogram`
        prediction_histogram.counts = [0] * round(
            (prediction_histogram.bins[0] - histogram_bin_start) /
            bin_width) + prediction_histogram.counts + [0] * round(
                (histogram_bin_end - prediction_histogram.bins[-1]) / bin_width)
        prediction_histogram.num_bins = len(prediction_histogram.counts)
        prediction_histogram.bins = label_histogram.bins

        prediction_mean_value = prediction_df[prediction_col].mean()
        (
            prediction_percentile_25,
            prediction_median_value,
            prediction_percentile_75,
        ) = prediction_df[prediction_col].quantile([0.25, 0.5,
                                                    0.75]).values.tolist()

        prediction_distribution = TargetHistogramAndStats(
            target_mean=_json_safe_float(prediction_mean_value),
            target_min=_json_safe_float(prediction_min_value),
            target_percentile_25=_json_safe_float(prediction_percentile_25),
            target_median=_json_safe_float(prediction_median_value),
            target_percentile_75=_json_safe_float(prediction_percentile_75),
            target_max=_json_safe_float(prediction_max_value),
            histogram=prediction_histogram,
        )

        histogram_and_stats_pair = TargetHistogramAndStatsPair(
            label=label_distribution,
            prediction=prediction_distribution,
        )
        return histogram_and_stats_pair

    def save_prediction_histogram_and_stats(
        self,
        root_dir: str,
        stage: TrainingStage,
        task: TaskOutputContext,
    ) -> None:
        # Support label vs prediction histogram for regression and
        # forecasting tasks. Forecasting produces numerical predictions
        #  similar to regression.
        if task.task_type not in {TaskType.REGRESSION, TaskType.FORECASTING}:
            return
        try:
            split_table_stats = task.get_train_table_stats(after_split=True)
            if split_table_stats is None:
                logger.debug(
                    f"Expect to have train table stats for {stage.value} set "
                    f"for label and prediction histogram generation.")
                return
            stage_table_stats = split_table_stats.get_stage_table_stats(stage)
            assert stage_table_stats is not None
            label_distribution = parse_histogram_and_stats(stage_table_stats)
            histogram_and_stats_pair = (
                self._generate_prediction_histogram_and_stats(
                    label_distribution,
                    prediction_df=self.df,
                    prediction_col='TARGET_PRED',
                ))
            path = self.get_prediction_histogram_path(root_dir, stage)
            _save_json(histogram_and_stats_pair, path)
        except Exception:
            logger.exception(
                f"Failed to save {stage.value} set histogram output")

    @staticmethod
    def _generate_prediction_heatmap(
        df: pd.DataFrame,
        target_col: str,
        prediction_col: str,
        cell_width: float,
        path: str,
    ) -> TargetPredictionHeatMap:
        # Calculate the boundary of the canvas used for heatmap plotting
        target_min_value = cast(float, df[target_col].min())
        target_max_value = cast(float, df[target_col].max())
        target_min_value_adjust = math.floor(
            target_min_value / cell_width) * cell_width
        target_max_value_adjust = max(
            target_min_value_adjust + cell_width,
            math.ceil(target_max_value / cell_width) * cell_width)
        if target_max_value_adjust == target_max_value:
            target_max_value_adjust += cell_width

        prediction_min_value = cast(float, df[prediction_col].min())
        prediction_max_value = cast(float, df[prediction_col].max())
        prediction_min_value_adjust = math.floor(
            prediction_min_value / cell_width) * cell_width
        prediction_max_value_adjust = max(
            prediction_min_value_adjust + cell_width,
            math.ceil(prediction_max_value / cell_width) * cell_width)
        if prediction_max_value_adjust == prediction_max_value:
            prediction_max_value_adjust += cell_width

        # Generate the mapping of non empty cells and counts. Note that we
        # use the bottom left corner of a cell in the heatmap to represent
        # a cell
        df[target_col] = (df[target_col] / cell_width).apply(
            math.floor) * cell_width
        df[prediction_col] = (df[prediction_col] / cell_width).apply(
            math.floor) * cell_width
        target_prediction_col = f'{target_col}_{prediction_col}'
        df[target_prediction_col] = tuple(
            zip(df[target_col], df[prediction_col]))
        bin_counts = dict(df[target_prediction_col].value_counts())
        cells = list(bin_counts.keys())
        counts = list(bin_counts.values())

        heatmap = TargetPredictionHeatMap(
            cell_width=cell_width,
            label_bin_start=target_min_value_adjust,
            label_bin_end=target_max_value_adjust,
            prediction_bin_start=prediction_min_value_adjust,
            prediction_bin_end=prediction_max_value_adjust,
            cells=cells,
            counts=counts,
            home_dir=path,
        )
        return heatmap

    def save_prediction_heatmap(
        self,
        root_dir: str,
        stage: TrainingStage,
        task: TaskOutputContext,
    ) -> None:
        # Support label vs prediction histogram for regression and forecasting tasks
        # Forecasting produces numerical predictions similar to regression
        if task.task_type not in {TaskType.REGRESSION, TaskType.FORECASTING}:
            return
        try:
            split_table_stats = task.get_train_table_stats(after_split=True)
            if split_table_stats is None:
                logger.debug(
                    f"Expect to have train table stats for {stage.value} set "
                    f"for label and prediction histogram generation.")
                return

            stage_table_stats = split_table_stats.get_stage_table_stats(stage)
            assert stage_table_stats is not None
            label_distribution = parse_histogram_and_stats(stage_table_stats)
            label_histogram = label_distribution.histogram

            cell_width = label_histogram.bin_width
            if cell_width <= 0:
                logger.error(
                    f"Label value histogram `bin_width` should always be a "
                    f"positive number for label vs prediction heatmap to "
                    f"work. However the value of `bin_width` is {cell_width}. "
                    f"This could be caused by target values having a very "
                    f"small range. For extremely small range, it is expected "
                    f"for regression heatmap to be unavailable.")
                return

            sampled_df = self.df.sample(min(self.df.shape[0], _MAX_NUM_SAMPLES))
            heatmap_dir = self.get_heatmap_dir_path(root_dir, stage)
            makedirs(heatmap_dir, exist_ok=True)
            sampled_df_path = osp.join(heatmap_dir, 'sampled_df.parquet')

            # Log the sampled data used for generating the heatmap of
            # label vs predictions
            write_frame(sampled_df, sampled_df_path)

            heatmap = self._generate_prediction_heatmap(
                df=sampled_df,
                target_col='TARGET',
                prediction_col='TARGET_PRED',
                cell_width=cell_width,
                path=heatmap_dir,
            )
            heatmap.save()
        except Exception as e:
            logger.error(
                f"Failed to save {stage.value} set heatmap output: {e}")

    @classmethod
    def load(cls, path: str) -> Self:
        """Load a versioned EvalOutput ZIP containing JSON and Parquet."""
        buffer = read_binary(path)
        with zipfile.ZipFile(io.BytesIO(buffer)) as archive:
            metadata = json.loads(archive.read(_EVAL_OUTPUT_METADATA_FILE))
            version = metadata.get('format_version')
            if version != _EVAL_OUTPUT_FORMAT_VERSION:
                raise ValueError(
                    f'Unsupported EvalOutput format version: {version}')
            df = cls._read_frame(
                archive,
                _EVAL_OUTPUT_DF_FILE,
                metadata['has_df'],
            )
            embedding = cls._read_frame(
                archive,
                _EVAL_OUTPUT_EMBEDDING_FILE,
                metadata['has_embedding'],
            )
        return cls(
            metrics=metadata['metrics'],
            df=df,
            embedding=embedding,
        )

    @staticmethod
    def _read_frame(
        archive: zipfile.ZipFile,
        name: str,
        present: bool,
    ) -> pd.DataFrame | None:
        if not present:
            return None
        return pd.read_parquet(io.BytesIO(archive.read(name)))

    @staticmethod
    def get_df_path(root_dir: str, stage: TrainingStage) -> str:
        return osp.join(root_dir, f'{stage.value}_df.parquet')

    @staticmethod
    def get_df_meta_path(root_dir: str, stage: TrainingStage) -> str:
        return osp.join(root_dir, f'{stage.value}_df_meta.json')

    @staticmethod
    def get_heatmap_dir_path(root_dir: str, stage: TrainingStage) -> str:
        return osp.join(root_dir, f'{stage.value}_target_prediction_heatmap')

    @staticmethod
    def get_prediction_histogram_path(
        root_dir: str,
        stage: TrainingStage,
    ) -> str:
        return osp.join(
            root_dir,
            f'{stage.value}_prediction_histogram_and_stats.json',
        )

    @staticmethod
    def get_output_path(root_dir: str, stage: TrainingStage) -> str:
        return osp.join(root_dir, f'{stage.value}_output.pt')


@dataclass
class PredOutput:
    r"""A dataclass that contains information produced from performing
    prediction on a trained model. Exact same specification as EvalOutput,
    but without the metrics."""

    df: pd.DataFrame
    embedding: pd.DataFrame | None = None
    # If the prediction output dataframe is too large, it will be split into
    # multiple files. This splitting is done to avoid consuming too much
    # memory on the REST driver node when copying the prediction output
    # to target destination.
    max_size_per_file: int = 1024**3  # 1GiB

    def save(
        self,
        folder: str,
        partition_id: int = 0,
        total_partitions: int = 1,
    ) -> None:
        # TODO: Always create a folder even if we have only 1 partition
        """Save prediction output to a folder. If the prediction has one
        partition, the output will be saved to a single file. Otherwise, the
        output will be saved to a folder with multiple files.

        Args:
            folder (str): the output folder.
            partition_id (int, optional): the output partition id. Defaults to
                :obj:`0`.
            total_partitions (int, optional): the output total partitions.
                Defaults to :obj:`1`.
        """
        makedirs(folder, exist_ok=True)

        prediction_file = PredictionArtifactType.PREDICTIONS.output_file_name
        prediction_path = osp.join(folder, prediction_file)
        embedding_file = PredictionArtifactType.EMBEDDINGS.output_file_name
        embedding_path = osp.join(folder, embedding_file)

        # Write the prediction output:
        self._save_prediction(prediction_path, partition_id)
        if self.embedding is not None:
            self._save_embedding(embedding_path, partition_id)

    def _save_parquet(
        self,
        df: pd.DataFrame,
        dir_path: str,
        partition_id: str,
        save_index: bool = True,
    ):
        path_prefix = osp.join(dir_path, f"part-{partition_id:05d}")
        write_frame_chunks(
            df,
            path_prefix,
            max_chunk_size=self.max_size_per_file,
            index=save_index,
        )

    def _save_embedding(self, dir_path: str, partition_id: int) -> None:
        if self.embedding is None:
            # This should never trigger in production. If it does, then
            # there's probably a bug somewhere.
            raise ValueError(
                "Failed to write embeddings because no embeddings were computed. "
                "Please contact Kumo for support.")
        makedirs(dir_path, exist_ok=True)
        # Without explicit conversion here, writing to parquet fails. See
        # issue https://github.com/kumo-ai/kumo/issues/5841
        if ("ID" in self.embedding) and (self.embedding["ID"].dtype == object):
            self.embedding = self.embedding.astype({"ID": str}, copy=False)
        self._save_parquet(
            self.embedding,
            dir_path,
            partition_id=partition_id,
        )

    def _save_prediction(self, dir_path: str, partition_id: int) -> None:
        makedirs(dir_path, exist_ok=True)
        self._save_parquet(
            _normalize_mixed_type_class_column(self.df),
            dir_path,
            partition_id=partition_id,
            save_index=False,
        )


@dataclass
class ArrowPredOutput:
    """Arrow-native prediction output for high-throughput streaming paths."""

    table: pa.Table | pa.RecordBatch
    embedding: pa.Table | pa.RecordBatch | None = None

    def to_pandas(self) -> PredOutput:
        """Materialize only when a legacy caller consumes the return value."""
        table = (pa.Table.from_batches([self.table]) if isinstance(
            self.table, pa.RecordBatch) else self.table)
        embedding = self.embedding
        if isinstance(embedding, pa.RecordBatch):
            embedding = pa.Table.from_batches([embedding])
        return PredOutput(
            table.to_pandas(),
            embedding.to_pandas() if embedding is not None else None,
        )


class StreamingPredOutputWriter:
    """Write bounded prediction and embedding frames for one output partition."""
    def __init__(
        self,
        folder: str,
        partition_id: int,
        *,
        max_size_per_file: int = _ONE_GB,
        max_buffer_size: int = _STREAMING_BUFFER_SIZE,
        output_extra_fields: dict[str, list[str]] | None = None,
        job_start_time: Any = None,
        upload_workers: int = 2,
        spool_root: str | os.PathLike[str] | None = None,
        write_predictions: bool = True,
        artifact_stats_sample_rows: Mapping[PredictionArtifactType, int]
        | None = None,
    ) -> None:
        makedirs(folder, exist_ok=True)
        prediction_dir = osp.join(
            folder, PredictionArtifactType.PREDICTIONS.output_file_name)
        embedding_dir = osp.join(
            folder, PredictionArtifactType.EMBEDDINGS.output_file_name)
        self.cleanup_partition(folder, partition_id)
        self._folder = folder
        self._partition_id = partition_id
        self._closed = False
        self._prediction_writer: StreamingFrameWriter | None = None
        if write_predictions:
            makedirs(prediction_dir, exist_ok=True)
            self._prediction_writer = StreamingFrameWriter(
                osp.join(prediction_dir, f'part-{partition_id:05d}'),
                max_file_size=max_size_per_file,
                max_buffer_size=max_buffer_size,
                index=False,
                upload_workers=upload_workers,
                spool_root=spool_root,
                stats_sample_rows=(None if artifact_stats_sample_rows is None
                                   else artifact_stats_sample_rows.get(
                                       PredictionArtifactType.PREDICTIONS, 0)),
            )
        self._embedding_dir = embedding_dir
        self._embedding_prefix = osp.join(embedding_dir,
                                          f'part-{partition_id:05d}')
        self._max_size_per_file = max_size_per_file
        self._max_buffer_size = max_buffer_size
        self._output_extra_fields = output_extra_fields
        self._job_start_time = job_start_time
        self._upload_workers = upload_workers
        self._spool_root = spool_root
        self._collect_artifact_stats = artifact_stats_sample_rows is not None
        self._artifact_stats_sample_rows = dict(artifact_stats_sample_rows
                                                or {})
        self._embedding_writer: StreamingFrameWriter | None = None
        self.last_output: PredOutput | ArrowPredOutput | None = None

    @staticmethod
    def cleanup_partition(folder: str, partition_id: int) -> None:
        partition_prefix = f'part-{partition_id:05d}-'
        directories = (
            osp.join(folder,
                     PredictionArtifactType.PREDICTIONS.output_file_name),
            osp.join(folder,
                     PredictionArtifactType.EMBEDDINGS.output_file_name),
        )
        for directory in directories:
            if not filesystem.exists(directory):
                continue
            for path in filesystem.listdir(directory):
                name = osp.basename(path)
                final_part = (name.startswith(partition_prefix)
                              and name.endswith('.parquet'))
                local_publish_temp = (name.startswith(f'.{partition_prefix}')
                                      and name.endswith('.tmp'))
                if final_part or local_publish_temp:
                    filesystem.rm(path)

    def _finalize_schema(
        self,
        frame: pd.DataFrame,
        artifact_type: PredictionArtifactType,
    ) -> pd.DataFrame:
        if self._output_extra_fields is None:
            return frame
        requested = set(self._output_extra_fields.get(artifact_type.value, []))
        metadata_columns = {'ANCHOR_TIMESTAMP', 'JOB_TIMESTAMP'}
        drop_columns = [
            column for column in metadata_columns
            if column in frame and column not in requested
        ]
        if drop_columns:
            frame = frame.drop(columns=drop_columns)
        if ('JOB_TIMESTAMP' in requested and 'JOB_TIMESTAMP' not in frame
                and self._job_start_time is not None):
            frame = frame.copy(deep=False)
            frame['JOB_TIMESTAMP'] = pd.Timestamp(
                self._job_start_time).tz_localize(None)
        missing = requested.difference(frame.columns)
        if missing:
            raise ValueError(
                f'Requested {artifact_type.value} metadata field(s) were not '
                f'produced: {", ".join(sorted(missing))}')
        return frame

    def _finalize_arrow_schema(
        self,
        table: pa.Table,
        artifact_type: PredictionArtifactType,
    ) -> pa.Table:
        if self._output_extra_fields is None:
            return table
        requested = set(self._output_extra_fields.get(artifact_type.value, []))
        metadata_columns = {'ANCHOR_TIMESTAMP', 'JOB_TIMESTAMP'}
        keep = [
            name for name in table.column_names
            if name not in metadata_columns or name in requested
        ]
        table = table.select(keep)
        if ('JOB_TIMESTAMP' in requested
                and 'JOB_TIMESTAMP' not in table.column_names
                and self._job_start_time is not None):
            value = pd.Timestamp(self._job_start_time).tz_localize(None)
            column = pa.array(np.full(table.num_rows, value.to_datetime64()))
            if artifact_type is PredictionArtifactType.PREDICTIONS:
                column = column.cast(pa.timestamp('us'), safe=False)
            table = table.append_column('JOB_TIMESTAMP', column)
        missing = requested.difference(table.column_names)
        if missing:
            raise ValueError(
                f'Requested {artifact_type.value} metadata field(s) were not '
                f'produced: {", ".join(sorted(missing))}')
        return table

    @staticmethod
    def _normalize_arrow_timestamps(table: pa.Table) -> pa.Table:
        fields = []
        columns = []
        changed = False
        for field, column in zip(table.schema, table.columns):
            if pa.types.is_timestamp(field.type) and field.type.unit != 'us':
                field = pa.field(field.name, pa.timestamp('us'), field.nullable,
                                 field.metadata)
                column = column.cast(pa.timestamp('us'), safe=False)
                changed = True
            fields.append(field)
            columns.append(column)
        if not changed:
            return table
        return pa.Table.from_arrays(
            columns,
            schema=pa.schema(fields, metadata=table.schema.metadata),
        )

    def write(self, output: PredOutput | ArrowPredOutput) -> None:
        if isinstance(output, ArrowPredOutput):
            table = (pa.Table.from_batches([output.table]) if isinstance(
                output.table, pa.RecordBatch) else output.table)
            if self._prediction_writer is not None:
                table = self._finalize_arrow_schema(
                    table, PredictionArtifactType.PREDICTIONS)
                table = self._normalize_arrow_timestamps(table)
                self._prediction_writer.write(table)
            elif table.num_rows:
                raise ValueError(
                    'Prediction rows were produced when predictions were not '
                    'requested.')
            if output.embedding is not None:
                self.write_embedding(output.embedding)
            self.last_output = output
            return
        df = _normalize_mixed_type_class_column(output.df)
        if self._prediction_writer is not None:
            df = self._finalize_schema(df, PredictionArtifactType.PREDICTIONS)
            df = _normalize_prediction_timestamps(df)
            # A stream can have an all-numeric object batch followed by a string
            # batch. Canonicalize object-backed classes before fixing the first
            # Parquet schema; native numerical CLASS columns remain numerical.
            if 'CLASS' in df and df['CLASS'].dtype == object:
                df = df.copy(deep=False)
                df['CLASS'] = df['CLASS'].map(lambda value: None
                                              if pd.isna(value) else str(value))
            self._prediction_writer.write(df)
        elif not df.empty:
            raise ValueError(
                'Prediction rows were produced when predictions were not '
                'requested.')
        if output.embedding is not None:
            self.write_embedding(output.embedding)
        self.last_output = output

    def write_embedding(
        self,
        embedding: pd.DataFrame | pa.Table | pa.RecordBatch,
    ) -> None:
        if isinstance(embedding, (pa.Table, pa.RecordBatch)):
            table = (pa.Table.from_batches([embedding]) if isinstance(
                embedding, pa.RecordBatch) else embedding)
            table = self._finalize_arrow_schema(
                table, PredictionArtifactType.EMBEDDINGS)
            embedding = table
        else:
            embedding = self._finalize_schema(embedding,
                                              PredictionArtifactType.EMBEDDINGS)
            if 'ID' in embedding and embedding['ID'].dtype == object:
                embedding = embedding.astype({'ID': str}, copy=False)
        if self._embedding_writer is None:
            makedirs(self._embedding_dir, exist_ok=True)
            self._embedding_writer = StreamingFrameWriter(
                self._embedding_prefix,
                max_file_size=self._max_size_per_file,
                max_buffer_size=self._max_buffer_size,
                index=False,
                upload_workers=self._upload_workers,
                spool_root=self._spool_root,
                stats_sample_rows=(None if not self._collect_artifact_stats else
                                   self._artifact_stats_sample_rows.get(
                                       PredictionArtifactType.EMBEDDINGS, 0)),
            )
        self._embedding_writer.write(embedding)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._prediction_writer is not None:
                self._prediction_writer.close()
            if self._embedding_writer is not None:
                self._embedding_writer.close()
        except Exception:
            try:
                self.abort()
            except Exception:
                logger.exception(
                    'Failed to abort prediction output after close failure')
            raise

    @property
    def artifact_stats(self) -> dict[PredictionArtifactType, FrameWriterStats]:
        if not self._closed:
            raise RuntimeError(
                "Prediction artifact stats require a closed writer")
        result = {}
        if self._prediction_writer is not None:
            result[PredictionArtifactType.PREDICTIONS] = (
                self._prediction_writer.stats)
        if self._embedding_writer is not None:
            result[PredictionArtifactType.EMBEDDINGS] = (
                self._embedding_writer.stats)
        return result

    def abort(self) -> None:
        self._closed = True
        try:
            if self._prediction_writer is not None:
                self._prediction_writer.abort()
        finally:
            if self._embedding_writer is not None:
                self._embedding_writer.abort()
            self.cleanup_partition(self._folder, self._partition_id)

    def __enter__(self) -> 'StreamingPredOutputWriter':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
