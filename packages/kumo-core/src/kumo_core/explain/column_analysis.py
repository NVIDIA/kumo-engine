# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os.path as osp
from collections import defaultdict
from itertools import accumulate
from typing import Any, Literal

import pandas as pd
import torch
from kumoapi.explain import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiNestedTensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import one_hot, scatter
from torch_geometric.utils.sparse import ptr2index

from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper, TensorMapper, TimestampTensorMapper
from kumo_core.explain.base import Explainer, continue_on_error
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
from kumo_core.encoders.target.multicategorical import MultiCategoricalTargetEncoder
from kumo_core.types.features import NANOSECONDS_IN_SECONDS, TimeUnit
from kumo_core.features.time import get_rel_time
from kumo_core.io.filesystem import makedirs
from kumo_core.io.json import save_json

TableHop = tuple[str, int]
ColID = tuple[str, Stype | None]

# The column name in the final output that references count-based cohorts.
COUNT_COL = 'COUNT(*)'

# The column name in the final output that references remaining categories.
OTHER_COL = 'OTHER(*)'

NA_COHORT = 'N/A'


class ColumnAnalysis(Explainer):
    r"""A column analysis explainer that collects column-level cohorts from raw
    data and maps them to their average ground-truth/predictions.

    :class:`ColumnAnalysis` is implemented as a two-stage approach, where
    entity-level information is collected for each column and each hop within
    each mini-batch by directly operating on the raw data from
    :class:`TensorFrame` (see :meth:`explain_post_step`).
    Afterwards, entity-level data is aggregated into cohorts, separated per
    hop (see :meth:`explain_end`).

    Args:
        num_cohorts: The maximum number of cohorts to generate for a column.
        normalize: Whether to normalize final prediction and target scores
            across all columns.
        max_classes: The maximum number of classes to support for
            multi-class/multi-label tasks.
        max_categories: The maximum number of categorical values to collect as
            cohort candidates.
    """
    supported_task_types = {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
        TaskType.REGRESSION,
    }
    requires_grad: bool = False

    def __init__(
        self,
        task_type: TaskType,
        root_dir: str | None,
        num_cohorts: int = 10,
        normalize: bool = True,
        max_classes: int = 30,
        max_categories: int = 500,
    ):
        super().__init__(task_type, root_dir)

        self.num_cohorts = num_cohorts
        self.normalize = normalize
        self.max_classes = max_classes
        self.max_categories = max_categories

    def explain_start(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        stage: TrainingStage,
    ) -> None:
        self._dfs_dict: dict[TableHop, list[pd.DataFrame]] = defaultdict(list)
        self._num_nodes_dict: dict[TableHop, int] = defaultdict(int)
        self._cols_dict: dict[TableHop, set[ColID]] = defaultdict(set)
        self._time_window_dict: dict[tuple[str, str, int], int] = {}
        self._time_unit_dict: dict[tuple[str, str, int], TimeUnit] = {}

    @continue_on_error
    def explain_post_step(
        self,
        batch: HeteroData,
        module: torch.nn.Module,
        datamodule: Any,
        step_output: dict[str, Tensor],
        stage: TrainingStage,
        batch_idx: int,
    ) -> None:
        r""":meth:`explain_post_step` collects entity-level information for
        each column within each mini-batch by directly operating on the raw
        data from :class:`TensorFrame`.
        As such, it outputs a :class:`pandas.DataFrame`, where each row
        corresponds to an entity we are making a prediction for, and each
        column collects statistics about this entity (*e.g.*, number of facts
        linked to it, average of numerical values, distribution of categorical
        values, etc).
        """
        feature_store, graph_store = datamodule.data

        batch_size: int | None = None
        seed_time: Tensor | None = None
        for store in batch.node_stores:
            if 'batch_size' in store:
                batch_size = store.batch_size
                seed_time = store.get('seed_time', None)
        assert batch_size is not None

        # Remove the training table from the set of inputs:
        train_table_name = feature_store.train_table_name
        tf_dict = {
            table_name: tf
            for table_name, tf in batch.tf_dict.items()
            if table_name != train_table_name
        }
        batch_dict = {
            table_name: batch
            for table_name, batch in batch.batch_dict.items()
            if table_name != train_table_name
        }
        table_mapper_dict: dict[str, dict[str, TensorMapper]] = {}
        for table_name, tf in tf_dict.items():
            table_mapper_dict[table_name] = {}
            for col_names in tf.col_names_dict.values():
                for col_name in col_names:
                    m = feature_store.get_tensor_mapper(table_name, col_name)
                    table_mapper_dict[table_name][col_name] = m

        num_sampled_nodes_dict = batch.collect('num_sampled_nodes', True)
        num_sampled_nodes_dict = {
            table_name: num_sampled_nodes
            for table_name, num_sampled_nodes in
            num_sampled_nodes_dict.items() if table_name != train_table_name
        }

        return self.step(
            table_mapper_dict=table_mapper_dict,
            tf_dict=tf_dict,
            batch_dict=batch_dict,
            edge_index_dict=batch.collect('edge_index', allow_empty=True),
            num_sampled_nodes_dict=num_sampled_nodes_dict,
            batch_size=batch_size,
            seed_time=seed_time,
            exclude_cols_dict=module.encoder.exclude_cols_dict,
        )

    def step(
        self,
        table_mapper_dict: dict[str, dict[str, TensorMapper]],
        tf_dict: dict[str, TensorFrame],
        batch_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
        num_sampled_nodes_dict: dict[str, list[int]],
        batch_size: int,
        seed_time: Tensor | None,
        exclude_cols_dict: dict[str, list[str]] | None = None,
    ) -> None:

        if seed_time is not None:
            seed_time = seed_time // NANOSECONDS_IN_SECONDS

        for table_name, num_sampled_nodes in num_sampled_nodes_dict.items():
            cumsum_sampled_nodes = [0] + list(accumulate(num_sampled_nodes))
            mapper_dict = table_mapper_dict[table_name]
            tf = tf_dict[table_name]
            batch = batch_dict[table_name]
            exclude_cols = (exclude_cols_dict or {}).get(table_name, [])

            iterator = zip(cumsum_sampled_nodes[:-1], cumsum_sampled_nodes[1:])
            for hop, (start, end) in enumerate(iterator):
                dfs: list[pd.DataFrame] = []

                self._num_nodes_dict[(table_name, hop)] += end - start

                df = self.collect_count(
                    batch=batch[start:end],
                    batch_size=batch_size,
                )
                dfs.append(df)
                self._cols_dict[(table_name, hop)].add((COUNT_COL, None))

                for stype, feats in tf.feat_dict.items():
                    col_names = tf.col_names_dict[stype]

                    for i, col_name in enumerate(col_names):
                        if col_name in exclude_cols:
                            continue

                        if stype not in {
                                Stype.numerical,
                                Stype.categorical,
                                Stype.multicategorical,
                                Stype.timestamp,
                        }:
                            continue

                        kwargs = dict(
                            feat=feats[start:end, i],
                            batch=batch[start:end],
                            batch_size=batch_size,
                            col_name=col_name,
                            mapper=mapper_dict[col_name],
                        )

                        if stype == Stype.numerical:
                            df = self.collect_numerical(**kwargs)

                        elif stype == Stype.categorical:
                            df = self.collect_categorical(**kwargs)

                        elif stype == Stype.multicategorical:
                            df = self.collect_multicategorical(**kwargs)

                        else:
                            assert stype == Stype.timestamp
                            if seed_time is None:
                                continue

                            index = TimestampTensorMapper.index['unix']
                            time = feats[start:end, i:i + 1, index]

                            # Find time window and unit which we use to group
                            # into cohorts. We compute this from the first
                            # batch and re-use for subsequent batches.
                            key = (table_name, col_name, hop)
                            if key not in self._time_window_dict:
                                notna_mask = feats[start:end, i:i + 1, 0] != -1
                                notna_time = time[notna_mask]
                                if notna_time.numel() > 0:
                                    min_time = int(notna_time.min())
                                    max_time = int(notna_time.max())
                                else:
                                    min_time = max_time = 0

                                time_window, time_unit = self.get_time_cohort(
                                    min_time=min_time,
                                    max_time=max_time,
                                    num_cohorts=self.num_cohorts,
                                )

                                self._time_window_dict[key] = time_window
                                self._time_unit_dict[key] = time_unit
                            else:
                                time_window = self._time_window_dict[key]
                                time_unit = self._time_unit_dict[key]

                            rel_time = get_rel_time(
                                time=time,
                                seed_time=seed_time[batch[start:end]],
                                time_unit=time_unit,
                            ).view(-1)

                            df = self.collect_rel_time(
                                feat=rel_time,
                                batch=batch[start:end],
                                batch_size=batch_size,
                                time_window=time_window,
                                time_unit=time_unit,
                                col_name=col_name,
                            )

                        dfs.append(df)
                        self._cols_dict[(table_name, hop)].add(
                            (col_name, stype))

                if len(dfs) > 0:
                    df = pd.concat(dfs, axis=1)
                    self._dfs_dict[(table_name, hop)].append(df)

    @continue_on_error
    def explain_end(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        outputs: dict[str, Tensor],
        stage: TrainingStage,
    ) -> ColumnAnalysisOutput | None:
        r""":meth:`explain_end` aggregates entity-level information for
        each column into cohort-level information.
        """
        df_dict = {
            key: pd.concat(dfs).reset_index(drop=True)
            for key, dfs in self._dfs_dict.items()
            if self._num_nodes_dict[key] > 0
        }
        del self._dfs_dict
        del self._num_nodes_dict
        del self._time_window_dict
        del self._time_unit_dict

        target_mapper: TensorMapper | None = None
        if isinstance(module.target_encoder, CategoricalTargetEncoder
                      | MultiCategoricalTargetEncoder):
            target_mapper = module.target_encoder.tensor_mapper

        pred, target = self.transform_prediction(
            stage=stage,
            outputs=outputs,
            num_nodes=max([0] + [len(df) for df in df_dict.values()]),
            max_classes=self.max_classes,
            target_encoder=getattr(module, 'target_encoder', None),
        )
        pred = pred.cpu()
        if target is not None:
            target = target.cpu()

        out = ColumnAnalysisOutput(task_type=self.task_type)

        for (table_name, hop), df in df_dict.items():
            for col_name, stype in self._cols_dict[(table_name, hop)]:
                if col_name == COUNT_COL or stype == Stype.numerical:
                    out[table_name, hop, col_name] = self.aggregate_range(  #
                        df, pred, target, col_name, target_mapper, stype)

                else:
                    assert stype in {
                        Stype.timestamp,
                        Stype.categorical,
                        Stype.multicategorical,
                    }
                    out[table_name, hop, col_name] = self.aggregate_category(
                        df, pred, target, col_name, target_mapper, stype)

        del self._cols_dict

        if self.normalize:
            out = out.normalize()

        # Save column analysis results:
        if self.root_dir is not None:
            root_dir = osp.join(self.root_dir, stage)
            makedirs(root_dir, exist_ok=True)
            save_json(out, osp.join(root_dir, 'column_analysis.json'))
            return None

        return out

    # Collect Helper functions ################################################

    def collect_count(
        self,
        batch: Tensor,
        batch_size: int,
    ) -> pd.DataFrame:
        r"""Computes the number of rows in a table that are linked to a given
        entity. This boils down to a simple :meth:`scatter_sum` based on the
        :obj:`batch` vector.
        """
        out = scatter(torch.ones_like(batch), batch, 0, batch_size, 'sum')
        return pd.DataFrame({COUNT_COL: out.cpu().numpy()})

    def collect_numerical(
        self,
        feat: Tensor,
        batch: Tensor,
        batch_size: int,
        col_name: str,
        mapper: TensorMapper,
    ) -> pd.DataFrame:
        r"""Computes the average of numerical values in a column that are
        linked to a given entity. This boils down to a simple
        :meth:`scatter_mean` based on the :obj:`batch` vector.
        """
        # Handle NaN values gracefully:
        valid_mask = ~(feat.isnan() | feat.isinf())
        feat, batch = feat[valid_mask], batch[valid_mask]

        # Average values that belong to the same entity:
        out = feat.new_full((batch_size, ), fill_value=float('NaN'))
        out.scatter_reduce_(0, batch, feat, reduce='mean', include_self=False)

        return pd.DataFrame({col_name: out.cpu().numpy()})

    def collect_categorical(
        self,
        feat: Tensor,
        batch: Tensor,
        batch_size: int,
        col_name: str,
        mapper: TensorMapper,
    ) -> pd.DataFrame:
        r"""Computes the average appearance of individual categories in a
        column that are linked to a given entity. This boils down to a simple
        category-level :meth:`scatter_mean` on its one-hot representation based
        on the :obj:`batch` vector.

        Returns a :class:`pandas.DataFrame` with a column for every category,
        referenced via :obj:`{col_name}:{category_name}`.
        """
        # Handle NaN values and large cardinality gracefully:
        assert isinstance(mapper, CategoricalTensorMapper)
        mapping = mapper.mapping.tolist()

        valid_mask = feat < len(mapping)
        feat, batch = feat[valid_mask], batch[valid_mask]

        num_cohorts = self.max_categories
        if len(mapping) > num_cohorts:
            mapping = mapping[:num_cohorts - 1] + [OTHER_COL]
            feat[feat >= num_cohorts] = num_cohorts - 1

        # Assign N/A values to their own category:
        feat[feat < 0] = len(mapping)
        mapping.append(NA_COHORT)

        # Convert categories into a one-hot representation and average their
        # values that belong to the same entity:
        feat = one_hot(feat, num_classes=len(mapping))
        out = scatter(feat, batch, dim=0, dim_size=batch_size, reduce='mean')
        return pd.DataFrame(
            out.cpu().numpy(),
            columns=[f'{col_name}:{value}' for value in mapping],
        )

    def collect_multicategorical(
        self,
        feat: MultiNestedTensor,
        batch: Tensor,
        batch_size: int,
        col_name: str,
        mapper: TensorMapper,
    ) -> pd.DataFrame:
        r"""Computes the average appearance of individual categories in a
        column that are linked to a given entity. This boils down to a simple
        category-level :meth:`scatter_mean` on its dense 0/1 representation
        based on the :obj:`batch` vector.

        Returns a :class:`pandas.DataFrame` with a column for every category,
        referenced via :obj:`{col_name}:{category_name}`.
        """
        rowptr, col = feat.offset, feat.values
        row = ptr2index(rowptr)

        # Handle NaN values and large cardinality gracefully:
        assert isinstance(mapper, MultiCategoricalTensorMapper)
        mapping = mapper.mapping.tolist()

        valid_mask = (col >= 0) & (col < len(mapping))
        row, col = row[valid_mask], col[valid_mask]

        num_cohorts = self.max_categories
        if len(mapping) > num_cohorts:
            mapping = mapping[:num_cohorts - 1] + [OTHER_COL]
            col[col >= num_cohorts] = num_cohorts - 1

        size = (rowptr.numel() - 1, len(mapping))
        dense_feat = torch.zeros(size, device=rowptr.device)
        dense_feat[row, col] = 1.0

        out = scatter(dense_feat, batch, 0, batch_size, reduce='mean')

        return pd.DataFrame(
            out.cpu().numpy(),
            columns=[f'{col_name}:{value}' for value in mapping],
        )

    def collect_rel_time(
        self,
        feat: Tensor,
        batch: Tensor,
        batch_size: int,
        time_window: int,
        time_unit: TimeUnit,
        col_name: str,
    ) -> pd.DataFrame:

        valid_mask = feat >= 0 & ~feat.isnan()  # Drop future-looking time.
        feat, batch = feat[valid_mask], batch[valid_mask]

        bins = torch.arange(
            start=0,
            end=(self.num_cohorts + 1) * time_window,
            step=time_window,
            device=feat.device,
        )
        feat = torch.bucketize(feat, bins[1:-1])

        # Convert timestamps into a one-hot representation and average their
        # values that belong to the same entity:
        feat = one_hot(feat, num_classes=self.num_cohorts)
        out = scatter(feat, batch, 0, dim_size=batch_size, reduce='mean')

        columns = [
            f'{col_name}:[{s} {time_unit}, {e} {time_unit})'
            for s, e in zip(bins[:-1], bins[1:])
        ]

        return pd.DataFrame(out.cpu().numpy(), columns=columns)

    # Aggregation Helper Functions ############################################

    def aggregate_range(
        self,
        df: pd.DataFrame,
        predictions: Tensor,
        targets: Tensor | None,
        col_name: str,
        target_mapper: TensorMapper | None,
        stype: Stype | None,
    ) -> ColumnCohorts | None:

        if col_name not in df:
            return None

        values = torch.from_numpy(df[col_name].values)

        entity_mask = ~values.isnan()
        values = values[entity_mask]
        predictions = predictions[entity_mask]
        targets = targets[entity_mask] if targets is not None else None

        if values.numel() > 0:  # Bins are given via quantiles on the raw data:
            quantiles = torch.linspace(0, 1, self.num_cohorts + 1)
            bins = values.to(torch.float).quantile(quantiles)
        else:
            bins = values

        # TODO This logic is a bit flawed since we likely merge cohorts that
        # should be separate, e.g., [0.5, 1.5, 1.8, 2.5] would be merged to
        # [0, 1, 2] and hence group {0, 1} into the same cohort.
        if not values.is_floating_point():
            bins = bins.ceil().to(values.dtype)

        bins = bins.unique()
        if len(bins) == 1:  # Fix for columns with only a single entry:
            bins = torch.cat([bins, bins])

        # Map each entry in the raw data to a bin:
        if col_name == COUNT_COL and bins[0] == 0:
            # Keep zero count as a separate bin for COUNT(*):
            dim_size = max(bins.numel(), 0)
            boundaries = bins[:-1]
        else:
            dim_size = max(bins.numel() - 1, 0)
            boundaries = bins[1:-1]

        index = torch.bucketize(values, boundaries=boundaries)

        # Average model predictions and ground-truth according to bin mapping:
        count = scatter(torch.ones_like(index), index, 0, dim_size, 'sum')
        predictions = scatter(predictions, index, 0, dim_size, reduce='mean')

        if targets is not None:
            targets = scatter(targets, index, 0, dim_size, reduce='mean')

        # Filter bins with zero elements:
        mask = count > 0
        count = count[mask]
        predictions = predictions[mask]
        targets = targets[mask] if targets is not None else None

        if col_name == COUNT_COL and bins[0] == 0:
            # Add an extra zero to bins for final interval representation:
            bins = torch.cat([bins[:1], bins[mask]])
        else:
            bins = torch.cat([bins[:1], bins[1:][mask]])

        # Convert bins to a string representation:
        bins_repr = [f'{value:.4g}' for value in bins.tolist()]

        # Convert to range/interval representation, use as cohort description:
        intervals_repr = list(zip(bins_repr[:-1], bins_repr[1:]))
        cohorts: list[str] = []
        for i, (s, e) in enumerate(intervals_repr):
            if col_name == COUNT_COL and i == len(intervals_repr) - 1:
                # Since the end interval of `COUNT(*)` is approximated through
                # sampling, signalize that there potentially exists more links.
                e = f'{e}+'
            cohorts.append(f'[{s} - {e}]' if i == 0 else f'({s} - {e}]')

        return ColumnCohorts(
            stype=stype,
            cohorts=cohorts,
            populations=(count / count.sum()).tolist(),
            predictions=self.to_human_readable_prediction(
                outputs=predictions,
                tensor_mapper=target_mapper,
            ),
            targets=self.to_human_readable_prediction(
                outputs=targets,
                tensor_mapper=target_mapper,
            ) if targets is not None else None,
            prediction_score=self.get_score(predictions, count, method='std'),
            target_score=self.get_score(targets, count, method='std')
            if targets is not None else None,
        )

    def aggregate_category(
        self,
        df: pd.DataFrame,
        predictions: Tensor,
        targets: Tensor | None,
        col_name: str,
        target_mapper: TensorMapper | None,
        stype: Stype,
    ) -> ColumnCohorts | None:

        # Collect relative frequency of all categories and store in a matrix:
        # Include the colon because some columns have name as prefix of others
        col_names = [v for v in df.columns if v.startswith(f'{col_name}:')]
        if len(col_names) == 0:
            return None
        freq_mat = torch.from_numpy(df[col_names].values)

        # Determine per category cohorts:
        cohorts = [value.split(':')[-1] for value in col_names]
        # Population count is given by the frequency of categories:
        count = freq_mat.sum(dim=0)

        # For predictions/targets, we weight the relative frequency of a
        # (entity, category) by the score of its entity, and divide that by the
        # global relative frequency of this category:
        freq_mat = freq_mat.unsqueeze(-1)  # [num_rows, num_categories, 1]

        for _ in range(3 - predictions.dim()):  # [num_rows, 1, num_classes]
            predictions = predictions.unsqueeze(1)
        predictions = (freq_mat * predictions).sum(dim=0) / freq_mat.sum(dim=0)

        if targets is not None:
            for _ in range(3 - targets.dim()):  # [num_rows, 1, num_classes]
                targets = targets.unsqueeze(1)
            targets = (freq_mat * targets).sum(dim=0) / freq_mat.sum(dim=0)

        # Filter out all categories with zero/NaN population count:
        mask = count > 0
        cohorts = [val for val, keep in zip(cohorts, mask.tolist()) if keep]
        count = count[mask]
        predictions = predictions[mask]
        targets = targets[mask] if targets is not None else None

        # NOTE Discard `OTHER_COL` here when computing the score.
        prediction_score = self.get_score(
            predictions[:-1] if OTHER_COL in cohorts else predictions,
            count[:-1] if OTHER_COL in cohorts else count,
            method='std',
        )
        target_score = self.get_score(
            targets[:-1] if OTHER_COL in cohorts else targets,
            count[:-1] if OTHER_COL in cohorts else count,
            method='std',
        ) if targets is not None else None

        # Post-process final cohorts in case of high-cardinality columns:
        if len(cohorts) > self.num_cohorts:
            split = self.num_cohorts - 1
            weight = count[split:].view(-1, 1) / count[split:].sum()

            cohorts = cohorts[:split] + [OTHER_COL]
            count = torch.cat([
                count[:split],
                count[split:].sum(dim=0, keepdim=True),
            ], dim=0)
            predictions = torch.cat([
                predictions[:split],
                (weight * predictions[split:]).sum(dim=0, keepdim=True),
            ], dim=0)
            if targets is not None:
                targets = torch.cat([
                    targets[:split],
                    (weight * targets[split:]).sum(dim=0, keepdim=True),
                ], dim=0)

        return ColumnCohorts(
            stype=stype,
            cohorts=cohorts,
            populations=(count / count.sum()).tolist(),
            predictions=self.to_human_readable_prediction(
                outputs=predictions,
                tensor_mapper=target_mapper,
            ),
            targets=self.to_human_readable_prediction(
                outputs=targets,
                tensor_mapper=target_mapper,
            ) if targets is not None else None,
            prediction_score=prediction_score,
            target_score=target_score,
        )

    # Variance-based Score ####################################################

    @staticmethod
    def get_score(
        inputs: Tensor,
        count: Tensor,
        method: Literal['var', 'std'],
    ) -> float:
        r"""Computes the column-level score based on model predictions across
        cohorts, weighted by population count.
        If the input is two-dimensional (e.g., in multi-label or multi-class
        classification scenarios), will first compute the individual scores per
        class, and will then average scores across all classes.

        Column-level scores are either given via variance (:obj:`method="var"`)
        or standard deviation (:obj:`method="std"`).
        """
        assert method in ['var', 'std']

        assert inputs.dim() >= 1 and inputs.dim() <= 2
        assert count.dim() == 1
        assert inputs.size(0) == count.numel()

        count = count.view(-1, 1) if inputs.dim() > 1 else count

        if count.sum() <= 1:
            return 0.0

        weighted_mean = (inputs * count).sum(dim=0, keepdim=True) / count.sum()

        numerator = (count * (inputs - weighted_mean)**2).sum(dim=0)
        denominator = count.sum() - 1  # Bessel's correction.

        score = numerator / denominator
        score = score.sqrt() if method == 'std' else score

        return float(score.mean())

    @staticmethod
    def get_time_cohort(
        min_time: int,
        max_time: int,
        num_cohorts: int,
    ) -> tuple[int, TimeUnit]:

        seconds = max_time - min_time

        if seconds == 0:
            return (1, 'months')  # Fallback in case of empty or full N/A.

        hours = seconds // (60 * 60)
        days = hours // 24
        weeks = days // 7
        months = days // 30

        if hours < 7 * num_cohorts:  # Group no more than 6 hours:
            return max(hours // num_cohorts, 1), 'hours'
        if days < 5 * num_cohorts:  # Group no more than 4 days:
            return max(days // num_cohorts, 1), 'days'
        elif weeks < 3 * num_cohorts:  # Group no more than 2 weeks:
            return max(weeks // num_cohorts, 1), 'weeks'
        # Return months for everything else:
        return max(months // num_cohorts, 1), 'months'


def pretty_print(self: ColumnAnalysisOutput, top_k: int | None = None) -> None:
    from tabulate import tabulate

    cohorts = sorted(
        self.cohorts,
        key=lambda cohort: cohort.prediction_score,
        reverse=True,
    )

    if top_k is not None:
        cohorts = cohorts[:top_k]

    headers = ['Table', 'Column', 'Hop', 'Score']
    data = [(
        cohort.table_name,
        cohort.col_name,
        cohort.hop,
        cohort.prediction_score,
    ) for cohort in cohorts]

    table = tabulate(data, headers=headers, tablefmt='github')
    print(table)


ColumnAnalysisOutput.pretty_print = pretty_print
