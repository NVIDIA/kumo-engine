# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os.path as osp
import shutil
from collections import defaultdict
from datetime import datetime
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from kumoapi.explain import (
    AggregatedNodes,
    Feature,
    LinkPredRFM,
    NodePredRFM,
    Subgraph,
    SubgraphFeatureStore,
    SubgraphNode,
    SubgraphStore,
)
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype, TimeUnit
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils import cumsum, scatter
from torch_geometric.utils.sparse import index2ptr

from kumo_core.io import filesystem as fsutil
from kumo_core.io.json import save_json
from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper
from kumo_core.explain.base import Explainer, continue_on_error
from kumo_core.features.time import to_datetime
from kumo_core.explain.schemas import (
    ColumnData,
    EntityOutput,
    HistoricalItems,
    LinkData,
    NodeData,
    NodeOutputData,
    ResidualData,
)
from kumo_core.explain.schemas import Subgraph as OldSubgraph

if TYPE_CHECKING:
    from kumo_core.features.stores.feature import FeatureStore
    from kumo_core.features.stores.graph import GraphStore

FORMAT = Literal['csv', 'nested_json', 'output', 'historical_items', 'rfm']

# Max length for the extraction of sequence stype column
SEQUENCE_MAX_LENGTH = 2


class SubgraphCollector(Explainer):
    r"""A Subgraph Collector that collects graph-level information from
    example-wise sampled subgraph.

    Args:
        formats (list[str]): The list of output formats.

            * :obj:`"csv"`: Returns the subgraph as a set of CSV files.
            * :obj:`"nested_json"`: Returns the subgraph as a nested JSON
              file, to be displayed in the UI.
            * :obj:`"output"`: Returns the individual predictions and
              ground-truth labels of a single subgraph.
            * :obj:`"historical_items"`: Returns the historical items within
              a single subgraph. Only applicable in link prediction tasks.

        max_num_facts (int, optional): The maximum number of facts to return
            per foreign key (for :obj:`formats="nested_json"`).
            (default: :obj:`16`)
        remote_sync_dir (str, optional): A remote directory to sync subgraph
            collector results in batches. Improves efficiency if specified.
            (default: :obj:`None`)
        target_to_entity_paths: The paths that connect the target to the
            entity.
    """
    supported_task_types = {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
        TaskType.REGRESSION,
        TaskType.STATIC_LINK_PREDICTION,
        TaskType.TEMPORAL_LINK_PREDICTION,
        # TaskType.MULTILABEL_RANKING,  # TODO: add back after supported
    }

    # Whether this explainer requires gradients to be computed:
    requires_grad: bool = True

    def __init__(
        self,
        task_type: TaskType,
        root_dir: str,
        formats: list[FORMAT],
        max_num_facts: int = 16,
        remote_sync_dir: str | None = None,
        target_to_entity_paths: list[list[EdgeType]] | None = None,
    ):
        super().__init__(task_type, root_dir)

        if not task_type.is_link_pred and 'historical_items' in formats:
            raise ValueError("Format 'historical items' is only supported "
                             "for link prediction tasks")

        self.formats = formats
        self.max_num_facts = max_num_facts
        self.remote_sync_dir = remote_sync_dir
        self.target_to_entity_paths = target_to_entity_paths

        self._idx = 0

    def explain_pre_step(
        self,
        batch: HeteroData,
        module: torch.nn.Module,
        datamodule: Any,
        stage: TrainingStage,
        batch_idx: int,
    ) -> None:
        if module.training:
            raise RuntimeError(f"Execution of '{self.__class__.__name__}' is "
                               f"only applicable during model inference")

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
        df_dict: dict[str, pd.DataFrame] = {}
        grad_dict: dict[str, dict[str, Tensor]] = {}
        feature_store: FeatureStore = datamodule.data[0]
        graph_store: GraphStore = datamodule.data[1]

        # Gradients ###########################################################
        if requires_grad(step_output):
            grad_dict = Explainer.get_grad(self.task_type, batch.tf_dict,
                                           module, step_output)

        # Calibrate logits if module is node pred and the calibrator is fitted
        if hasattr(module, 'calibrator') and module.calibrator.is_fitted:
            step_output['logits'] = module.calibrator(step_output['logits'])

        for table_name in feature_store.table_names:
            df = pd.DataFrame()

            if pkey_name := feature_store.get_pkey_name(table_name):
                df[pkey_name] = batch[table_name].pkey

            fkeys = graph_store.get_foreign_keys(table_name)
            for fkey_name, dst_table_name in fkeys:
                df[fkey_name] = torch.full(  # Fill with dummy values.
                    (batch[table_name].num_nodes, ),
                    fill_value=-1,
                    dtype=torch.long,
                ).numpy()

                # Currently, it is a bit tricky to recover foreign key data.
                # For this, we iterate over each `edge_index` representation
                # that either links from or to this foreign key and collect
                # their source or destination node indices:
                # NOTE Not all foreign key data may be filled by this.
                for edge_type, edge_index in batch.edge_index_dict.items():
                    src, rel, dst = edge_type

                    if src == table_name and rel.endswith(fkey_name):
                        index = edge_index[0]
                        local_dst_node_id = edge_index[1]
                    elif dst == table_name and rel.endswith(fkey_name):
                        index = edge_index[1]
                        local_dst_node_id = edge_index[0]
                    else:
                        continue

                    dst_pkey = batch[dst_table_name].get('pkey')
                    if dst_pkey is not None:
                        _set_column_values(
                            df, fkey_name, index.cpu().numpy(),
                            dst_pkey[local_dst_node_id.cpu().numpy()])

            tf: TensorFrame = batch[table_name].tf
            feat_df = _tf_to_df(tf, table_name, feature_store)
            df = pd.concat([df, feat_df], axis=1)
            unsupported_feat_dict = getattr(tf, '_unsupported_feat_dict', {})
            for col_name, data in unsupported_feat_dict.items():
                df[col_name] = data

            time_col_name = feature_store.get_time_col_name(table_name)
            if time_col_name is not None and time_col_name not in df:
                # The sampler only attaches `time` to node types it reached;
                # an unreached type is an empty block without it:
                if 'time' in batch[table_name] or len(df) > 0:
                    time = batch[table_name].time
                else:
                    time = torch.empty(0, dtype=torch.long)
                ser = to_datetime(time.cpu().numpy())
                df[time_col_name] = ser

            df_dict[table_name] = df

        # Collect seed entity IDs:
        seed_table = batch.input_type
        assert isinstance(seed_table, str)
        batch_size = batch[seed_table].batch_size
        seed_entities = batch[seed_table].pkey[:batch_size].tolist()

        # pred is the RHS target prediction
        pred: Tensor | None = None
        rhs_index_map: pd.DataFrame | None = None
        pred_df: pd.DataFrame | None = None
        # Whether need to fetch RHS pkeys for some formats in link prediction
        need_fetch = any(value in ['output', 'nested_json']
                         for value in self.formats)
        y_idx: Tensor | None = None
        y_batch: Tensor | None = None
        if need_fetch and self.task_type.is_link_pred:
            assert isinstance(module.entity_table_names, tuple | list)
            rhs_name = module.entity_table_names[1]
            pred = step_output['target'].cpu()
            pred_idx, pred_inverse = pred.view(-1).unique(return_inverse=True)

            # Get ground truth RHS target predictions #########################
            if stage != TrainingStage.PRED:
                y_idx, y_batch = get_link_pred_ground_truth(
                    module, batch, stage)
                y_idx, y_batch = y_idx.cpu(), y_batch.cpu()
            rhs_idx = pred_idx if y_idx is None else torch.cat(
                [y_idx, pred_idx]).unique()
            with feature_store.safe_batch_mode(batch_size=50_000):
                rhs_arr = feature_store[rhs_name, 'pkey', rhs_idx]

            # Use n_id as the index for the index mapping
            rhs_index_map = pd.DataFrame(
                {feature_store.get_pkey_name(rhs_name): rhs_arr},
                index=rhs_idx.numpy())

            # Retrieve raw prediction RHS features ############################
            tf = feature_store[rhs_name, 'tf', pred_idx]
            pred_df = _tf_to_df(tf, rhs_name, feature_store)
            unsupported_feat_dict = getattr(tf, '_unsupported_feat_dict', {})
            for col_name, data in unsupported_feat_dict.items():
                pred_df[col_name] = data
            if len(pred_df.columns) > 0:
                pred_df = pred_df.iloc[pred_inverse.numpy()].reset_index(
                    drop=True)
            pkey_name = feature_store.get_pkey_name(rhs_name)
            pred_df[pkey_name] = rhs_index_map.loc[pred.view(
                -1).numpy()][pkey_name].tolist()

        for value in self.formats:
            if value == 'csv':
                self.format_csv(  #
                    batch, seed_entities, df_dict, grad_dict, stage)
            elif value == 'nested_json':
                self.format_subgraph_json(  #
                    batch, seed_entities, df_dict, grad_dict, stage,
                    feature_store, graph_store, module, rhs_index_map, pred,
                    pred_df)
            elif value == 'output':
                self.format_output(  #
                    module, seed_entities, step_output, stage, feature_store,
                    rhs_index_map, pred, pred_df, y_idx, y_batch)
            elif value == 'historical_items':
                self.format_historical_items(  #
                    batch, seed_entities, df_dict, stage, feature_store)
            elif value == 'rfm':
                self.format_rfm(  #
                    seed_table, feature_store, batch, seed_entities, stage,
                    step_output, module)
            else:
                raise ValueError(f"Found unsupported format '{format}'")
        self._idx += len(seed_entities)

    @continue_on_error
    def explain_end(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        outputs: dict[str, Tensor],
        stage: TrainingStage,
    ) -> None:
        if (self.remote_sync_dir is not None
                and self.root_dir != self.remote_sync_dir):
            fsutil.copy(
                osp.join(self.root_dir, stage.value),
                osp.join(self.remote_sync_dir, stage.value),
                overwrite=True,
            )

    # Formatting ##############################################################

    def format_csv(
        self,
        batch: HeteroData,
        seed_entities: list[Any],
        df_dict: dict[str, pd.DataFrame],
        grad_dict: dict[str, dict[str, Tensor]],
        stage: TrainingStage,
    ) -> None:

        sorted_df_dict: dict[str, pd.DataFrame] = {}
        offset_dict: dict[str, Tensor] = {}
        for table_name, df in df_dict.items():
            df = df.copy(deep=False)
            # Append column scores to data frame:
            for col_name, grad in grad_dict.get(table_name, {}).items():
                df[f'{col_name}_KUMO_SCORE'] = grad.cpu().numpy()
            # Sort data frame by examples ...
            batch_vec, perm = batch[table_name].batch.sort(stable=True)
            sorted_df_dict[table_name] = df.iloc[perm.cpu().numpy()]
            # ... and store their boundary offsets:
            offset = index2ptr(batch_vec, size=len(seed_entities)).cpu()
            offset_dict[table_name] = offset

        # Save `*.csv` files by slicing all data frames:
        for i in range(len(seed_entities)):
            path = osp.join(self.root_dir, stage.value, 'csv',
                            str(self._idx + i))
            fsutil.makedirs(path, exist_ok=True)
            for table_name, df in sorted_df_dict.items():
                start = int(offset_dict[table_name][i])
                end = int(offset_dict[table_name][i + 1])
                csv_path = osp.join(path, f'{table_name}.csv')
                df[start:end].to_csv(csv_path, escapechar='\\', index=False)

            shutil.make_archive(path, 'zip', path)

    def format_subgraph_json(
        self,
        batch: HeteroData,
        seed_entities: list[Any],
        df_dict: dict[str, pd.DataFrame],
        grad_dict: dict[str, dict[str, Tensor]],
        stage: TrainingStage,
        feature_store: FeatureStore,
        graph_store: GraphStore,
        module: torch.nn.Module,
        rhs_index_map: pd.DataFrame | None = None,
        pred: Tensor | None = None,
        pred_df: pd.DataFrame | None = None,
    ) -> None:
        seed_table = batch.input_type
        assert isinstance(seed_table, str)

        batch_size = batch[seed_table].batch_size
        score_dict = get_normalized_grad_dict(batch, seed_table, grad_dict)

        num_sampled_edges_dict = batch.collect('num_sampled_edges',
                                               allow_empty=True)
        if len(num_sampled_edges_dict) > 0:
            num_hops = len(list(batch.num_sampled_edges_dict.values())[0])
        else:
            # When graph just contains a single table:
            num_hops = 0

        # Use original directional edges in case they are available.
        # Otherwise, ensure that `edge_index` is on CPU:
        edge_index_dict: dict[EdgeType, Tensor] = {}
        for edge_type, store in batch.edge_items():
            if hasattr(store, '_orig_edge_index'):
                edge_index_dict[edge_type] = store._orig_edge_index
            else:
                edge_index_dict[edge_type] = store.edge_index.cpu()

        seed_time: pd.Series | None = None
        if 'seed_time' in batch[seed_table]:
            seed_time = to_datetime(batch[seed_table].seed_time.cpu().numpy())

        # The edges in `edge_index_dict` are guaranteed to be supported by hop
        # and by example. As such, we generate an offset dictionary to allow
        # for fast slicing of edges for a specific (hop, example) in O(1):
        offset_dict: dict[EdgeType, Tensor] = {}
        for edge_type, edge_index in edge_index_dict.items():
            num_sampled_edges = batch[edge_type].num_sampled_edges

            # `edge_batch` (num_edges, ): [0, ...,  batch_size - 1]
            edge_batch = batch[edge_type[0]].cpu().batch[edge_index[0]]
            # `hop_batch` (num_edges, ): [0, ..., num_hops * batch_size - 1]
            hop_batch = torch.arange(
                0,
                batch_size * len(num_sampled_edges),
                batch_size,
            ).repeat_interleave(
                torch.tensor(num_sampled_edges),
                output_size=edge_index.size(1),
            )
            offset = index2ptr(edge_batch + hop_batch, num_hops * batch_size)
            offset_dict[edge_type] = offset

        cumsum_sampled_nodes_dict = {
            node_type: [0] + list(accumulate(num_sampled_nodes))
            for node_type, num_sampled_nodes in
            batch.num_sampled_nodes_dict.items()
        }

        # Allow for fast retrieval of links from or to a specific table:
        table_link_dict: dict[str, list[EdgeType]] = defaultdict(list)
        for edge_type in batch.edge_types:
            table_link_dict[edge_type[0]].append(edge_type)
            table_link_dict[edge_type[-1]].append(edge_type)

        def _get_cols(
            table_name: str,
            query_id: int,
            is_subgraph: bool = True,
        ) -> list[Feature] | list[ColumnData]:
            r"""Returns the column data of a particular row in the subgraph."""
            ser = df_dict[table_name].iloc[query_id]

            score: dict[str, Tensor] | None = score_dict.get(table_name, None)

            cols = []
            col_names = df_dict[table_name].columns
            pkey_name = feature_store.get_pkey_name(table_name)
            fkey_names = {
                fkey[0]
                for fkey in graph_store.get_foreign_keys(table_name)
            }
            for col_name in col_names:
                if col_name == pkey_name:
                    continue
                if col_name in fkey_names:
                    continue
                grad_score: float | None = None
                if score is not None and col_name in score:
                    grad_score = float(score[col_name][query_id])

                value = to_python_scalar(ser[col_name])
                stype = feature_store.get_tensor_mapper(
                    table_name,
                    col_name,
                ).stype

                if is_subgraph:
                    col_data = Feature(
                        table_name=table_name,
                        name=col_name,
                        value=value,
                        stype=stype,
                        score=grad_score,
                    )
                else:
                    col_data = ColumnData(
                        name=col_name,
                        value=value,
                        stype=stype,
                        grad_score=grad_score,
                    )

                cols.append(col_data)

            time_col_name = feature_store.get_time_col_name(table_name)
            if time_col_name is not None and time_col_name not in col_names:
                grad_score = None
                if score is not None and time_col_name in score:
                    grad_score = float(score[time_col_name][query_id])

                if is_subgraph:
                    col_data = Feature(
                        table_name=table_name,
                        name=time_col_name,
                        value=ser[time_col_name],
                        stype=Stype.timestamp,
                        score=grad_score,
                    )
                else:
                    col_data = ColumnData(
                        name=time_col_name,
                        value=ser[time_col_name],
                        stype=Stype.timestamp,
                        grad_score=grad_score,
                    )
                cols.append(col_data)

            return cols

        def _get_links(
            table_name: str,
            query_id: int,
            seed_id: int,
            hop: int,
        ) -> list[LinkData]:
            r"""Recursively constructs :class:`LinkData` objects."""
            if hop >= num_hops:
                return []

            links: list[LinkData] = []
            for edge_type in table_link_dict[table_name]:
                src, rel, dst = edge_type

                offset = offset_dict[edge_type]
                start = offset[hop * batch_size + seed_id]
                end = offset[hop * batch_size + seed_id + 1]

                if end - start == 0:
                    continue

                edge_index = edge_index_dict[edge_type][:, start:end]

                fkey_name = rel[4:] if rel.startswith('rev_') else rel
                if table_name == src:
                    dst_table_name = dst
                    fkeys = edge_index[1, edge_index[0] == query_id]
                else:
                    dst_table_name = src
                    fkeys = edge_index[0, edge_index[1] == query_id]

                if fkeys.numel() == 0:
                    continue

                # Ignore "back-links" to tables sampled in the previous layer.
                # We consider links as back-links in case all sampled nodes
                # refer to nodes that have been already sampled previously.
                num_prev_nodes = cumsum_sampled_nodes_dict[dst_table_name][hop]
                if fkeys.max() < num_prev_nodes:
                    continue

                # Reverse foreign keys such that they are ordered by recency:
                fkeys = fkeys.flip(0)
                fkey_names = {
                    fkey[0]
                    for fkey in graph_store.get_foreign_keys(dst_table_name)
                }
                fkey_ids: list[Any] | None = None
                if fkey_name in fkey_names:
                    fkey_ser = df_dict[dst_table_name][fkey_name].values
                    fkey_ids = fkey_ser[fkeys.numpy()].tolist()

                pkey_ids: list[Any] | None = None
                if pkey_name := feature_store.get_pkey_name(dst_table_name):
                    pkey_ser = df_dict[dst_table_name][pkey_name].values
                    pkey_ids = pkey_ser[fkeys.numpy()].tolist()

                nodes: list[NodeData] = []
                for i, fkey in enumerate(fkeys[:self.max_num_facts].tolist()):
                    node_data = NodeData(
                        fkey_name=fkey_name
                        if fkey_name != pkey_name else None,
                        fkey=fkey_ids[i] if fkey_ids is not None else None,
                        pkey_name=pkey_name,
                        pkey=pkey_ids[i] if pkey_ids is not None else None,
                        cols=_get_cols(
                            dst_table_name,
                            query_id=fkey,
                            is_subgraph=False,
                        ),
                        links=_get_links(
                            dst_table_name,
                            query_id=fkey,
                            seed_id=seed_id,
                            hop=hop + 1,
                        ),
                    )
                    nodes.append(node_data)

                residual: ResidualData | None = None
                if fkeys.numel() > self.max_num_facts:
                    start_time = end_time = None
                    if 'time' in batch[dst_table_name]:
                        time = batch[dst_table_name].time
                        time = time[fkeys[self.max_num_facts:]]
                        ser = to_datetime(time.cpu().numpy())
                        start_time = ser.min().to_pydatetime()
                        end_time = ser.max().to_pydatetime()

                    residual = ResidualData(
                        count=fkeys.numel() - self.max_num_facts,
                        start_time=start_time,
                        end_time=end_time,
                    )

                link_data = LinkData(
                    table_name=dst_table_name,
                    fkey_name=fkey_name,
                    nodes=nodes,
                    residual=residual,
                )
                links.append(link_data)

            return links

        nested_path = osp.join(self.root_dir, stage.value, 'nested_json')
        fsutil.makedirs(nested_path, exist_ok=True)
        path = osp.join(self.root_dir, stage.value, 'subgraph_json')
        fsutil.makedirs(path, exist_ok=True)

        id_name = feature_store.get_pkey_name(seed_table)

        # Handle RHS nodes for link prediction tasks ##########################
        rhs_name: str | None = None
        rhs_pkey_name: str | None = None
        if rhs_index_map is not None:
            assert isinstance(module.entity_table_names, tuple | list)
            assert isinstance(module.entity_table_names[1], str)
            rhs_name = module.entity_table_names[1]
            assert rhs_name is not None
            rhs_pkey_name = feature_store.get_pkey_name(rhs_name)

        for seed_id, seed_entity in enumerate(seed_entities):
            seed: datetime | None = None
            if seed_time is not None:
                seed = seed_time[seed_id].to_pydatetime()

            # To be removed:
            nested_subgraph = OldSubgraph(
                id=seed_entity,
                id_name=id_name,
                table_name=seed_table,
                task_type=self.task_type,
                seed_time=seed,
                cols=_get_cols(seed_table, seed_id, is_subgraph=False),
                links=_get_links(seed_table, seed_id, seed_id, hop=0),
            )

            self.residual_id: int = -1
            root = SubgraphNode(
                id=seed_id + batch.node_offsets[seed_table],
                pkey_name=id_name,
                pkey=seed_entity,
            )
            subgraph_store = SubgraphStore(
                root_id=root.id,
                task_type=self.task_type,
                seed_time=seed,
            )
            subgraph_store.add_node(root, seed_table)
            subgraph_feature_store = SubgraphFeatureStore()
            subgraph_feature_store.add(
                root.id,
                cols=_get_cols(seed_table, seed_id),
            )
            visited: set[tuple[str, int]] = {(seed_table, seed_id)}
            src_stack: list[tuple[str, int]] = [(seed_table, seed_id)]
            while len(src_stack) > 0:
                table_name, idx = src_stack.pop()
                src_node_id = idx + batch.node_offsets[table_name]
                src_node = subgraph_store[src_node_id]
                edge_types = get_edge_types(table_name, batch)
                edge_types += get_edge_types(table_name, batch, reverse=True)
                for edge_type in edge_types:
                    src, rel, dst = edge_type
                    rel = rel[4:] if rel.startswith('rev_') else rel
                    if src == table_name:
                        mask = (batch[edge_type].edge_index[0, :] == idx)
                        dst_select_idx = 1
                        dst_table_name = dst
                    else:
                        mask = (batch[edge_type].edge_index[1, :] == idx)
                        dst_select_idx = 0
                        dst_table_name = src
                    if dst_table_name not in df_dict:
                        continue
                    if mask.sum() == 0:
                        continue
                    dst_indices = batch[edge_type].edge_index[dst_select_idx]
                    dst_indices = dst_indices[mask]
                    if dst_score_dict := score_dict.get(dst_table_name):
                        dst_score = torch.stack(
                            list(dst_score_dict.values()),
                            dim=-1,
                        )[dst_indices].sum(dim=-1)
                        perm = dst_score.argsort(descending=True)
                        dst_indices = dst_indices[perm]  # Sort by score.
                    dst_indices = dst_indices.cpu()
                    pkeys: list[Any] | None = None
                    pkey_name: str | None = feature_store.get_pkey_name(
                        dst_table_name)
                    if pkey_name is not None:
                        pkey_ser = df_dict[dst_table_name][pkey_name].values
                        pkeys = pkey_ser.tolist()
                    for i, dst_index in enumerate(dst_indices):
                        dst_index = int(dst_index)
                        if i >= self.max_num_facts:
                            start_time = end_time = None
                            if 'time' in batch[dst_table_name]:
                                time = batch[dst_table_name].time
                                time = time[dst_indices[i:]]
                                ser = to_datetime(time.cpu().numpy())
                                start_time = ser.min().to_pydatetime()
                                end_time = ser.max().to_pydatetime()

                            residual_nodes = AggregatedNodes(
                                id=self.residual_id,
                                count=len(dst_indices) - self.max_num_facts,
                                start_time=start_time,
                                end_time=end_time,
                                pkey_name=pkey_name,
                            )
                            assert isinstance(self.residual_id, int)
                            self.residual_id -= 1
                            subgraph_store.add_node(residual_nodes,
                                                    dst_table_name)
                            # Get one column data
                            cols = _get_cols(
                                dst_table_name,
                                query_id=int(dst_indices[self.max_num_facts]),
                            )
                            score: dict[str, Tensor] | None = score_dict.get(
                                dst_table_name, None)
                            index = dst_indices[self.max_num_facts:]
                            for col in cols:
                                assert isinstance(col, Feature)
                                # Set value to None as the data is aggregated
                                col.value = None
                                if score is not None and col.name in score:
                                    # Sum aggregate the gradient scores
                                    grads: Tensor = score[col.name][index]
                                    col.score = float(grads.sum())
                            subgraph_feature_store.add(
                                residual_nodes.id,
                                cols=cols,
                            )
                            subgraph_store.add_edge(
                                src_node,
                                residual_nodes,
                                rel,
                            )
                            # Mark all nodes inside the residual as visited.
                            for res_index in dst_indices[i:]:
                                visited.add((dst_table_name, int(res_index)))
                            break

                        if (dst_table_name, dst_index) in visited:
                            continue
                        pkey = pkeys[dst_index] if pkeys is not None else None
                        n_id = dst_index + batch.node_offsets[dst_table_name]
                        dst_node = SubgraphNode(
                            id=n_id,
                            pkey_name=pkey_name,
                            pkey=pkey,
                        )
                        subgraph_store.add_node(dst_node, dst_table_name)
                        subgraph_store.add_edge(src_node, dst_node, rel)
                        subgraph_feature_store.add(
                            dst_node.id,
                            cols=_get_cols(dst_table_name, dst_index),
                        )
                        visited.add((dst_table_name, dst_index))
                        src_stack.append((dst_table_name, dst_index))

            if self.task_type.is_link_pred:
                # For each seed entity, find the RHS nodes that do not exist
                # in the batch subgraph:
                batch_mask = (batch[rhs_name].batch == seed_id)
                n_id = batch[rhs_name].n_id[batch_mask]
                assert pred is not None
                rhs_mask = ~torch.isin(pred[seed_id], n_id)
                rhs_n_ids = pred[seed_id][rhs_mask]
                for rhs_n_id in rhs_n_ids:
                    # Add RHS target node as a "new" node to the subgraph
                    assert rhs_index_map is not None
                    pkey = rhs_index_map.loc[int(rhs_n_id)][rhs_pkey_name]
                    pkey = to_python_scalar(pkey)
                    node = SubgraphNode(
                        id=self.residual_id,
                        pkey_name=rhs_pkey_name,
                        pkey=pkey,
                    )
                    subgraph_store.add_node(node, rhs_name)
                    assert pred_df is not None
                    mask = (pred_df[rhs_pkey_name] == pkey)
                    # It's possible that there are multiple rows here
                    # as the pred may contain duplicated predictions
                    features = pred_df[mask].iloc[0].to_dict()
                    # TODO (zecheng): Add gradient scores for each RHS node.
                    assert rhs_name is not None
                    cols = [
                        Feature(
                            table_name=rhs_name,
                            name=col_name,
                            # Convert value to Python scalar
                            value=to_python_scalar(value),
                            stype=feature_store.get_tensor_mapper(
                                rhs_name,
                                col_name,
                            ).stype,
                        ) for col_name, value in features.items()
                        if col_name != rhs_pkey_name
                    ]
                    subgraph_feature_store.add(
                        node.id,
                        cols=cols,
                    )
                    self.residual_id -= 1
            self.residual_id = -1

            name = f'{self._idx + seed_id}.json'
            save_json(nested_subgraph, osp.join(nested_path, name))
            subgraph = Subgraph(
                subgraph_store=subgraph_store,
                feature_store=subgraph_feature_store,
            )
            save_json(subgraph, osp.join(path, name))

    def format_rfm(
        self,
        seed_table: str,
        feature_store: FeatureStore,
        batch: HeteroData,
        seed_entities: list[Any],
        stage: TrainingStage,
        step_output: dict[str, Tensor],
        module: torch.nn.Module,
    ) -> None:
        seed_times: pd.Series | None = None
        if 'seed_time' in batch[seed_table]:
            seed_times = to_datetime(batch[seed_table].seed_time.cpu().numpy())

        path = osp.join(self.root_dir, stage.value, 'rfm')
        fsutil.makedirs(path, exist_ok=True)

        entity_table = feature_store.entity_table_names[0]
        edge_types = get_edge_types(entity_table, batch)

        pred: Tensor | None = None
        y_idx: Tensor | None = None
        y_batch: Tensor | None = None
        rhs_idx: Tensor | None = None
        lhs_idx: Tensor | None = None
        if self.task_type.is_link_pred:
            pred = step_output['target'].cpu()
            if stage != TrainingStage.PRED:
                y_idx, y_batch = get_link_pred_ground_truth(
                    module, batch, stage)
                y_idx, y_batch = y_idx.cpu(), y_batch.cpu()
            paths = self.target_to_entity_paths
            assert paths is not None
            rhs_idx, lhs_idx = get_target_rhs_id(paths, batch)
            if rhs_idx.numel() > 0:  # To global ids, like `pred`.
                rhs_table_name = _strip_edge_type(paths[1][0])[2]
                rhs_idx = batch[rhs_table_name].n_id.cpu()[rhs_idx]

        for seed_id, _ in enumerate(seed_entities):
            rfm = get_default_rfm(seed_table, self.task_type, edge_types,
                                  batch, seed_id, seed_times)
            if self.task_type.is_node_pred:
                assert isinstance(rfm, NodePredRFM)
                rfm.unique_counts = get_unique_counts(edge_types, batch,
                                                      seed_id, seed_table,
                                                      feature_store)
            elif self.task_type.is_link_pred:
                assert isinstance(rfm, LinkPredRFM)
                assert pred is not None
                assert rhs_idx is not None
                assert lhs_idx is not None
                set_link_pred_rfm(rfm, seed_id, pred, rhs_idx, lhs_idx, y_idx,
                                  y_batch)
            name = f'{self._idx + seed_id}.json'
            save_json(rfm, osp.join(path, name))

    def format_output(
        self,
        module: torch.nn.Module,
        seed_entities: list[Any],
        step_output: dict[str, Tensor],
        stage: TrainingStage,
        feature_store: FeatureStore,
        rhs_index_map: pd.DataFrame | None = None,
        pred: Tensor | None = None,
        pred_df: pd.DataFrame | None = None,
        y_idx: Tensor | None = None,
        y_batch: Tensor | None = None,
    ) -> None:
        path = osp.join(self.root_dir, stage.value, 'output')
        fsutil.makedirs(path, exist_ok=True)

        if not self.task_type.is_link_pred:
            pred, target = self.transform_prediction(
                stage=stage,
                outputs=step_output,
                target_encoder=module.target_encoder,
            )

            if self.task_type in {
                    TaskType.MULTICLASS_CLASSIFICATION,
                    TaskType.MULTILABEL_CLASSIFICATION,
            }:
                assert module.target_encoder is not None
                target_mapper = module.target_encoder.tensor_mapper
                assert isinstance(
                    target_mapper,
                    CategoricalTensorMapper | MultiCategoricalTensorMapper)
                mapping = target_mapper.mapping

                preds = [{
                    v: row[i]
                    for i, v in enumerate(mapping)
                } for row in pred.tolist()]
                targets = None
                if target is not None:
                    targets = [{
                        v: row[i]
                        for i, v in enumerate(mapping)
                    } for row in target.tolist()]

            else:
                preds = pred.view(-1).tolist()
                targets = None if target is None else target.view(-1).tolist()

            assert pred is not None
            for i in range(len(seed_entities)):
                output = EntityOutput(
                    task_type=self.task_type,
                    prediction=preds[i],
                    target=targets[i] if targets is not None else None,
                )
                save_json(
                    output,
                    osp.join(path, f'{self._idx + i}.json'),
                )
            return

        assert self.task_type.is_link_pred

        # Retrieve raw ground-truth RHS features ##############################
        rhs_table_name = module.rhs_name
        pkey_name = feature_store.get_pkey_name(rhs_table_name)
        y_df: pd.DataFrame | None = None
        y_offsets: list[int] | None = None
        if stage != TrainingStage.PRED:
            assert y_idx is not None
            assert y_batch is not None
            assert rhs_index_map is not None
            tf = feature_store[rhs_table_name, 'tf', y_idx]
            y_df = _tf_to_df(tf, rhs_table_name, feature_store)
            unsupported_feat_dict = getattr(tf, '_unsupported_feat_dict', {})
            for col_name, data in unsupported_feat_dict.items():
                y_df[col_name] = data
            y_df[pkey_name] = rhs_index_map.loc[
                y_idx.numpy()][pkey_name].tolist()

            y_count = scatter(
                torch.ones_like(y_batch),
                y_batch,
                dim_size=len(seed_entities),
                reduce='sum',
            )
            y_offsets = cumsum(y_count).tolist()

        # Construct output features ###########################################
        for i in range(len(seed_entities)):
            assert pred is not None
            assert pred_df is not None
            start = i * pred.size(1)
            end = start + pred.size(1)
            predictions: list[NodeOutputData] = []
            for j in range(start, end):
                cols: list[ColumnData] = []
                for col_name in pred_df.columns:
                    if col_name == pkey_name:
                        continue

                    value = to_python_scalar(pred_df[col_name][j])
                    # Convert NaType to None as it's not serializable
                    if pd.isna(value):
                        value = None

                    stype = feature_store.get_tensor_mapper(
                        rhs_table_name, col_name).stype
                    col_data = ColumnData(
                        name=col_name,
                        value=value,
                        stype=stype,
                    )
                    cols.append(col_data)
                node = NodeOutputData(
                    id=to_python_scalar(pred_df[pkey_name][j]),
                    cols=cols,
                    id_name=pkey_name,
                )
                predictions.append(node)

            targets: list[NodeOutputData] | None = None
            if stage != TrainingStage.PRED:
                assert y_df is not None and y_offsets is not None
                start = y_offsets[i]
                end = y_offsets[i + 1]

                targets = []
                for j in range(start, end):
                    cols = []
                    for col_name in y_df.columns:
                        if col_name == pkey_name:
                            continue

                        value = to_python_scalar(y_df[col_name][j])
                        # Convert NaType to None as it's not serializable
                        if pd.isna(value):
                            value = None

                        stype = feature_store.get_tensor_mapper(
                            rhs_table_name, col_name).stype

                        col_data = ColumnData(
                            name=col_name,
                            value=value,
                            stype=stype,
                        )
                        cols.append(col_data)
                    node = NodeOutputData(
                        id=to_python_scalar(y_df[pkey_name][j]),
                        cols=cols,
                        id_name=pkey_name,
                    )
                    targets.append(node)

            output = EntityOutput(
                task_type=self.task_type,
                prediction=predictions,
                target=targets,
            )
            save_json(output, osp.join(path, f'{self._idx + i}.json'))

    def format_historical_items(
        self,
        batch: HeteroData,
        seed_entities: list[Any],
        df_dict: dict[str, pd.DataFrame],
        stage: TrainingStage,
        feature_store: FeatureStore,
    ) -> None:
        # TODO: Remove Task dependency from this method.
        assert self.target_to_entity_paths is not None
        rhs_idx, lhs_idx = get_target_rhs_id(self.target_to_entity_paths,
                                             batch)
        rhs_table_name = _strip_edge_type(self.target_to_entity_paths[1][0])[2]

        df = df_dict[rhs_table_name]
        path = osp.join(self.root_dir, stage.value, 'historical_items')
        pkey_name = feature_store.get_pkey_name(rhs_table_name)

        for i in range(len(seed_entities)):
            nodes: list[NodeOutputData] = []

            # Edges are not grouped by seed:
            for node_idx in rhs_idx[lhs_idx == i].unique().tolist():
                ser = df.iloc[node_idx]

                cols: list[ColumnData] = []
                for col_name in df.columns:
                    if col_name == pkey_name:
                        continue

                    value = to_python_scalar(ser[col_name])
                    # Convert NaType to None as it's not serializable
                    if pd.isna(value):
                        value = None

                    stype = feature_store.get_tensor_mapper(
                        rhs_table_name, col_name).stype
                    col_data = ColumnData(
                        name=col_name,
                        value=value,
                        stype=stype,
                    )
                    cols.append(col_data)
                node = NodeOutputData(
                    id=to_python_scalar(ser[pkey_name]),
                    cols=cols,
                    id_name=pkey_name,
                )
                nodes.append(node)

            output = HistoricalItems(nodes)
            save_json(output, osp.join(path, f'{self._idx + i}.json'))


def requires_grad(step_output: dict[str, Tensor]) -> bool:
    r"""Whether a `step_output` requires gradients."""
    if 'logits' in step_output and step_output['logits'].requires_grad:
        return True
    if 'score' in step_output and step_output['score'].requires_grad:
        return True
    return False


def _set_column_values(
    df: pd.DataFrame,
    column: str,
    rows: Any,
    values: Any,
) -> None:
    r"""Sets ``df.loc[rows, column] = values``. DiskGraph attaches primary
    keys as an object array; a numeric column is widened to ``object``
    first, as pandas did implicitly with a deprecation warning before
    pandas 3 rejects the assignment."""
    if getattr(values, 'dtype', None) == object and df[column].dtype != object:
        df[column] = df[column].astype(object)
    df.loc[rows, column] = values


def _tf_to_df(
    tf: TensorFrame,
    table_name: str,
    feature_store: FeatureStore,
) -> pd.DataFrame:
    data = {}
    for stype, feats in tf.feat_dict.items():
        for i, col_name in enumerate(tf.col_names_dict[stype]):
            if Stype(stype) == Stype.sequence:
                ser = pd.Series(feats[:, i].values.tolist(), dtype=object)
                ser = ser.map(lambda x: '[' + ', '.join(
                    [f'{v:.2g}' for v in x[:SEQUENCE_MAX_LENGTH]]) + ', ...]')
            else:
                mapper = feature_store.get_tensor_mapper(table_name, col_name)
                ser = mapper.backward(feats[:, i])
            data[col_name] = ser
    return pd.DataFrame(data)


def get_default_rfm(
    seed_table: str,
    task_type: TaskType,
    edge_types: list[EdgeType],
    batch: HeteroData,
    seed_id: int,
    seed_times: pd.Series | None,
) -> NodePredRFM | LinkPredRFM:
    if task_type.is_node_pred:
        rfm_cls = NodePredRFM
    elif task_type.is_link_pred:
        rfm_cls = LinkPredRFM
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    rfm = rfm_cls(entity_table_name=seed_table)
    count: dict[str, int | float] = {}
    rel_time: dict[str, Tensor] = {}
    for edge_type in edge_types:
        dst_table = edge_type[-1]
        if len(batch[edge_type].num_sampled_edges) < 1:
            continue
        dst_nodes = get_linked_nodes(
            edge_type,
            batch,
            seed_id,
            start=0,
            end=batch[edge_type].num_sampled_edges[0],
        )
        rel = edge_type[1]
        rel = rel[4:] if rel.startswith('rev_') else f'rev_{rel}'
        rev_edge_type = (edge_type[2], rel, edge_type[0])
        rev_dst_nodes = dst_nodes.new_empty((0, ))
        if len(batch[rev_edge_type]) > 0:
            # Get the source nodes of the reverse edge type
            rev_dst_nodes = get_linked_nodes(
                rev_edge_type,
                batch,
                seed_id,
                start=0,
                end=batch[rev_edge_type].num_sampled_edges[0],
                dst=False,
            )
        dst_nodes = torch.cat([dst_nodes, rev_dst_nodes]).cpu()
        if dst_nodes.numel() == 0:
            continue
        count[dst_table] = dst_nodes.unique().numel()
        if seed_times is None or 'time' not in batch[dst_table]:
            continue
        # Convert time to days
        factor = 1000**3 * 3600 * 24
        seed_time = seed_times[seed_id].to_numpy().astype(int) // factor
        time: Tensor = batch[dst_table]['time'][dst_nodes] // factor
        rel_time[dst_table] = seed_time - time

    avg_rel_time: dict[str, tuple[float, TimeUnit]] = {
        key: (int(sum(rel_time[key])) / rel_time[key].numel(), TimeUnit.DAYS)
        for key in rel_time
    }

    most_recent_rel_time: dict[str, tuple[int | float, TimeUnit]] = {
        key: (int(min(rel_time[key])), TimeUnit.DAYS)
        for key in rel_time
    }
    rfm.count = count
    rfm.avg_rel_time = avg_rel_time
    rfm.most_recent_rel_time = most_recent_rel_time
    return rfm


def get_unique_counts(
    edge_types: list[EdgeType],
    batch: HeteroData,
    seed_id: int,
    seed_table: str,
    feature_store: FeatureStore,
) -> dict[str, dict[str, int | float]]:
    r"""Get the second-hop unique counts for items in node prediction task."""
    unique_counts: dict[str, dict[str, int | float]] = {}
    for edge_type in edge_types:
        dst_table = edge_type[-1]
        # Skip if there is no second-hop edges
        if len(batch[edge_type].num_sampled_edges) < 2:
            continue
        rel = edge_type[1]
        dst_nodes = get_linked_nodes(
            edge_type,
            batch,
            seed_id,
            start=0,
            end=batch[edge_type].num_sampled_edges[0],
        )
        rel = rel[4:] if rel.startswith('rev_') else f'rev_{rel}'
        rev_edge_type = (edge_type[2], rel, edge_type[0])
        rev_dst_nodes = dst_nodes.new_empty((0, ))
        if len(batch[rev_edge_type]) > 0:
            # Get the source nodes of the reverse edge type
            rev_dst_nodes = get_linked_nodes(
                rev_edge_type,
                batch,
                seed_id,
                start=0,
                end=batch[rev_edge_type].num_sampled_edges[0],
                dst=False,
            )
        dst_nodes = torch.cat([dst_nodes, rev_dst_nodes])
        if dst_nodes.numel() == 0:
            continue
        linked_edge_types = get_edge_types(dst_table, batch)
        linked_edge_types = [
            edge_type for edge_type in linked_edge_types
            if edge_type[2] != seed_table
        ]
        for linked_edge_type in linked_edge_types:
            table = linked_edge_type[-1]
            pkey_name = feature_store.get_pkey_name(table)
            # Skip if there is no pkey
            if pkey_name is None:
                continue
            start = batch[linked_edge_type].num_sampled_edges[0]
            end = start + batch[linked_edge_type].num_sampled_edges[1]
            nodes = get_linked_nodes(
                linked_edge_type,
                batch,
                dst_nodes,
                start=start,
                end=end,
            )
            rel = linked_edge_type[1]
            rel = rel[4:] if rel.startswith('rev_') else f'rev_{rel}'
            rev_linked_edge_type = (linked_edge_type[2], rel,
                                    linked_edge_type[0])
            rev_nodes: Tensor = nodes.new_empty((0, ))
            if len(batch[rev_linked_edge_type]) > 0:
                start = batch[rev_linked_edge_type].num_sampled_edges[0]
                end = start + batch[rev_linked_edge_type].num_sampled_edges[1]
                # Get the source nodes of the reverse edge type
                rev_nodes = get_linked_nodes(
                    rev_linked_edge_type,
                    batch,
                    dst_nodes,
                    start=start,
                    end=end,
                    dst=False,
                )
            nodes = torch.cat([nodes, rev_nodes])
            num_nodes = nodes.unique().numel()
            if num_nodes == 0:
                continue
            if dst_table in unique_counts:
                unique_counts[dst_table][table] = num_nodes
            else:
                unique_counts[dst_table] = {table: num_nodes}
    return unique_counts


def set_link_pred_rfm(
    rfm: LinkPredRFM,
    seed_id: int,
    pred: Tensor,
    rhs_idx: Tensor,
    lhs_idx: Tensor,
    y_idx: Tensor | None,
    y_batch: Tensor | None,
) -> None:
    r"""Set the RFM for link prediction task."""
    pred_idx = pred[seed_id].cpu().unique()
    hist_idx = rhs_idx[lhs_idx == seed_id].unique()
    rfm.num_hist_items = hist_idx.unique().numel()
    rfm.num_pred_hist_items = torch.isin(pred_idx, hist_idx).sum().item()
    if y_idx is None:
        return
    y = y_idx[y_batch == seed_id]
    rfm.num_true_items = y.unique().numel()
    rfm.num_true_hist_items = torch.isin(y, hist_idx).sum().item()


def get_link_pred_ground_truth(
    module: torch.nn.Module,
    batch: HeteroData,
    stage: TrainingStage,
) -> tuple[Tensor, Tensor]:
    if module.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
        y_idx, y_batch, _ = module.get_ground_truth(batch)
    else:
        assert module.task_type == TaskType.STATIC_LINK_PREDICTION
        y_idx, y_batch, _ = module.get_ground_truth(batch, stage)
    return y_idx, y_batch


def get_target_rhs_id(
    target_to_entity_paths: list[list[EdgeType]],
    batch: HeteroData,
) -> tuple[Tensor, Tensor]:
    r"""Returns the batch-local second-hop RHS nodes and their seeds."""
    if (len(target_to_entity_paths) != 2
            or any(len(path) != 1 for path in target_to_entity_paths)):
        raise ValueError(f"Expected one entity and one RHS relation for the "
                         f"link prediction target, got "
                         f"{target_to_entity_paths}")
    fact_name, rhs_fkey, rhs_table_name = _strip_edge_type(
        target_to_entity_paths[1][0])
    edge_type = (rhs_table_name, f'rev_{rhs_fkey}', fact_name)
    if edge_type not in batch.edge_types:
        raise ValueError(f"Batch has no {edge_type} relation")
    store = batch[edge_type]
    edge_index = _directed_edge_index(store)
    num_sampled_edges = store.num_sampled_edges
    if len(num_sampled_edges) >= 2:
        start = num_sampled_edges[0]
        rhs_idx = edge_index[0, start:start + num_sampled_edges[1]]
    elif edge_index.size(1) == 0:  # One-hop plan.
        rhs_idx = torch.empty(0, dtype=torch.long)
    else:  # Counts not split by hop.
        rhs_idx = _rhs_linked_to_seeds(target_to_entity_paths, batch,
                                       edge_index)
    lhs_idx = batch[rhs_table_name].batch[rhs_idx].cpu()
    return rhs_idx, lhs_idx


def _rhs_linked_to_seeds(
    target_to_entity_paths: list[list[EdgeType]],
    batch: HeteroData,
    rhs_edge_index: Tensor,
) -> Tensor:
    # The entity path may point entity -> target (joined on the target PK):
    fact_name = _strip_edge_type(target_to_entity_paths[1][0])[0]
    src, key, dst = _strip_edge_type(target_to_entity_paths[0][0])
    entity_name = dst if src == fact_name else src
    candidates = [(fact_name, key, entity_name)]
    if src != fact_name:
        candidates.append((fact_name, f'rev_{key}', entity_name))
    edge_types = [e for e in candidates if e in batch.edge_types]
    if not edge_types:
        raise ValueError(f"Batch has none of the {candidates} relations")
    batch_size = batch[entity_name].batch_size
    fact_idx = torch.cat([
        edge_index[0, edge_index[1] < batch_size]
        for edge_index in (_directed_edge_index(batch[edge_type])
                           for edge_type in edge_types)
    ])
    return rhs_edge_index[0, torch.isin(rhs_edge_index[1], fact_idx)]


def _strip_edge_type(edge_type: EdgeType) -> EdgeType:
    src, rel, dst = edge_type
    src = src.split('.', maxsplit=1)[-1]
    dst = dst.split('.', maxsplit=1)[-1]
    return src, rel, dst


def _directed_edge_index(store: Any) -> Tensor:
    if hasattr(store, '_orig_edge_index'):
        return store._orig_edge_index.cpu()
    return store.edge_index.cpu()


def get_linked_nodes(
    edge_type: EdgeType,
    batch: HeteroData,
    seed_id: int | Tensor,
    start: int,
    end: int,
    dst: bool = True,
) -> Tensor:
    r"""Get the linked nodes from the given edge type.

    Args:
        edge_type (EdgeType): The edge type to get the linked nodes from.
        batch (HeteroData): The batch to get the linked nodes from.
        seed_id (int | Tensor): The seed node id to get the linked nodes from.
        start (int): The start index of the edge index.
        end (int): The end index of the edge index.
        dst (bool): Whether to get the destination nodes. If getting
            destination nodes, the seed node should be the source node
            of the edge type.
    """
    edge_index = batch[edge_type].edge_index[:, start:end]
    if isinstance(seed_id, int):
        seed_id = torch.tensor([seed_id],
                               dtype=edge_index.dtype).to(edge_index.device)
    if dst:
        nodes = edge_index[1, torch.isin(edge_index[0], seed_id)]
    else:
        nodes = edge_index[0, torch.isin(edge_index[1], seed_id)]
    return nodes


def get_edge_types(
    table_name: str,
    batch: HeteroData,
    reverse: bool = False,
) -> list[EdgeType]:
    r"""Get all edge types with the given source or destination table."""
    if not reverse:
        return [
            edge_type for edge_type in batch.edge_types
            if edge_type[0] == table_name
        ]
    return [
        edge_type for edge_type in batch.edge_types
        if edge_type[2] == table_name
    ]


def get_normalized_grad_dict(
    batch: HeteroData,
    seed_table: str,
    grad_dict: dict[str, dict[str, Tensor]],
) -> dict[str, dict[str, Tensor]]:
    r"""Normalize gradient scores to interval `[0, 1]`.

    Args:
        batch (HeteroData): The batch data.
        seed_table (str): The seed table name.
        grad_dict (dict[str, dict[str, Tensor]]): The gradient score
            dictionary where the first key is the table name and the
            second key is the column name.

    Returns:
        dict[str, dict[str, Tensor]]: The normalized gradient score
            dictionary where the first key is the table name and the
            second key is the column name.
    """
    batch_size: int = batch[seed_table].batch_size
    max_grad: Tensor = torch.zeros(
        batch_size,
        device=batch[seed_table].batch.device,
    )

    for table_name, table_grad in grad_dict.items():
        batch_vec = batch[table_name].batch
        for grad in table_grad.values():
            max_grad = torch.maximum(
                max_grad,
                scatter(grad, batch_vec, dim_size=batch_size, reduce='max'),
            )

    score_dict: dict[str, dict[str, Tensor]] = defaultdict(dict)
    for table_name, table_grad in grad_dict.items():
        batch_vec = batch[table_name].batch
        for col_name, grad in table_grad.items():
            score = grad / max_grad[batch_vec]
            score = score.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            score_dict[table_name][col_name] = score

    return score_dict


def to_python_scalar(obj: Any) -> Any:
    r"""Convert a numpy or pandas object to Python scalar values."""
    # Convert NaType to None as it's not serializable
    if pd.isna(obj):
        return None
    return obj.item() if isinstance(obj, np.generic) else obj
