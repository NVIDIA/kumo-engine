# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Type, TypeVar

from kumo_core.training.explain_aggregation import (  # noqa: F401
    AGGR_FEATURE_FILE_NAME,
    GLOBAL_AGGR_FEATURE_DIR,
    MAX_NUM_POST_XAI_ENTITIES,
    aggr_list_features,
)
from kumoapi.explain import Feature, LinkPredRFM, NodePredRFM, Subgraph
from kumoapi.explain.rfm import BaseRFM
from kumoapi.typing import Stype, TimeUnit

T = TypeVar('T', bound=BaseRFM)

# Name of the directory to save the category-level RFMs
CATEGORY_RFM_DIR = 'category_rfm'


def _avg_rfm(cls: Type[T], rfms: List[T]) -> T:
    r"""Averages the base RFM in the list.

    Args:
        cls: The type of the RFM to average.
        rfms: The list of RFMs to average.
    Returns:
        The averaged base RFM.
    """
    assert len(rfms) > 0, "RFMs must contain at least one RFM"
    count: Dict[str, float] = {}
    for rfm in rfms:
        for key in rfm.count:
            count[key] = count.get(key, 0) + rfm.count[key]
    count = {key: count[key] / len(rfms) for key in count}

    avg_rel_time: Dict[str, Tuple[float, TimeUnit]] = {}
    for rfm in rfms:
        for key in rfm.avg_rel_time:
            if key not in avg_rel_time:
                avg_rel_time[key] = rfm.avg_rel_time[key]
            else:
                avg_rel_time[key] = (
                    avg_rel_time[key][0] + rfm.avg_rel_time[key][0],
                    rfm.avg_rel_time[key][1],
                )
    avg_rel_time = {
        key: (avg_rel_time[key][0] / len(rfms), avg_rel_time[key][1])
        for key in avg_rel_time
    }

    most_recent_rel_time: Dict[str, Tuple[float, TimeUnit]] = {}
    for rfm in rfms:
        for key in rfm.most_recent_rel_time:
            time, time_unit = rfm.most_recent_rel_time[key]
            if key not in most_recent_rel_time:
                most_recent_rel_time[key] = (time, time_unit)
            else:
                most_recent_rel_time[key] = (
                    most_recent_rel_time[key][0] +
                    rfm.most_recent_rel_time[key][0],
                    rfm.most_recent_rel_time[key][1],
                )
    most_recent_rel_time = {
        key:
        (most_recent_rel_time[key][0] / len(rfms), most_recent_rel_time[key][1])
        for key in most_recent_rel_time
    }

    return cls(
        entity_table_name=rfms[0].entity_table_name,
        count=count,
        avg_rel_time=avg_rel_time,
        most_recent_rel_time=most_recent_rel_time,
    )


def node_pred_rfm_avg(rfms: List[NodePredRFM]) -> NodePredRFM:
    r"""Averages the node prediction RFMs in the list.
    Args:
        rfms: The list of node prediction RFMs to average.
    Returns:
        The averaged node prediction RFM.
    """
    base_rfm = _avg_rfm(NodePredRFM, rfms)

    # Here we average the unique counts for the second-hop tables
    unique_counts: Dict[str, Dict[str, float]] = {}
    for rfm in rfms:
        for key in rfm.unique_counts:
            if key not in unique_counts:
                unique_counts[key] = rfm.unique_counts[key]
            else:
                for k in rfm.unique_counts[key]:
                    if k not in unique_counts[key]:
                        unique_counts[key][k] = rfm.unique_counts[key][k]
                    else:
                        unique_counts[key][k] = unique_counts[key][
                            k] + rfm.unique_counts[key][k]

    unique_counts = {
        key: {
            k: unique_counts[key][k] / len(rfms)
            for k in unique_counts[key]
        }
        for key in unique_counts
    }

    return NodePredRFM(
        entity_table_name=base_rfm.entity_table_name,
        count=base_rfm.count,
        avg_rel_time=base_rfm.avg_rel_time,
        most_recent_rel_time=base_rfm.most_recent_rel_time,
        unique_counts=unique_counts,
    )


def link_pred_rfm_avg(rfms: List[LinkPredRFM]) -> LinkPredRFM:
    r"""Averages the link prediction RFMs in the list.
    Args:
        rfms: The list of link prediction RFMs to average.
    Returns:
        The averaged link prediction RFM.
    """
    base_rfm = _avg_rfm(LinkPredRFM, rfms)
    num_hist_items: float = sum(rfm.num_hist_items for rfm in rfms) / len(rfms)
    num_pred_hist_items: float = sum(rfm.num_pred_hist_items
                                     for rfm in rfms) / len(rfms)

    num_true_items: Optional[float] = None
    all_none = all(rfm.num_true_items is None for rfm in rfms)
    if not all_none:
        num_true_items = sum(rfm.num_true_items for rfm in rfms
                             if rfm.num_true_items is not None)
        num_true_items = num_true_items / len(rfms)

    num_true_hist_items: Optional[float] = None
    all_none = all(rfm.num_true_hist_items is None for rfm in rfms)
    if not all_none:
        num_true_hist_items = sum(rfm.num_true_hist_items for rfm in rfms
                                  if rfm.num_true_hist_items is not None)
        num_true_hist_items = num_true_hist_items / len(rfms)

    return LinkPredRFM(
        entity_table_name=base_rfm.entity_table_name,
        count=base_rfm.count,
        avg_rel_time=base_rfm.avg_rel_time,
        most_recent_rel_time=base_rfm.most_recent_rel_time,
        num_hist_items=num_hist_items,
        num_pred_hist_items=num_pred_hist_items,
        num_true_items=num_true_items,
        num_true_hist_items=num_true_hist_items,
    )


def group_merge_topk_feats(subgraph: Subgraph, topk: int) -> List[Feature]:
    r"""Groups, aggregates, and selects the top-k most important features.

    This function processes features within a subgraph to identify the `topk`
    most influential ones based on their gradient scores. Features without a
    score or with a null value are excluded.

    The process involves:
    1. Extracting all features from the subgraph's feature store.
    2. Filtering out features with null scores or values.
    3. Sorting the features in descending order based on their scores.
    4. Identifying unique features by their (table_name, name) tuple.
    5. Selecting the `topk` unique features based on the highest score of
       their first appearance in the sorted list.
    6. For each selected unique feature, collecting all its instances that
       fall within the initial `topk` positions of the sorted list.
    7. If multiple instances of a unique feature are collected, they are
       aggregated into a single feature:
        - Numerical features (`Stype.numerical`) are aggregated using 'mean'.
        - Categorical features are aggregated using 'most_frequent'.
        - The aggregation uses the `aggr_list_features` function.
    8. Returning the final list containing up to `topk` features (original or
       aggregated).

    Args:
        subgraph: The subgraph containing the features to process.
        topk: The maximum number of features to return.

    Returns:
        A list of the top `k` features, potentially aggregated if multiple
        instances of the same feature appeared within the initial `topk`
        ranking based on score.
    """
    features: List[Feature] = []
    for _, cols in subgraph.feature_store.features.items():
        for col in cols.values():
            features.append(col)
    # Filter out features without scores or has null value
    features = [
        feat for feat in features
        if feat.score is not None and feat.value is not None
    ]
    # Sort features by score in descending order
    features.sort(key=lambda feat: feat.score, reverse=True)

    idx_map: dict[tuple[str, str], list[int]] = {}
    for i, feat in enumerate(features):
        name = (feat.table_name, feat.name)
        if name not in idx_map:
            idx_map[name] = [i]
        else:
            idx_map[name].append(i)

    # Sort the index map by the first appeared position
    sorted_items = sorted(idx_map.items(), key=lambda item: item[1][0])
    sorted_items = sorted_items[:topk]
    final_feats: list[Feature] = []
    for item in sorted_items:
        _, indices = item
        valid_indices = [indices[0]]
        if len(indices) > 1:
            for index in indices[1:]:
                if index < topk:
                    valid_indices.append(index)
        feats = [features[idx] for idx in valid_indices]
        if len(feats) == 1:
            final_feats.append(feats[0])
        else:
            op = ('mean'
                  if feats[0].stype == Stype.numerical else 'most_frequent')
            final_feats.append(aggr_list_features(feats, op))
    return final_feats
