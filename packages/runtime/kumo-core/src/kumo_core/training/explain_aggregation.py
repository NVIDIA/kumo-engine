# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-RFM subgraph feature aggregation, shared with hosted XAI."""

import logging
import os.path as osp
from copy import deepcopy
from typing import Any, Callable, Literal

from kumo_core.io.filesystem import exists
from kumo_core.io.json import load_json, save_json
from kumoapi.explain import (
    EntityMappings,
    Feature,
    Subgraph,
    SubgraphFeatureStore,
)
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype

log = logging.getLogger(__name__)
GLOBAL_AGGR_FEATURE_DIR = 'global_aggr_feature'
AGGR_FEATURE_FILE_NAME = 'feature'
MAX_NUM_POST_XAI_ENTITIES = 50


def aggr_list_features(
    features: list[Feature],
    op: Literal[
        'sum',
        'mean',
        'max',
        'min',
        'most_frequent',
        'least_frequent',
    ],
) -> Feature:
    r"""Aggregate a list of features by the given operation.
    If the aggregation operation is numerical and all the values are None,
    return an empty feature.

    Args:
        features (list[Feature]): The list of features to aggregate.
        op (str): The operation to aggregate the features.

    Returns:
        The aggregated feature.
    """
    if len(features) == 0:
        return Feature()
    numerical_ops = {'sum', 'mean', 'max', 'min'}
    count: dict[str, int] = {}
    aggr_feat = deepcopy(features[0])

    # Number of valid features to be aggregated
    # Categorical aggregations count None as a frequency bucket.
    num_valid_feats: int = int(aggr_feat.value is not None
                               or op not in numerical_ops)
    num_valid_scores: int = 1 if aggr_feat.score is not None else 0
    orig_val_mapping: dict[str, Any] = {}
    if op in {'most_frequent', 'least_frequent'}:
        key = str(aggr_feat.value)
        count[key] = 1
        orig_val_mapping[key] = aggr_feat.value
    for feat in features[1:]:
        if op in numerical_ops:
            # Skip numerical aggergations if the feature value is None
            if feat.value is None:
                continue
            # Initialize the aggregated feature value if it is None
            # until the first valid feature is found
            if aggr_feat.value is None:
                aggr_feat.value = feat.value
                num_valid_feats += 1
                if aggr_feat.score is None and feat.score is not None:
                    aggr_feat.score = feat.score
                    num_valid_scores += 1
                elif feat.score is not None:
                    aggr_feat.score += feat.score
                    num_valid_scores += 1
                continue
        if op in {'sum', 'mean'}:
            aggr_feat.value += feat.value
        elif op == 'max':
            aggr_feat.value = max(aggr_feat.value, feat.value)
        elif op == 'min':
            aggr_feat.value = min(aggr_feat.value, feat.value)
        elif op in {'most_frequent', 'least_frequent'}:
            key = str(feat.value)
            if key not in orig_val_mapping:
                orig_val_mapping[key] = feat.value
            count[key] = count.get(key, 0) + 1
        num_valid_feats += 1
        if feat.score is not None:
            if aggr_feat.score is None:
                aggr_feat.score = feat.score
                num_valid_scores += 1
            else:
                aggr_feat.score += feat.score
                num_valid_scores += 1
    # If the aggregation operation is numerical and all
    # the values are None, return an empty feature
    if op in numerical_ops and aggr_feat.value is None:
        return Feature()

    assert num_valid_feats > 0
    if op == 'mean':
        aggr_feat.value /= num_valid_feats
    elif op == 'most_frequent' and len(count) > 0:
        aggr_feat.value = orig_val_mapping[max(count, key=lambda x:
                                               (count[x], x))]
    elif op == 'least_frequent' and len(count) > 0:
        aggr_feat.value = orig_val_mapping[min(count, key=lambda x:
                                               (count[x], x))]
    if aggr_feat.score is not None:
        aggr_feat.score /= num_valid_scores
    return aggr_feat


def aggregate_subgraph_features(
    subgraph_collector_dir: str,
    stage: TrainingStage,
    *,
    exists: Callable = exists,
    load_json: Callable = load_json,
    save_json: Callable = save_json,
) -> None:
    """Aggregate the hosted random/test or first/prediction feature cohort.

    The directory contains strategy_mappings.json and subgraph_json/<ordinal>.
    Hosted callers retain their filesystem/serialization adapters; local callers
    use core I/O. Missing output (including unsupported tasks) is a no-op.
    This does not generate RFMs or aggregate RFM artifacts.
    """
    mapping_path = osp.join(subgraph_collector_dir, 'strategy_mappings.json')
    if not exists(mapping_path):
        return
    mappings = load_json(mapping_path, EntityMappings).mappings
    category = 'random' if stage != TrainingStage.PRED else 'first'
    # We need to aggregate random / first features to be used for global topk
    if category not in mappings:
        log.error(f"No {category} category found in mappings")
        return None
    random_entities = mappings[category][:MAX_NUM_POST_XAI_ENTITIES]
    if len(random_entities) == 0:
        log.error(f"No {category} entities found in mappings")
        return None
    # Get all columns that need to be aggregated for global features
    # Keys are (table_name, column_name)
    # Values are operation either 'mean' for numerical columns or
    # 'most_frequent' for categorical columns
    feats: dict[tuple[str, str], list[Feature]] = {}
    for entity in random_entities:
        subgraph_json_path = osp.join(subgraph_collector_dir, "subgraph_json",
                                      f"{entity}.json")
        if not exists(subgraph_json_path):
            log.error(f"Subgraph json path {subgraph_json_path} "
                      f"does not exist")
            continue
        subgraph = load_json(subgraph_json_path, Subgraph)
        feature_store: SubgraphFeatureStore | None = subgraph.feature_store
        if feature_store is None:
            log.error(f"No feature store found for {subgraph_json_path}")
            continue
        for _, cols_dict in feature_store.features.items():
            for col_name, col in cols_dict.items():
                key = (col.table_name, col_name)
                if key not in feats:
                    feats[key] = [col]
                else:
                    feats[key].append(col)

    # Aggregate features and save the results to:
    # <global_aggr_feat_dir>/<category>/<table_name>/<col_name>/
    # <op>/<AGGR_FEATURE_FILE_NAME>.json
    global_aggr_feat_dir = osp.join(subgraph_collector_dir,
                                    GLOBAL_AGGR_FEATURE_DIR)
    category_aggr_feat_dir = osp.join(global_aggr_feat_dir, category)
    for (table_name, col_name), features in feats.items():
        if len(features) == 0:
            log.error(f"No features found for {table_name}.{col_name}")
            continue
        stype = features[0].stype
        # Use 'mean' for numerical columns and 'most_frequent' for others
        op = 'mean' if stype == Stype.numerical else 'most_frequent'
        aggr_feat = aggr_list_features(features, op)
        table_dir = osp.join(category_aggr_feat_dir, table_name)
        col_dir = osp.join(table_dir, col_name)
        op_dir = osp.join(col_dir, op)
        save_json(aggr_feat, osp.join(op_dir, f"{AGGR_FEATURE_FILE_NAME}.json"))
