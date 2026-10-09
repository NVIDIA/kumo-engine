# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.types import TaskType
from kumoapi.explain import (
    Feature,
    LinkPredRFM,
    NodePredRFM,
    Subgraph,
    SubgraphFeatureStore,
    SubgraphStore,
)
from kumoapi.typing import Stype, TimeUnit

from rest.utils.explain import (
    group_merge_topk_feats,
    link_pred_rfm_avg,
    node_pred_rfm_avg,
)


def test_node_pred_rfm_avg():
    rfms = [
        NodePredRFM(
            entity_table_name='table1', count={
                'a': 1,
                'b': 2
            }, avg_rel_time={
                'a': (5, TimeUnit.DAYS),
                'b': (6, TimeUnit.DAYS)
            }, most_recent_rel_time={
                'a': (1, TimeUnit.DAYS),
                'b': (2, TimeUnit.DAYS)
            }, unique_counts={
                'a': {
                    'c': 1,
                    'd': 2
                },
                'b': {
                    'e': 3,
                }
            }),
        NodePredRFM(
            entity_table_name='table1', count={
                'a': 3,
                'b': 4
            }, avg_rel_time={
                'a': (3, TimeUnit.DAYS),
                'b': (4, TimeUnit.DAYS)
            }, most_recent_rel_time={
                'a': (0, TimeUnit.DAYS),
                'b': (1, TimeUnit.DAYS)
            }, unique_counts={
                'a': {
                    'c': 5,
                    'd': 6
                },
                'b': {
                    'e': 5,
                    'f': 6
                }
            }),
    ]
    avg_rfm = node_pred_rfm_avg(rfms)
    assert avg_rfm.count == {'a': 2.0, 'b': 3.0}
    assert avg_rfm.avg_rel_time == {
        'a': (4.0, TimeUnit.DAYS),
        'b': (5.0, TimeUnit.DAYS)
    }
    assert avg_rfm.most_recent_rel_time == {
        'a': (0.5, TimeUnit.DAYS),
        'b': (1.5, TimeUnit.DAYS)
    }
    assert avg_rfm.unique_counts == {
        'a': {
            'c': 3.0,
            'd': 4.0
        },
        'b': {
            'e': 4.0,
            'f': 3.0
        }
    }


def test_link_pred_rfm_avg():
    rfms = [
        LinkPredRFM(
            entity_table_name='table1', count={
                'a': 1,
                'b': 2
            }, avg_rel_time={
                'a': (5, TimeUnit.DAYS),
                'b': (6, TimeUnit.DAYS)
            }, most_recent_rel_time={
                'a': (1, TimeUnit.DAYS),
                'b': (2, TimeUnit.DAYS)
            }, num_hist_items=1, num_pred_hist_items=2, num_true_items=3,
            num_true_hist_items=4),
        LinkPredRFM(
            entity_table_name='table1', count={
                'a': 3,
                'b': 4
            }, avg_rel_time={
                'a': (3, TimeUnit.DAYS),
                'b': (4, TimeUnit.DAYS)
            }, most_recent_rel_time={
                'a': (0, TimeUnit.DAYS),
                'b': (1, TimeUnit.DAYS)
            }, num_hist_items=5, num_pred_hist_items=6, num_true_items=7,
            num_true_hist_items=8),
    ]
    avg_rfm = link_pred_rfm_avg(rfms)
    assert avg_rfm.count == {'a': 2.0, 'b': 3.0}
    assert avg_rfm.avg_rel_time == {
        'a': (4.0, TimeUnit.DAYS),
        'b': (5.0, TimeUnit.DAYS)
    }
    assert avg_rfm.most_recent_rel_time == {
        'a': (0.5, TimeUnit.DAYS),
        'b': (1.5, TimeUnit.DAYS)
    }
    assert avg_rfm.num_hist_items == 3.0
    assert avg_rfm.num_pred_hist_items == 4.0
    assert avg_rfm.num_true_items == 5.0
    assert avg_rfm.num_true_hist_items == 6.0


def test_group_merge_topk_feats():
    features = {
        0: {
            'feature1':
            Feature(table_name='table1', name='feature1', value=1,
                    stype=Stype.numerical, score=0.5),
            'feature2':
            Feature(table_name='table1', name='feature2', value=2,
                    stype=Stype.categorical, score=0.4),
        },
        1: {
            'feature1':
            Feature(table_name='table1', name='feature1', value=3,
                    stype=Stype.numerical, score=0.5),
            'feature2':
            Feature(table_name='table1', name='feature2', value=5,
                    stype=Stype.categorical, score=0.4),
        },
        2: {
            'feature1':
            Feature(table_name='table2', name='feature1', value=3,
                    stype=Stype.numerical, score=0.3),
            'feature2':
            Feature(table_name='table2', name='feature2', value=5,
                    stype=Stype.categorical, score=0.2),
        },
        3: {
            'feature2':
            Feature(table_name='table1', name='feature2', value=2,
                    stype=Stype.categorical, score=0.3),
        },
    }
    subgraph = Subgraph(
        subgraph_store=SubgraphStore(
            root_id=0,
            task_type=TaskType.BINARY_CLASSIFICATION,
        ),
        feature_store=SubgraphFeatureStore(features=features, ),
    )
    feats = group_merge_topk_feats(subgraph, topk=3)
    assert len(feats) == 3
    assert feats[0].name == 'feature1'
    assert feats[0].table_name == 'table1'
    # (1 + 3) / 2 = 2.0
    assert feats[0].value == 2.0

    assert feats[1].name == 'feature2'
    assert feats[1].table_name == 'table1'
    # Most frequent value
    assert feats[1].value == 2

    assert feats[2].name == 'feature1'
    assert feats[2].table_name == 'table2'
    assert feats[2].value == 3
