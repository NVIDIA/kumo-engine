# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from kumo_core.io.json import load_json, save_json
from kumo_core.training.explain_aggregation import (
    aggr_list_features,
    aggregate_subgraph_features,
)
from kumoapi.explain import (
    EntityMappings,
    Feature,
    Subgraph,
    SubgraphFeatureStore,
)
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype


def test_aggr_list_features():
    features = [
        Feature(table_name='table1', name='feature1', value=1),
        Feature(table_name='table1', name='feature1', value=1),
        Feature(table_name='table1', name='feature1', value=2),
        Feature(table_name='table1', name='feature1', value=2),
        Feature(table_name='table1', name='feature1', value=3),
        Feature(table_name='table1', name='feature1', value=None),
    ]
    assert aggr_list_features(features, 'sum').value == 9
    assert aggr_list_features(features, 'mean').value == 1.8
    assert aggr_list_features(features, 'max').value == 3
    assert aggr_list_features(features, 'min').value == 1
    assert aggr_list_features(features, 'most_frequent').value == 2
    assert aggr_list_features(features, 'least_frequent').value == 3

    features = []
    assert aggr_list_features(features, 'sum').name is None

    features = [
        Feature(table_name='table1', name='feature1', value='a'),
    ]
    assert aggr_list_features(features, 'most_frequent').value == 'a'

    features = [Feature(table_name='table1', name='feature1', value=None)]
    assert aggr_list_features(features, 'most_frequent').value is None

    # Most frequent value over all None values returns None values
    features = [
        Feature(table_name='table1', name='feature1', value=None),
        Feature(table_name='table1', name='feature1', value=None),
        Feature(table_name='table1', name='feature1', value=1),
    ]
    assert aggr_list_features(features, 'most_frequent').value is None

    # Sum over all None values returns empty feature
    features = [
        Feature(table_name='table1', name='feature1', value=None),
        Feature(table_name='table1', name='feature1', value=None),
    ]
    feat = aggr_list_features(features, 'sum')
    assert feat.table_name is None
    assert feat.stype is None
    assert feat.score is None
    assert feat.name is None
    assert feat.value is None


@pytest.mark.parametrize('stage,category', [(TrainingStage.TEST, 'random'),
                                            (TrainingStage.PRED, 'first')])
def test_aggregate_selected_subgraphs_without_rfm(tmp_path, stage, category):
    # Artifact ordinals, deliberately distinct from original entity IDs.
    save_json(
        EntityMappings(mappings={category: ['0', '1', 'missing']},
                       orig_mappings={category: [1007, 'user-9', 33]}),
        str(tmp_path / 'strategy_mappings.json'))
    for ordinal, value in enumerate([2., 6.]):
        subgraph = Subgraph(feature_store=SubgraphFeatureStore(
            features={
                7: {
                    'amount':
                    Feature(table_name='users', name='amount', value=value,
                            stype=Stype.numerical, score=0.2 + ordinal * 0.4),
                    'group':
                    Feature(table_name='users', name='group', value='A',
                            stype=Stype.categorical)
                },
            }))
        save_json(subgraph, str(tmp_path / 'subgraph_json' / f'{ordinal}.json'))
    # Unselected files must not contribute.
    save_json(Subgraph(), str(tmp_path / 'subgraph_json' / '1007.json'))
    aggregate_subgraph_features(str(tmp_path), stage)
    root = tmp_path / 'global_aggr_feature' / category / 'users'
    amount = load_json(str(root / 'amount/mean/feature.json'), Feature)
    group = load_json(str(root / 'group/most_frequent/feature.json'), Feature)
    assert amount.value == 4.
    assert amount.score == pytest.approx(0.4)
    assert group.value == 'A'
    assert not (tmp_path / 'category_rfm').exists()
    restored = load_json(str(tmp_path / 'strategy_mappings.json'),
                         EntityMappings)
    assert restored.orig_mappings[category] == [1007, 'user-9', 33]


def test_aggregation_keeps_hosted_fifty_entity_bound(tmp_path):
    save_json(EntityMappings(mappings={'random': [str(i) for i in range(51)]}),
              str(tmp_path / 'strategy_mappings.json'))
    read_paths = []

    def read(path, cls):
        read_paths.append(path)
        if cls is EntityMappings:
            return load_json(path, cls)
        return Subgraph()

    aggregate_subgraph_features(str(tmp_path), TrainingStage.TEST,
                                exists=lambda path: True, load_json=read)
    assert len(read_paths) == 51  # Mapping plus at most 50 entities.
    assert read_paths[-1].endswith('/subgraph_json/49.json')


def test_missing_subgraph_output_is_noop(tmp_path):
    aggregate_subgraph_features(str(tmp_path), TrainingStage.TEST)
    assert list(tmp_path.iterdir()) == []


def test_mapping_accepts_legacy_null_original_ids(tmp_path):
    path = tmp_path / 'strategy_mappings.json'
    path.write_text(
        json.dumps({
            'mappings': {
                'random': ['0']
            },
            'orig_mappings': None
        }))
    assert load_json(str(path), EntityMappings).orig_mappings is None


def test_mapping_defaults_preserve_hosted_null_original_ids(tmp_path):
    mapping = EntityMappings(mappings={"random": ["0"]})
    assert mapping.orig_mappings is None
    path = tmp_path / "strategy_mappings.json"
    save_json(mapping, str(path))
    assert json.loads(path.read_text())["orig_mappings"] is None
    assert load_json(str(path), EntityMappings).orig_mappings is None
