# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from types import SimpleNamespace

import pytest
from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
from kumo_core.training.config import LinkPredModuleType
from kumo_core.types import TaskType
from torch_geometric.sampler import NumNeighbors

DD_EDGE_RELATIONS = [
    {
        "name": "impressions_CONSUMER_ID_to_users_CONSUMER_ID",
        "src_type": "impressions",
        "dst_type": "users",
    },
    {
        "name": "impressions_CONSUMER_ID_to_users_CONSUMER_ID_reverse",
        "src_type": "users",
        "dst_type": "impressions",
    },
    {
        "name": "impressions_TEMPLATE_ID_to_templates_TEMPLATE_ID",
        "src_type": "impressions",
        "dst_type": "templates",
    },
    {
        "name": "impressions_TEMPLATE_ID_to_templates_TEMPLATE_ID_reverse",
        "src_type": "templates",
        "dst_type": "impressions",
    },
    {
        "name": "clicks_CONSUMER_ID_to_users_CONSUMER_ID",
        "src_type": "clicks",
        "dst_type": "users",
    },
    {
        "name": "clicks_CONSUMER_ID_to_users_CONSUMER_ID_reverse",
        "src_type": "users",
        "dst_type": "clicks",
    },
    {
        "name": "clicks_TEMPLATE_ID_to_templates_TEMPLATE_ID",
        "src_type": "clicks",
        "dst_type": "templates",
    },
    {
        "name": "clicks_TEMPLATE_ID_to_templates_TEMPLATE_ID_reverse",
        "src_type": "templates",
        "dst_type": "clicks",
    },
    {
        "name": "orders_CONSUMER_ID_to_users_CONSUMER_ID",
        "src_type": "orders",
        "dst_type": "users",
    },
    {
        "name": "orders_CONSUMER_ID_to_users_CONSUMER_ID_reverse",
        "src_type": "users",
        "dst_type": "orders",
    },
]


def _manifest(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "index_metadata.json").write_text(
        json.dumps({"edge_relations": DD_EDGE_RELATIONS}))
    return SimpleNamespace(
        graph_runtime_metadata={"graph_index_path": str(index_path)})


def _trainer(num_neighbors):
    return SimpleNamespace(
        sampler_kwargs=SimpleNamespace(
            num_neighbors=num_neighbors,
            temporal_strategy="last",
        ),
        task=SimpleNamespace(task_type=SimpleNamespace(is_link_pred=False)),
        module_kwargs=None,
    )


def _build_plan(num_neighbors, tmp_path):
    return DiskgraphNestedTrainer._build_ttng_sampling_plan(
        _trainer(num_neighbors),
        _manifest(tmp_path),
    )


def test_ttng_sampling_plan_resolves_kumo_edge_types_for_sparse_fanouts(
        tmp_path):
    num_neighbors = NumNeighbors(
        {
            ("users", "rev_CONSUMER_ID", "impressions"): [1, 0, 0],
            ("templates", "rev_TEMPLATE_ID", "impressions"): [1, 0, 0],
            ("clicks", "CONSUMER_ID", "users"): [0, 32, 0],
            ("orders", "CONSUMER_ID", "users"): [0, 32, 0],
            ("templates", "rev_TEMPLATE_ID", "clicks"): [0, 0, 1],
        },
        [0, 0, 0],
    )

    plan = _build_plan(num_neighbors, tmp_path)

    assert plan == {
        "layers": [
            {
                "fanouts": {
                    "impressions_CONSUMER_ID_to_users_CONSUMER_ID": {
                        "fanout": 1,
                        "mode": "temporal_most_recent",
                        "absent_key_policy": "uniform_fallback",
                    },
                    "impressions_TEMPLATE_ID_to_templates_TEMPLATE_ID": {
                        "fanout": 1,
                        "mode": "temporal_most_recent",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
                "default_fanout": None,
            },
            {
                "fanouts": {
                    "clicks_CONSUMER_ID_to_users_CONSUMER_ID_reverse": {
                        "fanout": 32,
                        "mode": "temporal_most_recent",
                        "absent_key_policy": "uniform_fallback",
                    },
                    "orders_CONSUMER_ID_to_users_CONSUMER_ID_reverse": {
                        "fanout": 32,
                        "mode": "temporal_most_recent",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
                "default_fanout": None,
            },
            {
                "fanouts": {
                    "clicks_TEMPLATE_ID_to_templates_TEMPLATE_ID": {
                        "fanout": 1,
                        "mode": "temporal_most_recent",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
                "default_fanout": None,
            },
        ],
    }


def test_ttng_sampling_plan_keeps_default_fanout_behavior(tmp_path):
    plan = _build_plan(NumNeighbors([5, 3]), tmp_path)

    assert plan["layers"] == [
        {
            "fanouts": {},
            "default_fanout": {
                "fanout": 5,
                "mode": "temporal_most_recent",
                "absent_key_policy": "uniform_fallback",
            },
        },
        {
            "fanouts": {},
            "default_fanout": {
                "fanout": 3,
                "mode": "temporal_most_recent",
                "absent_key_policy": "uniform_fallback",
            },
        },
    ]


def test_ttng_sampling_plan_rejects_empty_per_edge_plan(tmp_path):
    num_neighbors = NumNeighbors(
        {("missing", "EDGE", "users"): [1]},
        [0],
    )

    with pytest.raises(RuntimeError, match="did not resolve any"):
        _build_plan(num_neighbors, tmp_path)


def _rel(src, col, dst, pk):
    name = f"{src}_{col}_to_{dst}_{pk}"
    return [
        {
            "name": name,
            "src_type": src,
            "dst_type": dst
        },
        {
            "name": f"{name}_reverse",
            "src_type": dst,
            "dst_type": src
        },
    ]


LP_EDGE_RELATIONS = (
    _rel("interactions", "entity_id", "entities", "entity_id") +
    _rel("interactions", "item_id", "items", "item_id") +
    _rel("item_categories", "item_id", "items", "item_id") +
    _rel("item_categories", "category_name", "categories", "category_name"))

# Kumo 2's plan for kumo#31404.
LP_NUM_NEIGHBORS = {
    ("interactions", "entity_id", "entities"): [128, 0, 0, 0],
    ("items", "rev_item_id", "interactions"): [0, 128, 0, 0],
    ("item_categories", "item_id", "items"): [0, 0, 3, 0],
    ("categories", "rev_category_name", "item_categories"): [0, 0, 0, 1],
}


def _lp_plan(num_neighbors, module, rhs_emb_mode, tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "index_metadata.json").write_text(
        json.dumps({"edge_relations": LP_EDGE_RELATIONS}))
    trainer = SimpleNamespace(
        sampler_kwargs=SimpleNamespace(num_neighbors=num_neighbors,
                                       temporal_strategy="last"),
        task=SimpleNamespace(task_type=TaskType.TEMPORAL_LINK_PREDICTION),
        module_kwargs=SimpleNamespace(module=module, rhs_emb_mode=rhs_emb_mode),
    )
    manifest = SimpleNamespace(
        target_node_type="items",
        graph_runtime_metadata={"graph_index_path": str(index_path)})
    return DiskgraphNestedTrainer._build_ttng_sampling_plan(trainer, manifest)


def _fanout(n):
    return {
        "fanout": n,
        "mode": "temporal_most_recent",
        "absent_key_policy": "uniform_fallback",
    }


LP_MODULES = [
    (LinkPredModuleType.SHALLOW_RHS, "fusion"),
    (LinkPredModuleType.SHALLOW_RHS, "lookup"),
    (LinkPredModuleType.HYBRID, "feature"),
    (LinkPredModuleType.ID, "lookup"),
    (LinkPredModuleType.TWO_TOWER, "feature"),
]


@pytest.mark.parametrize("module,rhs_emb_mode", LP_MODULES)
def test_lp_plan_keeps_hops_through_target_table(module, rhs_emb_mode,
                                                 tmp_path):
    plan = _lp_plan(NumNeighbors(LP_NUM_NEIGHBORS, [0, 0, 0, 0]), module,
                    rhs_emb_mode, tmp_path)

    assert plan["layers"] == [
        {
            "fanouts": {
                "interactions_entity_id_to_entities_entity_id_reverse":
                _fanout(128)
            },
            "default_fanout": None,
        },
        {
            "fanouts": {
                "interactions_item_id_to_items_item_id": _fanout(128)
            },
            "default_fanout": None,
        },
        {
            "fanouts": {
                "item_categories_item_id_to_items_item_id_reverse":
                _fanout(3)
            },
            "default_fanout": None,
        },
        {
            "fanouts": {
                "item_categories_category_name_to_categories_category_name": _fanout(1)
            },
            "default_fanout": None,
        },
    ]


@pytest.mark.parametrize("module,rhs_emb_mode", LP_MODULES)
def test_lp_list_plan_uses_default_fanout_on_every_hop(module, rhs_emb_mode,
                                                       tmp_path):
    plan = _lp_plan(NumNeighbors([12, 8]), module, rhs_emb_mode, tmp_path)

    assert plan["layers"] == [
        {
            "fanouts": {},
            "default_fanout": _fanout(12)
        },
        {
            "fanouts": {},
            "default_fanout": _fanout(8)
        },
    ]


def test_explicit_zero_overrides_default_fanout(tmp_path):
    plan = _lp_plan(
        NumNeighbors({("interactions", "entity_id", "entities"): [0, 5]},
                     [12, 12]), LinkPredModuleType.ID, "lookup", tmp_path)

    rel = "interactions_entity_id_to_entities_entity_id_reverse"
    assert plan["layers"] == [
        {
            "fanouts": {
                rel: _fanout(0)
            },
            "default_fanout": _fanout(12)
        },
        {
            "fanouts": {
                rel: _fanout(5)
            },
            "default_fanout": _fanout(12)
        },
    ]


def test_unresolved_entry_is_reported(tmp_path, caplog):
    num_neighbors = NumNeighbors(
        {
            ("interactions", "entity_id", "entities"): [4],
            ("missing", "EDGE", "items"): [7],
        }, [0])

    with caplog.at_level(logging.WARNING):
        plan = _lp_plan(num_neighbors, LinkPredModuleType.SHALLOW_RHS, "fusion",
                        tmp_path)

    assert plan["layers"] == [{
        "fanouts": {
            "interactions_entity_id_to_entities_entity_id_reverse": _fanout(4)
        },
        "default_fanout": None,
    }]
    assert "('missing', 'EDGE', 'items')" in caplog.text
