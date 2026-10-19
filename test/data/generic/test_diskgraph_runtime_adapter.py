# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``runtime_batch_to_hetero``.

Pure data-transformation tests that do not require a real native
runtime; the synthetic batch dicts here mirror the shape produced by
``diskgraph.make_task_batch_iterator`` so the adapter is exercised
end-to-end against the runtime contract without standing up a graph fixture.
"""
from __future__ import annotations

from typing import Any

import kumo_core.diskgraph.runtime_adapter as runtime_adapter
import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.diskgraph.metadata import normalize_diskgraph_edge_type
from kumo_core.diskgraph.runtime_adapter import (
    _expand_lp_exclude,
    attach_tensor_frames,
    runtime_batch_to_hetero,
)
from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
from kumo_core.encoders.target.multicategorical import (
    MultiCategoricalTargetEncoder,
)
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    MultiCategoricalTensorMapper,
    SequenceTensorMapper,
    TensorMapper,
)
from kumo_core.train.autoregression import FUTURE_LABELS_COL
from kumoapi.encoder import Stype
from kumoapi.task import TaskType as ApiTaskType
from torch_frame.data import MultiNestedTensor
from torch_geometric.data import HeteroData


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


def _empty_batch(*, n_inputs: int = 0) -> dict[str, Any]:
    """Runtime batch shape with no engine attached: only the task block
    is populated.  ``graphs`` and ``raw_features`` are empty dicts."""
    return {
        "task": {
            "input_ids":
            pa.array(range(n_inputs), type=pa.uint64()),
            "tt_stage_node_ids":
            pa.array(range(n_inputs), type=pa.uint64()),
            "seed_node_ids":
            pa.array([100 + i for i in range(n_inputs)], type=pa.uint64()),
            "target":
            pa.array([float(i % 2) for i in range(n_inputs)],
                     type=pa.float64()),
        },
        "graphs": {},
        "raw_features": {},
        "metadata": {},
    }


def _populated_batch() -> dict[str, Any]:
    """Runtime batch with an engine attached: 3 seeds + 4 sampled
    User nodes, one User-(follows)->User edge, fused-feature record
    batches for User."""
    feat_schema = pa.schema([
        ("age", pa.int64()),
        ("country", pa.string()),
    ])
    user_features = pa.RecordBatch.from_arrays(
        [
            pa.array([41, 22, 35, 19], type=pa.int64()),
            pa.array(["US", "DE", "FR", "US"], type=pa.string()),
        ],
        schema=feat_schema,
    )
    return {
        "task": {
            "input_ids":
            pa.array([0, 1, 2], type=pa.uint64()),
            "tt_stage_node_ids":
            pa.array([10, 11, 12], type=pa.uint64()),
            "seed_node_ids":
            pa.array([100, 101, 102], type=pa.uint64()),
            "seed_timestamps":
            pa.array([1_700_000_000, 1_700_500_000, 1_700_600_000],
                     type=pa.int64()),
            "target":
            pa.array([1.0, 0.0, 1.0], type=pa.float64()),
            "weight":
            pa.array([0.5, 1.0, 1.5], type=pa.float64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array([100, 101, 102, 200], type=pa.uint64()),
                },
                "edge_index_dict": [
                    {
                        "edge_type": {
                            "src": "User",
                            "rel": "follows",
                            "dst": "User"
                        },
                        "src": pa.array([0, 1], type=pa.uint32()),
                        "dst": pa.array([3, 3], type=pa.uint32()),
                    },
                ],
                "batch_dict": {
                    "User": pa.array([0, 1, 2, 1], type=pa.uint32()),
                },
                "num_sampled_nodes_dict": {
                    "User": pa.array([3, 1], type=pa.uint64()),
                },
                "num_sampled_edges_dict": [
                    {
                        "edge_type": {
                            "src": "User",
                            "rel": "follows",
                            "dst": "User"
                        },
                        "counts": pa.array([2], type=pa.uint64()),
                    },
                ],
                "metadata": {},
            }
        },
        "raw_features": {
            "User": {
                "key": "User",
                "num_rows": 4,
                "columns": ["age", "country"],
                "metadata": {},
                "node_ids": pa.array([100, 101, 102, 200], type=pa.uint64()),
                "record_batches": [user_features],
            }
        },
        "metadata": {},
    }


def test_empty_batch_yields_seed_block_only() -> None:
    batch = _empty_batch(n_inputs=3)
    hetero, raw_features, raw_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table")

    # Seed block is populated.
    seed = hetero["train_table"]
    assert torch.equal(seed.input_id, torch.tensor([0, 1, 2]))
    assert torch.equal(seed.batch, torch.tensor([0, 1, 2]))
    assert torch.equal(seed.y, torch.tensor([0.0, 1.0, 0.0]))
    # No time or weight when the columns are all-null.
    assert "time" not in seed
    assert "weight" not in seed

    # No graph topology, no raw features.
    assert hetero.node_types == ["train_table"]
    assert hetero.edge_types == []
    assert raw_features == {}
    assert raw_row_ids == {}


def test_populated_batch_builds_graph_and_seed_block() -> None:
    batch = _populated_batch()
    hetero, raw_features, raw_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table")

    # Graph topology.
    assert torch.equal(hetero["User"].n_id, torch.tensor([100, 101, 102, 200]))
    assert torch.equal(hetero["User"].batch, torch.tensor([0, 1, 2, 1]))

    # Edge index is emitted as native columnar src/dst arrays.
    edge_type = ("User", "follows", "User")
    assert edge_type in hetero.edge_types
    assert torch.equal(
        hetero[edge_type].edge_index,
        torch.tensor([[0, 1], [3, 3]]),
    )

    # Seed block: input_id, time, y, weight all populated.
    seed = hetero["train_table"]
    assert torch.equal(seed.input_id, torch.tensor([0, 1, 2]))
    assert torch.equal(seed.batch, torch.tensor([0, 1, 2]))
    # Engine emits seconds (default unit); adapter normalizes to ns canonical.
    assert torch.equal(
        seed.time,
        torch.tensor([1_700_000_000, 1_700_500_000, 1_700_600_000]) *
        1_000_000_000,
    )
    assert torch.allclose(seed.y, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.allclose(seed.weight, torch.tensor([0.5, 1.0, 1.5]))


def test_epoch_second_seed_timestamps_are_normalized_to_ns() -> None:
    batch = _empty_batch(n_inputs=2)
    batch["task"]["seed_timestamps"] = pa.array([1_700_000_000, 1_700_000_001],
                                                type=pa.int64())

    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table")

    expected = torch.tensor(
        [1_700_000_000_000_000_000, 1_700_000_001_000_000_000])
    assert torch.equal(hetero["train_table"].time, expected)
    assert torch.equal(hetero["train_table"].seed_time, expected)


def test_diskgraph_fk_relation_names_are_normalized() -> None:
    assert normalize_diskgraph_edge_type(
        "transactions",
        "transactions_article_id_to_articles_article_id",
        "articles",
    ) == ("transactions", "article_id", "articles")

    assert normalize_diskgraph_edge_type(
        "articles",
        "transactions_article_id_to_articles_article_id_reverse",
        "transactions",
    ) == ("articles", "rev_article_id", "transactions")


def test_multiclass_target_record_batch_uses_target_encoder_mapping() -> None:
    batch = _empty_batch(n_inputs=3)
    batch["task"]["target"] = pa.RecordBatch.from_arrays(
        [pa.array(["silver", "gold", "silver"], type=pa.string())],
        names=["TARGET1"],
    )
    target_encoder = CategoricalTargetEncoder(
        ApiTaskType.MULTICLASS_CLASSIFICATION,
        CategoricalTensorMapper([["silver", 10], ["gold", 5]]),
    )

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="multiclass_classification",
        target_encoder=target_encoder,
    )

    assert torch.equal(
        hetero["train_table"].y,
        torch.tensor([[0], [1], [0]], dtype=torch.long),
    )


def test_multiclass_target_record_batch_maps_numeric_strings() -> None:
    batch = _empty_batch(n_inputs=3)
    batch["task"]["target"] = pa.RecordBatch.from_arrays(
        [pa.array(["1", "2", "1"], type=pa.string())],
        names=["TARGET1"],
    )
    target_encoder = CategoricalTargetEncoder(
        ApiTaskType.MULTICLASS_CLASSIFICATION,
        CategoricalTensorMapper([[2, 10], [1, 5]]),
    )

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="multiclass_classification",
        target_encoder=target_encoder,
    )

    assert torch.equal(
        hetero["train_table"].y,
        torch.tensor([[1], [0], [1]], dtype=torch.long),
    )


def test_multilabel_target_record_batch_builds_multi_nested_tensor() -> None:
    batch = _empty_batch(n_inputs=3)
    batch["task"]["target"] = pa.RecordBatch.from_arrays(
        [pa.array(["red;blue", "", "blue"], type=pa.string())],
        names=["TARGET1"],
    )
    target_encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([["red", 10], ["blue", 5]], sep=";"))

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="multilabel_classification",
        target_encoder=target_encoder,
    )

    y = hetero["train_table"].y
    assert isinstance(y, MultiNestedTensor)
    assert torch.equal(y.values, torch.tensor([0, 1, 1]))
    assert torch.equal(y.offset, torch.tensor([0, 2, 2, 3]))
    assert torch.equal(
        target_encoder.encode_tensor(y),
        torch.tensor([[1., 1.], [0., 0.], [0., 1.]]),
    )


def test_multilabel_target_record_batch_maps_numeric_strings() -> None:
    batch = _empty_batch(n_inputs=3)
    batch["task"]["target"] = pa.RecordBatch.from_arrays(
        [pa.array(["1|2", "", "2"], type=pa.string())],
        names=["TARGET1"],
    )
    target_encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([[2, 10], [1, 5]], sep="|"))

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="multilabel_classification",
        target_encoder=target_encoder,
    )

    y = hetero["train_table"].y
    assert isinstance(y, MultiNestedTensor)
    assert torch.equal(y.values, torch.tensor([1, 0, 0]))
    assert torch.equal(y.offset, torch.tensor([0, 2, 2, 3]))
    assert torch.equal(
        target_encoder.encode_tensor(y),
        torch.tensor([[1., 1.], [0., 0.], [1., 0.]]),
    )


def test_multilabel_ranking_target_record_batch_builds_multi_nested_tensor(
) -> None:
    batch = _empty_batch(n_inputs=3)
    batch["task"]["target"] = pa.RecordBatch.from_arrays(
        [pa.array(["red|blue", "", "blue"], type=pa.string())],
        names=["TARGET1"],
    )
    target_encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([["red", 10], ["blue", 5]], sep="|"))

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="multilabel_ranking",
        target_encoder=target_encoder,
    )

    y = hetero["train_table"].y
    assert isinstance(y, MultiNestedTensor)
    assert torch.equal(y.values, torch.tensor([0, 1, 1]))
    assert torch.equal(y.offset, torch.tensor([0, 2, 2, 3]))
    assert torch.equal(
        target_encoder.encode_tensor(y),
        torch.tensor([[1., 1.], [0., 0.], [0., 1.]]),
    )


def test_multilabel_ranking_target_record_batch_maps_numeric_strings() -> None:
    batch = _empty_batch(n_inputs=3)
    batch["task"]["target"] = pa.RecordBatch.from_arrays(
        [pa.array(["1|2", "", "2"], type=pa.string())],
        names=["TARGET1"],
    )
    target_encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([[2, 10], [1, 5]], sep="|"))

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="multilabel_ranking",
        target_encoder=target_encoder,
    )

    y = hetero["train_table"].y
    assert isinstance(y, MultiNestedTensor)
    assert torch.equal(y.values, torch.tensor([1, 0, 0]))
    assert torch.equal(y.offset, torch.tensor([0, 2, 2, 3]))
    assert torch.equal(
        target_encoder.encode_tensor(y),
        torch.tensor([[1., 1.], [0., 0.], [1., 0.]]),
    )


def test_forecasting_skips_seed_y_for_sequence_target() -> None:
    batch = _empty_batch(n_inputs=2)
    batch["task"]["target"] = pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        task_type="forecasting",
    )

    assert "y" not in hetero["train_table"]


def test_columnar_native_batch_shape_matches_diskgraph_iterator_output(
) -> None:
    features = pa.RecordBatch.from_arrays(
        [pa.array([41, 22, 35, 19], type=pa.int64())],
        schema=pa.schema([("age", pa.int64())]),
    )
    batch = {
        "task": {
            "input_ids": pa.array([0, 1, 2], type=pa.uint64()),
            "tt_stage_node_ids": pa.array([10, 11, 12], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101, 102], type=pa.uint64()),
            "seed_timestamps": pa.array([1700, 1701, 1702], type=pa.int64()),
            "target": pa.array([1.0, 0.0, 1.0], type=pa.float64()),
            "weight": pa.array([0.5, 1.0, 1.5], type=pa.float64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array([100, 101, 102, 200], type=pa.uint64())
                },
                "batch_dict": {
                    "User": pa.array([0, 1, 2, 1], type=pa.uint32())
                },
                "edge_index_dict": [{
                    "edge_type": {
                        "src": "User",
                        "rel": "follows",
                        "dst": "User"
                    },
                    "src": pa.array([0, 1], type=pa.uint32()),
                    "dst": pa.array([3, 3], type=pa.uint32()),
                }],
                "num_sampled_nodes_dict": {
                    "User": pa.array([3, 1], type=pa.uint64())
                },
                "num_sampled_edges_dict": [{
                    "edge_type": {
                        "src": "User",
                        "rel": "follows",
                        "dst": "User"
                    },
                    "counts":
                    pa.array([2], type=pa.uint64()),
                }],
            }
        },
        "raw_features": {
            "User": {
                "key": "User",
                "num_rows": 4,
                "node_ids": pa.array([100, 101, 102, 200], type=pa.uint64()),
                "record_batches": [features],
            }
        },
    }

    hetero, raw_features, raw_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table",
        task_type="binary_classification")

    assert hetero["train_table"].input_id.tolist() == [0, 1, 2]
    # seconds (default engine unit) normalized to ns canonical.
    assert hetero["train_table"].time.tolist() == [
        1700 * 1_000_000_000, 1701 * 1_000_000_000, 1702 * 1_000_000_000
    ]
    assert hetero["train_table"].y.tolist() == [[1], [0], [1]]
    assert hetero["train_table"].weight.tolist() == [0.5, 1.0, 1.5]
    assert hetero["User"].n_id.tolist() == [100, 101, 102, 200]
    assert hetero["User"].batch.tolist() == [0, 1, 2, 1]
    assert hetero[("User", "follows", "User")].edge_index.tolist() == [[0, 1],
                                                                       [3, 3]]
    assert hetero["User"].num_sampled_nodes == [3, 1]
    assert hetero[("User", "follows", "User")].num_sampled_edges == [2]
    assert raw_features["User"].column("age").to_pylist() == [41, 22, 35, 19]
    assert raw_row_ids["User"].tolist() == [100, 101, 102, 200]


def test_populated_batch_returns_raw_features_aligned_with_n_id() -> None:
    batch = _populated_batch()
    _, raw_features, raw_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table")

    assert "User" in raw_features
    rb = raw_features["User"]
    assert rb.num_rows == 4
    assert rb.column_names == ["age", "country"]

    # Row ids parallel the n_id ordering.
    assert torch.equal(raw_row_ids["User"], torch.tensor([100, 101, 102, 200]))


def test_record_batches_concat_when_engine_emits_multiple_hops() -> None:
    """Per-hop fused-feature emission produces multiple record batches
    per node type; the adapter must concatenate them in arrival order."""
    schema = pa.schema([("age", pa.int64())])
    seeds = pa.RecordBatch.from_arrays([pa.array([41, 22], pa.int64())],
                                       schema=schema)
    hop1 = pa.RecordBatch.from_arrays([pa.array([35, 19], pa.int64())],
                                      schema=schema)
    batch = _empty_batch(n_inputs=2)
    batch["graphs"] = {
        "main": {
            "node_ids": {
                "User": pa.array([100, 101, 102, 200], type=pa.uint64())
            },
            "edge_index_dict": [],
            "batch_dict": {},
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [],
            "metadata": {},
        }
    }
    batch["raw_features"] = {
        "User": {
            "key": "User",
            "num_rows": 4,
            "columns": ["age"],
            "metadata": {},
            "record_batches": [seeds, hop1],
        }
    }
    _, raw_features, raw_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table")

    assert raw_features["User"].num_rows == 4
    assert raw_features["User"].column("age").to_pylist() == [41, 22, 35, 19]
    assert torch.equal(raw_row_ids["User"], torch.tensor([100, 101, 102, 200]))


def test_empty_edge_type_lands_as_2x0_tensor() -> None:
    batch = _empty_batch(n_inputs=1)
    batch["graphs"] = {
        "main": {
            "node_ids": {
                "User": pa.array([], type=pa.uint64())
            },
            "edge_index_dict": [
                {
                    "edge_type": {
                        "src": "User",
                        "rel": "follows",
                        "dst": "User"
                    },
                    "src": pa.array([], type=pa.uint32()),
                    "dst": pa.array([], type=pa.uint32()),
                },
            ],
            "batch_dict": {},
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [],
            "metadata": {},
        }
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table")
    edge_type = ("User", "follows", "User")
    assert hetero[edge_type].edge_index.shape == (2, 0)
    assert hetero[edge_type].edge_index.dtype == torch.long


def test_bidirectional_output_reuses_pyg_reverse_edge_transform() -> None:
    batch = _empty_batch(n_inputs=1)
    batch["graphs"] = {
        "main": {
            "node_ids": {
                "transactions": pa.array([10, 11], type=pa.uint64()),
                "articles": pa.array([20], type=pa.uint64()),
            },
            "edge_index_dict": [{
                "edge_type": {
                    "src": "transactions",
                    "rel": "transactions_article_id_to_articles_article_id",
                    "dst": "articles",
                },
                "src": pa.array([0, 1], type=pa.uint32()),
                "dst": pa.array([0, 0], type=pa.uint32()),
            }],
            "batch_dict": {},
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [{
                "edge_type": {
                    "src": "transactions",
                    "rel": "transactions_article_id_to_articles_article_id",
                    "dst": "articles",
                },
                "counts":
                pa.array([2], type=pa.uint64()),
            }, {
                "edge_type": {
                    "src": "articles",
                    "rel":
                    "transactions_article_id_to_articles_article_id_reverse",
                    "dst": "transactions",
                },
                "counts":
                pa.array([0], type=pa.uint64()),
            }],
            "metadata": {},
        }
    }

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        bidirectional_output=True,
    )

    forward = ("transactions", "article_id", "articles")
    reverse = ("articles", "rev_article_id", "transactions")
    assert forward in hetero.edge_types
    assert reverse in hetero.edge_types
    assert hetero[forward].edge_index.tolist() == [[0, 1], [0, 0]]
    assert hetero[reverse].edge_index.tolist() == [[0, 0], [0, 1]]


def test_unsampled_schema_node_types_are_backfilled_empty() -> None:
    """The DG engine reports ``num_sampled_edges`` for EVERY schema relation,
    including all-zero ones, while ``node_ids`` only contains VISITED node
    types (``output.rs`` finalize). When the sampled neighborhood reaches only
    a subset of the schema (e.g. a single-timestamp grouped train batch at an
    early anchor that reaches no RHS ``products`` neighbors), the unvisited
    types are absent from ``node_ids``.

    The adapter must BACK-FILL every schema node type with a zero-length block
    (mirroring PyG ``filter_hetero_data``, loader/utils.py:180-186 "disconnected
    graph sampling"), so ``node_id_dict[<type>]`` is ALWAYS present (empty when
    unsampled). This matches PyG / legacy exactly and lets the LP modules read
    ``node_id_dict[rhs]`` without a per-module guard. The edges whose endpoints
    are now back-filled are kept (zero-column), exactly as PyG keeps them.
    """
    batch = _empty_batch(n_inputs=1)
    batch["graphs"] = {
        "main": {
            # Only ``users`` was sampled (anchor seeds, no product neighbors).
            "node_ids": {
                "users": pa.array([0, 1, 2], type=pa.uint64()),
            },
            "edge_index_dict": [],
            "batch_dict": {},
            "num_sampled_nodes_dict": {},
            # Engine reports the full schema, incl. zero-count relations whose
            # endpoint node types (``ratings`` / ``products``) were never
            # visited in this batch.
            "num_sampled_edges_dict": [
                {
                    "edge_type": {
                        "src": "users",
                        "rel": "u_to_r",
                        "dst": "ratings",
                    },
                    "counts": pa.array([0, 0], type=pa.uint64()),
                },
                {
                    "edge_type": {
                        "src": "ratings",
                        "rel": "r_to_p",
                        "dst": "products",
                    },
                    "counts": pa.array([0, 0], type=pa.uint64()),
                },
            ],
            "metadata": {},
        }
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table")
    # Every schema node type is present; the unsampled ones are empty blocks.
    assert "products" in hetero.node_types
    assert "ratings" in hetero.node_types
    assert hetero["products"].n_id.numel() == 0
    assert hetero["products"].n_id.dtype == torch.long
    assert hetero["ratings"].n_id.numel() == 0
    # The visited type keeps its real ids.
    assert hetero["users"].n_id.numel() == 3
    # Empty blocks carry consistent per-type attrs (num_sampled_nodes == [0])
    # AND an empty ``batch`` vector, so the feature encoder can index
    # ``batch_dict[type]`` for every type it sees (tensor_frame_encoder:434).
    assert hetero["products"].num_sampled_nodes == [0]
    assert "batch" in hetero["products"]
    assert hetero["products"].batch.numel() == 0
    assert "batch" in hetero["ratings"]
    # Edges with now-present (back-filled) endpoints are kept, zero-column.
    assert ("ratings", "r_to_p", "products") in hetero.edge_types
    assert hetero["ratings", "r_to_p", "products"].edge_index.shape == (2, 0)


def test_grouped_lp_batch_missing_rhs_has_empty_rhs_block() -> None:
    """Regression for the grouped temporal-LP ``KeyError: 'products'`` crash:
    a single-timestamp train batch whose neighborhood reaches zero RHS nodes
    must still expose ``node_id_dict['products']`` as an EMPTY block, so
    ``shallow_rhs``/``hybrid`` ``forward`` can read it without a guard."""
    batch = _empty_batch(n_inputs=2)
    batch["graphs"] = {
        "main": {
            "node_ids": {
                "users": pa.array([0, 1], type=pa.uint64()),
            },
            "edge_index_dict": [],
            "batch_dict": {},
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [
                {
                    "edge_type": {
                        "src": "ratings",
                        "rel": "r_to_p",
                        "dst": "products",
                    },
                    "counts": pa.array([0, 0], type=pa.uint64()),
                },
            ],
            "metadata": {},
        }
    }
    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
    )
    # node_id_dict (what shallow_rhs/hybrid forward read) always has the RHS.
    node_id_dict = {
        nt: hetero[nt].n_id
        for nt in hetero.node_types if "n_id" in hetero[nt]
    }
    assert "products" in node_id_dict
    assert node_id_dict["products"].numel() == 0
    assert node_id_dict["users"].numel() == 2


def test_partial_optional_columns_are_skipped() -> None:
    """Mixed-presence supervision columns (some None, some Some) are
    not surfaced on the seed block — the caller should validate
    upstream rather than inferring from a half-populated tensor."""
    batch = _empty_batch(n_inputs=3)
    batch["task"]["seed_timestamps"] = pa.array(
        [1_700_000_000, None, 1_700_600_000], type=pa.int64())
    batch["task"]["weight"] = pa.array([0.5, None, 1.5], type=pa.float64())
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table")
    seed = hetero["train_table"]
    assert "time" not in seed
    assert "weight" not in seed
    # target was fully populated, so it survives.
    assert torch.equal(seed.y, torch.tensor([0.0, 1.0, 0.0]))


def _numerical_mapper(values: list[float]) -> TensorMapper:
    """Fit a numerical TensorMapper on a small sample so tests can
    exercise the bridge end-to-end without standing up a full
    EncodersConfig + FeatureStore."""
    return TensorMapper.from_stype(Stype.numerical, pd.Series(values))


def _user_mapper_bank() -> dict[str, dict[str, TensorMapper]]:
    return {
        "users_v1": {
            "age": _numerical_mapper([18.0, 25.0, 30.0, 50.0]),
        }
    }


def _user_feature_schema() -> dict[str, Any]:
    return {
        "node_types": {
            "User": {
                "feature_columns": ["age"],
                "time_column": None,
                "pkey_column": None,
                "mapper_key": "users_v1",
            }
        },
        "edge_types": {},
        "raw_feature_format": "arrow_columnar_v1",
        "provenance_mode": "node_id_time_keyed_v1",
        "metadata": {},
    }


def test_attach_tensor_frames_populates_hetero_tf() -> None:
    """The bridge resolves mapper_key -> mapper_dict, converts the
    RecordBatch via get_tf, and attaches hetero[ns].tf."""
    batch = _populated_batch()
    hetero, raw_features, _ = runtime_batch_to_hetero(
        batch, train_table_name="train_table")
    # Replace the sample RecordBatch with an `age`-only one so the
    # mapper bank fitted above lines up.
    age_only = pa.RecordBatch.from_arrays(
        [pa.array([41, 22, 35, 19], type=pa.int64())],
        schema=pa.schema([("age", pa.int64())]),
    )
    raw_features["User"] = age_only

    attach_tensor_frames(
        hetero,
        raw_features,
        _user_mapper_bank(),
        _user_feature_schema(),
    )

    assert hasattr(hetero["User"], "tf")
    tf = hetero["User"].tf
    assert tf.num_rows == 4


def test_runtime_batch_to_hetero_attaches_tf_when_mapper_bank_passed() -> None:
    """End-to-end: passing mapper_bank + feature_schema to the top-
    level adapter attaches tf in one call."""
    batch = _populated_batch()
    age_only = pa.RecordBatch.from_arrays(
        [pa.array([41, 22, 35, 19], type=pa.int64())],
        schema=pa.schema([("age", pa.int64())]),
    )
    batch["raw_features"]["User"]["record_batches"] = [age_only]

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        mapper_bank=_user_mapper_bank(),
        feature_schema=_user_feature_schema(),
    )

    assert hasattr(hetero["User"], "tf")
    assert hetero["User"].tf.num_rows == 4


def test_record_batch_to_tf_uses_arrow_mapper_contract(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array([41, 22], type=pa.int64())],
        names=['age'],
    )
    expected = object()
    calls = []

    def fake_get_tf(frame, mapper_dict):
        calls.append(frame)
        assert isinstance(frame, (pa.RecordBatch, pa.Table))
        assert frame.column('age').to_pylist() == [41, 22]
        assert list(mapper_dict) == ['age']
        return expected

    monkeypatch.setattr(runtime_adapter, 'get_tf', fake_get_tf)

    assert runtime_adapter._record_batch_to_tf(record_batch,
                                               {'age': object()}) is expected
    assert len(calls) == 1


def test_record_batch_to_tf_ignores_columns_no_mapper_claims() -> None:
    """The batch is EXPECTED to carry non-encoder columns -- the node time
    column always, the primary key on the eval path -- and they must not reach
    the TensorFrame.

    ``get_tf`` iterates the mapper and looks columns up by name, so no
    projection of the batch is needed to achieve that. This pins the property
    the removed ``.select()`` used to appear to provide, so that deleting it
    stays safe if ``get_tf`` is ever reimplemented.
    """
    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array([41.0, 22.0], type=pa.float64()),
            pa.array([1_700_000_000, 1_700_000_100], type=pa.int64()),
            pa.array([7, 8], type=pa.int64()),
        ],
        names=["age", "created_at", "user_id"],
    )
    tf = runtime_adapter._record_batch_to_tf(
        record_batch, {"age": _numerical_mapper([18.0, 25.0, 30.0, 50.0])}
    )

    assert [c for cols in tf.col_names_dict.values() for c in cols] == ["age"]
    assert tf.num_rows == 2


def test_record_batch_to_tf_raises_when_a_mapper_column_is_missing() -> None:
    """A projection narrower than the mapper bank must FAIL, not silently
    encode fewer features than the model was fitted on.

    Before projection pushdown this could only mean the snapshot did not store
    the column; the batch always carried everything else. Now it can also mean
    `_feature_projection` did not ask for it, and that failure is otherwise
    invisible -- no exception, no log line, just a model quietly training on
    fewer inputs.
    """
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array([41.0, 22.0], type=pa.float64())], names=["age"]
    )
    mapper = _numerical_mapper([18.0, 25.0, 30.0, 50.0])

    with pytest.raises(RuntimeError, match="income"):
        runtime_adapter._record_batch_to_tf(
            record_batch,
            {
                "age": mapper,
                "income": mapper,
            },
            namespace="User",
        )


def test_record_batch_to_tf_zero_mapper_type_is_not_an_error() -> None:
    """A pure link table has NO fitted encoders, so an empty mapper is the
    normal case, not a missing-column one. It must yield the same zero-row,
    zero-stype TF the back-fill pass builds when the projection skips the type
    outright -- that equivalence is what makes skipping safe."""
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array([7, 8], type=pa.int64())], names=["ownership_id"]
    )

    tf = runtime_adapter._record_batch_to_tf(record_batch, {}, namespace="Ownership")
    assert len(tf.feat_dict) == 0
    assert tf.num_rows == 0


def test_backfilled_type_with_mapper_gets_empty_shaped_tf() -> None:
    """A schema type with a ``mapper_key`` that the batch did NOT sample (so it
    is parity back-filled as an empty block, with no raw RecordBatch) must still
    receive a MAPPER-SHAPED empty TF — i.e. its ``feat_dict`` carries the
    encoder's expected stypes at zero rows. A bare ``TensorFrame({}, {})`` would
    KeyError in the feature encoder (``tf.feat_dict[stype]``)."""
    # Batch samples NOTHING in the graph: User is reachable only via the schema
    # edge, so it is back-filled empty. No raw features for User.
    batch = _empty_batch(n_inputs=1)
    batch["graphs"] = {
        "main": {
            "node_ids": {},
            "edge_index_dict": [],
            "batch_dict": {},
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [{
                "edge_type": {
                    "src": "User",
                    "rel": "self",
                    "dst": "User",
                },
                "counts":
                pa.array([0, 0], type=pa.uint64()),
            }],
            "metadata": {},
        }
    }
    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        mapper_bank=_user_mapper_bank(),
        feature_schema=_user_feature_schema(),
    )
    assert "User" in hetero.node_types
    assert hetero["User"].n_id.numel() == 0
    tf = hetero["User"].tf
    assert tf.num_rows == 0
    # The encoder iterates feat_dict stypes; the mapper-shaped empty TF must
    # carry them (numerical, from the `age` numerical mapper) at zero rows.
    assert len(tf.feat_dict) > 0
    for stype, feat in tf.feat_dict.items():
        assert feat.shape[0] == 0


def test_forecasting_aliases_seed_stage_features_to_train_table_tf() -> None:
    batch = _populated_batch()
    future = pa.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                      type=pa.list_(pa.float64()))
    batch["raw_features"]["train_seeds"] = {
        "key":
        "train_seeds",
        "num_rows":
        3,
        "columns": [FUTURE_LABELS_COL],
        "metadata": {},
        "record_batches":
        [pa.RecordBatch.from_arrays([future], names=[FUTURE_LABELS_COL])],
    }
    feature_schema = _user_feature_schema()
    feature_schema["node_types"]["train_table"] = {
        "feature_columns": [FUTURE_LABELS_COL],
        "mapper_key": "train_table",
    }
    mapper_bank = _user_mapper_bank()
    mapper_bank["train_table"] = {
        FUTURE_LABELS_COL: SequenceTensorMapper(min_length=3, max_length=3),
    }

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        mapper_bank=mapper_bank,
        feature_schema=feature_schema,
        task_type="forecasting",
    )

    tf = hetero["train_table"].tf
    assert tf.col_names_dict["sequence"] == [FUTURE_LABELS_COL]
    assert torch.equal(
        tf.feat_dict["sequence"].values,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )


def test_runtime_batch_to_hetero_rejects_partial_mapper_args() -> None:
    """Passing one of (mapper_bank, feature_schema) without the other
    is a programmer error — surface it at the entry point."""
    batch = _empty_batch(n_inputs=1)
    with pytest.raises(ValueError, match="both be provided"):
        runtime_batch_to_hetero(
            batch,
            train_table_name="train_table",
            mapper_bank=_user_mapper_bank(),
        )
    with pytest.raises(ValueError, match="both be provided"):
        runtime_batch_to_hetero(
            batch,
            train_table_name="train_table",
            feature_schema=_user_feature_schema(),
        )


def test_attach_tensor_frames_skips_namespace_without_mapper_key() -> None:
    """A node type with no mapper_key in its feature_schema entry is
    skipped silently — typical for edge types or feature-less node
    types."""
    schema = pa.schema([("x", pa.int64())])
    raw_features = {
        "edge_namespace":
        pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int64())],
                                   schema=schema)
    }
    hetero = HeteroData()
    feature_schema = {
        "node_types": {
            "edge_namespace": {
                "feature_columns": ["x"],
                "time_column": None,
                "pkey_column": None,
                "mapper_key": None,  # no mapper -> skip
            }
        }
    }
    attach_tensor_frames(hetero, raw_features, _user_mapper_bank(),
                         feature_schema)
    assert "edge_namespace" not in hetero.node_types or not hasattr(
        hetero["edge_namespace"], "tf")


def test_attach_tensor_frames_raises_on_unknown_mapper_key() -> None:
    """A mapper_key in feature_schema that's not in the mapper bank
    is a configuration error — fail loudly rather than silently
    producing a feature-less batch."""
    schema = pa.schema([("age", pa.int64())])
    raw_features = {
        "User":
        pa.RecordBatch.from_arrays([pa.array([41, 22], type=pa.int64())],
                                   schema=schema)
    }
    hetero = HeteroData()
    feature_schema = {
        "node_types": {
            "User": {
                "feature_columns": ["age"],
                "time_column": None,
                "pkey_column": None,
                "mapper_key": "missing_v1",
            }
        }
    }
    with pytest.raises(KeyError, match="missing_v1"):
        attach_tensor_frames(hetero, raw_features, _user_mapper_bank(),
                             feature_schema)


def test_attach_tensor_frames_handles_empty_record_batch() -> None:
    """An empty (zero-row) record batch still produces a TensorFrame
    with the mapper-known schema, so downstream encoders see a
    consistent shape."""
    schema = pa.schema([("age", pa.int64())])
    empty = pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())],
                                       schema=schema)
    raw_features = {"User": empty}
    hetero = HeteroData()
    attach_tensor_frames(
        hetero,
        raw_features,
        _user_mapper_bank(),
        _user_feature_schema(),
    )
    assert hasattr(hetero["User"], "tf")
    assert hetero["User"].tf.num_rows == 0


def test_multi_channel_runtime_uses_requested_channel() -> None:
    """LP-style multi-channel runtime: caller picks the channel."""
    schema = pa.schema([("x", pa.int64())])
    rb = pa.RecordBatch.from_arrays([pa.array([1], pa.int64())], schema=schema)
    batch = _empty_batch(n_inputs=1)
    batch["graphs"] = {
        "lhs": {
            "node_ids": {
                "User": pa.array([10], type=pa.uint64())
            },
            "edge_index_dict": [],
            "batch_dict": {
                "User": pa.array([0], type=pa.uint32())
            },
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [],
            "metadata": {},
        },
        "rhs": {
            "node_ids": {
                "Item": pa.array([99], type=pa.uint64())
            },
            "edge_index_dict": [],
            "batch_dict": {
                "Item": pa.array([0], type=pa.uint32())
            },
            "num_sampled_nodes_dict": {},
            "num_sampled_edges_dict": [],
            "metadata": {},
        },
    }
    batch["raw_features"] = {
        "User": {
            "key": "User",
            "num_rows": 1,
            "columns": ["x"],
            "metadata": {},
            "record_batches": [rb],
        }
    }

    lhs, _, lhs_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table", channel="lhs")
    assert "User" in lhs.node_types
    assert "Item" not in lhs.node_types
    assert torch.equal(lhs_row_ids["User"], torch.tensor([10]))

    rhs, _, rhs_row_ids = runtime_batch_to_hetero(
        batch, train_table_name="train_table", channel="rhs")
    assert "Item" in rhs.node_types
    assert "User" not in rhs.node_types
    # rhs feature side: User block isn't in this channel, so no row_ids.
    assert "User" not in rhs_row_ids


def test_two_tower_lp_expands_seed_time_to_rhs_seed_layout() -> None:
    """Two-tower training: the deep RHS tower is seeded from the full RHS set
    ``[B positives, anchor-major B*M negatives]``, so RHS-side nodes carry an
    engine ``seed_index`` in ``[0, B*(1+M))``. The encoder indexes a single
    ``seed_time`` with that per-node batch vector, so the seed block's
    ``seed_time`` must be expanded from the B anchor times to that layout
    (positives keep the anchor time; each anchor-major negative repeats its
    anchor's time). Without it ``seed_time[batch]`` is out of bounds."""
    B, M = 2, 2
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101], type=pa.uint64()),
            "seed_timestamps": pa.array([1_700, 1_800], type=pa.int64()),
            "target": pa.array([1.0, 1.0], type=pa.float64()),
            "amount_m": M,
            "num_positives": B,
            "rhs_seed_type": "Item",
            "neg_mask": pa.array([True] * (B * M), type=pa.bool_()),
        },
        "graphs": {
            "main": {
                # B + B*M == 6 RHS seeds, batch == engine seed_index in [0, 6).
                "node_ids": {
                    "Item": pa.array(list(range(6)), type=pa.uint64()),
                },
                "batch_dict": {
                    "Item": pa.array(list(range(6)), type=pa.uint32()),
                },
                "metadata": {},
            }
        },
        "raw_features": {},
        "metadata": {},
    }

    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           seed_node_type="User",
                                           task_type="temporal_link_prediction")

    # Anchor seconds normalized to ns canonical, then expanded to RHS layout.
    NS = 1_000_000_000
    expected = torch.tensor([
        1_700 * NS,
        1_800 * NS,  # positives (anchors 0, 1)
        1_700 * NS,
        1_700 * NS,  # anchor 0's M negatives
        1_800 * NS,
        1_800 * NS
    ])  # anchor 1's M negatives
    seed = hetero["train_table"]
    assert torch.equal(seed.time, expected)
    assert torch.equal(seed.seed_time, expected)
    # Indexing the (length-6) seed_time with the RHS batch vector is now valid.
    assert int(hetero["Item"].batch.max()) < seed.seed_time.numel()


def test_expand_lp_exclude_expands_ragged_counts() -> None:
    """``lp_exclude_counts`` ragged-expands to an anchor-major lhs_batch via
    repeat_interleave; rhs_index passes through unchanged."""
    counts = pa.array([2, 0, 1], type=pa.uint32())
    rhs_global = pa.array([10, 11, 42], type=pa.int64())

    lhs_batch, rhs_index = _expand_lp_exclude(counts, rhs_global)

    # anchor 0 -> 2 excludes, anchor 1 -> none, anchor 2 -> 1 exclude.
    assert torch.equal(lhs_batch, torch.tensor([0, 0, 2]))
    assert torch.equal(rhs_index, torch.tensor([10, 11, 42]))


def test_expand_lp_exclude_all_zero_counts() -> None:
    """All-zero counts -> empty lhs_batch / rhs_index (no excludes)."""
    lhs_batch, rhs_index = _expand_lp_exclude(
        pa.array([0, 0], type=pa.uint32()), pa.array([], type=pa.int64()))
    assert lhs_batch.numel() == 0
    assert rhs_index.numel() == 0


def test_attach_lp_exclude_present_fields() -> None:
    """A static-LP batch carrying ``lp_exclude_*`` attaches expanded
    ``exclude_(lhs_batch, rhs_index)`` to the entity + seed blocks."""
    B = 3
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0, 1, 2], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101, 102], type=pa.uint64()),
            "target": pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            "lp_exclude_counts": pa.array([2, 0, 1], type=pa.uint32()),
            "lp_exclude_rhs_global": pa.array([10, 11, 42], type=pa.int64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array(list(range(B)), type=pa.uint64()),
                    "Item": pa.array(list(range(B)), type=pa.uint64()),
                },
                "batch_dict": {
                    "User": pa.array(list(range(B)), type=pa.uint32()),
                    "Item": pa.array(list(range(B)), type=pa.uint32()),
                },
                "metadata": {},
            }
        },
        "raw_features": {},
        "metadata": {},
    }

    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           seed_node_type="User",
                                           task_type="static_link_prediction")

    for block in (hetero["User"], hetero["train_table"]):
        assert torch.equal(block.exclude_lhs_batch, torch.tensor([0, 0, 2]))
        assert torch.equal(block.exclude_rhs_index, torch.tensor([10, 11, 42]))


def test_attach_lp_exclude_absent_fields_no_attribute() -> None:
    """Without ``lp_exclude_*`` (train / pred / temporal / legacy), the exclude
    attributes are not attached to the blocks."""
    B = 2
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101], type=pa.uint64()),
            "target": pa.array([1.0, 1.0], type=pa.float64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array(list(range(B)), type=pa.uint64()),
                    "Item": pa.array(list(range(B)), type=pa.uint64()),
                },
                "batch_dict": {
                    "User": pa.array(list(range(B)), type=pa.uint32()),
                    "Item": pa.array(list(range(B)), type=pa.uint32()),
                },
                "metadata": {},
            }
        },
        "raw_features": {},
        "metadata": {},
    }

    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           seed_node_type="User",
                                           task_type="static_link_prediction")

    assert getattr(hetero["User"], "exclude_lhs_batch", None) is None
    assert getattr(hetero["User"], "exclude_rhs_index", None) is None
    assert getattr(hetero["train_table"], "exclude_lhs_batch", None) is None


# ---------------------------------------------------------------------------
# Nanosecond-canonical timestamp normalization (pandas-2 resolution fix).
# The DiskGraph engine stamps seed times + node ``time`` in its configured unit
# (default seconds); the adapter must normalize BOTH to int64 NANOSECONDS so the
# temporal encoder's ``rel_time = node.time - seed_time`` is single-unit.
# ---------------------------------------------------------------------------


def test_seed_time_normalized_to_ns_default_seconds_unit() -> None:
    """Default engine unit is seconds; seed_timestamps must come out in ns."""
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101], type=pa.uint64()),
            "seed_timestamps": pa.array([1700, 1701], type=pa.int64()),
        },
        "graphs": {
            "main": {}
        },
        "raw_features": {},
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           task_type="temporal_link_prediction")
    seed = hetero["train_table"]
    assert seed.seed_time.tolist() == [
        1700 * 1_000_000_000, 1701 * 1_000_000_000
    ]
    assert seed.time.tolist() == seed.seed_time.tolist()


def test_seed_time_ns_unit_is_passthrough() -> None:
    """When the engine unit is already ns, no scaling is applied."""
    batch: dict[str, Any] = {
        "task": {
            "input_ids":
            pa.array([0], type=pa.uint64()),
            "seed_node_ids":
            pa.array([100], type=pa.uint64()),
            "seed_timestamps":
            pa.array([1_700_000_000_000_000_000], type=pa.int64()),
        },
        "graphs": {
            "main": {}
        },
        "raw_features": {},
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           task_type="temporal_link_prediction",
                                           index_np_time_unit="ns")
    assert hetero["train_table"].seed_time.tolist() == [
        1_700_000_000_000_000_000
    ]


def test_seed_time_ms_unit_scaled_to_ns() -> None:
    """Millisecond engine unit -> ns via the manifest-driven factor (1e6)."""
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0], type=pa.uint64()),
            "seed_node_ids": pa.array([100], type=pa.uint64()),
            "seed_timestamps": pa.array([1700], type=pa.int64()),
        },
        "graphs": {
            "main": {}
        },
        "raw_features": {},
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           task_type="temporal_link_prediction",
                                           index_np_time_unit="ms")
    assert hetero["train_table"].seed_time.tolist() == [1700 * 1_000_000]


def test_neighbor_node_time_normalized_to_ns_int64_seconds() -> None:
    """A bare int64 node ``time`` column (engine seconds) -> ns canonical."""
    features = pa.RecordBatch.from_arrays(
        [
            pa.array([100, 101], type=pa.uint64()),
            pa.array([1700, 1701], type=pa.int64())
        ],
        schema=pa.schema([("user_id", pa.uint64()),
                          ("event_time", pa.int64())]),
    )
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101], type=pa.uint64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array([100, 101], type=pa.uint64())
                },
                "batch_dict": {
                    "User": pa.array([0, 1], type=pa.uint32())
                },
            }
        },
        "raw_features": {
            "User": {
                "key": "User",
                "num_rows": 2,
                "node_ids": pa.array([100, 101], type=pa.uint64()),
                "record_batches": [features],
            }
        },
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           task_type="binary_classification",
                                           time_col_dict={"User": "event_time"})
    assert hetero["User"].time.tolist() == [
        1700 * 1_000_000_000, 1701 * 1_000_000_000
    ]


def test_neighbor_node_time_timestamp_resolution_normalized_to_ns() -> None:
    """A real Arrow ``timestamp[s]`` node ``time`` -> ns regardless of unit arg
    (the timestamp carries its own resolution)."""
    SEC = 1_609_459_200  # 2021-01-01 UTC
    features = pa.RecordBatch.from_arrays(
        [
            pa.array([100], type=pa.uint64()),
            pa.array([SEC], type=pa.timestamp("s"))
        ],
        schema=pa.schema([("user_id", pa.uint64()),
                          ("event_time", pa.timestamp("s"))]),
    )
    batch: dict[str, Any] = {
        "task": {
            "input_ids": pa.array([0], type=pa.uint64()),
            "seed_node_ids": pa.array([100], type=pa.uint64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "User": pa.array([100], type=pa.uint64())
                },
                "batch_dict": {
                    "User": pa.array([0], type=pa.uint32())
                },
            }
        },
        "raw_features": {
            "User": {
                "key": "User",
                "num_rows": 1,
                "node_ids": pa.array([100], type=pa.uint64()),
                "record_batches": [features],
            }
        },
    }
    hetero, _, _ = runtime_batch_to_hetero(batch,
                                           train_table_name="train_table",
                                           task_type="binary_classification",
                                           time_col_dict={"User": "event_time"})
    assert hetero["User"].time.tolist() == [SEC * 1_000_000_000]


def _ttng_nodepred_batch_with_seed_supervision_relation() -> dict[str, Any]:
    """Node-prediction runtime batch as the TT-NG engine emits it: the sampled
    graph node_ids cover only the real schema types (``products``), but the
    engine's ``num_sampled_edges_dict`` reports a per-relation count for the
    train-table extension's supervision relation
    ``<tt_id>/train_seeds -> products`` (and the val/test seed relations). Those
    seed endpoints are NOT graph schema node types and have no encoder, so the
    parity back-fill must not introduce blocks for them."""
    seed_ns = "gen-traintable-job-deadbeef/train_seeds"
    test_seed_ns = "gen-traintable-job-deadbeef/test_seeds"
    return {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101], type=pa.uint64()),
            "target": pa.array([1.0, 0.0], type=pa.float64()),
        },
        "graphs": {
            "main": {
                "node_ids": {
                    "products": pa.array([100, 101], type=pa.uint64()),
                },
                "edge_index_dict": [],
                "batch_dict": {
                    "products": pa.array([0, 1], type=pa.uint32()),
                },
                "num_sampled_nodes_dict": {
                    "products": pa.array([2], type=pa.uint64()),
                },
                "num_sampled_edges_dict": [
                    {
                        "edge_type": {
                            "src": seed_ns,
                            "rel": "supervises",
                            "dst": "products",
                        },
                        "counts": pa.array([0], type=pa.uint64()),
                    },
                    {
                        "edge_type": {
                            "src": test_seed_ns,
                            "rel": "supervises",
                            "dst": "products",
                        },
                        "counts": pa.array([0], type=pa.uint64()),
                    },
                ],
                "metadata": {},
            }
        },
        "raw_features": {
            "products": {
                "key":
                "products",
                "num_rows":
                2,
                "columns": ["age"],
                "metadata": {},
                "node_ids":
                pa.array([100, 101], type=pa.uint64()),
                "record_batches": [
                    pa.RecordBatch.from_arrays(
                        [pa.array([10, 20], type=pa.int64())],
                        schema=pa.schema([("age", pa.int64())])),
                ],
            }
        },
        "metadata": {},
    }


def test_ttng_seed_supervision_relation_not_backfilled_as_node_type() -> None:
    """REGRESSION: TT-NG node-pred train back-filled the engine's seed_table->
    entity supervision relations into encoder-less ``<tt_id>/<stage>_seeds``
    node blocks (+ placeholder TFs), making the feature encoder
    ``enc_dict[node_type]`` raise ``KeyError`` at the val sanity check. With the
    graph schema node-type allow-set (feature_schema present) the back-fill is
    restricted to schema types, so the seed namespaces never enter the batch."""
    batch = _ttng_nodepred_batch_with_seed_supervision_relation()
    mapper_bank = {
        "products": {
            "age": _numerical_mapper([10.0, 20.0, 30.0, 40.0])
        },
    }
    feature_schema = {
        "node_types": {
            "products": {
                "mapper_key": "products"
            },
        }
    }
    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        seed_node_type="products",
        task_type="binary_classification",
        mapper_bank=mapper_bank,
        feature_schema=feature_schema,
    )

    node_types = set(hetero.node_types)
    # The seed-table supervision namespaces must NOT be present.
    assert "gen-traintable-job-deadbeef/train_seeds" not in node_types
    assert "gen-traintable-job-deadbeef/test_seeds" not in node_types
    # The real schema node type + the seed-block placeholder remain.
    assert "products" in node_types
    assert "train_table" in node_types
    # Every node type that carries a TF (what the encoder iterates) is encodable:
    # a real schema type or the cleared seed block, never a seed namespace.
    tf_types = {nt for nt in hetero.node_types if "tf" in hetero[nt]}
    assert tf_types <= {"products", "train_table"}


def test_legacy_path_without_feature_schema_keeps_edge_endpoint_backfill(
) -> None:
    """Control: with no feature_schema (legacy / non-TT-NG path) the back-fill
    behavior is byte-identical — edge-endpoint node types are still created."""
    batch = _ttng_nodepred_batch_with_seed_supervision_relation()
    # No mapper_bank/feature_schema -> allowed_node_types stays None.
    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        seed_node_type="products",
        task_type="binary_classification",
    )
    node_types = set(hetero.node_types)
    # Legacy back-fill DOES create the edge-endpoint blocks (unchanged behavior).
    assert "gen-traintable-job-deadbeef/train_seeds" in node_types
    assert "gen-traintable-job-deadbeef/test_seeds" in node_types


def test_attach_pkeys_backfills_empty_unreached_blocks_with_empty_pkey(
) -> None:
    """The XAI SubgraphCollector reads ``batch[t].pkey`` for EVERY graph table.
    A TT-NG seed only materializes features (hence pkeys) for the tables it
    reaches; unreached tables are empty back-fill blocks. Those must get an
    EMPTY pkey array (NOT a fabricated identifier) so the collector builds an
    empty per-table frame instead of crashing on a missing attribute.
    """
    from kumo_core.diskgraph.runtime_adapter import _attach_pkeys_and_times

    hetero = HeteroData()
    # Reached table: has a materialized feature record batch with its pkey col.
    hetero["trans"].n_id = torch.tensor([10, 11], dtype=torch.long)
    # Unreached table: empty back-fill block (n_id empty).
    hetero["loan"].n_id = torch.tensor([], dtype=torch.long)
    # A table with no pkey column in the schema is left untouched.
    hetero["train_table"].n_id = torch.tensor([0], dtype=torch.long)

    raw_feature_tables = {
        "trans": pa.RecordBatch.from_pydict({"TRANS_ID": pa.array(["a", "b"])}),
    }
    pk_col_dict = {"trans": "TRANS_ID", "loan": "LOAN_ID"}

    _attach_pkeys_and_times(hetero, raw_feature_tables, pk_col_dict, {})

    # Reached table gets real pkeys from the record batch.
    assert list(hetero["trans"].pkey) == ["a", "b"]
    # Unreached (empty) table gets an EMPTY pkey array, not fabricated ids.
    assert "pkey" in hetero["loan"]
    assert len(hetero["loan"].pkey) == 0
    assert hetero["loan"].pkey.dtype == object
    # A table with no pkey column in ``pk_col_dict`` is left alone.
    assert "pkey" not in hetero["train_table"]


def test_attach_pkeys_does_not_fabricate_for_nonempty_unreached_block() -> None:
    """A non-empty block lacking a materialized pkey is a real gap; the adapter
    must NOT invent identifiers for it (leave it to surface loudly)."""
    from kumo_core.diskgraph.runtime_adapter import _attach_pkeys_and_times

    hetero = HeteroData()
    hetero["loan"].n_id = torch.tensor([5, 6, 7], dtype=torch.long)
    pk_col_dict = {"loan": "LOAN_ID"}

    _attach_pkeys_and_times(hetero, {}, pk_col_dict, {})

    assert "pkey" not in hetero["loan"]


# ---------------------------------------------------------------------------
# Feature projection pushdown: what the adapter sees when the engine narrows
# the RecordBatch to the plan's `feature_columns`.
#
# The whole safety argument is equivalence: for the SAME sampled batch, a
# projected RecordBatch and a select-all RecordBatch must produce identical
# `.tf`, `.time` and `.pkey`. The dangerous direction is the projected one
# silently losing a column the adapter reads by name, because every such read
# is guarded by `if <col> in column_names` and therefore fails SILENTLY.
# ---------------------------------------------------------------------------

_PROJECTION_STORED_COLUMNS = [
    "txn_id",  # primary key    -> hetero[ns].pkey
    "amount",  # encoder feature -> hetero[ns].tf
    "created_at",  # node time  -> hetero[ns].time  (NOT an encoder column)
    "user_id",  # foreign key    -> dead weight, projected away
    "receipt_url",  # ImageUrl   -> dead weight, projected away
]


def _txn_record_batch(columns: list[str]) -> pa.RecordBatch:
    """A `Transaction` feature batch restricted to `columns`, simulating what
    the engine emits under a given projection."""
    all_columns = {
        "txn_id": pa.array([9001, 9002, 9003, 9004], type=pa.int64()),
        "amount": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
        "created_at": pa.array(
            [1_700_000_000, 1_700_000_100, 1_700_000_200, 1_700_000_300],
            type=pa.int64(),
        ),
        "user_id": pa.array([1, 2, 3, 4], type=pa.int64()),
        "receipt_url": pa.array(["a", "b", "c", "d"], type=pa.string()),
    }
    return pa.RecordBatch.from_arrays([all_columns[c] for c in columns], names=columns)


def _txn_batch(columns: list[str]) -> dict[str, Any]:
    batch = _populated_batch()
    batch["graphs"]["main"]["node_ids"]["Transaction"] = pa.array(
        [200, 201, 202, 203], type=pa.uint64()
    )
    batch["graphs"]["main"]["batch_dict"]["Transaction"] = pa.array(
        [0, 1, 2, 0], type=pa.uint32()
    )
    batch["graphs"]["main"]["num_sampled_nodes_dict"]["Transaction"] = pa.array(
        [3, 1], type=pa.uint64()
    )
    batch["raw_features"]["Transaction"] = {
        "key": "Transaction",
        "num_rows": 4,
        "columns": columns,
        "metadata": {},
        "node_ids": pa.array([200, 201, 202, 203], type=pa.uint64()),
        "record_batches": [_txn_record_batch(columns)],
    }
    return batch


def _txn_feature_schema() -> dict[str, Any]:
    schema = _user_feature_schema()
    schema["node_types"]["Transaction"] = {
        # `created_at` is the time column and is deliberately absent here:
        # pquery excludes the time column from the encoder feature set.
        "feature_columns": ["amount"],
        "time_column": "created_at",
        "pkey_column": "txn_id",
        "mapper_key": "txns_v1",
    }
    return schema


def _txn_mapper_bank() -> dict[str, dict[str, TensorMapper]]:
    bank = _user_mapper_bank()
    bank["txns_v1"] = {
        "amount": _numerical_mapper([10.0, 20.0, 30.0, 40.0]),
    }
    return bank


def _to_hetero_with_txns(
    columns: list[str], *, attach_pkeys: bool = True
) -> HeteroData:
    hetero, _, _ = runtime_batch_to_hetero(
        _txn_batch(columns),
        train_table_name="train_table",
        mapper_bank=_txn_mapper_bank(),
        feature_schema=_txn_feature_schema(),
        pk_col_dict={"Transaction": "txn_id"},
        time_col_dict={"Transaction": "created_at"},
        attach_pkeys=attach_pkeys,
    )
    return hetero


def _link_batch() -> dict[str, Any]:
    """A batch where `Ownership` is SAMPLED but emits NO raw_features block.

    This is exactly what the engine produces for a type requested as `[]`:
    topology and node ids are present, the feature block is simply absent.
    """
    batch = _populated_batch()
    batch["graphs"]["main"]["node_ids"]["Ownership"] = pa.array(
        [300, 301], type=pa.uint64()
    )
    batch["graphs"]["main"]["batch_dict"]["Ownership"] = pa.array(
        [0, 1], type=pa.uint32()
    )
    batch["graphs"]["main"]["num_sampled_nodes_dict"]["Ownership"] = pa.array(
        [2], type=pa.uint64()
    )
    return batch


def _link_feature_schema() -> dict[str, Any]:
    schema = _user_feature_schema()
    # A pure link table: no encoder columns, no time column. pquery emits
    # exactly this for a join table.
    schema["node_types"]["Ownership"] = {
        "feature_columns": [],
        "time_column": None,
        "pkey_column": "ownership_id",
        "mapper_key": "links_v1",
    }
    return schema


def test_skipped_namespace_still_gets_a_tensor_frame() -> None:
    """A type requested as `[]` emits NO raw_features block, and the adapter
    has to survive that.

    `attach_tensor_frames`' back-fill pass builds a mapper-shaped empty TF for
    a schema type with no RecordBatch. For a link table the mapper bank entry
    is `{}` (no encoders were fitted), so this lands on the same
    `get_tf(empty, {})` call the select-all path already reached through
    `_record_batch_to_tf`'s no-overlap branch -- the two routes must agree,
    because that equivalence is the entire safety argument for skipping.
    """
    hetero, raw_features, _ = runtime_batch_to_hetero(
        _link_batch(),
        train_table_name="train_table",
        mapper_bank={
            **_user_mapper_bank(),
            "links_v1": {},
        },
        feature_schema=_link_feature_schema(),
    )

    assert "Ownership" not in raw_features, (
        "fixture must model a SKIPPED type: no raw_features block at all"
    )
    assert "Ownership" in hetero.node_types, (
        "skipping features must not remove the type from the graph"
    )
    assert "tf" in hetero["Ownership"], (
        "the back-fill pass must still attach a TensorFrame, or the feature "
        "encoder KeyErrors on a type it expects to encode"
    )
    assert len(hetero["Ownership"].tf.feat_dict) == 0, (
        "a link table has no encoders, so its TF carries no stypes"
    )


def test_skipping_a_namespace_leaves_topology_untouched() -> None:
    """`[]` declines a type's FEATURES, never its participation in the graph.

    The node ids and edges for a skipped type must be identical to what the
    select-all path produces -- edges come from the index's relation segments,
    which no projection can reach.
    """
    hetero, _, _ = runtime_batch_to_hetero(
        _link_batch(),
        train_table_name="train_table",
        mapper_bank={
            **_user_mapper_bank(),
            "links_v1": {},
        },
        feature_schema=_link_feature_schema(),
    )

    assert hetero["Ownership"].num_nodes == 2
    for edge_type in hetero.edge_types:
        assert hetero[edge_type].edge_index.shape[0] == 2


def test_naive_encoder_only_projection_silently_drops_node_time() -> None:
    """The failure mode this feature has to avoid, pinned as a fact.

    A projection built from `feature_columns` alone omits the node time
    column. `_attach_pkeys_and_times` guards its `.time` attach with
    `if time_col in column_names`, so nothing raises -- the temporal encoder
    just stops receiving `rel_time = node.time - seed_time`. If this test ever
    starts failing because `.time` IS present, the guard changed and the
    projection contract should be revisited.
    """
    hetero = _to_hetero_with_txns(["amount"])

    assert "time" not in hetero["Transaction"], (
        "the encoder-only projection is expected to lose `.time` silently; "
        "that is precisely why the time column must be requested explicitly"
    )


def test_projection_including_time_and_pkey_preserves_them() -> None:
    """The needed set: feature_columns + time_column + primary key."""
    hetero = _to_hetero_with_txns(["amount", "created_at", "txn_id"])

    block = hetero["Transaction"]
    assert "time" in block, "the node time column must survive projection"
    assert torch.equal(
        block.time,
        torch.tensor(
            [
                1_700_000_000_000_000_000,
                1_700_000_100_000_000_000,
                1_700_000_200_000_000_000,
                1_700_000_300_000_000_000,
            ],
            dtype=torch.long,
        ),
    )
    assert block.pkey.tolist() == [9001, 9002, 9003, 9004]


def test_every_non_empty_namespace_keeps_time_after_projection() -> None:
    """Sweep: every namespace that has rows and a known time column must come
    out of the projected path with `.time` attached."""
    hetero = _to_hetero_with_txns(["amount", "created_at", "txn_id"])
    time_col_dict = {"Transaction": "created_at"}

    checked = 0
    for namespace in hetero.node_types:
        if time_col_dict.get(namespace) is None:
            continue
        block = hetero[namespace]
        n_id = getattr(block, "n_id", None)
        if n_id is None or int(n_id.numel()) == 0:
            continue
        assert "time" in block, (
            f"namespace {namespace} is non-empty but lost `.time` under projection"
        )
        checked += 1
    assert checked, "the sweep must actually inspect a namespace"


@pytest.mark.parametrize("attach_pkeys", [True, False])
def test_projected_batch_equals_select_all(attach_pkeys: bool) -> None:
    """EQUIVALENCE: same seeds, same plan, projected vs select-all.

    Every retained artifact -- the TensorFrame, `.time`, `.pkey` -- must be
    identical. Only the columns the model never reads may differ.
    """
    needed = ["amount", "created_at"]
    if attach_pkeys:
        needed.append("txn_id")

    select_all = _to_hetero_with_txns(
        _PROJECTION_STORED_COLUMNS, attach_pkeys=attach_pkeys
    )
    projected = _to_hetero_with_txns(needed, attach_pkeys=attach_pkeys)

    assert set(select_all.node_types) == set(projected.node_types)
    for namespace in select_all.node_types:
        lhs, rhs = select_all[namespace], projected[namespace]

        assert ("tf" in lhs) == ("tf" in rhs)
        if "tf" in lhs:
            assert lhs.tf.col_names_dict == rhs.tf.col_names_dict, namespace
            assert lhs.tf.num_rows == rhs.tf.num_rows, namespace
            for stype, feat in lhs.tf.feat_dict.items():
                other = rhs.tf.feat_dict[stype]
                if isinstance(feat, MultiNestedTensor):
                    assert torch.equal(feat.values, other.values), namespace
                else:
                    assert torch.equal(feat, other), namespace

        assert ("time" in lhs) == ("time" in rhs), namespace
        if "time" in lhs:
            assert torch.equal(lhs.time, rhs.time), namespace

        assert hasattr(lhs, "pkey") == hasattr(rhs, "pkey"), namespace
        if hasattr(lhs, "pkey"):
            assert lhs.pkey.tolist() == rhs.pkey.tolist(), namespace

        assert ("n_id" in lhs) == ("n_id" in rhs), namespace
        if "n_id" in lhs:
            assert torch.equal(lhs.n_id, rhs.n_id), namespace


def test_forecasting_alias_matching_survives_a_narrowed_batch() -> None:
    """`_alias_forecasting_seed_features` matches the seed namespace by
    intersecting the schema's `feature_columns` with `record_batch.column_names`.

    Narrowing a GRAPH namespace must not disturb that: the candidate set is
    restricted to namespaces absent from `feature_schema["node_types"]`, and
    the seed namespace rides the FULL seed rowpack (`fetch_seed_rowpack_full`),
    which projection never touches.
    """
    future = pa.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], type=pa.list_(pa.float64())
    )
    feature_schema = _txn_feature_schema()
    feature_schema["node_types"]["train_table"] = {
        "feature_columns": [FUTURE_LABELS_COL],
        "mapper_key": "train_table",
    }
    mapper_bank = _txn_mapper_bank()
    mapper_bank["train_table"] = {
        FUTURE_LABELS_COL: SequenceTensorMapper(min_length=3, max_length=3),
    }

    for columns in (_PROJECTION_STORED_COLUMNS, ["amount", "created_at", "txn_id"]):
        batch = _txn_batch(columns)
        batch["raw_features"]["train_seeds"] = {
            "key": "train_seeds",
            "num_rows": 3,
            "columns": [FUTURE_LABELS_COL],
            "metadata": {},
            "record_batches": [
                pa.RecordBatch.from_arrays([future], names=[FUTURE_LABELS_COL])
            ],
        }

        hetero, _, _ = runtime_batch_to_hetero(
            batch,
            train_table_name="train_table",
            mapper_bank=mapper_bank,
            feature_schema=feature_schema,
            task_type="forecasting",
            pk_col_dict={"Transaction": "txn_id"},
            time_col_dict={"Transaction": "created_at"},
        )

        tf = hetero["train_table"].tf
        assert tf.col_names_dict["sequence"] == [FUTURE_LABELS_COL], columns
        assert torch.equal(
            tf.feat_dict["sequence"].values,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )
