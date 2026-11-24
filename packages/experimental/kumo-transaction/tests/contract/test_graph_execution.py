# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU validation, mutation boundaries, and value-preserving graph transfers."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pytest
import torch

from tfm.compositions.gnn import GNNEncoder
from tfm.contracts.graph import (
    STATIC_TIME,
    FutureSubgraphTargets,
    GraphAnchor,
    GraphBatch,
    GraphSupervisedBatch,
    GraphTrainingBatch,
    utc_us,
)
from tfm.data.graph import GraphLink, GraphRow, GraphSampler, GraphTable, RelationalGraph

T = datetime(2024, 1, 10, tzinfo=UTC)
ROLE = ("users", "events", "events")
REVERSE = ("events", "rev_events", "users")
DAY = 86_400_000_000


@pytest.fixture
def batch():
    anchors = tuple(
        GraphAnchor(str(i), str(i), T + timedelta(days=i), T + timedelta(days=i - 2))
        for i in range(4)
    )
    cutoffs = torch.tensor([utc_us(a.cutoff) for a in anchors])
    edges = torch.stack((torch.arange(4).repeat_interleave(2), torch.arange(8)))
    times = cutoffs.repeat_interleave(2) - torch.tensor([2 * DAY, 0] * 4)
    return GraphBatch(
        anchors,
        "users",
        {
            "users": torch.ones(4, 1),
            "events": torch.randn(8, 1, generator=torch.Generator().manual_seed(9)),
        },
        {"users": torch.arange(4), "events": torch.arange(4).repeat_interleave(2)},
        {"users": torch.full((4,), STATIC_TIME), "events": times},
        {ROLE: edges, REVERSE: edges.flip(0)},
        {ROLE: times, REVERSE: times},
        torch.arange(4),
    )


@pytest.mark.parametrize("anchor", [0, 3])
@pytest.mark.parametrize(
    "kind", ["node_future", "node_past", "edge_future", "owner", "cross_edge", "duplicate", "root"]
)
def test_vectorized_checks_reject_bad_graphs_at_every_owner(batch, anchor, kind):
    index = 2 * anchor
    if kind in {"node_future", "node_past"}:
        times = batch.node_available_us["events"].clone()
        times[index] = (
            utc_us(batch.anchors[anchor].cutoff) + 1
            if kind == "node_future"
            else utc_us(batch.anchors[anchor].observation_start) - 1
        )
        with pytest.raises(ValueError, match="Future node|predates"):
            replace(batch, node_available_us={**batch.node_available_us, "events": times})
    elif kind == "edge_future":
        times = batch.edge_available_us[ROLE].clone()
        times[index] = utc_us(batch.anchors[anchor].cutoff) + 1
        with pytest.raises(ValueError, match="Future edge"):
            replace(batch, edge_available_us={**batch.edge_available_us, ROLE: times})
    elif kind == "owner":
        owners = batch.node_sample["events"].clone()
        owners[index] = -1 if anchor == 0 else 4
        with pytest.raises(ValueError, match="owner outside"):
            replace(batch, node_sample={**batch.node_sample, "events": owners})
    elif kind in {"cross_edge", "duplicate"}:
        edges = batch.edge_index[ROLE].clone()
        edges[:, index] = (
            edges[:, index + 1] if kind == "duplicate" else torch.tensor([(anchor + 1) % 4, index])
        )
        with pytest.raises(ValueError, match="crosses|Duplicate"):
            replace(batch, edge_index={**batch.edge_index, ROLE: edges})
    else:
        roots = batch.root_index.clone()
        roots[anchor] = 100
        with pytest.raises(ValueError, match="bounds"):
            replace(batch, root_index=roots)


def targets(batch):
    return FutureSubgraphTargets(
        batch.anchors,
        torch.ones(4, 1),
        torch.ones(4, 1, dtype=torch.bool),
        ("x",),
        DAY,
        DAY,
        torch.tensor([utc_us(a.cutoff) + 2 * DAY for a in batch.anchors]),
        utc_us(T + timedelta(days=8)),
    )


@pytest.mark.parametrize("index", [0, 3])
def test_vectorized_future_bounds_still_reject_incomplete_and_unavailable(batch, index):
    target = targets(batch)
    for value in (utc_us(target.anchors[index].cutoff) + 2 * DAY - 1, target.partition_end_us + 1):
        clocks = target.available_us.clone()
        clocks[index] = value
        with pytest.raises(ValueError, match="Incomplete future horizon"):
            replace(target, available_us=clocks)


@pytest.mark.parametrize(
    "field",
    [
        "node_features",
        "node_sample",
        "node_available_us",
        "edge_index",
        "edge_available_us",
        "root_index",
    ],
)
def test_tracked_mutation_invalidates_transfer(batch, field):
    value = getattr(batch, field)
    tensor = value if isinstance(value, torch.Tensor) else next(iter(value.values()))
    tensor.add_(0)  # Even value-preserving in-place operations invalidate the record.
    with pytest.raises(ValueError, match="mutated"):
        batch.to(torch.device("cpu"))


def test_mappings_are_snapshots_and_same_device_aliases_are_read_only(batch):
    with pytest.raises(TypeError):
        batch.node_features["new"] = torch.ones(1)
    copied = batch.to(torch.device("cpu"))
    copied.node_features["users"].add_(1)
    for invalid in (batch, copied):
        with pytest.raises(ValueError, match="mutated"):
            invalid.assert_validated()
    rebuilt = replace(batch, node_features={k: v.clone() for k, v in batch.node_features.items()})
    rebuilt.assert_validated()  # Normal CPU reconstruction revalidates intentional changes.


def test_transfer_never_calls_constructors_or_tensor_predicates(batch, monkeypatch):
    target = targets(batch)
    supervised = GraphSupervisedBatch(
        batch, torch.tensor([0.0, 1.0, 0.0, 1.0]), target.available_us, target.partition_end_us
    )
    training = GraphTrainingBatch(batch, target)

    def forbidden(*args, **kwargs):
        raise AssertionError("transfer reran tensor validation")

    with monkeypatch.context() as m:
        for cls in (GraphBatch, GraphSupervisedBatch, GraphTrainingBatch, FutureSubgraphTargets):
            m.setattr(cls, "__post_init__", forbidden)
        m.setattr(torch.Tensor, "__bool__", forbidden)
        m.setattr(torch.Tensor, "unique", forbidden)
        copied = [
            b.to(torch.device("cpu"), non_blocking=True) for b in (batch, supervised, training)
        ]
        assert copied[2].has_supervision
        assert copied[2].targets.valid_count == 4
    assert copied[0].single_parent_relations == frozenset({ROLE})
    for name in batch.node_features:
        torch.testing.assert_close(copied[0].node_features[name], batch.node_features[name])


def test_nested_label_and_target_mutations_are_rejected(batch):
    target = targets(batch)
    training = GraphTrainingBatch(batch, target)
    target.values.add_(1)
    with pytest.raises(ValueError, match="mutated"):
        training.to(torch.device("cpu"))
    supervised = GraphSupervisedBatch(batch, torch.zeros(4), torch.zeros(4, dtype=torch.long), 0)
    supervised.labels.requires_grad_()
    with pytest.raises(ValueError, match="mutated"):
        supervised.to(torch.device("cpu"))


def model_inputs(sharing, trim):
    rows = tuple(GraphRow("users", str(i), {"x": float(i + 1)}) for i in range(4))
    events = tuple(GraphRow("events", str(i), {"x": float(i - 4)}, T, T) for i in range(8))
    graph = RelationalGraph(
        (GraphTable("users", ("x",), immutable=True), GraphTable("events", ("x",))),
        (ROLE,),
        (*rows, *events),
        tuple(GraphLink(ROLE, str(i // 2), str(i), T) for i in range(8)),
        frozenset({"label"}),
        T,
    )
    anchors = tuple(GraphAnchor(str(i), str(i), T) for i in range(4))
    batch = GraphSampler(graph, depth=2, fanout=10).sample(
        anchors, "users", sharing=sharing, hop_trim=trim
    )
    spec = GNNEncoder(
        (("users", 1), ("events", 1)),
        (ROLE, REVERSE),
        "users",
        (REVERSE,),
        hidden_size=8,
        depth=2,
        trim=trim,
    )
    return batch, spec


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("sharing", ["shared", "disjoint"])
@pytest.mark.parametrize("trim", [True, False])
def test_outputs_and_gradients_match_direct_engine_without_forward_uniqueness(
    sharing, trim, device, monkeypatch
):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.layers.gnn import GraphEncoder

    torch.manual_seed(81)
    batch, spec = model_inputs(sharing, trim)
    actual, expected = GraphEncoder(spec).to(device).train(), GraphEncoder(spec).to(device).train()
    expected.load_state_dict(actual.state_dict())
    copied = batch.for_execution(torch.device(device))
    batch = copied
    x = {
        name: layer(batch.node_features[name])
        for name, layer in zip(expected.types, expected.input, strict=True)
    }
    edges = {expected.engine_roles[r]: batch.edge_index[r] for r in expected.engine_roles}
    kwargs = (
        {}
        if not trim
        else {
            "num_sampled_nodes_dict": {k: list(v) for k, v in batch.num_sampled_nodes.items()},
            "num_sampled_edges_dict": {
                expected.engine_roles[k]: list(v) for k, v in batch.num_sampled_edges.items()
            },
        }
    )
    reference = expected.gnn(x, edges, **kwargs)[batch.root_type][batch.root_index]

    def forbidden(*args, **kwargs):
        raise AssertionError("model forward repeated FK uniqueness")

    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "unique", forbidden)
        result = actual(copied)
    torch.testing.assert_close(
        result,
        reference,
        rtol=1e-5 if device == "cuda" else 0,
        atol=1e-6 if device == "cuda" else 0,
    )
    weights = torch.arange(result.numel(), dtype=result.dtype, device=result.device).reshape_as(
        result
    )
    (result * weights).sum().backward()
    (reference * weights).sum().backward()
    for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(
            left.grad,
            right.grad,
            rtol=1e-4 if device == "cuda" else 0,
            atol=1e-5 if device == "cuda" else 0,
        )
    for left, right in zip(actual.buffers(), expected.buffers(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    copied.node_features["users"].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        actual(copied)


def test_inference_tensors_cannot_bypass_mutation_tracking(batch):
    with torch.inference_mode():
        values = torch.ones(4, 1)
    with pytest.raises(ValueError, match="versioned tensors.*clone inference inputs"):
        replace(batch, node_features={**batch.node_features, "users": values})
    with torch.inference_mode():
        with torch.inference_mode(False), torch.no_grad():
            normal = values.clone()
        rebuilt = replace(batch, node_features={**batch.node_features, "users": normal})
        rebuilt.assert_validated()
    assert not normal.is_inference()


def test_history_transfer_and_attribute_mutation_are_guarded(monkeypatch):
    from pathlib import Path

    import yaml

    from tfm.config import RecipeConfig
    from tfm.contracts.graph import GraphHistoryBatch
    from tfm.data.graph_recipe import prepare_graph_data

    config = RecipeConfig.model_validate(
        yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    )
    batch = prepare_graph_data(config).probes["train"][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("history transfer reran validation")

    with monkeypatch.context() as m:
        m.setattr(GraphHistoryBatch, "__post_init__", forbidden)
        m.setattr(torch.Tensor, "__bool__", forbidden)
        copied = batch.to(torch.device("cpu"))
    copied.assert_validated()
    batch.history.positions = batch.history.positions.clone()
    with pytest.raises(ValueError, match="mutated"):
        batch.to(torch.device("cpu"))
    copied.assert_validated()  # Metadata/container ownership is independent.
    copied.history.token_mask.logical_not_()
    with pytest.raises(ValueError, match="mutated"):
        copied.assert_validated()


def test_direct_device_construction_is_rejected_without_reading_tensor_values(batch):
    with pytest.raises(ValueError, match="on CPU"):
        replace(batch, root_index=torch.empty(4, dtype=torch.long, device="meta"))


def test_cpu_serialization_revalidates_new_tensor_identities(batch):
    import pickle

    original = GraphTrainingBatch(batch, targets(batch))
    restored = pickle.loads(pickle.dumps(original))
    restored.assert_validated()
    restored.graph.assert_validated()
    assert restored.targets.valid_count == original.targets.valid_count
    torch.testing.assert_close(
        restored.graph.node_features["events"], batch.node_features["events"]
    )
    with pytest.raises(TypeError):
        restored.graph.edge_index[ROLE] = torch.empty(2, 0, dtype=torch.long)
    batch.node_features["users"].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        pickle.dumps(original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("method", ["to", "for_execution"])
def test_gpu_transfer_has_no_tensor_validation_or_scalar_reads(batch, inference, method):
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode(inference), profile(activities=[ProfilerActivity.CPU]) as trace:
        transferred = getattr(batch, method)(torch.device("cuda"))
    torch.cuda.synchronize()
    transferred.assert_validated()
    names = {event.key for event in trace.key_averages()}
    assert not names.intersection(
        {
            "aten::_local_scalar_dense",
            "aten::item",
            "aten::_unique2",
            "aten::unique_dim",
            "aten::all",
            "aten::any",
            "aten::min",
            "aten::max",
            "aten::sum",
            "aten::nonzero",
        }
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_all_validated_batch_transfers_work_inside_inference_mode(batch, device):
    from pathlib import Path

    import yaml

    from tfm.config import RecipeConfig
    from tfm.data.graph_recipe import prepare_graph_data

    config = RecipeConfig.model_validate(
        yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    )
    history = prepare_graph_data(config).probes["train"][0]
    target = targets(batch)
    supervised = GraphSupervisedBatch(batch, torch.zeros(4), torch.zeros(4, dtype=torch.long), 0)
    for original in (batch, target, supervised, GraphTrainingBatch(batch, target), history):
        with torch.inference_mode():
            copied = original.to(torch.device(device), non_blocking=True)
            copied.assert_validated()  # Every reachable tensor must retain its version counter.
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        assert torch.is_grad_enabled()
    copied.history.token_mask.logical_not_()
    with torch.inference_mode(), pytest.raises(ValueError, match="mutated"):
        copied.to(torch.device(device))
    assert not torch.is_inference_mode_enabled() and torch.is_grad_enabled()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("sharing", ["shared", "disjoint"])
def test_sampling_transfer_and_frozen_encoder_inside_inference_loop(device, sharing, monkeypatch):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.layers.gnn import GraphEncoder

    torch.manual_seed(41)
    expected_batch, spec = model_inputs(sharing, True)
    encoder = GraphEncoder(spec).to(device).eval().requires_grad_(False)
    with torch.no_grad():
        expected = encoder(expected_batch.to(torch.device(device)))
    original = GraphBatch.__post_init__
    observed = []

    def validate_under_local_context(self):
        observed.append(True)
        assert not torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        original(self)

    monkeypatch.setattr(GraphBatch, "__post_init__", validate_under_local_context)
    with torch.inference_mode():
        actual_batch, _ = model_inputs(sharing, True)
        copied = actual_batch.to(torch.device(device))
        result = encoder(copied)
        assert torch.is_inference_mode_enabled()
        copied.node_features["users"].add_(0)
        with pytest.raises(ValueError, match="mutated"):
            encoder(copied)
    assert observed
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    torch.testing.assert_close(result, expected)
    assert all(p.grad is None for p in encoder.parameters())


@pytest.mark.parametrize("mode", ["binary", "future_subgraph"])
def test_recipe_preparation_inside_inference_mode_returns_versioned_batches(mode):
    from pathlib import Path

    import yaml

    from tfm.config import RecipeConfig
    from tfm.data.graph_recipe import prepare_graph_data

    variant = "config.smoke.yaml" if mode == "binary" else "config.pretrain_smoke.yaml"
    config = RecipeConfig.model_validate(
        yaml.safe_load((Path("recipes/gnn_sequence") / variant).read_text())
    )
    with torch.inference_mode():
        prepared = prepare_graph_data(config)
        for b in prepared.pretrain + prepared.validation + prepared.hybrid_train:
            b.assert_validated()
            b.to(torch.device("cpu")).assert_validated()
        assert torch.is_inference_mode_enabled()
        if prepared.transform is not None:
            assert not prepared.transform.mean.is_inference()
            assert not prepared.transform.scale.is_inference()
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("size", [0, 1, 2, 31, 256])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_packed_duplicate_edges_match_pair_oracle(size, strided, overflow, monkeypatch):
    from tfm.contracts.graph import _has_duplicate_edges

    generator = torch.Generator().manual_seed(9 + size)
    for source_size, target_size in [(1, 1), (3, 17), (41, 7)]:
        index = torch.stack(
            [
                torch.randint(source_size, (size * 2,), generator=generator),
                torch.randint(target_size, (size * 2,), generator=generator),
            ]
        )
        index = index[:, ::2] if strided else index[:, :size].contiguous()
        before = index.clone()
        expected = len(set(zip(index[0].tolist(), index[1].tolist(), strict=True))) != size
        domain = (
            (2**62 + source_size, 2**62 + target_size) if overflow else (source_size, target_size)
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("duplicate detection used generic unique")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "unique", forbidden)
            patch.setattr(torch.Tensor, "unique", forbidden)
            assert _has_duplicate_edges(index, *domain) == expected
        torch.testing.assert_close(index, before, atol=0, rtol=0)


def test_packed_duplicate_edges_avoid_integer_overflow():
    from tfm.contracts.graph import _has_duplicate_edges

    # The distinct pairs would collide if packed with overflowing int64 math.
    index = torch.tensor([[0, 2**62], [0, 0]])
    assert not _has_duplicate_edges(index, 2**62 + 1, 4)
    assert _has_duplicate_edges(torch.cat([index, index[:, :1]], dim=1), 2**62 + 1, 4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_execution_views_keep_cpu_proof_and_guard_both_sources(batch, device):
    import weakref

    training = GraphTrainingBatch(batch, targets(batch))
    with torch.inference_mode():
        view = training.for_execution(torch.device(device))
        assert torch.is_inference_mode_enabled()
    assert view.graph.root_index.device.type == device
    assert view.targets.values.device.type == device
    assert view.targets.valid.device.type == device
    assert view.graph.node_sample is batch.node_sample
    assert view.graph.node_available_us is batch.node_available_us
    assert view.graph.edge_available_us is batch.edge_available_us
    assert view.targets.available_us is training.targets.available_us
    assert view.for_execution(torch.device(device)) is view
    view.assert_validated()
    ref, graph_ref = weakref.ref(view), weakref.ref(view.graph)
    del view
    assert ref() is None and graph_ref() is None  # No GC cycle retaining GPU memory.
    view = training.for_execution(torch.device(device))
    batch.node_available_us["events"].add_(0)
    with pytest.raises(ValueError, match="mutated"):
        view.assert_validated()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("location", ["source", "device"])
def test_execution_views_detect_mutation_of_copied_inputs(batch, location):
    view = batch.for_execution(torch.device("cuda"))
    (batch if location == "source" else view).node_features["users"].add_(0)
    with pytest.raises(ValueError, match="mutated"):
        view.assert_validated()


def test_execution_and_full_transfer_use_compiled_plans(batch, monkeypatch):
    from tfm.contracts import validation as _validated

    original = GraphTrainingBatch(batch, targets(batch))

    def forbidden(*args, **kwargs):
        raise AssertionError("hot path rebuilt validation layout")

    monkeypatch.setattr(_validated, "fields", forbidden)
    monkeypatch.setattr(_validated._Layout, "build", forbidden)
    monkeypatch.setattr(_validated.ValidatedBatch, "_seal", forbidden)
    for _ in range(3):
        original.assert_validated()
        original.to(torch.device("cpu")).assert_validated()
        original.for_execution(torch.device("cpu")).assert_validated()


def test_history_execution_projection_is_cached_and_mutations_are_guarded():
    from pathlib import Path

    import yaml

    from tfm.config import RecipeConfig
    from tfm.contracts.graph import SequenceSupervisedBatch
    from tfm.data.graph_recipe import prepare_graph_data

    config = RecipeConfig.model_validate(
        yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    )
    original = prepare_graph_data(config).probes["train"][0]
    projection = SequenceSupervisedBatch.from_graph_history(original)
    assert SequenceSupervisedBatch.from_graph_history(original) is projection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    view = projection.for_execution(device)
    assert view.history.key_ids.device.type == device.type
    assert view.history.numeric_features is original.history.numeric_features
    assert view.history.fraud_labels is original.history.fraud_labels
    assert view.label_available_us is projection.label_available_us
    original.history.value_ids = original.history.value_ids.clone()
    for value in (original, projection, view):
        with pytest.raises(ValueError, match="mutated"):
            value.assert_validated()
    with pytest.raises(ValueError, match="mutated"):
        SequenceSupervisedBatch.from_graph_history(original)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("arm", ["future_subgraph", "hybrid", "binary_sequence"])
def test_full_copy_and_execution_view_have_matching_training_results(
    device, arm, monkeypatch, deterministic_graph_execution
):
    import copy
    from pathlib import Path

    import yaml

    from tfm.api.graph import graph_specs
    from tfm.compositions.gnn import GNNSequence
    from tfm.config import RecipeConfig
    from tfm.data.graph_recipe import prepare_graph_data
    from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel, SequenceBinaryModel

    if arm != "binary_sequence":
        pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    else:

        def forbidden(*args, **kwargs):
            raise AssertionError("Sequence-only control must not construct a graph encoder")

        monkeypatch.setattr(GNNEncoderModel, "__init__", forbidden)
    config = RecipeConfig.model_validate(
        yaml.safe_load(Path("recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text())
    )
    data = prepare_graph_data(config)
    graph, sequence = graph_specs(config, data)
    torch.manual_seed(74)
    if arm == "future_subgraph":
        encoder = GNNEncoderModel(graph, data.pretrain[0].targets.columns)
        original, model = data.pretrain[0], encoder
    elif arm == "hybrid":
        encoder = GNNEncoderModel(graph, data.pretrain[0].targets.columns)
        original = data.hybrid_train[0]
        model = GNNSequenceModel(GNNSequence(graph, sequence), encoder.encoder)
    else:
        original = data.sequence_supervised_train[0]
        model = SequenceBinaryModel(sequence)
    expected = copy.deepcopy(model).to(device).train()
    actual = model.to(device).train()
    left = expected(original.to(torch.device(device)))
    right = actual(original.for_execution(torch.device(device)))
    torch.testing.assert_close(
        left.embeddings,
        right.embeddings,
        rtol=1e-5 if device == "cuda" else 0,
        atol=1e-6 if device == "cuda" else 0,
    )
    torch.testing.assert_close(
        left.loss,
        right.loss,
        rtol=1e-5 if device == "cuda" else 0,
        atol=1e-6 if device == "cuda" else 0,
    )
    left.loss.backward()
    right.loss.backward()
    for x, y in zip(expected.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(
            x.grad,
            y.grad,
            rtol=1e-4 if device == "cuda" else 0,
            atol=1e-5 if device == "cuda" else 0,
        )
    for x, y in zip(expected.buffers(), actual.buffers(), strict=True):
        torch.testing.assert_close(
            x, y, rtol=1e-5 if device == "cuda" else 0, atol=1e-6 if device == "cuda" else 0
        )


@pytest.mark.parametrize("kind", ["tensor", "clock"])
def test_execution_records_reject_frozen_field_replacement(batch, kind):
    view = batch.for_execution(torch.device("cpu"))
    if kind == "tensor":
        object.__setattr__(batch, "root_index", batch.root_index.clone())
    else:
        object.__setattr__(batch.anchors[0], "cutoff", T + timedelta(days=30))
    with pytest.raises(ValueError, match="mutated"):
        view.assert_validated()


def test_execution_transfer_visits_only_declared_model_tensors(batch, monkeypatch):
    calls = []
    original = torch.Tensor.to

    def record(tensor, *args, **kwargs):
        calls.append(id(tensor))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record)
    batch.for_execution(torch.device("cpu"))
    expected = {
        id(t) for mapping in (batch.node_features, batch.edge_index) for t in mapping.values()
    } | {id(batch.root_index)}
    assert len(calls) == len(expected)
    assert set(calls) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("inference", [False, True])
def test_off_cpu_execution_sources_normalize_metadata_and_release_cuda_owner(
    batch, nested, device, inference
):
    import weakref

    original = GraphTrainingBatch(batch, targets(batch)) if nested else batch
    cuda_batch = original.to(torch.device("cuda"))
    cuda_graph = cuda_batch.graph if nested else cuda_batch
    owner_ref = weakref.ref(cuda_batch)
    node_clock_ref = weakref.ref(cuda_graph.node_available_us["events"])
    input_ref = weakref.ref(cuda_graph.node_features["events"])
    target_clock_ref = weakref.ref(cuda_batch.targets.available_us) if nested else None
    del cuda_graph
    with torch.inference_mode(inference):
        view = cuda_batch.for_execution(torch.device(device))
        assert torch.is_inference_mode_enabled() == inference
    graph = view.graph if nested else view
    assert graph.root_index.device.type == device
    assert graph.node_available_us["events"].device.type == "cpu"
    assert graph.node_sample["events"].device.type == "cpu"
    assert all(state.cpu for state in view._execution_source._record.tensors)
    if nested:
        assert view.targets.values.device.type == device
        assert view.targets.available_us.device.type == "cpu"
    if device == "cpu":
        assert all(state.cpu for state in view._record.tensors)
    view.assert_validated()
    torch.testing.assert_close(graph.node_features["events"].cpu(), batch.node_features["events"])
    (cuda_batch.graph if nested else cuda_batch).node_available_us["events"].add_(0)
    view.assert_validated()  # Normalization created independent CPU ownership.
    del cuda_batch
    assert owner_ref() is None and node_clock_ref() is None and input_ref() is None
    assert target_clock_ref is None or target_clock_ref() is None
    graph.node_available_us["events"].add_(0)
    with pytest.raises(ValueError, match="mutated"):
        view.assert_validated()


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_duplicate_pair_sort_handles_maximal_int64_coordinates(duplicate, strided):
    from tfm.contracts.graph import _has_duplicate_edges

    limit = torch.iinfo(torch.int64).max
    pairs = torch.tensor([[0, limit, limit, 1, 0], [limit, 0, limit, 1, 0]])
    if duplicate:
        pairs = torch.cat((pairs, pairs[:, 1:2]), dim=1)
    if strided:
        pairs = pairs.repeat_interleave(2, dim=1)[:, ::2]
    before, version = pairs.clone(), pairs._version
    assert _has_duplicate_edges(pairs, limit + 1, limit + 1) == duplicate
    assert pairs._version == version
    torch.testing.assert_close(pairs, before, rtol=0, atol=0)


def test_graph_construction_never_uses_column_wise_uniqueness(batch, monkeypatch):
    original = torch.unique

    def no_columns(tensor, *args, **kwargs):
        if tensor.ndim == 2:
            raise AssertionError("graph conversion used column-wise uniqueness")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "unique", no_columns)
    rebuilt = replace(batch)
    rebuilt.assert_validated()
    edges = batch.edge_index[ROLE].clone()
    edges[:, 0] = edges[:, 1]
    with pytest.raises(ValueError, match="Duplicate graph edges"):
        replace(batch, edge_index={**batch.edge_index, ROLE: edges})


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("size", [0, 1, 2, 9, 256])
def test_combined_edge_classification_matches_previous_checks(size, strided, overflow):
    from tfm.contracts.graph import _edge_properties

    rng = torch.Generator().manual_seed(234 + size)
    for source_count, receiver_count in [(1, 1), (3, 17), (41, 7)]:
        pairs = torch.stack(
            (
                torch.randint(source_count, (2 * size,), generator=rng),
                torch.randint(receiver_count, (2 * size,), generator=rng),
            )
        )
        pairs = pairs[:, ::2] if strided else pairs[:, :size].contiguous()
        before, version = pairs.clone(), pairs._version
        expected = (torch.unique(pairs, dim=1).shape[1] != size, pairs[1].unique().numel() == size)
        domain = (
            (2**62 + source_count, 2**62 + receiver_count)
            if overflow
            else (source_count, receiver_count)
        )
        assert _edge_properties(pairs, *domain) == expected
        assert pairs._version == version
        torch.testing.assert_close(pairs, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "columns,expected",
    [
        ([], (False, True)),
        ([(0, 0)], (False, True)),
        ([(0, 0), (1, 1), (0, 2)], (False, True)),
        ([(0, 0), (1, 0)], (False, False)),
        ([(0, 0), (0, 0)], (True, False)),
        ([(0, 2), (1, 0), (0, 2)], (True, False)),
    ],
)
def test_receiver_classification_uses_one_sort(columns, expected, monkeypatch):
    from tfm.contracts.graph import _edge_properties

    pairs = torch.tensor(columns, dtype=torch.long).reshape(-1, 2).T
    original_sort = torch.sort
    calls = []

    def sort(value, *args, **kwargs):
        calls.append(value.ndim)
        return original_sort(value, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("classification performed another sort/unique")

    monkeypatch.setattr(torch, "sort", sort)
    monkeypatch.setattr(torch, "argsort", forbidden)
    monkeypatch.setattr(torch, "unique", forbidden)
    monkeypatch.setattr(torch.Tensor, "unique", forbidden)
    assert _edge_properties(pairs, 2, 3) == expected
    assert calls == ([] if len(columns) < 2 else [1])


def test_graph_conversion_does_not_sort_receivers_again(batch, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("graph conversion separately sorted receiver IDs")

    monkeypatch.setattr(torch, "unique", forbidden)
    monkeypatch.setattr(torch.Tensor, "unique", forbidden)
    copied = replace(batch)
    assert copied.single_parent_relations == frozenset({ROLE})
    copied.assert_validated()


def test_construction_accumulates_once_and_reuses_nested_records(batch, monkeypatch):
    from tfm.contracts import validation as _validated

    target = targets(batch)
    graph_record, target_record = batch._record, target._record
    finishes = []
    original_finish = _validated._RecordAccumulator.finish

    def finish(accumulator):
        finishes.append(True)
        return original_finish(accumulator)

    def forbidden(*args, **kwargs):
        raise AssertionError("construction recursively merged records")

    monkeypatch.setattr(_validated._Record, "merge", forbidden)
    monkeypatch.setattr(_validated._RecordAccumulator, "finish", finish)
    training = GraphTrainingBatch(batch, target)
    assert finishes == [True]
    assert training.graph._record is graph_record
    assert training.targets._record is target_record
    assert len(training._record.tensors) == len(
        {id(state.value) for state in training._record.tensors}
    )
    rebuilt = replace(batch)
    assert finishes == [True, True]
    training.assert_validated()
    rebuilt.assert_validated()
    target.values.requires_grad_()
    with pytest.raises(ValueError, match="mutated"):
        training.assert_validated()


def test_constant_metadata_guards_survive_full_transfer_after_accumulation(batch):
    copied = batch.to(torch.device("cpu"))
    object.__setattr__(batch.anchors[0], "observation_start", T - timedelta(days=30))
    with pytest.raises(ValueError, match="mutated"):
        copied.assert_validated()


@pytest.mark.parametrize("mutation", ["append", "replace", "key", "gradient"])
def test_accumulated_records_cover_mutable_metadata_after_full_copy(mutation):
    from tfm.contracts.validation import ValidatedBatch, require_cpu

    @dataclass(frozen=True)
    class Sample(ValidatedBatch):
        value: torch.Tensor
        metadata: dict[str, list[int]]
        aliases: list[torch.Tensor]

        def __post_init__(self):
            require_cpu(self)
            self._seal()

    value = torch.zeros(2)
    source = Sample(value, {"owners": [1, 2]}, [value, value])
    copied = source.to(torch.device("cpu"))
    assert len(source._record.tensors) == len(copied._record.tensors) == 1
    assert copied.aliases[0] is copied.aliases[1]
    if mutation == "append":
        source.metadata["owners"].append(3)
    elif mutation == "replace":
        source.metadata["owners"] = [1, 2]
    elif mutation == "key":
        source.metadata["extra"] = []
    else:
        value.requires_grad_()
    for batch in (source, copied):
        with pytest.raises(ValueError, match="mutated"):
            batch.assert_validated()


@pytest.mark.parametrize("mutation", ["feature", "clock", "owner"])
def test_hop_reordering_rejects_invalidated_source_before_reading(batch, mutation):
    from tfm.data.graph import order_graph_by_hop

    valid = order_graph_by_hop(batch, depth=1)
    valid.assert_validated()
    value = {
        "feature": batch.node_features["events"],
        "clock": batch.node_available_us["events"],
        "owner": batch.node_sample["events"],
    }[mutation]
    value.add_(0)
    with pytest.raises(ValueError, match="mutated"):
        order_graph_by_hop(batch, depth=1)
