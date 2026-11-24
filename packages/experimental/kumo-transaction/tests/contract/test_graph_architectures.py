# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import torch

from tfm.compositions.gnn import GNNEncoder, GNNSequence
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.config import OptimizerConfig, TrainingConfig
from tfm.contracts.graph import (
    FutureSubgraphTargets,
    GraphAnchor,
    GraphHistoryBatch,
    GraphTrainingBatch,
    utc_us,
)
from tfm.contracts.schema import TransactionEvent, TransactionSequence
from tfm.data.adapters.kumo_graph import from_kumo_graph
from tfm.data.graph import (
    GraphLink,
    GraphRow,
    GraphSampler,
    GraphTable,
    RelationalGraph,
    future_subgraph_targets,
    order_graph_by_hop,
)
from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel
from tfm.objectives.future_subgraph import FutureSubgraph
from tfm.training.checkpointing import load_checkpoint, save_checkpoint
from tfm.training.selection import fit_selected, validation_loss
from tfm.training.trainer import train_model

T = datetime(2024, 1, 10, tzinfo=UTC)
DAY = timedelta(days=1)
DAY_US = 86_400_000_000
UA = ("users", "accounts", "accounts")
AE = ("accounts", "events", "events")
ALTERNATE = ("accounts", "other_events", "events")


@pytest.fixture
def engine():
    return pytest.importorskip(
        "kumo_core.modeling.gnn.hetero", reason="requires the optional kumo extra"
    )


@pytest.fixture
def graph() -> RelationalGraph:
    rows = [GraphRow("users", "u", {"x": 1.0})]
    for name, day in [("old", -5), ("new", 1), ("after", 3)]:
        rows.append(GraphRow("accounts", name, {"x": 2.0}, T + day * DAY, T + day * DAY))
    for name, day, value in [
        ("past", -1, 2.0),
        ("at", 0, 3.0),
        ("inside", 1, 5.0),
        ("end", 2, 9.0),
        ("after", 3, 99.0),
    ]:
        rows.append(
            GraphRow("events", name, {"x": value, "amount": value}, T + day * DAY, T + day * DAY)
        )
    links = [
        GraphLink(UA, "u", name, T + day * DAY)
        for name, day in [("old", -5), ("new", 1), ("after", 3)]
    ]
    links += [
        GraphLink(AE, "old", name, T + day * DAY)
        for name, day in [("past", -1), ("at", 0), ("inside", 1), ("end", 2), ("after", 3)]
    ]
    # A distinct FK role must retain a separate target group. Each physical
    # FK has at most one parent per child, as required by Engine gather.
    links += [
        GraphLink(ALTERNATE, "new", "inside", T + DAY),
    ]
    return RelationalGraph(
        (
            GraphTable("users", ("x",), immutable=True),
            GraphTable("accounts", ("x",)),
            GraphTable("events", ("x",), ("amount",)),
        ),
        (UA, AE, ALTERNATE),
        tuple(rows),
        tuple(links),
        frozenset({"fraud", "id", "clock"}),
        T + 5 * DAY,
    )


def labels(graph: RelationalGraph, *, cutoff: datetime = T) -> FutureSubgraphTargets:
    return future_subgraph_targets(
        GraphSampler(graph, depth=2, fanout=100),
        (GraphAnchor("s", "u", cutoff),),
        "users",
        horizon_us=2 * DAY_US,
        depth=2,
        partition_end_us=utc_us(T + 5 * DAY),
    )


def spec(graph: RelationalGraph, *, depth: int = 2) -> GNNEncoder:
    return GNNEncoder(
        tuple((t.name, len(t.features)) for t in graph.tables),
        graph.directed_relations,
        "users",
        foreign_key_relations=tuple((t, f"rev_{r}", s) for s, r, t in graph.relations),
        hidden_size=8,
        depth=depth,
    )


def test_future_oracle_window_end_membership_roles_and_pk_dedup(graph: RelationalGraph) -> None:
    target = labels(graph)
    values = dict(zip(target.columns, target.values[0].tolist(), strict=True))
    assert values["1/users/accounts/accounts/count"] == 1  # only new, not old/after
    assert values["2/accounts/events/events/count"] == 2  # inside + inclusive end
    assert values["2/accounts/events/events/amount/sum"] == 14
    assert values["2/accounts/events/events/amount/mean"] == 7
    assert values["2/accounts/events/events/amount/std"] == pytest.approx(8**0.5)
    assert values["2/accounts/other_events/events/count"] == 1
    # New account has no past input path, but contributes to label membership.
    assert values["2/accounts/other_events/events/amount/sum"] == 5


@pytest.mark.usefixtures("engine")
def test_future_mutation_invariance_and_repeated_anchors(graph: RelationalGraph) -> None:
    torch.manual_seed(3)
    sampler = GraphSampler(graph, depth=2, fanout=100)
    anchor = (GraphAnchor("s", "u", T),)
    model = GNNEncoderModel(spec(graph), labels(graph).columns).eval()
    before = model(sampler.sample(anchor, "users"), compute_loss=False).embeddings
    future = tuple(
        replace(r, values={**r.values, "x": 700.0, "amount": 700.0})
        if r.row_id == "inside" and r.table == "events"
        else r
        for r in graph.rows
    )
    changed = replace(graph, rows=future)
    after = model(
        GraphSampler(changed, depth=2, fanout=100).sample(anchor, "users"), compute_loss=False
    ).embeddings
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.equal(labels(graph).values, labels(changed).values)
    later = replace(
        graph,
        rows=tuple(
            replace(r, values={**r.values, "x": 999.0, "amount": 999.0})
            if r.row_id == "after"
            else r
            for r in graph.rows
        ),
    )
    torch.testing.assert_close(labels(graph).values, labels(later).values)
    mixed = sampler.sample((anchor[0], GraphAnchor("later", "u", T + DAY)), "users")
    output = model(mixed, compute_loss=False).embeddings
    assert output is not None and output.shape == (2, 8)
    torch.testing.assert_close(output[:1], before)
    assert mixed.root_index.unique().numel() == 2


def test_missing_empty_masks_late_availability_and_coverage(graph: RelationalGraph) -> None:
    missing = replace(
        graph,
        rows=tuple(
            replace(r, values={"x": 1.0, "amount": float("nan")}) if r.table == "events" else r
            for r in graph.rows
        ),
    )
    target = labels(missing)
    lookup = {c: i for i, c in enumerate(target.columns)}
    assert target.values[0, lookup["2/accounts/events/events/amount/finite_count"]] == 0
    assert not target.valid[0, lookup["2/accounts/events/events/amount/sum"]]
    assert target.valid[0, lookup["1/accounts/events/events/amount/sum"]]  # empty group
    assert not target.valid[0, lookup["1/accounts/events/events/amount/mean"]]
    late = replace(
        graph,
        rows=tuple(
            replace(r, available_at=T + 4 * DAY)
            if r.table == "events" and r.row_id == "inside"
            else r
            for r in graph.rows
        ),
    )
    assert int(labels(late).available_us[0]) == utc_us(T + 4 * DAY)
    with pytest.raises(ValueError, match="unavailable"):
        future_subgraph_targets(
            GraphSampler(late, depth=2, fanout=100),
            target.anchors,
            "users",
            horizon_us=2 * DAY_US,
            depth=2,
            partition_end_us=utc_us(T + 3 * DAY),
        )
    with pytest.raises(ValueError, match="coverage"):
        labels(replace(graph, observed_through=T + DAY))
    with pytest.raises(ValueError, match="partition"):
        replace(target, label_delay_us=10 * DAY_US)
    with pytest.raises(ValueError, match="Forbidden"):
        replace(graph, tables=(replace(graph.tables[0], features=("fraud",)), *graph.tables[1:]))


@pytest.mark.parametrize("late", ["terminal", "delay"])
def test_future_availability_must_fit_declared_source_watermark(graph, late):
    source = replace(
        graph,
        rows=tuple(
            replace(row, available_at=T + 4 * DAY)
            if late == "terminal" and row.table == "events" and row.row_id == "inside"
            else row
            for row in graph.rows
        ),
        observed_through=T + 3 * DAY,
    )
    kwargs = dict(
        anchors=(GraphAnchor("s", "u", T),),
        root_type="users",
        horizon_us=2 * DAY_US,
        depth=2,
        partition_end_us=utc_us(T + 5 * DAY),
        label_delay_us=2 * DAY_US if late == "delay" else 0,
    )
    with pytest.raises(ValueError, match="availability.*source coverage"):
        future_subgraph_targets(GraphSampler(source, depth=2, fanout=100), **kwargs)
    boundary = replace(source, observed_through=T + 4 * DAY)
    target = future_subgraph_targets(GraphSampler(boundary, depth=2, fanout=100), **kwargs)
    assert target.available_us.tolist() == [utc_us(T + 4 * DAY)]


def test_historical_validation_weights_actual_target_tokens_and_event_mask():
    from tfm.layers.profile_event_decoder import ProfileEventDecoderModel

    sequences = [
        TransactionSequence(
            str(i),
            tuple(
                TransactionEvent(str(i), str(j), T - (n - j) * DAY, numeric={"amount": i + 1.0})
                for j in range(n)
            ),
            T,
            None,
            sequence_id=str(i),
        )
        for i, n in enumerate((1, 3))
    ]
    tokenizer = KeyValueTimeTokenizer().fit(sequences)
    batches = [tokenizer.collate([sequence], max_length=8) for sequence in sequences]
    batches[1].event_mask[:, -1] = False
    torch.manual_seed(112)
    model = ProfileEventDecoderModel(
        ProfileEventDecoder(
            hidden_size=8,
            num_heads=2,
            profile_depth=1,
            event_depth=1,
            decoder_depth=1,
            dropout=0.0,
            vocab_size=tokenizer.vocab_size,
        )
    ).eval()
    losses = [float(model(batch).loss.detach()) for batch in batches]
    counts = [
        int((batch.token_mask.bool() & batch.event_mask.bool().unsqueeze(-1)).sum())
        for batch in batches
    ]
    assert counts[0] != counts[1]
    assert counts[1] < int(batches[1].token_mask.sum())
    expected = sum(loss * count for loss, count in zip(losses, counts, strict=True)) / sum(counts)
    assert expected != pytest.approx(sum(losses) / len(losses))
    model.train()
    assert validation_loss(model, batches, torch.device("cpu")) == pytest.approx(expected)
    assert model.training
    empty = replace(batches[0], event_mask=torch.zeros_like(batches[0].event_mask))
    with pytest.raises(ValueError, match="no valid target"):
        validation_loss(model, [empty], torch.device("cpu"))
    assert model.training


def test_graph_bounds_future_clocks_edge_orientation_and_alignment(graph: RelationalGraph) -> None:
    batch = GraphSampler(graph, depth=2, fanout=100).sample((GraphAnchor("s", "u", T),), "users")
    assert batch.edge_index[UA].shape[0] == 2
    # Message edges from the outermost sampled shell must point inward too.
    assert batch.edge_index[("events", "rev_events", "accounts")].shape[1] > 0
    with pytest.raises(ValueError, match="bounds"):
        replace(batch, root_index=torch.tensor([999]))
    times = {**batch.node_available_us, "users": torch.tensor([utc_us(T + DAY)])}
    with pytest.raises(ValueError, match="Future node"):
        replace(batch, node_available_us=times)
    with pytest.raises(ValueError, match="Future edge"):
        replace(
            batch,
            edge_available_us={
                r: torch.full_like(v, utc_us(T + DAY)) for r, v in batch.edge_available_us.items()
            },
        )
    with pytest.raises(ValueError, match="identities"):
        GraphTrainingBatch(batch, replace(labels(graph), anchors=(GraphAnchor("wrong", "u", T),)))


@pytest.mark.usefixtures("engine")
def test_masked_loss_no_gradient_or_adamw_update(graph: RelationalGraph) -> None:
    torch.manual_seed(0)
    target = labels(graph)
    masked = replace(
        target,
        values=torch.full_like(target.values, float("nan")),
        valid=torch.zeros_like(target.valid),
    )
    prediction = torch.randn_like(target.values, requires_grad=True)
    loss = FutureSubgraph().loss(prediction, masked)
    loss.backward()
    assert prediction.grad is not None and prediction.grad.count_nonzero() == 0
    graph_batch = GraphSampler(graph, depth=2, fanout=4).sample(target.anchors, "users")
    model = GNNEncoderModel(spec(graph), target.columns)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    train_model(
        model,
        [GraphTrainingBatch(graph_batch, masked)],
        max_steps=2,
        learning_rate=0.1,
        seed=0,
        device=torch.device("cpu"),
        optimizer=OptimizerConfig(weight_decay=0.9),
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="no valid"):
        validation_loss(model, [GraphTrainingBatch(graph_batch, masked)], torch.device("cpu"))
    with pytest.raises(ValueError, match="no valid"):
        fit_selected(
            model,
            [GraphTrainingBatch(graph_batch, masked)],
            [GraphTrainingBatch(graph_batch, masked)],
            TrainingConfig(max_steps=1),
            seed=0,
            device=torch.device("cpu"),
        )


def history_batch(graph: RelationalGraph) -> GraphHistoryBatch:
    anchor = GraphAnchor("s", "u", T)
    events = (TransactionEvent("u", "e", T - DAY, numeric={"amount": 1.0}),)
    sequence = TransactionSequence("u", events, T, None, sequence_id="s")
    tokenizer = KeyValueTimeTokenizer().fit([sequence])
    history = tokenizer.collate([sequence], max_length=8)
    return GraphHistoryBatch(
        GraphSampler(graph, depth=2, fanout=100).sample((anchor,), "users"),
        history,
        (anchor,),
        torch.tensor([1.0]),
        torch.tensor([utc_us(T)]),
        utc_us(T),
    )


@pytest.mark.usefixtures("engine")
def test_frozen_hybrid_profile_bridge_gradients_and_checkpoint(
    graph: RelationalGraph, tmp_path
) -> None:
    torch.manual_seed(1)
    batch = history_batch(graph)
    g = spec(graph)
    pretrained = GNNEncoderModel(g, labels(graph).columns)
    sequence = ProfileEventDecoder(
        hidden_size=8, num_heads=2, profile_depth=1, event_depth=1, decoder_depth=1, vocab_size=64
    )
    model = GNNSequenceModel(GNNSequence(g, sequence), pretrained.encoder)
    with pytest.raises(ValueError, match="next-event"):
        GNNSequenceModel(
            GNNSequence(g, sequence), pretrained.encoder, objective="next_event_fields"
        )
    # Buffer freezing must survive model.train(), not just requires_grad_(False).
    model.graph_encoder.register_buffer("normalization_marker", torch.tensor([2.0]))
    before = {k: v.clone() for k, v in model.graph_encoder.state_dict().items()}
    model.train()
    assert not model.graph_encoder.training
    output = model(batch)
    assert output.embeddings is not None and output.embeddings.shape == (1, 8)
    assert output.loss is not None
    output.loss.backward()
    assert model.bridge.weight.grad is not None and model.bridge.weight.grad.abs().sum() > 0
    assert model.graph_field.grad is not None and model.graph_field.grad.abs().sum() > 0
    assert model.embed.weight.grad is not None and model.embed.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.graph_encoder.parameters())
    train_model(model, [batch], max_steps=2, learning_rate=0.01, seed=1, device=torch.device("cpu"))
    for key, value in model.graph_encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model.eval()
    expected = model(batch, compute_loss=False).embeddings
    save_checkpoint(model, tmp_path / "graph.pt", metadata={"architecture": "gnn_sequence"})
    for p in model.bridge.parameters():
        p.data.zero_()
    load_checkpoint(model, tmp_path / "graph.pt")
    torch.testing.assert_close(model(batch, compute_loss=False).embeddings, expected)
    with pytest.raises(ValueError, match="anchors"):
        replace(batch, history_anchors=(GraphAnchor("s", "u", T + DAY),))
    with pytest.raises(ValueError, match="order"):
        replace(batch, history=replace(batch.history, sequence_ids=("other",)))
    with pytest.raises(ValueError, match="available"):
        replace(batch, label_available_us=torch.tensor([utc_us(T + DAY)]))


@pytest.mark.usefixtures("engine")
@pytest.mark.parametrize("inference", [False, True])
def test_kumo_adapter_preserves_native_ids_roles_and_sample_order(
    graph: RelationalGraph, inference: bool
) -> None:
    batch = GraphSampler(graph, depth=2, fanout=100).sample(
        (GraphAnchor("s", "u", T),), "users", hop_trim=False
    )
    ids = {name: torch.arange(len(x)) + 100 for name, x in batch.node_features.items()}
    native = SimpleNamespace(
        metadata={"sample_ids": ["s"]},
        inputs={
            "graph": {
                "nodes": {
                    name: {
                        "ids": ids[name],
                        "batch": batch.node_sample[name],
                        "features": {"x": {"values": x[:, 0]}},
                    }
                    for name, x in batch.node_features.items()
                },
                "edges": {
                    r: {"index": x, "source_type": s, "destination_type": t}
                    for (s, r, t), x in batch.edge_index.items()
                },
            }
        },
    )
    kwargs = dict(
        anchors=batch.anchors,
        root_type="users",
        root_node_ids=(100,),
        feature_columns={name: ("x",) for name in batch.node_features},
        relations=graph.directed_relations,
        forbidden_fields=graph.forbidden_fields,
        node_available_us=batch.node_available_us,
        edge_available_us=batch.edge_available_us,
        hop_depth=2,
    )
    with torch.inference_mode(inference):
        adapted = from_kumo_graph(native, **kwargs)
        adapted.assert_validated()
    model = GNNEncoderModel(spec(graph), labels(graph).columns).eval()
    torch.testing.assert_close(
        model(adapted, compute_loss=False).embeddings,
        model(order_graph_by_hop(batch, 2), compute_loss=False).embeddings,
    )
    # The native fixture uses relation-name strings, not EdgeType tuple keys.
    assert all(isinstance(name, str) for name in native.inputs["graph"]["edges"])
    native.inputs["graph"]["edges"] = {
        name: edge
        for name, edge in native.inputs["graph"]["edges"].items()
        if not name.startswith("rev_")
    }
    assert set(native.inputs["graph"]["edges"]) == {name for _, name, _ in graph.relations}
    physical_only = from_kumo_graph(native, **kwargs)
    torch.testing.assert_close(
        model(physical_only, compute_loss=False).embeddings,
        model(adapted, compute_loss=False).embeddings,
    )
    shared_anchors = (GraphAnchor("left", "u", T), GraphAnchor("right", "u", T))
    native.metadata["sample_ids"] = [a.sample_id for a in shared_anchors]
    shared_kwargs = {
        **kwargs,
        "anchors": shared_anchors,
        "root_node_ids": (100, 100),
        "sharing": "shared",
    }
    shared_native = from_kumo_graph(native, **shared_kwargs)
    assert shared_native.root_index.tolist() == [0, 0] and shared_native.sharing == "shared"
    torch.testing.assert_close(
        model(shared_native, compute_loss=False).embeddings,
        model(adapted, compute_loss=False).embeddings.repeat(2, 1),
    )
    native.metadata["sample_ids"] = ["wrong"]
    with pytest.raises(ValueError, match="sample order"):
        from_kumo_graph(native, **kwargs)


@pytest.mark.usefixtures("engine")
def test_graph_valid_loss_reaches_encoder_and_stage_optimizers_are_fresh(graph, monkeypatch):
    from tfm.training import trainer

    torch.manual_seed(2)
    target = labels(graph)
    training = GraphTrainingBatch(
        GraphSampler(graph, depth=2, fanout=100).sample(target.anchors, "users"), target
    )
    graph_model = GNNEncoderModel(spec(graph), target.columns)
    output = graph_model(training)
    output.loss.backward()
    assert graph_model.head.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in graph_model.encoder.parameters()
    )
    assert not training.targets.values.requires_grad
    created = []
    original = trainer.build_optimizers

    def record(model, **kwargs):
        optimizers = original(model, **kwargs)
        assert all(not optimizer.state for optimizer in optimizers)
        created.append(
            {
                id(p)
                for optimizer in optimizers
                for group in optimizer.param_groups
                for p in group["params"]
            }
        )
        return optimizers

    monkeypatch.setattr(trainer, "build_optimizers", record)
    settings = TrainingConfig(max_steps=2, learning_rate=0.001)
    fit_selected(graph_model, [training], [training], settings, seed=2, device=torch.device("cpu"))
    seq = ProfileEventDecoder(
        hidden_size=8, num_heads=2, event_depth=1, decoder_depth=1, vocab_size=64
    )
    hybrid = GNNSequenceModel(GNNSequence(spec(graph), seq), graph_model.encoder)
    history = history_batch(graph)
    fit_selected(hybrid, [history], [history], settings, seed=2, device=torch.device("cpu"))
    assert len(created) == 2 and created[0].isdisjoint(created[1])


@pytest.mark.usefixtures("engine")
def test_padding_does_not_change_hybrid_embedding(graph):
    import torch.nn.functional as functional

    torch.manual_seed(0)
    batch = history_batch(graph)
    g = spec(graph)
    model = GNNSequenceModel(
        GNNSequence(
            g,
            ProfileEventDecoder(
                hidden_size=8, num_heads=2, event_depth=1, decoder_depth=1, vocab_size=64
            ),
        ),
        GNNEncoderModel(g, labels(graph).columns).encoder,
    ).eval()
    original = model(batch, compute_loss=False).embeddings
    h = batch.history
    fields = {
        name: functional.pad(getattr(h, name), (0, 0, 0, 2))
        for name in ("key_ids", "value_ids", "positions", "token_mask")
    }
    fields.update(
        {
            name: functional.pad(getattr(h, name), (0, 2))
            for name in ("event_mask", "time_log", "elapsed_log")
        }
    )
    fields["events"] = {name: functional.pad(value, (0, 2)) for name, value in h.events.items()}
    padded = replace(batch, history=replace(h, **fields))
    torch.testing.assert_close(model(padded, compute_loss=False).embeddings, original)


def test_future_link_after_window_is_excluded_and_recent_ties_are_stable(graph):
    later_link = GraphLink(ALTERNATE, "old", "end", T + 3 * DAY)
    torch.testing.assert_close(
        labels(replace(graph, links=(*graph.links, later_link))).values, labels(graph).values
    )
    tied = GraphRow("events", "aaa", {"x": 42.0, "amount": 42.0}, T, T)
    expanded = replace(
        graph, rows=(*graph.rows, tied), links=(*graph.links, GraphLink(AE, "old", "aaa", T))
    )
    sampled = GraphSampler(expanded, depth=2, fanout=1).sample((GraphAnchor("s", "u", T),), "users")
    assert sampled.node_features["events"].flatten().tolist() == [42.0]


def test_shared_graph_contract_deduplicates_but_preserves_repeated_predictions(graph, engine):
    anchors = (GraphAnchor("s1", "u", T), GraphAnchor("s2", "u", T))
    sampler = GraphSampler(graph, depth=2, fanout=100)
    shared = sampler.sample(anchors, "users", sharing="shared")
    separate = sampler.sample(anchors, "users")
    assert shared.root_index.tolist() == [0, 0]
    assert shared.batch_size == separate.batch_size == 2
    assert all(
        2 * len(shared.node_features[k]) == len(separate.node_features[k])
        for k in shared.node_features
    )
    assert all(
        2 * shared.edge_index[k].shape[1] == separate.edge_index[k].shape[1]
        for k in shared.edge_index
    )
    model = GNNEncoderModel(spec(graph), labels(graph).columns).eval()
    prediction = model(shared, compute_loss=False).embeddings
    assert prediction.shape == (2, 8)
    torch.testing.assert_close(prediction[0], prediction[1])
    with pytest.raises(ValueError, match="identical cutoffs"):
        sampler.sample((anchors[0], replace(anchors[1], cutoff=T + DAY)), "users", sharing="shared")
    with pytest.raises(ValueError, match="observation windows"):
        sampler.sample(
            (anchors[0], replace(anchors[1], observation_start=T - DAY)), "users", sharing="shared"
        )
    raw = labels(graph)
    targets = replace(
        raw,
        anchors=anchors,
        values=raw.values.repeat(2, 1),
        valid=raw.valid.repeat(2, 1),
        available_us=raw.available_us.repeat(2),
    )
    with pytest.raises(ValueError, match="Validation requires disjoint"):
        validation_loss(model, [GraphTrainingBatch(shared, targets)], torch.device("cpu"))
    # Schema and declared FK direction must instantiate the actual Engine class.
    from kumo_core.modeling.gnn.hetero import HeteroGNN, PrimaryKeyToForeignKeyConv

    assert isinstance(model.encoder.gnn, HeteroGNN)
    alias = model.encoder.engine_roles[UA]
    assert alias[1].startswith("rev_")
    assert isinstance(model.encoder.gnn.hetero_convs[0][alias], PrimaryKeyToForeignKeyConv)


def test_engine_rejects_multi_parent_gather_but_target_oracle_deduplicates(graph, engine):
    # This extra link is valid for the general graph oracle but not a scalar FK.
    duplicate_path = GraphLink(AE, "new", "inside", T + DAY)
    expanded = replace(graph, links=(*graph.links, duplicate_path))
    torch.testing.assert_close(labels(graph).values, labels(expanded).values)
    batch = GraphSampler(expanded, depth=2, fanout=100).sample(
        (GraphAnchor("s", "u", T + 2 * DAY),), "users", hop_trim=False
    )
    model = GNNEncoderModel(replace(spec(graph), trim=False), labels(graph).columns)
    with pytest.raises(ValueError, match="at most one parent"):
        model(batch, compute_loss=False)


def test_missing_engine_is_explicit_and_does_not_select_a_local_model(graph, monkeypatch):
    from tfm.contracts.errors import ExtraUnavailableError
    from tfm.extras import kumo

    original = kumo.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "kumo_core.modeling.gnn.hetero":
            raise ImportError("deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(kumo.importlib, "import_module", missing)
    with pytest.raises(ExtraUnavailableError, match="--extra kumo"):
        GNNEncoderModel(spec(graph), labels(graph).columns)


@pytest.mark.usefixtures("engine")
@pytest.mark.parametrize("depth", [1, 2, 6])
@pytest.mark.parametrize("sharing", ["shared", "disjoint"])
def test_engine_receives_signed_inputs_and_hop_counts_and_matches_untrimmed(graph, depth, sharing):
    from tfm.layers.gnn import GraphEncoder

    torch.manual_seed(31)
    anchors = (GraphAnchor("a", "u", T), GraphAnchor("b", "u", T))
    batch = GraphSampler(graph, depth=depth, fanout=100).sample(anchors, "users", sharing=sharing)
    encoder = GraphEncoder(spec(graph, depth=depth)).eval()
    with torch.no_grad():
        for layer in encoder.input:
            layer.weight.zero_()
            layer.bias.fill_(-2)
    seen = {}

    def capture(module, args, kwargs):
        seen.update(kwargs)
        assert all(bool((x < 0).all()) for x in args[0].values())

    handle = encoder.gnn.register_forward_pre_hook(capture, with_kwargs=True)
    trimmed = encoder(batch)
    handle.remove()
    assert seen["num_sampled_nodes_dict"] == {
        k: list(v) for k, v in batch.num_sampled_nodes.items()
    }
    assert seen["num_sampled_edges_dict"] == {
        encoder.engine_roles[k]: list(v) for k, v in batch.num_sampled_edges.items()
    }
    control = GraphEncoder(replace(spec(graph, depth=depth), trim=False)).eval()
    control.load_state_dict(encoder.state_dict())
    torch.testing.assert_close(trimmed, control(batch), atol=1e-6, rtol=1e-5)
    raw = GraphSampler(graph, depth=depth, fanout=100).sample(
        batch.anchors, "users", hop_trim=False, sharing=sharing
    )
    torch.testing.assert_close(trimmed, control(raw), atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="requires validated"):
        encoder(raw)
    with pytest.raises(ValueError, match="hop counts"):
        replace(batch, num_sampled_nodes={k: (0, 0, 0) for k in batch.node_features})
    with pytest.raises(ValueError, match="Hop metadata"):
        replace(batch, node_hop=None)
    name = next(k for k, h in batch.node_hop.items() if h.numel())
    with pytest.raises(ValueError, match="hop-prefix"):
        replace(batch, node_hop={**batch.node_hop, name: batch.node_hop[name] + 1})


@pytest.mark.usefixtures("engine")
def test_direct_supervised_gnn_gradients_and_hybrid_attention_parity(graph):
    from tfm.attention.eager import EagerAttention
    from tfm.attention.sdpa import SdpaAttention
    from tfm.contracts.graph import GraphSupervisedBatch
    from tfm.heads.binary import Binary

    torch.manual_seed(7)
    batch = history_batch(graph)
    supervised = GraphSupervisedBatch(
        batch.graph, batch.labels, batch.label_available_us, batch.label_cutoff_us
    )
    model = GNNEncoderModel(spec(graph), objective=Binary())
    output = model(supervised)
    output.loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    with pytest.raises(ValueError, match="available"):
        replace(supervised, label_available_us=supervised.label_available_us + DAY_US)
    seq = ProfileEventDecoder(
        hidden_size=8, num_heads=2, event_depth=1, decoder_depth=1, vocab_size=64
    )
    eager = GNNSequenceModel(
        GNNSequence(spec(graph), seq), model.encoder, attention=EagerAttention()
    ).eval()
    import copy

    sdpa = GNNSequenceModel(
        GNNSequence(spec(graph), seq), copy.deepcopy(model.encoder), attention=SdpaAttention()
    ).eval()
    sdpa.load_state_dict(eager.state_dict())
    a, b = eager(batch), sdpa(batch)
    torch.testing.assert_close(a.embeddings, b.embeddings, atol=2e-6, rtol=1e-5)
    a.loss.backward()
    b.loss.backward()
    for (name, left), (_, right) in zip(
        eager.named_parameters(), sdpa.named_parameters(), strict=True
    ):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=1e-4, msg=name)
    assert all(p.grad is None for p in sdpa.graph_encoder.parameters())


def test_supervised_data_never_builds_future_targets(monkeypatch):
    import yaml

    from tfm.config import RecipeConfig
    from tfm.contracts.graph import GraphSupervisedBatch
    from tfm.data import graph_recipe

    def forbidden(*args, **kwargs):
        raise AssertionError("supervised preparation requested future targets")

    monkeypatch.setattr(graph_recipe, "future_subgraph_targets", forbidden)
    monkeypatch.setattr(graph_recipe.FutureTargetTransform, "fit", forbidden)
    from pathlib import Path

    config = yaml.safe_load(Path("recipes/gnn_encoder/config.smoke.yaml").read_text())
    config["data"]["pretrain_end"] = "2099-01-01T00:00:00Z"
    data = graph_recipe.prepare_graph_data(RecipeConfig.model_validate(config))
    assert data.transform is None
    assert all(isinstance(b, GraphSupervisedBatch) for b in data.pretrain + data.validation)


def test_future_builders_and_hop_ordering_inside_inference_mode(graph):
    from tfm.data.graph import FutureTargetTransform, order_graph_by_hop

    sampler = GraphSampler(graph, depth=2, fanout=100)
    anchors = (GraphAnchor("first", "u", T), GraphAnchor("second", "u", T + DAY))
    raw = sampler.sample(anchors, "users", hop_trim=False)
    with torch.inference_mode():
        order_graph_by_hop(raw, 2).assert_validated()
        targets = future_subgraph_targets(
            sampler,
            anchors,
            "users",
            horizon_us=2 * DAY_US,
            depth=2,
            partition_end_us=utc_us(T + 5 * DAY),
        )
        transform = FutureTargetTransform.fit(targets)
        transformed = transform.apply(targets)
        targets.assert_validated()
        transformed.assert_validated()
        assert not transform.mean.is_inference() and not transform.scale.is_inference()
        assert torch.is_inference_mode_enabled()
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()


@pytest.mark.usefixtures("engine")
def test_supervised_sequence_control_uses_no_graph_and_shares_blueprint_encoding(
    graph, monkeypatch
):
    from tfm.attention.sdpa import SdpaAttention
    from tfm.contracts.graph import GraphBatch, SequenceSupervisedBatch
    from tfm.layers.gnn import SequenceBinaryModel
    from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
    from tfm.runtime.graph import _probabilities, _table

    torch.manual_seed(91)
    batch = history_batch(graph)
    batch = replace(
        batch,
        history=replace(batch.history, numeric_features=torch.zeros(batch.graph.batch_size, 5)),
    )
    control_batch = SequenceSupervisedBatch.from_graph_history(batch)
    seq = ProfileEventDecoder(
        hidden_size=8, num_heads=2, event_depth=1, decoder_depth=1, vocab_size=64
    )
    control = SequenceBinaryModel(seq, SdpaAttention()).eval()
    reference = ProfileEventDecoderModel(seq, attention=SdpaAttention()).eval()
    reference.load_state_dict(control.state_dict(), strict=False)
    a = control(control_batch, compute_loss=False)
    b = reference(batch.history, compute_loss=False)
    torch.testing.assert_close(a.hidden_states, b.hidden_states)
    torch.testing.assert_close(a.embeddings, b.embeddings)
    a.embeddings.square().sum().backward()
    b.embeddings.square().sum().backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in control.named_parameters():
        if name in reference_parameters:
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)
    control.zero_grad(set_to_none=True)
    with torch.no_grad():
        for parameter in control.parameters():
            parameter.zero_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Sequence-only control transferred unused graph")

    monkeypatch.setattr(GraphBatch, "to", forbidden)
    output = control(control_batch.to(torch.device("cpu")))
    assert float(output.loss.detach()) == pytest.approx(torch.log(torch.tensor(2.0)).item())
    output.loss.backward()
    assert control.head.bias.grad.abs().sum() > 0
    _table(control, [batch], torch.device("cpu"), view="sequence")
    _table(reference, [batch], torch.device("cpu"), view="sequence")
    _probabilities(control, [batch], torch.device("cpu"))
    with pytest.raises(ValueError, match="available"):
        replace(control_batch, label_available_us=control_batch.label_available_us + DAY_US)
    control_batch.labels.add_(0)
    with pytest.raises(ValueError, match="mutated"):
        control(control_batch)


@pytest.mark.parametrize("historical", [True, False])
@pytest.mark.parametrize("mutation", ["value_ids", "labels"])
def test_sequence_evaluation_rejects_mutation_before_history_copy_or_forward(
    graph, monkeypatch, historical, mutation
):
    from tfm.contracts.kvt import KeyValueTimeBatch
    from tfm.layers.gnn import SequenceBinaryModel
    from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
    from tfm.runtime.graph import _table

    batch = history_batch(graph)
    spec = ProfileEventDecoder(
        hidden_size=8, num_heads=2, event_depth=1, decoder_depth=1, vocab_size=64
    )
    model = ProfileEventDecoderModel(spec) if historical else SequenceBinaryModel(spec)
    if mutation == "value_ids":
        batch.history.value_ids.add_(0)
    else:
        batch.labels.add_(0)
    with pytest.raises(ValueError, match="mutated"):
        batch.assert_validated()

    def forbidden(*args, **kwargs):
        raise AssertionError("Used historical inputs before checking batch mutation")

    monkeypatch.setattr(KeyValueTimeBatch, "to", forbidden)
    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="mutated"):
        _table(model, [batch], torch.device("cpu"), view="sequence")


@pytest.mark.usefixtures("engine")
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize(
    "precision",
    [
        "cpu",
        pytest.param(
            "fp32",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
        pytest.param(
            "bf16",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                reason="CUDA BF16 unavailable",
            ),
        ),
    ],
)
def test_frozen_hybrid_event_first_matches_eager_context(graph, monkeypatch, varlen, precision):
    torch.manual_seed(5)
    device = torch.device("cpu" if precision == "cpu" else "cuda")
    batch = history_batch(graph).for_execution(device)
    graph_spec = spec(graph)
    pretrained = GNNEncoderModel(graph_spec, labels(graph).columns)
    sequence = ProfileEventDecoder(
        hidden_size=8,
        num_heads=2,
        profile_depth=1,
        event_depth=1,
        decoder_depth=1,
        vocab_size=64,
        dropout=0.0,
        varlen=varlen,
    )
    model = (
        GNNSequenceModel(GNNSequence(graph_spec, sequence), pretrained.encoder).to(device).train()
    )
    exact = precision == "cpu" or torch.are_deterministic_algorithms_enabled()
    tolerance = 0 if exact else (1e-2 if precision == "bf16" else 1e-5)
    frozen = {k: v.clone() for k, v in model.graph_encoder.state_dict().items()}
    order = []
    event_hook = model.event_encoder.register_forward_pre_hook(lambda *_: order.append("event"))
    graph_hook = model.graph_encoder.register_forward_pre_hook(lambda *_: order.append("graph"))
    encode_history = model.encode_history

    def eager(history, context):
        assert callable(context)
        return encode_history(history, context())

    results = []
    try:
        for deferred in [False, True]:
            model.zero_grad(set_to_none=True)
            order.clear()
            with monkeypatch.context() as patch:
                if not deferred:
                    patch.setattr(model, "encode_history", eager)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
                ):
                    output = model(batch)
                assert output.loss is not None and output.logits is not None
                output.loss.backward()
            assert order == (["event", "graph"] if deferred else ["graph", "event"])
            results.append(
                (
                    {
                        "logits": output.logits.detach().clone(),
                        "loss": output.loss.detach().clone(),
                        "embeddings": output.embeddings.detach().clone(),
                    },
                    {
                        k: p.grad.clone() if p.grad is not None else None
                        for k, p in model.named_parameters()
                    },
                )
            )
            assert not model.graph_encoder.training
            assert all(
                not p.requires_grad and p.grad is None for p in model.graph_encoder.parameters()
            )
            for key, value in model.graph_encoder.state_dict().items():
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    finally:
        event_hook.remove()
        graph_hook.remove()
    torch.testing.assert_close(results[0][0], results[1][0], rtol=tolerance, atol=tolerance)
    for key, gradient in results[0][1].items():
        if gradient is None:
            assert results[1][1][key] is None
        else:
            torch.testing.assert_close(gradient, results[1][1][key], rtol=tolerance, atol=tolerance)


@pytest.mark.usefixtures("engine")
@pytest.mark.parametrize("log_every", [None, 1])
def test_validation_selected_training_honors_logging_interval(graph, log_every, capsys):
    batch = GraphTrainingBatch(
        GraphSampler(graph, depth=2, fanout=100).sample(labels(graph).anchors, "users"),
        labels(graph),
    )
    model = GNNEncoderModel(spec(graph), batch.targets.columns)
    settings = TrainingConfig(max_steps=2, learning_rate=0.001, log_every=log_every)
    stats = fit_selected(model, [batch], [batch], settings, seed=0, device=torch.device("cpu"))
    output = capsys.readouterr().out
    assert len(stats["curve"]) == 2
    if log_every is None:
        assert output == ""
    else:
        assert "training 2 steps" in output
        assert "step 1/2" in output and "step 2/2" in output
