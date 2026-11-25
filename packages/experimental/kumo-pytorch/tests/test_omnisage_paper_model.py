# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent checks for paper equations 1–6 and the chunked embedding backward pass."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.e2e.omnisage_paper_model import (
    CountMinSketch,
    OmniSagePaperModel,
    PaperConfig,
    UserEncoder,
    optimizer_for,
    sampled_softmax,
)


@pytest.mark.parametrize("chunk", [1, 2, 17])
def test_chunked_loss_and_gradients_match_dense_sampled_softmax(chunk):
    generator = torch.Generator().manual_seed(42)
    q, p, n = (
        torch.randn(rows, 3, generator=generator, dtype=torch.double, requires_grad=True)
        for rows in (5, 5, 8)
    )
    pids = torch.tensor([1, 2, 2, 4, 7])
    nids = torch.tensor([1, 2, 2, 4, 7, 6, 8, 11])
    self_ids = torch.tensor([6, 6, 5, 6, 3])
    owners = torch.tensor([0, 0, 1, 2, 3])
    known = (torch.cat([owners, torch.tensor([2])]), torch.cat([pids, torch.tensor([8])]))
    weights = torch.tensor([0.3, 0.12, 0.2, 0.8, 0.4], dtype=torch.double)
    sketch = CountMinSketch()
    observed = torch.tensor([1, 1, 1, 2, 2, 4, 7, 8])
    sketch.observe(observed)

    def log_q(ids):
        frequency = torch.tensor([max(1, int((observed == k).sum())) for k in ids])
        return frequency.float().log() - torch.tensor(float(len(observed))).log()

    scale = 3.5
    pos = scale * (q * p).sum(1) - log_q(pids)
    neg = scale * q @ n.T - log_q(nids)
    mask = (
        (pids[:, None] == nids)
        | (self_ids[:, None] == nids)
        | torch.tensor(
            [
                [bool(((known[0] == owner) & (known[1] == k)).any()) for k in nids]
                for owner in owners
            ]
        )
    )
    logits = torch.cat([pos[:, None], neg.masked_fill(mask, -torch.inf)], 1)
    expected = -(logits.log_softmax(1)[:, 0] * weights).sum()
    expected_grads = torch.autograd.grad(expected, (q, p, n))
    actual = sampled_softmax(
        q,
        p,
        n,
        pids,
        nids,
        sketch,
        weights=weights,
        scale=scale,
        self_ids=self_ids,
        known=known,
        owners=owners,
        chunk_size=chunk,
    )
    torch.testing.assert_close(actual, expected)
    for value, reference in zip(
        torch.autograd.grad(actual, (q, p, n)), expected_grads, strict=True
    ):
        torch.testing.assert_close(value, reference)


def test_frequency_estimator_persists_across_batches_and_state_dict():
    sketch = CountMinSketch()
    sketch.observe(torch.tensor([2, 2, 3]))
    sketch.observe(torch.tensor([2, 4]))
    expected = torch.tensor([3.0, 1.0, 1.0, 1.0]).log() - torch.tensor(5.0).log()
    ids = torch.tensor([2, 3, 4, 99])
    torch.testing.assert_close(sketch.log_q(ids), expected)
    restored = CountMinSketch()
    restored.load_state_dict(sketch.state_dict())
    torch.testing.assert_close(restored.log_q(ids), expected)


def test_history_encoder_is_causal_and_padded_positions_do_not_propagate():
    torch.manual_seed(21)
    cfg = PaperConfig.cpu()
    encoder = UserEncoder(cfg)
    x = torch.randn(2, 4, cfg.embedding_width, requires_grad=True)
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    baseline = encoder(x, valid)
    changed = x.detach().clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 10
    torch.testing.assert_close(encoder(changed, valid)[:, :2], baseline[:, :2])
    assert not baseline[1, 2:].count_nonzero()
    baseline[:, 1, 0].sum().backward()
    assert x.grad[:, :2].count_nonzero() and not x.grad[:, 2:].count_nonzero()


def _toy_batch():
    generator = torch.Generator().manual_seed(77)
    ix = torch.arange(6)
    tokens = {
        "pins": {
            "ids": ix + 10,
            "features": {
                "image": {"values": torch.randn(6, 1152, generator=generator)},
                "degree": {"values": ix + 1},
            },
        },
        "boards": {
            "ids": ix + 100,
            "features": {
                "annotation": {"values": torch.randn(6, 128, generator=generator)},
                "degree": {"values": ix + 2},
            },
        },
    }

    def graph(banks):
        edges = {}
        for i, bank in enumerate(banks):
            src = ix[: 5 if i == 0 else 3]
            edges[bank] = {
                "index": torch.stack((src, (src + i + 1) % 6)),
                "score": 0.1 + src.float() / 10,
            }
        return {"nodes": tokens, "edges": edges}

    return SimpleNamespace(
        graphs={
            "pins": graph(["pb.pins", "pb.boards", "pp.pins"]),
            "boards": graph(["pb.pins", "pb.boards"]),
        },
        root_ids={"pins": ix + 10, "boards": ix + 100},
        features={"pins": torch.tensor([0, 5]), "boards": torch.tensor([0])},
        negatives={"pins": torch.tensor([4, 5]), "boards": torch.tensor([4, 5])},
        pairs={
            "graph": ("pins", torch.tensor([0, 2]), "boards", torch.tensor([1, 3])),
            "engaged": ("pins", torch.tensor([1, 3]), "pins", torch.tensor([2, 0])),
        },
        history=torch.tensor([[0, 1, 2, 3], [2, 3, -1, -1], [5, -1, -1, -1]]),
        future=torch.tensor([2, -1, 1]),
    )


def test_equal_weight_per_feature_root_and_sum_over_valid_history_positions(monkeypatch):
    # With unit per-observation CE the reductions have closed-form answers.
    model = OmniSagePaperModel(PaperConfig.cpu())

    def unit_ce(batch, embs, table, q, positive, in_batch, **kw):
        assert len(q) == len(positive)
        return kw.get("weights", torch.tensor([1.0])).sum()

    monkeypatch.setattr(model, "_loss", unit_ce)
    terms = model(_toy_batch())
    assert terms["pair"].item() == pytest.approx(1)
    assert terms["feature"].item() == pytest.approx(1)
    assert terms["next"].item() == pytest.approx((3 + 1 + 0) / 3)
    assert terms["future"].item() == pytest.approx((4 + 1) / 2)


def test_overlapping_banks_match_explicit_typed_union_for_losses_and_gradients():
    torch.manual_seed(18)
    model = OmniSagePaperModel(PaperConfig.cpu())
    reference = deepcopy(model)
    batch = _toy_batch()
    batch.features["pins"] = torch.tensor([0, 1, 2])
    graph = batch.graphs["pins"]
    # A second graph-local row for the same public Pin also exercises ID-level union.
    pins = graph["nodes"]["pins"]
    pins["ids"] = torch.cat([pins["ids"], pins["ids"][1:2]])
    for field in pins["features"].values():
        field["values"] = torch.cat([field["values"], field["values"][1:2]])

    def bank(src, dst, score):
        return {"index": torch.tensor([src, dst]), "score": torch.tensor(score)}

    graph["edges"] = {
        "pb.pins": bank([0, 0, 1, 2], [1, 2, 1, 3], [0.2, 0.3, 0.8, 0.5]),
        "pp.pins": bank([0, 1, 2], [6, 1, 3], [0.7, 0.6, 0.5]),
        "pb.boards": bank([0], [1], [0.4]),
    }
    # The board shares the Pin's numeric ID, but its type makes it a distinct token.
    graph["nodes"]["boards"]["ids"][1] = pins["ids"][1]
    explicit = deepcopy(batch)
    explicit.graphs["pins"]["edges"] = {
        "pb.pins": bank([0, 1, 2, 0], [2, 1, 3, 1], [0.3, 0.8, 0.5, 0.7]),
        "pb.boards": graph["edges"]["pb.boards"],
    }
    expected, actual = reference(explicit), model(batch)
    for term, value in actual.items():
        torch.testing.assert_close(value, expected[term], rtol=0, atol=0)
    expected["total"].backward()
    actual["total"].backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_joint_step_updates_each_encoder_family_once():
    torch.manual_seed(1)
    batch = _toy_batch()
    config = replace(
        PaperConfig.cpu(),
        family_weights=(0.3, 0.6, 0.1),
        source_weights={"graph": 0.25, "engaged": 0.75},
        chunk_size=3,
    )
    model = OmniSagePaperModel(config)
    opt, scheduler = optimizer_for(model, 4, 2)
    terms = model(batch)
    torch.testing.assert_close(
        terms["pair"], 0.25 * terms["pair/graph"] + 0.75 * terms["pair/engaged"]
    )
    torch.testing.assert_close(
        terms["total"],
        0.3 * terms["pair"] + 0.6 * terms["feature"] + 0.1 * (terms["next"] + terms["future"]),
    )
    checked = [
        model.entities["pins"].head[-1].weight,
        model.entities["boards"].head[-1].weight,
        *(head[0].weight for head in model.feature_heads.values()),
        model.users.layers[0].linear1.weight,
    ]
    before = [p.detach().clone() for p in checked]
    terms["total"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in checked
    )
    opt.step()
    scheduler.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, checked, strict=True))
    counts = {t: int(sketch.total) for t, sketch in model.sketches.items()}
    assert counts == {"pins": 10, "boards": 3}
    with torch.no_grad():
        more = model.eval()(batch)
    assert all(torch.isfinite(term) for term in more.values())
    assert counts == {t: int(sketch.total) for t, sketch in model.sketches.items()}


def test_fp16_training_keeps_softmax_accumulation_in_fp32():
    q = torch.full((1, 2), 400.0, dtype=torch.float16, requires_grad=True)
    p = (-q.detach()).requires_grad_()
    n = q.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.float16):
        loss = sampled_softmax(
            q, p, n, torch.tensor([1]), torch.tensor([2]), CountMinSketch(), scale=1.0, chunk_size=1
        )
    torch.testing.assert_close(loss, torch.tensor(640000.0))
    loss.backward()
    torch.testing.assert_close(q.grad, torch.full_like(q, 800))
    torch.testing.assert_close(p.grad, -q.detach())
    torch.testing.assert_close(n.grad, q.detach())


def test_paper_loaders_match_graphs_and_fp16_training_trajectory(tmp_path):
    import kumo.settings

    pytest.importorskip("diskgraph")
    from kumo.pytorch import api
    from tests.e2e.omnisage_paper_custom import custom_graphs
    from tests.e2e.omnisage_paper_data import make_paper_fixture, paper_batches, paper_selection

    kumo.settings.configure(scratch_dir=tmp_path / "scratch")
    fixture = make_paper_fixture(
        tmp_path / "data", pins=32, steps=2, batch_size=3, history_length=4, random_negatives=3
    )
    engine = api.DiskGraphEngine(num_threads=1, random_seed=42)
    requests = paper_selection(fixture, engine)
    torch.manual_seed(52)
    first = OmniSagePaperModel(PaperConfig.cpu())
    models = (first, deepcopy(first))
    optimizers = [optimizer_for(m, 3, 1) for m in models]
    scalers = [torch.amp.GradScaler("cpu", init_scale=128) for _ in models]
    initial = first.entities["pins"].head[-1].weight.detach().clone()
    source = dict(engine=engine, requests=requests)
    with (
        paper_batches(fixture, **source) as package,
        paper_batches(fixture, graph_loader_factory=custom_graphs, **source) as custom,
    ):
        for a, b in zip(package, custom, strict=True):
            for root in ("pins", "boards"):
                left, right = a.graphs[root], b.graphs[root]
                assert set(left["edges"]) == set(right["edges"])
                for bank, edges in left["edges"].items():
                    for key in ("index", "rank", "score"):
                        torch.testing.assert_close(
                            edges[key], right["edges"][bank][key], rtol=0, atol=0
                        )
                for table, nodes in left["nodes"].items():
                    other = right["nodes"][table]
                    assert torch.equal(nodes["ids"], other["ids"])
                    assert torch.equal(nodes["batch"], other["batch"])
                    assert set(nodes["features"]) == set(other["features"])
                    for field, encoded in nodes["features"].items():
                        for key in ("values", "valid"):
                            assert torch.equal(encoded[key], other["features"][field][key])

            results = []
            for model, batch, (opt, sched), scaler in zip(
                models, (a, b), optimizers, scalers, strict=True
            ):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=torch.float16):
                    terms = model(batch)
                scaler.scale(terms["total"]).backward()
                scaler.unscale_(opt)
                assert all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
                )
                scaler.step(opt)
                scaler.update()
                sched.step()
                results.append(terms)
            for term, value in results[0].items():
                torch.testing.assert_close(
                    value.detach(), results[1][term].detach(), rtol=0, atol=0
                )
            for p, q in zip(*[m.parameters() for m in models], strict=True):
                assert torch.equal(p, q)
    assert not torch.equal(initial, first.entities["pins"].head[-1].weight)
