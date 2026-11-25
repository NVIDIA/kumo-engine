# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-device OmniSage reference for https://arxiv.org/html/2504.17811v3 §§3.2–3.4, A.3."""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PaperConfig:
    # Published transformer/feature-MLP dimensions and depths (Appendix A.3).
    graph_width: int = 768
    graph_heads: int = 12
    feature_width: int = 1024
    user_width: int = 512
    # The paper doesn't give final embedding width, task weights, or temperature values.
    embedding_width: int = 256
    inverse_temperature: float = 10.0
    family_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    source_weights: dict[str, float] = field(default_factory=dict)
    chunk_size: int = 128
    max_history: int = 256

    @classmethod
    def cpu(cls):
        return cls(graph_width=48, feature_width=64, user_width=32, embedding_width=32)


class CountMinSketch(nn.Module):
    """Persistent training-label frequencies for one entity type (§3.3)."""

    def __init__(self, width=8192, depth=4):
        super().__init__()
        generator = torch.Generator().manual_seed(527)
        self.prime = 2147483647
        self.register_buffer("a", torch.randint(1, self.prime, (depth, 1), generator=generator))
        self.register_buffer("b", torch.randint(0, self.prime, (depth, 1), generator=generator))
        self.register_buffer("table", torch.zeros(depth, width, dtype=torch.long))
        self.register_buffer("total", torch.zeros((), dtype=torch.long))

    def _buckets(self, ids):
        return ((self.a * (ids % self.prime) + self.b) % self.prime) % self.table.shape[1]

    @torch.no_grad()
    def observe(self, ids):
        indices = self._buckets(ids)
        self.table.scatter_add_(1, indices, torch.ones_like(indices))
        self.total.add_(ids.numel())

    def log_q(self, ids):
        # Unseen random negatives receive one pseudocount. The common normalizer
        # cancels in each same-type softmax.
        counts = self.table.gather(1, self._buckets(ids)).amin(0)
        return counts.clamp_min(1).float().log() - self.total.clamp_min(1).float().log()


class _ChunkedSoftmax(torch.autograd.Function):
    """Accumulate embedding-level gradients chunk-by-chunk (Algorithm 1)."""

    @staticmethod
    def forward(
        ctx,
        q,
        p,
        n,
        p_log_q,
        n_log_q,
        labels,
        negative_ids,
        owners,
        known_keys,
        cardinality,
        self_ids,
        weights,
        scale,
        chunk_size,
    ):
        dtype = torch.promote_types(q.dtype, torch.float32)
        query, positive, negative = (x.to(dtype) for x in (q, p, n))
        gq, gp, gn = (torch.zeros_like(x) for x in (query, positive, negative))
        total = query.new_zeros(())
        for start in range(0, len(query), chunk_size):
            take = slice(start, start + chunk_size)
            a, b = query[take], positive[take]
            pos = (a * b).sum(1) * scale - p_log_q[take]
            neg = (a @ negative.T) * scale - n_log_q
            invalid = labels[take, None] == negative_ids
            if self_ids is not None:
                invalid |= self_ids[take, None] == negative_ids
            if known_keys is not None:
                pairs = owners[take, None] * cardinality + negative_ids
                loc = torch.searchsorted(known_keys, pairs).clamp_max(len(known_keys) - 1)
                invalid |= known_keys[loc] == pairs
            logits = torch.cat([pos[:, None], neg.masked_fill(invalid, -torch.inf)], dim=1)
            log_prob = logits.log_softmax(dim=1)
            w = weights[take].to(dtype)
            total -= (log_prob[:, 0] * w).sum()
            grad = log_prob.exp()
            grad[:, 0] -= 1
            grad *= (scale * w)[:, None]
            gp[take] = grad[:, :1] * a
            gq[take] = grad[:, :1] * b + grad[:, 1:] @ negative
            gn.addmm_(grad[:, 1:].T, a)
        ctx.save_for_backward(gq.to(q.dtype), gp.to(p.dtype), gn.to(n.dtype))
        return total

    @staticmethod
    def backward(ctx, grad_output):
        return (*(g * grad_output for g in ctx.saved_tensors), *(None,) * 11)


def sampled_softmax(
    q,
    p,
    n,
    positive_ids,
    negative_ids,
    sketch,
    *,
    scale,
    chunk_size,
    weights=None,
    self_ids=None,
    owners=None,
    known=None,
):
    """Equation 1 with both negative pools, per-row weights and accidental-hit masking.

    Optional known pairs are (query occurrence, positive entity ID). Compact ID
    codes keep the query/candidate membership key independent of the PK range.
    """
    if not q.numel() or chunk_size < 1:
        raise ValueError("sampled_softmax needs a positive query count and chunk size")
    if weights is None:
        weights = q.new_full((len(q),), 1 / len(q))
    known_keys = None
    cardinality = 0
    pos_code, neg_code, self_code = positive_ids, negative_ids, self_ids
    if known is not None:
        values = [positive_ids, negative_ids, known[1]]
        if self_ids is not None:
            values.append(self_ids)
        vocabulary = torch.unique(torch.cat(values))
        pos_code, neg_code, known_code = [torch.searchsorted(vocabulary, v) for v in values[:3]]
        cardinality = len(vocabulary)
        known_keys = torch.unique(known[0] * cardinality + known_code)
        if self_ids is not None:
            self_code = torch.searchsorted(vocabulary, self_ids)
    # Matmul would otherwise autocast these FP32 accumulators back to FP16,
    # even after the explicit casts inside the custom autograd function.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _ChunkedSoftmax.apply(
            q,
            p,
            n,
            sketch.log_q(positive_ids),
            sketch.log_q(negative_ids),
            pos_code,
            neg_code,
            owners,
            known_keys,
            cardinality,
            self_code,
            weights,
            scale,
            chunk_size,
        )


def _feature_mlp(in_features, hidden, output):
    # Four affine layers, no layer norm and no output normalization (A.3.2).
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, output),
    )


def _modalities(graph, table):
    features = graph["nodes"][table]["features"]
    if table == "pins":
        packed = features["image"]["values"]
        return {"visual": packed[:, :1024], "pin_text": packed[:, 1024:]}
    return {"board_text": features["annotation"]["values"]}


def _union_edges(graph):
    """Union top-k results per root and typed public ID (§3.1.1)."""
    by_type = {}
    for name in sorted(graph["edges"]):
        edge = graph["edges"][name]
        table = name.split(".")[-1]
        src, dst = edge["index"].long()
        by_type.setdefault(table, []).append((src, dst, edge["score"].flatten()))
    for table, banks in by_type.items():
        # Each scored walk already emits distinct destinations per root.
        if len(banks) == 1:
            yield table, *banks[0]
            continue
        src, dst, score = (torch.cat(parts) for parts in zip(*banks, strict=True))
        if len(src):
            ids = graph["nodes"][table]["ids"][dst]
            # The paper doesn't specify a score for overlaps. Retain the strongest
            # score, breaking exact ties by canonical bank name and sampler rank.
            order = torch.argsort(score, descending=True, stable=True)
            order = order[torch.argsort(ids[order], stable=True)]
            order = order[torch.argsort(src[order], stable=True)]
            first = torch.ones_like(order, dtype=torch.bool)
            first[1:] = (src[order][1:] != src[order][:-1]) | (ids[order][1:] != ids[order][:-1])
            keep = order[first].sort().values
            src, dst, score = src[keep], dst[keep], score[keep]
        yield table, src, dst, score


class EntityEmbedder(nn.Module):
    """Per-entity-type CLS encoder with learned score gates and descending-score ordering."""

    def __init__(self, root_type, config):
        super().__init__()
        self.root_type = root_type
        h = config.graph_width
        self.own = nn.ModuleDict(
            {
                k: nn.Sequential(nn.Linear(d, h), nn.ReLU())
                for k, d in _MODALITIES[root_type].items()
            }
        )
        self.own_degree = nn.Linear(1, h)
        self.neighbor = nn.ModuleDict(
            {
                t: nn.Sequential(nn.Linear(sum(dims.values()), h), nn.ReLU(), nn.Linear(h, h))
                for t, dims in _MODALITIES.items()
            }
        )
        self.score_gate = nn.Linear(1, h)
        self.cls = nn.Parameter(torch.empty(1, 1, h))
        # Up to 25 + 75 + 50 neighbors in §4.1 plus CLS, self modalities and degree.
        self.position = nn.Parameter(torch.empty(1, 154, h))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        self.transformer = nn.TransformerEncoderLayer(
            h, config.graph_heads, 4 * h, dropout=0.0, activation="relu", batch_first=True
        )
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, config.embedding_width))

    def forward(self, graph, count, edges):
        own = _modalities(graph, self.root_type)
        degrees = graph["nodes"][self.root_type]["features"]["degree"]["values"]
        tokens = [self.cls.expand(count, -1, -1)]
        tokens.extend(self.own[k](x[:count])[:, None] for k, x in own.items())
        tokens.append(self.own_degree(degrees[:count].float().log1p().reshape(-1, 1))[:, None])
        owners, values, scores = [], [], []
        for table, src, dst, score in edges:
            raw = torch.cat([v[dst] for v in _modalities(graph, table).values()], dim=1)
            values.append(self.neighbor[table](raw) * self.score_gate(score.log1p()[:, None]))
            owners.append(src)
            scores.append(score)
        self_tokens = torch.cat(tokens, dim=1)
        seq = self_tokens
        mask = torch.zeros(count, self_tokens.shape[1], dtype=torch.bool, device=seq.device)
        if owners and sum(len(s) for s in owners):
            owner, score, x = torch.cat(owners), torch.cat(scores), torch.cat(values)
            # Stable ties retain canonical bank order and the sampler's local rank.
            order = torch.argsort(score, descending=True, stable=True)
            order = order[torch.argsort(owner[order], stable=True)]
            owner = owner[order]
            lengths = owner.bincount(minlength=count)
            slots = (
                torch.arange(len(owner), device=owner.device) - (lengths.cumsum(0) - lengths)[owner]
            )
            padded = x.new_zeros(count, int(lengths.max()), x.shape[1])
            padded[owner, slots] = x[order]
            seq = torch.cat((self_tokens, padded), dim=1)
            mask = torch.cat(
                (mask, torch.arange(padded.shape[1], device=x.device) >= lengths[:, None]), 1
            )
        hidden = self.transformer(seq + self.position[:, : seq.shape[1]], src_key_padding_mask=mask)
        return F.normalize(self.head(hidden[:, 0]), dim=1)


_MODALITIES = {"pins": {"visual": 1024, "pin_text": 128}, "boards": {"board_text": 128}}


class UserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.user_width
        self.projection = nn.Linear(config.embedding_width, h)
        self.position = nn.Parameter(torch.empty(config.max_history, h))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                h, 4, 4 * h, dropout=0.0, activation="gelu", batch_first=True
            )
            for _ in range(4)
        )
        self.output = nn.Linear(h, config.embedding_width)

    def forward(self, history, valid):
        length = history.shape[1]
        if length > len(self.position):
            raise ValueError("history exceeds configured maximum")
        x = self.projection(history) + self.position[:length]
        # Right padding can never reach a valid prefix through causal attention.
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        for layer in self.layers:
            x = layer(x, src_mask=causal)
        return F.normalize(self.output(x), dim=-1).masked_fill(~valid[:, :, None], 0)


def _selected_edges(src, dst, roots):
    """Collect each selected root's neighbors, retaining repeated selections."""
    if not len(src):
        return src, dst
    order = torch.argsort(src, stable=True)
    ordered = src[order]
    starts = torch.searchsorted(ordered, roots)
    counts = torch.searchsorted(ordered, roots, right=True) - starts
    owner = torch.repeat_interleave(torch.arange(len(roots), device=roots.device), counts)
    slots = torch.arange(len(owner), device=owner.device) - (counts.cumsum(0) - counts)[owner]
    return owner, dst[order[starts[owner] + slots]]


class OmniSagePaperModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PaperConfig()
        c = self.config
        self.entities = nn.ModuleDict({t: EntityEmbedder(t, c) for t in _MODALITIES})
        self.feature_heads = nn.ModuleDict(
            {
                name: _feature_mlp(size, c.feature_width, c.embedding_width)
                for name, size in {
                    "visual": 1024,
                    "pin_text": 128,
                    "board_text": 128,
                    "degree": 1,
                }.items()
            }
        )
        self.users = UserEncoder(c)
        self.sketches = nn.ModuleDict({t: CountMinSketch() for t in _MODALITIES})

    def _loss(self, batch, embs, table, q, positive, in_batch, **kwargs):
        negatives = torch.cat((in_batch, batch.negatives[table]))
        ids = batch.root_ids[table]
        return sampled_softmax(
            q,
            embs[table][positive],
            embs[table][negatives],
            ids[positive],
            ids[negatives],
            self.sketches[table],
            scale=self.config.inverse_temperature,
            chunk_size=self.config.chunk_size,
            **kwargs,
        )

    def _feature_queries(self, graph, table, roots, edges):
        qs, owners = [], []
        root_owner = torch.arange(len(roots), device=roots.device)

        def add(fields, selected, owner):
            for name, x in fields.items():
                qs.append(self.feature_heads[name](x[selected]))
                owners.append(owner)

        degree = (
            graph["nodes"][table]["features"]["degree"]["values"].float().log1p().reshape(-1, 1)
        )
        add({**_modalities(graph, table), "degree": degree}, roots, root_owner)
        for neighbor, src, dst, _ in edges:
            owner, indices = _selected_edges(src, dst, roots)
            add(_modalities(graph, neighbor), indices, owner)
        return torch.cat(qs), torch.cat(owners)

    def forward(self, batch):
        c = self.config
        neighborhoods = {t: list(_union_edges(g)) for t, g in batch.graphs.items()}
        embs = {
            t: module(batch.graphs[t], len(batch.root_ids[t]), neighborhoods[t])
            for t, module in self.entities.items()
        }
        history, future = batch.history, batch.future
        valid = history >= 0
        steps = valid[:, :-1] & valid[:, 1:]
        next_pos = history[:, 1:][steps]
        has_future = valid.any(1) & (future >= 0)
        future_pos = future[has_future]

        # Count sampled label occurrences once per source observation. Reuse the same
        # sketch snapshot for all terms; random negatives never update the sketch.
        if self.training:
            observed = {t: [r] for t, r in batch.features.items()}
            for _, _, t, p in batch.pairs.values():
                observed[t].append(p)
            observed["pins"].extend((next_pos, future_pos))
            for t, selections in observed.items():
                self.sketches[t].observe(batch.root_ids[t][torch.cat(selections)])

        losses = {}
        zero = embs["pins"].sum() * 0
        pair = zero
        for name, (qt, q, pt, p) in batch.pairs.items():
            if len(q):
                # Mask every known positive for this query occurrence and typed target,
                # including overlaps between graph and engagement examples.
                matches = [
                    (qs, batch.root_ids[pt][ps])
                    for u, qs, v, ps in batch.pairs.values()
                    if u == qt and v == pt
                ]
                known = tuple(torch.cat(v) for v in zip(*matches, strict=True))
                self_ids = batch.root_ids[qt][q] if qt == pt else None
                value = self._loss(
                    batch, embs, pt, embs[qt][q], p, p, owners=q, known=known, self_ids=self_ids
                )
            else:
                value = zero
            losses["pair/" + name] = value
            weight = c.source_weights.get(name, 1 / len(batch.pairs))
            pair = pair + weight * value

        feature = zero
        nroots = sum(map(len, batch.features.values()))
        for table, roots in batch.features.items():
            if len(roots):
                queries, owners = self._feature_queries(
                    batch.graphs[table], table, roots, neighborhoods[table]
                )
                weights = 1 / (owners.bincount(minlength=len(roots))[owners].float() * nroots)
                feature = feature + self._loss(
                    batch, embs, table, queries, roots[owners], roots, weights=weights
                )

        user = self.users(embs["pins"][history.clamp_min(0)], valid)
        if len(next_pos):
            # Eq. 4 averages per sequence after summing its unpadded positions.
            w = user.new_ones(len(next_pos)) / history.shape[0]
            next_loss = self._loss(
                batch, embs, "pins", user[:, :-1][steps], next_pos, next_pos, weights=w
            )
        else:
            next_loss = zero
        if len(future_pos):
            mask = valid & has_future[:, None]
            row = torch.where(mask)[0]
            w = user.new_ones(len(row)) / len(future_pos)
            future_loss = self._loss(
                batch, embs, "pins", user[mask], future[row], future_pos, weights=w
            )
        else:
            future_loss = zero
        a, b, d = c.family_weights
        return {
            **losses,
            "pair": pair,
            "feature": feature,
            "next": next_loss,
            "future": future_loss,
            "total": a * pair + b * feature + d * (next_loss + future_loss),
        }


def optimizer_for(model, steps, warmup_steps):
    """Appendix A.3.3; durations, warmup ratio and weight decay are run parameters."""
    if not 0 < warmup_steps < steps:
        raise ValueError("supply at least one warmup step and one cosine step")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    def factor(i):
        if i < warmup_steps:
            return (i + 1) / warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * (i - warmup_steps + 1) / (steps - warmup_steps + 1)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
