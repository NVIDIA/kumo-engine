# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Higher-order walk relevance for :class:`HeteroGNN` predictions.

Implements GNN-GI and GNN-LRP from Schnake et al., "Higher-Order Explanations
of Graph Neural Networks via Relevant Walks" (TPAMI 2022, arXiv:2006.03589).

A walk is a sequence of ``num_mp_layers + 1`` sampled nodes
``(I, J, ..., K)``: node ``I`` at the GNN input, ``J`` after the first message
passing layer, and so on up to node ``K`` whose final embedding enters the
readout. Consecutive nodes are connected by a sampled edge or by the root skip
connection (``J == K``). Each walk receives one signed score. Walk scores
are computed with the paper's masking trick: for each walk prefix, the
gradient is only allowed to flow through the prefix nodes, and the relevance of
all walks sharing that prefix is read off as gradient times input at the GNN
input.

``method='gi'`` differentiates the unmodified forward pass (nested
gradient times input). ``method='lrp'`` differentiates a rewritten forward pass
whose values are identical to the model's, but whose gradient implements LRP:

* Linear maps (message passing, root skip, pre-MP and head layers): LRP-gamma
  generalized to signed inputs, with an epsilon stabilizer. Positive
  contributions are boosted for positive pre-activations and negative
  contributions for negative ones. Biases absorb relevance.
* ``sum``/``mean`` aggregation: linear, so part of the linear map above.
* ``max`` aggregation: each channel is routed to its arg-max neighbor (ties
  split evenly), which is then treated as a linear selection.
* ``layer_norm``: linear map with frozen statistics; the standard deviation is
  detached (Ali et al., ICML 2022). Centering stays linear.
* Activations (e.g. GELU): gate rule ``x * [act(x) / x]``, which passes
  relevance through unchanged.
* Bilinear terms in link prediction heads (``<h_lhs, h_rhs>``): split evenly
  between both factors.

Usage for a hybrid link prediction module and a sampled batch::

    results = explain_hybrid_link(module, batch, seed=0, rhs_ids=[73, 74])
    results[74].topk(5)  # Most positive walks for candidate 74.
    results[74].to_records(batch.n_id_dict)

Other models use :class:`WalkRelevance` with a :class:`WalkReadout`, e.g.
:class:`NodeReadout` for node-level heads.
"""
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from kumoapi.task import TaskType
from torch import Tensor
from torch_geometric.nn import MLP, BatchNorm, LayerNorm
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import scatter

from kumo_core.modeling.gnn.hetero import (
    ForeignKeyToPrimaryKeyConv,
    HeteroGNN,
    HeteroIdentity,
    HeteroLinear,
    HeteroPostLayer,
    PrimaryKeyToForeignKeyConv,
)

Node = tuple[NodeType, int]
Walk = tuple[Node, ...]
WalkMethod = Literal['gi', 'lrp']

_GATE_ACTIVATIONS = (
    torch.nn.GELU,
    torch.nn.ReLU,
    torch.nn.LeakyReLU,
    torch.nn.SiLU,
    torch.nn.ELU,
)


class WalkOps:
    r"""Propagation rules shared by the GNN rewrite and the readouts.

    With ``lrp=False`` every operation is the plain forward computation.
    With ``lrp=True`` values are unchanged but gradients follow LRP.
    """
    def __init__(self, lrp: bool, gamma: float = 0.0,
                 epsilon: float = 1e-6) -> None:
        self.lrp = lrp
        self.gamma = gamma
        self.epsilon = epsilon

    def affine(
        self,
        fn: Callable[[str], Any],
        bias: Any,
        gamma: float | None = None,
    ) -> Any:
        r"""Applies an affine map with the LRP-gamma/epsilon rule.

        ``fn(mode)`` evaluates the bias-free linear part. ``mode`` is a pair
        of characters selecting (inputs, weights): ``'i'`` keeps all values,
        ``'p'`` the positive and ``'n'`` the negative part. ``fn`` may return
        a tensor or a dict of tensors; ``bias`` must have the same structure
        (entries may be ``None``).
        """
        gamma = self.gamma if gamma is None else gamma
        z = _add(fn('ii'), bias)
        if not self.lrp:
            return z
        if gamma > 0:
            pos = _add(fn('pp'), fn('nn'))
            neg = _add(fn('pn'), fn('np'))
            p = _map3(
                lambda z_, a, b: z_ + gamma * torch.where(
                    z_.detach() >= 0, a, b), z, pos, neg)
        else:
            p = z

        def stabilize(z_: Tensor, p_: Tensor) -> Tensor:
            sign = torch.where(p_.detach() >= 0, 1.0, -1.0)
            p_ = p_ + self.epsilon * sign
            return p_ * (z_ / p_).detach()

        if isinstance(z, dict):
            return {key: stabilize(z[key], p[key]) for key in z}
        return stabilize(z, p)

    def linear(self, x: Tensor, lin: torch.nn.Linear,
               gamma: float | None = None) -> Tensor:
        def fn(mode: str) -> Tensor:
            return _sign(x, mode[0]) @ _sign(lin.weight, mode[1]).t()

        return self.affine(fn, lin.bias, gamma)

    def activation(self, x: Tensor, act: torch.nn.Module) -> Tensor:
        if not self.lrp:
            return act(x)
        if not isinstance(act, _GATE_ACTIVATIONS):
            raise NotImplementedError(
                f"Walk relevance does not support activation '{act}'")
        out = act(x)
        safe = torch.where(x == 0, torch.ones_like(x), x)
        gate = torch.where(x == 0, torch.zeros_like(x), out / safe)
        return x * gate.detach()

    def layer_norm(self, x: Tensor, norm: LayerNorm) -> Tensor:
        if norm.mode != 'node':
            raise NotImplementedError(
                f"Walk relevance only supports node-wise layer norm "
                f"(got mode='{norm.mode}')")
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        std = (var + norm.eps).sqrt()
        if self.lrp:
            std = std.detach()
        out = (x - mean) / std
        if norm.weight is not None:
            out = out * norm.weight
        if norm.bias is not None:
            out = out + norm.bias
        return out

    def module(self, x: Tensor, module: torch.nn.Module) -> Tensor:
        r"""Applies a node-wise post-processing module."""
        if isinstance(module, torch.nn.Dropout | torch.nn.Identity):
            return x
        if isinstance(module, LayerNorm):
            return self.layer_norm(x, module)
        if isinstance(module, BatchNorm):
            if module.training:
                raise RuntimeError("Walk relevance requires eval mode")
            return module(x)  # Affine in eval mode.
        return self.activation(x, module)

    def bilinear(self, a: Tensor, b: Tensor) -> Tensor:
        r"""Dot product along the last dimension, attributing half of the
        relevance to each factor (used in both GI and LRP mode)."""
        return 0.5 * ((a * b.detach()).sum(-1) + (a.detach() * b).sum(-1))

    def mlp(self, x: Tensor, mlp: MLP) -> Tensor:
        r"""Applies a :class:`torch_geometric.nn.MLP` in eval mode."""
        if not getattr(mlp, 'plain_last', True):
            raise NotImplementedError("Only 'plain_last' MLPs are supported")
        for lin, norm in zip(mlp.lins[:-1], mlp.norms):
            x = self.linear(x, lin)
            if mlp.act is not None and mlp.act_first:
                x = self.activation(x, mlp.act)
            x = self.module(x, norm)
            if mlp.act is not None and not mlp.act_first:
                x = self.activation(x, mlp.act)
        return self.linear(x, mlp.lins[-1])


class WalkReadout(Protocol):
    r"""Maps final node embeddings to the scalar being explained.

    ``nodes`` lists every node whose final embedding the readout uses.
    """
    nodes: list[Node]

    def __call__(self, h_dict: dict[NodeType, Tensor], ops: WalkOps) -> Tensor:
        ...


@dataclass
class WalkRelevanceResult:
    r"""Signed walk relevance scores for one scalar prediction.

    Attributes:
        walks: Walks as node sequences ordered from the GNN input to the
            readout node. Nodes are ``(node_type, index)`` into the batch.
        relevance: Relevance of each walk.
        score: The explained model output.
        method: ``'gi'`` or ``'lrp'``.
    """
    walks: list[Walk]
    relevance: Tensor
    score: float
    method: str

    @property
    def total(self) -> float:
        return float(self.relevance.sum())

    @property
    def conservation_error(self) -> float:
        r"""``|sum(walk relevance) - score|``; non-zero due to biases,
        layer norm shifts and stabilizers."""
        return abs(self.total - self.score)

    def topk(self, k: int, signed: bool = True) -> list[tuple[Walk, float]]:
        r"""Returns the ``k`` most relevant walks, by signed value or by
        magnitude."""
        key = self.relevance if signed else self.relevance.abs()
        k = min(k, key.numel())
        index = key.topk(k).indices.tolist()
        return [(self.walks[i], float(self.relevance[i])) for i in index]

    def node_relevance(self, position: int = 0) -> dict[Node, float]:
        r"""Pools walk relevance onto the node at ``position`` of each walk.
        ``position=0`` yields an input-node attribution."""
        out: dict[Node, float] = defaultdict(float)
        for walk, rel in zip(self.walks, self.relevance.tolist()):
            out[walk[position]] += rel
        return dict(out)

    def to_records(
        self,
        node_ids: dict[NodeType, Tensor] | None = None,
        pkeys: dict[NodeType, Sequence[Any]] | None = None,
    ) -> list[dict[str, Any]]:
        r"""Converts walks to JSON-serializable records, optionally mapping
        batch indices to global node ids and primary keys."""
        def describe(node: Node) -> dict[str, Any]:
            node_type, index = node
            out: dict[str, Any] = {'table': node_type, 'index': index}
            if node_ids is not None and node_type in node_ids:
                out['node_id'] = int(node_ids[node_type][index])
            if pkeys is not None and node_type in pkeys:
                pkey = pkeys[node_type][index]
                out['pkey'] = pkey.item() if isinstance(pkey, Tensor) else pkey
            return out

        return [{
            'walk': [describe(node) for node in walk],
            'relevance': rel,
        } for walk, rel in zip(self.walks, self.relevance.tolist())]


class WalkRelevance:
    r"""Computes walk relevance for a :class:`HeteroGNN` in eval mode.

    Args:
        gnn: The message passing model.
        method: ``'gi'`` (GNN-GI) or ``'lrp'`` (GNN-LRP).
        gamma: LRP-gamma for linear maps inside the GNN. Either a single value
            or one value per message passing layer (bottom to top). Pre-MP
            layers use the first value.
        epsilon: LRP-epsilon stabilizer.
        head_gamma: LRP-gamma for linear maps in the readout.
        min_prefix_relevance: Skips walk prefixes whose absolute relevance is
            below this fraction of ``|score|``. ``0`` enumerates all walks.
    """
    def __init__(
        self,
        gnn: HeteroGNN,
        method: WalkMethod = 'lrp',
        gamma: float | Sequence[float] = 0.0,
        epsilon: float = 1e-6,
        head_gamma: float = 0.0,
        min_prefix_relevance: float = 0.0,
    ) -> None:
        if method not in ('gi', 'lrp'):
            raise ValueError(f"Unknown walk relevance method '{method}'")
        aggrs = gnn.aggr if isinstance(gnn.aggr, list) else [gnn.aggr]
        unsupported = set(aggrs) - {'sum', 'mean', 'max'}
        if unsupported:
            raise NotImplementedError(
                f"Walk relevance does not support aggregations {unsupported}")
        num_layers = gnn.num_mp_layers
        if isinstance(gamma, (int, float)):
            gamma = [float(gamma)] * num_layers
        if len(gamma) != num_layers:
            raise ValueError(f"Expected {num_layers} gamma values "
                             f"(got {len(gamma)})")
        self.gnn = gnn
        self.method = method
        self.aggrs: list[str] = list(aggrs)
        self.gamma = list(gamma)
        self.epsilon = epsilon
        self.head_gamma = head_gamma
        self.min_prefix_relevance = min_prefix_relevance

    # Forward rewrite #########################################################

    def _ops(self, gamma: float) -> WalkOps:
        return WalkOps(self.method == 'lrp', gamma, self.epsilon)

    def _pre_mp(
        self,
        x_dict: dict[NodeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        ops = self._ops(self.gamma[0] if self.gamma else 0.0)
        for lin, post in zip(self.gnn.lins, self.gnn.post_lins):
            assert isinstance(lin, HeteroLinear)
            x_dict = {
                key: ops.linear(x, lin.lin_dict[key])
                for key, x in x_dict.items()
            }
            x_dict = self._post(post, x_dict, ops)
        return x_dict

    @staticmethod
    def _post(post: HeteroPostLayer, x_dict: dict[NodeType, Tensor],
              ops: WalkOps) -> dict[NodeType, Tensor]:
        out = {}
        for key, x in x_dict.items():
            for module in post.layer_dict[key]:
                x = ops.module(x, module)
            out[key] = x
        return out

    def _mp_layer(
        self,
        i: int,
        h_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
    ) -> dict[NodeType, Tensor]:
        ops = self._ops(self.gamma[i])
        hetero_conv = cast(ModuleDict, self.gnn.hetero_convs[i])
        root_skip = self.gnn.root_skips[i]

        # Arg-max routing weights and degrees, fixed by the forward values:
        max_weight: dict[EdgeType, Tensor] = {}
        deg: dict[EdgeType, Tensor] = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, _, dst = edge_type
            if edge_index.numel() == 0:
                continue
            num_dst = h_dict[dst].size(0)
            msg = h_dict[src].detach()[edge_index[0]]
            ones = msg.new_ones(edge_index.size(1), 1)
            deg[edge_type] = scatter(ones, edge_index[1], 0, num_dst, 'sum')
            if 'max' in self.aggrs:
                amax = scatter(msg, edge_index[1], 0, num_dst, 'max')
                tie = (msg == amax[edge_index[1]]).to(msg.dtype)
                count = scatter(tie, edge_index[1], 0, num_dst, 'sum')
                max_weight[edge_type] = tie / count[edge_index[1]]

        def fn(mode: str) -> dict[NodeType, Tensor]:
            x_mode, w_mode = mode[0], mode[1]
            out: dict[NodeType, Tensor] = {}
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                if edge_index.numel() == 0:
                    continue
                conv = hetero_conv[edge_type]
                num_dst = h_dict[dst].size(0)
                msg = _sign(h_dict[src], x_mode)[edge_index[0]]
                if isinstance(conv, ForeignKeyToPrimaryKeyConv):
                    parts = []
                    for aggr in self.aggrs:
                        if aggr == 'max':
                            msg_aggr = msg * max_weight[edge_type]
                        else:
                            msg_aggr = msg
                        part = scatter(msg_aggr, edge_index[1], 0, num_dst,
                                       'sum')
                        if aggr == 'mean':
                            part = part / deg[edge_type].clamp(min=1)
                        parts.append(part)
                    agg = torch.cat(parts, dim=-1)
                elif isinstance(conv, PrimaryKeyToForeignKeyConv):
                    agg = scatter(msg, edge_index[1], 0, num_dst, 'any')
                else:
                    raise NotImplementedError(
                        f"Unsupported convolution '{type(conv).__name__}'")
                lin = conv.lin
                out_ = agg @ _sign(lin.weight, w_mode).t()
                out[dst] = out[dst] + out_ if dst in out else out_
            for key, h in h_dict.items():
                if isinstance(root_skip, HeteroIdentity):
                    # Identity weights are non-negative:
                    skip = _sign(h, x_mode)
                    if w_mode == 'n':
                        skip = torch.zeros_like(skip)
                else:
                    lin = cast(HeteroLinear, root_skip).lin_dict[key]
                    skip = _sign(h, x_mode) @ _sign(lin.weight, w_mode).t()
                out[key] = out[key] + skip if key in out else skip
            return out

        def biases() -> dict[NodeType, Tensor | None]:
            out: dict[NodeType, Tensor | None] = {key: None for key in h_dict}
            for edge_type in edge_index_dict:
                bias = hetero_conv[edge_type].lin.bias
                out[edge_type[2]] = _add(out[edge_type[2]], bias)
            if isinstance(root_skip, HeteroLinear):
                for key in h_dict:
                    out[key] = _add(out[key], root_skip.lin_dict[key].bias)
            return out

        out_dict = ops.affine(fn, biases())
        return self._post(self.gnn.post_convs[i], out_dict, ops)

    def forward(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
    ) -> list[dict[NodeType, Tensor]]:
        r"""Runs the (rewritten) forward pass and returns the GNN input
        followed by the output of every message passing layer."""
        hs = [x_dict]
        h_dict = self._pre_mp(x_dict)
        for i in range(self.gnn.num_mp_layers):
            h_dict = self._mp_layer(i, h_dict, edge_index_dict)
            hs.append(h_dict)
        return hs

    # Walk enumeration ########################################################

    @torch.no_grad()
    def _in_neighbors(
        self,
        node: Node,
        edge_index_dict: dict[EdgeType, Tensor],
    ) -> list[Node]:
        node_type, index = node
        out: list[Node] = [node]  # Root skip connection.
        seen = {node}
        for (src, _, dst), edge_index in edge_index_dict.items():
            if dst != node_type or edge_index.numel() == 0:
                continue
            for j in edge_index[0, edge_index[1] == index].tolist():
                if (src, j) not in seen:
                    seen.add((src, j))
                    out.append((src, j))
        return out

    def explain(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        readout: WalkReadout,
    ) -> WalkRelevanceResult:
        r"""Explains ``readout`` applied to the GNN output.

        Args:
            x_dict: GNN input features (encoder output plus any additive
                identity or lookup embeddings), indexed by node type.
            edge_index_dict: Sampled edges, indexed by edge type.
            readout: The scalar readout to explain.
        """
        if self.gnn.training:
            raise RuntimeError("Walk relevance requires eval mode")
        num_layers = self.gnn.num_mp_layers
        head_ops = WalkOps(self.method == 'lrp', self.head_gamma,
                           self.epsilon)

        with torch.enable_grad():
            x_dict = {
                key: x.detach().requires_grad_(True)
                for key, x in x_dict.items()
            }
            hs = self.forward(x_dict, edge_index_dict)
            score = readout(hs[-1], head_ops)
            if score.numel() != 1:
                raise ValueError("Readout must return a scalar")
            keys = list(hs[-1].keys())
            grads = torch.autograd.grad(score, [hs[-1][k] for k in keys],
                                        retain_graph=True, allow_unused=True)
            cot_top = {
                k: g if g is not None else torch.zeros_like(hs[-1][k])
                for k, g in zip(keys, grads)
            }

            score_value = float(score.detach())
            threshold = self.min_prefix_relevance * abs(score_value)
            walks: list[Walk] = []
            values: list[Tensor] = []

            def recurse(t: int, cot: dict[NodeType, Tensor],
                        suffix: list[Node]) -> None:
                # `cot` is the cotangent at `hs[t]`, non-zero only at the
                # first node of `suffix`.
                if t == 0:
                    for key, x in hs[0].items():
                        if key not in cot:
                            continue
                        rel = (x.detach() * cot[key]).sum(dim=-1)
                        for i in rel.nonzero().view(-1).tolist():
                            walks.append(tuple([(key, i)] + suffix))
                            values.append(rel[i])
                    return
                outs = [hs[t][k] for k in cot]
                inputs_keys = list(hs[t - 1].keys())
                inputs = [hs[t - 1][k] for k in inputs_keys]
                if t == 1:  # Pre-MP layers are node-wise; skip to input.
                    inputs = [hs[0][k] for k in inputs_keys]
                grads = torch.autograd.grad(outs, inputs,
                                            grad_outputs=list(cot.values()),
                                            retain_graph=True,
                                            allow_unused=True)
                prev = {
                    k: g
                    for k, g in zip(inputs_keys, grads) if g is not None
                }
                if t == 1:
                    recurse(0, prev, suffix)
                    return
                for node in self._in_neighbors(suffix[0], edge_index_dict):
                    key, j = node
                    if key not in prev:
                        continue
                    masked = {key: _mask_row(prev[key], j)}
                    if threshold > 0:
                        rel = float((hs[t - 1][key][j].detach() *
                                     prev[key][j]).sum())
                        if abs(rel) < threshold:
                            continue
                    recurse(t - 1, masked, [node] + suffix)

            for key, g in cot_top.items():
                rows = g.abs().sum(dim=-1).nonzero().view(-1).tolist()
                for k in rows:
                    recurse(num_layers, {key: _mask_row(g, k)}, [(key, k)])

        relevance = (torch.stack(values).detach()
                     if values else torch.zeros(0))
        return WalkRelevanceResult(walks, relevance, score_value, self.method)

    def node_relevance(
        self,
        x_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        readout: WalkReadout,
    ) -> dict[NodeType, Tensor]:
        r"""First-order node attribution under the same rules (one backward
        pass, no walk masks). Equals walk relevance pooled onto the first
        node of each walk."""
        head_ops = WalkOps(self.method == 'lrp', self.head_gamma,
                           self.epsilon)
        with torch.enable_grad():
            x_dict = {
                key: x.detach().requires_grad_(True)
                for key, x in x_dict.items()
            }
            hs = self.forward(x_dict, edge_index_dict)
            score = readout(hs[-1], head_ops)
            keys = list(x_dict.keys())
            grads = torch.autograd.grad(score, [x_dict[k] for k in keys],
                                        allow_unused=True)
        return {
            k: (x_dict[k].detach() * g).sum(dim=-1)
            if g is not None else torch.zeros(x_dict[k].size(0))
            for k, g in zip(keys, grads)
        }


# Readouts ####################################################################


class NodeReadout:
    r"""Explains ``head(h[node])[output_index]`` for node-level tasks.

    Args:
        node: The seed node.
        mlp: A :class:`torch_geometric.nn.MLP` head (e.g.
            ``NodePredHead.mlp``).
        output_index: Output channel to explain.
        extra: Optional constant features appended to the embedding (e.g.
            prediction time encodings).
    """
    def __init__(self, node: Node, mlp: MLP,
                 output_index: int = 0, extra: Tensor | None = None) -> None:
        self.nodes = [node]
        self.mlp = mlp
        self.output_index = output_index
        self.extra = extra

    def __call__(self, h_dict: dict[NodeType, Tensor], ops: WalkOps) -> Tensor:
        node_type, index = self.nodes[0]
        x = h_dict[node_type][index].view(1, -1)
        if self.extra is not None:
            x = torch.cat([x, self.extra.detach().view(1, -1)], dim=-1)
        return ops.mlp(x, self.mlp)[0, self.output_index]


class HybridLinkReadout:
    r"""Explains the score of one (LHS, RHS candidate) pair of a
    :class:`~kumo_core.train.hybrid_link_gnn_module.HybridLinkGNNLightningModule`
    as computed at prediction time.

    The embedding score is ``lhs_projector(h_lhs) . e_rhs +
    lin_offset_embgnn(h_lhs)`` (the offset is skipped if the module has
    none). If the candidate was sampled in the LHS
    subgraph (``rhs_node`` is given), the ID-GNN score ``mlp_head(h_rhs) +
    <h_lhs, h_rhs>`` is computed as well and the larger of both is explained,
    mirroring ``postprocess_eval_step``.

    Args:
        module: The hybrid link prediction module.
        lhs_node: The seed node ``(lhs_table, index)``.
        rhs_id: Global RHS node id of the candidate.
        rhs_node: The candidate's ``(rhs_table, index)`` in the subgraph of
            ``lhs_node``, if sampled.
    """
    def __init__(
        self,
        module: Any,
        lhs_node: Node,
        rhs_id: int,
        rhs_node: Node | None = None,
    ) -> None:
        if getattr(module, 'distance_measure', 'dot_product') != 'dot_product':
            raise NotImplementedError("Only 'dot_product' is supported")
        if not getattr(module.head, 'use_mlp_head', True):
            rhs_node = None  # Shallow-RHS modules only use embeddings.
        self.module = module
        self.lhs_node = lhs_node
        self.rhs_node = rhs_node
        self.nodes = [lhs_node] + ([rhs_node] if rhs_node is not None else [])
        device = next(module.lhs_projector.parameters()).device
        with torch.no_grad():
            index = torch.tensor([rhs_id], dtype=torch.long, device=device)
            self.rhs_emb = module.rhs_embedding(index).view(-1).detach()
        self.branch: str | None = None

    def __call__(self, h_dict: dict[NodeType, Tensor], ops: WalkOps) -> Tensor:
        module = self.module
        lhs_type, lhs_index = self.lhs_node
        h_lhs = h_dict[lhs_type][lhs_index].view(1, -1)
        projector = module.lhs_projector
        lin = getattr(projector, 'projector', projector)
        if getattr(projector, 'normalize', False):
            raise NotImplementedError("Normalized projectors are unsupported")
        emb_score = (ops.linear(h_lhs, lin) * self.rhs_emb).sum()
        offset = getattr(module, 'lin_offset_embgnn', None)
        if offset is not None:  # Absent in older checkpoints.
            emb_score = emb_score + ops.linear(h_lhs, offset).sum()
        self.branch = 'embedding'
        if self.rhs_node is None:
            return emb_score
        rhs_type, rhs_index = self.rhs_node
        h_rhs = h_dict[rhs_type][rhs_index].view(1, -1)
        id_score = ops.mlp(h_rhs, module.head.mlp_head).sum() + \
            ops.bilinear(h_lhs, h_rhs).sum()
        if bool(id_score.detach() >= emb_score.detach()):
            self.branch = 'id_gnn'
            return id_score
        return emb_score


def hybrid_link_inputs(
    module: Any,
    batch: Any,
) -> tuple[dict[NodeType, Tensor], dict[EdgeType, Tensor]]:
    r"""Recomputes the GNN input of a hybrid (or shallow-RHS) link prediction
    module for a sampled batch, mirroring its ``forward``."""
    tf_dict = dict(batch.tf_dict)
    batch_dict = dict(batch.batch_dict)
    tf_dict.pop(module.train_table_name, None)
    batch_dict.pop(module.train_table_name, None)
    seed_time: Tensor | None = None
    if module.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
        seed_time = batch[module.lhs_name].seed_time
    with torch.no_grad():
        x_dict = dict(
            module.encoder(tf_dict, batch_dict,
                           batch.collect('time', allow_empty=True),
                           seed_time))
        batch_size = batch[module.lhs_name].batch_size
        x = x_dict[module.lhs_name].clone()
        x[:batch_size] += module.identity_awareness.weight
        x_dict[module.lhs_name] = x
        if module.rhs_embedding.lookup_embedding is not None:
            n_id = batch.n_id_dict[module.rhs_name]
            emb = module.rhs_projector(
                module.rhs_embedding.lookup_embedding(n_id))
            x_dict[module.rhs_name] = x_dict[module.rhs_name] + emb
    edge_index_dict = batch.collect('edge_index', allow_empty=True)
    return x_dict, edge_index_dict


def explain_hybrid_link(
    module: Any,
    batch: Any,
    seed: int,
    rhs_ids: Sequence[int],
    method: WalkMethod = 'lrp',
    **kwargs: Any,
) -> dict[int, WalkRelevanceResult]:
    r"""Explains the scores of ``rhs_ids`` for one seed of a sampled batch.

    Restricts computation to the seed's (disjoint) subgraph. Walks in the
    returned results index nodes of ``batch``, so
    ``result.to_records(batch.n_id_dict)`` maps them to global node ids.

    Args:
        module: A hybrid link prediction module in eval mode.
        batch: The sampled :class:`~torch_geometric.data.HeteroData` batch.
        seed: Index of the LHS seed within the batch.
        rhs_ids: Global RHS node ids of the candidates to explain.
        method: ``'gi'`` or ``'lrp'``.
        **kwargs: Further arguments of :class:`WalkRelevance`.
    """
    x_dict, edge_index_dict = hybrid_link_inputs(module, batch)
    batch_dict = {key: batch[key].batch for key in x_dict}
    x_dict, edge_index_dict, perm = _seed_subgraph(x_dict, edge_index_dict,
                                                   batch_dict, seed)
    explainer = WalkRelevance(module.model, method, **kwargs)
    lhs_index = int((perm[module.lhs_name] == seed).nonzero())
    lhs_node = (module.lhs_name, lhs_index)
    rhs_n_id = batch[module.rhs_name].n_id[perm[module.rhs_name]]

    out: dict[int, WalkRelevanceResult] = {}
    for rhs_id in rhs_ids:
        match = (rhs_n_id == rhs_id).nonzero().view(-1)
        # Later duplicates overwrite earlier ones in `postprocess_step`:
        rhs_node = ((module.rhs_name, int(match[-1]))
                    if match.numel() > 0 else None)
        readout = HybridLinkReadout(module, lhs_node, int(rhs_id), rhs_node)
        res = explainer.explain(x_dict, edge_index_dict, readout)
        res.walks = [
            tuple((key, int(perm[key][i])) for key, i in walk)
            for walk in res.walks
        ]
        out[int(rhs_id)] = res
    return out


def _seed_subgraph(
    x_dict: dict[NodeType, Tensor],
    edge_index_dict: dict[EdgeType, Tensor],
    batch_dict: dict[NodeType, Tensor],
    seed: int,
) -> tuple[dict[NodeType, Tensor], dict[EdgeType, Tensor],
           dict[NodeType, Tensor]]:
    perm = {
        key: (batch_dict[key] == seed).nonzero().view(-1)
        for key in x_dict
    }
    mapping = {}
    for key, index in perm.items():
        m = torch.full((x_dict[key].size(0), ), -1, dtype=torch.long,
                       device=index.device)
        m[index] = torch.arange(index.numel(), device=index.device)
        mapping[key] = m
    sub_edges = {}
    for (src, rel, dst), edge_index in edge_index_dict.items():
        row, col = mapping[src][edge_index[0]], mapping[dst][edge_index[1]]
        mask = (row >= 0) & (col >= 0)
        sub_edges[(src, rel, dst)] = torch.stack([row[mask], col[mask]])
    sub_x = {key: x_dict[key][index] for key, index in perm.items()}
    return sub_x, sub_edges, perm


# Helpers #####################################################################


def _sign(x: Tensor, mode: str) -> Tensor:
    if mode == 'i':
        return x
    if mode == 'p':
        return x.clamp(min=0)
    return x.clamp(max=0)


def _add(a: Any, b: Any) -> Any:
    if isinstance(a, dict):
        out = dict(a)
        for key, value in (b or {}).items():
            if key in out:
                out[key] = _add(out[key], value)
            elif value is not None:
                out[key] = value
        return out
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _map3(fn: Callable[[Tensor, Tensor, Tensor], Tensor], a: Any, b: Any,
          c: Any) -> Any:
    if isinstance(a, dict):
        return {key: fn(a[key], b[key], c[key]) for key in a}
    return fn(a, b, c)


def _mask_row(x: Tensor, index: int) -> Tensor:
    out = torch.zeros_like(x)
    out[index] = x[index]
    return out
