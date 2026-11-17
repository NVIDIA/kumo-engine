# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent oracles and fixtures for the degree-T (GNN-GI) tests.

Test-only. Nothing here imports `kumo_core.explain` or another test module
(an AST test and a fresh-process import check enforce it). `GatedOracle`
reimplements the gated forward of contract 6.1 from the module parameters
and the raw batch, with its own trimming, frozen mean denominators and walk
enumeration, layer-specific or tied gates, free ReLU or fixed masks, and two
negative controls (gate-weighted denominators, biases). Walk scores are
nested autograd mixed partial derivatives at gates 1 (paper Eq. 7), bag
scores tied-gate derivatives divided by `alpha!` (Prop. 1 and 2), and the
first-order oracle is `grad_{H_0} f * H_0`.
"""
import math
from dataclasses import dataclass

import torch

from kumo_core.modeling.gnn.hetero import (
    ForeignKeyToPrimaryKeyConv,
    HeteroGNN,
    HeteroIdentity,
)
from kumo_core.modeling.heads.node_prediction import NodePredHead

F64 = torch.float64
FK = ('transactions', 'customer_id', 'customer')
REV = ('customer', 'rev_customer_id', 'transactions')
ART_FK = ('transactions', 'article_id', 'article')
ART_REV = ('article', 'rev_article_id', 'transactions')
TABLES = ('customer', 'transactions', 'article')
RELATIONS = (FK, REV, ART_FK, ART_REV)


# Fixtures ###################################################################

@dataclass
class Fixture:
    """A scoring-module stand-in and one retained batch; the seed is row 0
    of the entity table. `nodes` and `hops` are the sampled node and edge
    counts per hop (None: no trimming)."""
    module: torch.nn.Module
    x: dict
    edges: dict
    nodes: dict | None
    hops: dict | None

    def forward(self, x=None):
        model, head = self.module.model, self.module.head
        out = model(dict(self.x if x is None else x), dict(self.edges),
                    None if self.nodes is None else {k: list(v) for k, v in self.nodes.items()},
                    None if self.hops is None else {k: list(v) for k, v in self.hops.items()})
        return head(out[self.module.entity_table_name][:1])

    def sample(self):
        """Primary keys and node ids per occurrence (`100 * table + row`)."""
        nodes = {}
        for index, (table, x) in enumerate(sorted(self.x.items())):
            keys = [100 * (index + 1) + row for row in range(x.size(0))]
            nodes[table] = {'pkeys': keys, 'node_ids': [10 * key for key in keys],
                            'num_nodes': x.size(0)}
        return {'nodes': nodes}


def scoring_module(channels, layers, *, relations=RELATIONS, tables=TABLES, aggr='mean',
                   act='relu', learn_skip=True, pre=0, head_layers=1, gnn_bias=False,
                   head_bias=False, dtype=F64):
    module = torch.nn.Module()
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(channels, layers, metadata=(list(tables), list(relations)),
                             num_pre_mp_layers=pre, aggr=[aggr], norm=None, act=act,
                             bias=gnn_bias, learn_skip=learn_skip)
    module.head = NodePredHead(channels, 1, num_layers=head_layers, act=act, norm=None,
                               param_keys=[], bias=head_bias)
    return module.to(dtype).eval()


def fill(module, seed, scale=0.8, bias_scale=0.3):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            noise = torch.randn(param.shape, generator=generator, dtype=F64).to(param.dtype)
            param.copy_((bias_scale if name.endswith('bias') else scale) * noise)
    return module


def b1(values=None, x=None, edges=None, nodes=None, hops=None, dtype=F64, **options):
    """P0 fixture B1: all weights 1 (biases 0.5), customer 1, transactions 2
    and 3, article 4; `t0 -> c` and `t1 -> c` at hop 1, `c -> t0`, `c -> t1`
    and `a -> t0` at hop 2. Here f = 9 (contract 6.4)."""
    module = scoring_module(1, 2, dtype=dtype, **options)
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.fill_(0.5 if name.endswith('bias') else 1.0)
            if values and name in values:
                param.copy_(torch.as_tensor(values[name], dtype=param.dtype))
    x = x or {'customer': [[1.]], 'transactions': [[2.], [3.]], 'article': [[4.]]}
    x = {table: torch.tensor(rows, dtype=dtype) for table, rows in x.items()}
    default_edges = {FK: [[0, 1], [0, 0]], REV: [[0, 0], [0, 1]], ART_REV: [[0], [0]],
                     ART_FK: [[], []]}
    edges = {rel: torch.tensor(e, dtype=torch.long).reshape(2, -1)
             for rel, e in (edges or default_edges).items()}
    if nodes is None:
        nodes = {'customer': [1, 0, 0], 'transactions': [0, 2, 0], 'article': [0, 0, 1]}
    if hops is None:
        hops = {FK: [2, 0], REV: [0, 2], ART_REV: [0, 1], ART_FK: [0, 0]}
    return Fixture(module, x, edges, nodes, hops)


#: Article of each hop-1 transaction in the Rel-HM shaped fixture (two shared).
RELHM_ARTICLES = (0, 1, 2, 0, 3, 1, 4, 5, 2, 6, 3, 7)


def relhm(seed, channels=4, transactions=8, dtype=F64, scale=0.8, **options):
    """Rel-HM shape at T = 2: the seed customer, its hop-1 transactions, and
    at hop 2 their articles (shared ones deduplicated) and the seed customer
    again (the `rev_customer_id` edges point back to occurrence 0)."""
    articles = RELHM_ARTICLES[:transactions]
    count = max(articles) + 1
    module = fill(scoring_module(channels, 2, dtype=dtype, **options), seed, scale=scale)
    generator = torch.Generator().manual_seed(1000 + seed)
    x = {'customer': torch.rand(1, channels, generator=generator, dtype=F64) + 0.05,
         'transactions': torch.rand(transactions, channels, generator=generator,
                                    dtype=F64) + 0.05,
         'article': torch.rand(count, channels, generator=generator, dtype=F64) + 0.05}
    x = {table: value.to(dtype) for table, value in x.items()}
    rows = list(range(transactions))
    edges = {FK: torch.tensor([rows, [0] * transactions]),
             REV: torch.tensor([[0] * transactions, rows]),
             ART_REV: torch.tensor([list(articles), rows]),
             ART_FK: torch.zeros(2, 0, dtype=torch.long)}
    nodes = {'customer': [1, 0, 0], 'transactions': [0, transactions, 0],
             'article': [0, 0, count]}
    hops = {FK: [transactions, 0], REV: [0, transactions], ART_REV: [0, transactions],
            ART_FK: [0, 0]}
    return Fixture(module, x, edges, nodes, hops)


def t3_trimmed(seed, channels=3, **options):
    """T = 3 deduplicated sample with Engine trimming: seed c0; hop 1 t0, t1;
    hop 2 a0, a1 and c0 again; hop 3 t2, t3 (sources only). Defaults: one
    pre-message-passing layer, identity roots and a two-layer ReLU head."""
    options = {'pre': 1, 'learn_skip': False, 'head_layers': 2, **options}
    module = fill(scoring_module(channels, 3, **options), seed)
    generator = torch.Generator().manual_seed(2000 + seed)
    x = {'customer': torch.rand(1, channels, generator=generator, dtype=F64) + 0.05,
         'transactions': torch.rand(4, channels, generator=generator, dtype=F64) + 0.05,
         'article': torch.rand(2, channels, generator=generator, dtype=F64) + 0.05}
    edges = {FK: torch.tensor([[0, 1], [0, 0]]), REV: torch.tensor([[0, 0], [0, 1]]),
             ART_FK: torch.tensor([[0, 2, 1, 3], [0, 0, 1, 1]]),
             ART_REV: torch.tensor([[0, 1], [0, 1]])}
    nodes = {'customer': [1, 0, 0, 0], 'transactions': [0, 2, 0, 2],
             'article': [0, 0, 2, 0]}
    hops = {FK: [2, 0, 0], REV: [0, 2, 0], ART_FK: [0, 0, 4], ART_REV: [0, 2, 0]}
    return Fixture(module, x, edges, nodes, hops)


def t3_untrimmed(seed, channels=3, signed=True, **options):
    """T = 3 without trimming (every node a destination at every layer).

    Customers c0, c1; transactions t0..t3 of customers (0, 0, 0, 1); article
    a0 receives three `article_id` edges and a1 none (an empty mean or sum
    segment), while a1 still sends a `rev_article_id` edge to t1. Defaults:
    sum aggregation, identity activations and signed `H_0`.
    """
    options = {'aggr': 'sum', 'act': None, **options}
    module = fill(scoring_module(channels, 3, **options), seed, scale=0.7)
    generator = torch.Generator().manual_seed(3000 + seed)

    def draw(rows):
        if signed:
            return torch.randn(rows, channels, generator=generator, dtype=F64)
        return torch.rand(rows, channels, generator=generator, dtype=F64) + 0.05

    x = {'customer': draw(2), 'transactions': draw(4), 'article': draw(2)}
    edges = {FK: torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]]),
             REV: torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]]),
             ART_FK: torch.tensor([[0, 2, 3], [0, 0, 0]]),
             ART_REV: torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]])}
    return Fixture(module, x, edges, None, None)


def precision_relhm(seed, transactions=12, channels=32, dtype=torch.float32):
    """Rel-HM shape whose 32-channel linears see more than 8 rows, so bf16
    internals under `medium` change captured tensors on hosts that have them."""
    return relhm(seed, channels=channels, transactions=transactions, dtype=dtype,
                 scale=1.0 / math.sqrt(channels))


def as_float64(fixture):
    """The same weights and inputs in float64 (exact widening)."""
    import copy
    module = copy.deepcopy(fixture.module).to(F64)
    return Fixture(module, {k: v.to(F64) for k, v in fixture.x.items()}, fixture.edges,
                   fixture.nodes, fixture.hops)


# Oracle #####################################################################

def gate_of(step):
    """Contract 4.5 step key to the oracle's `(gate tensor key, index)`."""
    layer, kind, source, relation, destination, index = step
    if kind == 'self_update':
        return (layer, 'self', destination), index
    return (layer, 'rel', (source, relation, destination)), index


def tied_variable(step):
    """Contract 6.2 tied variable of a step key (the layer index dropped)."""
    _, kind, source, relation, destination, index = step
    if kind == 'self_update':
        return ('self', destination, index)
    return ('rel', (source, relation, destination), index)


class GatedOracle:
    """Gated forward of `HeteroGNN` plus `NodePredHead` from parameters.

    `rows[t][table]` are the destinations of layer t and `count[t][rel]` the
    retained edges, both recomputed here from the hop counts the way
    `HeteroGNN.forward` trims them (destinations drop the rows sampled in the
    last t hops, edges the last t - 1 hops).
    """

    def __init__(self, fixture, output_index=0):
        module = fixture.module
        self.model, self.head = module.model, module.head
        self.entity = module.entity_table_name
        self.seed = (self.entity, 0)
        self.output_index = output_index
        self.T = len(self.model.hetero_convs)
        self.x = {k: v.detach().to(F64) for k, v in fixture.x.items()}
        self.edges = {tuple(k): v.clone() for k, v in fixture.edges.items()}
        self.size = {k: v.size(0) for k, v in self.x.items()}
        T = self.T
        self.rows, self.count = {}, {}
        for t in range(1, T + 1):
            if fixture.nodes is None:
                self.rows[t] = dict(self.size)
            else:
                hops = {k: list(v) + [0] * (T + 1 - len(v)) for k, v in fixture.nodes.items()}
                self.rows[t] = {k: self.size[k] - sum(hops[k][-j] for j in range(1, t + 1))
                                for k in self.size}
            self.count[t] = {}
            for rel, index in self.edges.items():
                if fixture.hops is None:
                    self.count[t][rel] = index.size(1)
                else:
                    hops = list(fixture.hops[rel]) + [0] * (T - len(fixture.hops[rel]))
                    self.count[t][rel] = index.size(1) - sum(hops[-j] for j in range(1, t))
        self.relu = self.model.act is not None
        self.head_relu = self.head.mlp.act is not None

    # Gates ------------------------------------------------------------------
    def gates(self, value=1.0, grad=False, scale=None):
        """Layer-specific gates; `scale` maps a layer to a factor."""
        lam = {}
        for t in range(1, self.T + 1):
            factor = float(value) * (1.0 if scale is None else scale.get(t, 1.0))
            for table, rows in self.rows[t].items():
                lam[(t, 'self', table)] = torch.full((rows, ), factor, dtype=F64,
                                                     requires_grad=grad)
            for rel in self.edges:
                lam[(t, 'rel', rel)] = torch.full((self.count[t][rel], ), factor,
                                                  dtype=F64, requires_grad=grad)
        return lam

    def tied_variables(self):
        mu = {('self', table): torch.ones(rows, dtype=F64, requires_grad=True)
              for table, rows in self.size.items()}
        mu.update({('rel', rel): torch.ones(index.size(1), dtype=F64, requires_grad=True)
                   for rel, index in self.edges.items()})
        return mu

    def tied(self, mu):
        """Layer gates as prefixes of one variable per edge occurrence or
        destination occurrence (trimming keeps prefixes)."""
        lam = {}
        for t in range(1, self.T + 1):
            for table, rows in self.rows[t].items():
                lam[(t, 'self', table)] = mu[('self', table)][:rows]
            for rel in self.edges:
                lam[(t, 'rel', rel)] = mu[('rel', rel)][:self.count[t][rel]]
        return lam

    # Forward ----------------------------------------------------------------
    def _act(self, values, mask, relu):
        if mask is not None:
            return values * mask
        return torch.relu(values) if relu else values

    def forward(self, lam, *, x=None, masks=None, gate_weighted=False, biases=False,
                record=None):
        """Unsigned selected head output on the gated graph.

        `masks` fixes the ReLU patterns (keys `('pre', p, table)`,
        `('mp', t, table)`, `('head', j)`); `record` collects pre-activations
        under the same keys.
        """
        model, head = self.model, self.head
        masks = masks or {}
        H = dict(self.x if x is None else x)
        for p, layer in enumerate(model.lins):  # preprocessing: fixed, never gated
            out = {}
            for table, values in H.items():
                lin = layer.lin_dict[table]
                a = values @ lin.weight.T
                if biases and lin.bias is not None:
                    a = a + lin.bias
                if record is not None:
                    record[('pre', p, table)] = a.detach()
                out[table] = self._act(a, masks.get(('pre', p, table)), self.relu)
            H = out
        for t in range(1, self.T + 1):
            root = model.root_skips[t - 1]
            a = {}
            for table, rows in self.rows[t].items():
                h = H[table][:rows]
                if isinstance(root, HeteroIdentity):
                    branch = h
                else:
                    lin = root.lin_dict[table]
                    branch = h @ lin.weight.T
                    if biases and lin.bias is not None:
                        branch = branch + lin.bias
                a[table] = lam[(t, 'self', table)][:, None] * branch
            for rel, full in self.edges.items():
                src, _, dst = rel
                conv = model.hetero_convs[t - 1][rel]
                if biases and conv.lin.bias is not None:
                    a[dst] = a[dst] + conv.lin.bias  # every destination row of the table
                index = full[:, :self.count[t][rel]]
                if index.size(1) == 0:
                    continue
                gate = lam[(t, 'rel', rel)]
                message = (H[src][index[0]] @ conv.lin.weight.T) * gate[:, None]
                if isinstance(conv, ForeignKeyToPrimaryKeyConv) and _is_mean(conv):
                    weight = gate if gate_weighted else torch.ones_like(gate)
                    degree = torch.zeros(self.rows[t][dst], dtype=F64).index_add(
                        0, index[1], weight)
                    message = message / degree[index[1]][:, None]
                a[dst] = a[dst].index_add(0, index[1], message)
            out = {}
            for table, values in a.items():
                if record is not None:
                    record[('mp', t, table)] = values.detach()
                out[table] = self._act(values, masks.get(('mp', t, table)), self.relu)
            H = out
        z = H[self.entity][:1]
        lins = head.mlp.lins
        for j, lin in enumerate(lins):
            z = z @ lin.weight.T
            if biases and lin.bias is not None:
                z = z + lin.bias
            if j < len(lins) - 1:
                if record is not None:
                    record[('head', j)] = z.detach()
                z = self._act(z, masks.get(('head', j)), self.head_relu)
        return z.reshape(-1)[self.output_index]

    def evaluate(self, lam=None, **options):
        """`forward` as a float, without autograd."""
        with torch.no_grad():
            return float(self.forward(self.gates() if lam is None else lam, **options))

    def masks_at_one(self, x=None):
        """The free ReLU patterns at gates 1 (as Engine captures them, `a > 0`),
        for the layers that have a ReLU."""
        record = {}
        with torch.no_grad():
            self.forward(self.gates(), x=x, record=record)
        gated = {'pre': self.relu, 'mp': self.relu, 'head': self.head_relu}
        return {key: (value > 0).to(F64) for key, value in record.items() if gated[key[0]]}

    # Walks ------------------------------------------------------------------
    def gates_into(self, t, node):
        """`(step key, gate key, index, source)` of every gate into `node`."""
        table, occurrence = node
        into = []
        if occurrence < self.rows[t].get(table, 0):
            into.append(((t, 'self_update', table, '', table, occurrence),
                         (t, 'self', table), occurrence, node))
        for rel in sorted(self.edges):
            if rel[2] != table:
                continue
            index = self.edges[rel]
            for edge in range(self.count[t][rel]):
                if int(index[1, edge]) == occurrence:
                    into.append(((t, 'relation', rel[0], rel[1], rel[2], edge),
                                 (t, 'rel', rel), edge, (rel[0], int(index[0, edge]))))
        return into

    def walks(self):
        """Every walk key (forward order) with its source."""
        found = {}

        def visit(t, node, suffix):
            if t == 0:
                found[tuple(suffix)] = node
                return
            for step, _, _, source in self.gates_into(t, node):
                visit(t - 1, source, [step] + suffix)

        visit(self.T, self.seed, [])
        return found

    def walk_scores(self, sign=1.0, masks=None):
        """`R_W = d^T f / d lambda[1, v_1] ... d lambda[T, v_T]` at gates 1, by
        one nested `autograd.grad` per walk prefix (layer T first)."""
        lam = self.gates(grad=True)
        value = sign * self.forward(lam, masks=masks)
        keys = {t: [key for key in lam if key[0] == t] for t in range(1, self.T + 1)}
        scores = {}

        def visit(t, node, value, suffix):
            if t == 0:
                scores[tuple(suffix)] = float(value.detach())
                return
            grads = {}
            if isinstance(value, torch.Tensor) and value.requires_grad:
                computed = torch.autograd.grad(value, [lam[key] for key in keys[t]],
                                               create_graph=True, allow_unused=True)
                grads = {key: g for key, g in zip(keys[t], computed) if g is not None}
            for step, gate, index, source in self.gates_into(t, node):
                child = grads[gate][index] if gate in grads else torch.zeros((), dtype=F64)
                visit(t - 1, source, child, [step] + suffix)

        visit(self.T, self.seed, value, [])
        return scores

    def bag_score(self, variables, sign=1.0):
        """`(1 / alpha!) d^T f / d mu^alpha` for a bag `{tied variable: alpha}`."""
        mu = self.tied_variables()
        value = sign * self.forward(self.tied(mu))
        factorial = 1
        for (kind, key, index), alpha in sorted(variables.items(), key=repr):
            factorial *= math.factorial(alpha)
            for _ in range(alpha):
                if not value.requires_grad:
                    return 0.0
                grad, = torch.autograd.grad(value, mu[(kind, key)], create_graph=True,
                                            allow_unused=True)
                if grad is None:
                    return 0.0
                value = grad[index]
        return float(value.detach()) / factorial

    def input_gradient_times_input(self, sign=1.0, masks=None):
        """First-order oracle `grad_{H_0} f * H_0` per table, at gates 1."""
        x = {k: v.clone().requires_grad_(True) for k, v in self.x.items()}
        value = sign * self.forward(self.gates(), x=x, masks=masks)
        grads = torch.autograd.grad(value, list(x.values()), allow_unused=True)
        return {k: (torch.zeros_like(v) if g is None else g) * v.detach()
                for (k, v), g in zip(x.items(), grads)}

    def polynomial(self, scores, lam):
        """`sum_W R_W prod_t lambda[t, v_t]` at gate values `lam`."""
        total = []
        for key, score in scores.items():
            product = score
            for step in key:
                gate, index = gate_of(step)
                product *= float(lam[gate][index])
            total.append(product)
        return math.fsum(total)

    def margin(self, x=None):
        """`(min |a|, mixed)` over relevant gated neurons at gates 1: message
        passing at destinations on walks, head hidden neurons, preprocessing
        neurons of walk sources; `mixed` says both ReLU states occur."""
        record = {}
        with torch.no_grad():
            self.forward(self.gates(), x=x, record=record)
        walks = self.walks()
        values = []
        if self.relu:
            for t in range(1, self.T + 1):
                nodes = {self.seed} if t == self.T else set()
                for key in walks:
                    if t < self.T:
                        step = key[t]  # the layer t + 1 step reads H[t] at its source
                        nodes.add(_source_of(step, self))
                for table, occurrence in nodes:
                    values.append(record[('mp', t, table)][occurrence])
            for key, (table, occurrence) in walks.items():
                for p in range(len(self.model.lins)):
                    values.append(record[('pre', p, table)][occurrence])
        if self.head_relu:
            values += [record[('head', j)].reshape(-1)
                       for j in range(len(self.head.mlp.lins) - 1)]
        if not values:
            return None, False
        flat = torch.cat([v.reshape(-1) for v in values])
        return float(flat.abs().min()), bool((flat > 0).any() and (flat < 0).any())


def _source_of(step, oracle):
    """The `H[t - 1]` occurrence a layer-t step reads."""
    layer, kind, source, relation, destination, index = step
    if kind == 'self_update':
        return (destination, index)
    return (source, int(oracle.edges[(source, relation, destination)][0, index]))


def _is_mean(conv):
    aggr = conv.aggr
    if hasattr(aggr, 'aggrs'):
        aggr = aggr.aggrs[0]
    return type(aggr).__name__ == 'MeanAggregation'


def interior_seed(build, seeds=range(80), margin=1e-2):
    """The first seed whose fixture is a nondegenerate interior point: every
    relevant gated pre-activation at least `margin` away from 0, |f| >= 0.1
    and, with ReLU, both mask states present (no vacuous agreement)."""
    for seed in seeds:
        oracle = GatedOracle(build(seed))
        low, mixed = oracle.margin()
        if abs(oracle.evaluate()) < 0.1:
            continue
        if low is None or (low >= margin and (mixed or not oracle.relu)):
            return seed
    raise AssertionError('No interior fixture found')
