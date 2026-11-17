# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable fixtures for binary walk evidence (frozen with P2).

Two kinds of fixtures live here:

- End-to-end tables and pinned plans for tiny Engine fits. They use only
  public APIs that already exist at 1909c5b9, so the same builders produce
  base-wheel bundles for compatibility oracles.
- Small CPU scoring modules with a real `HeteroGNN`, `NodePredHead`,
  `BinaryCrossEntropyLoss`, `CategoricalTargetEncoder` and `PlatScaling`,
  for unit tests that do not train.

Nothing here asserts behavior; tests own their expectations.
"""
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

BINARY_QUERY = 'PREDICT users.churn FOR EACH users.id'
REGRESSION_QUERY = 'PREDICT entities.target FOR EACH entities.id'


# End-to-end tables and plans #################################################

def write_binary_tables(data_dir, *, seed=11, users=128, events_per_user=10,
                        items=24):
    """Write users, events and items parquet files for a binary churn task.

    `churn = age > 35` with 5% flipped labels, so `True` is the majority
    label, the fitted mapping is `[True, False]` and the target encoder is
    reversed. The first two events are identical apart from their keys, so
    duplicate records must stay distinct occurrences.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 70, users)
    churn = (age > 35) ^ (rng.random(users) < 0.05)
    count = users * events_per_user
    item_id = rng.integers(0, items, count)
    units = rng.uniform(1, 20, count)
    item_id[1], units[1] = item_id[0], units[0]
    pd.DataFrame({'id': np.arange(users), 'age': age,
                  'churn': churn}).to_parquet(data_dir / 'users.parquet')
    pd.DataFrame({'id': np.arange(count),
                  'user_id': np.repeat(np.arange(users), events_per_user),
                  'item_id': item_id,
                  'units': units}).to_parquet(data_dir / 'events.parquet')
    pd.DataFrame({'id': np.arange(items),
                  'price': rng.uniform(1, 100, items)}).to_parquet(
                      data_dir / 'items.parquet')
    return {'users': users, 'events': count, 'items': items,
            'churn_true': int(churn.sum())}


def binary_graph(kumo, connector):
    from kumoapi.typing import Stype
    users = kumo.Table.from_source_table(connector['users.parquet'],
                                         primary_key='id').infer_metadata()
    users = users.with_column('age', stype=Stype.numerical)
    events = kumo.Table.from_source_table(connector['events.parquet'],
                                          primary_key='id').infer_metadata()
    events = (events.with_column('user_id', stype=Stype.ID).with_column(
        'item_id', stype=Stype.ID).with_column('units', stype=Stype.numerical))
    items = kumo.Table.from_source_table(connector['items.parquet'],
                                         primary_key='id').infer_metadata()
    items = items.with_column('price', stype=Stype.numerical)
    return kumo.Graph(
        tables={'users': users, 'events': events, 'items': items},
        edges=[kumo.Edge(src_table='events', fkey='user_id',
                         dst_table='users'),
               kumo.Edge(src_table='events', fkey='item_id',
                         dst_table='items')])


def pinned_plan(*, loss, hops, channels=32, aggregation='mean', epochs=4,
                steps=8, batch_size=16, seed=42):
    """A single-trial plan in the contract 1.6 profile shape."""
    from kumoapi.model_plan import GNNModelPlan, ModelPlan
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.training_job.manual_seed = seed
    plan.training_job.tune_metric = 'loss'
    plan.optimization.loss = [loss]
    plan.optimization.max_epochs = epochs
    plan.optimization.max_steps_per_epoch = steps
    plan.optimization.batch_size = [batch_size]
    plan.neighbor_sampling.num_neighbors = [list(hops)]
    plan.model_architecture.model = [
        GNNModelPlan(channels=[channels], aggregation=[[aggregation]],
                     dropout=[0.0])
    ]
    plan.model_architecture.activation = ['relu']
    plan.model_architecture.normalization = [None]
    plan.model_architecture.num_pre_message_passing_layers = [0]
    plan.model_architecture.num_post_message_passing_layers = [1]
    plan.model_architecture.use_seq_id = [False]
    # The fixtures are static tasks, which reject `prediction_time_encodings`
    # (the head then defaults to no prediction-time encoding).
    return plan


def binary_plan(**kwargs):
    kwargs.setdefault('hops', (10, 10))
    return pinned_plan(loss='binary_cross_entropy', **kwargs)


def write_regression_tables(data_dir, *, seed=5, entities=32,
                            events_per_entity=12):
    """Regression fixture whose 32-channel linears see more than 8 rows."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(1, 20, entities)
    pd.DataFrame({'id': np.arange(entities), 'quantity': values,
                  'target': 3 * values}).to_parquet(data_dir /
                                                    'entities.parquet')
    count = entities * events_per_entity
    pd.DataFrame({'id': np.arange(count),
                  'entity_id': np.repeat(np.arange(entities),
                                         events_per_entity),
                  'units': rng.uniform(1, 20, count)}).to_parquet(
                      data_dir / 'events.parquet')
    return {'entities': entities, 'events': count}


def regression_graph(kumo, connector):
    from kumoapi.typing import Stype
    entity = kumo.Table.from_source_table(connector['entities.parquet'],
                                          primary_key='id').infer_metadata()
    entity = entity.with_column('quantity', stype=Stype.numerical).with_column(
        'target', stype=Stype.numerical)
    event = kumo.Table.from_source_table(connector['events.parquet'],
                                         primary_key='id').infer_metadata()
    event = event.with_column('entity_id', stype=Stype.ID).with_column(
        'units', stype=Stype.numerical)
    return kumo.Graph(tables={'entities': entity, 'events': event},
                      edges=[kumo.Edge(src_table='events', fkey='entity_id',
                                       dst_table='entities')])


def regression_plan(**kwargs):
    kwargs.setdefault('hops', (12, 12))
    kwargs.setdefault('epochs', 1)
    kwargs.setdefault('steps', 2)
    return pinned_plan(loss='mse', **kwargs)


# Unit-test scoring modules ###################################################

FK = ('transactions', 'customer_id', 'customer')
REV = ('customer', 'rev_customer_id', 'transactions')
ART_REV = ('article', 'rev_article_id', 'transactions')
ART_FK = ('transactions', 'article_id', 'article')
B1_TABLES = ('customer', 'transactions', 'article')


def b1_batch(dtype=torch.float64, order=None, x=None):
    """P0 fixture B1 inputs: customer 1, transactions 2 and 3, article 4."""
    x = x or {'customer': [[1.]], 'transactions': [[2.], [3.]], 'article': [[4.]]}
    x = {table: torch.tensor(values, dtype=dtype) for table, values in x.items()}
    edges = {FK: torch.tensor([[0, 1], [0, 0]]),
             REV: torch.tensor([[0, 0], [0, 1]]),
             ART_REV: torch.tensor([[0], [0]]),
             ART_FK: torch.zeros(2, 0, dtype=torch.long)}
    if order is not None:
        edges = {key: edges[key] for key in order}
    nodes = {'customer': [1, 0, 0], 'transactions': [0, 2, 0],
             'article': [0, 0, 1]}
    nse = {FK: [2, 0], REV: [0, 2], ART_REV: [0, 1], ART_FK: [0, 0]}
    return x, edges, nodes, nse


def b1_module(dtype=torch.float64, *, gnn_bias=False, head_bias=False,
              root_weight=1.0, head_weight=1.0):
    """B1 GNN (mean, learned roots, all weights 1) and a one-layer head."""
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from kumo_core.modeling.heads.node_prediction import NodePredHead
    module = cast(Any, torch.nn.Module())  # a scoring-module stand-in
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(1, 2, metadata=(list(B1_TABLES),
                                             [FK, REV, ART_REV, ART_FK]),
                             aggr=['mean'], norm=None, bias=gnn_bias,
                             learn_skip=True)
    module.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[])
    if not head_bias:
        module.head.mlp.lins[0].register_parameter('bias', None)
    head = cast(Any, module.head).mlp.lins[0]
    with torch.no_grad():
        for param in module.parameters():
            param.fill_(1.0)
        for root in module.model.root_skips:
            for lin in cast(Any, root).lin_dict.values():
                lin.weight.fill_(root_weight)
        head.weight.fill_(head_weight)
        if gnn_bias:
            for name, param in module.model.named_parameters():
                if name.endswith('bias'):
                    param.fill_(0.5)
        if head_bias:
            head.bias.fill_(-0.25)
    return module.to(dtype).eval()


def b1_forward(module: Any, batch):
    x, edges, nodes, nse = batch
    out = module.model(dict(x), dict(edges),
                       {k: list(v) for k, v in nodes.items()},
                       {k: list(v) for k, v in nse.items()})
    return module.head(out[module.entity_table_name][:1])


def target_encoder(labels_with_counts, task='binary'):
    """A fitted target encoder built from `[[label, count], ...]`."""
    from kumo_core.encoders.target.base import TargetEncoder
    from kumoapi.encoder import Index
    from kumoapi.task import TaskType
    from kumoapi.typing import ColStatType
    encoder = Index()
    encoder._stats = {ColStatType.CATEGORY_COUNTS: labels_with_counts}
    return TargetEncoder.from_encoder(TaskType.BINARY_CLASSIFICATION, encoder)


class B1Encoder(torch.nn.Module):
    """Returns the fixed B1 `H_0` rows; stands in for a fitted encoder."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        x, _, _, _ = b1_batch(self.dtype)
        return {**x, 'task': torch.zeros(1, 1, dtype=self.dtype)}


def binary_scoring_module(*, counts=((True, 123), (False, 37)),
                          dtype=torch.float32, calibration=None,
                          gnn_bias=False, head_bias=False, head_weight=1.0,
                          loss='bce', batch_order=None):
    """A binary scoring module on fixture B1 with `common_step(batch)`.

    `calibration` is `None` (unfitted Platt) or `(weight_raw, bias)`.
    `common_step` ignores its argument and scores the fixed B1 batch, so
    capture and replay see one seed with eight walks.
    """
    from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
    from kumo_core.modeling.calibration import PlatScaling
    from kumo_core.train.loss import BinaryCrossEntropyLoss, FocalLoss
    from kumo_core.types import TaskType
    module: Any = b1_module(dtype, gnn_bias=gnn_bias, head_bias=head_bias,
                            head_weight=head_weight)
    module.task_type = TaskType.BINARY_CLASSIFICATION
    module.loss = (BinaryCrossEntropyLoss(TaskType.BINARY_CLASSIFICATION)
                   if loss == 'bce' else FocalLoss(
                       TaskType.BINARY_CLASSIFICATION))
    module.target_encoder = target_encoder([list(c) for c in counts])
    assert isinstance(module.target_encoder, CategoricalTargetEncoder)
    module.calibrator = PlatScaling(TaskType.BINARY_CLASSIFICATION)
    if calibration is not None:
        with torch.no_grad():
            module.calibrator.weight_and_bias.copy_(
                torch.tensor([[calibration[0]], [calibration[1]]]))
        module.calibrator.set_extra_state({'is_fitted': True})
    module.calibrator.to(dtype)
    module.encoder = B1Encoder(dtype)
    module.device = torch.device('cpu')
    module.eval()
    default = b1_batch(dtype, order=batch_order)

    def common_step(batch=None):
        # A replayed batch may carry tampered B1 features as `b1_x`.
        features = getattr(batch, 'b1_x', None)
        return b1_forward(module, default if features is None else
                          b1_batch(dtype, order=batch_order, x=features))

    module.common_step = common_step
    return module


def emitted_frame(module, logits):
    """Exactly what `BaseNodeLightningModule.on_predict_batch_end` emits."""
    if module.calibrator.is_fitted:
        logits = module.calibrator(logits)
    prob = module.loss.to_probability(logits).cpu()
    return SimpleNamespace(false_prob=prob[:, 0].float(),
                           true_prob=prob[:, 1].float())


# Native capture versus offline derivation ###################################

#: Fields that legitimately differ between a native row and its replay.
REPLAY_FIELDS = ('origin', 'numerics', 'replay_logit_abs_diff',
                 'replay_probability_abs_diff')


def comparable_row(row):
    """A walk row without the fields that differ on offline replay."""
    return {key: value for key, value in row.items() if key not in REPLAY_FIELDS}


def b1_sample():
    """A retained-graph stand-in for fixture B1 (keys and node ids only)."""
    return {'origin': 'original_forward', 'seed_table': 'customer',
            'train_table_name': 'task', 'entity_table_name': 'customer',
            'nodes': {'customer': {'pkeys': [17], 'node_ids': [170], 'num_nodes': 1},
                      'transactions': {'pkeys': [41, 42], 'node_ids': [410, 420],
                                       'num_nodes': 2},
                      'article': {'pkeys': [9], 'node_ids': [90], 'num_nodes': 1}},
            'edges': []}


def seed_batch(entity='customer', input_id=8):
    from torch_geometric.data import HeteroData
    batch = HeteroData()
    setattr(batch, 'input_type', entity)
    batch[entity].batch_size = 1
    batch[entity].input_id = torch.tensor([input_id])
    return batch


def capture_record(module: Any, root, config=None, *, sample=None, pkey=17,
                   numerical_sensitivity=False, on_unsupported='raise'):
    """Run `LocalSensitivity` around one `module.common_step` forward.

    Returns the record, the native manifest, the logits, the batch and the
    collector. `config` is a `WalkAttributionConfig` or None.
    """
    import json

    from kumo_core.explain.local_sensitivity import LocalSensitivity
    sample = b1_sample() if sample is None else sample
    provenance = {'run_id': 'r', 'sampling_plan': {},
                  'model_reconstruction': {'task': {'task_type': module.task_type.value}}}
    collector = LocalSensitivity(
        str(root), provenance, numerical_sensitivity=numerical_sensitivity,
        walk_attribution=None if config is None else config.to_dict(),
        walk_on_unsupported=on_unsupported)
    data = SimpleNamespace(base_seed=7, sampling_plan={})
    collector.explain_start(module, data, None)
    setattr(collector, '_sample', lambda *args: sample)
    batch = seed_batch(module.entity_table_name)
    collector.explain_pre_step(batch, module, data, None, 0)
    logits = module.common_step(batch)
    output = {'logits': logits, 'pkey': np.array([pkey]), 'time': None}
    collector.explain_post_step(batch, module, data, output, None, 0)
    collector.explain_end(module, data, {}, None)
    identifier = output['prediction_id'][0]
    record = json.loads((root / 'rows' / f'{identifier}.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    return SimpleNamespace(record=record, manifest=manifest, logits=logits,
                           sample=sample, batch=batch, collector=collector)


def derive_offline(captured, module, config, output, *, numerics: Any = 'recorded'):
    """Derive a captured record offline with `attribute_bundle` and `module`.

    `numerics` is 'recorded' (keep the native record), None (a bundle from
    before the record existed) or a replacement dict. Model loading and batch
    reconstruction are routed to the fixture module for the duration of the
    call only, so tests never patch these entry points themselves.
    """
    import kumo_core.explain.local_sensitivity as local
    import kumo_core.explain.perturbation as perturbation

    from kumo.engine.evidence.walks import attribute_bundle
    parent = output.parent / f'{output.name}-parent'
    parent.mkdir(parents=True)
    (parent / 'manifest.json').write_text('{}')
    native = {**captured.manifest, 'bundle_sha256': 'b' * 64}
    if numerics is None:
        native.pop('numerics', None)
    elif numerics != 'recorded':
        native['numerics'] = numerics
    bundle = SimpleNamespace(
        root=parent, validate=lambda: None,
        manifest={'native_provenance': native}, iter_predictions=lambda: iter([{}]),
        iter_numerical_evidence=lambda: iter([captured.record]),
        scoring_graph=lambda _: captured.sample)
    saved = local.load_scoring_module, perturbation.reconstruct_batch
    local.load_scoring_module = lambda *args: module
    perturbation.reconstruct_batch = lambda _: captured.batch
    try:
        return attribute_bundle(bundle, model='trusted', config=config, output=output)
    finally:
        local.load_scoring_module, perturbation.reconstruct_batch = saved


def native_offline_parity(module, config, root, **options):
    """Native row and offline row of one forward of any registered method.

    Returns `(native_row, offline_row, offline_bundle)`; compare the rows
    with `comparable_row`.
    """
    captured = capture_record(module, root / 'native', config, **options)
    offline = derive_offline(captured, module, config, root / 'offline')
    return captured.record['walk_attribution'], next(offline.iter_rows()), offline


# Replay precision ###########################################################

EVENT = ('events', 'entity_id', 'entities')
REV_EVENT = ('entities', 'rev_entity_id', 'events')


def precision_module(*, events=12, channels=32, seed=3):
    """A deterministic regression scorer whose 32-channel linears see more
    than 8 rows (12 events), so `medium` and `highest` differ on hosts with
    bf16 matmul support. Rebuilt identically in any process from `seed`."""
    from kumo_core.encoders.target.numerical import NumericalTargetEncoder
    from kumo_core.modeling.gnn.hetero import HeteroGNN
    from kumo_core.modeling.heads.node_prediction import NodePredHead
    from kumo_core.train.loss import MSELoss
    from kumo_core.types import TaskType
    generator = torch.Generator().manual_seed(seed)
    module = cast(Any, torch.nn.Module())  # a scoring-module stand-in
    module.entity_table_name = 'entities'
    module.train_table_name = 'task'
    module.model = HeteroGNN(channels, 2, metadata=(['entities', 'events'],
                                                    [EVENT, REV_EVENT]),
                             aggr=['mean'], norm=None, bias=True)
    module.head = NodePredHead(channels, 1, num_layers=1, norm=None, param_keys=[])
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) / channels ** 0.5)
    module.task_type = TaskType.REGRESSION
    module.loss = MSELoss(TaskType.REGRESSION)
    module.target_encoder = NumericalTargetEncoder(0, 1)
    module.encoder = torch.nn.Identity()
    module.device = torch.device('cpu')
    module.eval()
    x = {'entities': torch.rand(1, channels, generator=generator),
         'events': torch.rand(events, channels, generator=generator)}
    edges = {EVENT: torch.stack([torch.arange(events), torch.zeros(events, dtype=torch.long)]),
             REV_EVENT: torch.stack([torch.zeros(events, dtype=torch.long),
                                     torch.arange(events)])}
    nodes = {'entities': [1, 0, 0], 'events': [0, events, 0]}
    edge_counts = {EVENT: [events, 0], REV_EVENT: [0, events]}

    def common_step(_=None):
        out = module.model(dict(x), dict(edges), {k: list(v) for k, v in nodes.items()},
                           {k: list(v) for k, v in edge_counts.items()})
        return module.head(out['entities'][:1])

    module.common_step = common_step
    return module


def precision_sample(events=12):
    return {'origin': 'original_forward', 'seed_table': 'entities',
            'train_table_name': 'task', 'entity_table_name': 'entities',
            'nodes': {'entities': {'pkeys': [1], 'node_ids': [1], 'num_nodes': 1},
                      'events': {'pkeys': list(range(100, 100 + events)),
                                 'node_ids': list(range(events)), 'num_nodes': events}},
            'edges': []}
