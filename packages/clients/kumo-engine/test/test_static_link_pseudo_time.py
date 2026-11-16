# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.table import Column
from kumoapi.typing import Dtype, Stype

import kumo.engine as kumo
from kumo.engine import FileConnector, Graph, PredictiveQuery, Table, Trainer
from kumo.engine.table import TableDefinition

QUERY = ('PREDICT LIST_DISTINCT(orders.item_id) RANK TOP 10 '
         'FOR EACH users.id')
LAYER = {
    'fanouts': {},
    'default_fanout': {
        'fanout': 1000,
        'mode': 'uniform',
        'absent_key_policy': 'uniform_fallback',
    },
}


def _snapshot(tmp_path, num_users=60, num_items=40, per_user=8):
    """Users order random items: no signal beyond the labels themselves."""
    rng = random.Random(0)
    orders = [(user, item) for user in range(num_users)
              for item in rng.sample(range(num_items), per_user)]
    frames = {
        'users': pa.table({'id': list(range(num_users))}),
        'items': pa.table({'id': list(range(num_items))}),
        'orders': pa.table({
            'id': list(range(len(orders))),
            'user_id': [user for user, _ in orders],
            'item_id': [item for _, item in orders],
        }),
    }
    source = tmp_path / 'inputs'
    source.mkdir()
    connector = FileConnector(root_dir=str(source))
    tables = []
    for name, frame in frames.items():
        pq.write_table(frame, source / f'{name}.parquet')
        tables.append(
            Table(
                connector, name,
                TableDefinition(
                    source_table=connector.source_table_for(name), pkey='id',
                    cols=[
                        Column(name=col, dtype=Dtype.int64, stype=Stype.ID)
                        for col in frame.column_names
                    ])))
    graph = Graph(tables, [
        Edge('orders', 'user_id', 'users'),
        Edge('orders', 'item_id', 'items'),
    ])
    return graph, graph.snapshot(), orders


@pytest.fixture(params=['generated', 'custom'])
def datamodule(request, tmp_path):
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
    from kumo_core.training.table import PreparedCustomTrainingTable
    from kumo_core.training.workflow import (
        PreparedGraph,
        TrainingWorkflowRequest,
        materialize_training_task,
    )

    kumo.configure(scratch_dir=tmp_path / 'scratch')
    _, snapshot, orders = _snapshot(tmp_path)
    custom = None
    if request.param == 'custom':
        rows = tmp_path / 'custom'
        rows.mkdir()
        users = sorted({user for user, _ in orders})
        pq.write_table(
            pa.table({
                'ENTITY1': users,
                'TARGET1': [[i for u, i in orders if u == user]
                            for user in users],
            }), rows / 'part-00000.parquet')
        custom = PreparedCustomTrainingTable(str(rows))
    task = materialize_training_task(
        TrainingWorkflowRequest.from_spec(
            graph=snapshot.graph_config,
            predictive_query=QUERY,
            prepared_graph=PreparedGraph(index_id=snapshot.id,
                                         index_path=str(snapshot.index_path)),
            run_id='leak',
            scratch_dir=str(tmp_path / 'scratch'),
            artifact_dir=str(tmp_path / 'run'),
            custom_training_table=custom,
        ))
    return DiskgraphLightningDataModule(
        task.manifest_path,
        {},
        sampling_plan={'layers': [LAYER, LAYER]},
        train_batch_size=16,
        eval_batch_size=16,
        task_type='static_link_prediction',
        neg_rhs_node_type='items',
        neg_amount_m=1,
        resolve_entity_node_type='users',
        resolve_target_node_type='items',
    )


def _batches(loader):
    return list(loader.dataset)


def _ids(batch, node_type):
    graph = batch['graphs']['main']
    ids = graph['node_ids'].get(node_type)
    seeds = graph['batch_dict'].get(node_type)
    if ids is None:
        return [], []
    return np.asarray(ids).tolist(), np.asarray(seeds).tolist()


def _labels(batches):
    """``(user, item)`` node-id links scored by these batches."""
    links = set()
    for batch in batches:
        task = batch['task']
        users = np.repeat(np.asarray(task['seed_node_ids']),
                          np.asarray(task['lp_pos_counts']))
        items = np.asarray(task['lp_pos_rhs_global'])
        keep = ~np.asarray(task['lp_pos_mask'])
        links |= set(zip(users[keep].tolist(), items[keep].tolist()))
    return links


def _user_side(batches):
    """``(seed user, item)`` pairs each user seed reaches in its subgraph."""
    reached = set()
    for batch in batches:
        users = np.asarray(batch['task']['seed_node_ids']).tolist()
        for item, seed in zip(*_ids(batch, 'items')):
            if seed < len(users):
                reached.add((users[seed], item))
    return reached


def _item_side(batches):
    """``(user, seed item)`` pairs each item seed reaches in its subgraph."""
    reached = set()
    for batch in batches:
        items = np.asarray(batch['task']['seed_node_ids']).tolist()
        for user, seed in zip(*_ids(batch, 'users')):
            reached.add((user, items[seed]))
    return reached


def test_static_link_subgraphs_hide_later_split_labels(datamodule):
    train = _labels(_batches(datamodule.train_dataloader()))
    val = _labels(_batches(datamodule.val_dataloader()))
    test = _labels(_batches(datamodule.test_dataloader()))
    assert train and val and test and not (val & test)

    seen = {
        'train': _user_side(_batches(datamodule.train_dataloader())),
        'val': _user_side(_batches(datamodule.val_dataloader())),
        'test': _user_side(_batches(datamodule.test_dataloader())),
        'rhs_val': _item_side(_batches(datamodule.rhs_val_dataloader())),
        'rhs_test': _item_side(_batches(datamodule.rhs_test_dataloader())),
    }
    for name in ('train', 'val', 'rhs_val'):
        assert seen[name] & train, name
        assert not seen[name] & (val | test), name
    for name in ('test', 'rhs_test'):
        assert seen[name] & (train | val), name
        assert not seen[name] & test, name
    prediction = _item_side(_batches(datamodule.rhs_predict_dataloader()))
    assert train | val | test <= prediction


def test_static_link_fit_on_random_links_scores_like_random(tmp_path):
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    graph, snapshot, _ = _snapshot(tmp_path, num_users=300, num_items=200,
                                   per_user=10)
    query = PredictiveQuery(graph, QUERY)
    plan = query.suggest_model_plan(snapshot)
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 10
    plan.model_architecture.module = 'shallow_rhs'
    result = Trainer(plan).fit(query.generate_training_table(snapshot),
                               snapshot=snapshot)
    # Random ranking recalls ~10/190; reading labels off subgraphs gives ~0.3.
    assert result.metrics['test_recall@10'] < 0.15


def test_native_iterator_infers_cutoff_without_loader_options(datamodule):
    from diskgraph import RuntimeIndex

    dataset = datamodule.train_dataloader().dataset
    index = RuntimeIndex(dataset.manifest_path, rnd_seed=7, num_threads=1)

    def sample(stage, **options):
        iterator, _ = index.iter(
            stage=stage,
            sampling_plan={'layers': [LAYER, LAYER]},
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=16,
            entity_node_type='users',
            target_node_type='items',
            **options,
        )
        return list(iterator)

    for stage, cutoff in [('train', 1), ('val', 1), ('test', 2)]:
        inferred = _user_side(sample(stage))
        assert inferred == _user_side(sample(stage, pseudo_seed_time=cutoff))
        unrestricted = sample(stage, disable_pseudo_time=True)
        if stage != 'train':
            held_out = _labels(unrestricted)
            assert held_out
            assert not inferred & held_out
            assert _user_side(unrestricted) & held_out


def test_stage_visibility_overrides_generic_runtime_options(datamodule):
    loaders = [
        (datamodule.train_dataloader, _user_side),
        (datamodule.val_dataloader, _user_side),
        (datamodule.test_dataloader, _user_side),
        (datamodule.rhs_val_dataloader, _item_side),
        (datamodule.rhs_test_dataloader, _item_side),
        (datamodule.rhs_predict_dataloader, _item_side),
    ]
    expected = [read(_batches(loader())) for loader, read in loaders]
    datamodule.runtime_options.update({
        "pseudo_seed_time": 3,
        "disable_pseudo_time": True,
    })
    for (loader, read), visible in zip(loaders, expected):
        assert read(_batches(loader())) == visible
