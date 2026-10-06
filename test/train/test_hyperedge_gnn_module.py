# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from kumo_core.train.hyperedge_gnn_module import (
    HyperedgeGNNLightningModule,
    HyperedgeLightningData,
)
from torch.utils.data import RandomSampler
from torch_geometric.data import HeteroData
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
)


class _RootOnlySampler(BaseSampler):
    """Return seed nodes without native sampling to exercise loader behavior."""
    def sample_from_nodes(self, inputs: NodeSamplerInput, **kwargs):
        return HeteroSamplerOutput(
            node={inputs.input_type: inputs.node},
            row={},
            col={},
            edge={},
            metadata=(inputs.input_id, inputs.time),
        )

    def sample_from_edges(self, inputs, neg_sampling=None):
        raise NotImplementedError('This fixture only samples seed nodes')


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(num_workers=4, filter_per_worker=True),
    dict(sampler=RandomSampler(range(20), num_samples=15)),
    dict(eval_loader_kwargs=dict(batch_size=20))
])
def test_datamodule(kwargs):
    data = HeteroData()
    data['paper'].x = torch.arange(60)
    data['author'].x = torch.arange(60)

    datamodule = HyperedgeLightningData(
        (data, data),
        graph_sampler=_RootOnlySampler(),
        input_train_fkeys=[
            ('paper', torch.arange(0, 20)),
            ('author', torch.arange(0, 20)),
        ],
        input_val_fkeys=[
            ('paper', torch.arange(20, 40)),
            ('author', torch.arange(20, 40)),
        ],
        input_test_fkeys=[
            ('paper', torch.arange(40, 60)),
            ('author', torch.arange(40, 60)),
        ],
        batch_size=10,
        **kwargs,
    )

    loader_kwargs = datamodule.loader_kwargs
    assert len(loader_kwargs) in {4, 5}
    assert loader_kwargs['batch_size'] == 10
    assert loader_kwargs['pin_memory']
    assert loader_kwargs['num_workers'] == kwargs.get('num_workers', 0)
    persistent_workers = kwargs.get('num_workers', 0) > 0
    assert loader_kwargs['persistent_workers'] == persistent_workers
    if 'filter_per_worker' in kwargs:
        filter_per_worker = kwargs.get('filter_per_worker', False)
        assert loader_kwargs['filter_per_worker'] == filter_per_worker
    if 'sampler' in kwargs:
        assert loader_kwargs['sampler'] == kwargs['sampler']

    loader_kwargs = datamodule.eval_loader_kwargs
    assert len(loader_kwargs) in {4, 5}
    batch_size = kwargs.get('eval_loader_kwargs', {}).get('batch_size', 10)
    assert loader_kwargs['batch_size'] == batch_size
    assert loader_kwargs['pin_memory']
    assert loader_kwargs['num_workers'] == kwargs.get('num_workers', 0)
    persistent_workers = kwargs.get('num_workers', 0) > 0
    assert loader_kwargs['persistent_workers'] == persistent_workers
    if 'filter_per_worker' in kwargs:
        filter_per_worker = kwargs.get('filter_per_worker', False)
        assert loader_kwargs['filter_per_worker'] == filter_per_worker
    assert 'sampler' not in loader_kwargs

    train_loader = datamodule.train_dataloader()
    assert len(train_loader) == 2
    for i, (batch1, batch2) in enumerate(train_loader):
        batch_size = batch1['paper'].batch_size
        assert batch_size == (5 if i == 1 and 'sampler' in kwargs else 10)
        assert batch2['author'].batch_size == batch_size
        assert torch.equal(batch1['paper'].x, batch2['author'].x)
        assert torch.equal(batch1['paper'].n_id, batch2['author'].n_id)
        assert torch.equal(batch1['paper'].input_id, batch2['author'].input_id)
        assert batch1['paper'].n_id.numel() == batch_size
        assert batch1['paper'].n_id.min() >= 0
        assert batch1['paper'].n_id.max() < 20

    val_loader = datamodule.val_dataloader()
    assert len(val_loader) == 1 if 'eval_loader_kwargs' in kwargs else 2
    for batch1, batch2 in val_loader:
        batch_size = batch1['paper'].batch_size
        assert batch_size == (20 if 'eval_loader_kwargs' in kwargs else 10)
        assert batch2['author'].batch_size == batch_size
        assert torch.equal(batch1['paper'].x, batch2['author'].x)
        assert torch.equal(batch1['paper'].n_id, batch2['author'].n_id)
        assert torch.equal(batch1['paper'].input_id, batch2['author'].input_id)
        assert batch1['paper'].n_id.numel() == batch_size
        assert batch1['paper'].n_id.min() >= 20
        assert batch1['paper'].n_id.max() < 40

    test_loader = datamodule.test_dataloader()
    assert len(val_loader) == 1 if 'eval_loader_kwargs' in kwargs else 2
    for batch1, batch2 in test_loader:
        batch_size = batch1['paper'].batch_size
        assert batch_size == (20 if 'eval_loader_kwargs' in kwargs else 10)
        assert batch2['author'].batch_size == batch_size
        assert torch.equal(batch1['paper'].x, batch2['author'].x)
        assert torch.equal(batch1['paper'].n_id, batch2['author'].n_id)
        assert torch.equal(batch1['paper'].input_id, batch2['author'].input_id)
        assert batch1['paper'].n_id.numel() == batch_size
        assert batch1['paper'].n_id.min() >= 40
        assert batch1['paper'].n_id.max() < 60


def test_predict_step_returns_requested_embedding_per_batch():
    module = object.__new__(HyperedgeGNNLightningModule)
    torch.nn.Module.__init__(module)
    module.entity_table_names = ['paper', 'author']
    module.train_table_name = 'TRAIN_TABLE'
    module._kumo_return_predict_embeddings = True
    embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state = {'hook': None, 'removed': False}

    class _Handle:
        def remove(self):
            state['removed'] = True

    def register_hook(hook):
        state['hook'] = hook
        return _Handle()

    def common_step(_batches):
        state['hook'](None, (embedding, ))
        return torch.zeros(2, 1)

    module._get_register_embedding_hook_fn = lambda: register_hook
    module.common_step = common_step
    paper_batch = HeteroData()
    paper_batch['paper'].batch_size = 2
    paper_batch['paper'].pkey = torch.tensor([10, 11])
    paper_batch['TRAIN_TABLE'].time = torch.tensor([100, 200])
    author_batch = HeteroData()
    author_batch['author'].pkey = torch.tensor([20, 21])

    output = module.predict_step((paper_batch, author_batch), batch_idx=0)

    assert state['removed'] is True
    assert torch.equal(output['embedding'], embedding)
