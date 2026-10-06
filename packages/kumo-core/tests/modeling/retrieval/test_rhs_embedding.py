# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from typing import Literal

import pandas as pd
import pytest
import torch
from kumoapi.encoder import Index, Numerical
from kumoapi.model_plan import RHSEmbeddingMode
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_geometric.testing import onlyCUDA, withCUDA

import kumo_core.modeling.embedding as embedding_module
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from kumo_core.encoders.tensor_mapper import NumericalTensorMapper, TensorMapper
from kumo_core.encoders.selection import compute_missing_stats_
from kumo_core.modeling.embedding import Embedding
from kumo_core.modeling.retrieval.rhs_embedding import RHSEmbedding


@withCUDA
@pytest.mark.parametrize('primary_keys', [[10, 20, 30], ['a', 'b', 'c']])
@pytest.mark.parametrize('old_count,new_count', [(5, 7), (7, 5)])
def test_rhs_embedding_sparse_physical_index_checkpoint_round_trip(
        primary_keys: list[object], old_count: int, new_count: int,
        device: torch.device) -> None:
    old_ids = [0, 2, old_count - 1]
    old_map = pd.DataFrame({'arange': old_ids}, index=primary_keys)
    old = RHSEmbedding(
        emb_mode=RHSEmbeddingMode.LOOKUP,
        num_nodes=old_count,
        embedding_dim=2,
        index_map=old_map,
    ).to(device)
    assert old.lookup_embedding is not None
    with torch.no_grad():
        old.lookup_embedding.weight.copy_(
            torch.arange(old_count * 2, device=device).reshape(old_count, 2))
    state = copy.deepcopy(old.state_dict())

    new_keys = [primary_keys[2], primary_keys[0], 'new']
    new_ids = [1, 3, new_count - 1]
    new_map = pd.DataFrame({'arange': new_ids}, index=new_keys)
    new = RHSEmbedding(
        emb_mode=RHSEmbeddingMode.LOOKUP,
        num_nodes=new_count,
        embedding_dim=2,
        index_map=new_map.copy(),
    ).to(device)
    new.load_state_dict(state)
    new.order_by(new_map)

    assert new.lookup_embedding is not None
    assert new.lookup_embedding.weight.shape == (new_count, 2)
    assert torch.equal(new.lookup_embedding.weight[1],
                       old.lookup_embedding.weight[old_ids[2]])
    assert torch.equal(new.lookup_embedding.weight[3],
                       old.lookup_embedding.weight[0])
    expected_mean = old.lookup_embedding.weight[old_ids].mean(dim=0)
    assert torch.equal(new.lookup_embedding.weight[new_ids[2]], expected_mean)
    canonical_ids = torch.tensor(new_ids, dtype=torch.int64, device=device)
    new.filter(canonical_ids)
    assert new.get_mips_index().num_keys == 3


def test_sparse_embedding_chunked_mean_preserves_dtype(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_module, '_SPARSE_REMAP_CHUNK_SIZE', 2)
    old_ids = [0, 2, 4]
    old_map = pd.DataFrame({'arange': old_ids}, index=['a', 'b', 'c'])
    embedding = Embedding(5, 2, index_map=old_map).to(torch.float64)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(10, dtype=torch.float64).reshape(5, 2))
    expected_mean = embedding.weight[old_ids].mean(dim=0)

    new_map = pd.DataFrame({'arange': [0, 2, 5]}, index=['a', 'b', 'new'])
    embedding.order_by(new_map, num_embeddings=6)

    assert embedding.weight.dtype == torch.float64
    assert torch.equal(embedding.weight[5], expected_mean)


def test_sparse_embedding_bounds_accelerator_remap_indices(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedding_module, '_SPARSE_REMAP_CHUNK_SIZE', 2)
    original_from_numpy = torch.from_numpy
    transferred_lengths: list[int] = []

    def tracked_from_numpy(array):
        transferred_lengths.append(len(array))
        return original_from_numpy(array)

    monkeypatch.setattr(torch, 'from_numpy', tracked_from_numpy)
    old_map = pd.DataFrame({'arange': [0, 2, 4]}, index=['a', 'b', 'c'])
    embedding = Embedding(5, 2, index_map=old_map)
    new_map = pd.DataFrame({'arange': [1, 3, 5]}, index=['c', 'a', 'new'])
    transferred_lengths.clear()

    embedding.order_by(new_map, num_embeddings=6)

    assert transferred_lengths
    assert max(transferred_lengths) <= 2


@withCUDA
@pytest.mark.parametrize('emb_mode', list(RHSEmbeddingMode))
@pytest.mark.parametrize('rhs_feat_bytes_threshold', [None, 0])
def test_rhs_embedding(
    emb_mode: RHSEmbeddingMode,
    rhs_feat_bytes_threshold: int | None,
    device: torch.device,
) -> None:
    df = pd.DataFrame({
        'A': [0.1, 0.2, 0.3, 0.4, 0.5],
        'B': [0, 1, 0, 1, 0],
    })
    encoder_dict = {
        'A': compute_missing_stats_(Numerical(), df['A']),
        'B': compute_missing_stats_(Index(), df['B']),
    }
    mapper_dict = {
        col_name: TensorMapper.from_encoder(encoder)
        for col_name, encoder in encoder_dict.items()
    }
    feature_store = InMemoryFeatureStore(
        df_dict={'RHS': df},
        table_schema_dict={'RHS': dict(mapper_dict=mapper_dict)},
        train_table_schema=None,
    )

    model = RHSEmbedding(
        emb_mode=emb_mode,
        num_nodes=5,
        embedding_dim=8,
        encoder_dict=encoder_dict,
        rhs_feat_bytes_threshold=rhs_feat_bytes_threshold,
    ).to(device)

    assert model.device == device
    assert str(model) == (f"RHSEmbedding(emb_mode={emb_mode.value}, "
                          f"num_nodes=5, embedding_dim=8, "
                          f"distance_measure=dot_product)")

    if emb_mode.use_rhs_feature:
        with pytest.raises(RuntimeError, match="are not yet available"):
            model()

    model.fetch_feats(feature_store, table_name='RHS')
    if emb_mode.use_rhs_feature:
        assert model._feats is not None
        assert all([x.size(0) == 5 for x in model._feats])
        assert not model._filtered_feats

        if rhs_feat_bytes_threshold == 0:
            for feat in model._feats:
                assert feat.is_cpu
        else:
            for feat in model._feats:
                assert feat.device == device
    else:
        assert model._feats is None

    out = model()
    assert out.device == device
    assert out.size() == (5, 8)

    index = torch.tensor([1, 3], device=device)
    assert model(index).allclose(out[index])

    assert model.get_mips_index().num_keys == 5

    model.cache_output()
    if emb_mode.use_rhs_feature:
        assert model._out_cache is not None
        assert model._out_cache.allclose(out)
        with pytest.raises(RuntimeError, match="only available in inference"):
            model()
        with pytest.raises(RuntimeError, match="oversubscription of memory"):
            model.get_mips_index()
    else:
        assert model._out_cache is None
        model.get_mips_index()

    with torch.no_grad():
        assert model().allclose(out)
        assert model(index).allclose(out[index])

    # Test RHS embedding filtering logic ######################################

    index = torch.tensor([1, 3], device=device)
    model.filter(index)
    assert model._out_cache is None
    assert model._filtered_index is not None
    assert model._filtered_index.equal(index)
    assert model._filtered_index_map is None

    if emb_mode.use_rhs_feature:  # Fetched features are untouched:
        assert model._feats is not None
        assert all([x.size(0) == 5 for x in model._feats])
        assert not model._filtered_feats
    else:
        assert model._feats is None

    assert model().allclose(out[index])
    assert model(index).allclose(out[index])
    with pytest.raises(IndexError, match="invalid RHS nodes"):
        model(torch.tensor([0, 1], device=device))

    assert model.get_mips_index().num_keys == 2

    model.cache_output()
    if emb_mode.use_rhs_feature:
        assert model._out_cache is not None
        assert model._out_cache.allclose(out[index])
        with pytest.raises(RuntimeError, match="oversubscription of memory"):
            model.get_mips_index()
    else:
        assert model._out_cache is None
        model.get_mips_index()

    with torch.no_grad():
        assert model().allclose(out[index])
        assert model(index).allclose(out[index])
        with pytest.raises(IndexError, match="invalid RHS nodes"):
            model(torch.tensor([0, 1], device=device))

    # Test filtered feature fetching ##########################################

    model._feats = None
    model._filtered_feats = False

    index = torch.tensor([2, 0, 4], device=device)
    model.filter(index)

    assert model._out_cache is None
    assert model._filtered_index is not None
    assert model._filtered_index.equal(index)
    assert model._filtered_index_map is not None
    assert model._filtered_index_map.equal(
        torch.tensor([1, -1, 0, -1, 2], device=device))
    assert model._feats is None
    assert not model._filtered_feats

    if emb_mode.use_rhs_feature:
        with pytest.raises(RuntimeError, match="are not yet available"):
            model()

    model.fetch_feats(feature_store, table_name='RHS')
    if emb_mode.use_rhs_feature:
        assert model._feats is not None
        assert all([x.size(0) == 3 for x in model._feats])
        assert model._filtered_feats
    else:
        assert model._feats is None

    assert model().allclose(out[index])
    assert model(index).allclose(out[index])
    with pytest.raises(IndexError, match="invalid RHS nodes"):
        model(torch.tensor([0, 1], device=device))

    assert model.get_mips_index().num_keys == 3

    model.cache_output()
    if emb_mode.use_rhs_feature:
        assert model._out_cache is not None
        assert model._out_cache.allclose(out[index])
        with pytest.raises(RuntimeError, match="oversubscription of memory"):
            model.get_mips_index()
    else:
        assert model._out_cache is None
        model.get_mips_index()

    with torch.no_grad():
        assert model().allclose(out[index])
        assert model(index).allclose(out[index])
        with pytest.raises(IndexError, match="invalid RHS nodes"):
            model(torch.tensor([0, 1], device=device))


@withCUDA
def test_sorted_rhs_filter_uses_bounded_implicit_mapping(
        device: torch.device) -> None:
    model = RHSEmbedding(
        emb_mode=RHSEmbeddingMode.LOOKUP,
        num_nodes=1_000_000,
        embedding_dim=1,
    ).to(device)
    index = torch.tensor([1, 10, 999_999], device=device)

    model.filter(index)

    assert model._filtered_index_map is None
    assert model._map_filtered_index(index).equal(
        torch.arange(3, device=device))
    with pytest.raises(IndexError, match="invalid RHS nodes"):
        model._map_filtered_index(torch.tensor([2], device=device))


@withCUDA
@pytest.mark.parametrize('emb_mode', list(RHSEmbeddingMode))
def test_no_features(emb_mode: RHSEmbeddingMode, device: torch.device) -> None:
    feature_store = InMemoryFeatureStore(
        df_dict={'RHS': pd.DataFrame({'INDEX': range(5)})},
        table_schema_dict={'RHS': dict(mapper_dict={})},
        train_table_schema=None,
    )

    model = RHSEmbedding(
        emb_mode=emb_mode,
        num_nodes=5,
        embedding_dim=8,
        encoder_dict={},
    ).to(device)
    model.fetch_feats(feature_store, table_name='RHS')

    assert model.device == device
    assert str(model) == (f"RHSEmbedding(emb_mode={emb_mode.value}, "
                          f"num_nodes=5, embedding_dim=8, "
                          f"distance_measure=dot_product)")

    index = torch.tensor([2, 0, 4], device=device)

    out = model()
    assert out.size() == (5, 8)
    assert model(index).allclose(out[index])

    model.filter(index)
    assert model().allclose(out[index])
    assert model(index).allclose(out[index])


@withCUDA
@pytest.mark.parametrize('distance_measure', ['dot_product', 'cosine'])
def test_distance_measure(distance_measure: Literal['dot_product', 'cosine'],
                          device: torch.device) -> None:
    feature_store = InMemoryFeatureStore(
        df_dict={'RHS': pd.DataFrame({'INDEX': range(5)})},
        table_schema_dict={'RHS': dict(mapper_dict={})},
        train_table_schema=None,
    )

    model = RHSEmbedding(
        emb_mode=RHSEmbeddingMode('feature'),
        num_nodes=5,
        embedding_dim=8,
        encoder_dict={},
        distance_measure=distance_measure,
    ).to(device)
    model.fetch_feats(feature_store, table_name='RHS')

    assert model.device == device
    assert str(model) == (f"RHSEmbedding(emb_mode=feature, "
                          f"num_nodes=5, embedding_dim=8, "
                          f"distance_measure={distance_measure})")
    out = model()
    assert out.size() == (5, 8)
    norm = torch.linalg.vector_norm(out, 2, dim=1)
    if distance_measure == 'dot_product':
        assert not norm.allclose(torch.ones(5).to(device))
    else:
        assert norm.allclose(torch.ones(5).to(device))


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rhs_embedding = RHSEmbedding(
            emb_mode=RHSEmbeddingMode.FEATURE,
            num_nodes=5,
            embedding_dim=8,
            encoder_dict={},
        )
        self.rhs_embedding._feats = [
            # float
            torch.ones(1, 1, dtype=torch.float),
            MultiNestedTensor.from_tensor_mat(
                [[torch.ones(1, dtype=torch.float)]]),
            MultiEmbeddingTensor.from_tensor_list(
                [torch.ones(1, 1, dtype=torch.float)]),
            # int64
            torch.ones(1, 1, dtype=torch.int64),
            MultiNestedTensor.from_tensor_mat(
                [[torch.ones(1, dtype=torch.int64)]]),
            MultiEmbeddingTensor.from_tensor_list(
                [torch.ones(1, 1, dtype=torch.int64)]),
        ]


@onlyCUDA
def test_cache_device_dtype() -> None:
    """Test that calling to(), cuda(), and cpu() on the wrapper nn.Module
    moves the caches in RHSEmbedding module to the correct device without
    changing their dtypes.
    """
    model = MyModule()
    assert model.rhs_embedding._feats is not None
    model.to('cuda')
    assert all(x.device.type == 'cuda' for x in model.rhs_embedding._feats)
    model.to('cpu')
    assert all(x.device.type == 'cpu' for x in model.rhs_embedding._feats)
    model.cuda()
    assert all(x.device.type == 'cuda' for x in model.rhs_embedding._feats)
    model.cpu()
    assert all(x.device.type == 'cpu' for x in model.rhs_embedding._feats)
    model.half()
    assert all(x.dtype == torch.float for x in model.rhs_embedding._feats[:3])
    assert all(x.dtype == torch.int64 for x in model.rhs_embedding._feats[3:])
    model.float()
    assert all(x.dtype == torch.float for x in model.rhs_embedding._feats[:3])
    assert all(x.dtype == torch.int64 for x in model.rhs_embedding._feats[3:])
    model.double()
    assert all(x.dtype == torch.float for x in model.rhs_embedding._feats[:3])
    assert all(x.dtype == torch.int64 for x in model.rhs_embedding._feats[3:])


@withCUDA
@pytest.mark.parametrize('filter', [False, True])
def test_fusion_on_the_fly(filter: bool, device: torch.device) -> None:
    df = pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4, 0.5]})
    encoder_dict = {'A': compute_missing_stats_(Numerical(), df['A'])}
    mapper_dict = {'A': NumericalTensorMapper()}

    feature_store = InMemoryFeatureStore(
        df_dict={'RHS': df},
        table_schema_dict={'RHS': dict(mapper_dict=mapper_dict)},
        train_table_schema=None,
    )

    model = RHSEmbedding(
        emb_mode='fusion',
        num_nodes=5,
        embedding_dim=8,
        encoder_dict=encoder_dict,
    ).to(device)

    with pytest.raises(RuntimeError, match="not yet available"):
        model()

    with torch.no_grad():
        with pytest.raises(RuntimeError, match="on-the-fly in batch mode"):
            model()

    if filter:
        index = torch.tensor([1, 3], device=device)
        model.filter(index)
    else:
        index = torch.tensor([0, 1, 2, 3, 4], device=device)

    with torch.no_grad():
        with pytest.raises(ValueError, match="needs to be provided"):
            model(index)

    with torch.no_grad():
        out1 = model(index, feature_store, table_name='RHS')

    model.fetch_feats(feature_store, table_name='RHS')
    out2 = model()
    assert torch.allclose(out1, out2)
