# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import random
from collections.abc import Generator
from dataclasses import is_dataclass
from typing import Literal

import pytest
import torch
from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_geometric.testing import onlyFullTest, withCUDA

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder, TensorFrameEncoder
from kumo_core.encoders.fused.categorical import FusedCategoricalEncoder
from kumo_core.encoders.fused.multicategorical import FusedMultiCategoricalEncoder
from kumo_core.encoders.fused.numerical import FusedNumericalEncoder
from kumo_core.encoders.fused.sequence import FusedSequenceEncoder
from kumo_core.encoders.fused.text import FusedTextEncoder
from kumo_core.encoders.fused.timestamp import FusedTimestampEncoder
from kumo_core.types.features import TIMESTAMP_PREFIX

# TODO(akihironitta): Use stype instead of stype.value once PyTorch
# addresses https://github.com/pytorch/pytorch/issues/111603
numerical = Stype.numerical.value
categorical = Stype.categorical.value
timestamp = Stype.timestamp.value
text = Stype.text.value
sequence = Stype.sequence.value
multicategorical = Stype.multicategorical.value

STYPE_TO_ENCODER = {
    numerical: FusedNumericalEncoder,
    categorical: FusedCategoricalEncoder,
    text: FusedTextEncoder,
    timestamp: FusedTimestampEncoder,
    sequence: FusedSequenceEncoder,
    multicategorical: FusedMultiCategoricalEncoder,
}


def get_tf_and_col_encoder_dict(
    remove_kumo_dependency: bool = False,
    empty_time: bool = False,
) -> tuple[TensorFrame, dict[str, Encoder]]:
    num_rows = 4
    mapped_text_tensor_list = ([
        [
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
        ],
        [
            torch.tensor([5], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
        ],
        [
            torch.tensor([6], dtype=torch.long),
            torch.tensor([6, 7], dtype=torch.long),
        ],
        [
            torch.tensor([5], dtype=torch.long),
            torch.tensor([6], dtype=torch.long),
        ],
    ])
    tensor_mat = []
    for _ in range(num_rows):
        length = random.randint(1, 5)
        tensor = torch.randint(0, length, (length, ))
        tensor_list = [tensor]
        tensor_mat.append(tensor_list)

    feat_dict = {
        numerical:
        torch.randn(num_rows, 3),
        categorical:
        torch.arange(num_rows).unsqueeze(1).repeat(1, 2),
        timestamp:
        torch.stack(
            [
                torch.randint(0, 60, (num_rows, 1)),  # minute
                torch.randint(0, 24, (num_rows, 1)),  # hour
                torch.randint(1990, 2023, (num_rows, 1)),  # year
                torch.randint(0, 7, (num_rows, 1)),  # day_of_week
                torch.randint(0, 28, (num_rows, 1)),  # day_of_month
                torch.randint(0, 365, (num_rows, 1)),  # day_of_year
                torch.randint(28, 31, (num_rows, 1)),  # days_per_month
                torch.randint(365, 366, (num_rows, 1)),  # days_per_year
                torch.randint(52, 53, (num_rows, 1)),  # weeks_per_year
                torch.randint(0, 12, (num_rows, 1)),  # month
                torch.randint(0, 52, (num_rows, 1)),  # week_of_year
                torch.randint(0, 10, (num_rows, 1)),  # days_to_holiday
                torch.randint(0, 10, (num_rows, 1)),  # days_from_holiday
                torch.randint(0, 10, (num_rows, 1)),  # to_holiday_name
                torch.randint(0, 10, (num_rows, 1)),  # from_holiday_name
                torch.randint(0, 1000000000, (num_rows, 1)),  # unix
            ],
            dim=-1),  # [num_rows, num_cols, 16]
        text:
        MultiNestedTensor.from_tensor_mat(mapped_text_tensor_list),
        sequence:
        MultiEmbeddingTensor.from_tensor_list([torch.randn(num_rows, 11)]),
        multicategorical:
        MultiNestedTensor.from_tensor_mat(tensor_mat),
    }
    col_names_dict = {
        numerical: ['num0', 'num1', 'num2'],
        categorical: ['cat0', 'cat1'],
        timestamp: ['time0'],
        text: ['text0', 'text1'],
        sequence: ['seq0'],
        multicategorical: ['mulcat0'],
    }
    if empty_time:
        del feat_dict[timestamp]
        del col_names_dict[timestamp]

    tf = TensorFrame(
        feat_dict=feat_dict,
        col_names_dict=col_names_dict,
    )
    # NOTE: Just to simplify the test, we use the same category counts for all
    # categorical columns and the number of categories is equal to the number
    # of rows. There's no other speficic reason for using num_rows here.
    category_counts = [['A', 4], ['B', 3], ['C', 2], ['D', 1]]

    num0 = Numerical()
    num0._stats = {
        ColStatType.MEAN: float(torch.mean(tf.feat_dict[numerical][:, 0])),
    }
    num1 = MaxLogNumerical()
    num1._stats = {
        ColStatType.MIN: float(torch.min(tf.feat_dict[numerical][:, 1])),
        ColStatType.MEAN: float(torch.mean(tf.feat_dict[numerical][:, 1])),
    }
    num2 = MinLogNumerical()
    num2._stats = {
        ColStatType.MAX: float(torch.max(tf.feat_dict[numerical][:, 2])),
        ColStatType.MEAN: float(torch.mean(tf.feat_dict[numerical][:, 2])),
    }
    cat0 = Index()
    cat0._stats = {ColStatType.CATEGORY_COUNTS: category_counts}
    cat1 = Hash(num_components=num_rows)
    cat1._stats = {ColStatType.CATEGORY_COUNTS: category_counts}
    time0 = Datetime(num_year_periods=1)
    time0._stats = {
        ColStatType.MIN: f'{TIMESTAMP_PREFIX}1970-01-01',
        ColStatType.MAX: f'{TIMESTAMP_PREFIX}2022-12-31',
    }
    text0 = TextEmbedding('glove.test-10d')
    text1 = TextEmbedding('glove.test-10d')
    seq0 = NumericalList()
    seq0._stats = {
        ColStatType.SEQUENCE_MAX_LENGTH: tf.feat_dict[sequence].values.size(1),
        ColStatType.SEQUENCE_MIN_LENGTH: tf.feat_dict[sequence].values.size(1),
    }
    mulcat0 = MultiCategorical()
    mulcat0._stats = {
        ColStatType.MULTI_CATEGORY_COUNTS: category_counts,
        ColStatType.MULTI_CATEGORIES_SEPARATOR: None,
    }

    col_encoder_dict: dict[str, Encoder] = {
        'num0': num0,
        'num1': num1,
        'num2': num2,
        'cat0': cat0,
        'cat1': cat1,
        'time0': time0,
        'text0': text0,
        'text1': text1,
        'seq0': seq0,
        'mulcat0': mulcat0,
    }

    if empty_time:
        del col_encoder_dict['time0']

    if remove_kumo_dependency:
        # This is necessary for test cases that expect `kumo` package to be
        # available in the test environment:
        for encoder in col_encoder_dict.values():
            dataclass_name = encoder.__class__.__name__
            encoder._target_ = f'kumoapi.encoder.{dataclass_name}'

    return tf, col_encoder_dict


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
@pytest.mark.parametrize('use_seq_id', [True, False])
@pytest.mark.parametrize('act', [None, 'leaky_relu'])
@pytest.mark.parametrize('time_col', [None, 'time0', 'time_diff_name'])
def test_tf_encoder_forward(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
    use_seq_id: bool,
    act: Literal['leaky_relu'] | None,
    time_col: str | None,
    teardown_glove_helper: Generator,
) -> None:
    tf, col_encoder_dict = get_tf_and_col_encoder_dict()
    tf = tf.to(device)
    num_rows = tf.num_rows
    batch = torch.randint(1, 4, (num_rows, ), device=device)
    seed_time = torch.zeros(num_rows, device=device, dtype=torch.long)
    if time_col is not None:
        time = torch.ones(num_rows, device=device, dtype=torch.long)
    else:
        time = None
    out_channels = 2
    tf_encoder = TensorFrameEncoder(
        name='test_table',
        col_encoder_dict=col_encoder_dict,
        out_channels=out_channels,
        enc_mode=enc_mode,
        act=act,
        use_seq_id=use_seq_id,
        exclude_cols=[],
        time_col=time_col,
        device=device,
    )
    for stype, encoder_cls in STYPE_TO_ENCODER.items():
        assert isinstance(
            tf_encoder.stype_encoder_dict[stype],
            encoder_cls,
        )
    assert tf_encoder.out_channels == out_channels
    assert tf_encoder.enc_mode == enc_mode
    assert tf_encoder.time_col == time_col
    if act is None:
        assert tf_encoder.act is None
    else:
        assert isinstance(tf_encoder.act, torch.nn.LeakyReLU)
    assert tf_encoder.use_seq_id is use_seq_id
    assert tf_encoder.exclude_cols == []
    out = tf_encoder(tf, batch, time=time, seed_time=seed_time)
    assert out.size() == (tf.num_rows, out_channels)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
@pytest.mark.parametrize('use_seq_id', [True, False])
@pytest.mark.parametrize('act', [None, 'leaky_relu'])
@pytest.mark.parametrize('time_col', [None, 'time0'])
def test_tf_encoder_no_feature(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
    use_seq_id: bool,
    act: Literal['leaky_relu'] | None,
    time_col: str | None,
    teardown_glove_helper: Generator,
) -> None:
    num_rows = 3
    # Null columns are excluded from TensorFrame already:
    tf = TensorFrame({}, {})
    batch = torch.randint(0, num_rows, (num_rows, ), device=device)
    seed_time = time = None
    if time_col is not None:
        seed_time = torch.arange(num_rows, device=device)
        time = torch.arange(num_rows, device=device)
    out_channels = 2
    encoder = TensorFrameEncoder(
        name='test_table',
        col_encoder_dict={'col1': Null()},
        out_channels=out_channels,
        enc_mode=enc_mode,
        act=act,
        use_seq_id=use_seq_id,
        time_col=time_col,
        device=device,
    )
    out = encoder(tf, batch, time=time, seed_time=seed_time)
    if time_col is not None:
        assert not torch.allclose(
            out,
            torch.ones(num_rows, out_channels, device=device),
        )
    else:
        assert torch.allclose(
            out,
            torch.ones(num_rows, out_channels, device=device),
        )


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_hetero_tf_encoder_forward(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
    teardown_glove_helper: Generator,
) -> None:
    tf_dict = {}
    batch_dict = {}
    time_dict: dict[str, Tensor | None] = {}
    table_encoder_dict = {}
    exclude_cols_dict: dict[str, list[str]] = {}

    tf, col_encoder_dict = get_tf_and_col_encoder_dict()
    tf = tf.to(device)
    tf_dict['table'] = tf
    batch_dict['table'] = torch.randint(1, 4, (tf.num_rows, ), device=device)
    time_dict['table'] = None
    table_encoder_dict['table'] = col_encoder_dict
    exclude_cols_dict['table'] = []

    tf, col_encoder_dict = get_tf_and_col_encoder_dict()
    tf = tf.to(device)
    tf_dict['temporal_table'] = tf
    batch_dict['temporal_table'] = torch.randint(1, 4, (tf.num_rows, ),
                                                 device=device)
    time_dict['temporal_table'] = torch.arange(tf.num_rows, device=device)
    table_encoder_dict['temporal_table'] = col_encoder_dict
    seed_time = torch.zeros(tf.num_rows, device=device, dtype=torch.long)
    exclude_cols_dict['temporal_table'] = []

    out_channels = 2
    hetero_tf_encoder = HeteroTensorFrameEncoder(
        table_encoder_dict=table_encoder_dict,
        out_channels=out_channels,
        enc_mode=enc_mode,
        exclude_cols_dict=exclude_cols_dict,
        time_col_dict={'temporal_table': 'time_col'},
        device=device,
    )

    out = hetero_tf_encoder(
        tf_dict,
        batch_dict=batch_dict,
        time_dict=time_dict,
        seed_time=seed_time,
    )
    for table_name in ['table', 'temporal_table']:
        tf = tf_dict[table_name]
        assert out[table_name].size() == (tf.num_rows, out_channels)


@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_hetero_tf_encoder_empty_time_forward(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
) -> None:
    tf_dict = {}
    batch_dict = {}
    time_dict: dict[str, Tensor | None] = {}
    table_encoder_dict = {}
    exclude_cols_dict: dict[str, list[str]] = {}

    tf, col_encoder_dict = get_tf_and_col_encoder_dict(empty_time=True)
    tf = tf.to(device)
    tf_dict['table'] = tf
    batch_dict['table'] = torch.randint(1, 4, (tf.num_rows, ), device=device)
    time_dict['table'] = None
    table_encoder_dict['table'] = col_encoder_dict
    exclude_cols_dict['table'] = []

    tf, col_encoder_dict = get_tf_and_col_encoder_dict(empty_time=True)
    tf = tf[:0].to(device)
    tf = tf.to(device)
    tf_dict['temporal_table'] = tf
    batch_dict['temporal_table'] = torch.empty(0, dtype=torch.long,
                                               device=device)
    # Empty time feature
    time_dict['temporal_table'] = torch.arange(0, device=device)
    table_encoder_dict['temporal_table'] = col_encoder_dict
    exclude_cols_dict['temporal_table'] = []

    seed_time = torch.zeros(4, device=device, dtype=torch.long)

    out_channels = 2
    hetero_tf_encoder = HeteroTensorFrameEncoder(
        table_encoder_dict=table_encoder_dict,
        out_channels=out_channels,
        enc_mode=enc_mode,
        exclude_cols_dict=exclude_cols_dict,
        time_col_dict={'temporal_table': 'time_col'},
        device=device,
    )

    with modeling_context.explaining():
        outs = hetero_tf_encoder(
            tf_dict,
            batch_dict=batch_dict,
            time_dict=time_dict,
            seed_time=seed_time,
        )
    out = torch.cat(list(outs.values()), dim=0)
    out.backward(torch.randn_like(out))
    rel_time_encoder = hetero_tf_encoder.enc_dict[
        'temporal_table'].rel_time_encoder
    grad = rel_time_encoder.get_grad_wrt_input()
    assert grad.size() == (0, 1)


def test_config() -> None:
    tf, col_encoder_dict = get_tf_and_col_encoder_dict(
        remove_kumo_dependency=True)
    tf_dict = {'table': tf}
    batch_dict = {'table': torch.zeros(tf.num_rows, dtype=torch.long)}
    time_dict = {'table': torch.arange(tf.num_rows)}
    seed_time = torch.ones(tf.num_rows, dtype=torch.long)

    model1 = HeteroTensorFrameEncoder(
        table_encoder_dict={'table': col_encoder_dict},
        out_channels=4,
        enc_mode='fuse',
        exclude_cols_dict={},
        time_col_dict={'table': 'time_col'},
        device=torch.device('cpu'),
    )

    out1 = model1(
        tf_dict,
        batch_dict=batch_dict,
        time_dict=time_dict,
        seed_time=seed_time,
    )
    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == (
        'kumo_core.encoders.tensor_frame_encoder.HeteroTensorFrameEncoder')

    model2 = HeteroTensorFrameEncoder.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    out2 = model2(
        tf_dict,
        batch_dict=batch_dict,
        time_dict=time_dict,
        seed_time=seed_time,
    )
    assert torch.allclose(out1['table'], out2['table'])


@onlyFullTest
@withCUDA
@pytest.mark.parametrize('enc_mode', ['fuse', 'column'])
def test_compile(
    device: torch.device,
    enc_mode: Literal['fuse', 'column'],
    teardown_glove_helper: Generator,
) -> None:
    tf, col_encoder_dict = get_tf_and_col_encoder_dict()
    tf = tf.to(device)
    tf_dict = {'table': tf}
    batch_dict = {'table': torch.zeros(tf.num_rows, dtype=torch.long)}
    time_dict = {'table': torch.arange(tf.num_rows).to(device)}
    seed_time = torch.zeros(tf.num_rows, device=device, dtype=torch.long)

    encoder = HeteroTensorFrameEncoder(
        table_encoder_dict={'table': col_encoder_dict},
        out_channels=4,
        enc_mode=enc_mode,
        exclude_cols_dict={},
        time_col_dict={},
        device=device,
    )

    explanation = torch._dynamo.explain(encoder)(
        tf_dict,
        batch_dict,
        time_dict=time_dict,
        seed_time=seed_time,
    )
    assert explanation.graph_break_count == 0

    compiled_encoder = torch.compile(encoder)
    # Note: We call forward/backward twice on purpose. The second call uses
    # 'guards' cached during the first call. The second call is to catch
    # unexpected errors like https://github.com/pytorch/pytorch/issues/111603.
    for _ in range(2):
        for out in compiled_encoder(tf_dict, batch_dict).values():
            out.sum().backward()


@withCUDA
def test_debug(device: torch.device) -> None:
    tf = TensorFrame(feat_dict={}, col_names_dict={}, num_rows=5).to(device)
    num_rows = 5
    batch = torch.zeros(num_rows, dtype=torch.long, device=device)
    seed_time = torch.arange(num_rows, device=device, dtype=torch.long)
    time = torch.arange(num_rows, device=device, dtype=torch.float)

    tf_encoder = TensorFrameEncoder(
        name='test_table',
        col_encoder_dict={},
        out_channels=16,
        time_col='time',
        device=device,
    )
    assert tf_encoder.rel_time_encoder is not None
    tf_encoder.rel_time_encoder.time_lin.bias.data.fill_(float('NaN'))

    out = tf_encoder(tf, batch, time=time, seed_time=seed_time)
    assert out.isnan().sum() > 0

    with modeling_context.debugging():
        with pytest.raises(
                ValueError,
                match="NaN values for the table 'test_table'",
        ):
            tf_encoder(tf, batch, time=time, seed_time=seed_time)


@withCUDA
def test_set_exclude_cols_dict(
    device: torch.device,
    teardown_glove_helper: Generator,
) -> None:

    tf_dict = {}
    batch_dict = {}
    table_encoder_dict = {}

    for i in range(1, 3):
        tf, col_encoder_dict = get_tf_and_col_encoder_dict()
        tf = tf.to(device)
        tf_dict[f'table{i}'] = tf
        batch_dict[f'table{i}'] = torch.zeros(tf.num_rows, device=device)
        table_encoder_dict[f'table{i}'] = col_encoder_dict

    model = HeteroTensorFrameEncoder(
        table_encoder_dict=table_encoder_dict,
        out_channels=8,
        exclude_cols_dict={},
        time_col_dict={},
        device=device,
    )

    assert model.exclude_cols_dict == {}
    for encoder in model.enc_dict.values():
        assert encoder.exclude_cols == []
        for stype_encoder in encoder.stype_encoder_dict.values():
            assert stype_encoder.exclude_cols == []
            assert stype_encoder.exclude_cols_mask.sum() == 0

    out1 = model(tf_dict, batch_dict=batch_dict)

    exclude_cols_dict = {'table1': ['num0'], 'table2': ['cat0']}
    with model.set_exclude_cols_dict(exclude_cols_dict):
        assert model.exclude_cols_dict == exclude_cols_dict
        for node_type, encoder in model.enc_dict.items():
            assert encoder.exclude_cols == exclude_cols_dict[node_type]
            for stype, stype_encoder in encoder.stype_encoder_dict.items():
                assert stype_encoder.exclude_cols == (
                    exclude_cols_dict[node_type])
                if node_type == 'table1' and stype == Stype.numerical:
                    assert stype_encoder.exclude_cols_mask.sum() == 1
                elif node_type == 'table2' and stype == Stype.categorical:
                    assert stype_encoder.exclude_cols_mask.sum() == 1
                else:
                    assert stype_encoder.exclude_cols_mask.sum() == 0

        out2 = model(tf_dict, batch_dict=batch_dict)

    assert model.exclude_cols_dict == {}
    for encoder in model.enc_dict.values():
        assert encoder.exclude_cols == []
        for stype_encoder in encoder.stype_encoder_dict.values():
            assert stype_encoder.exclude_cols == []
            assert stype_encoder.exclude_cols_mask.sum() == 0

    out3 = model(tf_dict, batch_dict=batch_dict)

    for node_type in out1.keys():
        assert out1[node_type].allclose(out3[node_type])
        assert not out1[node_type].allclose(out2[node_type])
