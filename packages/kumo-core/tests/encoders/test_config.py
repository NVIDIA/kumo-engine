# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest
from kumoapi.encoder import Index, MultiCategorical, NAStrategy, Numerical
from kumoapi.typing import ColStatType, Stype

from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig


def test_encoders() -> None:
    encoders = EncodersConfig()

    encoders['TABLE_A']['COL_1'] = Numerical()
    encoders['TABLE_A']['COL_2'] = Index()
    encoders.target = TargetEncoderConfig(col_name='TARGET', encoder=Index())
    assert 'TABLE_A' in encoders
    assert 'TABLE_B' not in encoders

    assert list(encoders.keys()) == ['TABLE_A']
    assert list(encoders.values()) == [{
        'COL_1': Numerical(),
        'COL_2': Index()
    }]
    assert list(encoders.items()) == [('TABLE_A', {
        'COL_1': Numerical(),
        'COL_2': Index()
    })]

    assert str(encoders) == (
        'EncodersConfig(\n'
        '  encoders={\n'
        '    TABLE_A: {\n'
        '      COL_1: Numerical(scaler=None, na_strategy=mean),\n'
        '      COL_2: Index(min_occ=1, na_strategy=separate),\n'
        '    },\n'
        '  },\n'
        '  target=(TARGET, Index(min_occ=1, na_strategy=separate)),\n'
        ')')


@pytest.mark.parametrize('target_stype', [
    Stype.numerical,
    Stype.categorical,
    Stype.multicategorical,
])
def test_from_stypes(target_stype: Stype) -> None:
    df = pd.DataFrame({
        'AGE': [20, 30, 40, float('NaN')],
        'GENDER': ['male', 'female', 'female', None],
    })
    df_dict = {'USERS': df}
    stype_dict = {
        'AGE': Stype.numerical,
        'GENDER': Stype.categorical,
    }
    table_stype_dict = {'USERS': stype_dict}

    out = EncodersConfig.from_stypes(df_dict, table_stype_dict)
    assert list(out.encoders.keys()) == ['USERS']
    assert len(out.encoders['USERS']) == 2
    assert isinstance(out['USERS']['AGE'], Numerical)
    assert isinstance(out['USERS']['GENDER'], Index)
    assert out.target is None

    df_dict['TRAIN_TABLE'] = pd.DataFrame({'TARGET': [0, 10, 20]})
    out = EncodersConfig.from_stypes(
        df_dict=df_dict,
        table_stype_dict=table_stype_dict,
        train_table_name='TRAIN_TABLE',
        target_col_name='TARGET',
        target_stype=target_stype,
    )
    assert out.target is not None
    assert out.target.col_name == 'TARGET'
    encoder = out.target.encoder
    if target_stype == Stype.numerical:
        assert encoder == Numerical(scaler='standard')
    elif target_stype == Stype.categorical:
        assert encoder == Index(min_occ=1)
    else:
        assert target_stype == Stype.multicategorical
        assert isinstance(encoder, MultiCategorical)
        assert encoder.na_strategy == NAStrategy.ZERO
        assert encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] == '|'
