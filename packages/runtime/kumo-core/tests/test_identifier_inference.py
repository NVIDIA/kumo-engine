# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.stype_inference import (
    contains_identifier,
    infer_identifier_stype,
)
from kumoapi.typing import Dtype, Stype


@pytest.mark.parametrize('name', [
    'user_id',
    'USER_ID',
    'customerid',
    'id',
    'transaction_key',
    'code_value',
    'class_label',
    'product_category',
    'hashed_id_value',
])
@pytest.mark.parametrize('dtype', [Dtype.int64, Dtype.string])
def test_identifier_names_use_hosted_id_policy(name, dtype):
    assert contains_identifier(name)
    assert infer_identifier_stype(name, dtype) == Stype.ID
    assert infer_identifier_stype(name, dtype.value) == Stype.ID


@pytest.mark.parametrize('name',
                         ['bid', 'grid', 'paid', 'covid', 'liquid', 'age'])
def test_ordinary_words_are_not_identifiers(name):
    assert not contains_identifier(name)
    assert infer_identifier_stype(name, Dtype.int64) is None


@pytest.mark.parametrize('dtype', [
    None,
    Dtype.float64,
    Dtype.bool,
    Dtype.date,
    Dtype.intlist,
    Dtype.unsupported,
])
def test_identifier_names_do_not_override_incompatible_dtypes(dtype):
    assert infer_identifier_stype('user_id', dtype) is None


@pytest.mark.parametrize('name', [
    'num_items', 'item_count', 'total_sales', 'avg_score', 'price', 'ratio',
    'value'
])
@pytest.mark.parametrize('dtype', [Dtype.int64, Dtype.float64])
def test_numeric_names_take_precedence_over_low_cardinality(name, dtype):
    from kumo_core.stype_inference import infer_stype_from_name
    assert infer_stype_from_name(name, dtype) == Stype.numerical


@pytest.mark.parametrize('dtype,expected', [(Dtype.int64, Stype.ID),
                                            (Dtype.float64, Stype.numerical)])
def test_identifier_name_precedence_is_preserved(dtype, expected):
    from kumo_core.stype_inference import infer_stype_from_name
    assert infer_stype_from_name('num_users_id', dtype) == expected


@pytest.mark.parametrize('dtype',
                         [None, Dtype.string, Dtype.bool, Dtype.intlist])
def test_numeric_name_requires_numeric_dtype(dtype):
    from kumo_core.stype_inference import infer_stype_from_name
    assert infer_stype_from_name('num_items', dtype) is None
