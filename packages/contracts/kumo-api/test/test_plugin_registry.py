# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from kumoapi.plugin import check_model_alias, registry_lineage


@pytest.mark.parametrize('alias', [
    '', ' ', 'default', 'LAST', 'Latest', 'v3', 'my-alias', '2nd', 'a' * 256,
    None
])
def test_check_model_alias_refuses_what_a_registry_reserves(alias):
    with pytest.raises(ValueError):
        check_model_alias(alias)


def test_check_model_alias_accepts_a_plain_name():
    check_model_alias('champion')
    check_model_alias('_v2_prod')


def test_registry_lineage_keeps_kumo_keys_as_strings():
    recorded = {
        'kumo_engine': '1',
        'kumo_bundle_id': 'b',
        'auc': 0.9,
        'kumo_seconds': 3
    }
    assert registry_lineage(recorded) == {
        'kumo_bundle_id': 'b',
        'kumo_seconds': '3'
    }
