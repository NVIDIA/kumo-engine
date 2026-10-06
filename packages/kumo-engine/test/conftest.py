# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.engine.graph import GraphSnapshot


@pytest.fixture
def snapshot(tmp_path):
    config = {
        'tables': [{
            'table_name':
            'users',
            'source_table_name':
            'db.schema.users',
            'pkey': {
                'name': 'id',
                'dtype': 'int64',
                'stype': 'ID'
            },
            'cols': [
                {
                    'name': 'label',
                    'dtype': 'bool',
                    'stype': 'categorical'
                },
                {
                    'name': 'age',
                    'dtype': 'int64',
                    'stype': 'numerical'
                },
            ],
            'fkeys': [],
        }]
    }
    return GraphSnapshot('snapshot', tmp_path / 'index', config)
