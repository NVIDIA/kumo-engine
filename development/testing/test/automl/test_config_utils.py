# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from kumoapi.encoder import Hash, Index, MultiCategorical, Numerical

from kumo.automl.config_utils import get_encoders_from_pquery


def test_get_encoders_from_pquery():
    pquery = MagicMock()
    pquery.graph.tables = [MagicMock(), MagicMock()]
    pquery.graph.tables[0].name = "table0"
    pquery.graph.tables[1].name = "table1"
    pquery.graph.tables[0].cols = [MagicMock(), MagicMock()]
    pquery.graph.tables[0].cols[0].name = "col0"
    pquery.graph.tables[0].cols[1].name = "col1"
    pquery.graph.tables[0].cols[0].encoder = Index()
    pquery.graph.tables[0].cols[1].encoder = Numerical()
    pquery.graph.tables[1].cols = [MagicMock(), MagicMock()]
    pquery.graph.tables[1].cols[0].name = "col2"
    pquery.graph.tables[1].cols[1].name = "col3"
    pquery.graph.tables[1].cols[0].encoder = Hash(num_components=1)
    pquery.graph.tables[1].cols[1].encoder = MultiCategorical()
    encoders_string = get_encoders_from_pquery(pquery)
    assert encoders_string == {
        "table0.col0": "Index(min_occ=1, na_strategy='separate')",
        "table0.col1": "Numerical(scaler=None, na_strategy='mean')",
        "table1.col2": "Hash(num_components=1, na_strategy='separate')",
        "table1.col3": "MultiCategorical(min_occ=1, na_strategy='zero')",
    }
