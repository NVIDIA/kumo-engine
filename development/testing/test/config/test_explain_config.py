# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from kumoapi.explain import GraphGradientScore


def test_graph_gradient_score():
    score = GraphGradientScore()

    score['TABLE_1']['COL_1'] = 1.0
    score['TABLE_1']['COL_2'] = 2.0
    assert score['TABLE_1'].total_score == 3.0

    score['TABLE_2']['COL_3'] = 3.0
    score['TABLE_2']['COL_4'] = 4.0
    assert score['TABLE_2'].total_score == 7.0

    df = score.normalize().to_pandas()
    expected = pd.DataFrame({
        'table': ['TABLE_2', 'TABLE_2', 'TABLE_1', 'TABLE_1'],
        'column': ['COL_4', 'COL_3', 'COL_2', 'COL_1'],
        'score': [0.4, 0.3, 0.2, 0.1],
    })
    pd.testing.assert_frame_equal(df, expected)
