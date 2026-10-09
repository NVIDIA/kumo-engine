# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import pathlib

import pandas as pd
from kumo_core.train.output import EvalOutput
from kumoapi.train import TrainingStage


def test_save_eval_output(tmp_path: pathlib.Path):
    temp_path = tmp_path.resolve().as_posix()
    original = EvalOutput(
        metrics={
            'a': 1,
            'b': 2
        },
        df=pd.DataFrame({
            'ID': [1, 2],
            'TIMESTAMP': [
                pd.Timestamp('2026-07-21'),
                pd.Timestamp('2026-07-22'),
            ],
            'CLASS': [1, 'two'],
        }),
        embedding=pd.DataFrame({
            'ID': [1],
            'EMB': [2]
        }),
    )

    original.save(temp_path, TrainingStage.VAL)
    loaded = EvalOutput.load(osp.join(temp_path, 'val_output.pt'))

    assert original.metrics == loaded.metrics
    expected_df = original.df.copy()
    expected_df['CLASS'] = ['1', 'two']
    pd.testing.assert_frame_equal(expected_df, loaded.df)
    pd.testing.assert_frame_equal(original.embedding, loaded.embedding)
