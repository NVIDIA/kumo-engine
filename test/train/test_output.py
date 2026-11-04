# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os.path as osp
import pathlib
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pandas as pd
from kumo_core.train import output
from kumo_core.train.output import EvalFrames, EvalOutput
from kumo_core.types import TaskType
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


def _spooled(frame: pd.DataFrame, chunk_rows: int) -> EvalFrames:
    frames = EvalFrames()
    for start in range(0, len(frame), chunk_rows):
        frames.append(frame.iloc[start:start + chunk_rows])
    return frames


def test_iterator_keeps_spool_alive():
    frames = _spooled(pd.DataFrame({'SCORE': [1.0, 2.0]}), 1)
    it = iter(frames)
    del frames
    gc.collect()

    assert pd.concat(it)['SCORE'].tolist() == [1.0, 2.0]


def test_save_df_keeps_edits_to_materialized_df(tmp_path: pathlib.Path):
    out = EvalOutput({}, frames=_spooled(pd.DataFrame({'SCORE': [1., 2.]}), 1))
    out.df['SCORE'] = [3., 4.]
    out.save_df(str(tmp_path), TrainingStage.TEST)

    saved = pd.read_parquet(tmp_path / 'test_df.parquet')
    assert saved['SCORE'].tolist() == [3., 4.]


def test_holdout_keeps_leading_rows_that_fit(tmp_path: pathlib.Path):
    sizes = np.repeat([200, 100, 1], 1000)
    frame = pd.DataFrame({'ENTITY': ['x' * int(k) for k in sizes]})
    cap = int(frame.memory_usage(deep=True).sum() / 4)

    EvalOutput({}, frames=_spooled(frame, 256)).save_df(
        str(tmp_path), TrainingStage.TEST, max_size=cap)

    saved = pd.read_parquet(tmp_path / 'test_df.parquet')
    pd.testing.assert_frame_equal(saved, frame.head(len(saved)))
    assert saved.memory_usage(deep=True, index=False).sum() <= cap
    meta = json.loads((tmp_path / 'test_df_meta.json').read_text())
    assert meta == {
        'total_rows': 3000,
        'saved_rows': len(saved),
        'truncated': True
    }


def _save_peak(tmp_path: pathlib.Path, num_chunks: int) -> int:
    chunk = pd.DataFrame({
        'ENTITY': [f'cust_{i}' for i in range(20_000)],
        'SCORE': np.random.rand(20_000),
    })
    frames = EvalFrames()
    for _ in range(num_chunks):
        frames.append(chunk)
    tracemalloc.start()
    output = EvalOutput({}, frames=frames)
    output.save(str(tmp_path), TrainingStage.TEST)
    output.save_df(str(tmp_path), TrainingStage.TEST)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def test_save_memory_does_not_grow_with_rows(tmp_path: pathlib.Path,
                                             monkeypatch):
    monkeypatch.setattr(output, '_ROW_GROUP_ROWS', 50_000)
    small = _save_peak(tmp_path / 'small', num_chunks=4)
    large = _save_peak(tmp_path / 'large', num_chunks=40)

    assert large < 1.5 * small


def test_regression_stats_match_whole_frame(tmp_path: pathlib.Path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        'ENTITY': np.arange(30_000),
        'TARGET': rng.normal(size=30_000),
        'TARGET_PRED': rng.normal(size=30_000),
    })
    distribution = {
        'target_mean': 0.0,
        'histogram': {
            'bins': [-1.0, 0.0, 1.0],
            'counts': [10, 10],
            'bin_width': 1.0,
            'num_bins': 2,
        },
    }
    stage_stats = SimpleNamespace(target_distribution=distribution)
    task = SimpleNamespace(
        task_type=TaskType.REGRESSION,
        get_train_table_stats=lambda after_split: SimpleNamespace(
            get_stage_table_stats=lambda stage: stage_stats),
    )
    saved = {}
    for name, out in [
        ('whole', EvalOutput({}, df=frame)),
        ('chunked', EvalOutput({}, frames=_spooled(frame, 4_096))),
    ]:
        root = tmp_path / name
        np.random.seed(0)
        out.save_all(str(root), TrainingStage.TEST, task)
        saved[name] = root

    for path in [
            'test_prediction_histogram_and_stats.json',
            'test_target_prediction_heatmap/sampled_df.parquet',
            'test_df.parquet',
    ]:
        assert ((saved['whole'] / path).read_bytes() == (saved['chunked'] /
                                                         path).read_bytes())


def test_zero_row_holdout_keeps_readable_schema(tmp_path: pathlib.Path):
    frame = pd.DataFrame({'ENTITY': ['large row'], 'SCORE': [1.]})
    EvalOutput({}, frame).save_df(str(tmp_path), TrainingStage.TEST,
                                  max_size=1)

    saved = pd.read_parquet(tmp_path / 'test_df.parquet')
    assert saved.empty
    assert list(saved.dtypes) == [object, np.float64]
    meta = json.loads((tmp_path / 'test_df_meta.json').read_text())
    assert meta['saved_rows'] == 0
    assert meta['truncated']


def test_empty_frames_sample_and_heatmap_keep_schema(tmp_path: pathlib.Path):
    frame = pd.DataFrame({
        'TARGET': pd.Series([], dtype=float),
        'TARGET_PRED': pd.Series([], dtype=float),
    })
    assert EvalFrames.of(frame).sample(0).dtypes.equals(frame.dtypes)

    stage_stats = SimpleNamespace(target_distribution={
        'target_mean': 0.0,
        'histogram': {
            'bins': [0.0, 1.0],
            'counts': [1],
            'bin_width': 1.0,
            'num_bins': 1,
        },
    })
    task = SimpleNamespace(
        task_type=TaskType.REGRESSION,
        get_train_table_stats=lambda after_split: SimpleNamespace(
            get_stage_table_stats=lambda stage: stage_stats),
    )
    EvalOutput({}, frame).save_prediction_heatmap(str(tmp_path),
                                                  TrainingStage.TEST, task)

    sampled = pd.read_parquet(tmp_path / 'test_target_prediction_heatmap' /
                              'sampled_df.parquet')
    assert sampled.empty
    assert list(sampled.columns) == ['TARGET', 'TARGET_PRED']
