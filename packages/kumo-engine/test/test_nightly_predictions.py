# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check bounded Arrow ordering and retained ranked cutoff evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_predictions.py'
SPEC = importlib.util.spec_from_file_location('nightly_predictions', SOURCE)
predictions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictions)


def write_shards(frame, root, count):
    paths = []
    for index, rows in enumerate(np.array_split(np.arange(len(frame)), count)):
        path = root / f'input-{index}.parquet'
        pq.write_table(
            pa.Table.from_pandas(frame.iloc[rows], preserve_index=False), path)
        paths.append(path)
    return paths


def test_arrow_partition_sort_preserves_uint64_and_nanoseconds(
        tmp_path, monkeypatch):
    frame = pd.DataFrame(
        dict(
            ENTITY=np.repeat(np.arange(2**63, 2**63 + 70, dtype='uint64'), 13),
            CLASS=np.tile(np.arange(13, dtype='int64'), 70),
            SCORE=np.tile(np.arange(13, dtype='float64') % 3,
                          70), ANCHOR_TIMESTAMP=pd.to_datetime(
                              np.arange(910) + 1735689600000000001)))
    shuffled = frame.sample(frac=1, random_state=3)
    paths = write_shards(shuffled, tmp_path, 70)
    expected = frame.sort_values(['ENTITY', 'SCORE', 'CLASS'],
                                 ascending=[True, False,
                                            True], ignore_index=True)
    monkeypatch.setattr(predictions, '_SORT_ROWS', 64)
    monkeypatch.setattr(predictions, '_BATCH_ROWS', 17)
    monkeypatch.setattr(predictions, '_SAMPLE_KEYS', 16)
    read, read_sizes = pq.ParquetFile.read, []

    def bounded_read(reader, *args, **kwargs):
        read_sizes.append(reader.metadata.num_rows)
        assert reader.metadata.num_rows <= 64
        return read(reader, *args, **kwargs)

    def reject_python_rows(*args, **kwargs):
        raise AssertionError('Prediction sorting must stay in Arrow')

    monkeypatch.setattr(pd.DataFrame, 'from_records', reject_python_rows)
    monkeypatch.setattr(pq.ParquetFile, 'read', bounded_read)
    work = tmp_path / 'spill'
    batches = list(predictions.ordered_batches(paths, work))
    assert max(map(len, batches)) <= 17
    assert len(read_sizes) > 1
    pd.testing.assert_frame_equal(pd.concat(batches, ignore_index=True),
                                  expected)
    assert list(work.iterdir()) == []


def test_training_keys_sort_exact_timestamp_without_prediction_columns(
        tmp_path, monkeypatch):
    frame = pd.DataFrame(
        dict(
            ENTITY1=pd.Series([2**63 + 1, 2**63, 2**63, 2**63 + 1],
                              dtype='uint64'), TIMESTAMP=pd.to_datetime([
                                  1735689600000000002, 1735689600000000002,
                                  1735689600000000001, 1735689600000000001
                              ]), TARGET1=[1, 2, 3, 4]))
    paths = write_shards(frame, tmp_path, 2)
    monkeypatch.setattr(predictions, '_SORT_ROWS', 2)
    actual = pd.concat(
        predictions.ordered_batches(
            paths, tmp_path / 'spill', sort_keys=[('ENTITY1', 'ascending'),
                                                  ('TIMESTAMP', 'ascending')]),
        ignore_index=True)
    pd.testing.assert_frame_equal(
        actual, frame.sort_values(['ENTITY1', 'TIMESTAMP'], ignore_index=True))


@pytest.mark.parametrize('keys,dtype', [
    ([2**64 - 1, 2**63 + 1, 2**53 + 1, 1, 0], 'uint64'),
    ([2**63 - 1, 2**53 + 1, 0, -2**53 - 1, -2**63], 'int64'),
])
def test_sparse_extreme_keys_and_maximum_sample_split_are_exact(
        tmp_path, monkeypatch, keys, dtype):
    frame = pd.DataFrame(
        dict(ENTITY=pd.Series(np.repeat(keys, 3), dtype=dtype),
             VALUE=np.arange(15)))
    paths = write_shards(frame, tmp_path, 3)
    monkeypatch.setattr(predictions, '_SORT_ROWS', 4)
    monkeypatch.setattr(predictions, '_SAMPLE_KEYS', 1)
    actual = pd.concat(
        predictions.ordered_batches(paths, tmp_path / 'spill', ranked=False),
        ignore_index=True)
    pd.testing.assert_frame_equal(
        actual, frame.sort_values('ENTITY', kind='stable', ignore_index=True))


def test_skewed_split_sample_cannot_cause_quadratic_partition_reads(
        tmp_path, monkeypatch):
    frame = pd.DataFrame(
        dict(ENTITY=np.arange(69, -1, -1), CLASS=np.arange(70),
             SCORE=np.arange(70.)))
    paths = write_shards(frame, tmp_path, 1)
    monkeypatch.setattr(predictions, '_SORT_ROWS', 1)
    monkeypatch.setattr(predictions, '_SAMPLE_KEYS', 1)
    batches, rows = pq.ParquetFile.iter_batches, []

    def counted_batches(reader, *args, **kwargs):
        for batch in batches(reader, *args, **kwargs):
            rows.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', counted_batches)
    actual = pd.concat(predictions.ordered_batches(paths, tmp_path / 'spill'),
                       ignore_index=True)
    assert actual.ENTITY.tolist() == list(range(70))
    assert sum(rows) <= len(frame) * 10


@pytest.mark.parametrize('damage', [
    lambda frame: frame.assign(ENTITY=['1', '2']),
    lambda frame: frame.assign(ENTITY=pd.Series([1, None], dtype='Int64')),
    lambda frame: frame.assign(CLASS=pd.Series([1, None], dtype='Int64')),
    lambda frame: frame.drop(columns='CLASS'),
    lambda frame: frame.assign(SCORE=[float('inf'), .2]),
    lambda frame: frame.assign(SCORE=[None, .2]),
    lambda frame: frame.assign(SCORE=[True, False]),
])
def test_invalid_prediction_schema_or_values_fail_and_clean_spills(
        tmp_path, damage):
    frame = damage(
        pd.DataFrame(dict(ENTITY=[1, 2], CLASS=[1, 2], SCORE=[.1, .2])))
    paths = write_shards(frame, tmp_path, 1)
    work = tmp_path / 'spill'
    with pytest.raises(ValueError):
        list(predictions.ordered_batches(paths, work))
    assert list(work.iterdir()) == []


def test_schema_mismatch_is_checked_even_for_empty_shards(tmp_path):
    frame = pd.DataFrame(dict(ENTITY=[1], CLASS=[1], SCORE=[.1]))
    paths = write_shards(frame, tmp_path, 1)
    empty = tmp_path / 'empty.parquet'
    pq.write_table(pa.table(dict(ENTITY=pa.array([], type=pa.int64()))), empty)
    work = tmp_path / 'spill'
    with pytest.raises(ValueError, match='schemas|columns'):
        list(predictions.ordered_batches(paths + [empty], work))
    assert list(work.iterdir()) == []


def test_reader_close_removes_spills_and_empty_input_is_valid(
        tmp_path, monkeypatch):
    frame = pd.DataFrame(
        dict(ENTITY=np.arange(200), CLASS=np.arange(200),
             SCORE=np.arange(200, dtype=float)))
    paths = write_shards(frame.sample(frac=1, random_state=1), tmp_path, 7)
    monkeypatch.setattr(predictions, '_SORT_ROWS', 16)
    reader = predictions.ordered_batches(paths, tmp_path / 'spill')
    assert len(next(reader))
    assert list((tmp_path / 'spill').iterdir())
    reader.close()
    assert list((tmp_path / 'spill').iterdir()) == []
    assert list(predictions.ordered_batches([], tmp_path / 'empty')) == []
    assert list((tmp_path / 'empty').iterdir()) == []


def test_one_entity_cannot_overflow_the_sort_memory_bound(
        tmp_path, monkeypatch):
    frame = pd.DataFrame(
        dict(ENTITY=[1] * 17, CLASS=np.arange(17),
             SCORE=np.arange(17, dtype=float)))
    paths = write_shards(frame, tmp_path, 1)
    monkeypatch.setattr(predictions, '_SORT_ROWS', 16)
    with pytest.raises(ValueError, match='entity|Entity'):
        list(predictions.ordered_batches(paths, tmp_path / 'spill'))
    assert list((tmp_path / 'spill').iterdir()) == []


def ranked_frame():
    return pd.DataFrame(
        dict(ENTITY=[100] * 5 + [200] * 5, CLASS=[1, 2, 3, 4, 5] * 2,
             SCORE=[.9, .8, .7, .2, .1] * 2))


def test_ranked_stream_keeps_cross_batch_cutoff_and_exclusions():
    frame = ranked_frame()
    result = predictions.ranked_rows(
        [frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]], [100, 200],
        list(range(1, 7)), [[1], []], 2, 5)
    assert result['ranked_ids'] == [[2, 3], [1, 2]]
    assert result['boundary_ids'] == [[2, 3, 4, 5], [1, 2, 3, 4]]
    assert result['raw_rows'] == 10


def test_cutoff_tie_retries_only_ambiguous_entities():
    frame = pd.DataFrame(
        dict(ENTITY=[100] * 3 + [200] * 3, CLASS=[1, 2, 3] * 2,
             SCORE=[.9, .8, .8, .9, .8, .7]))
    requests = []

    def retry(ids, limit):
        requests.append((ids, limit))
        return [
            pd.DataFrame(
                dict(ENTITY=[100] * 6, CLASS=list(range(1, 7)),
                     SCORE=[.9, .8, .8, .7, .6, .5]))
        ]

    result = predictions.ranked_rows([frame], [100, 200], list(range(1, 7)),
                                     [[], []], 2, 3, retry=retry)
    assert requests == [([100], 6)]
    assert result['ranked_ids'] == [[1, 2], [1, 2]]
    assert result['boundary_ids'][0] == [1, 2, 3, 4]
    assert result['raw_rows'] == 12
    assert result['cutoff_retries'] == [dict(ids=[100], raw_limit=6)]


def test_cutoff_retry_stays_bounded_and_rejects_unresolved_ties():
    frame = pd.DataFrame(dict(ENTITY=[100] * 3, CLASS=[1, 2, 3],
                              SCORE=[1.] * 3))
    requests = []

    def retry(ids, limit):
        requests.append(limit)
        return [
            pd.DataFrame(
                dict(ENTITY=[100] * limit, CLASS=list(range(1, limit + 1)),
                     SCORE=[1.] * limit))
        ]

    with pytest.raises(ValueError, match='cannot certify'):
        predictions.ranked_rows([frame], [100], list(range(1, 1001)), [[]], 2,
                                3, retry=retry)
    assert requests == [6, 12, 24, 48, 96, 192, 384, 500]


def test_cutoff_retry_does_not_hide_invalid_coordinates():
    frame = pd.DataFrame(
        dict(ENTITY=[100] * 3, CLASS=[1, 1, 3], SCORE=[.9, .8, .8]))

    def retry(*args):
        pytest.fail('Invalid output must fail without a prediction retry')

    with pytest.raises(ValueError, match='duplicate or invalid'):
        predictions.ranked_rows([frame], [100], list(range(1, 7)), [[]], 2, 3,
                                retry=retry)


def test_ranked_stream_requires_the_exact_anchor():
    frame = ranked_frame().assign(
        ANCHOR_TIMESTAMP=pd.Timestamp('2025-01-01 00:00:00.000000001'))
    anchor = '2025-01-01 00:00:00.000000001'
    assert predictions.ranked_rows([frame], [100, 200], list(range(1, 7)),
                                   [[1], []], 2, 5,
                                   anchor=anchor)['raw_rows'] == 10
    frame.loc[0, 'ANCHOR_TIMESTAMP'] += pd.Timedelta(1, unit='ns')
    with pytest.raises(ValueError, match='anchors differ'):
        predictions.ranked_rows([frame], [100, 200], list(range(1, 7)),
                                [[1], []], 2, 5, anchor=anchor)


@pytest.mark.parametrize('damage', [
    lambda frame: frame.iloc[:5],
    lambda frame: frame.assign(CLASS=[1] * 10),
    lambda frame: frame.assign(ENTITY=[999] * 5 + [200] * 5),
    lambda frame: frame.assign(SCORE=[.9] * 10),
])
def test_ranked_stream_rejects_missing_duplicate_foreign_or_uncertified_rows(
        damage):
    with pytest.raises(ValueError):
        predictions.ranked_rows([damage(ranked_frame())], [100, 200],
                                list(range(1, 7)), [[1], []], 2, 5)


def test_ranked_membership_change_requires_retained_boundary():
    before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.9, .8]],
                  boundary_ids=[[1, 2, 3]], boundary_scores=[[.9, .8, .79999]])
    after = dict(ranked_ids=[[1, 3]], ranked_scores=[[.9, .80001]],
                 boundary_ids=[[1, 3, 2]], boundary_scores=[[.9, .80001, .8]])
    result = predictions.ranked_difference(before, after, atol=.0001)
    assert result['status'] == 'PASS'
    assert result['added_items'] == result['removed_items'] == 1
    assert result['rank_inversions'] == 1
    with pytest.raises(ValueError, match='cutoff coordinates'):
        predictions.ranked_difference(
            before,
            {key: after[key]
             for key in ('ranked_ids', 'ranked_scores')}, atol=.0001)


def test_quality_keeps_full_population_metric():
    truth = pd.DataFrame(dict(TARGET1=[[1, 2], [3]]))
    assert predictions.quality('static_link', truth,
                               dict(ranked_ids=[[2, 4], [3, 5]]), []) == .75
