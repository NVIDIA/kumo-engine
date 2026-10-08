# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skipped-row reports reach Python application logs."""

import pytest
from diskgraph import tt_artifacts


@pytest.mark.parametrize('prediction', [False, True])
def test_builder_warns_with_distinct_rows_and_all_reasons(
        prediction, monkeypatch, caplog):
    result = {
        'row_count': 3,
        'row_failures': {
            'rows_read': 5,
            'skipped_rows': 2,
            'by_condition': [
                {'condition': 'null_weight', 'column': 'WEIGHT', 'count': 2},
                {'condition': 'empty_target_list', 'column': 'TARGET', 'count': 1},
            ],
        },
    }
    native_name = '_native_build_predict' if prediction else '_native_build'
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(tt_artifacts, native_name, build)
    kwargs = dict(output_path='table.parquet', input_timestamp_unit='nanoseconds',
                  tt_id='tt', index_path='index',
                  train_table_col_stats_config={'target_stype': 'numerical'},
                  split_stats_config={'entity_table': 'users',
                                      'entity_pk_column': 'id'})
    if prediction:
        builder = tt_artifacts.build_prediction_table_artifacts_for_existing_table
    else:
        builder = tt_artifacts.build_train_table_artifacts_for_existing_table
        kwargs['split_config'] = None
    assert builder(**kwargs) is result
    assert calls[0]['ingest_on_error'] == 'skip_row_on_error'
    assert '2 of 5 rows were skipped; training on 3 rows' in caplog.text
    assert 'null weight[WEIGHT]=2' in caplog.text
    assert 'empty target list[TARGET]=1' in caplog.text


def test_no_warning_without_skipped_rows(caplog):
    tt_artifacts._warn_skipped_rows({'row_count': 3, 'row_failures': {
        'skipped_rows': 0,
    }})
    assert not caplog.records
