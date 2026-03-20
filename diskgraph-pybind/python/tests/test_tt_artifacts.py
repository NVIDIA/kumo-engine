from diskgraph.tt_artifacts import _build_stats_artifact


def test_build_stats_artifact_includes_required_keys():
    native_summary = {
        'stats': {
            'num_total_rows': 100,
            'num_train_rows': 80,
            'num_entities': 10,
            'num_targets': 7,
            'num_links': None,
            'exact_stats': {
                'timestamp': {
                    'num_unique': 5,
                },
            },
        },
    }

    payload = _build_stats_artifact(native_summary, row_count=100)

    assert payload['version'] == 1
    assert payload['kind'] == 'tt_stats'
    assert payload['num_total_rows'] == 100
    assert payload['num_train_rows'] == 80
    assert payload['num_target_rows'] == 80
    assert payload['num_timeframes'] == 5
    assert payload['num_timeframes_post_filter'] == 5
    assert payload['avg_entities_per_timeframe'] == 16

    # Strict native_full-required keys must always be present.
    for key in (
        'avg_entities_per_timeframe',
        'node_degree_stats',
        'num_target_rows',
        'num_timeframes',
        'num_timeframes_post_filter',
        'scoring_table_column_stats',
        'stats_per_timeframe',
        'target_distribution',
    ):
        assert key in payload


def test_build_stats_artifact_defaults_when_timestamp_stats_missing():
    payload = _build_stats_artifact({'stats': {}}, row_count=11)

    assert payload['num_total_rows'] == 11
    assert payload['num_train_rows'] == 11
    assert payload['num_target_rows'] == 11
    assert payload['num_timeframes'] == 0
    assert payload['num_timeframes_post_filter'] == 0
    assert payload['avg_entities_per_timeframe'] == 0


def test_build_stats_artifact_temporal_rows_sum_exact():
    plan_json = '{"temporal": {"timestamps": [100, 200, 300]}}'
    payload = _build_stats_artifact({'stats': {'num_train_rows': 10}}, row_count=10, plan_json=plan_json)
    assert sum(int(x['num_train_rows']) for x in payload['stats_per_timeframe']) == 10


def test_build_stats_artifact_temporal_timestamp_ms_is_normalized():
    plan_json = '{"temporal": {"timestamps": [1700000000000, 1700000001000]}}'
    payload = _build_stats_artifact({'stats': {'num_train_rows': 2}}, row_count=2, plan_json=plan_json)
    assert payload['stats_per_timeframe'][0]['anchor_date'].startswith('2023-11')
