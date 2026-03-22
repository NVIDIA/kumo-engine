from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import diskgraph.tt_artifacts as tt_artifacts
from diskgraph import SplitSpec
from diskgraph.tt_artifacts import _normalize_downsampling_config, execute_tt_plan_with_artifacts


def _write_min_table(path: Path) -> None:
    table = pa.table({
        'ENTITY': [1, 2],
        'TIMESTAMP': [10, 11],
        'TARGET': [0, 1],
        'arange': [0, 1],
    })
    pq.write_table(table, path)


def _mock_native_materialize_ok(**_kwargs):
    return {
        'rows_written': 2,
        'summary_json': '{"stats": {"num_total_rows": 2, "num_train_rows": 2}}',
        'split_assignments': bytes([0, 1]),
        'split_family': 'FixedSplit',
        'split_authoritative': True,
        'split_reason': None,
        'downsampling_handled': True,
        'downsampling_applied': False,
        'pre_downsample_row_count': 2,
        'train_table_col_stats_json': '{"version":1,"kind":"tt_train_table_col_stats","stats":{},"provided_fields":["stats"]}',
    }


def _mock_native_tt_stats(**_kwargs):
    return {
        'payload_json': '{"version":2,"kind":"tt_stats","num_total_rows":2,"num_train_rows":2,'
                        '"num_entities":2,"num_targets":2,"num_target_rows":2,'
                        '"num_timeframes":0,"num_timeframes_post_filter":0,'
                        '"avg_entities_per_timeframe":0,"node_degree_stats":null,'
                        '"stats_per_timeframe":[],"target_distribution":{},'
                        '"scoring_table_column_stats":null,'
                        '"provided_fields":["num_total_rows","num_train_rows","num_entities",'
                        '"num_targets","num_target_rows","num_timeframes",'
                        '"num_timeframes_post_filter","avg_entities_per_timeframe",'
                        '"node_degree_stats","stats_per_timeframe","target_distribution",'
                        '"scoring_table_column_stats"]}'
    }


def _mock_native_split_stats(**_kwargs):
    return {
        'payload_json': '{"version":1,"kind":"tt_split_stats","train_table_stats":{"num_train_rows":1},'
                        '"val_table_stats":{"num_train_rows":1},"test_table_stats":null,'
                        '"provided_fields":["train_table_stats","val_table_stats","test_table_stats"]}'
    }


def _mock_native_persist(**_kwargs):
    return {'manifest_path': '/tmp/fake.manifest.json', 'artifacts_dir': '/tmp/fake.artifacts'}


def test_downsampling_config_ignores_unknown_strategy_field():
    # strategy field is no longer used; unknown values should not cause errors
    cfg = _normalize_downsampling_config({'enabled': True, 'max_rows': 10, 'strategy': 'anything'})
    assert cfg.enabled is True
    assert cfg.max_rows == 10


def test_downsampling_config_rejects_removed_knobs():
    with pytest.raises(RuntimeError, match='no longer accepts seed'):
        _normalize_downsampling_config({'enabled': True, 'max_rows': 10, 'seed': 1})
    with pytest.raises(RuntimeError, match='no longer accepts fail_on_unsupported'):
        _normalize_downsampling_config({'enabled': True, 'max_rows': 10, 'fail_on_unsupported': True})


def test_rejects_non_split_spec(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', _mock_native_materialize_ok)
    with pytest.raises(TypeError, match='requires split_config as SplitSpec'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config={'split_kind': 'FixedSplit', 'ratio': [0.5, 0.25, 0.25]},
        )


def test_fails_without_native_materialize_with_split(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', None)
    with pytest.raises(RuntimeError, match=r'missing native helper\(s\): execute_tt_plan_materialize_with_split'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
        )


def test_fails_without_downsampling_metadata(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')

    def bad_materialize(**_kwargs):
        out = _mock_native_materialize_ok()
        out['downsampling_handled'] = False
        return out

    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', bad_materialize)
    with pytest.raises(RuntimeError, match='did not handle downsampling metadata'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
            downsampling={'enabled': True, 'max_rows': 1},
        )


def test_fails_when_materialize_omits_train_table_col_stats_json(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')

    def bad_materialize(**_kwargs):
        out = _mock_native_materialize_ok()
        out.pop('train_table_col_stats_json', None)
        return out

    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', bad_materialize)
    monkeypatch.setattr(tt_artifacts, '_build_tt_stats_artifact_native', _mock_native_tt_stats)
    monkeypatch.setattr(tt_artifacts, '_build_split_stats_artifact_native', _mock_native_split_stats)
    monkeypatch.setattr(tt_artifacts, '_persist_tt_artifact_bundle', _mock_native_persist)
    with pytest.raises(RuntimeError, match='did not provide train_table_col_stats_json'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
        )


def test_fails_without_native_tt_stats_builder(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', _mock_native_materialize_ok)
    monkeypatch.setattr(tt_artifacts, '_build_tt_stats_artifact_native', None)
    monkeypatch.setattr(tt_artifacts, '_build_split_stats_artifact_native', _mock_native_split_stats)
    monkeypatch.setattr(tt_artifacts, '_persist_tt_artifact_bundle', _mock_native_persist)
    with pytest.raises(RuntimeError, match=r'missing native helper\(s\): build_tt_stats_artifact'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
        )


def test_fails_without_native_persist_helper(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', _mock_native_materialize_ok)
    monkeypatch.setattr(tt_artifacts, '_build_tt_stats_artifact_native', _mock_native_tt_stats)
    monkeypatch.setattr(tt_artifacts, '_build_split_stats_artifact_native', _mock_native_split_stats)
    monkeypatch.setattr(tt_artifacts, '_persist_tt_artifact_bundle', None)
    with pytest.raises(RuntimeError, match=r'missing native helper\(s\): persist_tt_artifact_bundle'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
        )


def test_fails_with_multiple_missing_native_helpers_lists_all(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', None)
    monkeypatch.setattr(tt_artifacts, '_build_tt_stats_artifact_native', None)
    with pytest.raises(RuntimeError) as exc:
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
        )
    msg = str(exc.value)
    assert 'execute_tt_plan_materialize_with_split' in msg
    assert 'build_tt_stats_artifact' in msg


def test_uses_split_family_hint_when_split_kind_missing(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    captured = {}

    def materialize_capture(**kwargs):
        captured.update(kwargs)
        return _mock_native_materialize_ok()

    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', materialize_capture)
    monkeypatch.setattr(tt_artifacts, '_build_tt_stats_artifact_native', _mock_native_tt_stats)
    monkeypatch.setattr(tt_artifacts, '_build_split_stats_artifact_native', _mock_native_split_stats)
    monkeypatch.setattr(tt_artifacts, '_persist_tt_artifact_bundle', _mock_native_persist)

    execute_tt_plan_with_artifacts(
        plan_json='{}',
        index_path='unused',
        output_path=str(tmp_path / 'tt.parquet'),
        split_config=SplitSpec(kind=None, ratio=[0.5, 0.25, 0.25]),
        split_family_hint='FixedSplit',
    )
    assert captured.get('split_family_hint') == 'FixedSplit'


def test_requires_split_family_when_split_kind_missing(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', _mock_native_materialize_ok)
    with pytest.raises(RuntimeError, match='requires split family'):
        execute_tt_plan_with_artifacts(
            plan_json='{}',
            index_path='unused',
            output_path=str(tmp_path / 'tt.parquet'),
            split_config=SplitSpec(kind=None, ratio=[0.5, 0.25, 0.25]),
        )


def test_success_with_native_helpers_only(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'tt.parquet')
    captured = {}

    def persist_capture(**kwargs):
        captured.update(kwargs)
        return _mock_native_persist(**kwargs)

    monkeypatch.setattr(tt_artifacts, '_execute_tt_plan_materialize_with_split', _mock_native_materialize_ok)
    monkeypatch.setattr(tt_artifacts, '_build_tt_stats_artifact_native', _mock_native_tt_stats)
    monkeypatch.setattr(tt_artifacts, '_build_split_stats_artifact_native', _mock_native_split_stats)
    monkeypatch.setattr(tt_artifacts, '_persist_tt_artifact_bundle', persist_capture)

    out = execute_tt_plan_with_artifacts(
        plan_json='{}',
        index_path='unused',
        output_path=str(tmp_path / 'tt.parquet'),
        split_config=SplitSpec(kind='FixedSplit', ratio=[0.5, 0.25, 0.25]),
    )
    assert out['row_count'] == 2
    assert out['manifest_path'] == '/tmp/fake.manifest.json'
    assert isinstance(captured.get('train_table_col_stats_json'), str)
    assert captured['train_table_col_stats_json']
