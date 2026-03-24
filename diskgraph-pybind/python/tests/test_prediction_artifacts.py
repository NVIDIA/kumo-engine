import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_prediction_artifacts_module():
    root = Path(__file__).resolve().parents[1] / 'diskgraph' / 'prediction_artifacts.py'
    spec = importlib.util.spec_from_file_location('diskgraph_prediction_artifacts', root)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prediction_artifacts = _load_prediction_artifacts_module()


def _write_min_table(path: Path) -> None:
    table = pa.table({'ENTITY': [1, 2], 'arange': [0, 1]})
    pq.write_table(table, path)


def test_validate_prediction_artifact_kind():
    with pytest.raises(RuntimeError, match='kind must be tt_prediction_table_edges'):
        prediction_artifacts.validate_prediction_table_edges_artifact({'kind': 'tt_train_table_edges'})


def test_parse_prediction_artifact_rejects_non_json():
    with pytest.raises(RuntimeError, match='invalid prediction edge artifact JSON'):
        prediction_artifacts.parse_prediction_table_edges_artifact('{')


def test_build_prediction_edges_requires_explicit_prediction_mode(tmp_path: Path):
    _write_min_table(tmp_path / 'pred.parquet')
    with pytest.raises(RuntimeError, match='task_table_edge_kind must be "prediction"'):
        prediction_artifacts.build_prediction_table_edges_artifact(
            prediction_table_path=str(tmp_path / 'pred.parquet'),
            index_path='unused',
            cfg={
                'enabled': True,
                'task_table_edge_kind': 'train',
                'fkeys': [{'column_name': 'ENTITY', 'target_table': 'users'}],
            },
        )


def test_validate_prediction_artifact_mask_present_requires_path():
    payload = {
        'version': 1,
        'kind': 'tt_prediction_table_edges',
        'format': 'tt_pred_edge_bundle_v1',
        'reverse_edges': {'k': {'path': '/tmp/edge.parquet', 'format': 'edge_parquet'}},
        'fkey_cols': {'path': '/tmp/fkey', 'format': 'fkey_cols_parquet'},
        'mask': {'present': True},
        'checksums': {'fkey_cols_sha256': 'abc'},
    }
    with pytest.raises(RuntimeError, match='mask.path is required'):
        prediction_artifacts.validate_prediction_table_edges_artifact(payload)


def test_parse_prediction_col_stats_artifact_rejects_non_json():
    with pytest.raises(RuntimeError, match='invalid prediction col-stats artifact JSON'):
        prediction_artifacts.parse_prediction_table_col_stats_artifact('{')


def test_validate_prediction_col_stats_artifact_kind():
    with pytest.raises(RuntimeError, match='kind must be tt_prediction_table_col_stats'):
        prediction_artifacts.validate_prediction_table_col_stats_artifact({'kind': 'tt_train_table_col_stats'})


def test_build_prediction_edges_success(tmp_path: Path, monkeypatch):
    _write_min_table(tmp_path / 'pred.parquet')

    class _FakeFkeyConfig:
        def __init__(self, column_name: str, target_table: str):
            self.column_name = column_name
            self.target_table = target_table

    class _FakeExportConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    captured = {}

    def _fake_export(cfg):
        captured['export_cfg'] = cfg

    def _fake_build(**kwargs):
        payload = {
            'version': 1,
            'kind': 'tt_prediction_table_edges',
            'format': 'tt_pred_edge_bundle_v1',
            'reverse_edges': {
                'users__rev_user_id__custom_prediction_table': {
                    'path': '/tmp/edge.parquet',
                    'format': 'edge_parquet',
                }
            },
            'fkey_cols': {'path': '/tmp/fkey', 'format': 'fkey_cols_parquet'},
            'mask': {'present': False, 'reason': 'mask_not_required'},
            'checksums': {'fkey_cols_sha256': 'abc'},
            'provided_fields': ['reverse_edges', 'fkey_cols', 'mask'],
        }
        return {'payload_json': json.dumps(payload)}

    monkeypatch.setattr(prediction_artifacts, '_export_tt_artifacts_native', _fake_export)
    monkeypatch.setattr(
        prediction_artifacts,
        '_build_prediction_table_edge_bundle_artifact_native',
        _fake_build,
    )
    monkeypatch.setattr(prediction_artifacts, '_fkey_config_cls', _FakeFkeyConfig)
    monkeypatch.setattr(prediction_artifacts, '_export_config_cls', _FakeExportConfig)

    out = prediction_artifacts.build_prediction_table_edges_artifact(
        prediction_table_path=str(tmp_path / 'pred.parquet'),
        index_path='unused',
        cfg={
            'enabled': True,
            'fkeys': [{'column_name': 'ENTITY', 'target_table': 'users'}],
        },
    )

    payload = json.loads(out)
    assert payload['kind'] == 'tt_prediction_table_edges'
    assert 'export_cfg' in captured


def test_build_prediction_artifact_bundle_persists_manifest(tmp_path: Path, monkeypatch):
    pred = tmp_path / 'pred.parquet'
    _write_min_table(pred)

    class _FakeFkeyConfig:
        def __init__(self, column_name: str, target_table: str):
            self.column_name = column_name
            self.target_table = target_table

    class _FakeExportConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _fake_export(_cfg):
        return None

    def _fake_build(**_kwargs):
        payload = {
            'version': 1,
            'kind': 'tt_prediction_table_edges',
            'format': 'tt_pred_edge_bundle_v1',
            'reverse_edges': {
                'users__rev_user_id__custom_prediction_table': {
                    'path': '/tmp/edge.parquet',
                    'format': 'edge_parquet',
                }
            },
            'fkey_cols': {'path': '/tmp/fkey', 'format': 'fkey_cols_parquet'},
            'mask': {'present': False, 'reason': 'mask_not_required'},
            'checksums': {'fkey_cols_sha256': 'abc'},
            'provided_fields': ['reverse_edges', 'fkey_cols', 'mask'],
        }
        return {'payload_json': json.dumps(payload)}

    def _fake_build_col_stats(**_kwargs):
        payload = {
            'version': 1,
            'kind': 'tt_prediction_table_col_stats',
            'stats_format': 'kumo_col_stats_v1',
            'row_count': 2,
            'stats': {'ENTITY__kumo.numerical__MEAN__True': {'value': 1.5}},
            'provided_fields': ['stats'],
        }
        return {'payload_json': json.dumps(payload)}

    def _fake_persist(**kwargs):
        assert kwargs['output_path'] == str(pred)
        assert kwargs['row_count'] == 2
        assert isinstance(kwargs['prediction_table_edges_json'], str)
        assert isinstance(kwargs['prediction_table_col_stats_json'], str)
        return {
            'manifest_path': str(pred) + '.prediction.manifest.json',
            'artifacts_dir': str(pred) + '.artifacts',
        }

    monkeypatch.setattr(prediction_artifacts, '_export_tt_artifacts_native', _fake_export)
    monkeypatch.setattr(
        prediction_artifacts,
        '_build_prediction_table_edge_bundle_artifact_native',
        _fake_build,
    )
    monkeypatch.setattr(
        prediction_artifacts,
        '_build_prediction_table_col_stats_artifact_native',
        _fake_build_col_stats,
    )
    monkeypatch.setattr(prediction_artifacts, '_persist_prediction_artifact_bundle_native', _fake_persist)
    monkeypatch.setattr(prediction_artifacts, '_fkey_config_cls', _FakeFkeyConfig)
    monkeypatch.setattr(prediction_artifacts, '_export_config_cls', _FakeExportConfig)

    out = prediction_artifacts.build_prediction_table_artifact_bundle(
        prediction_table_path=str(pred),
        index_path='unused',
        cfg={
            'enabled': True,
            'row_count': 2,
            'fkeys': [{'column_name': 'ENTITY', 'target_table': 'users'}],
        },
        producer_summary={'path': 'native_materialize'},
    )

    assert out['manifest_path'].endswith('.prediction.manifest.json')
    assert out['artifacts_dir'].endswith('.artifacts')
    assert isinstance(out['prediction_table_col_stats_json'], str)
    assert out['prediction_table_size_bytes'] > 0


def test_build_prediction_artifact_bundle_can_disable_col_stats(tmp_path: Path, monkeypatch):
    pred = tmp_path / 'pred.parquet'
    _write_min_table(pred)

    class _FakeFkeyConfig:
        def __init__(self, column_name: str, target_table: str):
            self.column_name = column_name
            self.target_table = target_table

    class _FakeExportConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _fake_export(_cfg):
        return None

    def _fake_build_edges(**_kwargs):
        payload = {
            'version': 1,
            'kind': 'tt_prediction_table_edges',
            'format': 'tt_pred_edge_bundle_v1',
            'reverse_edges': {'e': {'path': '/tmp/e', 'format': 'edge_parquet'}},
            'fkey_cols': {'path': '/tmp/f', 'format': 'fkey_cols_parquet'},
            'mask': {'present': False, 'reason': 'mask_not_required'},
            'checksums': {'fkey_cols_sha256': 'abc'},
            'provided_fields': ['reverse_edges', 'fkey_cols', 'mask'],
        }
        return {'payload_json': json.dumps(payload)}

    def _fake_persist(**kwargs):
        assert kwargs['prediction_table_col_stats_json'] is None
        return {
            'manifest_path': str(pred) + '.prediction.manifest.json',
            'artifacts_dir': str(pred) + '.artifacts',
        }

    monkeypatch.setattr(prediction_artifacts, '_export_tt_artifacts_native', _fake_export)
    monkeypatch.setattr(
        prediction_artifacts,
        '_build_prediction_table_edge_bundle_artifact_native',
        _fake_build_edges,
    )
    monkeypatch.setattr(prediction_artifacts, '_persist_prediction_artifact_bundle_native', _fake_persist)
    monkeypatch.setattr(prediction_artifacts, '_fkey_config_cls', _FakeFkeyConfig)
    monkeypatch.setattr(prediction_artifacts, '_export_config_cls', _FakeExportConfig)

    out = prediction_artifacts.build_prediction_table_artifact_bundle(
        prediction_table_path=str(pred),
        index_path='unused',
        cfg={
            'enabled': True,
            'row_count': 2,
            'fkeys': [{'column_name': 'ENTITY', 'target_table': 'users'}],
            'prediction_table_col_stats_config': {'enabled': False},
        },
    )
    assert out['prediction_table_col_stats_json'] is None
