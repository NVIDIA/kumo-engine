# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.engine.evidence import (
    EvidenceConfig,
    EvidenceValidationError,
    PredictionEvidenceBundle,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


@pytest.mark.parametrize('status', ['violated', 'unavailable'])
def test_temporal_failure_cannot_claim_complete(source, tmp_path, status):
    local = source.run_root / 'xai/local'
    path = local / 'rows' / ('a' * 64 + '.json')
    row = json.loads(path.read_text())
    row['temporal_validation'] = {'status': status}
    path.write_bytes(canonical(row))
    (local / 'index.jsonl').write_bytes(
        canonical({
            key: value
            for key, value in row.items() if key != 'cells'
        }) + b'\n')
    with pytest.raises(EvidenceValidationError, match='temporal coverage'):
        source.export(tmp_path / 'invalid-complete')


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    local = root / 'xai/local'
    (local / 'samples').mkdir(parents=True)
    (local / 'rows').mkdir()
    identifier = 'a' * 64
    graph = {
        'origin': 'original_forward',
        'nodes': {
            'products': {
                'pkeys': [7],
                'num_nodes': 1
            }
        },
        'edges': []
    }
    row = {
        'prediction_id':
        identifier,
        'prediction':
        2.5,
        'origin':
        'original_forward',
        'entity':
        7,
        'task_row_id': [2],
        'time_ns': [3],
        'status':
        'zero_signal',
        'reason':
        None,
        'sample_sha256':
        hashlib.sha256(canonical(graph)).hexdigest(),
        'cells': [{
            'abs_gradient': 0.,
            'signed_gradient': 0.,
            'table': 'products',
            'pkey': 7,
            'column': 'quantity',
            'mapper_value': 0.,
            'input_dtype': 'torch.float32',
            'missing': False
        }]
    }
    (local / 'samples' / f'{identifier}.json').write_bytes(canonical(graph))
    (local / 'rows' / f'{identifier}.json').write_bytes(canonical(row))
    (local / 'index.jsonl').write_bytes(
        canonical({
            k: v
            for k, v in row.items() if k != 'cells'
        }) + b'\n')
    (local / 'manifest.json').write_bytes(canonical({'complete': True}))
    parquet = root / 'output.parquet'
    pq.write_table(
        pa.table({
            'PREDICTION_ID': [identifier],
            'ENTITY': [7],
            'TARGET_PRED': [2.5],
            'CUTOFF_NS': [3],
            'TASK_ROW_ID': [2]
        }), parquet)
    result = SimpleNamespace(files=[str(parquet)],
                             explanations=SimpleNamespace(run_root=str(root)))
    return PredictionEvidenceBundle.from_result(result)


def test_portable_reopen_without_source(source, tmp_path):
    bundle = source.export(tmp_path / 'bundle')
    shutil.rmtree(source.run_root)
    moved = bundle.export(tmp_path / 'moved')
    shutil.rmtree(bundle.root)
    loaded = PredictionEvidenceBundle.load(moved.root)
    assert loaded.validate()['prediction_rows'] == 1
    assert loaded.validate()['evidence_complete']
    assert list(loaded.iter_predictions())[0]['TARGET_PRED'] == 2.5
    assert list(loaded.iter_numerical_evidence())[0]['status'] == 'zero_signal'
    assert loaded.scoring_graph('a' * 64)['edges'] == []


def test_corruption_and_extra_file_rejected(source, tmp_path):
    bundle = source.export(tmp_path / 'bundle')
    (bundle.root / 'unlisted').write_text('bad')
    with pytest.raises(EvidenceValidationError, match='Unlisted'):
        bundle.validate()
    (bundle.root / 'unlisted').unlink()
    (bundle.root / 'xai/local/index.jsonl').write_text('{}')
    with pytest.raises(EvidenceValidationError, match='integrity'):
        bundle.validate()


def test_prediction_mismatch_never_published(source, tmp_path):
    pq.write_table(
        pa.table({
            'PREDICTION_ID': ['a' * 64],
            'TARGET_PRED': [3.5]
        }), source.files[0])
    with pytest.raises(EvidenceValidationError, match='emitted'):
        source.export(tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()
    assert list(tmp_path.glob('bundle.partial-*'))


def test_duplicate_prediction_rejected(source, tmp_path):
    pq.write_table(
        pa.table({
            'PREDICTION_ID': ['a' * 64] * 2,
            'TARGET_PRED': [2.5] * 2
        }), source.files[0])
    with pytest.raises(EvidenceValidationError, match='duplicate'):
        source.export(tmp_path / 'bundle')


def test_paths_and_overwrite_rejected(source, tmp_path):
    bundle = source.export(tmp_path / 'bundle')
    with pytest.raises(EvidenceValidationError):
        bundle.scoring_graph('../escape')
    with pytest.raises(FileExistsError):
        source.export(bundle.root)
    (source.run_root / 'xai/link').symlink_to('/etc/passwd')
    with pytest.raises(EvidenceValidationError, match='symlink'):
        source.export(tmp_path / 'other')


def test_legacy_export(source, tmp_path):
    bundle = source.export(tmp_path / 'bundle')
    destination = bundle.export_legacy(tmp_path / 'legacy')
    assert (destination / 'xai/local/rows' / ('a' * 64 + '.json')).is_file()


def test_configuration():
    assert EvidenceConfig().numerical_sensitivity == 'signed_and_absolute'
    with pytest.raises(ValueError):
        EvidenceConfig(numerical_sensitivity='contribution')
    with pytest.raises(TypeError):
        EvidenceConfig(scoring_graph=1)


def test_calibration_child_bundle(source, tmp_path):
    from kumo.engine.evidence import PredictionEvidenceSource
    from kumo.engine.uncertainty import (
        CalibrationSpec,
        PredictionIntervalCalibrator,
    )
    model, policy = 'b' * 64, 'c' * 64
    source = PredictionEvidenceSource(source.files, source.run_root, {
        'model_sha256': model,
        'scoring_sha256': policy
    })
    bundle = source.export(tmp_path / 'bundle')
    calibrator = PredictionIntervalCalibrator.fit(
        [{
            'prediction_id': str(i),
            'prediction': 0.
        } for i in range(20)], {str(i): 1.
                                for i in range(20)},
        spec=CalibrationSpec(model_sha256=model,
                             scoring_sha256=policy), population_id='fresh',
        separation_declaration='Synthetic independent fixture')
    child = bundle.with_intervals(calibrator, output=tmp_path / 'child')
    interval = list(child.iter_intervals())[0]
    assert interval['lower'] == 1.5
    assert interval['upper'] == 3.5
    assert not list(bundle.iter_intervals())
    copied = child.export(tmp_path / 'copied')
    assert list(copied.iter_intervals()) == [interval]
    assert copied.manifest['parent_manifest_sha256']


def test_perturbation_child_publication(source, tmp_path, monkeypatch):
    from kumo.engine.evidence import RecordRemovalPlan

    def responses(model, sample, record, provenance, plan):
        yield {
            'origin': 'perturbation',
            'prediction_id': record['prediction_id'],
            'original_sample_sha256': record['sample_sha256'],
            'baseline': record['prediction'],
            'perturbed_prediction': 1.5,
            'signed_change': 1.,
            'absolute_change': 1.,
            'resampled': False
        }

    monkeypatch.setattr('kumo.engine.perturbation.responses_from_trainer',
                        responses)
    bundle = source.export(tmp_path / 'bundle')
    plan = RecordRemovalPlan(groups=({
        'table': 'events',
        'pkey': 9
    }, ), removable_tables=('events', ), foreign_keys=())
    child = bundle.perturb(model=object(), plan=plan, prediction_ids=['a' * 64],
                           output=tmp_path / 'child')
    assert child.manifest['perturbations']['responses'] == 1
    assert 'perturbations' not in bundle.manifest
    with pytest.raises(EvidenceValidationError, match='not retained'):
        bundle.perturb(model=object(), plan=plan, prediction_ids=['b' * 64],
                       output=tmp_path / 'unknown')
    assert not (tmp_path / 'unknown').exists()


def test_source_lineage_mapper_join(source, tmp_path):
    record = {
        'table': 'products',
        'pkey': 7,
        'column': 'quantity',
        'source_value': 0.,
        'source': {
            'file': 'input.csv',
            'row': 2,
            'column': 'quantity',
            'sha256': 'd' * 64
        }
    }
    bundle = source.export(tmp_path / 'bundle', source_records=iter([record]))
    assert bundle.manifest['source_lineage'] == 'provided_mapper_verified'
    assert list(bundle.iter_source_records()) == [record]
    record['source_value'] = 5.
    with pytest.raises(EvidenceValidationError, match='mapper coordinate'):
        source.export(tmp_path / 'bad', source_records=iter([record]))
    assert not (tmp_path / 'bad').exists()
    with pytest.raises(EvidenceValidationError, match='Missing source'):
        source.export(tmp_path / 'missing', source_records=iter([]))
