# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent adversarial review of bundle and public capability boundaries."""
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_prediction_evidence_bundle import source  # noqa: F401

from kumo.engine.evidence import EvidenceConfig, EvidenceValidationError
from kumo.engine.trainer.trainer import Trainer


def resign(root, manifest):
    data = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    (root / 'manifest.json').write_bytes(data)
    (root / '_SUCCESS').write_text(hashlib.sha256(data).hexdigest())


def test_evidence_must_match_emitted_entity(source, tmp_path):
    pq.write_table(
        pa.table({
            'PREDICTION_ID': ['a' * 64],
            'ENTITY': [99],
            'TARGET_PRED': [2.5],
            'CUTOFF_NS': [3],
            'TASK_ROW_ID': [2]
        }), source.files[0])
    with pytest.raises(EvidenceValidationError):
        source.export(tmp_path / 'bundle')
    assert pq.read_table(source.files[0]).num_rows == 1


@pytest.mark.parametrize('column,value', [('CUTOFF_NS', 4), ('TASK_ROW_ID', 3)])
def test_evidence_must_match_emitted_task_identity(source, tmp_path, column,
                                                   value):
    data = pq.read_table(source.files[0]).to_pydict()
    data[column] = [value]
    pq.write_table(pa.table(data), source.files[0])
    with pytest.raises(EvidenceValidationError):
        source.export(tmp_path / 'bundle')
    assert pq.read_table(source.files[0]).num_rows == 1


def test_false_complete_claim_rejected(source, tmp_path):
    local = source.run_root / 'xai/local'
    (local /
     'manifest.json').write_text('{"complete":false,"generation_closed":true}')
    bundle = source.export(tmp_path / 'bundle')
    manifest = bundle.manifest
    manifest['evidence_complete'] = True
    resign(bundle.root, manifest)
    with pytest.raises(EvidenceValidationError):
        bundle.validate()


def test_graph_failure_remains_explicit(source, tmp_path):
    local = source.run_root / 'xai/local'
    path = local / 'rows' / ('a' * 64 + '.json')
    row = json.loads(path.read_text())
    row.update(sample_sha256=None, graph_status='failed',
               graph_reason='original_graph_capture_failed')
    path.write_text(json.dumps(row))
    (local / 'index.jsonl').write_text(
        json.dumps({
            k: v
            for k, v in row.items() if k != 'cells'
        }) + '\n')
    (local /
     'manifest.json').write_text('{"complete":false,"generation_closed":true}')
    bundle = source.export(tmp_path / 'bundle')
    assert not bundle.manifest['evidence_complete']
    assert bundle.availability('a' * 64)['scoring_graph'] == 'failed'


def test_manifest_parent_traversal_rejected_even_with_valid_hash(
        source, tmp_path):
    bundle = source.export(tmp_path / 'bundle')
    inventory = bundle.root / 'inventory.jsonl'
    entry = json.loads(inventory.read_text().splitlines()[0])
    entry['path'] = '../outside'
    inventory.write_text(json.dumps(entry) + '\n')
    manifest = bundle.manifest
    manifest['inventory_sha256'] = hashlib.sha256(
        inventory.read_bytes()).hexdigest()
    resign(bundle.root, manifest)
    with pytest.raises(EvidenceValidationError, match='Unsafe'):
        bundle.validate()


def test_graph_only_request_needs_explicit_supported_batch():
    trainer = object.__new__(Trainer)
    trainer._accelerator = 'cpu'
    with pytest.raises(ValueError, match='batch_size=1'):
        trainer.predict(evidence=EvidenceConfig(numerical_sensitivity=None))


def test_record_policy_does_not_raise_runtime_gate():
    trainer = object.__new__(Trainer)
    trainer._accelerator = 'gpu'
    # Missing snapshot is encountered after unsupported evidence is disabled.
    with pytest.raises(TypeError, match='snapshot'):
        trainer.predict(evidence=EvidenceConfig(on_unsupported='record'))


def test_replay_only_does_not_require_cpu_or_single_seed():
    trainer = object.__new__(Trainer)
    trainer._accelerator = 'gpu'
    with pytest.raises(TypeError, match='snapshot'):
        trainer.predict(evidence=EvidenceConfig(
            scoring_graph=False, numerical_sensitivity=None, replay=True))


def test_removal_cannot_omit_unsampled_inbound_schema():
    from kumo.engine.perturbation import RecordRemovalPlan, responses_from_trainer
    plan = RecordRemovalPlan(({
        'table': 'events',
        'pkey': 1
    }, ), ('events', ), (('events', 'entity_id', 'entities'), ))
    relations = [('events', 'entity_id', 'entities'),
                 ('entities', 'rev_entity_id', 'events'),
                 ('children', 'event_id', 'events'),
                 ('events', 'rev_event_id', 'children')]
    provenance = {
        'model_reconstruction': {
            'graph_runtime_metadata': {
                'edge_types': [{
                    'src': src,
                    'rel': rel,
                    'dst': dst
                } for src, rel, dst in relations]
            }
        }
    }
    with pytest.raises(ValueError, match='complete retained relation schema'):
        list(responses_from_trainer(None, {}, {}, provenance, plan))
