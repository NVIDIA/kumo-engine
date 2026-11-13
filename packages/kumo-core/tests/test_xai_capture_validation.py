# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent acceptance checks for prediction capture identities/failures."""
import json

import pytest
import torch
from test_local_sensitivity import fixture


def test_repeated_entity_cutoffs_and_input_occurrences_are_distinct(tmp_path):
    identifiers = []
    for index, (cutoff, task_id) in enumerate([(10, 19), (11, 19), (11, 20)]):
        path = tmp_path / str(index)
        collector, module, batch, output, _ = fixture(path)
        output['time'] = torch.tensor([cutoff])
        batch['events'].input_id = torch.tensor([task_id])
        collector.explain_post_step(batch, module, None, output, None, 0)
        identifiers.append(output['prediction_id'][0])
        record = json.loads((path / 'index.jsonl').read_text())
        assert record['time_ns'] == [cutoff]
        assert record['task_row_id'] == [task_id]
    assert len(set(identifiers)) == 3


def test_graph_failure_preserves_gradient_and_marks_partial(
        tmp_path, monkeypatch):
    collector, module, batch, output, _ = fixture(tmp_path)

    def fail(*args):
        raise ValueError('private record details')

    monkeypatch.setattr(collector, '_sample', fail)
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    record = json.loads((tmp_path / 'index.jsonl').read_text())
    assert record['status'] == 'available'
    assert record['graph_status'] == 'failed'
    assert record['sample_sha256'] is None
    assert 'private record' not in str(record)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['generation_closed'] and not manifest['complete']
    assert manifest['graph_failures'] == 1


def test_closed_empty_generation_does_not_fabricate_rows(tmp_path):
    collector, module, _, _, _ = fixture(tmp_path)
    collector.explain_end(module, None, {}, None)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['attempted_rows'] == manifest['rows'] == 0
    assert (tmp_path / 'index.jsonl').read_text() == ''


@pytest.mark.parametrize(('source_time', 'status'), [(99, 'valid'),
                                                     (100, 'violated'),
                                                     (101, 'violated'),
                                                     (None, 'unavailable')])
def test_temporal_audit_preserves_original_graph(tmp_path, source_time, status):
    collector, module, batch, output, _ = fixture(tmp_path)
    collector.provenance['model_reconstruction'] = {
        'metadata': {
            'is_temporal': True
        }
    }
    collector.provenance['temporal_boundary']['source_time_columns'] = {
        'events': 'event_time'
    }
    if source_time is not None:
        batch['events'].time = torch.tensor([source_time])
    output['time'] = torch.tensor([100])
    collector.explain_post_step(batch, module, None, output, None, 0)
    collector.explain_end(module, None, {}, None)
    row = json.loads((tmp_path / 'index.jsonl').read_text())
    assert row['temporal_validation']['status'] == status
    retained = json.loads(
        (tmp_path / 'samples' / (row['prediction_id'] + '.json')).read_text())
    assert retained['nodes']['events']['times_ns'] == (
        None if source_time is None else [source_time])
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['complete'] == (status == 'valid')
