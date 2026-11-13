"""Installed-runtime collector RSS check, separate from the native model gate."""
import json
import resource

import kumo_core.modeling.context as context
import torch
from test_local_sensitivity import fixture


def test_ten_thousand_rows_keep_collector_memory_bounded(tmp_path):
    collector, module, batch, output, encoder = fixture(tmp_path)
    checkpoints = {}
    for index in range(10000):
        batch['events'].input_id = torch.tensor([index])
        collector.explain_pre_step(batch, module, None, None, index)
        with context.explaining():
            output['logits'] = encoder(
                batch['events'].tf.feat_dict['numerical'])
        collector.explain_post_step(batch, module, None, output, None, index)
        if index + 1 in (1000, 10000):
            checkpoints[str(index + 1)] = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss
    collector.explain_end(module, None, {}, None)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['rows'] == 10000
    # Linux ru_maxrss uses KiB; fixed tensors and flushed row writes should not
    # grow proportionally with the additional 9000 predictions.
    assert checkpoints['10000'] - checkpoints['1000'] < 32 * 1024
    print(
        json.dumps({
            'collector_only': True,
            'rss_peak_kib': checkpoints,
            'max_growth_kib': 32 * 1024
        }))
