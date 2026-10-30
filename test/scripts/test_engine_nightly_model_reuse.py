# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the model-reuse oracle without fitting models or requiring CUDA."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


def worker():
    path = (Path(__file__).resolve().parents[2] /
            'packages/kumo-engine/test/test_nightly_model_reuse.py')
    spec = importlib.util.spec_from_file_location('nightly_model_reuse', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def states():
    prefix = 'rhs_embedding.lookup_embedding.'
    old_map = pd.DataFrame({'arange': range(20)},
                           index=pd.Index(range(20),
                                          name='__ttng_rhs_primary_key__'))
    old_weight = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    old = {
        prefix + 'weight': old_weight,
        prefix + 'mask': torch.zeros(20, dtype=torch.bool),
        prefix + '_extra_state': {
            'index_map': old_map
        }
    }
    # Change physical order so positional comparisons cannot pass by accident.
    new_map = pd.DataFrame({'arange': range(21)},
                           index=pd.Index([20, *reversed(range(20))],
                                          name=old_map.index.name))
    new = {
        prefix + 'weight':
        torch.cat([old_weight.mean(dim=0, keepdim=True),
                   old_weight.flip(0)]),
        prefix + 'mask':
        torch.tensor([True] + [False] * 20),
        prefix + '_extra_state': {
            'index_map': new_map
        }
    }
    return old, new


def test_reuse_accepts_exact_raw_key_remap_and_new_item_mean():
    old, new = states()
    worker().check_remap(old, new)


def test_isolated_child_reaches_request_without_importing_pytest(tmp_path):
    path = Path(worker().__file__)
    missing = tmp_path / 'missing-request.json'
    code = '''
import importlib.abc, runpy, sys
class BlockPytest(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pytest' or fullname.startswith('pytest.'):
            raise AssertionError('The isolated child imported pytest')
sys.meta_path.insert(0, BlockPytest())
worker, request, output = sys.argv[1:]
sys.argv = [worker, request, output]
try:
    runpy.run_path(worker, run_name='__main__')
except FileNotFoundError as error:
    assert error.filename == request, error
    print('CHILD_REQUEST_REACHED')
else:
    raise AssertionError('The child did not read its request')
'''
    result = subprocess.run([
        sys.executable, '-I', '-c', code,
        str(path),
        str(missing),
        str(tmp_path / 'unused.json')
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'CHILD_REQUEST_REACHED'


def test_reuse_rejects_product_imports_outside_runtime_prefix(
        tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'runtime'))
    check = worker().check_installed_module
    check(SimpleNamespace(__file__=tmp_path / 'runtime/lib/kumo/__init__.py'))
    with pytest.raises(AssertionError):
        check(SimpleNamespace(__file__=tmp_path / 'other/kumo/__init__.py'))


@pytest.mark.parametrize('change', [
    None,
    'missing_call',
    'skipped',
    'deselected',
    'collection_error',
    'empty_inventory',
    'incomplete_fit',
    'zero_steps',
    'zero_cuda',
])
def test_behavior_receipt_requires_every_test_and_every_cuda_fit(change):
    proof = dict(collected=['a', 'b'], calls={
        'a': 'passed',
        'b': 'passed'
    }, bad=[], deselected=0, collection_errors=[], fits=1, completed_fits=1,
                 fit_ledger=[dict(status='PASS', steps=2, cuda_batches=2)])
    if change == 'missing_call':
        proof['calls'].pop('b')
    elif change == 'skipped':
        proof['bad'].append('b:setup:skipped')
    elif change == 'deselected':
        proof['deselected'] = 1
    elif change == 'collection_error':
        proof['collection_errors'].append('test.py')
    elif change == 'empty_inventory':
        proof['collected'] = []
    elif change == 'incomplete_fit':
        proof['completed_fits'] = 0
    elif change == 'zero_steps':
        proof['fit_ledger'][0]['steps'] = 0
    elif change == 'zero_cuda':
        proof['fit_ledger'][0]['cuda_batches'] = 0
    if change is None:
        worker().check_behavior(proof)
    else:
        with pytest.raises(AssertionError):
            worker().check_behavior(proof)


@pytest.mark.parametrize('change', [
    None,
    'missing_child',
    'duplicate_child',
    'missing_log',
    'hidden_fit',
    'missing_state',
    'no_cuda',
    'incomplete_rows',
])
def test_behavior_pins_original_child_and_retained_files(tmp_path, change):
    proof = dict(fits=0, fit_attempts=0, state_checks=2, restored_tensors=12,
                 prediction_batches={'reuse': 2}, rows=1071)
    if change == 'hidden_fit':
        proof['fit_attempts'] = 1
    elif change == 'missing_state':
        proof['state_checks'] = 1
    elif change == 'no_cuda':
        proof['prediction_batches']['reuse'] = 0
    elif change == 'incomplete_rows':
        proof['rows'] = 1070
    directory = tmp_path / 'pytest-temp/reuse0'
    directory.mkdir(parents=True)
    child = directory / 'child-result.json'
    child.write_text(json.dumps(proof))
    for name in ('child.stdout', 'child.stderr', 'request.json'):
        (directory / name).write_text('retained')
    (tmp_path / 'junit.xml').write_text('<testsuites/>')
    (tmp_path / 'result.json').write_text('not part of its own artifact map')
    if change == 'missing_child':
        child.unlink()
    elif change == 'duplicate_child':
        other = directory.parent / 'other'
        other.mkdir()
        (other / child.name).write_bytes(child.read_bytes())
    elif change == 'missing_log':
        (directory / 'child.stderr').unlink()
    result = {}
    if change is None:
        worker().pin_behavior_artifacts(tmp_path, result)
        assert result['b7_receipt']['path'] == str(child.relative_to(tmp_path))
        assert result['b7_receipt']['sha256'] == result['artifacts'][str(
            child.relative_to(tmp_path))]
        assert 'pytest-temp/reuse0/child.stderr' in result['artifacts']
        assert 'junit.xml' in result['artifacts']
        assert 'result.json' not in result['artifacts']
    else:
        with pytest.raises((ValueError, AssertionError)):
            worker().pin_behavior_artifacts(tmp_path, result)


@pytest.mark.parametrize('change', [
    'old_weight',
    'new_weight',
    'mask',
    'duplicate_key',
    'foreign_key',
    'duplicate_id',
    'dtype',
    'extra_state',
    'index_name',
    'missing_item',
])
def test_reuse_rejects_changed_weights_or_invalid_catalogue(change):
    old, actual = states()
    actual = copy.deepcopy(actual)
    prefix = 'rhs_embedding.lookup_embedding.'
    mapping = actual[prefix + '_extra_state']['index_map']
    if change == 'old_weight':
        actual[prefix + 'weight'][1, 0] += 1
    elif change == 'new_weight':
        actual[prefix + 'weight'][0, 0] += 1
    elif change == 'mask':
        actual[prefix + 'mask'][0] = False
    elif change == 'duplicate_key':
        mapping.index = [0] * 21
    elif change == 'foreign_key':
        mapping.index = pd.Index([99, *range(20)], name=mapping.index.name)
    elif change == 'duplicate_id':
        mapping['arange'] = 0
    elif change == 'dtype':
        actual[prefix + 'weight'] = actual[prefix + 'weight'].double()
    elif change == 'extra_state':
        actual[prefix + '_extra_state']['unexpected'] = True
    elif change == 'index_name':
        mapping.index.name = None
    else:
        actual[prefix + '_extra_state']['index_map'] = mapping.iloc[1:]
    with pytest.raises((ValueError, AssertionError)):
        worker().check_remap(old, actual)
