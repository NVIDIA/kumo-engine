# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required CUDA reuse check. The child loads the installed wheel in isolation."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

RHS = 'rhs_embedding.lookup_embedding.'
REMAP_KEYS = tuple(RHS + name for name in ('weight', 'mask', '_extra_state'))
QUERY = ('PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
         'RANK TOP 3 FOR EACH ents.ID')


def hashes(root):
    return {
        str(path.relative_to(root)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*')) if path.is_file()
    }


def check_remap(saved, actual):
    """Old rows remain exact by raw key; only the new item's row is initialized."""
    import numpy as np
    import torch

    old, new = [state[RHS + '_extra_state'] for state in (saved, actual)]
    assert set(old) == set(new) == {'index_map'}
    old, new = old['index_map'], new['index_map']
    for mapping, count in ((old, 20), (new, 21)):
        assert mapping.index.name == '__ttng_rhs_primary_key__'
        assert list(mapping) == ['arange']
        assert mapping.index.is_unique and set(mapping.index) == set(
            range(count))
        assert np.issubdtype(mapping['arange'].dtype, np.integer)
        assert sorted(mapping['arange']) == list(range(count))
    before = saved[RHS + 'weight'].detach().cpu()
    after = actual[RHS + 'weight'].detach().cpu()
    assert before.shape[0] == 20 and after.shape == (21, before.shape[1])
    assert after.dtype == before.dtype
    # Loading remaps on CPU. Preserve the original storage order for its mean.
    expected = before.mean(dim=0).expand(21, -1).clone()
    mask = torch.ones(21, dtype=torch.bool)
    for key, physical_id in old['arange'].items():
        expected[int(new.loc[key, 'arange'])] = before[int(physical_id)]
        mask[int(new.loc[key, 'arange'])] = False
    assert torch.equal(after,
                       expected), 'RHS learned rows or new-item mean changed'
    actual_mask = actual[RHS + 'mask'].detach().cpu()
    assert actual_mask.dtype == mask.dtype and torch.equal(actual_mask, mask)
    assert saved[RHS + 'mask'].dtype == torch.bool
    assert not saved[RHS + 'mask'].any()


def check_installed_module(module):
    assert Path(module.__file__).resolve().is_relative_to(
        Path(sys.prefix).resolve())


def check_child_proof(proof):
    for name in ('fits', 'fit_attempts', 'state_checks', 'restored_tensors',
                 'rows'):
        assert type(proof[name]) is int
    assert proof['fits'] == proof['fit_attempts'] == 0
    assert proof['state_checks'] == 2 and proof['restored_tensors'] > 0
    batches = proof['prediction_batches']['reuse']
    assert type(batches) is int and batches > 0
    assert proof['rows'] == 51 * 21


def check_model_reuse(graph, tmp_path):
    import pandas as pd
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumoapi.jobs import MetadataField
    from wheel_gpu_smoke import make_plan, observe_cuda, prediction_values

    import kumo.engine as kumo

    assert torch.cuda.is_available(), 'This required test needs CUDA'
    original_graph, snapshot, root = graph
    plan = make_plan('temporal_link')
    plan.optimization.max_epochs = 1
    plan.optimization.min_steps_per_epoch = 2
    plan.optimization.max_steps_per_epoch = 2
    state = dict(stage='fit', fits=0, completed_fits=0, optimizer_steps=0,
                 training_batches=0, prediction_batches={})
    query = kumo.PredictiveQuery(graph=original_graph, query=QUERY)
    trainer = kumo.Trainer(model_plan=plan, accelerator='gpu')
    with observe_cuda(state):
        fitted = trainer.fit(query.generate_training_table(snapshot),
                             snapshot=snapshot)
    checkpoint = local_path(str(fitted.artifact_paths['selected_checkpoint']))
    assert load_kumo_checkpoint(str(checkpoint),
                                map_location='cpu')['global_step'] > 0
    assert state['fits'] == state['completed_fits'] == 1
    assert state['optimizer_steps'] > 0 and state['training_batches'] > 0
    # Reuse this fit for catalog saturation and both query-filter boundaries.
    events = pd.read_parquet(root / 'data/events.parquet')
    times, anchor = pd.to_datetime(events['TS']), pd.Timestamp('2024-03-01')
    lower = anchor - pd.Timedelta(days=7)
    assert all([(times == lower).any(), (times == anchor).any(),
                (times > anchor).any()])
    counts = events.loc[(times > lower)
                        & (times <= anchor)].groupby('ENT_ID').size()
    eligible = sorted(counts[counts > 2].index.tolist())
    assert eligible and set(eligible) < set(range(50))
    for wrong in ((times >= lower) & (times <= anchor),
                  (times > lower) & (times < anchor)):
        wrong_counts = events.loc[wrong].groupby('ENT_ID').size()
        assert set(eligible) != set(wrong_counts[wrong_counts > 2].index)
    state.update(stage='behavior', prediction_batches={'behavior': 0})
    with observe_cuda(state):
        for suffix, ids in [('', list(range(50))),
                            (' WHERE COUNT(events.*, -7, 0, days) > 2',
                             eligible)]:
            selected = kumo.PredictiveQuery(original_graph, QUERY + suffix)
            table = selected.generate_prediction_table(
                snapshot, prediction_time=anchor.to_pydatetime(),
                plan=trainer.training_table_generation_plan)
            staged = pd.read_parquet(table.table_path)
            assert staged['ENTITY1'].is_unique and set(
                staged['ENTITY1']) == set(ids)
            name = 'filtered' if suffix else 'catalog-cap'
            output = kumo.PredictionOutputConfig(
                kumo.FileConnector(root_dir=str(tmp_path / name)),
                output_paths={'predictions': 'rows'},
                extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
            frame = trainer.predict(table, snapshot=snapshot,
                                    num_classes_to_return=25,
                                    output_config=output).frame
            prediction_values('temporal_link', frame, ids,
                              classes=list(range(20)), anchor=anchor)
            staged.to_parquet(tmp_path / f'{name}-inputs.parquet', index=False)
            frame.to_parquet(tmp_path / f'{name}-predictions.parquet',
                             index=False)
    assert state['prediction_batches']['behavior'] > 0
    bundle = tmp_path / 'saved-model'
    shutil.copytree(fitted.bundle_dir, bundle)
    source = root / 'data'
    original = dict(data=hashes(source), bundle=hashes(bundle))
    request = dict(
        source=str(source), bundle=str(bundle), snapshot_id=snapshot.id,
        graph_id=original_graph.id,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    report = tmp_path / 'child-result.json'
    child = subprocess.run([
        sys.executable, '-I',
        str(Path(__file__).resolve()),
        str(request_path),
        str(report)
    ], capture_output=True, text=True, timeout=300)
    (tmp_path / 'child.stdout').write_text(child.stdout)
    (tmp_path / 'child.stderr').write_text(child.stderr)
    assert child.returncode == 0, child.stderr
    proof = json.loads(report.read_text())
    check_child_proof(proof)
    assert dict(data=hashes(source), bundle=hashes(bundle)) == original


def child(request_path, report_path):
    request = json.loads(request_path.read_text())
    import kumo_core
    import pandas as pd
    import torch
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumo_core.train.trainer import Trainer as CoreTrainer
    from kumoapi.graph import Edge
    from kumoapi.jobs import MetadataField
    from kumoapi.typing import Stype
    from pytorch_lightning import Callback
    from pytorch_lightning import Trainer as LightningTrainer
    from wheel_gpu_smoke import (
        observe_cuda,
        prediction_values,
        verify_checkpoint,
    )

    import kumo.engine as kumo

    assert sys.flags.isolated, 'The prediction process must use isolated Python'
    for module in (kumo, kumo_core):
        check_installed_module(module)
    source, bundle = Path(request['source']), Path(request['bundle'])
    original = dict(data=hashes(source), bundle=hashes(bundle))
    data = report_path.parent / 'new-data'
    shutil.copytree(source, data)
    for name, key, value in [('ents', 'ID', 50), ('items', 'IID', 20)]:
        path = data / f'{name}.parquet'
        frame = pd.read_parquet(path)
        pd.concat([frame, pd.DataFrame({key: [value]})],
                  ignore_index=True).to_parquet(path)
    frozen_inputs = hashes(data)
    kumo.configure(scratch_dir=report_path.parent / 'child-scratch')
    connector = kumo.FileConnector(root_dir=str(data))
    tables = {
        name: kumo.Table.from_source_table(connector[f'{name}.parquet'],
                                           primary_key=key).infer_metadata()
        for name, key in [('ents', 'ID'), ('items', 'IID')]
    }
    tables['events'] = kumo.Table.from_source_table(
        connector['events.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables=tables, edges=[
            Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
            Edge(src_table='events', fkey='ITEM_ID', dst_table='items')
        ])
    snapshot = graph.snapshot()
    assert graph.id != request['graph_id']
    assert snapshot.id != request['snapshot_id']
    checkpoint = bundle / 'model' / 'checkpoint'
    saved = load_kumo_checkpoint(str(checkpoint),
                                 map_location='cpu')['state_dict']
    state = dict(stage='reuse', fits=0, fit_attempts=0, state_checks=0,
                 restored_tensors=0, prediction_batches={'reuse': 0})

    class ExactState(Callback):
        def check(self, module):
            state['restored_tensors'] = verify_checkpoint(
                module, checkpoint, request['checkpoint_sha256'],
                exclude=REMAP_KEYS)
            check_remap(saved, module.state_dict())
            state['state_checks'] += 1

        def on_predict_start(self, trainer, module):
            self.check(module)

        def on_predict_end(self, trainer, module):
            self.check(module)

    def forbidden_fit(*args, **kwargs):
        state['fit_attempts'] += 1
        raise AssertionError('The fresh prediction process attempted to fit')

    with ExitStack() as stack:
        for cls in (kumo.Trainer, CoreTrainer, LightningTrainer):
            stack.enter_context(patch.object(cls, 'fit', forbidden_fit))
        stack.enter_context(observe_cuda(state))
        create = CoreTrainer.create_pl_trainer

        def create_with_state_checks(trainer):
            result = create(trainer)
            trainer.pl_trainer.callbacks.append(ExactState())
            return result

        stack.enter_context(
            patch.object(CoreTrainer, 'create_pl_trainer',
                         create_with_state_checks))
        trainer = kumo.Trainer.load(bundle, accelerator='gpu')
        query = kumo.PredictiveQuery(graph=graph, query=QUERY)
        table = query.generate_prediction_table(snapshot)
        output = kumo.PredictionOutputConfig(
            connector=kumo.FileConnector(root_dir=str(report_path.parent /
                                                      'output')),
            output_paths={'predictions':
                          'all'}, extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
        frame = trainer.predict(table, snapshot=snapshot,
                                num_classes_to_return=21,
                                output_config=output).frame
    values = prediction_values(
        'temporal_link', frame, list(range(51)), classes=list(range(21)),
        anchor=pd.Timestamp(table.timestamps[0], unit='ns'))
    assert values.shape == (51, 21) and torch.cuda.is_available()
    assert state['fit_attempts'] == state['fits'] == 0
    assert state[
        'state_checks'] == 2 and state['prediction_batches']['reuse'] > 0
    assert dict(data=hashes(source), bundle=hashes(bundle)) == original
    assert hashes(data) == frozen_inputs
    state['rows'] = len(frame)
    state['engine_module'] = str(Path(kumo.__file__).resolve())
    report_path.write_text(json.dumps(state, indent=2))


BEHAVIOR = (
    'test_readme_quickstart.py',
    'test_warm_start.py',
    'test_training_table_export.py',
    'test_static_link_split.py',
    'test_static_link_pseudo_time.py::test_static_link_subgraphs_hide_later_split_labels',
    'test_static_link_pseudo_time.py::test_native_iterator_infers_cutoff_without_loader_options',
    'test_static_link_pseudo_time.py::test_stage_visibility_overrides_generic_runtime_options',
    'test_static_query_time_cutoff.py',
    'test_timestamp_batch_cap.py',
    'test_prediction_workers.py::test_bad_worker_count_fails_at_public_entry',
    'test_prediction_workers.py::test_bad_class_count_fails_before_prediction_work',
    'test_prediction_output_columns.py::test_link_prediction_respects_explicit_count',
    'test_nightly_model_reuse.py',
)


def check_behavior(result):
    assert result['collected'] and len(set(result['collected'])) == len(
        result['collected'])
    assert set(result['calls']) == set(result['collected'])
    assert set(result['calls'].values()) == {'passed'}
    assert not result['bad'] and not result['deselected'] and not result[
        'collection_errors']
    ledger = result['fit_ledger']
    assert result['fits'] == result['completed_fits'] == len(ledger) > 0
    assert all(
        row['status'] == 'PASS' and type(row['steps']) is int and row['steps'] >
        0 and type(row['cuda_batches']) is int and row['cuda_batches'] > 0
        for row in ledger)


def pin_behavior_artifacts(output, result):
    [path] = output.glob('pytest-temp/**/child-result.json')
    check_child_proof(json.loads(path.read_text()))
    assert (output / 'junit.xml').is_file()
    assert all((path.parent / name).is_file()
               for name in ('child.stdout', 'child.stderr', 'request.json'))
    artifacts = hashes(output)
    artifacts.pop('result.json', None)
    relative = str(path.relative_to(output))
    result.update(artifacts=artifacts,
                  b7_receipt=dict(path=relative, sha256=artifacts[relative]))


def behavior(output):
    sys.path.append(
        str(
            Path(os.environ['ENGINE_NIGHTLY_PYTEST_TOOLS']).resolve(
                strict=True)))
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    os.environ.pop('PYTEST_ADDOPTS', None)
    import kumo_core
    import pytest
    from wheel_gpu_smoke import observe_cuda

    import kumo.engine as kumo

    for module in (kumo, kumo_core):
        check_installed_module(module)
    directory = Path(__file__).resolve().parent
    output = output.resolve()
    output.mkdir(parents=True)
    os.chdir(output)
    result = dict(status='FAIL', stage='behavior', fits=0, completed_fits=0,
                  optimizer_steps=0, training_batches=0,
                  prediction_batches={'behavior':
                                      0}, fit_ledger=[], collected=[], calls={},
                  bad=[], deselected=0, collection_errors=[])

    class Required:
        def pytest_collection_finish(self, session):
            result['collected'] = [item.nodeid for item in session.items]

        def pytest_deselected(self, items):
            result['deselected'] += len(items)

        def pytest_collectreport(self, report):
            if report.failed or report.skipped:
                result['collection_errors'].append(report.nodeid)

        def pytest_runtest_logreport(self, report):
            if report.failed or report.skipped:
                result['bad'].append(
                    f'{report.nodeid}:{report.when}:{report.outcome}')
            if report.when == 'call':
                assert report.nodeid not in result['calls']
                result['calls'][report.nodeid] = report.outcome

    try:
        with observe_cuda(result):
            result['pytest_exit_code'] = int(
                pytest.main([
                    '-q',
                    '-o',
                    'addopts=',
                    f'--confcutdir={directory}',
                    f'--basetemp={output / "pytest-temp"}',
                    f'--junitxml={output / "junit.xml"}',
                    *[str(directory / item) for item in BEHAVIOR],
                ], plugins=[Required()]))
        assert result['pytest_exit_code'] == 0
        check_behavior(result)
        pin_behavior_artifacts(output, result)
        result.update(
            status='PASS', physical_fits=result['fits'], coverage={
                name: True
                for name in ('I2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8')
            })
    finally:
        (output / 'result.json').write_text(json.dumps(result, indent=2))


if __name__ != '__main__':
    import pytest

    @pytest.fixture(scope='module')
    def graph(tmp_path_factory):
        from test_prediction_output_columns import graph as fixture
        yield from fixture.__wrapped__(tmp_path_factory)

    def test_saved_model_predicts_new_graph_without_refitting(graph, tmp_path):
        check_model_reuse(graph, tmp_path)

else:
    # Only test helpers enter sys.path; product modules come from the installed wheel.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if sys.argv[1] == '--behavior':
        behavior(Path(sys.argv[2]))
    else:
        child(Path(sys.argv[1]), Path(sys.argv[2]))
