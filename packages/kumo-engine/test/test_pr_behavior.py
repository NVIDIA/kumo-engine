# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR contracts reuse the existing 50-entity link fixture and its model."""

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def eligible_ids(events, anchor, threshold):
    import pandas as pd

    times = pd.to_datetime(events['TS'])
    window = (times > anchor - pd.Timedelta(days=7)) & (times <= anchor)
    counts = events.loc[window].groupby('ENT_ID').size()
    return sorted(counts[counts > threshold].index.tolist())


def verify_recommendations(frame, ids, catalog, count):
    import numpy as np

    expected = min(count, len(catalog))
    if (frame.empty or frame[['ENTITY', 'CLASS']].isna().any().any()
            or frame.duplicated(['ENTITY', 'CLASS']).any()
            or set(frame['ENTITY']) != set(ids)
            or not set(frame['CLASS']) <= set(catalog)
            or not frame.groupby('ENTITY').size().eq(expected).all()
            or not np.isfinite(frame['SCORE'].to_numpy(dtype=float)).all()):
        raise ValueError('Incomplete, repeated, invalid or nonfinite '
                         'recommendations')
    return dict(entities=len(ids), catalog=len(catalog), requested=count,
                returned_per_entity=expected, rows=len(frame))


def _hash_files(root):
    return {
        str(path.relative_to(root)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()
    }


def verify_remapped_state(module, expected, catalog, new_key):
    import torch
    from kumo_core.train.mapping import TTNG_RHS_PRIMARY_KEY_INDEX_NAME

    verify_extra = runpy.run_path(
        str(Path(__file__).with_name(
            'wheel_gpu_learning.py')))['verify_checkpoint_extra_state']
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    keys = [
        key for key in expected
        if 'rhs_embedding.' in key and key.endswith('lookup_embedding.weight')
    ]
    assert len(keys) == 1, 'A real learned RHS lookup is required'
    weight_key = keys[0]
    prefix = weight_key.removesuffix('weight')
    extra_key, mask_key = prefix + '_extra_state', prefix + 'mask'
    old_map, new_map = (state[extra_key]['index_map']
                        for state in (expected, actual))
    for mapping in (old_map, new_map):
        assert mapping.index.name == TTNG_RHS_PRIMARY_KEY_INDEX_NAME
        assert mapping.index.is_unique and not mapping.index.hasnans
        assert list(mapping.columns) == ['arange'
                                         ] and mapping['arange'].is_unique
    assert set(new_map.index) == set(catalog) == set(old_map.index) | {new_key}
    assert new_key not in old_map.index
    weight = actual[weight_key]
    original = expected[weight_key]
    assert original.dtype == weight.dtype and original.shape[
        1:] == weight.shape[1:]
    old_ids = torch.tensor(old_map['arange'].tolist(), device=original.device)
    new_ids = torch.tensor(new_map.loc[old_map.index, 'arange'].tolist(),
                           device=weight.device)
    assert torch.equal(weight[new_ids].detach().cpu(),
                       original[old_ids].cpu()), 'Old RHS weights changed'
    # Kumo restores CPU storage before Lightning places the model on its device.
    values = original if sorted(old_ids.tolist()) == list(range(
        len(original))) else original[old_ids]
    mean = values.mean(dim=0)
    assert torch.equal(
        weight[int(new_map.loc[new_key, 'arange'])].detach().cpu(),
        mean.cpu()), 'New RHS weight is not the existing mean initialization'
    mask = torch.ones(weight.shape[0], device=weight.device, dtype=torch.bool)
    mask[new_ids] = False
    assert torch.equal(actual[mask_key],
                       mask), 'New-key or sparse lookup mask changed'
    for key in expected.keys() - {weight_key, mask_key, extra_key}:
        left, right = expected[key], actual[key]
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
            assert left.shape == right.shape and torch.equal(
                left.cpu(),
                right.detach().cpu()), key
        else:
            verify_extra(left, right)
    lookup = module.rhs_embedding.lookup_embedding
    return dict(
        module=f'{type(module).__module__}.{type(module).__name__}',
        lookup=f'{type(lookup).__module__}.{type(lookup).__name__}',
        lookup_mode=module.rhs_embedding.emb_mode.value,
        device=str(weight.device), old_entities_preserved=len(old_map),
        new_entities=1, reindexed_entities=int(
            (old_map['arange'] != new_map.loc[old_map.index, 'arange']).sum()),
        checkpoint_state_entries=len(expected),
        remapped_state_entries=[weight_key, mask_key, extra_key])


def _link_fixture(tmp_path_factory):
    helpers = runpy.run_path(
        str(Path(__file__).with_name('test_prediction_output_columns.py')))
    # Reuse its unchanged generated data and fitted public API recipe.
    graph_fixture = helpers['graph'].__wrapped__(tmp_path_factory)
    graph = next(graph_fixture)
    try:
        query = ('PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
                 'RANK TOP 3 FOR EACH ents.ID')
        yield graph, helpers['_fit'](graph, query)
    finally:
        next(graph_fixture, None)


def test_link_counts_are_complete_unique_and_catalog_bounded(pr_link, tmp_path):
    import pandas as pd

    (_, snapshot, root), (trainer, _, table) = pr_link
    ids = sorted(pd.read_parquet(root / 'data/ents.parquet')['ID'])
    catalog = sorted(pd.read_parquet(root / 'data/items.parquet')['IID'])
    assert len(ids) == 50 and len(catalog) == 20
    reports = []
    for count in (2, 5, 25):
        frame = trainer.predict(table, snapshot=snapshot,
                                num_classes_to_return=count).frame
        reports.append(verify_recommendations(frame, ids, catalog, count))
        frame.to_parquet(tmp_path / f'count-{count}.parquet', index=False)
    (tmp_path / 'counts.json').write_text(json.dumps(reports, indent=2))


def test_prediction_filter_obeys_both_time_boundaries(pr_link, tmp_path):
    import pandas as pd

    import kumo.engine as kumo

    (graph, snapshot, root), (trainer, query, _) = pr_link
    events = pd.read_parquet(root / 'data/events.parquet')
    anchor = pd.Timestamp('2024-03-01')
    ids = eligible_ids(events, anchor, 2)
    all_ids = set(pd.read_parquet(root / 'data/ents.parquet')['ID'])
    assert ids and set(ids) < all_ids
    times = pd.to_datetime(events['TS'])
    boundaries = dict(lower=int((times == anchor - pd.Timedelta(days=7)).sum()),
                      upper=int((times == anchor).sum()), future=int(
                          (times > anchor).sum()))
    assert all(boundaries.values())
    filtered = kumo.PredictiveQuery(
        graph, query.query + ' WHERE COUNT(events.*, -7, 0, days) > 2')
    table = filtered.generate_prediction_table(
        snapshot, prediction_time=anchor.to_pydatetime(),
        plan=trainer.training_table_generation_plan)
    staged = pd.read_parquet(table.table_path)
    assert staged['ENTITY1'].is_unique
    assert set(staged['ENTITY1']) == set(ids)
    frame = trainer.predict(table, snapshot=snapshot,
                            num_classes_to_return=3).frame
    catalog = pd.read_parquet(root / 'data/items.parquet')['IID'].tolist()
    report = verify_recommendations(frame, ids, catalog, 3)
    staged.to_parquet(tmp_path / 'filtered-inputs.parquet', index=False)
    frame.to_parquet(tmp_path / 'filtered-predictions.parquet', index=False)
    (tmp_path / 'filter.json').write_text(
        json.dumps(dict(**report, boundaries=boundaries, expected_ids=ids),
                   indent=2))


def test_retained_tables_and_model_predict_in_fresh_process(pr_link, tmp_path):
    from importlib import metadata

    (_, _, root), (trainer, query, _) = pr_link
    source = dict(data=_hash_files(root / 'data'),
                  bundle=_hash_files(trainer._training_result.bundle_dir))
    data = tmp_path / 'retained-data'
    bundle = tmp_path / 'retained-model'
    shutil.copytree(root / 'data', data)
    shutil.copytree(trainer._training_result.bundle_dir, bundle)
    original = dict(data=_hash_files(data), bundle=_hash_files(bundle))
    request = tmp_path / 'request.json'
    request.write_text(
        json.dumps(
            dict(data=str(data), bundle=str(bundle),
                 output=str(tmp_path / 'child'),
                 version=metadata.version('kumo-engine'), query=query.query,
                 accelerator=trainer._accelerator)))
    completed = subprocess.run(
        [sys.executable, '-I',
         str(Path(__file__).resolve()),
         str(request)], capture_output=True, text=True)
    (tmp_path / 'child.log').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / 'child/result.json').read_text())
    assert result['status'] == 'PASS' and result['fits_started'] == 0
    assert result['requested_accelerator'] == trainer._accelerator
    assert result['prediction_batches'] > 0
    assert result['snapshot_id'] != result['training_snapshot_id']
    assert result['entities'] == 51 and result['catalog'] == 21
    assert len(result['state_checks']) == 2
    assert all(
        check['old_entities_preserved'] == 20 and check['new_entities'] == 1
        for check in result['state_checks'])
    if trainer._accelerator == 'gpu':
        assert result['cuda_prediction_batches'] == result['prediction_batches']
        assert set(result['observed_devices']) == {'cuda:0'}
    assert _hash_files(data) == original['data']
    assert _hash_files(bundle) == original['bundle']
    assert _hash_files(root / 'data') == source['data']
    assert _hash_files(trainer._training_result.bundle_dir) == source['bundle']


def _predict_retained(request):
    from importlib import metadata

    import pandas as pd
    import torch
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumo_core.train.trainer import Trainer as CoreTrainer
    from kumoapi.graph import Edge
    from kumoapi.jobs import MetadataField
    from kumoapi.typing import Stype
    from pytorch_lightning.callbacks import Callback

    import kumo.engine as kumo
    from kumo.engine.trainer._spec import read_bundle_artifact_paths

    options = json.loads(request.read_text())
    output = Path(options['output'])
    output.mkdir()
    if (metadata.version('kumo-engine') != options['version']
            or kumo.__version__ != options['version'] or
            not Path(kumo.__file__).resolve().is_relative_to(Path(sys.prefix))):
        raise ValueError('Require the selected installed wheel')
    accelerator = options['accelerator']
    expected_device = {
        'cpu': 'cpu',
        'gpu': 'cuda',
        'mps': 'mps',
        'auto': None
    }[accelerator]
    if expected_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable for requested GPU')
    result = dict(status='RUNNING', fits_started=0, prediction_batches=0,
                  cuda_prediction_batches=0, requested_accelerator=accelerator,
                  observed_devices=[], state_checks=[],
                  installed_import=kumo.__file__, version=kumo.__version__)

    def forbidden_fit(*args, **kwargs):
        result['fits_started'] += 1
        raise AssertionError('The retained-model check must not train')

    class Proof(Callback):
        def on_predict_start(self, trainer, module):
            result['state_checks'].append(
                dict(
                    phase='post_restore',
                    **verify_remapped_state(module, expected_state, catalog,
                                            new_item)))

        def on_predict_end(self, trainer, module):
            result['state_checks'].append(
                dict(
                    phase='post_prediction',
                    **verify_remapped_state(module, expected_state, catalog,
                                            new_item)))

        def on_predict_batch_end(self, trainer, module, values, batch,
                                 batch_idx, dataloader_idx=0):
            logits = (values.get('logits', values.get('score')) if isinstance(
                values, dict) else None)
            tensors = (list(logits.values())
                       if isinstance(logits, dict) else [logits])
            device = trainer.strategy.root_device
            if ((expected_device is not None and device.type != expected_device)
                    or device.type not in ('cpu', 'cuda', 'mps') or not tensors
                    or any(not isinstance(value, torch.Tensor)
                           or not value.numel() or value.device != device
                           for value in tensors)):
                raise AssertionError(
                    'Missing prediction work on selected device')
            if device.type == 'cuda':
                torch.cuda.synchronize()
                result['cuda_prediction_batches'] += 1
            result['prediction_batches'] += 1
            result['observed_devices'].append(str(device))

    original_create = CoreTrainer.create_pl_trainer

    def create(trainer):
        original_create(trainer)
        trainer.pl_trainer.callbacks.append(Proof())

    CoreTrainer.fit = forbidden_fit
    kumo.Trainer.fit = forbidden_fit
    CoreTrainer.create_pl_trainer = create
    bundle = Path(options['bundle'])
    source_data = Path(options['data'])
    original = dict(data=_hash_files(source_data), bundle=_hash_files(bundle))
    spec = json.loads((bundle / 'training-spec.json').read_text())
    assert spec['predictive_query'] == options['query']
    checkpoint = Path(read_bundle_artifact_paths(bundle)['selected_checkpoint'])
    expected_state = load_kumo_checkpoint(str(checkpoint),
                                          map_location='cpu')['state_dict']
    result['checkpoint_sha256'] = hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    data = output / 'data'
    shutil.copytree(source_data, data)
    entities_frame = pd.read_parquet(data / 'ents.parquet')
    items_frame = pd.read_parquet(data / 'items.parquet')
    assert len(entities_frame) == 50 and len(items_frame) == 20
    new_entity, new_item = int(entities_frame.ID.max()) + 1, int(
        items_frame.IID.max()) + 1
    pd.concat(
        [entities_frame, pd.DataFrame({'ID': [new_entity]})],
        ignore_index=True).to_parquet(data / 'ents.parquet', index=False)
    pd.concat([items_frame, pd.DataFrame({'IID': [new_item]})],
              ignore_index=True).to_parquet(data / 'items.parquet', index=False)
    kumo.configure(scratch_dir=output / 'scratch')
    connector = kumo.FileConnector(root_dir=str(data))
    entities = kumo.Table.from_source_table(connector['ents.parquet'],
                                            primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(connector['items.parquet'],
                                         primary_key='IID').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    graph = kumo.Graph(
        dict(ents=entities, items=items, events=events),
        [Edge('events', 'ENT_ID', 'ents'),
         Edge('events', 'ITEM_ID', 'items')])
    snapshot = graph.snapshot()
    assert snapshot.id != spec['snapshot_id']
    model = kumo.Trainer.load(bundle, accelerator=accelerator)
    query = kumo.PredictiveQuery(graph, options['query'])
    anchor = pd.Timestamp('2024-03-01')
    table = query.generate_prediction_table(
        snapshot, prediction_time=anchor.to_pydatetime(),
        plan=model.training_table_generation_plan)
    ids = pd.read_parquet(data / 'ents.parquet')['ID'].tolist()
    catalog = pd.read_parquet(data / 'items.parquet')['IID'].tolist()
    staged = pd.read_parquet(table.table_path)
    assert staged.ENTITY1.is_unique and set(staged.ENTITY1) == set(ids)
    assert (pd.to_datetime(staged.TIMESTAMP) == anchor).all()
    staged.to_parquet(output / 'prediction-inputs.parquet', index=False)
    config = kumo.PredictionOutputConfig(
        kumo.FileConnector(root_dir=str(output / 'exports')),
        output_paths={'predictions': 'predictions'},
        extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
    frame = model.predict(table, snapshot=snapshot, output_config=config,
                          num_classes_to_return=len(catalog)).frame
    result.update(verify_recommendations(frame, ids, catalog, len(catalog)))
    assert (pd.to_datetime(frame.ANCHOR_TIMESTAMP) == anchor).all()
    if not result['prediction_batches']:
        raise AssertionError('No prediction batches completed')
    assert len(result['state_checks']) == 2
    assert _hash_files(source_data) == original['data']
    assert _hash_files(bundle) == original['bundle']
    frame.to_parquet(output / 'predictions.parquet', index=False)
    result.update(status='PASS', snapshot_id=snapshot.id,
                  training_snapshot_id=spec['snapshot_id'],
                  new_entity=new_entity, new_item=new_item, anchor=str(anchor),
                  source_unchanged=True, bundle_unchanged=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2))


def test_window_helper_excludes_lower_boundary_and_future_rows():
    import pandas as pd

    events = pd.DataFrame({
        'ENT_ID': [0, 0, 1, 1, 2, 2],
        'TS':
        pd.to_datetime([
            '2024-02-24', '2024-03-01', '2024-03-02', '2024-02-25',
            '2024-02-23', '2024-02-25'
        ]),
    })
    assert eligible_ids(events, pd.Timestamp('2024-03-01'), 1) == [0]


def test_recommendation_helper_rejects_missing_repeated_and_invalid_rows():
    import pandas as pd
    import pytest

    frame = pd.DataFrame({
        'ENTITY': [0, 0, 1, 1],
        'CLASS': [10, 11, 10, 11],
        'SCORE': [.1, .2, .3, .4],
    })
    assert verify_recommendations(frame, [0, 1], [10, 11], 5)['rows'] == 4
    for bad in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                frame.assign(SCORE=float('nan')),
                frame.assign(CLASS=[10, 12, 10, 11]),
                frame.assign(ENTITY=[0, 0, 2, 2])):
        with pytest.raises(ValueError):
            verify_recommendations(bad, [0, 1], [10, 11], 5)


def test_retained_gpu_request_rejects_unavailable_cuda(tmp_path, monkeypatch):
    from importlib import metadata

    import pytest
    import torch

    request = tmp_path / 'request.json'
    request.write_text(
        json.dumps(
            dict(output=str(tmp_path / 'child'),
                 version=metadata.version('kumo-engine'), accelerator='gpu')))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError,
                       match='CUDA is unavailable for requested GPU'):
        _predict_retained(request)


def test_backfill_helper_requires_old_weights_new_mean_and_other_parameters():
    import copy

    import pandas as pd
    import pytest
    import torch
    from kumo_core.modeling.retrieval.rhs_embedding import RHSEmbedding
    from kumo_core.train.mapping import TTNG_RHS_PRIMARY_KEY_INDEX_NAME

    # Real lookup modules exercise the checker without training a model.
    module = torch.nn.Module()
    index = pd.Index([10, 20, 30], name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME)
    module.rhs_embedding = RHSEmbedding(
        'lookup', 5, 2, index_map=pd.DataFrame({'arange': [0, 2, 4]},
                                               index=index))
    module.other = torch.nn.Linear(2, 1)
    with torch.no_grad():
        module.rhs_embedding.lookup_embedding.weight.copy_(
            torch.arange(10).reshape(5, 2))
    expected = copy.deepcopy(module.state_dict())
    new_map = pd.DataFrame({'arange': [2, 0, 6, 1]},
                           index=pd.Index([10, 20, 30, 40],
                                          name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME))
    module.rhs_embedding.num_nodes = 7
    module.rhs_embedding.order_by(new_map)
    report = verify_remapped_state(module, expected, [10, 20, 30, 40], 40)
    assert report['old_entities_preserved'] == 3 and report['new_entities'] == 1
    assert report['reindexed_entities'] == 3
    for name, index in [('rhs_embedding.lookup_embedding.weight', (2, 0)),
                        ('rhs_embedding.lookup_embedding.weight', (1, 0)),
                        ('other.weight', (0, 0))]:
        changed = copy.deepcopy(module)
        with torch.no_grad():
            dict(changed.named_parameters())[name][index] += 1
        with pytest.raises(AssertionError):
            verify_remapped_state(changed, expected, [10, 20, 30, 40], 40)
    assert torch.equal(expected['rhs_embedding.lookup_embedding.weight'],
                       torch.arange(10).reshape(5, 2))


def test_backfill_helper_does_not_reorder_dense_mean_reduction():
    import copy

    import pandas as pd
    import torch
    from kumo_core.modeling.retrieval.rhs_embedding import RHSEmbedding
    from kumo_core.train.mapping import TTNG_RHS_PRIMARY_KEY_INDEX_NAME

    module = torch.nn.Module()
    index = pd.Index([10, 20, 30], name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME)
    module.rhs_embedding = RHSEmbedding(
        'lookup', 3, 1, index_map=pd.DataFrame({'arange': [2, 0, 1]},
                                               index=index))
    with torch.no_grad():
        module.rhs_embedding.lookup_embedding.weight.copy_(
            torch.tensor([[1e20], [-1e20], [3.]]))
    expected = copy.deepcopy(module.state_dict())
    module.rhs_embedding.num_nodes = 4
    module.rhs_embedding.order_by(
        pd.DataFrame({'arange': [2, 0, 1, 3]},
                     index=pd.Index([10, 20, 30, 40],
                                    name=TTNG_RHS_PRIMARY_KEY_INDEX_NAME)))
    verify_remapped_state(module, expected, [10, 20, 30, 40], 40)


if __name__ == '__main__':
    _predict_retained(Path(sys.argv[1]))
else:
    import pytest

    pr_link = pytest.fixture(scope='module')(_link_fixture)
