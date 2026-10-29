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
                 version=metadata.version('kumo-engine'), query=query.query)))
    completed = subprocess.run(
        [sys.executable, '-I',
         str(Path(__file__).resolve()),
         str(request)], capture_output=True, text=True)
    (tmp_path / 'child.log').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / 'child/result.json').read_text())
    assert result['status'] == 'PASS' and result['fits_started'] == 0
    assert result['cuda_prediction_batches'] > 0
    assert _hash_files(data) == original['data']
    assert _hash_files(bundle) == original['bundle']


def _predict_retained(request):
    from importlib import metadata

    import pandas as pd
    import torch
    from kumo_core.train.trainer import Trainer as CoreTrainer
    from kumoapi.graph import Edge
    from kumoapi.typing import Stype
    from pytorch_lightning.callbacks import Callback

    import kumo.engine as kumo

    options = json.loads(request.read_text())
    output = Path(options['output'])
    output.mkdir()
    if (metadata.version('kumo-engine') != options['version']
            or kumo.__version__ != options['version'] or
            not Path(kumo.__file__).resolve().is_relative_to(Path(sys.prefix))
            or not torch.cuda.is_available()):
        raise ValueError('Require the selected installed wheel and real CUDA')
    result = dict(status='RUNNING', fits_started=0, cuda_prediction_batches=0,
                  installed_import=kumo.__file__, version=kumo.__version__)

    def forbidden_fit(*args, **kwargs):
        result['fits_started'] += 1
        raise AssertionError('The retained-model check must not train')

    class Proof(Callback):
        def on_predict_batch_end(self, trainer, module, values, batch,
                                 batch_idx, dataloader_idx=0):
            logits = (values.get('logits', values.get('score')) if isinstance(
                values, dict) else None)
            tensors = (list(logits.values())
                       if isinstance(logits, dict) else [logits])
            if (trainer.strategy.root_device.type != 'cuda' or not tensors
                    or any(not isinstance(value, torch.Tensor)
                           or not value.numel() or value.device.type != 'cuda'
                           for value in tensors)):
                raise AssertionError('Missing completed CUDA prediction work')
            torch.cuda.synchronize()
            result['cuda_prediction_batches'] += 1

    original_create = CoreTrainer.create_pl_trainer

    def create(trainer):
        original_create(trainer)
        trainer.pl_trainer.callbacks.append(Proof())

    CoreTrainer.fit = forbidden_fit
    kumo.Trainer.fit = forbidden_fit
    CoreTrainer.create_pl_trainer = create
    data = Path(options['data'])
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
    model = kumo.Trainer.load(Path(options['bundle']))
    query = kumo.PredictiveQuery(graph, options['query'])
    table = query.generate_prediction_table(
        snapshot, plan=model.training_table_generation_plan)
    frame = model.predict(table, snapshot=snapshot,
                          num_classes_to_return=3).frame
    ids = pd.read_parquet(data / 'ents.parquet')['ID'].tolist()
    catalog = pd.read_parquet(data / 'items.parquet')['IID'].tolist()
    result.update(verify_recommendations(frame, ids, catalog, 3))
    if not result['cuda_prediction_batches']:
        raise AssertionError('No CUDA prediction batches completed')
    frame.to_parquet(output / 'predictions.parquet', index=False)
    result.update(status='PASS', snapshot_id=snapshot.id)
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


if __name__ == '__main__':
    _predict_retained(Path(sys.argv[1]))
else:
    import pytest

    pr_link = pytest.fixture(scope='module')(_link_fixture)
