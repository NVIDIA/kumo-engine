# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-local trainer options.

`BiasOptions`: API, bundle records, load and warm start. `run_name`
(contract C10): validation, run ids, refusal of reuse, bitwise-equal fits in
fresh scratch directories, and prediction identities of a named trainer.
"""
import contextlib
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.explain.walk_schema import WalkSchemaError, validate_v2_row
from kumo_core.training.orchestration import BiasOptions
from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumoapi.train import TrainingTableSpec
from walk_binary_fixtures import (
    BINARY_QUERY,
    binary_graph,
    binary_plan,
    write_binary_tables,
)

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.evidence import (
    EvidenceConfig,
    PredictionEvidenceBundle,
    WalkAttributionConfig,
)
from kumo.engine.trainer._spec import (
    MANIFEST_FILE,
    SPEC_FILE,
    read_training_spec,
    write_training_spec,
)

GNN = 'kumo_core.modeling.gnn.hetero.HeteroGNN'
HEAD = 'kumo_core.modeling.heads.node_prediction.NodePredHead'


def test_bias_options_is_exported_and_type_checked():
    assert kumo.BiasOptions is BiasOptions
    assert 'BiasOptions' in kumo.__all__
    assert kumo.Trainer(ModelPlan()).bias_options is None
    options = BiasOptions(gnn=False, head=False)
    assert kumo.Trainer(ModelPlan(), bias_options=options).bias_options is (
        options)
    with pytest.raises(TypeError):
        kumo.Trainer(ModelPlan(), bias_options={'gnn': False})


def _bundle(root, *, gnn, head, recorded):
    bundle = root / 'published-model'
    (bundle / 'model').mkdir(parents=True)
    config = json.dumps({'model': {'_target_': GNN, 'bias': gnn},
                         'head': {'_target_': HEAD, 'bias': head}})
    (bundle / 'model' / 'config.json').write_text(config)
    manifest = {'artifacts': [{
        'role': 'trainer_config', 'relative_path': 'model/config.json',
        'size_bytes': len(config),
        'sha256': hashlib.sha256(config.encode()).hexdigest()}]}
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest))
    write_training_spec(bundle, predictive_query=BINARY_QUERY,
                        model_plan=ModelPlan(), snapshot_id='snap',
                        training_table_generation=TrainingTableGenerationPlan(),
                        bias_options=recorded)
    return bundle


def test_spec_records_only_non_default_options(tmp_path):
    default = _bundle(tmp_path / 'a', gnn=True, head=True, recorded=None)
    assert 'bias_options' not in json.loads((default / SPEC_FILE).read_text())
    assert read_training_spec(default).bias_options is None
    free = _bundle(tmp_path / 'b', gnn=False, head=False,
                   recorded={'gnn': False, 'head': False})
    assert json.loads((free / SPEC_FILE).read_text())['bias_options'] == {
        'gnn': False, 'head': False}
    assert read_training_spec(free).bias_options == {'gnn': False,
                                                     'head': False}


def test_load_restores_recorded_options(tmp_path):
    free = _bundle(tmp_path / 'free', gnn=False, head=True,
                   recorded={'gnn': False, 'head': True})
    assert kumo.Trainer.load(free).bias_options == BiasOptions(gnn=False)
    default = _bundle(tmp_path / 'default', gnn=True, head=True,
                      recorded=None)
    assert kumo.Trainer.load(default).bias_options is None


@pytest.mark.parametrize('config,recorded', [
    ((False, False), None),
    ((True, True), {'gnn': False, 'head': False}),
    ((False, True), {'gnn': False, 'head': False}),
])
def test_load_rejects_spec_that_disagrees_with_trained_config(
        tmp_path, config, recorded):
    bundle = _bundle(tmp_path, gnn=config[0], head=config[1],
                     recorded=recorded)
    with pytest.raises(ValueError, match='bias options'):
        kumo.Trainer.load(bundle)


# End-to-end fits ##############################################################


@pytest.fixture(scope='module')
def binary_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp('bias-options')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    try:
        data = root / 'data'
        data.mkdir()
        write_binary_tables(data)
        connector = kumo.FileConnector(root_dir=str(data))
        graph = binary_graph(kumo, connector)
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=BINARY_QUERY)
        yield snapshot, query.generate_training_table(snapshot), root
    finally:
        kumo.configure(scratch_dir=previous)


def _state_keys(bundle_dir):
    path = Path(bundle_dir) / 'model' / 'checkpoint'
    state = torch.load(path, map_location='cpu',
                       weights_only=False)['state_dict']
    return set(state)


def _fit(snapshot_table, *, bias_options=None, warm_start=None, epochs=1):
    snapshot, table, _ = snapshot_table
    trainer = kumo.Trainer(binary_plan(epochs=epochs, steps=2),
                           accelerator='cpu', bias_options=bias_options)
    trainer.fit(table, snapshot=snapshot, warm_start=warm_start)
    return trainer


def _bundle_dir(trainer):
    result = trainer.result
    assert result is not None
    return result.bundle_dir


def test_bias_free_fit_records_and_reloads(binary_snapshot):
    trainer = _fit(binary_snapshot, bias_options=BiasOptions(gnn=False,
                                                             head=False))
    bundle = _bundle_dir(trainer)
    config = json.loads((bundle / 'model' / 'config.json').read_text())
    assert config['model']['bias'] is False and config['head']['bias'] is False
    keys = _state_keys(bundle)
    assert not [k for k in keys if k.startswith(('model.', 'head.'))
                and k.endswith('.bias')]
    assert [k for k in keys if k.startswith('encoder.')]
    assert json.loads((bundle / SPEC_FILE).read_text())['bias_options'] == {
        'gnn': False, 'head': False}
    copy = binary_snapshot[2] / 'bias-free-copy'
    shutil.copytree(bundle, copy)
    assert kumo.Trainer.load(copy).bias_options == BiasOptions(gnn=False,
                                                               head=False)


def test_default_fit_keeps_spec_bytes_and_biases(binary_snapshot):
    trainer = _fit(binary_snapshot)
    bundle = _bundle_dir(trainer)
    config = json.loads((bundle / 'model' / 'config.json').read_text())
    assert config['model']['bias'] is True and config['head']['bias'] is True
    assert 'bias_options' not in json.loads((bundle / SPEC_FILE).read_text())
    keys = _state_keys(bundle)
    assert 'head.mlp.lins.0.bias' in keys
    assert kumo.Trainer.load(bundle).bias_options is None


def test_warm_start_inherits_flags_and_rejects_explicit_conflict(
        binary_snapshot):
    source = _fit(binary_snapshot, bias_options=BiasOptions(gnn=False,
                                                            head=False))
    inherited = _fit(binary_snapshot, warm_start=source)
    config = json.loads((_bundle_dir(inherited) / 'model' / 'config.json').read_text())
    assert config['model']['bias'] is False and config['head']['bias'] is False
    assert json.loads((_bundle_dir(inherited) / SPEC_FILE).read_text())['bias_options'] == {
                           'gnn': False, 'head': False}
    with pytest.raises(ValueError, match='conflict with the warm-start'):
        _fit(binary_snapshot, bias_options=BiasOptions(), warm_start=source)


# Named runs (contract C10) ####################################################

NAME = 'p2-c10'


@contextlib.contextmanager
def _scratch(path):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=path)
    try:
        yield
    finally:
        kumo.configure(scratch_dir=previous)


@contextlib.contextmanager
def _strict_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _snapshot_stub(index_path) -> Any:
    """The one attribute the run-id check reads from a snapshot."""
    return SimpleNamespace(index_path=index_path)


@pytest.mark.parametrize('name', ['a', '0', 'relhm-churn-1', 'x' * 48, 'a--'])
def test_valid_run_names(name):
    trainer = kumo.Trainer(ModelPlan(), run_name=name)
    assert trainer.run_name == name and trainer.scratch_dir.name == name


@pytest.mark.parametrize('name,error', [
    ('', ValueError), ('-a', ValueError), ('Ab', ValueError), ('a_b', ValueError),
    ('a b', ValueError), ('x' * 49, ValueError), ('café', ValueError),
    ('a\n', ValueError), ('../a', ValueError), (1, TypeError), (b'a', TypeError)])
def test_invalid_run_names_raise(name, error):
    with pytest.raises(error, match='run_name'):
        kumo.Trainer(ModelPlan(), run_name=name)


def test_run_ids_stay_random_by_default_and_count_fits_when_named(tmp_path):
    snapshot = _snapshot_stub(tmp_path / 'index')
    with _scratch(tmp_path / 'scratch'):
        first, second = kumo.Trainer(ModelPlan()), kumo.Trainer(ModelPlan())
        assert first.run_name is None
        assert re.fullmatch('[0-9a-f]{32}', first.scratch_dir.name)
        assert first.scratch_dir != second.scratch_dir
        ids = [first._next_fit_run_id(snapshot) for _ in range(3)]
        assert len(set(ids)) == 3 and all(re.fullmatch(
            f'{first.scratch_dir.name}-fit-[0-9a-f]{{32}}', i) for i in ids)
        named = kumo.Trainer(ModelPlan(), run_name=NAME)
        assert named.scratch_dir == tmp_path / 'scratch' / 'trainers' / NAME
        assert [named._next_fit_run_id(snapshot) for _ in range(3)] == [
            f'{NAME}-fit-0', f'{NAME}-fit-1', f'{NAME}-fit-2']


@pytest.mark.parametrize('taken', ['run_directory', 'extension'])
def test_named_run_refuses_reuse_without_consuming_an_ordinal(tmp_path, taken):
    snapshot = _snapshot_stub(tmp_path / 'index')
    with _scratch(tmp_path / 'scratch'):
        trainer = kumo.Trainer(ModelPlan(), run_name=NAME)
        path = (trainer.scratch_dir / 'runs' / f'{NAME}-fit-0'
                if taken == 'run_directory' else
                tmp_path / 'index' / 'train_tables' / f'tt-{NAME}-fit-0')
        path.mkdir(parents=True)
        for _ in range(2):
            with pytest.raises(FileExistsError, match=f'{NAME}-fit-0'):
                trainer._next_fit_run_id(snapshot)
        path.rmdir()
        assert trainer._next_fit_run_id(snapshot) == f'{NAME}-fit-0'


def _training_rows(data):
    """Caller-supplied rows with a manual split, as the Rel-HM pipeline
    trains: a fixed row order. (`generate_training_table` may emit its rows in
    another order on every call, so fits on two generated tables differ.)"""
    users = pd.read_parquet(data / 'users.parquet')
    fold = np.arange(len(users)) % 8
    pd.DataFrame({'ENTITY1': users['id'], 'TARGET1': users['churn'],
                  'SPLIT': np.where(fold < 6, 0, np.where(fold == 6, 1, 2))
                  }).to_parquet(data / 'rows.parquet')


def _fresh_fit(scratch, data, run_name):
    """Snapshot, training table and one strict fit in a fresh `scratch`."""
    with _scratch(scratch):
        connector = kumo.FileConnector(root_dir=str(data))
        graph = binary_graph(kumo, connector)
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=BINARY_QUERY)
        table = kumo.TrainingTable.from_source_table(
            connector['rows.parquet'], TrainingTableSpec(split_col='SPLIT'),
            pq=query, snapshot=snapshot)
        trainer = kumo.Trainer(binary_plan(epochs=2, steps=4),
                               accelerator='cpu', run_name=run_name)
        with _strict_determinism():
            trainer.fit(table, snapshot=snapshot)
    return trainer, snapshot, table, connector


@pytest.fixture(scope='module')
def named_fits(tmp_path_factory):
    root = tmp_path_factory.mktemp('named-runs')
    data = root / 'data'
    data.mkdir()
    write_binary_tables(data)
    _training_rows(data)
    fits = {label: _fresh_fit(root / label, data, name)
            for label, name in (('named-a', NAME), ('named-b', NAME),
                                ('random-a', None), ('random-b', None))}
    return fits, root


def _state(trainer):
    return torch.load(_bundle_dir(trainer) / 'model' / 'checkpoint',
                      map_location='cpu', weights_only=False)['state_dict']


def _files(bundle):
    return {p.relative_to(bundle) for p in bundle.rglob('*') if p.is_file()}


def test_same_run_name_in_fresh_scratches_gives_bitwise_equal_fits(named_fits):
    fits, root = named_fits
    a, b = (_state(fits[label][0]) for label in ('named-a', 'named-b'))
    assert list(a) == list(b)
    assert all(torch.equal(v, b[k]) if torch.is_tensor(v) else v == b[k]
               for k, v in a.items())
    # Control: the same fits with random run ids differ, so the name is
    # what makes them equal.
    c, d = (_state(fits[label][0]) for label in ('random-a', 'random-b'))
    assert any(torch.is_tensor(v) and not torch.equal(v, d[k]) for k, v in c.items())
    for label in ('named-a', 'named-b'):
        assert _bundle_dir(fits[label][0]) == (root / label / 'trainers' / NAME / 'runs'
                                               / f'{NAME}-fit-0' / 'published-model')
    # Bundles keep their format: the same files and spec keys as a random run.
    named, random = (_bundle_dir(fits[label][0]) for label in ('named-a', 'random-a'))
    assert _files(named) == _files(random)
    spec = json.loads((named / SPEC_FILE).read_text())
    assert set(spec) == set(json.loads((random / SPEC_FILE).read_text()))
    assert NAME not in json.dumps(spec)


def test_named_runs_refuse_a_used_run_directory_or_extension_in_fit(named_fits):
    fits, root = named_fits
    _, snapshot, table, _ = fits['named-a']
    plan = binary_plan(epochs=2, steps=4)
    with _scratch(root / 'named-a'):
        again = kumo.Trainer(plan, accelerator='cpu', run_name=NAME)
        with pytest.raises(FileExistsError, match=f'runs/{NAME}-fit-0'):
            again.fit(table, snapshot=snapshot)
    with _scratch(root / 'fresh-trainers'):  # a fresh trainer, the used index
        again = kumo.Trainer(plan, accelerator='cpu', run_name=NAME)
        with pytest.raises(FileExistsError, match=f'train_tables/tt-{NAME}-fit-0'):
            again.fit(table, snapshot=snapshot)
    assert again.result is None
    assert not (root / 'fresh-trainers' / 'trainers').exists()


@pytest.fixture(scope='module')
def named_evidence(named_fits):
    """Two predict calls of one named trainer, each with walk evidence."""
    fits, root = named_fits
    trainer, snapshot, _, connector = fits['named-a']
    pd.DataFrame({'ENTITY1': np.arange(4)}).to_parquet(root / 'data' / 'score.parquet')
    config = WalkAttributionConfig(selected_class=True, top_k=10000, max_walks=100000)
    evidence = EvidenceConfig(numerical_sensitivity=None, walk_attribution=config)
    bundles = []
    with _scratch(root / 'named-a'):
        for k in range(2):
            result = trainer.predict(snapshot=snapshot, entities=connector['score.parquet'],
                                     batch_size=1, evidence=evidence)
            bundles.append(result.evidence.export(root / f'named-evidence-{k}'))
    return bundles, root


def test_named_trainer_predictions_keep_unique_identities(named_evidence):
    bundles, root = named_evidence
    ids = [{r['prediction_id'] for r in bundle.iter_numerical_evidence()}
           for bundle in bundles]
    assert len(ids[0]) == len(ids[1]) == 4 and not ids[0] & ids[1]
    runs = sorted(path.name for path in (root / 'named-a' / 'trainers' / NAME / 'runs').iterdir()
                  if path.name.startswith(f'{NAME}-predict-'))
    assert len(runs) == 2 and all(
        re.fullmatch(f'{NAME}-predict-[0-9a-f]{{32}}', run) for run in runs)
    for bundle in bundles:  # each call wrote complete evidence to a fresh directory
        report = PredictionEvidenceBundle.load(bundle.root).validate()
        assert report['valid'] and report['evidence_complete']
        assert report['evidence_rows'] == 4


def test_real_rows_bind_relation_steps_to_the_retained_graph(named_evidence):
    """m2 on a real trimmed graph: denominators are the retained in-degree."""
    bundle = named_evidence[0][0]
    record = next(iter(bundle.iter_numerical_evidence()))
    row, sample = record['walk_attribution'], bundle.scoring_graph(record['prediction_id'])
    assert row['status'] == 'available'
    assert all(edge['num_sampled_edges'] for edge in sample['edges'])
    validate_v2_row(row, sample)
    relation = [step for walk in row['walks'] for step in walk['steps']
                if step['kind'] == 'relation']
    mean = max((s for s in relation if s['operator'] == 'mean'),
               key=lambda s: s['denominator'])
    assert mean['denominator'] > 1
    mean['denominator'] += 1
    with pytest.raises(WalkSchemaError, match='Mean denominator'):
        validate_v2_row(row, sample)
    mean['denominator'] -= 1
    validate_v2_row(row, sample)
    # A retained edge whose source is not the step's source occurrence.
    entry = next(e for e in sample['edges'] if e['type'] == mean['relation'])
    entry['edge_index'][0][mean['edge_occurrence']] += 1
    with pytest.raises(WalkSchemaError, match='endpoints differ'):
        validate_v2_row(row, sample)


def test_sample_sha256_is_the_canonical_encoding_of_the_retained_sample(named_evidence):
    """The frozen `sample_sha256` form (n8) that downstream consumers recompute."""
    bundle = named_evidence[0][0]
    for record in bundle.iter_numerical_evidence():
        identifier = record['prediction_id']
        stored = (bundle.root / 'xai' / 'local' / 'samples' / f'{identifier}.json').read_bytes()
        digest = hashlib.sha256(stored).hexdigest()
        assert digest == record['sample_sha256'] == record['walk_attribution']['sample_sha256']
        encoded = json.dumps(json.loads(stored), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
