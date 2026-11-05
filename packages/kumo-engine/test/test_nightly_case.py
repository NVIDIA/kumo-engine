# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks for the shared nightly case execution contract."""

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_future_target_import

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_case.py'
future_targets = test_future_target_import.future_targets


@pytest.fixture
def case_runner():
    if not SOURCE.exists():
        pytest.fail('The shared nightly case module is not implemented')
    spec = importlib.util.spec_from_file_location('nightly_case_tested', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nightly_plan_caps_only_the_reviewed_budget(case_runner):
    from kumoapi.model_plan import ModelPlan

    suggested = ModelPlan()
    suggested.training_job.num_experiments = 4
    suggested.optimization.max_epochs = 8
    suggested.optimization.max_steps_per_epoch = 1000
    suggested.optimization.max_val_steps = 1000
    suggested.optimization.max_test_steps = 2000
    suggested.optimization.batch_size = [512, 1024]
    suggested.optimization.loss = ['huber']
    query = SimpleNamespace(suggest_model_plan=lambda snapshot: suggested)
    before = copy.deepcopy(suggested.as_json())
    plan = case_runner.capped_plan(query, object())
    assert plan.training_job.num_experiments == 2
    assert plan.optimization.max_epochs == 3
    assert plan.optimization.max_steps_per_epoch == 200
    assert plan.optimization.max_val_steps == 100
    assert plan.optimization.max_test_steps == 200
    after = plan.as_json()
    for block, fields in [('training_job', ['num_experiments']),
                          ('optimization', [
                              'max_epochs', 'max_steps_per_epoch',
                              'max_val_steps', 'max_test_steps'
                          ])]:
        for name in fields:
            after[block].pop(name)
            before[block].pop(name)
    assert after == before
    assert suggested.optimization.max_epochs == 8


@pytest.mark.parametrize('depth', [2, 4, 5, 6])
def test_deep_variants_preserve_fanouts_and_only_extend_short_sampler(
        case_runner, depth):
    from kumoapi.model_plan import HopConfig, ModelPlan, NumNeighborsConfig

    suggested = ModelPlan()
    suggested.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(
            **{f'hop{i + 1}': HopConfig(default=12 + i)
               for i in range(depth)})
    ]
    original = copy.deepcopy(suggested.as_json())
    query = SimpleNamespace(suggest_model_plan=lambda snapshot: suggested)
    plan = case_runner.capped_plan(query, object(), min_hops=4, max_hops=6)
    hops = plan.neighbor_sampling.num_neighbors[0].to_dict()
    assert len(hops) == max(depth, 4)
    assert [hop.default
            for hop in hops.values()] == (list(range(12, 12 + depth)) +
                                          [11 + depth] * max(4 - depth, 0))
    assert suggested.as_json() == original
    assert case_runner.neighbor_depths(plan) == [max(depth, 4)]


@pytest.mark.parametrize('bounds', [(0, 6), (5, 4), (4, 7), (True, 6),
                                    (4, None)])
def test_invalid_graph_depth_bounds_are_rejected(case_runner, bounds):
    from kumoapi.model_plan import ModelPlan

    query = SimpleNamespace(suggest_model_plan=lambda snapshot: ModelPlan())
    with pytest.raises(ValueError, match='hop'):
        case_runner.capped_plan(query, object(), min_hops=bounds[0],
                                max_hops=bounds[1])


def test_deep_variants_reject_gaps_and_zero_only_hops(case_runner):
    from kumoapi.model_plan import HopConfig, ModelPlan, NumNeighborsConfig

    for options in [
            NumNeighborsConfig(hop1=HopConfig(default=12),
                               hop3=HopConfig(default=12)),
            NumNeighborsConfig(hop1=HopConfig(default=12),
                               hop2=HopConfig(default=0))
    ]:
        suggested = ModelPlan()
        suggested.neighbor_sampling.num_neighbors = [options]
        query = SimpleNamespace(suggest_model_plan=lambda snapshot: suggested)
        with pytest.raises(ValueError, match='hop'):
            case_runner.capped_plan(query, object(), min_hops=4, max_hops=6)


def test_normal_case_matching_keeps_binary_variants_separate(case_runner):
    core = dict(task='binary', arm='normal', scale='100M')
    payments = dict(core, variant='payments')
    fraud = dict(core, variant='fraud')
    large = dict(payments, scale='1B')
    manifest = dict(cases=[core, fraud, large, payments])
    assert case_runner.normal_case(manifest, dict(payments,
                                                  arm='control')) is payments
    assert case_runner.normal_case(manifest, dict(fraud,
                                                  arm='control')) is fraud
    assert case_runner.normal_case(manifest, dict(large,
                                                  arm='control')) is large
    assert case_runner.normal_case(manifest, dict(core, arm='control')) is core


def test_streaming_uses_fifty_results_and_fifty_one_for_cutoff(case_runner):
    case = dict(variant='streaming', task='temporal_link', top_k=50,
                raw_limit=500)
    assert case_runner.prediction_top_k('temporal_link', case) == 50
    assert case_runner.prediction_limit('temporal_link', case,
                                        list(range(100))) == 51
    assert case_runner.prediction_limit('temporal_link', case,
                                        list(range(30))) == 30
    with pytest.raises(ValueError, match='top K'):
        case_runner.prediction_limit('temporal_link', dict(case, raw_limit=49),
                                     list(range(100)))


def test_streaming_popularity_baseline_uses_same_fifty_results(
        case_runner, tmp_path):
    pd.DataFrame(dict(ENTITY1=[1],
                      TARGET1=[list(range(60))
                               ])).to_parquet(tmp_path / 'train.parquet',
                                              index=False)
    truth = pd.DataFrame(dict(ENTITY1=[2], TARGET1=[[49]]))
    observed = {}

    def quality(task, truth, values, classes):
        observed.update(values)
        return 1.

    helpers = SimpleNamespace(quality=quality,
                              validate_quality_measurement=lambda *args: None)
    result = case_runner.baseline_score(
        'temporal_link', SimpleNamespace(staging_dir=tmp_path),
        dict(train=dict(parts=['train.parquet'])), truth, [], list(range(60)),
        helpers=helpers, top_k=50)
    assert observed['ranked_ids'] == [list(range(50))]
    assert result['score'] == 1.


@pytest.mark.parametrize('count,sample,subset,covered', [
    (2000, 1000, 1000, True),
    (1000, 1000, 500, True),
    (600, 600, 300, True),
    (2, 2, 1, True),
    (1, 1, 0, False),
])
def test_samples_preserve_identity_and_a_proper_subset(case_runner, count,
                                                       sample, subset, covered):
    frame = pd.DataFrame(
        dict(ENTITY1=range(count),
             TIMESTAMP=[pd.Timestamp('2024-01-01')] * count))
    selected = case_runner.sample_populations(frame)
    assert len(selected['consistency']) == sample
    assert len(selected['partial']) == subset
    assert selected['proper_subset'] is covered
    assert set(selected['consistency'].ENTITY1) <= set(frame.ENTITY1)
    assert set(selected['partial'].ENTITY1) <= set(frame.ENTITY1)
    again = case_runner.sample_populations(frame.iloc[::-1])
    assert selected['consistency'].ENTITY1.tolist(
    ) == again['consistency'].ENTITY1.tolist()
    assert selected['partial'].ENTITY1.tolist(
    ) == again['partial'].ENTITY1.tolist()


def test_sorted_population_reuses_order_with_identical_samples(
        case_runner, monkeypatch):
    frame = pd.DataFrame(dict(ENTITY1=range(20)))
    expected = case_runner.sample_populations(frame, size=7)

    def reject_sort(*args, **kwargs):
        pytest.fail('An already sorted population was sorted again')

    monkeypatch.setattr(frame, 'sort_values', reject_sort)
    actual = case_runner.sample_populations(frame, size=7, ordered=True)
    for key in ('consistency', 'partial'):
        pd.testing.assert_frame_equal(actual[key], expected[key])
    assert actual['proper_subset'] == expected['proper_subset']


def test_empty_prediction_population_is_rejected(case_runner):
    with pytest.raises(ValueError):
        case_runner.sample_populations(pd.DataFrame(dict(ENTITY1=[])))


def test_coverage_counts_distinct_candidates_and_completed_epochs(case_runner):
    first = dict(
        trial_id='candidate-a', kind='candidate', status='PASS', steps=3,
        cuda_batches=3, epochs=[
            dict(epoch=0, batches=2, completed=True),
            dict(epoch=1, batches=1, completed=True)
        ])
    retry = dict(first)
    refit = dict(first, trial_id='refit-a', kind='refit')
    assert case_runner.check_training_coverage([first, retry, refit]) == dict(
        status='FAIL', distinct_trials=1, max_completed_epochs=2)
    second = dict(first, trial_id='candidate-b')
    assert case_runner.check_training_coverage([first, second, retry]) == dict(
        status='PASS', distinct_trials=2, max_completed_epochs=2)
    incomplete = dict(first, epochs=[dict(epoch=0, batches=1, completed=False)])
    assert case_runner.check_training_coverage(
        [incomplete, dict(incomplete,
                          trial_id='candidate-b')])['status'] == 'FAIL'


def sorting_helpers():
    def ordered(files, directory, *, ranked=False, sort_keys=None):
        table = pa.concat_tables([pq.read_table(path) for path in files])
        yield table.sort_by(sort_keys).to_pandas()

    return SimpleNamespace(ordered_batches=ordered)


def label_rows():
    return pd.DataFrame(
        dict(ENTITY1=[2, 1, 2, 1, 2,
                      1], TIMESTAMP=pd.to_datetime(['2024-01-01'] * 6),
             SPLIT=[0, 0, 1, 1, 2, 2], TARGET1=[2., 1., 4., 3., 6., 5.]))


def test_logical_stage_hashes_ignore_shards_and_detect_changed_answers(
        case_runner, tmp_path):
    frame = label_rows()
    first, shuffled, corrupt = (
        tmp_path / name
        for name in ['first.parquet', 'shuffled.parquet', 'corrupt.parquet'])
    frame.to_parquet(first, index=False)
    frame.iloc[::-1].to_parquet(shuffled, index=False)
    changed = frame.copy()
    changed.loc[0, 'TARGET1'] = 100.
    changed.to_parquet(corrupt, index=False)
    expected = case_runner.stage_identities([str(first)], 'regression',
                                            tmp_path / 'sort-a',
                                            helpers=sorting_helpers())
    assert {
        key: value['rows']
        for key, value in expected.items()
    } == dict(train=2, val=2, test=2)
    assert case_runner.stage_identities([str(shuffled)], 'regression',
                                        tmp_path / 'sort-b',
                                        helpers=sorting_helpers()) == expected
    actual = case_runner.stage_identities([str(corrupt)], 'regression',
                                          tmp_path / 'sort-c',
                                          helpers=sorting_helpers())
    assert actual['train']['sha256'] != expected['train']['sha256']
    assert actual['test'] == expected['test']


def test_control_payload_preserves_generated_inputs_and_stage_keys(case_runner):
    normal = label_rows()
    control = normal.copy()
    control['TARGET1'] = [1., 2., 3., 4., 5., 6.]
    actual = case_runner.control_payload(normal, control, 'regression')
    pd.testing.assert_frame_equal(actual[['ENTITY1', 'TIMESTAMP', 'SPLIT']],
                                  normal[['ENTITY1', 'TIMESTAMP', 'SPLIT']])
    assert actual.TARGET1.tolist() == control.TARGET1.tolist()
    bad = control.copy()
    bad.loc[0, 'ENTITY1'] = 99
    with pytest.raises(ValueError):
        case_runner.control_payload(normal, bad, 'regression')


def test_forecast_control_keeps_history_and_moves_the_whole_answer(case_runner):
    normal = label_rows()
    normal['autoregression_labels'] = [[10., 20.]] * 6
    normal['future_labels'] = [[value, value + 1] for value in normal.TARGET1]
    control = normal.copy(deep=True)
    control['TARGET1'] = [1., 2., 3., 4., 5., 6.]
    control['future_labels'] = [[value, value + 1] for value in control.TARGET1]
    actual = case_runner.control_payload(normal, control, 'forecasting')
    assert actual.autoregression_labels.tolist(
    ) == normal.autoregression_labels.tolist()
    assert actual.future_labels.tolist() == control.future_labels.tolist()


def test_replay_failure_is_retained_while_fresh_work_continues(case_runner):
    events = []

    def replay():
        events.append('replay')
        raise ValueError('reference predictions differ')

    def fit():
        events.append('fit')
        return 'fresh result'

    result, evidence = case_runner.with_reference_replay(replay, fit)
    assert result == 'fresh result'
    assert events == ['replay', 'fit']
    assert evidence['status'] == 'FAIL'
    assert 'reference predictions differ' in evidence['error']


def test_prediction_archive_roundtrip_preserves_large_ids_and_boundaries(
        case_runner, tmp_path):
    frame = pd.DataFrame(
        dict(ENTITY1=pd.Series([2**63 + 1, 2**63 + 2], dtype='uint64')))
    values = dict(ranked_ids=[[2**63 + 5], [2**63 + 6]], ranked_scores=[[.9],
                                                                        [.8]],
                  boundary_ids=[[2**63 + 5, 7], [2**63 + 6, 8]],
                  boundary_scores=[[.9, .1], [.8, .2]], raw_rows=4)
    path = tmp_path / 'predictions.parquet'
    case_runner.write_prediction(path, frame, values, values['ranked_ids'],
                                 'static_link')
    restored, decisions, ids = case_runner.read_prediction(path, 'static_link')
    assert ids == frame.ENTITY1.tolist()
    assert {
        key: restored[key]
        for key in values if key != 'raw_rows'
    } == {
        key: values[key]
        for key in values if key != 'raw_rows'
    }
    assert decisions == values['ranked_ids']


def test_repeat_is_compared_directly_with_the_first_sample(case_runner):
    import numpy as np

    def compare(before, after, **kwargs):
        return dict(status='PASS' if np.max(np.abs(before -
                                                   after)) <= .001 else 'FAIL')

    helpers = SimpleNamespace(smoke=SimpleNamespace(
        compare_predictions=compare))
    result = case_runner.compare_values('regression', np.array([[.0009]]),
                                        np.array([[-.0009]]), dict(atol=.001),
                                        helpers=helpers)
    assert result['status'] == 'FAIL'


def comparison_helpers():
    modules = []
    for name, path in [('smoke', SOURCE.parents[1] / 'test/wheel_gpu_smoke.py'),
                       ('predictions',
                        SOURCE.with_name('nightly_predictions.py'))]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return SimpleNamespace(smoke=modules[0],
                           ranked_difference=modules[1].ranked_difference)


def test_pending_prediction_bound_records_measurement_without_approving(
        case_runner):
    import numpy as np

    result = case_runner.compare_values('regression', np.array([[1.]]),
                                        np.array([[1.1]]), dict(atol=None),
                                        helpers=comparison_helpers(),
                                        measure_only=True)
    assert result['status'] == 'MEASURED'
    assert result['atol'] is None
    assert result['max_abs_difference'] == pytest.approx(.1)
    with pytest.raises(ValueError):
        case_runner.compare_values('regression', np.array([[1.]]),
                                   np.array([[1.1]]), dict(atol=None),
                                   helpers=comparison_helpers())


def test_pending_bound_does_not_hide_missing_cutoff_evidence(case_runner):
    before = dict(ranked_ids=[[1]], ranked_scores=[[.9]], boundary_ids=[[1, 3]],
                  boundary_scores=[[.9, .1]])
    after = dict(ranked_ids=[[2]], ranked_scores=[[.91]], boundary_ids=[[2, 3]],
                 boundary_scores=[[.91, .1]])
    result = case_runner.compare_values('static_link', before, after,
                                        dict(atol=None),
                                        helpers=comparison_helpers(),
                                        measure_only=True)
    assert result['status'] == 'FAIL'
    assert result['unproven_membership_changes'] == 1


def test_static_exclusions_come_from_prior_generated_stages(
        case_runner, tmp_path):
    from types import SimpleNamespace

    train = pd.DataFrame(dict(ENTITY1=[1, 2], TARGET1=[[10, 11], [20]]))
    val = pd.DataFrame(dict(ENTITY1=[1], TARGET1=[[12]]))
    truth = pd.DataFrame(dict(ENTITY1=[1, 2], TARGET1=[[13], [21]]))
    train.to_parquet(tmp_path / 'train.parquet', index=False)
    val.to_parquet(tmp_path / 'val.parquet', index=False)
    stages = dict(train=dict(parts=['train.parquet']),
                  val=dict(parts=['val.parquet']))
    actual = case_runner.static_exclusions(
        SimpleNamespace(staging_dir=tmp_path), stages, truth)
    assert actual.exclude_target_keys.tolist() == [[10, 11, 12], [20]]
    corrupt = truth.assign(exclude_target_keys=[[10], [20]])
    with pytest.raises(ValueError):
        case_runner.static_exclusions(SimpleNamespace(staging_dir=tmp_path),
                                      stages, corrupt)


@pytest.mark.parametrize('task,requested,catalog_size,expected', [
    ('temporal_link', 500, 100, 21),
    ('temporal_link', 500, 6, 6),
    ('static_link', 500, 1000, 500),
    ('static_link', 500, 100, 100),
])
def test_temporal_prediction_budget_does_not_inherit_legacy_500(
        case_runner, task, requested, catalog_size, expected):
    assert case_runner.prediction_limit(task, dict(raw_limit=requested),
                                        list(range(catalog_size))) == expected


def run_cpu_case(case_runner, tmp_path, monkeypatch, count, *,
                 replay_error=None, case_fields=None, resolved_depth=None,
                 rank_tie=False):
    import json
    from contextlib import contextmanager

    import torch
    from kumoapi.model_plan import HopConfig, ModelPlan, NumNeighborsConfig

    import kumo.engine as kumo

    spec = importlib.util.spec_from_file_location(
        'case_shared_helpers', SOURCE.with_name('nightly_check.py'))
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    helpers = SimpleNamespace(**shared.__dict__)
    helpers.input_signature = lambda manifest, case, **kwargs: dict(fixture=True
                                                                    )
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    full = pd.DataFrame(
        dict(ENTITY1=range(count),
             TIMESTAMP=[pd.Timestamp('2024-01-01')] * count))
    full.to_parquet(inputs / 'full.parquet', index=False)
    (inputs / 'descriptor.json').write_text(json.dumps(
        dict(query='test query')))
    case = dict(task='regression', arm='normal', seed=0, scale='100M',
                descriptor='descriptor.json', prediction_input='full.parquet')
    case.update(case_fields or {})
    task = case['task']
    if task.endswith('link'):
        pd.DataFrame(dict(item_id=range(60))).to_parquet(
            inputs / 'catalog.parquet', index=False)
        case['catalog'] = dict(path='catalog.parquet', column='item_id')
    if replay_error:
        case['replay_error'] = replay_error
    manifest = dict(
        cases=[case], files={
            name: shared.digest(inputs / name)
            for name in ('descriptor.json', 'full.parquet')
        })
    if task.endswith('link'):
        manifest['files']['catalog.parquet'] = shared.digest(inputs /
                                                             'catalog.parquet')
    table_root = tmp_path / 'table'
    table_root.mkdir()
    train = full.iloc[:min(2, count)].assign(TARGET1=[0.] * min(2, count))
    truth = full.assign(TARGET1=(full.ENTITY1 %
                                 2 == 0) if task == 'binary' else full.ENTITY1 /
                        10.)
    if task.endswith('link'):
        train['TARGET1'] = [list(range(60))] * len(train)
        truth['TARGET1'] = [[49]] * len(truth)
    train.to_parquet(table_root / 'train.parquet', index=False)
    truth.to_parquet(table_root / 'test.parquet', index=False)
    table = SimpleNamespace(staging_dir=table_root)
    stages = dict(train=dict(parts=['train.parquet']),
                  test=dict(parts=['test.parquet']))
    proof = dict(normal_stages={}, stages={}, generation_plan={})
    monkeypatch.setattr(case_runner, 'generate_training_table',
                        lambda *args, **kwargs: (table, stages, proof))
    plan = ModelPlan()
    plan.optimization.batch_size = [512, 1024]
    plan.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(hop1=HopConfig(default=12),
                           hop2=HopConfig(default=12))
    ]
    plan.training_job.tune_metric = 'auroc' if task == 'binary' else 'mae'
    query = SimpleNamespace(
        validate=lambda: None, get_task_type=lambda: SimpleNamespace(
            value=shared.smoke.TASKS[task]['task_type']),
        suggest_model_plan=lambda snapshot: plan)
    monkeypatch.setattr(kumo, 'PredictiveQuery', lambda *args: query)
    monkeypatch.setattr(kumo, 'configure', lambda **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    active = {}

    @contextmanager
    def observe(state, **kwargs):
        active['state'] = state
        yield

    monkeypatch.setattr(case_runner, 'observe_cuda', observe)
    monkeypatch.setattr(
        kumo.PredictionTable, 'from_source_table',
        staticmethod(lambda source, **kwargs: pd.read_parquet(
            Path(source.connector.config.root_dir) / source.table_name)))

    class Trainer:
        training_table_generation_plan = None

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, table, **kwargs):
            state = active['state']
            bundle = tmp_path / 'fresh-model'
            bundle.mkdir()
            checkpoint = bundle / 'model.ckpt'
            torch.save(dict(global_step=2), checkpoint)
            digest = shared.digest(checkpoint)
            state.update(
                fits=2, completed_fits=2, optimizer_steps=4, training_batches=4,
                fit_ledger=[
                    dict(
                        status='PASS', steps=2, cuda_batches=2,
                        kind='candidate', trial_id=name,
                        checkpoint_sha256=digest, validation=[dict(score=1.)],
                        epochs=[
                            dict(epoch=i, batches=1, batch_sizes=[64],
                                 completed=True) for i in range(2)
                        ]) for name in ('a', 'b')
                ])
            artifacts = {'selected_checkpoint': str(checkpoint)}
            if resolved_depth is not None:
                resolved = ModelPlan()
                resolved.neighbor_sampling.num_neighbors = [
                    NumNeighborsConfig(
                        **{
                            f'hop{i + 1}': HopConfig(default=12)
                            for i in range(resolved_depth)
                        })
                ]
                path = bundle / 'resolved-model-plan.json'
                path.write_text(json.dumps(resolved.as_json()))
                artifacts['resolved_model_plan'] = str(path)
            return SimpleNamespace(bundle_dir=bundle, metrics={},
                                   artifact_paths=artifacts)

        @classmethod
        def load(cls, path):
            return cls()

        def predict(self, population, **kwargs):
            state = active['state']
            state['prediction_batches'][state['stage']] += 1
            if state['stage'] == 'reload':
                state.update(reload_checks=1, restored_tensors=1)
            frame = pd.DataFrame(
                dict(ENTITY=population.ENTITY1,
                     TARGET_PRED=population.ENTITY1 / 10.,
                     ANCHOR_TIMESTAMP=population.TIMESTAMP))
            if task == 'binary':
                positive = population.ENTITY1 % 2 == 0
                frame['TARGET_PRED'] = positive
                frame['True_PROB'] = positive.astype(float) * .8 + .1
                frame['False_PROB'] = 1 - frame.True_PROB
            if task.endswith('link'):
                frame = pd.DataFrame([
                    dict(
                        ENTITY=entity, CLASS=item,
                        SCORE=1 - (49 if rank_tie and entity == 0 and item == 50
                                   else item) / 100, ANCHOR_TIMESTAMP=stamp)
                    for entity, stamp in zip(population.ENTITY1,
                                             population.TIMESTAMP, strict=True)
                    for item in range(kwargs['num_classes_to_return'])
                ])
            config = kwargs['output_config']
            destination = Path(config.connector.config.root_dir)
            destination.mkdir(parents=True, exist_ok=True)
            path = destination / (
                config.output_paths['predictions'].rstrip('/') + '.parquet')
            frame.to_parquet(path, index=False)
            return SimpleNamespace(files=[str(path)], num_rows=len(frame),
                                   frame=frame)

    monkeypatch.setattr(kumo, 'Trainer', Trainer)
    return case_runner.run_case(inputs, manifest, case, object(), object(),
                                tmp_path / 'result', tmp_path / 'scratch',
                                dict(atol=None), helpers=helpers,
                                measure_only=True)


@pytest.mark.parametrize('count,partial', [(2000, 1000), (600, 300)])
def test_case_scores_full_population_and_repeats_reloads_sample(
        case_runner, tmp_path, monkeypatch, count, partial):
    result = run_cpu_case(case_runner, tmp_path, monkeypatch, count)
    assert result['score'] == 0.
    assert result['predictions']['predict']['entities'] == count
    assert result['predictions']['repeat']['entities'] == min(1000, count)
    assert result['predictions']['reloaded_prediction']['entities'] == min(
        1000, count)
    name = 'proper-subset' if count <= 1000 else 'sample'
    assert result['predictions'][name]['entities'] == partial
    assert result['training_coverage']['status'] == 'PASS'
    assert result['prediction_checks']['status'] == 'MEASURED'


@pytest.mark.parametrize('variant,arm', [('payments', 'normal'),
                                         ('fraud', 'control')])
def test_deep_binary_variants_record_selected_sampler_and_keep_quality_gates(
        case_runner, tmp_path, monkeypatch, variant, arm):
    result = run_cpu_case(
        case_runner, tmp_path, monkeypatch, 12,
        case_fields=dict(task='binary', variant=variant, arm=arm, min_hops=4,
                         max_hops=6), resolved_depth=4)
    assert result['variant'] == variant
    assert result['arm'] == arm
    assert result['metric'] == 'auroc'
    assert result['graph_depth'] == dict(min_hops=4, max_hops=6,
                                         configured_hops=[4], resolved_hops=[4])
    assert result['score'] == 1.
    assert result['training_coverage']['status'] == 'PASS'
    assert result['prediction_checks']['status'] == 'MEASURED'


def test_deep_variant_rejects_selected_model_that_lost_requested_hops(
        case_runner, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='resolved.*hop'):
        run_cpu_case(
            case_runner, tmp_path, monkeypatch, 12,
            case_fields=dict(task='binary', variant='payments', min_hops=4,
                             max_hops=6), resolved_depth=2)


def test_streaming_full_case_keeps_fifty_ranks_and_cutoff_in_replay_artifact(
        case_runner, tmp_path, monkeypatch):
    result = run_cpu_case(
        case_runner, tmp_path, monkeypatch, 12,
        case_fields=dict(task='temporal_link', variant='streaming', top_k=50,
                         raw_limit=51, metric='recall@50'))
    assert result['variant'] == 'streaming'
    assert result['metric'] == 'recall@50'
    assert result['top_k'] == 50
    assert result['raw_limit'] == 51
    assert result['score'] == result['baseline'] == 1.
    assert result['predictions']['predict']['rows'] == 12 * 51
    values, _, ids = case_runner.read_prediction(
        tmp_path / 'result' / result['canonical_predictions']['path'],
        'temporal_link')
    assert len(ids) == 12
    assert all(len(ranks) == 50 for ranks in values['ranked_ids'])
    assert all(len(boundary) >= 2 for boundary in values['boundary_ids'])
    assert result['training_coverage']['status'] == 'PASS'
    assert result['prediction_checks']['status'] == 'MEASURED'


def test_streaming_case_retries_cutoff_without_changing_scoring_population(
        case_runner, tmp_path, monkeypatch):
    result = run_cpu_case(
        case_runner, tmp_path, monkeypatch, 12, rank_tie=True,
        case_fields=dict(task='temporal_link', variant='streaming', top_k=50,
                         raw_limit=51, metric='recall@50'))
    assert result['status'] == 'MEASURED'
    assert result['top_k'] == 50
    assert result['predictions']['predict']['entities'] == 12
    assert result['predictions']['predict']['rows'] == 12 * 51 + 60
    assert result['predictions']['predict']['cutoff_retries'] == [
        dict(ids=[0], raw_limit=60)
    ]
    retry_input = pd.read_parquet(
        tmp_path / 'result/prediction-inputs/predict-cutoff-1.parquet')
    assert retry_input.ENTITY1.tolist() == [0]
    assert result['score'] == result['baseline'] == 1.
    assert result['training_coverage']['status'] == 'PASS'
    assert result['prediction_checks']['status'] == 'MEASURED'


def test_community_variant_keeps_its_identity_and_metric(
        case_runner, tmp_path, monkeypatch):
    result = run_cpu_case(case_runner, tmp_path, monkeypatch, 12,
                          case_fields=dict(variant='community', metric='mae'))
    assert result['variant'] == 'community'
    assert result['metric'] == 'mae'
    assert result['score'] == 0.


def test_replay_admission_error_survives_successful_fresh_training(
        case_runner, tmp_path, monkeypatch):
    result = run_cpu_case(case_runner, tmp_path, monkeypatch, 10,
                          replay_error='Reference model checksum differs')
    assert result['physical_fits'] == 2
    assert result['checkpoint_sha256']
    assert result['reference_replay']['status'] == 'FAIL'
    assert 'checksum differs' in result['reference_replay']['error']


def test_public_static_generation_and_control_import_preserve_logical_stages(
        case_runner, tmp_path, monkeypatch):
    import json

    from kumoapi.graph import Edge
    from kumoapi.typing import Stype

    import kumo.engine as kumo
    from kumo.engine._scratch import get_scratch_dir

    original_import = kumo.TrainingTable.from_source_table
    options = []

    def import_static(source, spec, **kwargs):
        options.append(kwargs.get('allow_new_target_entities', False))
        return original_import(source, spec, **kwargs)

    monkeypatch.setattr(kumo.TrainingTable, 'from_source_table',
                        staticmethod(import_static))
    data = tmp_path / 'data'
    data.mkdir()
    pd.DataFrame({'id': range(12)}).to_parquet(data / 'customers.parquet')
    pd.DataFrame({'id': range(12)}).to_parquet(data / 'merchants.parquet')
    pd.DataFrame([
        dict(id=entity * 6 + item, customer_id=entity, merchant_id=item)
        for entity in range(12) for item in range(6)
    ]).to_parquet(data / 'purchases.parquet')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'engine-scratch')
    try:
        connector = kumo.FileConnector(root_dir=str(data))
        customers = kumo.Table.from_source_table(
            connector['customers.parquet'], primary_key='id').infer_metadata()
        merchants = kumo.Table.from_source_table(
            connector['merchants.parquet'], primary_key='id').infer_metadata()
        purchases = kumo.Table.from_source_table(
            connector['purchases.parquet'], primary_key='id').infer_metadata()
        for column in ('customer_id', 'merchant_id'):
            purchases = purchases.with_column(column, stype=Stype.ID)
        graph = kumo.Graph(
            dict(customers=customers, merchants=merchants, purchases=purchases),
            [
                Edge(src_table='purchases', fkey='customer_id',
                     dst_table='customers'),
                Edge(src_table='purchases', fkey='merchant_id',
                     dst_table='merchants')
            ])
        snapshot = graph.snapshot()
        query_text = ('PREDICT LIST_DISTINCT(purchases.merchant_id) '
                      'RANK TOP 5 FOR EACH customers.id')
        query = kumo.PredictiveQuery(graph, query_text)
        descriptor = dict(
            query=query_text, query_generation_plan=dict(
                split='RandomLinkSplit([0.6, 0.2, 0.2])'))
        (data / 'normal.json').write_text(json.dumps(descriptor))
        spec = importlib.util.spec_from_file_location(
            'case_native_helpers', SOURCE.with_name('nightly_check.py'))
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        normal = dict(scale='100M', task='static_link', arm='normal', seed=0,
                      descriptor='normal.json')
        manifest = dict(
            cases=[normal],
            files={'normal.json': helpers.digest(data / 'normal.json')})
        table, _, proof = case_runner.generate_training_table(
            data, manifest, normal, query, snapshot, tmp_path / 'first',
            helpers=helpers, bootstrap=True)
        exported = case_runner._export(table, tmp_path / 'control-export')
        controlled = pd.concat([pd.read_parquet(path) for path in exported],
                               ignore_index=True)
        controlled['TARGET1'] = [[int(value) + 6 for value in row]
                                 for row in controlled.TARGET1]
        controlled.to_parquet(data / 'control.parquet', index=False)
        expected = case_runner.stage_identities([str(data / 'control.parquet')],
                                                'static_link',
                                                tmp_path / 'expected',
                                                helpers=helpers)
        normal['stage_identity'] = proof['normal_stages']
        control = dict(normal, arm='control', control_rows='control.parquet',
                       stage_identity=expected)
        manifest['cases'].append(control)
        manifest['files']['control.parquet'] = helpers.digest(data /
                                                              'control.parquet')
        imported, stages, actual = case_runner.generate_training_table(
            data, manifest, control, query, snapshot, tmp_path / 'second',
            helpers=helpers)
        assert actual['normal_stages'] == proof['normal_stages']
        assert actual['stages'] == expected
        truth = helpers.stage_frame(imported, stages['test'])
        truth = case_runner.static_exclusions(imported, stages, truth)
        assert all(6 <= int(value) < 12 for row in truth.TARGET1
                   for value in row)
        assert all(not set(row.TARGET1) & set(row.exclude_target_keys)
                   for row in truth.itertuples())
        control['stage_identity'] = copy.deepcopy(expected)
        control['stage_identity']['test']['sha256'] = '0' * 64
        with pytest.raises(ValueError, match='control stage counts or labels'):
            case_runner.generate_training_table(data, manifest, control, query,
                                                snapshot, tmp_path / 'third',
                                                helpers=helpers)
        assert options == [False, False]
    finally:
        kumo.configure(scratch_dir=previous)


@pytest.mark.parametrize('variant', [None, 'streaming'])
def test_temporal_control_import_preserves_future_target_labels(
        case_runner, future_targets, tmp_path, variant):
    import json

    from kumoapi.json_serde import to_json_dict

    query, snapshot, plan, _, _, normal_rows = future_targets
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'normal.json').write_text(
        json.dumps(
            dict(query=query.query, query_generation_plan=to_json_dict(plan))))
    pq.write_table(normal_rows, data / 'control.parquet')
    spec = importlib.util.spec_from_file_location(
        'temporal_control_helpers', SOURCE.with_name('nightly_check.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    expected = case_runner.stage_identities([str(data / 'control.parquet')],
                                            'temporal_link',
                                            tmp_path / 'expected',
                                            helpers=helpers)
    normal = dict(scale='100M', task='temporal_link', arm='normal', seed=0,
                  descriptor='normal.json', stage_identity=expected)
    if variant:
        normal['variant'] = variant
    control = dict(normal, arm='control', control_rows='control.parquet')
    manifest = dict(
        cases=[normal, control], files={
            name: helpers.digest(data / name)
            for name in ('normal.json', 'control.parquet')
        })
    _, _, proof = case_runner.generate_training_table(data, manifest, control,
                                                      query, snapshot,
                                                      tmp_path / 'import',
                                                      helpers=helpers)
    assert proof['normal_stages'] == expected
    assert proof['stages'] == expected


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('task', [
    'binary', 'multiclass', 'multilabel', 'multilabel_ranking', 'regression',
    'forecasting', 'static_link', 'temporal_link'
])
def test_label_batches_preserve_legacy_hash_bytes(case_runner, tmp_path, task,
                                                  workers):
    rows = []
    for entity in (9007199254740993, 9007199254740994):
        for split in (0, 1, 2):
            row = dict(ENTITY1=entity, SPLIT=split,
                       TIMESTAMP=pd.Timestamp('2024-01-01'),
                       TARGET1=-0.0 if entity == 9007199254740993 else 1.25)
            if task == 'forecasting':
                row.update(autoregression_labels=[-0., 2.],
                           future_labels=[1.25, -0.])
            elif task == 'temporal_link':
                row.update(TARGET1=[12, 3],
                           __kumo_internal_frequency_column__=[2., 1.5])
            elif task == 'binary':
                row['TARGET1'] = entity == 9007199254740993
            elif task == 'multiclass':
                row['TARGET1'] = 'µ' if entity == 9007199254740993 else 'z'
            elif task in ('multilabel', 'multilabel_ranking'):
                row['TARGET1'] = 'z|a'
            elif task == 'static_link':
                row.update(TARGET1=[12, 3], exclude_target_keys=[9, 8])
            rows.append(row)
    frame = pd.DataFrame(rows)
    expected = {}
    for split, stage in enumerate(('train', 'val', 'test')):
        digest = hashlib.sha256(b'nightly-logical-labels-v1\n')
        for entity in (9007199254740993, 9007199254740994):
            target = '0' if entity == 9007199254740993 else '1.25'
            if task in ('static_link', 'temporal_link'):
                target = '[3,12]'
            elif task == 'binary':
                target = 'true' if entity == 9007199254740993 else 'false'
            elif task == 'multiclass':
                target = r'"\u00b5"' if entity == 9007199254740993 else '"z"'
            elif task in ('multilabel', 'multilabel_ranking'):
                target = '["a","z"]'
            text = (f'{{"ENTITY1":{entity},"SPLIT":{split},"TARGET1":{target},'
                    '"TIMESTAMP":{"timestamp_ns":1704067200000000000}')
            if task == 'forecasting':
                text += ',"autoregression_labels":[0,2],"future_labels":[1.25,0]'
            elif task == 'temporal_link':
                text += ',"__kumo_internal_frequency_column__":[1.5,2]'
            elif task == 'static_link':
                text += ',"exclude_target_keys":[8,9]'
            digest.update((text + '}\n').encode())
        expected[stage] = dict(rows=2, sha256=digest.hexdigest())
    actual = case_runner.label_identities(
        (frame.iloc[start:start + 2] for start in range(0, len(frame), 2)),
        task, tmp_path / 'labels', workers=workers)
    assert actual == expected
    assert not list((tmp_path / 'labels').rglob('*.arrow'))


@pytest.mark.parametrize('workers', [1, 2])
def test_label_batches_reject_duplicate_keys_across_blocks(
        case_runner, tmp_path, workers):
    frame = label_rows().sort_values(['ENTITY1', 'TIMESTAMP', 'SPLIT'])
    with pytest.raises(ValueError, match='Duplicate or unordered'):
        case_runner.label_identities(
            [frame.iloc[:3], frame.iloc[:3], frame.iloc[3:]], 'regression',
            tmp_path / 'labels', workers=workers)
    assert not list((tmp_path / 'labels').rglob('*.arrow'))


@pytest.mark.parametrize('workers', [1, 2])
def test_label_batches_reject_nonfinite_answers(case_runner, tmp_path, workers):
    frame = label_rows().sort_values(['ENTITY1', 'TIMESTAMP', 'SPLIT'])
    frame.loc[0, 'TARGET1'] = float('inf')
    with pytest.raises(ValueError, match='Nonfinite training label'):
        case_runner.label_identities([frame.iloc[:3], frame.iloc[3:]],
                                     'regression', tmp_path / 'labels',
                                     workers=workers)
    assert not list((tmp_path / 'labels').rglob('*.arrow'))


def test_saved_normal_payload_is_compared_only_in_candidate_mode(
        case_runner, future_targets, tmp_path):
    import json

    from kumoapi.json_serde import to_json_dict

    query, snapshot, plan, _, _, normal_rows = future_targets
    descriptor = tmp_path / 'normal.json'
    descriptor.write_text(
        json.dumps(
            dict(query=query.query, query_generation_plan=to_json_dict(plan))))
    frozen = tmp_path / 'normal.parquet'
    pq.write_table(normal_rows, frozen)
    spec = importlib.util.spec_from_file_location(
        'bootstrap_payload_helpers', SOURCE.with_name('nightly_check.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    normal = dict(scale='100M', task='temporal_link', arm='normal', seed=0,
                  descriptor=descriptor.name, normal_rows=[frozen.name])
    manifest = dict(
        cases=[normal], files={
            name: helpers.digest(tmp_path / name)
            for name in (descriptor.name, frozen.name)
        })
    _, _, proof = case_runner.generate_training_table(tmp_path, manifest,
                                                      normal, query, snapshot,
                                                      tmp_path / 'matching',
                                                      helpers=helpers)
    assert sum(stage['rows'] for stage in proof['normal_stages'].values()) == 6
    changed = normal_rows.to_pandas()
    changed.at[0, 'TARGET1'] = [10, 40]
    changed.to_parquet(frozen, index=False)
    manifest['files'][frozen.name] = helpers.digest(frozen)
    with pytest.raises(ValueError,
                       match='normal stage counts or labels differ'):
        case_runner.generate_training_table(tmp_path, manifest, normal, query,
                                            snapshot, tmp_path / 'changed',
                                            helpers=helpers)
    _, _, reference_proof = case_runner.generate_training_table(
        tmp_path, manifest, normal, query, snapshot, tmp_path / 'reference',
        helpers=helpers, bootstrap=True)
    assert reference_proof['normal_stages'] == proof['normal_stages']
    assert reference_proof['answer_policy'] == 'record_reference'
