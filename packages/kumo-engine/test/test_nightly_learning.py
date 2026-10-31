# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject a nonlearning model and a scrambled-answer leak on CPU."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_learning', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


def rows(task='binary', normal=.8, control=.5, baseline=.5):
    return [
        dict(task=task, arm=arm, score=score, baseline=baseline)
        for arm, score in [('normal', normal), ('control', control)]
    ]


def test_learning_gate_accepts_learning_and_chance_control():
    result = nightly.check_learning(rows())
    assert result['status'] == 'PASS'
    assert result['tasks']['binary']['normal_gain'] == pytest.approx(.3)


@pytest.mark.parametrize('normal,control', [(.5, .5), (.8, .8), (.8, .2),
                                            (float('nan'), .5), (.8, None)])
def test_learning_gate_rejects_nonlearning_leak_and_missing_scores(
        normal, control):
    assert nightly.check_learning(rows(normal=normal,
                                       control=control))['status'] == 'FAIL'


def test_learning_gate_uses_lower_error_for_regression():
    assert nightly.check_learning(rows('regression', .7, 1.,
                                       1.))['status'] == 'PASS'
    assert nightly.check_learning(rows('regression', 1., 1.,
                                       1.))['status'] == 'FAIL'
    assert nightly.check_learning(rows('regression', .7, .3,
                                       1.))['status'] == 'FAIL'


def test_learning_margin_scales_with_a_rare_link_baseline():
    result = nightly.check_learning(rows('static_link', .012, .01, .01))
    assert result['status'] == 'PASS'
    assert result['tasks']['static_link']['normal_margin'] == pytest.approx(
        .001)


@pytest.mark.parametrize('normal,control,status', [(.96, .95, 'PASS'),
                                                   (.96, .96, 'FAIL'),
                                                   (.95, .95, 'FAIL')])
def test_high_baseline_keeps_a_reachable_learning_margin(
        normal, control, status):
    result = nightly.check_learning(
        rows('multilabel_ranking', normal, control, .95))
    assert result['status'] == status
    assert result['tasks']['multilabel_ranking'][
        'normal_margin'] == pytest.approx(.005)
    assert result['tasks']['multilabel_ranking'][
        'control_tolerance'] == pytest.approx(.005)


@pytest.mark.parametrize('normal,baseline', [(1.05, .95), (1.5, 1.1)])
def test_learning_gate_rejects_scores_outside_the_metric_range(
        normal, baseline):
    assert nightly.check_learning(
        rows(normal=normal, control=baseline,
             baseline=baseline))['status'] == 'FAIL'


def test_learning_gate_rejects_missing_or_duplicate_controls():
    assert nightly.check_learning(rows()[:1])['status'] == 'FAIL'
    assert nightly.check_learning(rows() + rows()[1:])['status'] == 'FAIL'


def test_rank_changes_fail_even_when_scores_match():
    before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.7, .3]])
    after = dict(ranked_ids=[[2, 1]], ranked_scores=[[.3, .7]])
    with pytest.raises(ValueError, match='rank'):
        nightly.ranked_difference(before, after)


def test_runtime_accepts_package_count_and_driver_changes_but_keeps_gpu_class():
    nightly.check_dependencies({
        'torch': '2.14.1',
        'extra': '1.0'
    }, {'torch': '2.14.1'})
    with pytest.raises(ValueError):
        nightly.check_dependencies({'torch': '2.14.1'}, {'torch': '2.15.0'})
    reference = dict(reference_runtime=dict(reference_hardware=dict(
        gpu_class='RTX PRO 6000')))
    runtime = dict(gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition',
                   gpu_count=1, architecture='x86_64', driver='new-version',
                   torch='new-version', cuda='new-version',
                   gpu_memory_bytes=96 * 1024**3)
    nightly.check_reference_runtime(reference, runtime, False)
    with pytest.raises(ValueError):
        nightly.check_reference_runtime(reference, dict(runtime,
                                                        gpu='Tesla T4'), False)


def test_completed_training_allows_partial_batches_and_actual_early_stopping():
    work = dict(
        core_fit_attempts=1, fits=1, completed_fits=1, optimizer_steps=80,
        training_batches=80, training_epochs=[
            dict(batches=20, batch_sizes=[64] * 19 + [32]) for _ in range(4)
        ], early_stopping=dict(stopped_epoch=3, patience=2, wait_count=2))
    nightly.check_training(work, 40)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, optimizer_steps=0), 40)
    with pytest.raises(ValueError):
        nightly.check_training(dict(work, early_stopping={}), 40)


@pytest.mark.parametrize('method', ['predict_v2', 'test'])
def test_nightly_keeps_product_oom_recovery(monkeypatch, tmp_path, method):
    from types import SimpleNamespace

    import torch
    from kumo_core.train.trainer import Trainer, handle_gpu_oom

    batches = []

    @handle_gpu_oom
    def predict(self):
        size = self.datamodule.eval_loader_kwargs['batch_size']
        batches.append(size)
        if size > 32:
            raise torch.cuda.OutOfMemoryError('injected allocation failure')
        return 'complete prediction'

    monkeypatch.setattr(Trainer, method, predict)
    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(eval_loader_kwargs={'batch_size': 64}),
        pl_model=SimpleNamespace(reset_embedding_hook=lambda: None))
    state = dict(core_fit_attempts=0, native_ingests=[], pseudo_options={},
                 ledger_path=str(tmp_path / 'ledger.json'))
    with nightly.native_calls(state):
        assert getattr(Trainer, method)(trainer) == 'complete prediction'
    assert batches == [64, 32]
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 64


@pytest.mark.parametrize('damage', [None, 'no_learning', 'leaked_control'])
def test_final_comparison_enforces_learning_on_retained_results(
        tmp_path, damage):
    """Exercise the full comparison reader and writer, with no GPU execution claim."""
    from types import SimpleNamespace

    results, inputs = tmp_path / 'results', tmp_path / 'results/input-metadata'
    nightly.save(inputs / 'profile.json', {'torch': '2.14.1'})
    manifest = dict(
        candidate_dependencies='profile.json',
        reference_dependencies='profile.json', cases=[],
        files={'profile.json': nightly.digest(inputs / 'profile.json')})
    reference_rows, runs = [], []
    for task in nightly.smoke.TASKS:
        for arm in ('normal', 'control'):
            name = task + '-' + arm
            case = dict(task=task, arm=arm, seed=0,
                        descriptor=name + '/table.json',
                        prediction_input=name + '/full.parquet',
                        partial_input=name + '/partial.parquet',
                        requested_plan_sha256='1' * 64)
            manifest['cases'].append(case)
            manifest['files'].update({
                case[k]: '2' * 64
                for k in ('descriptor', 'prediction_input', 'partial_input')
            })
            baseline = 1. if task in ('regression', 'forecasting') else .5
            score = baseline if arm == 'control' else .7 if baseline == 1 else .8
            if task == 'binary' and damage == 'no_learning' and arm == 'normal':
                score = .5
            if task == 'binary' and damage == 'leaked_control' and arm == 'control':
                score = .8
            work = dict(
                core_fit_attempts=1, fits=1, completed_fits=1,
                optimizer_steps=200, training_batches=200, training_epochs=[
                    dict(batches=20, batch_sizes=[64] * 20) for _ in range(10)
                ], restored_tensors=1, reload_checks=1)
            work.update({
                key: [dict(status='PASS', steps=200, cuda_batches=200)]
                for key in ('core_fit_ledger', 'fit_ledger')
            })
            directory = results / name
            names = [
                'checkpoint', 'requested-plan.json', 'generation-plan.json',
                'selected-model-plan.json', 'resolved_model_plan.json',
                'trainer_config.json', 'feature_schema.json',
                'runtime_manifest.json', 'metadata.json', 'encoders.json',
                'target_encoder.json', 'core-fits.json'
            ]
            predictions = {}
            for stage in nightly.PREDICTIONS:
                names.extend([
                    stage + '-raw.parquet',
                    f'prediction-inputs/{stage}.parquet', stage +
                    ('-ranked.json' if task.endswith('link') else '.parquet')
                ])
                predictions[stage] = dict(files=[stage + '-raw.parquet'])
            for path in names:
                nightly.save(directory / path, {})
            nightly.save(
                directory / 'trainer_config.json',
                dict(refit_cfg=dict(refitted_full=False,
                                    refitted_trainval=False)))
            artifacts = {
                path: nightly.digest(directory / path)
                for path in names
            }
            row = dict(
                task=task, arm=arm, seed=0, status='MEASURED', physical_fits=1,
                fit_id=name, checkpoint_sha256=artifacts['checkpoint'],
                checkpoint_artifact='checkpoint', artifacts=artifacts,
                selected_checkpoint_steps=200, predictions=predictions,
                prediction_batches={stage: 1
                                    for stage in nightly.PREDICTIONS},
                work=work, input_signature=nightly.input_signature(
                    manifest, case), metric=nightly.smoke.TASKS[task]['metric'],
                score=score, seconds=1, differences={})
            nightly.save(directory / 'result.json', row)
            runs.append(row)
            reference_rows.append(
                dict(row, baseline=baseline, fit_id='reference-' + name))
    nightly.save(inputs / 'manifest.json', manifest)
    input_hash = nightly.digest(inputs / 'manifest.json')
    reference = dict(
        format='kumo-engine-nightly-reference-v1', version='1.0.0',
        wheel_sha256=nightly.REFERENCE_WHEEL, cases=reference_rows,
        input_manifest_sha256=input_hash, reference_runtime=dict(
            reference_hardware=dict(gpu_class='RTX PRO 6000')))
    nightly.save(results / 'reference.json', reference)
    reference_hash = nightly.digest(results / 'reference.json')
    runtime = dict(gpu='NVIDIA RTX PRO 6000', gpu_count=1,
                   architecture='x86_64', dependencies={'torch': '2.14.1'},
                   version='candidate', wheel_sha256='3' * 64)
    nightly.save(results / 'setup/wheel.json',
                 dict(version='candidate', sha256='3' * 64, installed_files=1))
    nightly.save(
        results / 'setup/build.json',
        dict(version='candidate', wheel_sha256='3' * 64, revision='4' * 40,
             tree='5' * 40))
    nightly.save(
        results / 'nightly-result.json',
        dict(format='kumo-engine-nightly-result-v1', status='MEASURED',
             input_manifest_sha256=input_hash, reference_sha256=reference_hash,
             reference_only=False, expected_runs=16, completed_runs=16,
             runs=runs, runtime=runtime, runner_sha256=nightly.digest(SOURCE),
             helper_sha256=nightly.digest(nightly.HELPER), seconds=16))
    args = SimpleNamespace(input_root=inputs, input_sha256=input_hash,
                           reference=results / 'reference.json',
                           reference_sha256=reference_hash,
                           reference_only=False, results_dir=results,
                           output_dir=tmp_path / 'summary')
    if damage:
        with pytest.raises(ValueError, match='learning'):
            nightly.compare(args)
    else:
        nightly.compare(args)
    report = nightly.read(args.output_dir / 'comparison.json')
    assert report['status'] == ('FAIL' if damage else 'PASS')
    assert report['learning']['tasks']['binary']['status'] == report['status']


def test_installed_wheel_check_rejects_changed_runtime_files(tmp_path):
    import importlib.metadata
    import zipfile

    dist_info = tmp_path / 'example-1.0.dist-info'
    dist_info.mkdir()
    metadata = b'Name: example\nVersion: 1.0\n'
    (dist_info / 'METADATA').write_bytes(metadata)
    source = tmp_path / 'example.py'
    source.write_bytes(b'VALUE = 1\n')
    wheel = tmp_path / 'example-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('example.py', source.read_bytes())
        archive.writestr('example-1.0.dist-info/METADATA', metadata)
        archive.writestr('example-1.0.dist-info/RECORD', b'wheel record')
    distribution = importlib.metadata.Distribution.at(dist_info)
    check = nightly.smoke.verify_installed_files
    assert check(wheel, distribution) == 2
    source.write_bytes(b'VALUE = 2\n')
    with pytest.raises(ValueError):
        check(wheel, distribution)
    source.unlink()
    with pytest.raises(ValueError):
        check(wheel, distribution)
