"""Native gate for binary walk evidence and the replay precision record.

Trains tiny CPU models on a static binary churn fixture, once with the
default biases and once with `BiasOptions(gnn=False, head=False)`, then
checks: the effective architecture, bundle records and reload; exact
prediction parity with and without evidence; class orientation against an
independent recomputation and a semantic oracle; sign folding for the other
class; bundle validation, including relation steps and `mean` denominators
against the retained trimmed graphs; offline replay in-process and in a
fresh process under the recorded precision; and the refusal of methods that
are not implemented. A regression phase checks that a fresh-process replay of a
`medium` capture whose 32-channel linears see more than 8 rows passes with
the recorded precision applied and fails when the record is overridden.

Batch-size differences are measured and reported, never asserted. Passing
this gate is accounting and plumbing evidence, not explanation faithfulness.

  runtime/bin/heavy runtime/bin/epy scripts/xai/walk_binary_native_gate.py \
      --development --output <new directory> [--methods ID,ID] [--gamma 0 0.25]
"""
import argparse
import copy
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import kumo.engine as kumo
from kumo.engine.evidence import (
    EvidenceConfig,
    PredictionEvidenceBundle,
    WalkAttributionConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'packages' /
                       'kumo-engine' / 'test'))
from walk_binary_fixtures import (  # noqa: E402
    BINARY_QUERY,
    REGRESSION_QUERY,
    binary_graph,
    binary_plan,
    regression_graph,
    regression_plan,
    write_binary_tables,
    write_regression_tables,
)

LEGACY = 'hetero_lrp_gamma_reference_v1'
DEGREE_T = 'hetero_gnn_gi_degree_t_v1'
DEFAULT_METHODS = (LEGACY, 'hetero_gnn_lrp_weight_gamma_v1', DEGREE_T)
COLUMNS = ['ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB']

REPLAY = r'''
import json, sys
from pathlib import Path
import torch
import kumo.engine as kumo
from kumo.engine.evidence import PredictionEvidenceBundle, WalkAttributionConfig
root, evidence, model, output, config = (Path(sys.argv[1]), sys.argv[2], sys.argv[3],
                                         sys.argv[4], json.loads(sys.argv[5]))
kumo.configure(scratch_dir=root / ('scratch-' + output))
before = torch.get_float32_matmul_precision()
bundle = PredictionEvidenceBundle.load(root / evidence)
out = bundle.attribute_walks(model=kumo.Trainer.load(root / model, accelerator='cpu'),
                             config=WalkAttributionConfig(**config), output=root / output)
print(json.dumps({'before': before, 'after': torch.get_float32_matmul_precision(),
                  'statuses': [r['status'] for r in out.iter_rows()]}))
'''


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def fresh_replay(root, evidence, model, output, config):
    process = subprocess.run(
        [sys.executable, '-c', REPLAY, str(root), evidence, model, output,
         json.dumps(config.to_dict())], capture_output=True, text=True)
    lines = [line for line in process.stdout.splitlines() if line.startswith('{')]
    check(process.returncode == 0 and lines,
          f'fresh replay failed: {process.stderr[-2000:]}')
    return json.loads(lines[-1])


def auroc(scores, labels):
    """Mann-Whitney AUROC with ties counted as one half."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positive, negative = scores[labels], scores[~labels]
    greater = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def frame(result):
    return result.frame[COLUMNS].sort_values('ENTITY').reset_index(drop=True)


def comparable(row):
    """A walk row without the fields that legitimately differ on replay
    (origin, numerics and the replay differences). Method `diagnostics` are
    computed from the captured forward alone, so they must agree too (D14c)."""
    keep = ('walks', 'sources', 'ledger', 'score', 'output', 'coverage',
            'exactness', 'architecture', 'operations', 'seed', 'method_settings',
            'diagnostics')
    return {key: row[key] for key in keep}


def encoder_boundary(trainer, bundle, records):
    """`module.encoder` on each reconstructed batch, under the recorded mode,
    reproduces every source `input_value` bitwise (what P5 consumes)."""
    import tempfile

    from kumo_core.artifacts import stage_model_bundle
    from kumo_core.explain.local_sensitivity import load_scoring_module
    from kumo_core.explain.perturbation import reconstruct_batch
    from kumo_core.explain.walk_graph import float32_matmul_precision
    provenance = bundle.manifest['native_provenance']
    result = trainer.result
    check(result is not None, 'a loaded trainer has a result')
    module: Any  # a loaded node scorer, duck-typed
    with tempfile.TemporaryDirectory(prefix='encoder-boundary-') as temp:
        model_dir = stage_model_bundle(result.artifact_paths, Path(temp) / 'model')
        module = load_scoring_module(str(model_dir), provenance)
    checked = 0
    with torch.no_grad(), float32_matmul_precision(provenance['numerics']['precision_mode']):
        for record in records:
            batch = reconstruct_batch(bundle.scoring_graph(record['prediction_id']))
            encoded = module.encoder(batch.tf_dict, batch.batch_dict,
                                     batch.collect('time', allow_empty=True),
                                     batch[module.train_table_name].get('time'))
            for source in record['walk_attribution']['sources']:
                actual = encoded[source['table']][source['occurrence']].tolist()
                check(actual == source['input_value'], 'encoder boundary is bitwise')
                checked += 1
    return {'sources_checked': checked, 'bitwise_equal': True,
            'mode': provenance['numerics']['precision_mode']}


def retained_denominators(bundle, records):
    """Contract 4.10 on the real trimmed graphs: bundle validation binds every
    displayed relation step to an edge of its layer and every `mean` step to
    its destination's retained in-degree; a shifted denominator is rejected."""
    from kumo_core.explain.walk_schema import WalkSchemaError, validate_v2_row
    rows = steps = largest = rejected = 0
    for record in records:
        row = copy.deepcopy(record['walk_attribution'])  # the records stay intact
        sample = bundle.scoring_graph(record['prediction_id'])
        mean = [step for walk in row['walks'] for step in walk['steps']
                if step['kind'] == 'relation' and step['operator'] == 'mean']
        if not mean:
            continue
        rows += 1
        steps += len(mean)
        shifted = max(mean, key=lambda step: step['denominator'])
        largest = max(largest, shifted['denominator'])
        shifted['denominator'] += 1
        try:
            validate_v2_row(row, sample)
        except WalkSchemaError:
            rejected += 1
    check(rows > 0 and largest > 1 and rejected == rows,
          'a shifted mean denominator is rejected on every row')
    return {'rows_with_mean_steps': rows, 'mean_steps': steps,
            'max_denominator': largest, 'rows_rejecting_a_shifted_denominator': rejected}


def binary_phase(root, data, label, bias_options, methods, gammas):
    started = time.monotonic()
    report = {'bias_options': None if bias_options is None else bias_options.to_dict()}
    connector = kumo.FileConnector(root_dir=str(data))
    graph = binary_graph(kumo, connector)
    snapshot = graph.snapshot()
    table = kumo.PredictiveQuery(graph=graph, query=BINARY_QUERY).generate_training_table(
        snapshot)
    trainer = kumo.Trainer(binary_plan(epochs=6), accelerator='cpu',
                           bias_options=bias_options)
    fitted = trainer.fit(table, snapshot=snapshot)
    saved = root / f'model-{label}'
    shutil.copytree(fitted.bundle_dir, saved)

    # Effective architecture from the trained modules, never from options.
    config = json.loads((saved / 'model' / 'config.json').read_text())
    state = torch.load(saved / 'model' / 'checkpoint', map_location='cpu',
                       weights_only=False)['state_dict']
    biases = sorted(k for k in state if k.startswith(('model.', 'head.'))
                    and k.endswith('.bias'))
    spec = json.loads((saved / 'training-spec.json').read_text())
    expect_bias = bias_options is None
    check(config['model']['bias'] is expect_bias and config['head']['bias'] is expect_bias,
          'model/config.json bias flags')
    check(bool(biases) is expect_bias, 'checkpoint bias tensors')
    check(('bias_options' in spec) is (not expect_bias), 'training-spec bias_options')
    restored = kumo.Trainer.load(saved, accelerator='cpu')
    check(restored.bias_options == bias_options, 'reloaded bias options')
    report['architecture'] = {
        'gnn_bias': config['model']['bias'], 'head_bias': config['head']['bias'],
        'channels': config['model']['channels'], 'mp_layers': config['model']['num_mp_layers'],
        'aggr': config['model']['aggr'], 'act': config['model']['act'],
        'norm': config['model']['norm'], 'head_layers': config['head']['num_layers'],
        'bias_tensors': len(biases), 'spec_bias_options': spec.get('bias_options')}

    users = pd.read_parquet(data / 'users.parquet')
    pd.DataFrame({'ENTITY1': users['id']}).to_parquet(data / f'all-{label}.parquet')
    pd.DataFrame({'ENTITY1': users['id'][:24]}).to_parquet(data / f'score-{label}.parquet')
    every, scored = connector[f'all-{label}.parquet'], connector[f'score-{label}.parquet']

    # Semantic orientation oracle (training users, not an accuracy claim).
    batched = frame(restored.predict(snapshot=snapshot, entities=every))
    truth = users.set_index('id').loc[batched['ENTITY'], 'churn'].to_numpy()
    report['orientation_auroc_true_prob_vs_churn'] = auroc(batched['True_PROB'], truth)
    check(report['orientation_auroc_true_prob_vs_churn'] >= 0.9,
          'True_PROB must rank churn == True users first')

    ordinary = frame(restored.predict(snapshot=snapshot, entities=scored, batch_size=1))
    config_true = WalkAttributionConfig(selected_class=True, feature_vectors=True,
                                        top_k=10000, max_walks=100000)
    explained = restored.predict(
        snapshot=snapshot, entities=scored, batch_size=1,
        evidence=EvidenceConfig(numerical_sensitivity=None, walk_attribution=config_true))
    pd.testing.assert_frame_equal(ordinary, frame(explained), check_exact=True)
    precision = torch.get_float32_matmul_precision()
    report['capture_precision_mode'] = precision
    default = frame(restored.predict(snapshot=snapshot, entities=scored))
    report['batch_size_effect'] = {
        'max_abs_true_prob_difference_batch1_vs_default': float(
            (default['True_PROB'] - ordinary['True_PROB']).abs().max()),
        'rows_differing': int((default['True_PROB'] != ordinary['True_PROB']).sum()),
        'note': 'measured and reported, not asserted (contract 9.3)'}
    bundle = explained.evidence.export(root / f'evidence-{label}')
    report['bundle_validation'] = PredictionEvidenceBundle.load(bundle.root).validate()
    records = list(bundle.iter_numerical_evidence())
    check(len(records) == 24, 'one record per scored user')
    report['encoder_boundary'] = encoder_boundary(restored, bundle, records)
    report['retained_denominators'] = retained_denominators(bundle, records)

    # Orientation, selection and calibration on every row, recomputed from
    # checkpoint tensors independently of the loaded module.
    weight, bias = state['calibrator.weight_and_bias'].reshape(2)
    fitted = bool(state['calibrator._extra_state']['is_fitted'])
    rows, emitted = [], frame(explained).set_index('ENTITY')
    for record in records:
        row = record['walk_attribution']
        check(row['status'] == 'available', f'walk row status {row["status"]}')
        binary = row['output']['binary']
        z = row['output']['head_output']
        check(binary['encoded_positive_label'] is True and binary['target_encoder_reverse'] is True
              and binary['logit_sign'] == 1 and row['score'] == z, 'class orientation')
        true_prob = float(emitted.loc[record['entity'], 'True_PROB'])
        check(binary['emitted_encoded_positive_probability'] == true_prob
              == record['prediction'], 'True_PROB binding')
        logit = (max(float(weight), 1e-3) * z + float(bias)) if fitted else z
        check(abs(1 / (1 + math.exp(-logit)) - true_prob) <= 1e-6,
              'True_PROB recomputed from checkpoint calibrator')
        check(binary['calibration']['fitted'] is fitted, 'calibration record')
        rows.append(row)
    report['rows'] = len(rows)
    report['walks_per_row'] = sorted({r['coverage']['total_walks'] for r in rows})
    report['max_abs_accounting_error'] = max(abs(r['ledger']['accounting_error']) for r in rows)
    report['bias_relevance_by_kind_max_abs'] = {
        kind: max(abs(r['ledger']['bias_relevance_by_kind'][kind]) for r in rows)
        for kind in ('preprocessing', 'message_passing', 'head')}
    if bias_options is not None:
        check(all(not r['architecture']['gnn_bias'] and not r['architecture']['head_bias']
                  for r in rows), 'bias-free architecture record')
        check(report['bias_relevance_by_kind_max_abs'] == {
            'preprocessing': 0.0, 'message_passing': 0.0, 'head': 0.0},
            'no bias relevance without biases')

    # The other class: the explained coordinate is -z, gamma-0 LRP relevance
    # is exactly negated, and the selected probability is False_PROB.
    config_false = WalkAttributionConfig(selected_class=False, top_k=10000,
                                         max_walks=100000)
    negative = restored.predict(
        snapshot=snapshot, entities=scored, batch_size=1,
        evidence=EvidenceConfig(numerical_sensitivity=None, walk_attribution=config_false))
    negative_bundle = negative.evidence.export(root / f'evidence-{label}-false')
    negative_records = {r['entity']: r for r in negative_bundle.iter_numerical_evidence()}
    negative_frame = frame(negative).set_index('ENTITY')
    for record in records:
        positive_row = record['walk_attribution']
        row = negative_records[record['entity']]['walk_attribution']
        check(row['score'] == -positive_row['score'], 'score == -z')
        check([w['relevance'] for w in row['walks']]
              == [-w['relevance'] for w in positive_row['walks']], 'gamma-0 folding negates')
        check(row['output']['binary']['emitted_probability_selected']
              == float(negative_frame.loc[record['entity'], 'False_PROB']), 'False_PROB binding')

    # Offline replay: in-process from `highest` and in a fresh process.
    torch.set_float32_matmul_precision('highest')
    offline = bundle.attribute_walks(model=restored, config=config_true,
                                     output=root / f'offline-{label}')
    check(torch.get_float32_matmul_precision() == 'highest', 'replay restored the mode')
    replayed = {r['prediction_id']: r for r in offline.iter_rows()}
    for record in records:
        row = replayed[record['prediction_id']]
        check(row['status'] == 'available' and row['numerics']['precision_mode'] == precision
              and row['numerics']['replay_precision_mode'] == precision, 'replay numerics')
        check(comparable(row) == comparable(record['walk_attribution']), 'replay equality')
    fresh = fresh_replay(root, f'evidence-{label}', f'model-{label}', f'fresh-{label}',
                         config_true)
    check(fresh['before'] == 'highest' and fresh['after'] == 'highest'
          and set(fresh['statuses']) == {'available'}, f'fresh replay {fresh}')
    fresh_rows = {r['prediction_id']: r for r in
                  kumo.WalkAttributionBundle.load(root / f'fresh-{label}').iter_rows()}
    check(all(comparable(fresh_rows[r['prediction_id']]) == comparable(r['walk_attribution'])
              for r in records), 'fresh replay equality')
    report['offline'] = {'in_process': 'available', 'fresh_process': fresh}

    # Every requested method and gamma, in `record` mode: rows are produced
    # or refused (never another rule), the bundle validates, predictions are
    # unchanged, and produced rows replay offline to the same values.
    report['methods'] = {}
    for method in methods:
        for gamma in gammas:
            if method == DEGREE_T and gamma != 0:
                continue  # GNN-GI has no gamma rule (configuration error).
            key = f'{method}@gamma={gamma:g}'
            config = WalkAttributionConfig(
                method=method, gamma=gamma, selected_class=True, feature_vectors=True,
                bags=method == DEGREE_T, top_k=10000, max_walks=100000)
            recorded = restored.predict(
                snapshot=snapshot, entities=scored, batch_size=1,
                evidence=EvidenceConfig(numerical_sensitivity=None, walk_attribution=config,
                                        on_unsupported='record'))
            pd.testing.assert_frame_equal(ordinary, frame(recorded), check_exact=True)
            directory = f'evidence-{label}-{len(report["methods"])}'
            method_bundle = recorded.evidence.export(root / directory)
            PredictionEvidenceBundle.load(method_bundle.root).validate()
            rows = [r['walk_attribution'] for r in method_bundle.iter_numerical_evidence()]
            entry: dict[str, Any] = {'record_statuses': sorted({r['status'] for r in rows})}
            if set(entry['record_statuses']) <= {'available', 'incomplete'}:
                offline = method_bundle.attribute_walks(
                    model=restored, config=config, output=root / f'offline-{directory}')
                derived = {r['prediction_id']: r for r in offline.iter_rows()}
                entry['offline_equal'] = all(
                    comparable(derived[r['prediction_id']]) == comparable(r) for r in rows)
                check(entry['offline_equal'], f'{key}: offline replay equality')
                entry['max_abs_accounting_error'] = max(
                    abs(r['ledger']['accounting_error']) for r in rows)
                entry['decompositions'] = sorted({r['exactness']['decomposition'] for r in rows})
                entry['degree_t_claims'] = sum(r['exactness']['degree_t_claim'] for r in rows)
            else:
                try:
                    restored.predict(snapshot=snapshot, entities=scored, batch_size=1,
                                     evidence=EvidenceConfig(numerical_sensitivity=None,
                                                             walk_attribution=config))
                    entry['raise_mode'] = None
                except ValueError as error:
                    entry['raise_mode'] = f'{type(error).__name__}: {error}'
                check(entry['raise_mode'] is not None, f'{key}: raise mode must refuse')
            report['methods'][key] = entry
    report['seconds'] = round(time.monotonic() - started, 1)
    return report


def regression_phase(root):
    """Contract C4: replay applies the recorded precision for v1 and v2."""
    from kumo.engine.evidence.bundle import _json, _read, _seal
    data = root / 'regression-data'
    data.mkdir()
    write_regression_tables(data)
    connector = kumo.FileConnector(root_dir=str(data))
    graph = regression_graph(kumo, connector)
    snapshot = graph.snapshot()
    table = kumo.PredictiveQuery(graph=graph, query=REGRESSION_QUERY).generate_training_table(
        snapshot)
    fitted = kumo.Trainer(regression_plan(), accelerator='cpu').fit(table, snapshot=snapshot)
    shutil.copytree(fitted.bundle_dir, root / 'model-regression')
    restored = kumo.Trainer.load(root / 'model-regression', accelerator='cpu')
    pd.DataFrame({'ENTITY1': np.arange(4)}).to_parquet(data / 'score.parquet')
    report = {}
    for name, config in (('v1', WalkAttributionConfig(top_k=10000, max_walks=100000)),
                         ('v2', WalkAttributionConfig(feature_vectors=True, top_k=10000,
                                                      max_walks=100000))):
        explained = restored.predict(
            snapshot=snapshot, entities=connector['score.parquet'], batch_size=1,
            evidence=EvidenceConfig(numerical_sensitivity=None, walk_attribution=config))
        bundle = explained.evidence.export(root / f'evidence-regression-{name}')
        native = bundle.manifest['native_provenance']['numerics']
        rows = [r['walk_attribution'] for r in bundle.iter_numerical_evidence()]
        sample = bundle.scoring_graph(rows[0]['prediction_id'])
        fresh = fresh_replay(root, f'evidence-regression-{name}', 'model-regression',
                             f'fresh-regression-{name}', config)
        check(set(fresh['statuses']) == {'available'}, f'{name} fresh replay {fresh}')
        # Override the record: the same replay at `highest` must fail loudly.
        tampered = root / f'evidence-regression-{name}-highest'
        shutil.copytree(bundle.root, tampered)
        local = _read(tampered / 'xai/local/manifest.json')
        local['numerics']['precision_mode'] = 'highest'
        local['numerics']['mkldnn_matmul_fp32_precision'] = 'ieee'
        (tampered / 'xai/local/manifest.json').write_bytes(_json(local))
        manifest = _read(tampered / 'manifest.json')
        manifest['native_provenance'] = local
        _seal(tampered, manifest)
        overridden = fresh_replay(root, tampered.name, 'model-regression',
                                  f'fresh-regression-{name}-highest', config)
        report[name] = {
            'native_numerics': native, 'native_statuses': sorted({r['status'] for r in rows}),
            'max_rows_per_table': max(n['num_nodes'] for n in sample['nodes'].values()),
            'fresh_replay_recorded_mode': fresh['statuses'],
            'fresh_replay_overridden_highest': overridden['statuses']}
    medium_differs = all(report[name]['native_numerics']['mkldnn_matmul_fp32_precision']
                         == 'bf16' for name in ('v1', 'v2'))
    report['host_bf16_matmul'] = medium_differs
    if medium_differs:
        for name in ('v1', 'v2'):
            check('failed' in report[name]['fresh_replay_overridden_highest'],
                  f'{name}: replay at an overridden precision must fail')
    return report


def run(root, methods, gammas):
    torch.set_num_threads(2)
    kumo.configure(scratch_dir=root / 'scratch')
    data = root / 'data'
    data.mkdir()
    tables = write_binary_tables(data)
    report = {'tables': tables, 'methods_requested': list(methods),
              'gammas_requested': list(gammas)}
    report['default_bias'] = binary_phase(root, data, 'default', None, methods, gammas)
    report['bias_free'] = binary_phase(root, data, 'bias-free',
                                       kumo.BiasOptions(gnn=False, head=False), methods,
                                       gammas)
    report['regression_replay_precision'] = regression_phase(root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS),
                        help='comma-separated walk method IDs to exercise')
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.0, 0.25],
                        help='LRP gamma values per method (GNN-GI runs at 0 only)')
    args = parser.parse_args()
    methods = tuple(m for m in args.methods.split(',') if m)
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.development:
        if args.wheel is None:
            parser.error('--wheel required for installed acceptance')
        with zipfile.ZipFile(args.wheel) as archive:
            for name in ('kumo.engine.evidence.walks',
                         'kumo_core.explain.walk_attribution',
                         'kumo_core.explain.walk_graph',
                         'kumo_core.explain.walk_schema'):
                module_file = importlib.import_module(name).__file__
                assert module_file is not None
                path = Path(module_file).resolve()
                assert path.is_relative_to(Path(sys.prefix).resolve())
                assert path.read_bytes() == archive.read(
                    name.replace('.', '/') + '.py')
    started = time.monotonic()
    report = run(args.output.resolve(), methods, tuple(args.gamma))
    report.update(
        seconds=round(time.monotonic() - started, 1),
        installed_wheel_acceptance=not args.development,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='CPU, one scorer, one seed per batch; plumbing and accounting '
              'evidence only, not explanation faithfulness')
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
