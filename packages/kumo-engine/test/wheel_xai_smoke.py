"""Exercise the installed Engine, using a tiny disposable saved regression.

Source development can set PYTHONPATH explicitly. Release evidence requires an
unset PYTHONPATH and imports outside the Engine checkout.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from kumo_core.explain.local_sensitivity import LocalSensitivity
from kumo_core.train.node_gnn_module import NodeGNNLightningModule
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo


def run(root: Path, *, temporal=False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    kumo.configure(scratch_dir=root / "scratch")
    data = root / "data"
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    quantity = rng.uniform(1, 50, 96)
    pd.DataFrame({
        "id": np.arange(96),
        "quantity": quantity,
        "target": 3 * quantity + rng.normal(0, 1, 96)
    }).to_parquet(data / "entities.parquet", index=False)
    connector = kumo.FileConnector(root_dir=str(data))
    table = kumo.Table.from_source_table(connector["entities.parquet"],
                                         primary_key="id").infer_metadata()
    table = table.with_column("quantity", stype=Stype.numerical)
    table = table.with_column("target", stype=Stype.numerical)
    if temporal:
        days = pd.date_range('2024-01-01', periods=120)
        events = pd.DataFrame({
            'event_id':
            np.arange(480),
            'entity_id':
            np.tile(np.arange(4), 120),
            'date':
            np.repeat(days + pd.Timedelta(hours=23, minutes=59, seconds=59), 4),
            'units':
            rng.uniform(10, 50, 480),
        })
        events.loc[len(events)] = [480, 0, pd.Timestamp('2024-04-29'), 31.]
        events.to_parquet(data / 'events.parquet', index=False)
        event_table = kumo.Table.from_source_table(
            connector['events.parquet'], primary_key='event_id',
            time_column='date').infer_metadata().with_column(
                'entity_id',
                stype=Stype.ID).with_column('units', stype=Stype.numerical)
        graph = kumo.Graph(
            tables={
                'entities': table,
                'events': event_table
            }, edges=[
                kumo.Edge(src_table='events', fkey='entity_id',
                          dst_table='entities')
            ])
        pql = 'PREDICT SUM(events.units, 0, 7, days) FOR EACH entities.id'
    else:
        graph = kumo.Graph(tables={"entities": table}, edges=[])
        pql = "PREDICT entities.target FOR EACH entities.id"
    snapshot = graph.snapshot()
    query = kumo.PredictiveQuery(graph=graph, query=pql)
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.training_job.manual_seed = 42
    plan.optimization.loss = ["mse"]
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    plan.optimization.batch_size = [16]
    plan.neighbor_sampling.num_neighbors = [[8]]
    plan.model_architecture.channels = [16]
    trained = kumo.Trainer(plan, accelerator="cpu").fit(
        query.generate_training_table(snapshot), snapshot=snapshot)
    bundle = root / "saved-model"
    shutil.copytree(trained.bundle_dir, bundle)
    restored = kumo.Trainer.load(bundle, accelerator="cpu")
    scoring_rows = {"ENTITY1": [0, 1, 2, 3]}
    if temporal:
        scoring_rows['TIMESTAMP'] = [datetime(2024, 4, 29)] * 4
    pd.DataFrame(scoring_rows).to_parquet(data / "score.parquet")
    source = connector["score.parquet"]
    ordinary = restored.predict(snapshot=snapshot, entities=source,
                                batch_size=1)
    assert not ordinary.explanations.local_availability.available
    counters = {"forward": 0, "autograd": 0}
    captured = []
    forward = NodeGNNLightningModule.predict_step
    gradient = torch.autograd.grad
    cells = LocalSensitivity._cells

    def count_forward(module, *args, **kwargs):
        counters["forward"] += 1
        return forward(module, *args, **kwargs)

    def count_grad(*args, **kwargs):
        counters["autograd"] += 1
        return gradient(*args, **kwargs)

    def capture(collector, module, batch, output):
        result = cells(collector, module, batch, output)
        if not captured:
            captured.append((module, batch, result))
        return result

    with patch.object(NodeGNNLightningModule, "predict_step", count_forward), \
         patch.object(torch.autograd, "grad", count_grad), \
         patch.object(LocalSensitivity, "_cells", capture):
        local = restored.predict(snapshot=snapshot, entities=source,
                                 local_sensitivity=True)
    pd.testing.assert_frame_equal(
        ordinary.frame.sort_values("ENTITY").reset_index(drop=True),
        local.frame.drop(columns=["PREDICTION_ID", "TASK_ROW_ID", "CUTOFF_NS"]
                         ).sort_values("ENTITY").reset_index(drop=True),
        check_exact=False, rtol=1e-6, atol=1e-6)
    from kumo.engine.evidence import PredictionEvidenceBundle
    exported = local.evidence.export(root / "evidence-bundle")
    assert exported.validate()["valid"]
    copied = root / "copied-evidence"
    shutil.copytree(exported.root, copied)
    assert PredictionEvidenceBundle.load(copied).validate()["valid"]
    xai = kumo.PredictionExplanations.load(local.explanations.run_root)
    rows = xai.local_sensitivities()
    assert len(rows) == 4, rows
    assert counters == {"forward": 4, "autograd": 4}, counters
    manifest = xai.local_sensitivity_manifest()
    assert manifest["rows"] == 4
    assert manifest["complete"] is (not temporal)
    if temporal:
        assert manifest["temporal_status_counts"]["violated"] >= 1
    assert manifest["bundle_files_sha256"]
    frame = local.frame
    entity_col = next(
        c for c in frame if
        c not in {"TARGET_PRED", "PREDICTION_ID", "TASK_ROW_ID", "CUTOFF_NS"}
        and set(frame[c].dropna()) == {0, 1, 2, 3})
    for row in rows:
        assert row["status"] in {"available", "zero_signal"}, row
        assert row["prediction"] == float(
            frame.loc[frame[entity_col] == row["entity"],
                      "TARGET_PRED"].iloc[0])
        detail = xai.local_sensitivity(row["prediction_id"])
        sample = xai.scoring_sample(row["prediction_id"])
        assert sample["origin"] == "original_forward"
        if temporal:
            cutoff = row['time_ns'][0]
            violation = any(t >= cutoff
                            for t in sample['nodes']['events']['times_ns'])
            assert row['temporal_validation']['status'] == (
                'violated' if violation else 'valid')
            assert row['graph_status'] == 'available'
        encoded = json.dumps(sample, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(
            encoded.encode()).hexdigest() == row["sample_sha256"]
        assert any(c["column"] == "quantity" for c in detail["cells"])
        for cell in detail["cells"]:
            assert cell["pkey"] == sample["nodes"][cell["table"]]["pkeys"][
                cell["local_id"]]
    # Finite differences use the captured native model/batch only AFTER the
    # measured production run, perturbing exactly its transformed coordinate.
    module, batch, (recorded, _, _) = captured[0]
    cell = max((c for c in recorded if c["column"] == "quantity"),
               key=lambda c: c["abs_gradient"] or 0)
    encoder = next(e for e in module.encoder.enc_dict[
        cell["table"]].stype_encoder_dict.values()
                   if cell["column"] in e.col_names)
    original_learnable = encoder.forward_learnable
    column = encoder.col_names.index(cell["column"])

    def perturb(delta):
        def apply(value):
            value = value.clone()
            value[cell["local_id"], column] += delta
            return original_learnable(value)

        with torch.no_grad(), patch.object(encoder, "forward_learnable", apply):
            logits = module.predict_step(batch, 0)["logits"]
            return float(
                module.target_encoder.decode_tensor(
                    module.loss.to_prediction(logits)).reshape(()))

    step = 1e-3
    signed_difference = (perturb(step) - perturb(-step)) / (2 * step)
    assert np.isclose(signed_difference, cell["signed_gradient"], rtol=.01,
                      atol=.001), (signed_difference, cell["signed_gradient"], cell)
    difference = abs(signed_difference)
    assert np.isclose(difference, cell["abs_gradient"], rtol=.01,
                      atol=.001), (difference, cell)
    # A sink failure still emits every prediction and immutable identity.
    with patch.object(LocalSensitivity, '_write_manifest',
                      side_effect=OSError('injected sink failure')):
        failed = restored.predict(snapshot=snapshot, entities=source,
                                  local_sensitivity=True)
    assert len(failed.frame) == 4 and failed.frame.PREDICTION_ID.nunique() == 4
    assert not failed.explanations.local_availability.available
    pd.testing.assert_frame_equal(
        ordinary.frame.sort_values('ENTITY').reset_index(drop=True),
        failed.frame.drop(columns=['PREDICTION_ID', 'TASK_ROW_ID', 'CUTOFF_NS']
                          ).sort_values('ENTITY').reset_index(drop=True),
        check_exact=False, rtol=1e-6, atol=1e-6)
    from kumo.engine.evidence import EvidenceConfig
    with patch.object(torch.autograd, 'grad',
                      side_effect=AssertionError('graph-only differentiated')):
        graph_only = restored.predict(
            snapshot=snapshot, entities=source, batch_size=1,
            evidence=EvidenceConfig(numerical_sensitivity=None))
    graph_report = graph_only.evidence.export(root / 'graph-only').validate()
    assert graph_report['valid'] and graph_report['evidence_complete'] is (
        not temporal)
    if not temporal:
        from kumo_core.train.diskgraph_nested_trainer import (
            DiskgraphNestedTrainer,
        )
        native_load = DiskgraphNestedTrainer.load

        def unsupported_head(*args, **kwargs):
            trainer = native_load(*args, **kwargs)
            trainer.loss = 'mae_loss'
            return trainer

        with patch.object(DiskgraphNestedTrainer, 'load', unsupported_head):
            unsupported = restored.predict(
                snapshot=snapshot, entities=source, batch_size=1,
                evidence=EvidenceConfig(on_unsupported='record'))
        assert len(unsupported.frame) == 4
        assert unsupported.explanations.local_sensitivity_manifest(
        )['status'] == 'unsupported'
    replay = restored.predict(snapshot=snapshot, entities=source,
                              explanations=True, local_sensitivity=True)
    assert len(replay.explanations.local_sensitivities()) == 4
    assert replay.explanations.availability["cohorts"].available
    assert replay.explanations.availability["samples"].available
    if temporal:
        from kumo.engine.perturbation import RecordRemovalPlan
        first = next(row for row in rows
                     if row['temporal_validation']['status'] == 'valid')
        sample = exported.scoring_graph(first['prediction_id'])
        candidate = sample['nodes']['events']['pkeys'][0]
        removal = RecordRemovalPlan(
            groups=({
                'table': 'events',
                'pkey': candidate
            }, ), removable_tables=('events', ),
            foreign_keys=(('events', 'entity_id', 'entities'), ))
        shutil.rmtree(root / 'scratch')
        shutil.rmtree(root / 'data')
        derived = PredictionEvidenceBundle.load(copied).perturb(
            model=restored, plan=removal, output=root / 'removal-evidence',
            prediction_ids=[first['prediction_id']])
        assert derived.validate()['valid']
        assert derived.manifest['perturbations']['responses'] == 1
    if (root / "scratch").exists():
        shutil.rmtree(root / "scratch")
    assert PredictionEvidenceBundle.load(copied).validate()["valid"]
    result = {
        "passed": True,
        "engine_import": kumo.__file__,
        "prediction_rows": 4,
        "temporal": temporal,
        "original_forward_counts": counters,
        "finite_difference": difference,
        "analytic_abs_gradient": cell["abs_gradient"],
        "local_run_root": local.explanations.run_root,
        "replay_run_root": replay.explanations.run_root,
        "bundle": str(bundle),
        "provenance": manifest
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    return result


def verify_install(wheel, revision):
    import email
    import importlib
    import importlib.metadata
    import os
    import sys
    import zipfile

    if os.environ.get('PYTHONPATH') or os.environ.get('DISKGRAPH_INGEST_BIN'):
        raise ValueError('Source overlays are forbidden for wheel acceptance')
    prefix = Path(sys.prefix).resolve()
    origins = {}
    for name in ('kumo.engine', 'kumo_core', 'kumoapi', 'diskgraph'):
        origin = Path(importlib.import_module(name).__file__).resolve()
        if not origin.is_relative_to(
                prefix) or 'site-packages' not in origin.parts:
            raise ValueError(
                f'{name} does not resolve to the isolated installation')
        origins[name] = str(origin)
    with zipfile.ZipFile(wheel) as archive:
        metadata_path, = [
            p for p in archive.namelist() if p.endswith('.dist-info/METADATA')
        ]
        metadata = email.message_from_bytes(archive.read(metadata_path))
        assert importlib.metadata.version('kumo-engine') == metadata['Version']
        commit_suffix = metadata['Version'].partition('+')[2]
        assert len(commit_suffix) >= 7 and revision.startswith(commit_suffix)
        for name in ('kumo.engine', 'kumo_core.explain.local_sensitivity',
                     'kumo.engine.evidence', 'kumo.engine.uncertainty'):
            path = Path(importlib.import_module(name).__file__).resolve()
            relative = path.as_posix().split('/site-packages/', 1)[1]
            assert archive.read(relative) == path.read_bytes(), name
    return {
        'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'source_revision': revision,
        'version': metadata['Version'],
        'module_origins': origins,
        'torch': torch.__version__
    }


def calibration_gate(root):
    from kumo.engine.uncertainty import (
        CalibrationSpec,
        PredictionIntervalCalibrator,
    )
    rng = np.random.default_rng(20261113)
    rows = [{
        'prediction_id': f'cal-{i}',
        'prediction': 0.
    } for i in range(10000)]
    outcomes = {row['prediction_id']: float(rng.normal()) for row in rows}
    spec = CalibrationSpec(model_sha256='a' * 64, scoring_sha256='b' * 64)
    cal = PredictionIntervalCalibrator.fit(
        rows, outcomes, spec=spec, population_id='synthetic-calibration',
        separation_declaration='Independent synthetic population, never fitted')
    cal.save(root / 'calibration')
    cal = PredictionIntervalCalibrator.load(root / 'calibration')
    future = [{
        'prediction_id': f'test-{i}',
        'prediction': 0.
    } for i in range(20000)]
    intervals = list(
        cal.apply(future, model_sha256='a' * 64, scoring_sha256='b' * 64))
    y = rng.normal(size=len(future))
    covered = sum(row['lower'] <= value <= row['upper']
                  for row, value in zip(intervals, y))
    coverage = covered / len(y)
    z = 1.6448536269514722
    lower = (coverage + z * z /
             (2 * len(y)) - z * np.sqrt(coverage *
                                        (1 - coverage) / len(y) + z * z /
                                        (4 * len(y)**2))) / (1 + z * z / len(y))
    assert lower > .88, (coverage, lower)
    return {
        'coverage': coverage,
        'one_sided_95_lower': lower,
        'mean_width':
        float(np.mean([r['upper'] - r['lower'] for r in intervals])),
        'calibration_rows': len(rows),
        'evaluation_rows': len(future)
    }


def supply_chain_gate(root, prototype):
    import sys

    # Only application adapters are imported from this read-only tree.
    # Engine module origins were checked before this import.
    sys.path.insert(0, str(prototype))
    from supplychain.data import load_prepared, snapshot_at
    from supplychain.modeling import build_graph

    original = prototype / 'artifacts/runs/quality_v1/retained/model'
    inventory = {
        str(p.relative_to(original)):
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in original.rglob('*') if p.is_file()
    }
    shutil.copytree(original, root / 'saved-model')
    kumo.configure(scratch_dir=root / 'scratch')
    prepared = load_prepared(prototype / 'data/modeling')
    cutoff = pd.Timestamp('2023-07-06')
    snapshot_at(prepared, cutoff, root / 'historical-snapshot')
    graph = build_graph(root / 'historical-snapshot/tables', 'relational',
                        root / 'derived')
    snapshot = graph.snapshot()
    model = kumo.Trainer.load(root / 'saved-model', accelerator='cpu')
    products = pd.read_parquet(root /
                               'historical-snapshot/tables/products.parquet')
    population = root / 'population'
    population.mkdir()
    pd.DataFrame({
        'ENTITY1': products.product_id.iloc[:4],
        'TIMESTAMP': cutoff
    }).to_parquet(population / 'score.parquet', index=False)
    source = kumo.FileConnector(root_dir=str(population))['score.parquet']
    result = model.predict(snapshot=snapshot, entities=source, batch_size=1,
                           local_sensitivity=True, explanations=True)
    exported = result.evidence.export(root / 'evidence')
    report = exported.validate()
    assert report['valid'] and report['prediction_rows'] == 4 and report[
        'evidence_complete']
    from supplychain.model_export import (
        adapt_evidence,
        model_identity,
        source_lookup,
    )
    from supplychain.modeling import availability, utc_string
    tables, sources = source_lookup(prepared)
    legacy_rows = 0
    for summary in result.explanations.local_sensitivities():
        identifier = summary['prediction_id']
        adapted = adapt_evidence(
            result.explanations.local_sensitivity(identifier),
            result.explanations.scoring_sample(identifier),
            run_id='wheel-smoke', cutoff=utc_string(cutoff),
            snapshot_id='wheel-smoke-snapshot',
            model=model_identity(root / 'saved-model'), tables=tables,
            sources=sources, replay_availability=availability(
                'not_sampled', reason='Separate replay diagnostic'))
        assert adapted['prediction_id'] == identifier and adapted[
            'graph'] is not None
        legacy_rows += 1
    assert legacy_rows == 4
    assert inventory == {
        str(p.relative_to(original)):
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in original.rglob('*') if p.is_file()
    }
    return {
        'passed': True,
        'prediction_rows': 4,
        'cutoff': str(cutoff),
        'legacy_consumer_rows': legacy_rows,
        'heldout_evaluated': False,
        'model_unchanged': True,
        'bundle': report
    }


def main():
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['native', 'supply-chain'],
                        required=True)
    parser.add_argument('--development', action='store_true',
                        help='Source preflight only, never wheel acceptance')
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--prototype-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.suite == 'supply-chain' and args.prototype_root is None:
        parser.error('--suite supply-chain requires --prototype-root')
    prototype = args.prototype_root.resolve() if args.prototype_root else None
    if prototype is not None and output.is_relative_to(prototype):
        raise ValueError('Never write acceptance outputs inside the prototype')
    output.mkdir(parents=True, exist_ok=False)
    if args.development:
        identity = {
            'source_revision': args.source_revision,
            'development_only': True,
            'engine_import': kumo.__file__
        }
    else:
        if args.wheel is None:
            parser.error('--wheel is required without --development')
        identity = verify_install(args.wheel.resolve(), args.source_revision)
    started = time.monotonic()
    if args.suite == 'native':
        results = {
            'static': run(output / 'static'),
            'temporal': run(output / 'temporal', temporal=True),
            'calibration': calibration_gate(output)
        }
    else:
        results = supply_chain_gate(output, prototype)
    report = {
        'passed': True,
        'installed_wheel_acceptance': not args.development,
        'identity': identity,
        'results': results,
        'elapsed_seconds': time.monotonic() - started
    }
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
