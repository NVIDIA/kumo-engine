# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete sampling or retained-model two-GPU receipts."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/scripts/weekly_aux.py'


def api():
    return runpy.run_path(str(SCRIPT))


@pytest.mark.parametrize('changed', [None, 'missing', 'unbound', 'changed'])
def test_b9_binds_separate_reader_before_model_calls(tmp_path, changed):
    reader = tmp_path / 'reader.py'
    reader.write_text(
        'def ordered_batches(*args, **kwargs):\n    return iter(())\n')
    sha = hashlib.sha256(reader.read_bytes()).hexdigest()
    spec = dict(
        worker=dict(prediction_reader=dict(path=str(reader), sha256=sha)),
        inputs={str(reader): sha})
    if changed == 'missing':
        del spec['worker']['prediction_reader']
    elif changed == 'unbound':
        spec['inputs'].clear()
    elif changed == 'changed':
        reader.write_text('# changed')
    if changed:
        with pytest.raises(ValueError):
            api()['b9_prediction_reader'](spec)
    else:
        assert list(api()['b9_prediction_reader'](spec).ordered_batches()) == []


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True,
                   allow_nan=False).encode()).hexdigest()


def p1_spec():
    return dict(
        check='P1',
        work=dict(n=512, offset=0, hops=2, batch=128, procs=1, threads=8,
                  cache='no_drop_uncontrolled', features=True,
                  expected_work=dict(batches=4, requested_seeds=512),
                  historical_sampled_work=dict(nodes=5120, edges=6144)),
        data_sha256='a' * 64)


def p1_report():
    return dict(
        check='P1', status='MEASURED', physical_fits=0, fit_attempts=0,
        operations=[
            dict(
                operation=index, device='cpu',
                cache_policy='no_drop_uncontrolled', cache_qualified=False,
                work=p1_spec()['work'], data_sha256='a' * 64,
                sampled_work=dict(p1_spec()['work']['expected_work'],
                                  nodes=5120, edges=6144),
                workers=[
                    dict(
                        status='PASS', slot=0, pid=100,
                        allocation=dict(cpu_bytes=0), delay_seconds_per_batch=0,
                        retained_buffers=0,
                        work=dict(p1_spec()['work']['expected_work'],
                                  nodes=5120, edges=6144))
                ], result=dict(status='ok', cache='warm', hops=2, seeds=512,
                               batch=128, procs=1, threads=8, features=True,
                               wall_s=1.0, nodes_per_seed=10, rows=5120,
                               cpu_util=0.1, cores_busy=1, peak_rss_gb_sum=0.1))
            for index in range(5)
        ])


def proof(worker, *, checkpoint='b' * 64):
    return dict(
        status='PASS', worker_id=worker, pid=100 + worker,
        root_device=f'cuda:{worker}', score_device=f'cuda:{worker}',
        completed_batches=2, score_elements=20,
        physical_gpu_uuid=f'GPU-00000000-0000-4000-8000-{worker+1:012d}',
        gpu_name='NVIDIA L4', fit_attempts=0, batch_retries_disabled=True,
        eval_batch_size=1024, reload_identity_checks=[
            dict(status='PASS', checkpoint_sha256=checkpoint,
                 restore_path='lightning_strategy.load_model_state_dict',
                 tensor_count=10)
        ])


def b9_spec():
    return dict(
        check='B9', retained_fit_id='retained-P2-fit',
        checkpoint_sha256='b' * 64, gpu_name='NVIDIA L4',
        prediction_batch_size=1024, population=dict(
            expected_prediction_count=2,
            prediction_ids_sha256=hashlib.sha256(b'[1,2]').hexdigest()),
        metric='recall@10')


def b9_report():
    predicted = dict(
        entity_ids=[1, 2], catalog=list(range(1, 13)),
        ranked_ids=[list(range(1, 11)), list(range(1, 11))],
        ranked_scores=[[float(value) for value in range(10, 0, -1)]] * 2)
    return dict(
        check='B9', status='MEASURED', physical_fits=0, fit_attempts=0,
        retained_fit_id='retained-P2-fit', checkpoint_sha256='b' * 64, calls=[
            dict(workers=1, predictions=copy.deepcopy(predicted),
                 devices=[proof(0)]),
            dict(workers=2, predictions=copy.deepcopy(predicted),
                 devices=[proof(0), proof(1)])
        ])


def freshness():
    return dict(
        status='UNCHANGED_FULL_HASHED_PAYLOAD', lease=dict(owner_pid=123),
        seal_sha256='a' * 64, full_roster_sha256='b' * 64, payload_files=7858,
        payload_bytes=1399792381206, frozen_native_inputs=dict(
            status='FULL_NATIVE_INPUT_HASHES_VERIFIED',
            training_files={'train.parquet': 'c' * 64},
            prediction_files={'predict.parquet':
                              'd' * 64}, prediction_population_sha256='e' * 64),
        catalog_files_sha256={'merchants.parquet': 'f' * 64})


def test_p1_requires_five_complete_frozen_cpu_operations():
    api()['verify_p1'](p1_spec(), p1_report())


@pytest.mark.parametrize(
    'change',
    ['four', 'capped', 'gpu', 'changed_work', 'duplicate', 'no_memory', 'fit'])
def test_p1_rejects_missing_work_or_changed_profile(change):
    row = p1_report()
    if change == 'four':
        row['operations'].pop()
    elif change == 'capped':
        row['operations'][0]['result']['status'] = 'capped'
    elif change == 'gpu':
        row['operations'][0]['device'] = 'cuda:0'
    elif change == 'changed_work':
        row['operations'][0]['result']['seeds'] = 128
    elif change == 'duplicate':
        row['operations'][1]['operation'] = 0
    elif change == 'no_memory':
        row['operations'][0]['result']['peak_rss_gb_sum'] = None
    else:
        row['fit_attempts'] = 1
    with pytest.raises(ValueError):
        api()['verify_p1'](p1_spec(), row)


def test_b9_requires_both_matching_gpus_and_same_retained_checkpoint():
    assert api()['verify_b9'](b9_spec(),
                              b9_report())['removed_recommendations'] == 0


@pytest.mark.parametrize('change', [
    'same_gpu', 'wrong_gpu', 'no_batches', 'checkpoint', 'ids', 'fit',
    'missing_worker'
])
def test_b9_rejects_incomplete_or_mismatched_actual_work(change):
    row = b9_report()
    dual = row['calls'][1]
    if change == 'same_gpu':
        dual['devices'][1]['physical_gpu_uuid'] = dual['devices'][0][
            'physical_gpu_uuid']
    elif change == 'wrong_gpu':
        dual['devices'][1][
            'gpu_name'] = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
    elif change == 'no_batches':
        dual['devices'][1]['completed_batches'] = 0
    elif change == 'checkpoint':
        dual['devices'][1]['reload_identity_checks'][0][
            'checkpoint_sha256'] = 'c' * 64
    elif change == 'ids':
        dual['predictions']['entity_ids'] = [1, 3]
    elif change == 'fit':
        dual['devices'][0]['fit_attempts'] = 1
    else:
        dual['devices'].pop()
    with pytest.raises(ValueError):
        api()['verify_b9'](b9_spec(), row)


def test_b9_reports_membership_and_numeric_differences_without_a_tolerance_gate(
):
    row = b9_report()
    row['calls'][1]['predictions']['ranked_ids'][0][-1] = 11
    row['calls'][1]['predictions']['ranked_scores'][0][0] = 1000.0
    measured = api()['verify_b9'](b9_spec(), row)
    assert measured['removed_recommendations'] == 1
    assert measured['added_recommendations'] == 1
    assert measured['max_abs_difference'] == 990.0
    assert measured['qualified'] is False


def test_auxiliary_receipts_bind_the_case_and_raw_files(tmp_path):
    source = tmp_path / 'output'
    source.mkdir()
    (source / 'raw.json').write_text('{}')
    spec = p1_spec()
    (source / 'input-manifest.json').write_text(json.dumps(spec))
    manifest_sha = hashlib.sha256(
        (source / 'input-manifest.json').read_bytes()).hexdigest()
    row = p1_report()
    row.update(
        schema=1, manifest_sha256=manifest_sha, files={
            'raw.json': hashlib.sha256(b'{}').hexdigest(),
            'input-manifest.json': manifest_sha
        })
    (source / 'auxiliary-result.json').write_text(json.dumps(row))
    api()['load_evidence'](source, p1_spec(), manifest_sha)
    (source / 'raw.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):
        api()['load_evidence'](source, p1_spec(), manifest_sha)


def test_forbidden_fit_call_writes_its_attempt_before_raising(tmp_path):
    path = tmp_path / 'fit-attempt.json'
    with pytest.raises(RuntimeError, match='forbidden'):
        api()['reject_fit'](path)
    assert json.loads(path.read_text())['fit_attempts'] == 1


def write_receipt(root, spec, row):
    root.mkdir()
    (root / 'input-manifest.json').write_text(json.dumps(spec))
    pin = hashlib.sha256(
        (root / 'input-manifest.json').read_bytes()).hexdigest()
    (root / 'raw.json').write_text('{}')
    if spec['check'] == 'B9':
        (root / 'memory.json').write_text('{"overall":{"samples":1}}')
        for name in ('payload-precheck.json', 'payload-postcheck.json'):
            (root / name).write_text(json.dumps(freshness()))
    row.update(
        schema=1, manifest_sha256=pin, files={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.iterdir() if path.is_file()
        })
    destination = root / 'auxiliary-result.json'
    destination.write_text(json.dumps(row))
    return dict(path=str(destination),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


def test_bound_ancillary_receipts_complete_coverage_without_another_fit(
        tmp_path):
    wheel = dict(version='1.0.0', sha256='d' * 64)
    one, two = p1_spec(), b9_spec()
    one.update(wheel=wheel, dataset='commerce/100M')
    two['wheel'] = wheel
    records = dict(P1=write_receipt(tmp_path / 'p1', one, p1_report()),
                   B9=write_receipt(tmp_path / 'b9', two, b9_report()))
    weekly = runpy.run_path(str(SCRIPT.with_name('weekly_check.py')))
    fits = {
        'retained-P2-fit':
        dict(workload='P2', checkpoint_sha256='b' * 64,
             population=two['population'], gpu='NVIDIA L4')
    }
    result = weekly['milestone_coverage'](records, fits, ('1.0.0', 'd' * 64))
    assert result['weekly_milestone_complete'] is True
    assert result['ancillary_checks']['P1']['status'] == 'MEASURED'
    assert result['ancillary_checks']['B9']['status'] == 'MEASURED'
    with pytest.raises(ValueError, match='retained'):
        weekly['milestone_coverage'](records, {}, ('1.0.0', 'd' * 64))


def test_b9_rejects_batch_retries_or_changed_prediction_batches():
    for field, value in [('batch_retries_disabled', False),
                         ('eval_batch_size', 512)]:
        row = b9_report()
        row['calls'][1]['devices'][1][field] = value
        with pytest.raises(ValueError):
            api()['verify_b9'](b9_spec(), row)


def test_retained_loader_supports_the_released_1_0_0_signature(tmp_path):
    class ReleasedTrainer:
        @classmethod
        def load(cls, bundle_path, embedding_models=None):
            return dict(bundle_path=bundle_path,
                        embedding_models=embedding_models)

    bundle = tmp_path / 'retained-model'
    assert api()['load_saved_model'](ReleasedTrainer,
                                     bundle) == dict(bundle_path=bundle,
                                                     embedding_models=None)


def test_catalog_verification_does_not_rescan_the_catalog_for_every_entity():
    class CountedCatalog(list):
        traversals = 0

        def __iter__(self):
            self.traversals += 1
            return super().__iter__()

    catalog = CountedCatalog(range(1, 101))
    ids = list(range(1, 21))
    spec = dict(
        metric='recall@10', population=dict(
            expected_prediction_count=20, prediction_ids_sha256=hashlib.sha256(
                json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
    values = dict(
        entity_ids=ids, catalog=catalog,
        ranked_ids=[list(range(1, 11)) for _ in ids],
        ranked_scores=[[float(value) for value in range(10, 0, -1)]
                       for _ in ids])
    assert len(api()['ranked_coordinates'](spec, values)) == 200
    assert catalog.traversals <= 4


def make_p1_identity(root, *, tiny=False, wrong_counts=False):
    root.mkdir()
    index = root / 'index'
    index.mkdir()
    rows = {
        'customers': 100000,
        'purchases': 106407358
    } if not tiny else {
        'customers': 10
    }
    raw = dict(
        graph='commerce', rung='100M', total_rows=sum(rows.values()), tables={
            name: dict(rows=count, files=[])
            for name, count in rows.items()
        })

    def record(path, value):
        path.write_text(json.dumps(value))
        return dict(path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    raw_record = record(root / 'manifest.json', raw)
    counts = dict(rows)
    if wrong_counts:
        counts['customers'] -= 1
    record(
        index / 'index_metadata.json',
        dict(node_types=[
            dict(name=name, max_id=count - 1) for name, count in counts.items()
        ]))
    record(
        index / 'ingest_stats.json',
        dict(stats=dict(node_counts=counts, total_nodes=sum(counts.values()))))
    record(index / '_MANIFEST.json', {'sealed': True})
    record(index / 'diskgraph_config.json', {'tables': list(rows)})
    files = {
        path.name:
        dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             bytes=path.stat().st_size)
        for path in index.iterdir()
    }
    work = dict(n=300000, unique_customers=100000, offset=0, hops=3, batch=256,
                procs=2, threads=8, plan=[24, 8,
                                          6], cache='no_drop_uncontrolled',
                features=True, expected_work=dict(batches=1172,
                                                  requested_seeds=300000),
                historical_sampled_work=dict(nodes=173052351, edges=196315170))
    seedmap = record(root / 'seedmap.parquet',
                     {'fixture': 'no native execution'})
    profile = record(
        root / 'profile.json',
        dict(
            index_path=str(index),
            index_manifest_sha256=files['_MANIFEST.json']['sha256'],
            ingest_stats_sha256=files['ingest_stats.json']['sha256'],
            seedmap=str(root / 'seedmap.parquet'),
            seedmap_sha256=seedmap['sha256'], unique_customers=100000,
            seed_evaluations=300000, fanout=[24, 8, 6], batch_size=256,
            processes=2, threads_per_process=8,
            expected_work=dict(work['expected_work'],
                               **work['historical_sampled_work']),
            data_manifest_sha256='a' * 64))
    correction = record(
        root / 'correction.json',
        dict(graph='commerce', rung='100M', total_rows=sum(rows.values()),
             dataset_manifest_sha256=raw_record['sha256'],
             admission_audit_sha256='a' * 64,
             raw_profile_sha256=profile['sha256']))
    seeds = record(root / 'seeds.parquet', {'fixture': 'no native execution'})
    identity = dict(dataset_manifest=raw_record, profile=profile,
                    correction=correction,
                    snapshot=dict(index_path=str(index), files=files),
                    seed_population=seeds, seedmap=seedmap, work=work,
                    selection=dict(sha256='b' * 64, offset=0, n=300000))
    identity_record = record(root / 'identity.json', identity)
    canonical = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    spec = dict(
        data_sha256=canonical(identity), work=work,
        worker=dict(index=str(index), seedmap=seedmap,
                    native_identity=identity_record))
    return spec


def test_p1_identity_derives_data_digest_and_actual_native_table_counts(
        tmp_path):
    spec = make_p1_identity(tmp_path / 'good')
    assert api()['verify_p1_identity'](spec)['dataset_manifest']['sha256']
    spec['data_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        api()['verify_p1_identity'](spec)


@pytest.mark.parametrize('kind', ['tiny', 'wrong_counts'])
def test_p1_identity_rejects_pinned_tiny_or_incomplete_native_graph(
        tmp_path, kind):
    spec = make_p1_identity(tmp_path / kind, tiny=kind == 'tiny',
                            wrong_counts=kind == 'wrong_counts')
    with pytest.raises(ValueError):
        api()['verify_p1_identity'](spec)


def test_p1_requires_actual_native_work_and_no_unproved_warm_cache():
    row = p1_report()
    row['operations'][0]['cache_qualified'] = True
    with pytest.raises(ValueError):
        api()['verify_p1'](p1_spec(), row)


def test_p1_reports_consistent_node_edge_changes_without_a_historical_performance_gate(
):
    row = p1_report()
    item = row['operations'][0]
    item['workers'][0]['work'] = dict(item['workers'][0]['work'], nodes=6000,
                                      edges=7000)
    item['sampled_work'] = dict(item['sampled_work'], nodes=6000, edges=7000)
    item['result']['rows'] = 6000
    item['result']['nodes_per_seed'] = 6000 / 512
    api()['verify_p1'](p1_spec(), row)


@pytest.mark.parametrize('changed',
                         [None, 'identity', 'population', 'batch', 'profile'])
def test_b9_capacity_is_bound_to_the_retained_full_p2_workload(changed):
    spec = b9_spec()
    primary = dict(workload='P2', hardware_profile='scale-l4',
                   identity=dict(data_sha256='a' * 64),
                   population=copy.deepcopy(spec['population']))
    capacity = dict(identity=copy.deepcopy(primary['identity']),
                    population=copy.deepcopy(primary['population']),
                    hardware_profile='scale-l4', prediction_batch_size=1024)
    if changed == 'identity':
        capacity['identity']['data_sha256'] = 'f' * 64
    elif changed == 'population':
        capacity['population']['expected_prediction_count'] = 1
    elif changed == 'batch':
        capacity['prediction_batch_size'] = 512
    elif changed == 'profile':
        capacity['hardware_profile'] = 'tiny-fixture'
    if changed:
        with pytest.raises(ValueError, match='capacity'):
            api()['verify_b9_capacity'](spec, capacity, primary)
    else:
        api()['verify_b9_capacity'](spec, capacity, primary)


def test_b9_payload_postcheck_reuses_the_precheck_host_lease(tmp_path):
    precheck = freshness()
    calls = []

    class Native:
        @staticmethod
        def verify_inputs(config, *, previous_lease):
            calls.append((config, previous_lease))
            return dict(precheck, checked_utc='later')

    module = api()
    prepared = dict(payload_freshness=precheck)
    saved = module['b9_payload_precheck'](prepared, tmp_path)
    post = module['b9_payload_postcheck'](Native, {
        'frozen': True
    }, saved, tmp_path)
    assert calls == [({'frozen': True}, precheck['lease'])]
    assert json.loads(
        (tmp_path / 'payload-precheck.json').read_text()) == precheck
    assert json.loads((tmp_path / 'payload-postcheck.json').read_text()) == post


@pytest.mark.parametrize('changed', ['payload', 'lease'])
def test_b9_changed_payload_or_lease_cannot_complete_and_preserves_postcheck(
        tmp_path, changed):
    precheck = freshness()

    class Native:
        @staticmethod
        def verify_inputs(config, *, previous_lease):
            if changed == 'payload':
                raise ValueError('Payload identity changed')
            return dict(precheck, lease=dict(owner_pid=999))

    with pytest.raises(ValueError):
        api()['b9_payload_postcheck'](Native, {}, precheck, tmp_path)
    assert json.loads(
        (tmp_path / 'payload-postcheck.json').read_text())['status'] == 'FAIL'


@pytest.mark.parametrize('changed',
                         ['missing_post', 'lease', 'seal', 'payload'])
def test_collected_b9_requires_complete_same_payload_and_lease_receipts(
        tmp_path, changed):
    root = tmp_path / 'b9'
    write_receipt(root, b9_spec(), b9_report())
    post = root / 'payload-postcheck.json'
    if changed == 'missing_post':
        post.unlink()
    else:
        proof = freshness()
        if changed == 'lease':
            proof['lease']['owner_pid'] = 999
        elif changed == 'seal':
            proof['seal_sha256'] = 'f' * 64
        else:
            proof['payload_files'] = 1
        post.write_text(json.dumps(proof))
    row = json.loads((root / 'auxiliary-result.json').read_text())
    row['files'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir() if p.name != 'auxiliary-result.json'
    }
    (root / 'auxiliary-result.json').write_text(json.dumps(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        api()['load_evidence'](root, b9_spec(), row['manifest_sha256'])
