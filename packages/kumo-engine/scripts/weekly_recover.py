#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect one retained fit after a reviewed native staging-link failure."""

import argparse
import runpy
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

PORTABLE = runpy.run_path(str(Path(__file__).with_name('weekly_check.py')))
read, save, digest, pinned = [
    PORTABLE[name] for name in ('read', 'save', 'digest', 'pinned')
]


def bind_original_runtime(spec, validator, identity_path):
    original_root = Path(identity_path).parents[1]
    local_root = validator.resolve().parents[1]
    mappings = []
    for source, sha in spec['inputs'].items():
        source = Path(source)
        if not source.is_relative_to(original_root):
            continue
        local = local_root / source.relative_to(original_root)
        if not local.is_file() or local.is_symlink(
        ) or not local.resolve().is_relative_to(local_root):
            raise ValueError(
                'The original validator runtime has a missing or redirected helper'
            )
        pinned(local, sha)
        mappings.append(
            dict(identity_path=str(source), local_path=str(local), sha256=sha))
    required = {
        'scripts/weekly_check.py', 'scripts/pr_check.py',
        'test/wheel_gpu_learning.py', 'test/wheel_measurements.py'
    }
    if not required <= {
            str(Path(row['identity_path']).relative_to(original_root))
            for row in mappings
    }:
        raise ValueError(
            'The original validator runtime is not completely pinned')
    return mappings


def recover(source, output, audit_record, validator_record, *,
            source_identity_path=None, validator_identity_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    failed = read(source / 'weekly-case.json')
    spec = failed['case_spec']
    audit = read(pinned(audit_record['path'], audit_record['sha256']))
    validator = pinned(validator_record['path'], validator_record['sha256'])
    bound_validator = validator_identity_path or str(validator)
    bound_source = source_identity_path or str(source)
    if (failed.get('status') != 'FAIL' or failed.get('error')
            != 'ValueError: Evidence cannot contain symbolic links'
            or failed.get('physical_fits') != 1
            or spec['inputs'].get(bound_validator) != validator_record['sha256']
            or audit['source_case'] != bound_source):
        raise ValueError(
            'Recovery requires the exact failed case and its original validator'
        )
    runtime_mappings = bind_original_runtime(spec, validator, bound_validator)
    original = runpy.run_path(str(validator))
    manifest = original['load_manifest'](source / 'input-manifest.json',
                                         failed['manifest_sha256'])
    [recorded
     ] = [row for row in manifest['cases'] if row['id'] == failed['case_id']]
    if recorded != spec or failed['identity'] != spec['identity']:
        raise ValueError(
            'The retained case identity differs from its frozen manifest')
    result = read(source / 'result.json')
    if result != read(source / 'worker/result.json'):
        raise ValueError(
            'The completed worker result differs from the retained outer result'
        )
    original['verify_result'](spec, result)
    checkpoint = result['expected_reload_checkpoint_sha256']
    checkpoints = [
        path for path in source.rglob('checkpoint')
        if path.is_file() and path.as_posix().endswith('/model/checkpoint')
    ]
    if (not checkpoints or any(path.is_symlink() or digest(path) != checkpoint
                               for path in checkpoints)
            or audit['fit_id'] != result['fit_id']
            or audit['selected_checkpoint_sha256'] != checkpoint):
        raise ValueError(
            'The retained fit or checkpoint differs from the reviewed failure')
    links = PORTABLE['prediction_source_links'](source)
    reviewed = {row['path']: row for row in audit['links']}
    if not links or {row['path'] for row in links} != set(reviewed):
        raise ValueError(
            'The managed source links differ from the exact reviewed allowlist')
    for link in links:
        recorded = reviewed[link['path']]
        if (link['link_target'] != recorded['link_target']
                or link['bytes'] != recorded['target_bytes']
                or link['sha256'] != recorded['target_sha256']):
            raise ValueError(
                'A reviewed source link or target changed before recovery')
    started = time.monotonic()
    PORTABLE['copy_portable_case'](source, output, links)
    shutil.copyfile(source / 'weekly-case.json',
                    output / 'source-collection-failure.json')
    proof = dict(
        schema=1, source_root=str(source), source_identity_path=bound_source,
        source_failure_sha256=digest(source / 'weekly-case.json'),
        audit=audit_record, original_runtime=runtime_mappings,
        original_validator=dict(validator_record,
                                identity_path=bound_validator),
        collection_helper_sha256=digest(
            Path(__file__).with_name('weekly_check.py')),
        recovery_source_sha256=digest(__file__), fit_id=result['fit_id'],
        checkpoint_sha256=checkpoint, identity=spec['identity'], new_fits=0,
        new_predictions=0,
        copy_receipt_sha256=digest(output / 'portable-copy.json'))
    save(output / 'collection-recovery-provenance.json', proof)
    report = dict(
        failed, status='MEASURED', result=result,
        collection_recovery_sha256=digest(
            output / 'collection-recovery-provenance.json'))
    report.pop('error', None)
    report.pop('traceback', None)
    verification = dict(
        proof, status='VERIFYING',
        recovery_started_utc=datetime.now(timezone.utc).isoformat())
    try:
        report['files'] = original['inventory'](output)
        save(output / 'weekly-case.json', report)
        original['load_case'](output / 'weekly-case.json', spec, failed['mode'],
                              manifest)
        verification.update(
            status='PASS', original_source_unchanged=True,
            recovered_report_sha256=digest(output / 'weekly-case.json'))
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        save(output / 'weekly-case.json', report)
        verification.update(status='FAIL', error=report['error'])
        raise
    finally:
        verification['seconds'] = time.monotonic() - started
        save(output.with_name(output.name + '-recovery.json'), verification)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'link-audit', 'validator'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--link-audit-sha256', required=True)
    parser.add_argument('--validator-sha256', required=True)
    parser.add_argument('--source-identity-path')
    parser.add_argument('--validator-identity-path')
    args = parser.parse_args()
    recover(args.source, args.output,
            dict(path=str(args.link_audit), sha256=args.link_audit_sha256),
            dict(path=str(args.validator), sha256=args.validator_sha256),
            source_identity_path=args.source_identity_path,
            validator_identity_path=args.validator_identity_path)


if __name__ == '__main__':
    main()
