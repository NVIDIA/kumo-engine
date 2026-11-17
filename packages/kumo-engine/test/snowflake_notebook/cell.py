# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from snowflake.snowpark.context import get_active_session

session = get_active_session()
session.sql('USE DATABASE KUMO_TEST_DB').collect()
session.sql('USE SCHEMA KUMO_TEST_DB.KUMO_TEST').collect()
row = session.sql(
    'SELECT CURRENT_ACCOUNT(),CURRENT_USER(),CURRENT_ROLE(),CURRENT_DATABASE(),CURRENT_SCHEMA(),CURRENT_SESSION()'
).collect()[0]
identity = dict(
    zip(['account', 'user', 'role', 'database', 'schema', 'session'], row))
stage = sys.argv[1]
assert stage.startswith(
    '@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/notebook-output-probe/'
), 'Unexpected attempt stage'
attempt = stage.rsplit('/', 1)[1]
local = Path(tempfile.mkdtemp(prefix='kumo-installed-lifecycle-'))
payload = local / 'payload'
payload.mkdir()
evidence = local / 'evidence'
evidence.mkdir()
receipt = dict(attempt=attempt, status='failed', identity=identity,
               scope=('Installed-wheel notebook lifecycle; UI excluded'),
               artifacts={})
config = {}
try:
    downloads = session.file.get(stage + '/payload', str(payload))
    if not downloads:
        raise RuntimeError('No payload downloaded')
    import re
    expected_references = sys.argv[2]
    assert re.fullmatch(
        r'[0-9a-f]{64}',
        expected_references), 'Invalid submitted references hash'
    references_file = payload / 'payload-locations.json'
    with references_file.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest(
        ) == expected_references, 'Cache references differ from submission'
    receipt['payload_references_sha256'] = expected_references
    references = json.loads(references_file.read_text())
    required = {
        'lifecycle-config.json', 'snowflake.json', 'wheel_mlops_lifecycle.py',
        'wheel_mlops_platform.py', 'wheel_gpu_smoke.py', 'offline_support.py',
        'wheelhouse.tar', 'wheelhouse-manifest.json'
    }
    wheel_names = {
        name
        for name in references
        if name.startswith('kumo_engine-') and name.endswith('.whl')
    }
    assert len(wheel_names) == 1 and set(
        references) == required | wheel_names, 'Incomplete cached payload'
    for name, entry in references.items():
        assert re.fullmatch(
            r'[A-Za-z0-9_.+-]+',
            name) and entry['name'] == name, 'Invalid cached filename'
        assert re.fullmatch(r'[0-9a-f]{64}',
                            entry['sha256']), 'Invalid cached hash'
        location = entry['location']
        base = re.escape('@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE')
        legacy = base + r'/notebook-output-probe/[0-9a-f]{12}/payload/' + re.escape(
            name)
        cached = base + r'/mlops-validation/payload-cache/[0-9a-f]{64}/' + re.escape(
            name) + r'/objects/[0-9a-f]{12}/' + re.escape(name)
        assert re.fullmatch(legacy, location) or re.fullmatch(
            cached, location), 'Invalid cached stage location'
        folder = Path(tempfile.mkdtemp(prefix='cached-download-', dir=local))
        session.file.get(location, str(folder))
        files = list(folder.iterdir())
        assert len(files) == 1 and files[0].name == name and files[0].is_file(
        ) and not files[0].is_symlink(), 'Unexpected cached download'
        downloaded = files[0]
        assert downloaded.stat(
        ).st_size == entry['size'], 'Cached file size mismatch'
        with downloaded.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest(
            ) == entry['sha256'], 'Cached file checksum mismatch'
        shutil.move(str(downloaded), str(payload / name))
    (evidence / 'cached-files.json').write_text(json.dumps(
        references, indent=2))
    config = json.loads((payload / 'lifecycle-config.json').read_text())
    receipt.update(mode=config['mode'], wheel_sha256=config['wheel_sha256'])
    wheel = payload / config['wheel_file']
    assert hashlib.sha256(wheel.read_bytes()).hexdigest(
    ) == config['wheel_sha256'], 'Candidate wheel changed during stage transfer'
    receipt['runtime'] = dict(
        python=sys.version, cpus=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)),
        self_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    support_spec = importlib.util.spec_from_file_location(
        'offline_support', payload / 'offline_support.py')
    support = importlib.util.module_from_spec(support_spec)
    support_spec.loader.exec_module(support)
    wheelhouse = local / 'wheelhouse'
    dependencies = support.unpack(payload, wheelhouse)
    (evidence / 'dependency-manifest.json').write_text(
        json.dumps(dependencies, indent=2))
    venv = local / 'venv'
    with (evidence / 'venv.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'venv',
                        str(venv)], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=60)
    python = venv / 'bin/python'
    with (evidence / 'install.log').open('w') as log:
        subprocess.run([
            str(python), '-I', '-m', 'pip', '--isolated', 'install',
            '--no-index', '--find-links',
            str(wheelhouse),
            str(wheel) + '[snowflake]'
        ], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    with (evidence / 'pip-check.log').open('w') as log:
        pip_check = subprocess.run(
            [str(python), '-I', '-m', 'pip', '--isolated', 'check'], stdout=log,
            stderr=subprocess.STDOUT, check=False, timeout=60)
    receipt['pip_check_returncode'] = pip_check.returncode
    child_code = """import json, os
from pathlib import Path
import snowflake.connector
connection = snowflake.connector.connect(account=os.environ['SNOWFLAKE_ACCOUNT'], host=os.environ['SNOWFLAKE_HOST'], authenticator='oauth', token=Path('/snowflake/session/token').read_text(), user=None, role='KUMO_SNOWFLAKE_TEST_ROLE', database='KUMO_TEST_DB', schema='KUMO_TEST', warehouse='KUMO_TEST_WH', login_timeout=30, network_timeout=60)
try:
    with connection.cursor() as cursor:
        cursor.execute('SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE()')
        print(json.dumps(dict(zip(['account','user','role'],cursor.fetchone()))))
finally: connection.close()
"""
    with (evidence / 'child-auth.stderr.log').open('w') as log:
        child = subprocess.run([str(python), '-I', '-c', child_code],
                               stdout=subprocess.PIPE, stderr=log, text=True,
                               check=False, timeout=90)
    receipt['child_auth_returncode'] = child.returncode
    (evidence / 'child-auth.stdout.json').write_text(child.stdout)
    if child.returncode == 0:
        observed = json.loads(child.stdout)
        assert all(observed[key] == identity[key]
                   for key in ['account', 'user', 'role'
                               ]), 'Fresh child identity differs from notebook'
        receipt['child_identity'] = observed
    else:
        receipt['status'] = 'blocked'
        receipt[
            'reason'] = 'Fresh-process authentication failed; inspect retained logs'
    if config['mode'] == 'preflight':
        if pip_check.returncode:
            receipt['status'] = 'blocked'
            receipt[
                'reason'] = 'Candidate dependencies are not satisfied; inspect pip-check.log'
        elif child.returncode == 0:
            check_code = "import importlib.util,json; from pathlib import Path; p=Path(" + repr(
                str(payload / 'wheel_mlops_lifecycle.py')
            ) + "); s=importlib.util.spec_from_file_location('lifecycle',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.check_install(Path(" + repr(
                str(wheel)) + "), " + repr(
                    config['expected_version']) + ", managed=True)))"
            with (evidence / 'candidate-admission.stderr.log').open('w') as log:
                admission = subprocess.run(
                    [str(python), '-I', '-c', check_code],
                    stdout=subprocess.PIPE, stderr=log, text=True, check=False,
                    timeout=90, env=dict(os.environ, MPLBACKEND='Agg'))
            (evidence / 'candidate-admission.json').write_text(admission.stdout)
            receipt['candidate_admission_returncode'] = admission.returncode
            receipt[
                'status'] = 'passed' if admission.returncode == 0 else 'blocked'
            if admission.returncode:
                receipt[
                    'reason'] = 'Candidate import/admission failed; inspect retained logs'
    elif pip_check.returncode == 0 and child.returncode == 0:
        spec = importlib.util.spec_from_file_location(
            'platform_lifecycle', payload / 'wheel_mlops_platform.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        result = runner.run_notebook(
            config=payload / 'snowflake.json', wheel=wheel,
            expected_version=config['expected_version'],
            output_dir=evidence / 'lifecycle', python_executable=python,
            attempt=attempt,
            phase='negative' if config['mode'] == 'negative' else None)
        receipt['lifecycle'] = result
        receipt['status'] = (result.get('summary')
                             or {}).get('status', 'failed')
        if result['returncode'] and receipt['status'] == 'passed':
            receipt['status'] = 'failed'
            receipt[
                'reason'] = 'Notebook worker exited unsuccessfully after writing its receipt'
    else:
        receipt['status'] = 'blocked'
        receipt[
            'reason'] = 'Dependency/authentication prerequisites did not pass; zero fits started'
except (ImportError, OSError, subprocess.TimeoutExpired,
        subprocess.CalledProcessError):
    receipt['status'] = 'blocked'
    receipt['error'] = traceback.format_exc()
except Exception:
    receipt['error'] = traceback.format_exc()
finally:
    exports = local / 'exports'
    exports.mkdir()
    receipt['logical_files'] = {}
    for path in sorted(evidence.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(evidence).as_posix()
        filename = 'evidence-' + hashlib.sha256(
            relative.encode()).hexdigest() + path.suffix
        target = exports / filename
        shutil.copyfile(path, target)
        receipt['logical_files'][filename] = relative
        receipt['artifacts'][filename] = hashlib.sha256(
            target.read_bytes()).hexdigest()
    (exports / 'lifecycle-result.json').write_text(
        json.dumps(receipt, indent=2, default=str))
    for path in sorted(exports.iterdir()):
        rows = session.file.put(str(path), stage + '/receipts',
                                auto_compress=False, overwrite=False)
        if not rows or any(
                str(item.status).upper() != 'UPLOADED' for item in rows):
            raise RuntimeError('Evidence upload incomplete: ' + path.name)
print(json.dumps(receipt, default=str))
if receipt['status'] != 'passed':
    summary = (receipt.get('lifecycle') or {}).get('summary') or {}
    phases = summary.get('phases', {})
    engine_passed = (
        config.get('mode') == 'lifecycle'
        and set(phases) == {'fit', 'score', 'evaluate', 'negative'} and all(
            phase.get('status') == 'passed' and phase.get('returncode') == 0
            for phase in phases.values()))
    if not engine_passed:
        raise RuntimeError('Lifecycle prerequisite/test is ' +
                           receipt['status'] + '; evidence exported')
