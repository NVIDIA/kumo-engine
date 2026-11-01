# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise local process cleanup and retained-result packaging without models."""
import importlib.util
import json
import sys
import tarfile
import time
from pathlib import Path

import pytest

SOURCE = Path(
    __file__).parents[2] / 'packages/kumo-engine/scripts/nightly_host.py'


def host():
    assert SOURCE.is_file(), 'The VM-side nightly host is missing'
    spec = importlib.util.spec_from_file_location('nightly_host', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('reference_only', [True, False])
@pytest.mark.parametrize('tamper', [False, True])
def test_both_install_modes_use_their_frozen_dependency_profile(
        tmp_path, reference_only, tamper):
    module = host()
    profiles = {
        'reference_dependencies': 'reference.json',
        'candidate_dependencies': 'candidate.json',
        'files': {},
    }
    for name, version in [('reference.json', '2.6.5'),
                          ('candidate.json', '2.6.6')]:
        path = tmp_path / name
        path.write_text(json.dumps({'torch': '2.14.1',
                                   'pytorch-lightning': version}))
        profiles['files'][name] = module.digest(path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(profiles))
    checksum = module.digest(manifest)
    if tamper:
        (tmp_path / ('reference.json' if reference_only else
                     'candidate.json')).write_text('{}')
        with pytest.raises(ValueError, match='profile'):
            module.dependency_profile(tmp_path, checksum, reference_only)
    else:
        assert module.dependency_profile(tmp_path, checksum, reference_only) == {
            'torch': '2.14.1',
            'pytorch-lightning': '2.6.5' if reference_only else '2.6.6',
        }


def test_deadline_stops_the_process_and_its_child(tmp_path):
    module = host()
    marker = tmp_path / 'ticks'
    child = (
        'import time; from pathlib import Path; '
        f'p=Path({str(marker)!r}); '
        'exec("while True:\\n p.write_text(str(time.time_ns()))\\n time.sleep(.01)")'
    )
    parent = (
        'import subprocess,sys,time; '
        f'subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(30)')
    with pytest.raises(TimeoutError):
        module.run_process([sys.executable, '-c', parent],
                           tmp_path / 'process.log',
                           time.monotonic() + .7, lambda: False)
    assert marker.is_file(), 'The real child must start before cancellation'
    value = marker.read_bytes()
    time.sleep(.08)
    assert marker.read_bytes() == value


def test_cancel_signal_and_child_failure_cannot_pass(tmp_path):
    module = host()
    with pytest.raises(InterruptedError):
        module.run_process(
            [sys.executable, '-c', 'import time; time.sleep(30)'],
            tmp_path / 'cancel.log',
            time.monotonic() + 5, lambda: True)
    with pytest.raises(Exception, match='7'):
        module.run_process([sys.executable, '-c', 'raise SystemExit(7)'],
                           tmp_path / 'failed.log',
                           time.monotonic() + 5, lambda: False)


def test_host_lock_excludes_a_second_workload(tmp_path):
    module = host()
    path = tmp_path / 'host.lock'
    with module.host_lock(path):
        with pytest.raises(BlockingIOError):
            with module.host_lock(path):
                pytest.fail('A second job acquired the held lock')
    with module.host_lock(path):
        pass


def test_portable_results_preserve_failed_reports_and_skip_full_indexes(
        tmp_path):
    module = host()
    root = tmp_path / 'job'
    root.mkdir()
    for name in [
            'result/nightly-result.json',
            'result/static_link-normal/result.json', 'run.log',
            'result/work/graph/index.json', 'result/work/model.ckpt',
            'result/static_link-normal/reload-ranked.json'
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"FAIL"}')
    archive = tmp_path / 'portable.tar.gz'
    module.portable_archive(root, archive)
    with tarfile.open(archive) as source:
        assert set(source.getnames()) == {
            'result/nightly-result.json',
            'result/static_link-normal/result.json', 'run.log'
        }
        assert source.extractfile(
            'result/nightly-result.json').read() == b'{"status":"FAIL"}'


def test_gpu_query_failure_preserves_measured_host_capacity(
        tmp_path, monkeypatch):
    import subprocess
    module = host()
    monkeypatch.setattr(module, 'BASE', tmp_path)
    monkeypatch.setattr(module.sys, 'version_info', (3, 12))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == '/sys/devices/virtual/dmi/id/board_asset_tag':
            return 'i-06dd0e3c558780731'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(module.subprocess, 'check_output', unavailable)
    output = tmp_path / 'capacity.json'
    with pytest.raises(subprocess.CalledProcessError):
        module.capacity(output)
    import json
    result = json.loads(output.read_text())
    assert result['cpu_count'] > 0 and result['system_memory_bytes'] > 0
    assert result['nvme_available_bytes'] > 0 and result[
        'available_memory_bytes'] > 0


def test_host_keeps_runner_output_fresh_and_copies_reference_before_compare(
        tmp_path, monkeypatch):
    """Stub cloud/install/CUDA work; exercise real runner output admission."""
    from types import SimpleNamespace

    module = host()
    spec = importlib.util.spec_from_file_location(
        'nightly_admission', SOURCE.with_name('nightly_check.py'))
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    data = tmp_path / 'data'
    data.mkdir()
    profile = data / 'profile.json'
    profile.write_text('{"torch":"2.14.1"}')
    manifest = data / 'manifest.json'
    manifest.write_text(json.dumps(dict(
        reference_dependencies='profile.json',
        candidate_dependencies='profile.json',
        files={'profile.json': module.digest(profile)})))
    archive = tmp_path / 'inputs.tar'
    with tarfile.open(archive, 'w') as bundle:
        for path in data.iterdir():
            bundle.add(path, arcname=path.name)
    reference = tmp_path / 'original-reference.json'
    reference.write_text('{"reference":"preserve these exact bytes"}')
    wheel = tmp_path / 'candidate.whl'
    wheel.write_bytes(b'wheel installation is outside this test')
    job = tmp_path / 'job'
    job.mkdir()
    request = job / 'request.json'
    request.write_text(json.dumps(dict(
        instance_id=module.INSTANCE, mode='nightly', run_id='123-1',
        source_sha='a' * 40, reference_only=False, payload={'files': {}},
        wheel=dict(filename=wheel.name, key='wheel', sha256=module.digest(wheel),
                   version='test'), build={}, inputs=dict(
                       bucket='fixture', key='inputs', sha256=module.digest(archive),
                       input_manifest_sha256=module.digest(manifest),
                       reference_key='reference',
                       reference_sha256=module.digest(reference)))))
    monkeypatch.setenv('NIGHTLY_REQUEST_SHA256', module.digest(request))
    monkeypatch.setattr(module, 'BASE', tmp_path)
    monkeypatch.setattr(module, 'capacity', lambda path: {'status': 'MEASURED'})

    class ReachedInputs(Exception):
        pass

    def stop_before_cuda(*args):
        raise ReachedInputs

    monkeypatch.setattr(checker, 'load_inputs', stop_before_cuda)

    def external(command, *args, **kwargs):
        if command[0] == 'aws':
            key = command[5].rsplit('/', 1)[-1]
            Path(command[6]).write_bytes(
                {'inputs': archive, 'reference': reference,
                 'wheel': wheel}[key].read_bytes())
        elif len(command) > 3 and command[2].endswith('nightly_check.py'):
            if command[3] == 'run':
                assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
                with pytest.raises(ReachedInputs):
                    checker.run(SimpleNamespace(input_root=data,
                                                input_sha256='',
                                                output_dir=job / 'result'))
            elif command[3] == 'compare':
                assert (job / 'result/reference.json').read_bytes() == reference.read_bytes()

    monkeypatch.setattr(module, 'run_process', external)
    monkeypatch.setattr(module.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(
                            returncode=0, stdout='', stderr=''))
    module.main(request)
    receipt = json.loads((job / 'receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['processes_ended']
