# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import signal
import subprocess
from types import SimpleNamespace

import kumo_core.diskgraph.runner as ingest_runner
import pytest
from kumo_core.diskgraph.runner import (
    DiskgraphCancelledError,
    DiskgraphNoSpaceError,
    DiskgraphRetryExhaustedError,
    NonRetryableDiskgraphError,
)


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


def test_default_worker_threads_queries_cgroup_aware_binary(monkeypatch):
    monkeypatch.setattr(ingest_runner, "_resolve_ingest_bin",
                        lambda: "/tmp/ingest")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="34\n")

    monkeypatch.setattr(ingest_runner.subprocess, "run", fake_run)

    assert ingest_runner._default_worker_threads() == 34
    assert calls == [(["/tmp/ingest", "--print-resolved-threads"], {
        "capture_output": True,
        "text": True,
        "timeout": 30,
    })]


@pytest.mark.parametrize(
    "outcome",
    [
        SimpleNamespace(returncode=1, stdout="34\n"),
        SimpleNamespace(returncode=0, stdout="not-a-number\n"),
        SimpleNamespace(returncode=0, stdout="0\n"),
        OSError("missing binary"),
        subprocess.TimeoutExpired("ingest", 30),
    ],
)
def test_default_worker_threads_falls_back_on_query_failure(
        monkeypatch, outcome):
    monkeypatch.setattr(ingest_runner, "_resolve_ingest_bin",
                        lambda: "/tmp/ingest")
    monkeypatch.setattr(ingest_runner, "_default_worker_threads_from_cpu_count",
                        lambda: 68)

    def fake_run(*_args, **_kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(ingest_runner.subprocess, "run", fake_run)

    assert ingest_runner._default_worker_threads() == 68


def test_pdeathsig_allows_pid_one_parent(monkeypatch):
    monkeypatch.setattr(ingest_runner, "_PRCTL", lambda *args: 0)
    monkeypatch.setattr(ingest_runner.os, "getppid", lambda: 1)
    monkeypatch.setattr(ingest_runner.os, "getpid", lambda: 99)

    killed = []
    monkeypatch.setattr(ingest_runner.os, "kill",
                        lambda pid, sig: killed.append((pid, sig)))

    ingest_runner._set_pdeathsig(signal.SIGKILL, expected_parent_pid=1)

    assert killed == []


def test_pdeathsig_kills_when_parent_changed(monkeypatch):
    monkeypatch.setattr(ingest_runner, "_PRCTL", lambda *args: 0)
    monkeypatch.setattr(ingest_runner.os, "getppid", lambda: 123)
    monkeypatch.setattr(ingest_runner.os, "getpid", lambda: 99)

    killed = []
    monkeypatch.setattr(ingest_runner.os, "kill",
                        lambda pid, sig: killed.append((pid, sig)))

    ingest_runner._set_pdeathsig(signal.SIGKILL, expected_parent_pid=1)

    assert killed == [(99, signal.SIGKILL)]


def test_runner_uses_active_python_for_child_process(tmp_path, monkeypatch):
    captured = {}

    class FinishedProcess:
        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setenv("DISKGRAPH_PYTHON", "/obsolete/python")
    monkeypatch.setattr(
        ingest_runner.subprocess,
        "Popen",
        lambda cmd, **kwargs: captured.update(kwargs) or FinishedProcess(),
    )

    command = ["/bin/ingest", "--output", str(tmp_path / "output")]
    assert ingest_runner._run_with_heartbeat(
        command,
        mem_limit_bytes=None,
        heartbeat_interval_s=10,
    ) == (0, '')
    assert captured["env"]["DISKGRAPH_PYTHON"] == ingest_runner.sys.executable


def test_runner_removes_requested_environment_values(monkeypatch):
    captured = {}

    class FinishedProcess:
        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setenv("SOURCE_CREDENTIAL_FILE", "/tmp/stale.cfg")
    monkeypatch.setattr(
        ingest_runner.subprocess,
        "Popen",
        lambda cmd, **kwargs: captured.update(kwargs) or FinishedProcess(),
    )

    assert ingest_runner._run_with_heartbeat(
        ["/bin/ingest", "--output", "/tmp/output"],
        mem_limit_bytes=None,
        heartbeat_interval_s=10,
        subprocess_environment={"SOURCE_CREDENTIAL_FILE": None},
    ) == (0, '')
    assert "SOURCE_CREDENTIAL_FILE" not in captured["env"]


def test_runner_overlays_subprocess_environment_without_mutating_parent(
        monkeypatch):
    captured = {}

    class FinishedProcess:
        def wait(self, timeout):
            return 0

        def poll(self):
            return 0

    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'parent-key')
    monkeypatch.setattr(
        ingest_runner.subprocess,
        'Popen',
        lambda cmd, **kwargs: captured.update(kwargs) or FinishedProcess(),
    )

    assert ingest_runner._run_with_heartbeat(
        ['/bin/ingest', '--output', '/tmp/output'],
        mem_limit_bytes=None,
        heartbeat_interval_s=10,
        subprocess_environment={
            'AWS_ACCESS_KEY_ID': 'child-key',
            'AWS_SESSION_TOKEN': 'child-token',
        },
    ) == (0, '')
    assert captured['env']['AWS_ACCESS_KEY_ID'] == 'child-key'
    assert captured['env']['AWS_SESSION_TOKEN'] == 'child-token'
    assert ingest_runner.os.environ['AWS_ACCESS_KEY_ID'] == 'parent-key'


def test_ephemeral_file_is_removed_after_requested_phase(tmp_path):
    output = tmp_path / "output"
    marker_dir = output / "_tmp"
    marker_dir.mkdir(parents=True)
    credentials = tmp_path / "source.cfg"
    credentials.write_text("secret\n")
    cleanup = {"encode": (str(credentials), )}

    ingest_runner._remove_files_for_completed_phases(output, cleanup)
    assert credentials.is_file()
    (marker_dir / "_phase_complete_encode").touch()
    ingest_runner._remove_files_for_completed_phases(output, cleanup)
    assert not credentials.exists()


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        ((), "encode"),
        (("context", ), "encode"),
        (("encode", ), "pk_map"),
        (("context", "encode"), "pk_map"),
        (("encode", "pk_map"), "edges_map"),
        (("encode", "pk_map", "edges_map"), "index"),
        (("encode", "pk_map", "edges_map", "index"), "index"),
        (("context", "encode", "pk_map", "edges_map", "index"), "index"),
    ],
)
def test_resume_phase_uses_only_durable_cli_phases(
        tmp_path, completed, expected):
    output = tmp_path / "output"
    marker_dir = output / "_tmp"
    marker_dir.mkdir(parents=True)
    for phase in completed:
        (marker_dir / f"_phase_complete_{phase}").touch()

    assert ingest_runner._resume_phase_for_attempt(output) == expected


def test_resume_phase_without_output_or_tmp_starts_at_encode(tmp_path):
    assert ingest_runner._resume_phase_for_attempt(
        tmp_path / "missing-output") == "encode"


def test_resume_cli_mapping_matches_durable_phase_order():
    assert set(ingest_runner.PHASE_TO_CLI) == set(
        ingest_runner.RESUMABLE_PHASE_ORDER)


@pytest.mark.parametrize(
    ("phase", "cli_phase"),
    [
        ("encode", "encode"),
        ("pk_map", "pk-map"),
        ("edges_map", "edges-map"),
        ("index", "index"),
    ],
)
def test_build_ingest_cmd_spells_only_supported_resume_phases(
        tmp_path, phase, cli_phase):
    cmd = ingest_runner._build_ingest_cmd(
        "/bin/ingest",
        tmp_path / "config.json",
        tmp_path / "output",
        {name: 1 for name in ingest_runner.PHASE_NAMES},
        col_stats=False,
        cat_count_strategy="none",
        resume_from_phase=phase,
        output_dataplane_dir=None,
        success_marker_content=None,
        extra_args=None,
    )

    index = cmd.index("--resume-from-phase")
    assert cmd[index + 1] == cli_phase
    assert "context" not in cmd[index:index + 2]


def test_build_ingest_cmd_rejects_context_as_resume_phase(tmp_path):
    with pytest.raises(KeyError, match="context"):
        ingest_runner._build_ingest_cmd(
            "/bin/ingest",
            tmp_path / "config.json",
            tmp_path / "output",
            {name: 1 for name in ingest_runner.PHASE_NAMES},
            col_stats=False,
            cat_count_strategy="none",
            resume_from_phase="context",
            output_dataplane_dir=None,
            success_marker_content=None,
            extra_args=None,
        )


# =============================================================================
# run_ingest — exit-code classification + halving + retry budget
# =============================================================================
# These tests stub `_run_with_heartbeat` to inject specific exit codes and
# assert that `run_ingest` translates them into the right
# `NonRetryableDiskgraphError` / `DiskgraphRetryExhaustedError` and either
# halves the failing phase's thread count or surfaces the failure.


def _stub_runner(
    monkeypatch,
    outcome_sequence: list[tuple[int, str]],
) -> list[dict]:
    """Replace `_run_with_heartbeat` with exact outcomes in order.

    Captures each invocation's phase_threads snapshot so tests can assert
    halving behavior.
    """
    monkeypatch.setattr(ingest_runner, "_resolve_ingest_bin",
                        lambda: "/bin/true")

    calls: list[dict] = []
    iterator = iter(outcome_sequence)

    def _fake_run(cmd, *, mem_limit_bytes, heartbeat_interval_s):
        # Capture per-phase thread counts by parsing the constructed
        # argv. Each phase appears as `--<phase>-threads N` exactly once.
        phase_threads = {}
        for i, tok in enumerate(cmd):
            if tok.startswith('--') and tok.endswith('-threads'):
                phase = tok[2:-len('-threads')].replace('-', '_')
                phase_threads[phase] = int(cmd[i + 1])
        calls.append({'cmd': cmd, 'phase_threads': phase_threads})
        try:
            return next(iterator)
        except StopIteration:
            raise AssertionError(
                f"run_ingest spawned more attempts than outcome_sequence "
                f"provides ({len(outcome_sequence)}); current call "
                f"#{len(calls)} "
                f"with phase_threads={phase_threads}. Either the retry "
                f"budget regressed or outcome_sequence is too short.")

    monkeypatch.setattr(ingest_runner, "_run_with_heartbeat", _fake_run)
    return calls


def _diagnostic_envelope(*, visibility='ordinary', summary='encoder invariant'):
    chain_kind = ('restricted_boundary'
                  if visibility == 'restricted' else 'error')
    return {
        'version': 1,
        'event': 'diskgraph.ingest_failed',
        'diagnostic_id': f"dg-{'a' * 24}",
        'visibility': visibility,
        'component': 'diskgraph-ingest',
        'operation': 'ingest',
        'category': 'internal',
        'code': 'ingest.unexpected_failure',
        'retryable': False,
        'summary': summary,
        'public': None,
        'fields': {
            'attempt': 1,
            'phase': 'encode',
            'exit_code': None,
            'signal': None,
        },
        'chain': [{
            'kind': chain_kind,
            'summary': summary,
        }],
        'frames': [],
        'fingerprint': f"dg-v1-{'b' * 64}",
        'truncated': {
            'chain': False,
            'frames': False,
            'summary': False,
        },
    }


@pytest.mark.parametrize('mutate', [
    lambda value: value.update({'unexpected': 'raw-canary'}),
    lambda value: value.update({'version': 2}),
    lambda value: value.update({'summary': 'line one\nforged line'}),
    lambda value: value.update({
        'public': {
            'code': 'diskgraph_ingest_validation',
            'message': 'raw-canary',
            'non_retryable': True,
        },
    }),
    lambda value: value['frames'].append({
        'module': 'diskgraph_ingest::encode',
        'file': '/customer/absolute/path.rs',
        'line': 1,
    }),
])
def test_diagnostic_reader_rejects_untrusted_schema_without_logging_payload(
        mutate, tmp_path, caplog):
    payload = _diagnostic_envelope()
    mutate(payload)
    (tmp_path / '.diagnostic.json').write_text(json.dumps(payload))

    with caplog.at_level(logging.WARNING, logger=ingest_runner.logger.name):
        assert ingest_runner._read_diagnostic(tmp_path) is None

    assert 'raw-canary' not in caplog.text
    assert 'forged line' not in caplog.text
    assert '/customer/absolute/path.rs' not in caplog.text


_FULL_DISK_USAGE = SimpleNamespace(total=1000, used=995, free=5)
_ROOMY_DISK_USAGE = SimpleNamespace(total=1000, used=100, free=900)


def _stub_disk_usage(monkeypatch, outcomes_by_path):
    calls = []

    def fake_disk_usage(path):
        calls.append(path)
        if path not in outcomes_by_path:
            raise AssertionError(f'unexpected disk usage path: {path}')
        outcome = outcomes_by_path[path]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(ingest_runner.shutil, 'disk_usage', fake_disk_usage)
    return calls


def _run_fatal_ingest(monkeypatch, output_dir, output_dataplane_dir):
    _stub_runner(monkeypatch, outcome_sequence=[(1, '')])
    return ingest_runner.run_ingest(
        ingest_config={'tables': []},
        output_dir=output_dir,
        output_dataplane_dir=output_dataplane_dir,
        initial_threads=8,
    )


def test_run_ingest_applies_resolved_default_to_every_phase(
        monkeypatch, tmp_path):
    calls = _stub_runner(monkeypatch, outcome_sequence=[(1, '')])
    monkeypatch.setattr(ingest_runner, "_default_worker_threads", lambda: 34)

    with pytest.raises(NonRetryableDiskgraphError, match="exit code 1"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / "out",
            initial_threads=None,
        )

    assert calls[0]["phase_threads"] == {
        "context": 34,
        "encode": 34,
        "pk_map": 34,
        "edges_map": 34,
        "index": 34,
    }


def test_run_ingest_fatal_exit_raises_non_retryable(monkeypatch, tmp_path):
    """A non-zero, non-OOM exit code is treated as a deterministic
    application error — schema mismatch, invalid column type, etc. The
    binary's panic=abort means OOMs always show up as 134/137, so any
    other non-zero rc cannot be helped by halving and must surface as
    `NonRetryableDiskgraphError` after one attempt.
    """
    _stub_runner(monkeypatch, outcome_sequence=[(1, '')])

    with pytest.raises(NonRetryableDiskgraphError, match="exit code 1"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )


def test_run_ingest_completed_index_state_reruns_index_for_finalization(
        monkeypatch, tmp_path):
    output = tmp_path / "out"
    marker_dir = output / "_tmp"
    marker_dir.mkdir(parents=True)
    for phase in ("encode", "pk_map", "edges_map", "index"):
        (marker_dir / f"_phase_complete_{phase}").touch()
    calls = _stub_runner(monkeypatch, outcome_sequence=[(2, "")])

    with pytest.raises(NonRetryableDiskgraphError):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=output,
            initial_threads=8,
        )

    cmd = calls[0]["cmd"]
    index = cmd.index("--resume-from-phase")
    assert cmd[index:index + 2] == ["--resume-from-phase", "index"]


def test_run_ingest_no_space_raises_retryable(monkeypatch, tmp_path):
    """A non-OOM exit on a (nearly) full SSD is a disk-full, not a schema
    error. It must surface as a distinct, RETRYABLE ``DiskgraphNoSpaceError``
    so the workflow can re-place on a larger node — not the non-retryable
    schema-error path.
    """
    import types

    _stub_runner(monkeypatch, outcome_sequence=[(1, '')])
    # < _NO_SPACE_FREE_FRACTION (2%) free → treated as out-of-space.
    monkeypatch.setattr(
        ingest_runner.shutil, "disk_usage",
        lambda p: types.SimpleNamespace(total=1000, used=995, free=5))

    with pytest.raises(DiskgraphNoSpaceError, match="ran out of local SSD"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )
    assert DiskgraphNoSpaceError().non_retryable is False


def test_run_ingest_local_dataplane_no_space_is_actionable_non_retryable(
        monkeypatch, tmp_path):
    """A full local dataplane mirror is not a schema error and cannot be
    repaired by moving the ingest to a larger DiskGraph node."""
    output_dir = tmp_path / 'nvme' / 'out'
    dataplane_dir = (tmp_path / 'dataplane' / 'diskgraph_snapshots' / 'graph' /
                     'version')
    dataplane_dir.mkdir(parents=True)
    calls = _stub_disk_usage(monkeypatch, {
        dataplane_dir: _FULL_DISK_USAGE,
        output_dir: _ROOMY_DISK_USAGE,
    })

    with pytest.raises(NonRetryableDiskgraphError) as exc_info:
        _run_fatal_ingest(monkeypatch, output_dir, str(dataplane_dir))

    assert type(exc_info.value).__name__ == 'DiskgraphDataplaneNoSpaceError'
    assert exc_info.value.non_retryable is True
    message = str(exc_info.value)
    assert str(dataplane_dir) in message
    assert 'data_plane_storage_root' in message
    assert 'larger DiskGraph node' in message
    assert 'will not fix' in message
    assert 'Free or expand the dataplane volume' in message
    assert 'object storage' in message
    assert calls == [dataplane_dir]


def test_run_ingest_dataplane_no_space_precedes_nvme_no_space(
        monkeypatch, tmp_path):
    """If both filesystems are full, report the terminal dataplane ceiling
    instead of triggering futile NVMe node re-placement."""
    output_dir = tmp_path / 'nvme' / 'out'
    dataplane_dir = tmp_path / 'dataplane' / 'snapshot'
    dataplane_dir.mkdir(parents=True)
    calls = _stub_disk_usage(monkeypatch, {
        dataplane_dir: _FULL_DISK_USAGE,
        output_dir: _FULL_DISK_USAGE,
    })

    with pytest.raises(NonRetryableDiskgraphError) as exc_info:
        _run_fatal_ingest(monkeypatch, output_dir, str(dataplane_dir))

    assert type(exc_info.value).__name__ == 'DiskgraphDataplaneNoSpaceError'
    assert calls == [dataplane_dir]


def test_run_ingest_nvme_no_space_remains_retryable_with_dataplane_room(
        monkeypatch, tmp_path):
    """An ample local dataplane does not change the existing retryable NVMe
    classification and adaptive re-placement contract."""
    output_dir = tmp_path / 'nvme' / 'out'
    dataplane_dir = tmp_path / 'dataplane' / 'snapshot'
    dataplane_dir.mkdir(parents=True)
    calls = _stub_disk_usage(monkeypatch, {
        dataplane_dir: _ROOMY_DISK_USAGE,
        output_dir: _FULL_DISK_USAGE,
    })

    with pytest.raises(DiskgraphNoSpaceError, match='ran out of local SSD'):
        _run_fatal_ingest(monkeypatch, output_dir, str(dataplane_dir))

    assert calls == [dataplane_dir, output_dir]


@pytest.mark.parametrize('dataplane_uri', [
    's3://bucket/diskgraph_snapshots/graph/version',
    'gs://bucket/diskgraph_snapshots/graph/version',
    'az://account/container/diskgraph_snapshots/graph/version',
])
def test_run_ingest_remote_dataplane_is_not_probed(dataplane_uri, monkeypatch,
                                                   tmp_path):
    """Object-store dataplane URIs must never reach local disk_usage."""
    output_dir = tmp_path / 'nvme' / 'out'
    calls = _stub_disk_usage(monkeypatch, {
        output_dir: _ROOMY_DISK_USAGE,
    })

    with pytest.raises(NonRetryableDiskgraphError, match='exit code 1'):
        _run_fatal_ingest(monkeypatch, output_dir, dataplane_uri)

    assert calls == [output_dir]


def test_run_ingest_dataplane_probe_uses_closest_existing_parent(
        monkeypatch, tmp_path):
    """ENOSPC can prevent creation of the snapshot leaf, so capacity must be
    read from the closest existing parent on the same filesystem."""
    output_dir = tmp_path / 'nvme' / 'out'
    dataplane_root = tmp_path / 'dataplane'
    dataplane_root.mkdir()
    dataplane_dir = (dataplane_root / 'diskgraph_snapshots' / 'graph' /
                     'version')
    calls = _stub_disk_usage(monkeypatch, {
        dataplane_root: _FULL_DISK_USAGE,
        output_dir: _ROOMY_DISK_USAGE,
    })

    with pytest.raises(NonRetryableDiskgraphError) as exc_info:
        _run_fatal_ingest(monkeypatch, output_dir, str(dataplane_dir))

    assert type(exc_info.value).__name__ == 'DiskgraphDataplaneNoSpaceError'
    assert calls == [dataplane_root]


def test_run_ingest_dataplane_probe_error_falls_back(monkeypatch, tmp_path):
    """An unreadable local dataplane must preserve the existing fatal
    fallback instead of hiding the original ingest failure."""
    output_dir = tmp_path / 'nvme' / 'out'
    dataplane_dir = tmp_path / 'dataplane' / 'snapshot'
    dataplane_dir.mkdir(parents=True)
    calls = _stub_disk_usage(
        monkeypatch, {
            dataplane_dir: OSError('capacity unavailable'),
            output_dir: _ROOMY_DISK_USAGE,
        })

    with pytest.raises(NonRetryableDiskgraphError, match='exit code 1'):
        _run_fatal_ingest(monkeypatch, output_dir, str(dataplane_dir))

    assert calls == [dataplane_dir, output_dir]


def test_run_ingest_dataplane_stat_error_falls_back(monkeypatch, tmp_path):
    """An access/stat failure is not proof that the leaf is missing. Do not
    climb to and classify a potentially different ancestor filesystem."""
    output_dir = tmp_path / 'nvme' / 'out'
    dataplane_dir = tmp_path / 'dataplane' / 'snapshot'
    dataplane_dir.mkdir(parents=True)
    real_stat = ingest_runner.Path.stat
    real_exists = ingest_runner.Path.exists

    def fake_stat(path, *args, **kwargs):
        if path == dataplane_dir:
            raise PermissionError('dataplane stat denied')
        return real_stat(path, *args, **kwargs)

    def fake_exists(path):
        if path == dataplane_dir:
            raise PermissionError('dataplane exists denied')
        return real_exists(path)

    monkeypatch.setattr(ingest_runner.Path, 'exists', fake_exists)
    monkeypatch.setattr(ingest_runner.Path, 'stat', fake_stat)
    calls = _stub_disk_usage(monkeypatch, {
        output_dir: _ROOMY_DISK_USAGE,
    })

    with pytest.raises(NonRetryableDiskgraphError, match='exit code 1'):
        _run_fatal_ingest(monkeypatch, output_dir, str(dataplane_dir))

    assert calls == [output_dir]


def test_run_ingest_fatal_low_nofile_limit_adds_hint(monkeypatch, tmp_path):
    """A fatal exit under a low open-file limit (e.g. Docker's 1024 default)
    may be fd exhaustion in a writer thread; the error must say so."""
    _stub_runner(monkeypatch, outcome_sequence=[(1, '')])
    monkeypatch.setattr(ingest_runner.resource, "getrlimit", lambda r:
                        (1024, 524288))

    with pytest.raises(NonRetryableDiskgraphError,
                       match="open-file limit is only 1024"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )


def test_run_ingest_fatal_high_nofile_limit_no_hint(monkeypatch, tmp_path):
    """With an ample open-file limit the fd hint must not appear."""
    _stub_runner(monkeypatch, outcome_sequence=[(1, '')])
    monkeypatch.setattr(ingest_runner.resource, "getrlimit", lambda r:
                        (65535, 524288))

    with pytest.raises(NonRetryableDiskgraphError) as exc_info:
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )
    assert "open-file limit" not in str(exc_info.value)


def test_run_ingest_fatal_with_ample_disk_stays_non_retryable(
        monkeypatch, tmp_path):
    """The same non-OOM exit with plenty of free disk is a real schema/data
    error and must stay non-retryable (no false no-space classification)."""
    import types

    _stub_runner(monkeypatch, outcome_sequence=[(1, '')])
    monkeypatch.setattr(
        ingest_runner.shutil, "disk_usage",
        lambda p: types.SimpleNamespace(total=1000, used=100, free=900))

    with pytest.raises(NonRetryableDiskgraphError, match="exit code 1"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )


@pytest.mark.parametrize("rc", [143, -15])
def test_run_ingest_cancel_exit_raises_retryable_cancelled(rc, monkeypatch,
                                                           tmp_path):
    """SIGTERM means an external signal cancelled the binary. The
    wrapper must not retry — the caller wanted us to stop — and must
    surface `DiskgraphCancelledError`, which lets Temporal retry the whole
    activity on a fresh worker. Parametrized over both
    forms: shell-converted 143 (rare, since `_run_with_heartbeat` uses
    `Popen` directly) and the direct-Popen negative form -15.
    """
    _stub_runner(monkeypatch, outcome_sequence=[(rc, '')])

    with pytest.raises(DiskgraphCancelledError, match="cancelled") as caught:
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )
    assert caught.value.non_retryable is False


@pytest.mark.parametrize("rc", [134, 137, -6, -9])
def test_run_ingest_oom_at_floor_raises_non_retryable(rc, monkeypatch,
                                                      tmp_path):
    """When the initial thread count is already at the halving floor
    (4 by default) and the binary OOMs, there is no smaller thread
    count to halve to. The wrapper must give up immediately with a
    `NonRetryableDiskgraphError` mentioning the lowest thread count.

    Parametrized over all four OOM exit codes: SIGABRT 134 (panic=abort
    on allocator failure), SIGKILL 137 (kernel OOM-killer), and the
    direct-Popen negative forms -6 and -9 — production almost certainly
    sees the negative forms since `_run_with_heartbeat` uses `Popen`
    directly. Locks the rc-class mapping at `_classify_exit`.
    """
    _stub_runner(monkeypatch, outcome_sequence=[(rc, '')])

    with pytest.raises(NonRetryableDiskgraphError, match="lowest thread count"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=4,  # already at the _THREAD_FLOOR
        )


def test_run_ingest_oom_halves_failing_phase(monkeypatch, tmp_path):
    """Persistent OOMs (kernel SIGKILL = rc 137) should trigger halving
    on the failing phase only. With `initial_threads=16` the halving
    sequence on the lowest missing phase ("context") is 16 → 8 → 4
    (floor) before we give up. The wrapper exits with
    `NonRetryableDiskgraphError` once the floor is reached, and the
    halving sequence is observable in the per-call phase_threads
    snapshots.
    """
    calls = _stub_runner(
        monkeypatch,
        outcome_sequence=[(137, ''), (137, ''), (137, ''), (137, '')],
    )

    with pytest.raises(NonRetryableDiskgraphError, match="lowest thread count"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=16,
            max_inner_attempts=5,
        )

    # Halving touches only the "context" phase (the lowest missing one
    # since no phase markers were written). Other phases keep 16.
    context_threads = [c['phase_threads']['context'] for c in calls]
    assert context_threads == [
        16, 8, 4
    ], (f"expected halving 16→8→4 on the failing phase; got "
        f"{context_threads}")
    for c in calls:
        for phase in ('encode', 'pk_map', 'edges_map', 'index'):
            assert c['phase_threads'][phase] == 16, (
                f"non-failing phase {phase} must not be halved")
    assert "--resume-from-phase" not in calls[0]["cmd"]
    for call in calls[1:]:
        index = call["cmd"].index("--resume-from-phase")
        assert call["cmd"][index:index + 2] == [
            "--resume-from-phase", "encode"
        ]


def test_run_ingest_retry_budget_exhausted_raises(monkeypatch, tmp_path):
    """If OOMs keep happening but the halving budget (max_inner_attempts)
    is consumed before reaching the floor, the wrapper raises
    `DiskgraphRetryExhaustedError`. This separates "we can't go lower"
    (NonRetryable) from "we ran out of attempts" (RetryExhausted),
    which matter for downstream classification.
    """
    # Cap attempts at 2 so even a halving-friendly initial count exits
    # via the budget exhaustion path rather than via floor.
    calls = _stub_runner(
        monkeypatch,
        outcome_sequence=[(137, ''), (137, '')],
    )

    with pytest.raises(DiskgraphRetryExhaustedError,
                       match="exhausted automatic memory-recovery retries"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=64,  # plenty of room to halve
            max_inner_attempts=2,
        )

    # Both attempts ran (we didn't bail early via floor), and the
    # second attempt saw halved threads.
    assert len(calls) == 2
    assert calls[0]['phase_threads']['context'] == 64
    assert calls[1]['phase_threads']['context'] == 32


def test_run_ingest_success_without_stats_file_raises_retry_exhausted(
        monkeypatch, tmp_path):
    """Exit 0 with no `ingest_stats.json` written means the binary
    reported success but failed to produce its expected artifact. This
    is an internal inconsistency — neither a non-retryable bug nor an
    OOM — so the wrapper surfaces `DiskgraphRetryExhaustedError` with
    a "contact Kumo" message instead of silently returning empty stats.
    """
    _stub_runner(monkeypatch, outcome_sequence=[(0, '')])

    with pytest.raises(DiskgraphRetryExhaustedError,
                       match="stats output.*was not produced"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=tmp_path / 'out',
            initial_threads=8,
        )


def test_run_ingest_oom_with_all_phase_markers_raises_non_retryable(
        monkeypatch, tmp_path):
    """OOM after every phase already completed is the "finalize-step
    failure or race between marker write and crash" branch in
    `_halve_failing_phase`: `_next_phase_to_run` returns None, halving
    is impossible, and the wrapper must surface
    `NonRetryableDiskgraphError` with a "please contact Kumo" message
    rather than spinning on a phase it can't identify.
    """
    output_dir = tmp_path / 'out'
    output_dir.mkdir(parents=True)
    # Pre-seed every phase-complete marker so `_next_phase_to_run`
    # returns None — mimicking a crash between the final phase marker
    # write and process exit.
    (output_dir / '_tmp').mkdir()
    from kumo_core.diskgraph.runner import PHASE_ORDER
    for phase in PHASE_ORDER:
        (output_dir / '_tmp' / f'_phase_complete_{phase}').touch()

    _stub_runner(monkeypatch, outcome_sequence=[(137, '')])

    with pytest.raises(NonRetryableDiskgraphError,
                       match="after all pipeline phases had completed"):
        ingest_runner.run_ingest(
            ingest_config={"tables": []},
            output_dir=output_dir,
            initial_threads=16,
        )


def test_run_ingest_success_returns_parsed_stats(monkeypatch, tmp_path):
    """Happy path: rc=0 + ingest_stats.json present → returns an
    IngestRunResult carrying the parsed stats and the per-phase thread
    counts. This anchors the rest of the failure assertions above by
    proving the same scaffolding produces the success branch when the
    artifact is in place.
    """
    output_dir = tmp_path / 'out'
    output_dir.mkdir(parents=True)
    stats_blob = {"num_rows": 42, "tables": ["users"]}
    (output_dir / 'ingest_stats.json').write_text(json.dumps(stats_blob))

    _stub_runner(monkeypatch, outcome_sequence=[(0, '')])

    result = ingest_runner.run_ingest(
        ingest_config={"tables": []},
        output_dir=output_dir,
        initial_threads=8,
    )
    assert result.stats == stats_blob
    assert result.attempts == 1
    assert result.phase_threads['context'] == 8


def test_no_space_error_type_name_matches_pool_constant():
    """The workflow matches no-space failures by ``ApplicationError.type``,
    which Temporal sets to the exception class name. Pin the class name to
    the string constant workflow code uses (it cannot import this module)."""
    from pool.errors import NO_SPACE_ERROR_TYPE
    assert DiskgraphNoSpaceError.__name__ == NO_SPACE_ERROR_TYPE


def test_dataplane_no_space_error_does_not_trigger_pool_replacement():
    """The dataplane type must remain distinct from the exact error name that
    the workflow maps to a larger DiskGraph node."""
    from pool.errors import NO_SPACE_ERROR_TYPE
    error_type = getattr(ingest_runner, 'DiskgraphDataplaneNoSpaceError', None)

    assert error_type is not None
    assert issubclass(error_type, NonRetryableDiskgraphError)
    assert error_type.__name__ != NO_SPACE_ERROR_TYPE
    assert error_type().non_retryable is True
