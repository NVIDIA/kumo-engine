# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for pooled Kumo subprocess session supervision."""
import signal
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import pool.subprocess_supervisor as supervisor


def test_zombie_leader_with_live_sibling_is_live() -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='Z',
        task_states=((100, 'Z'), (101, 'R')),
    )

    assert member.has_live_tasks


def test_empty_task_snapshot_falls_back_to_live_leader() -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='S',
        task_states=(),
    )

    assert member.has_live_tasks


@pytest.mark.parametrize(('task_group_exited', 'expected_live'), [
    (False, True),
    (None, True),
    (True, False),
])
def test_empty_zombie_task_snapshot_uses_pidfd_group_state(
    task_group_exited: bool | None,
    expected_live: bool,
) -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='Z',
        task_states=(),
        task_group_exited=task_group_exited,
    )

    assert member.has_live_tasks is expected_live


@pytest.mark.parametrize(('task_group_exited', 'expected_live'), [
    (False, True),
    (None, True),
    (True, False),
])
def test_unavailable_task_snapshot_uses_pidfd_group_state(
    task_group_exited: bool | None,
    expected_live: bool,
) -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='Z',
        task_states=None,
        task_group_exited=task_group_exited,
    )

    assert member.has_live_tasks is expected_live


def test_task_states_distinguishes_unavailable_directory(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def _unavailable(_path: str) -> None:
        raise FileNotFoundError('synthetic unavailable task directory')

    monkeypatch.setattr(supervisor.os, 'scandir', _unavailable)

    assert supervisor._task_states(100) is None


def test_session_members_retains_unavailable_live_task_group(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor.os, 'getsid', lambda _pid: 200)
    monkeypatch.setattr(
        supervisor.os,
        'scandir',
        lambda _path: nullcontext([SimpleNamespace(name='100')]),
    )
    monkeypatch.setattr(
        supervisor.Path,
        'read_bytes',
        lambda _path: b'100 (probe) Z 1 100 200',
    )
    monkeypatch.setattr(supervisor, '_task_states', lambda _pid: None)
    monkeypatch.setattr(supervisor, 'pidfd_process_exited', lambda _pid: False)

    members = supervisor._session_members()

    assert members == [
        supervisor._SessionMember(
            pid=100,
            leader_state='Z',
            task_states=None,
            task_group_exited=False,
        )
    ]


def test_session_members_uses_pidfd_for_empty_zombie_snapshot(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor.os, 'getsid', lambda _pid: 200)
    monkeypatch.setattr(
        supervisor.os,
        'scandir',
        lambda _path: nullcontext([SimpleNamespace(name='100')]),
    )
    monkeypatch.setattr(
        supervisor.Path,
        'read_bytes',
        lambda _path: b'100 (probe) Z 1 100 200',
    )
    monkeypatch.setattr(supervisor, '_task_states', lambda _pid: ())
    pidfd_calls = []
    monkeypatch.setattr(supervisor, 'pidfd_process_exited',
                        lambda pid: pidfd_calls.append(pid) or False)

    members = supervisor._session_members()

    assert pidfd_calls == [100]
    assert members == [
        supervisor._SessionMember(
            pid=100,
            leader_state='Z',
            task_states=(),
            task_group_exited=False,
        )
    ]


def test_signal_members_signals_zombie_leader_with_live_sibling(
        monkeypatch) -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='Z',
        task_states=((100, 'Z'), (101, 'R')),
    )
    signals = []
    closes = []
    monkeypatch.setattr(supervisor, '_session_members', lambda: [member])
    monkeypatch.setattr(supervisor, 'pidfd_open', lambda pid: pid + 1000)
    monkeypatch.setattr(
        supervisor,
        'pidfd_send_signal',
        lambda pidfd, sig: signals.append((pidfd, sig)),
    )
    monkeypatch.setattr(supervisor.os, 'close', closes.append)

    supervisor._signal_members(signal.SIGKILL)

    assert signals == [(1100, signal.SIGKILL)]
    assert closes == [1100]


def test_signal_members_signals_unavailable_live_task_group(
        monkeypatch: pytest.MonkeyPatch) -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='Z',
        task_states=None,
        task_group_exited=False,
    )
    signals = []
    closes = []
    monkeypatch.setattr(supervisor, '_session_members', lambda: [member])
    monkeypatch.setattr(supervisor, 'pidfd_open', lambda pid: pid + 1000)
    monkeypatch.setattr(
        supervisor,
        'pidfd_send_signal',
        lambda pidfd, sig: signals.append((pidfd, sig)),
    )
    monkeypatch.setattr(supervisor.os, 'close', closes.append)

    supervisor._signal_members(signal.SIGTERM)

    assert signals == [(1100, signal.SIGTERM)]
    assert closes == [1100]


def test_signal_members_skips_all_zombie_task_group(monkeypatch) -> None:
    member = supervisor._SessionMember(
        pid=100,
        leader_state='Z',
        task_states=((100, 'Z'), ),
    )
    pidfd_open = []
    monkeypatch.setattr(supervisor, '_session_members', lambda: [member])
    monkeypatch.setattr(supervisor, 'pidfd_open', pidfd_open.append)

    supervisor._signal_members(signal.SIGTERM)

    assert pidfd_open == []
