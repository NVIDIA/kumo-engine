# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from tmux_tools import config as driver_config
from tmux_tools.tmux import Config, Driver, LaunchConfig, Window


def _driver() -> Driver:
    return Driver(Config(server_id='test', session_id='dev', config_file=''))


def test_wait_for_ports_to_drain_polls_until_listener_releases(
        monkeypatch) -> None:
    driver = _driver()
    listening = Mock(side_effect=[True, True, False])
    sleep = Mock()
    monkeypatch.setattr(driver, '_port_is_listening', listening)
    monkeypatch.setattr('tmux_tools.tmux.time.sleep', sleep)

    assert driver._wait_for_ports_to_drain([10001], timeout_seconds=10,
                                           poll_seconds=0.01)
    assert listening.call_count == 3
    assert sleep.call_count == 2


def test_wait_for_ports_to_drain_leaves_persistent_listener_untouched(
        monkeypatch) -> None:
    driver = _driver()
    monkeypatch.setattr(driver, '_port_is_listening', Mock(return_value=True))
    monotonic = Mock(side_effect=[0.0, 0.2])
    sleep = Mock()
    monkeypatch.setattr('tmux_tools.tmux.time.monotonic', monotonic)
    monkeypatch.setattr('tmux_tools.tmux.time.sleep', sleep)
    run_session = Mock()
    monkeypatch.setattr(driver, 'run_session', run_session)

    assert not driver._wait_for_ports_to_drain([10009], timeout_seconds=0.1,
                                               poll_seconds=0.01)
    sleep.assert_not_called()
    run_session.assert_not_called()


def test_restart_kills_session_before_waiting_for_reserved_ports(
        monkeypatch) -> None:
    driver = _driver()
    run_session = Mock()
    wait_for_ports = Mock(return_value=True)
    monkeypatch.setattr(driver, 'is_running', Mock(return_value=True))
    monkeypatch.setattr('builtins.input', Mock(return_value='n'))
    monkeypatch.setattr(driver, 'run_session', run_session)
    monkeypatch.setattr(driver, '_wait_for_ports_to_drain', wait_for_ports)

    assert not driver.maybe_reattach_if_running([10001, 10002])

    run_session.assert_called_once_with('kill-session')
    wait_for_ports.assert_called_once_with([10001, 10002])


def test_config_driver_waits_for_reserved_and_forwarded_ports(
        monkeypatch) -> None:
    tmux_driver = Mock()
    tmux_driver.check_tmux_is_installed.return_value = True
    tmux_driver.maybe_reattach_if_running.return_value = True
    monkeypatch.setattr(driver_config.tmux, 'Driver',
                        Mock(return_value=tmux_driver))
    driver = driver_config.Driver(
        name='dev',
        panels=[
            driver_config.Panel(title='rest', command='rest',
                                ssh_ports=[10003]),
        ],
        reserved_ports=[10001, 10002],
    )

    driver.run(detached=True)

    tmux_driver.maybe_reattach_if_running.assert_called_once_with(
        [10001, 10002, 10003])


def test_launch_starts_panel_commands_directly_without_send_keys(monkeypatch):
    driver = _driver()
    run = Mock()
    run_session = Mock()
    monkeypatch.setattr(driver, 'run', run)
    monkeypatch.setattr(driver, 'run_session', run_session)
    launch_config = LaunchConfig(
        landing_page=Window(name='landing', command='echo ready'),
        windows=[Window(name='rest', command='env A=1 python rest/main.py')],
        session_closed_callback=None,
    )

    driver.launch(launch_config)

    assert run_session.call_args_list[0].args[0] == 'new-window'
    window_args = run_session.call_args_list[0].args[1]
    assert 'send-keys' not in window_args
    assert 'env A=1 python rest/main.py' in window_args
