# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import time
from dataclasses import dataclass
from shlex import quote
from typing import List, Optional

from tmux_tools import config, util


@dataclass
class Window:
    name: str
    command: str


@dataclass
class Config:
    server_id: str
    session_id: str
    config_file: str


@dataclass
class LaunchConfig:
    landing_page: Window
    windows: List[Window]
    session_closed_callback: Optional[str]


class Driver:
    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def _port_is_listening(port: int) -> bool:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.05):
                return True
        except OSError:
            return False

    def _wait_for_ports_to_drain(
        self,
        ports: List[int],
        *,
        timeout_seconds: float = 10.0,
        poll_seconds: float = 0.05,
    ) -> bool:
        """Wait for listeners owned by a killed session to release ports.

        No listener is terminated here. A listener that survives the bounded
        wait is left for the normal dependency check to report as a conflict.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            if not any(self._port_is_listening(port) for port in ports):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_seconds)

    def maybe_reattach_if_running(
        self,
        configured_ports: Optional[List[int]] = None,
    ) -> bool:
        if not self.is_running():
            return False
        result = input("""
A devserver is already running. Would you like to re-attach to the existing
devserver? Note, the existing devserver may be running with a different
configuration than requested. (y/N) """)
        if result in ['y', 'Y', 'yes']:
            self.attach()
            return True
        self.run_session('kill-session')
        self._wait_for_ports_to_drain(configured_ports or [])
        return False

    def set_session_closed_callback(self, cmd: str) -> None:
        run_cmd = f'run {quote(cmd)}'
        self.run(f'set-hook -g "session-closed" {quote(run_cmd)}')

    def launch(self, launch_config: LaunchConfig) -> None:
        landing_page_command = f"{launch_config.landing_page.command};bash"
        command = (f"bash -c {quote(landing_page_command)}")
        self.run(
            f'-f {quote(self.config.config_file)} '
            f'new-session -d -s {quote(self.config.session_id)} '
            f'-n {quote(launch_config.landing_page.name)} {quote(command)}')
        for window in launch_config.windows:
            # Start the command as the window's process rather than creating
            # an interactive shell and racing a later ``send-keys`` into it.
            # A simultaneous pair of devservers can otherwise lose the first
            # characters of a long environment-prefixed worker command while
            # shell startup hooks are still reading the terminal.
            panel_command = f'{window.command}; exec bash -i'
            command = f'bash -lc {quote(panel_command)}'
            self.run_session(
                'new-window',
                f'-n {quote(window.name)} {quote(command)}')
        if launch_config.session_closed_callback:
            self.set_session_closed_callback(
                launch_config.session_closed_callback)
        self.run_session('select-window', '-n')

    def attach(self) -> None:
        self.run_session('attach')

    def run_session(self, command: str, args: str = "") -> None:
        self.run(f'{command} -t {quote(self.config.session_id)} {args}')

    def run(self, command: str) -> None:
        util.must_run_command(
            f'tmux -L {quote(self.config.server_id)} {command}')

    def is_running(self) -> bool:
        output = util.run_command(
            f'tmux -L {quote(self.config.server_id)} list-sessions -F "#S"')
        if output.returncode != 0:
            return False
        sessions = output.stdout.split('\n')
        return self.config.session_id in sessions

    def check_tmux_is_installed(self) -> bool:
        dep = config.Dependency(
            name="tmux is installed",
            command="tmux -V",
            stdout_search_pattern="tmux 3.",
            instructions="""
Please install tmux version 3 or above.
""",
        )
        print("Checking Dependency:", dep.name)
        result = dep.check()
        if not result.succeeded:
            print(result.message)
        return result.succeeded
