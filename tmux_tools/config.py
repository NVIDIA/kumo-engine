# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from shlex import quote
from subprocess import CompletedProcess
from typing import List, Optional, Tuple

from tmux_tools import aws, tmux, util

DEVSERVER_CONDA_ENV = os.getenv('CONDA_DEFAULT_ENV')
CONDA_COMMAND = (f"conda activate {DEVSERVER_CONDA_ENV};"
                 if DEVSERVER_CONDA_ENV else '')


@dataclass
class DependencyResult:
    succeeded: bool
    message: str


@dataclass
class Dependency:
    name: str
    command: str
    instructions: str
    stdout_search_pattern: Optional[str] = None

    def _check_cmd_output(self, cmd_output: CompletedProcess) -> bool:
        return (cmd_output.returncode == 0
                and (not self.stdout_search_pattern or re.search(
                    self.stdout_search_pattern, cmd_output.stdout)) is not None)

    def check(self) -> DependencyResult:
        message: List[str] = []
        try:
            bash_cmd = f"bash -c {quote(self.command)}"
            cmd_output = util.run_command(bash_cmd)
            if self._check_cmd_output(cmd_output):
                return DependencyResult(True, "")
            message.append(f"Failed to check self: {self.name}")
            message.append(util.format_cmd_error(self.command, cmd_output))
        except FileNotFoundError:
            message.append(f"Failed to check self: {self.name}")
            name = util.get_cmd_name(self.command)
            message.append(
                f"Could not find program `{name}` on the system path "
                f"when running command: {self.command}")

        message.append(self.instructions)

        return DependencyResult(False, '\n'.join(message))


@dataclass
class LandingPage:
    ssh_command: str
    web_urls: List[Tuple[str, str]]
    additional_instructions: str

    def render_text(self) -> str:
        parts: List[str] = []
        kumo = r"""
             ___           ___           ___           ___
            /__/|         /__/\         /__/\         /  /\
           |  |:|         \  \:\       |  |::\       /  /::\
           |  |:|          \  \:\      |  |:|:\     /  /:/\:\
         __|  |:|      ___  \  \:\   __|__|:|\:\   /  /:/  \:\
        /__/\_|:|____ /__/\  \__\:\ /__/::::| \:\ /__/:/ \__\:\
        \  \:\/:::::/ \  \:\ /  /:/ \  \:\~~\__\/ \  \:\ /  /:/
         \  \::/~~~~   \  \:\  /:/   \  \:\        \  \:\  /:/
          \  \:\        \  \:\/:/     \  \:\        \  \:\/:/
           \  \:\        \  \::/       \  \:\        \  \::/
            \__\/         \__\/         \__\/         \__\/
    """
        parts.append(f"\033[1;31m{kumo}\033[0m")

        parts.append("""

Welcome to the Kumo devserver. Here, you can use the following commands.

    * To quit, press Ctrl-Q
    * To switch windows, press `Shift <-` or `Shift ->`
    * To scroll up and down, press Ctrl-b then press `[`. Scroll using
      or Ctrl-U and Ctrl-D. Exit scroll mode with `Q`.
    * If your terminal supports mouse events, you can scroll with your
      trackpad and switch windows by clicking on the window name at the bottom
      of the screen. On the default mac terminal, you may need to press Cmd-R
      in order to enable/disable mouse events. On iTerm2, mouse reporting
      is configured in Preference -> Profiles -> Terminal, and can temporarily
      be disabled by holding ⌥ option.

If you're using a conda environment and it's not being activated in the panels
when you run the devserver, set the environment variable
`KUMO_DEVSERVER_CONDA_ENV` to the name of your environment.
    """)

        if self.ssh_command:
            parts.append(f"""
Please run the following command from your laptop to set up an SSH tunnel to
your devserver:

{util.indent(self.ssh_command)}
    """)

        if self.web_urls:
            url_parts: List[str] = [
                """
To open the service in your web browser, hold Cmd and double click one
of the following URLs:
"""
            ]
            max_len = max(len(name) for name, _ in self.web_urls)
            for name, url in self.web_urls:
                space = " " * (max(max_len - len(name), 0))
                url_parts.append(f"    {name}:{space} \033[4m{url}\033[0m")
            parts.append('\n'.join(url_parts) + '\n')

        parts.append(self.additional_instructions)

        return ''.join(parts)


@dataclass
class Panel:
    title: str
    command: str
    ssh_ports: List[int] = field(default_factory=list)
    web_urls: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Driver:
    name: str
    panels: List[Panel]
    tmux_server_id: str = 'devserver'
    additional_instructions: str = ""
    dependencies: List[Dependency] = field(default_factory=list)
    reserved_ports: List[int] = field(default_factory=list)
    session_closed_callback: Optional[str] = None

    def run(self, detached: bool = False) -> None:
        tmux_config = self.make_tmux_config()
        t = tmux.Driver(tmux_config)
        if not t.check_tmux_is_installed():
            exit(1)
        configured_ports = sorted(
            set(self.reserved_ports).union(port for panel in self.panels
                                           for port in panel.ssh_ports))
        if t.maybe_reattach_if_running(configured_ports):
            return
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [executor.submit(dep.check) for dep in self.dependencies]
            for dep, task in zip(self.dependencies, tasks):
                print("Checking Dependency:", dep.name)
                result = task.result()
                if not result.succeeded:
                    print(result.message)
                    exit(1)
        launch_config = self.make_launch_config()
        t.launch(launch_config)
        if not detached:
            t.attach()

    def make_tmux_config(self) -> tmux.Config:
        config_file = os.path.join(os.path.dirname(__file__), 'tmux.conf')

        return tmux.Config(session_id=self.name, server_id=self.tmux_server_id,
                           config_file=config_file)

    def make_launch_config(self) -> tmux.LaunchConfig:
        # Main Window
        landing_page_text = self.make_landing_page().render_text()
        landing_page = tmux.Window(
            name="Instructions",
            command=f'echo {quote(landing_page_text)}',
        )

        # Sub-windows
        windows: List[tmux.Window] = []
        for panel in self.panels:
            windows.append(
                tmux.Window(name=panel.title,
                            command=f"{CONDA_COMMAND} {panel.command}"))

        return tmux.LaunchConfig(
            landing_page=landing_page, windows=windows,
            session_closed_callback=self.session_closed_callback)

    def make_ssh_command(self, aws_info: aws.Info) -> str:
        ports: List[int] = []
        for panel in self.panels:
            ports.extend(panel.ssh_ports)
        parts: List[str] = []
        parts.append("ssh \\")
        parts.append(
            f"""    -o ProxyCommand="aws ssm start-session --target %h """
            f"""--region {aws_info.az} --document-name AWS-StartSSHSession """
            f"""--parameters 'portNumber=%p'" \\""")
        for port in ports:
            parts.append(f"    -L {port}:localhost:{port} \\")
        parts.append(f"    {aws_info.instance_id}")
        return '\n'.join(parts)

    def make_landing_page(self) -> LandingPage:
        web_urls: List[Tuple[str, str]] = []
        for panel in self.panels:
            web_urls.extend(panel.web_urls)

        ssh_command = ""
        if aws.Info.is_on_aws():
            aws_info = aws.Info.get()
            ssh_command = self.make_ssh_command(aws_info)

        return LandingPage(
            ssh_command=ssh_command,
            web_urls=web_urls,
            additional_instructions=self.additional_instructions,
        )
