# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex
import subprocess
from typing import List


def format_cmd_error(cmd: str, cmd_output: subprocess.CompletedProcess) -> str:
    result: List[str] = []
    result.append(f"Executed command `{cmd}` and got error "
                  f"code {cmd_output.returncode}.")
    if cmd_output.stdout:
        result.append("Standard Out:")
        result.append(indent(cmd_output.stdout))
    if cmd_output.stderr:
        result.append("Standard Error:")
        result.append(indent(cmd_output.stderr))
    return '\n'.join(result)


def run_command(cmd: str) -> subprocess.CompletedProcess:
    parts = shlex.split(cmd)
    return subprocess.run(
        parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def get_cmd_name(cmd: str) -> str:
    parts = shlex.split(cmd)
    return parts[0]


def must_run_command(cmd: str):
    output = run_command(cmd)
    if output.returncode != 0:
        print(format_cmd_error(cmd, output))
        exit(1)


def indent(text: str) -> str:
    parts = []
    for line in text.split('\n'):
        if line:
            parts.append(f'    {line}')
    return '\n'.join(parts)
