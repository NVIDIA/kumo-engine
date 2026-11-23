# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal
import subprocess
import sys
import time
from pathlib import Path

from tests.e2e.ci_command import run


def test_quiet_command_retains_output_and_status(tmp_path, capfd):
    log = tmp_path / "job" / "console.log"
    child = """
import os, sys, time
os.write(1, b'out\\n')
os.write(2, b'err\\n')
time.sleep(.2)
sys.exit(23)
"""
    assert run(log, [sys.executable, "-c", child], heartbeat=0.05) == 23
    assert log.read_bytes() == b"out\nerr\n"
    output = capfd.readouterr().out
    assert "out\nerr" in output and "Still running" in output and "exited 23 after" in output


def test_cancellation_reaches_grandchild(tmp_path):
    log = tmp_path / "console.log"
    grandchild = """
import signal, sys, time
def terminate(*_):
    time.sleep(0.3)
    print('grandchild terminated', flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, terminate)
print('ready', flush=True)
time.sleep(30)
"""
    child = f"""
import subprocess, sys
subprocess.run([sys.executable, '-c', {grandchild!r}])
"""
    wrapper = Path(__file__).with_name("ci_command.py")
    with subprocess.Popen([sys.executable, wrapper, log, sys.executable, "-c", child]) as process:
        try:
            deadline = time.monotonic() + 10
            while not log.exists() or "ready" not in log.read_text():
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.02)
        finally:
            process.terminate()
        assert process.wait(timeout=6) == 128 + signal.SIGTERM
    assert "grandchild terminated" in log.read_text()
