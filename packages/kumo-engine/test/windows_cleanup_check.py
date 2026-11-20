#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove the reload harness terminates a real native Windows process tree."""

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from importlib import util
from pathlib import Path

CHILD = '''import json, os, sys, time
from pathlib import Path
scratch = Path(sys.argv[1])
with (scratch / 'held.bin').open('wb') as held:
    held.write(b'held until termination')
    held.flush()
    ready = scratch / 'ready.tmp'
    ready.write_text(json.dumps({'parent': os.getppid(), 'child': os.getpid(),
                                 'executable': sys.executable,
                                 'platform': sys.platform, 'os_name': os.name}))
    ready.replace(scratch / 'ready.json')
    time.sleep(120)
'''
PARENT = '''import subprocess, sys
child = subprocess.Popen([sys.executable, '-Iu', '-c', sys.argv[2], sys.argv[1]],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
child.wait(timeout=120)
'''
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
SYNCHRONIZE = 0x00100000


def tree_command(scratch):
    # Windows venv python.exe can redirect to a different interpreter process.
    # This stdlib-only tree uses the base interpreter so Popen owns its real PID.
    executable = Path(sys._base_executable).resolve(strict=True)
    return [str(executable), '-Iu', '-c', PARENT, str(scratch), CHILD]


def kernel_api():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def wait_terminated(kernel, handle):
    result = kernel.WaitForSingleObject(handle, 10000)
    if result == WAIT_TIMEOUT:
        raise TimeoutError('Windows process remained alive after termination')
    if result != WAIT_OBJECT_0:
        raise ctypes.WinError(ctypes.get_last_error())


def cleanup(kernel, parent, pids, handles):
    """Bound fallback cleanup even if the proof fails before tree termination."""
    errors = []
    try:
        for pid in reversed(pids):
            handle = handles.get(pid)
            if (handle is not None
                    and kernel.WaitForSingleObject(handle, 0) == WAIT_OBJECT_0):
                continue
            try:
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                               capture_output=True, timeout=15, check=False)
                if handle is not None:
                    wait_terminated(kernel, handle)
            except (OSError, subprocess.SubprocessError, TimeoutError) as error:
                errors.append(error)
        try:
            parent.wait(timeout=15)
        except subprocess.TimeoutExpired:
            parent.kill()
            parent.wait(timeout=10)
        if errors:
            raise RuntimeError('Windows fallback cleanup failed') from errors[0]
    finally:
        for handle in handles.values():
            kernel.CloseHandle(handle)


def main():
    print(json.dumps(dict(platform=platform.platform(), sys_platform=sys.platform,
                          os_name=os.name, python=sys.version,
                          invoker_executable=sys.executable,
                          base_executable=sys._base_executable)), flush=True)
    if sys.platform != 'win32' or os.name != 'nt':
        raise RuntimeError('This cleanup proof requires native Windows')
    # Isolated Python (-I) omits the script directory from its import path.
    script = Path(__file__).with_name('desktop_reload_qualification.py')
    spec = util.spec_from_file_location('desktop_reload', script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load the reload harness at {script}')
    harness = util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    kernel = kernel_api()
    scratch = Path(tempfile.mkdtemp(prefix='drq-cleanup-'))
    try:
        command = tree_command(scratch)
        parent = subprocess.Popen(command, cwd=scratch,
                                  stdin=subprocess.DEVNULL)
    except OSError:
        shutil.rmtree(scratch)
        raise
    pids, handles = [parent.pid], {}
    try:
        deadline = time.monotonic() + 15
        ready = scratch / 'ready.json'
        while not ready.exists():
            if parent.poll() is not None:
                raise RuntimeError('Parent exited before the child was ready')
            if time.monotonic() >= deadline:
                raise TimeoutError('Child did not report readiness')
            time.sleep(.05)
        record = json.loads(ready.read_text())
        if record['parent'] != parent.pid:
            raise RuntimeError('Readiness did not come from the expected child')
        pids.append(record['child'])
        if (record['platform'] != 'win32' or record['os_name'] != 'nt'
                or not os.path.samefile(record['executable'], command[0])):
            raise RuntimeError('The process tree did not use native base Python')
        for pid in pids:
            handle = kernel.OpenProcess(SYNCHRONIZE, False, pid)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            handles[pid] = handle
        try:
            parent.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise RuntimeError('The parent did not reach its timeout')
        harness.finish_process(parent)
        for handle in handles.values():
            wait_terminated(kernel, handle)
    finally:
        try:
            cleanup(kernel, parent, pids, handles)
        finally:
            shutil.rmtree(scratch)
    if scratch.exists():
        raise RuntimeError('Worker scratch survived process cleanup')
    print(json.dumps(dict(status='PASS', parent_pid=pids[0], child_pid=pids[1],
                          tree_executable=record['executable'],
                          parent_wait_timed_out=True,
                          parent_terminated=True, child_terminated=True,
                          scratch_removed=True)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
