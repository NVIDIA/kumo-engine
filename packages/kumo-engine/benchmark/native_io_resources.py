# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linux cache control and cgroup accounting for owned benchmark files."""

import ctypes
import mmap
import os
import resource
from pathlib import Path

from io_utils import fsutil


def parse_counters(text: str) -> dict[str, int]:
    return {key: int(value) for key, value in
            (line.split() for line in text.splitlines() if line.strip())}


def parse_io(text: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for line in text.splitlines():
        for field in line.split()[1:]:
            key, value = field.split('=')
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def cgroup_path() -> str:
    for line in fsutil.read_text('/proc/self/cgroup').splitlines():
        if line.startswith('0::'):
            return '/sys/fs/cgroup' + line.split('::', 1)[1]
    raise RuntimeError('A unified cgroup v2 hierarchy is required')


def process_snapshot() -> dict:
    """Physical bytes belong to this PID; descendant I/O is not implied."""
    current = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(
        pid=os.getpid(), scope='process_only',
        io=parse_counters(fsutil.read_text('/proc/self/io').replace(':', '')),
        major_faults=current.ru_majflt, minor_faults=current.ru_minflt,
        user_seconds=current.ru_utime, system_seconds=current.ru_stime,
        peak_rss_kib=current.ru_maxrss,
        waited_children=dict(major_faults=children.ru_majflt,
                             minor_faults=children.ru_minflt,
                             user_seconds=children.ru_utime,
                             system_seconds=children.ru_stime),
    )


def group_snapshot(group: str) -> dict:
    """Include all descendants; RSS alone omits the mmap file cache."""
    read = lambda name: fsutil.read_text(f'{group}/{name}').strip()
    io_available = fsutil.exists(f'{group}/io.stat')
    return {
        'memory_current': int(read('memory.current')),
        'memory_peak': int(read('memory.peak')),
        'memory_max': read('memory.max'),
        'swap_max': read('memory.swap.max'),
        'memory_stat': parse_counters(read('memory.stat')),
        'memory_events': parse_counters(read('memory.events')),
        'io_available': io_available,
        'io': parse_io(read('io.stat')) if io_available else None,
        'cpu': parse_counters(read('cpu.stat')),
        'observer_process': process_snapshot(),
    }


def residency(path: str) -> tuple[int, int]:
    """Use mincore without touching pages or allocating the file contents."""
    size = fsutil.filesize(path)
    if size is None:
        raise FileNotFoundError(path)
    if not size:
        return 0, 0
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    # Raw file descriptors are necessary for mmap/fadvise; fsutil has no
    # equivalent cache-control or descriptor-mapping operation.
    descriptor = os.open(path, os.O_RDONLY)
    address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_PRIVATE,
                        descriptor, 0)
    try:
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'mmap failed', path)
        vector = (ctypes.c_ubyte * pages)()
        if libc.mincore(address, size, vector):
            raise OSError(ctypes.get_errno(), 'mincore failed', path)
        return sum(value & 1 for value in vector), pages
    finally:
        if address != ctypes.c_void_p(-1).value:
            libc.munmap(address, size)
        os.close(descriptor)


def owned_files(manifest: dict, prepared: str) -> list[str]:
    """Reject an eviction list that could target unrelated host files."""
    root = Path(prepared).resolve()
    result = []
    for value in manifest['owned_files']:
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Benchmark file is outside prepared directory: {path}')
        if fsutil.filesize(str(path)) is None:
            raise FileNotFoundError(str(path))
        result.append(str(path))
    if not result:
        raise ValueError('Manifest contains no owned files')
    return sorted(set(result))


def evict_files(paths: list[str]) -> dict:
    """Evict only finalized owned files before creating a fresh cgroup."""
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fdatasync(descriptor)
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    counts = [residency(path) for path in paths]
    resident = sum(count[0] for count in counts)
    pages = sum(count[1] for count in counts)
    fraction = resident / pages if pages else 0.
    if fraction > .01:
        raise RuntimeError(f'Cache eviction incomplete: {fraction:.2%} resident')
    return dict(resident_pages=resident, total_pages=pages,
                resident_fraction=fraction, method='fdatasync+fadvise+mincore')


def warm_files(paths: list[str]) -> dict:
    """Read inside the measured scope so file-cache charges stay in its budget."""
    for path in paths:
        with fsutil.get_binary_handle(path, 'rb') as stream:
            while stream.read(4 * 1024 * 1024):
                pass
    counts = [residency(path) for path in paths]
    resident = sum(count[0] for count in counts)
    pages = sum(count[1] for count in counts)
    fraction = resident / pages if pages else 0.
    if fraction < .99:
        raise RuntimeError(f'Resident regime cannot retain files: {fraction:.2%}')
    return dict(resident_pages=resident, total_pages=pages,
                resident_fraction=fraction, method='sequential-read+mincore')
