# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Condition and verify the Linux file pages of one idle loader's index."""

import ctypes
import mmap
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter_ns


def index_files(index):
    """List regular index files once per inode, without following symlinks."""
    files = {}
    for path in sorted(index.rglob("*")):
        info = path.lstat()
        if stat.S_ISREG(info.st_mode):
            files.setdefault((info.st_dev, info.st_ino), (path, info))
    return dict(files.values())


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@contextmanager
def _open_selected(path, expected):
    if not path.exists() or _identity(path.lstat()) != _identity(expected):
        raise RuntimeError(f"Selected index file changed: {path}")
    with path.open("rb", buffering=0) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.getuid() or _identity(info) != _identity(expected):
            raise RuntimeError(
                f"Selected index file changed or is not owned by this worker: {path}"
            )
        yield stream


def _check(result):
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _worker_mappings(libc, rows, cold):
    for line in Path("/proc/self/maps").read_text().splitlines():
        address, permissions, _, device, inode, *_ = line.split()
        major, minor = (int(part, 16) for part in device.split(":"))
        row = rows.get((os.makedev(major, minor), int(inode)))
        if row is None:
            continue
        if "w" in permissions:
            raise RuntimeError(f"Selected index file has a writable worker mapping: {row['path']}")
        if cold:
            start, end = (int(part, 16) for part in address.split("-"))
            _check(
                libc.madvise(
                    ctypes.c_void_p(start), ctypes.c_size_t(end - start), mmap.MADV_DONTNEED
                )
            )
            row["released_readonly_mappings"] += 1


def _residency(libc, fd, size):
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vector = (ctypes.c_ubyte * pages)()
    if pages:
        # Obtain an address without touching the pages we are measuring.
        with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            _check(libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vector))
    return pages, sum(value & 1 for value in vector)


class IndexPageCache:
    def __init__(self, index: Path):
        self.index = index
        self.selected = index_files(index)

    def condition(self, mode, record):
        """Condition the original index selection and retain evidence if verification fails."""
        record.update(mode=mode, start_ns=perf_counter_ns(), status="unavailable", files=[])
        try:
            if sys.platform != "linux" or mode not in ("cold", "warm"):
                raise ValueError("Index cache conditioning requires Linux and a cold or warm mode")
            current = index_files(self.index)
            rows = {}
            for path, info in self.selected.items():
                row = dict(
                    path=str(path.relative_to(self.index)),
                    device=info.st_dev,
                    inode=info.st_ino,
                    size=info.st_size,
                    mtime_ns=info.st_mtime_ns,
                    released_readonly_mappings=0,
                )
                rows[info.st_dev, info.st_ino] = row
                record["files"].append(row)
            record["unconditioned_new_files"] = [
                str(path.relative_to(self.index))
                for path, info in current.items()
                if (info.st_dev, info.st_ino) not in rows
            ]
            libc = ctypes.CDLL(None, use_errno=True)
            if mode == "cold":
                for path, info in self.selected.items():
                    with _open_selected(path, info) as stream:
                        os.fsync(stream.fileno())
            _worker_mappings(libc, rows, mode == "cold")
            for path, info in self.selected.items():
                with _open_selected(path, info) as stream:
                    if mode == "cold":
                        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    else:
                        while stream.read(1024 * 1024):
                            pass

            # Verify the entire selection after all eviction or warming operations have finished.
            record.update(pages=0, resident_pages=0)
            for path, info in self.selected.items():
                with _open_selected(path, info) as stream:
                    pages, resident = _residency(libc, stream.fileno(), info.st_size)
                rows[info.st_dev, info.st_ino].update(pages=pages, resident_pages=resident)
                record["pages"] += pages
                record["resident_pages"] += resident
            expected = 0 if mode == "cold" else record["pages"]
            if not record["pages"] or record["resident_pages"] != expected:
                raise RuntimeError(f"Requested {mode} index file-page residency was not verified")
            record["status"] = "verified"
        except Exception as error:
            record["reason"] = str(error)
            raise
        finally:
            record["end_ns"] = perf_counter_ns()
