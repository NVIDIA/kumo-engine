"""Explicit Linux file-page-cache experiments, without global cache eviction.

Only caller-selected, owned regular files are affected. mincore snapshots must
confirm residency after advisory eviction or sequential warming. This says
nothing about application, device, filesystem metadata or remote storage caches.
"""

import ctypes
import mmap
import os
import stat
import sys
from pathlib import Path


def residency(fd, size):
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    if not pages:
        return {"pages": 0, "resident_pages": 0}
    libc = ctypes.CDLL(None, use_errno=True)
    mincore = libc.mincore
    mincore.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    )
    mincore.restype = ctypes.c_int
    vector = (ctypes.c_ubyte * pages)()
    # Obtain the mapping address without reading/faulting in its contents.
    with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        if mincore(address, size, vector) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    return {"pages": pages, "resident_pages": sum(value & 1 for value in vector)}


def release_readonly_mappings(device, inode):
    """Drop this idle worker's file PTEs before requesting page-cache eviction.

    Caller must have stopped all readers. Writable mappings are rejected; no
    anonymous pages or files outside the explicit owned-file selection qualify.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.madvise.restype = ctypes.c_int
    released = 0
    for line in Path("/proc/self/maps").read_text().splitlines():
        address, permissions, _, encoded_device, encoded_inode, *_ = line.split()
        major, minor = (int(part, 16) for part in encoded_device.split(":"))
        if (os.makedev(major, minor), int(encoded_inode)) != (device, inode):
            continue
        if "w" in permissions:
            raise RuntimeError("Cannot condition a file with writable worker mappings")
        start, end = (int(part, 16) for part in address.split("-"))
        if libc.madvise(start, end - start, mmap.MADV_DONTNEED) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        released += 1
    return released


def condition_files(paths, mode, *, release_mappings=False):
    if sys.platform != "linux":
        raise RuntimeError("File-page-cache experiments require Linux")
    if mode not in {"cold", "warm"}:
        raise ValueError("Expected cold or warm cache condition")
    observations, seen = [], set()
    for path in sorted(map(Path, paths)):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Cache experiments do not follow symlinks")
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Cache experiments require regular files")
        with path.open("rb", buffering=0) as stream:
            fd = stream.fileno()
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
                raise ValueError("Cache experiments require owned regular files")
            key = (before.st_dev, before.st_ino)
            if key in seen:
                continue
            seen.add(key)
            released = 0
            if mode == "cold":
                os.fsync(fd)  # This file only; never a machine-wide sync/drop.
                if release_mappings:
                    released = release_readonly_mappings(*key)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            else:
                while stream.read(1024 * 1024):
                    pass
            after = os.fstat(fd)
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError("Cache input changed during conditioning")
            observations.append(
                {
                    "path": str(path.resolve()),
                    "device": before.st_dev,
                    "inode": before.st_ino,
                    "size": before.st_size,
                    "mtime_ns": before.st_mtime_ns,
                    "released_readonly_mappings": released,
                }
            )
    if not observations:
        raise ValueError("Cache experiment selected no files")
    # Check all files after conditioning the entire selection: warming a later
    # file may have evicted an earlier one. These remain instantaneous snapshots.
    for row in observations:
        with open(row["path"], "rb") as stream:
            current = os.fstat(stream.fileno())
            if (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            ) != (row["device"], row["inode"], row["size"], row["mtime_ns"]):
                raise RuntimeError("Cache input changed before residency verification")
            row.update(residency(stream.fileno(), row["size"]))
    pages = sum(row["pages"] for row in observations)
    resident = sum(row["resident_pages"] for row in observations)
    verified = pages > 0 and (resident == 0 if mode == "cold" else resident == pages)
    return {
        "mode": mode,
        "status": "verified" if verified else "unavailable",
        "cache_scope": "selected Linux file pages only; mincore snapshots, not locked residency",
        "pages": pages,
        "resident_pages": resident,
        "files": observations,
    }


class FilePageCache:
    """before_epoch callback; alternate cold/warm on explicitly selected files."""

    def __init__(self, files, *, release_mappings=False):
        self.files = files
        self.release_mappings = release_mappings

    def __call__(self, epoch, recorder):
        mode = "cold" if epoch % 2 == 0 else "warm"
        start = recorder.clock()
        try:
            result = condition_files(
                self.files(), mode, release_mappings=self.release_mappings
            )
        except Exception as error:
            recorder.write(
                "cache_condition",
                mode=mode,
                status="unavailable",
                reason=str(error),
                start_ns=start,
                end_ns=recorder.clock(),
            )
            raise
        recorder.write(
            "cache_condition", **result, start_ns=start, end_ns=recorder.clock()
        )
        if result["status"] != "verified":
            raise RuntimeError("Requested file-page-cache condition was not verified")
