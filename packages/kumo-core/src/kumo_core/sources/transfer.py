# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence, TypeVar
from uuid import uuid4

from .cpu import download_workers
from .diagnostics import Rule, transient
from .errors import PermanentSourceError, TransientSourceError
from .parallel import run_in_parallel

PARQUET_MAGIC = b"PAR1"
DOWNLOAD_CHUNK = 1 << 20

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StagedFile:
    """``size`` is unset where the fetch is checksummed end to end."""

    path: str
    size: int | None = None


Fetch = Callable[[StagedFile, str], None]
Unloaded = TypeVar("Unloaded")


def reset_destination(dest: str) -> None:
    """The directory, not the reported paths, is what is ingested, so a
    half-succeeded delete has to fail the read."""
    try:
        shutil.rmtree(dest)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise PermanentSourceError(
            f"could not empty {dest} before reading into it ({error}); what "
            f"a partial delete leaves behind is an earlier attempt's parquet "
            f"that ingest would read as part of this table") from error
    os.makedirs(dest, exist_ok=True)


def discard_destination(dest: str) -> None:
    """Never raises: a read fails for a reason, and cleanup is not it.

    Moved aside before it is deleted. A failure only asks the work beside it
    to stop, so a straggler can still be writing here; renaming the directory
    takes the path it writes to away in one step, where deleting the tree in
    place races the file it is still creating -- and loses, leaving the
    partial read the caller asked to be rid of.
    """
    quarantine = f"{dest}.discarded-{uuid4().hex}"
    try:
        os.rename(dest, quarantine)
    except FileNotFoundError:
        return
    except OSError:
        quarantine = dest
    try:
        shutil.rmtree(quarantine)
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning(
            "could not remove the partial read left at %s; the next attempt "
            "empties it before writing", quarantine, exc_info=True)


def verify_parquet(
    target: str,
    name: str,
    *,
    expected_size: int | None = None,
    diagnosis: str = "",
) -> None:
    size = os.path.getsize(target)
    if expected_size is not None and size != expected_size:
        raise TransientSourceError(
            f"downloaded {size} of {expected_size} bytes for {name}")
    with open(target, "rb") as handle:
        magic = handle.read(len(PARQUET_MAGIC))
    if magic != PARQUET_MAGIC:
        raise PermanentSourceError(f"{name} is not parquet{diagnosis}")


def fetch_all(
    files: Sequence[StagedFile],
    dest: str,
    *,
    fetch: Fetch,
    diagnosis: str = "",
    worker_ceiling: int | None = None,
) -> tuple[str, ...]:
    """Concurrent because serial fetching stretches the window a batch of
    presigned links must stay valid for. ``fetch`` writes the file itself, so
    a vendor client that checksums as it writes is passed straight in.

    Each invocation writes into an attempt-unique sibling of ``dest``. The
    worker pool reports a failure without waiting for callbacks already in
    progress, so an old callback may not open its target until a retry has
    recreated ``dest``. Giving that callback a retired generation path fences
    it from the retry; only a completely successful generation is promoted.
    """
    if not files:
        return ()
    _require_distinct_names(files)
    generation = f"{os.path.normpath(dest)}.downloading-{uuid4().hex}"
    os.makedirs(generation)
    stop = threading.Event()
    succeeded = False
    try:
        downloaded = tuple(
            run_in_parallel(
                [
                    partial(_fetch_one, staged, generation, fetch, diagnosis)
                    for staged in files
                ],
                workers=min(download_workers(worker_ceiling), len(files)),
                stop=stop,
            ))
        result = _promote_generation(generation, dest, downloaded)
        succeeded = True
        return result
    finally:
        if not succeeded:
            # Renaming takes the generation pathname away in one step. A
            # callback with an open descriptor follows it into quarantine; a
            # callback that opens later cannot recreate the missing parent.
            discard_destination(generation)
            # Preserve the fetch contract: a failed read leaves no public
            # destination that could be mistaken for a complete table.
            discard_destination(dest)


def _promote_generation(generation: str, dest: str,
                        downloaded: Sequence[str]) -> tuple[str, ...]:
    """Publish a complete generation and translate its paths to ``dest``."""
    discard_destination(dest)
    try:
        os.rename(generation, dest)
    except OSError as error:
        raise PermanentSourceError(
            f"could not publish the completed read at {dest} ({error})"
        ) from error
    return tuple(
        os.path.join(dest, os.path.basename(path)) for path in downloaded)


def _require_distinct_names(files: Sequence[StagedFile]) -> None:
    """Objects are flattened to their basename because what they were nested
    under is not a relative path this can rebuild: one warehouse lists a path
    relative to its stage, another an absolute volume path, a third a whole
    object name.
    """
    seen: dict[str, str] = {}
    for staged in files:
        name = os.path.basename(staged.path)
        if not name:
            raise PermanentSourceError(
                f"the source staged {staged.path!r}, which names no object")
        if name in seen:
            raise PermanentSourceError(
                f"the source staged {seen[name]!r} and {staged.path!r} under "
                f"the one name {name!r}; downloaded side by side, one would "
                f"overwrite the other")
        seen[name] = staged.path


def staged_fetch(
    *,
    clear: Callable[[], None],
    unload: Callable[[], Unloaded],
    fetch: Callable[[Unloaded], tuple[str, ...]],
    location: str,
) -> tuple[Unloaded, tuple[str, ...]]:
    """The prefix is cleared before the unload as well as after: an
    interrupted attempt's objects are still listed by the next one, whose row
    counts would not show the extra rows.
    """
    try:
        clear()
        unloaded = unload()
        return unloaded, fetch(unloaded)
    finally:
        remove_quietly(clear, location)


def remove_quietly(clear: Callable[[], None], location: str) -> None:
    """Never raises, so a cleanup hiccup cannot fail a read that landed; but
    this is the customer's data in their storage, so a leak is logged."""
    try:
        clear()
    except Exception:
        logger.warning(
            "could not clear the objects staged at %s; they are left in "
            "place", location, exc_info=True)


def remove_all(removals: Sequence[Callable[[], None]], *,
               rules: Sequence[Rule] = (),
               worker_ceiling: int | None = None) -> None:
    if not removals:
        return
    workers = min(download_workers(worker_ceiling), len(removals))
    # Wait for all: a straggler could delete a retry's restaged objects.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(removal) for removal in removals]
    failures = [
        error for future in futures
        if (error := future.exception()) is not None
    ]
    if failures:
        # A transient failure must not hide a permanent one.
        raise min(failures, key=lambda error: transient(error, rules))


def _fetch_one(
    staged: StagedFile,
    dest: str,
    fetch: Fetch,
    diagnosis: str,
) -> str:
    target = os.path.join(dest, os.path.basename(staged.path))
    fetch(staged, target)
    verify_parquet(target, staged.path, expected_size=staged.size,
                   diagnosis=diagnosis)
    return target
