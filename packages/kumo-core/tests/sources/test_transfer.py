# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bringing staged objects down must lose nothing and keep nothing partial."""

import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from kumo_core.sources.errors import PermanentSourceError, TransientSourceError
from kumo_core.sources.transfer import (
    StagedFile,
    discard_destination,
    fetch_all,
    remove_all,
    remove_quietly,
    reset_destination,
    verify_parquet,
)

PROMPTLY = 5

PARQUET = b"PAR1" + b"body"
SLOWLY = 0.02


class Warehouse:
    """Writes canned objects and records how many were fetched at once."""
    def __init__(self, bodies, raises=None, takes=0.0):
        self.bodies = dict(bodies)
        self.raises = raises
        self.takes = takes
        self.lock = threading.Lock()
        self.open = 0
        self.most_open = 0
        self.fetched = []

    def fetch(self, staged, target):
        with self.lock:
            self.open += 1
            self.most_open = max(self.most_open, self.open)
            self.fetched.append(staged.path)
        try:
            time.sleep(self.takes)
            if self.raises is not None:
                raise self.raises
            with open(target, "wb") as handle:
                handle.write(self.bodies[staged.path])
        finally:
            with self.lock:
                self.open -= 1


def files(count=2):
    return tuple(
        StagedFile(path=f"stage/part-{index}.parquet", size=len(PARQUET))
        for index in range(count))


def warehouse(count=2, **kwargs):
    return Warehouse({f.path: PARQUET for f in files(count)}, **kwargs)


def test_a_destination_is_left_empty_whatever_was_there_before(tmp_path):
    dest = tmp_path / "orders"
    dest.mkdir()
    (dest / "stale.parquet").write_bytes(b"PAR1stale")
    reset_destination(str(dest))
    assert list(dest.iterdir()) == []

    reset_destination(str(tmp_path / "absent" / "orders"))
    assert (tmp_path / "absent" / "orders").is_dir()


def test_a_destination_is_discarded_out_from_under_work_still_writing(tmp_path):
    """A failure only asks the reads beside it to stop, so one of them can
    still be writing when the partial read is thrown away. Deleting the tree
    in place walks it while that straggler adds to it, and gives up with the
    directory still there."""
    dest = tmp_path / "orders"
    dest.mkdir()
    for index in range(200):
        (dest / f"part-{index}.parquet").write_bytes(PARQUET)

    writing = threading.Event()
    stop = threading.Event()

    def straggler():
        for index in range(200, 20_000):
            if stop.is_set():
                return
            try:
                (dest / f"part-{index}.parquet").write_bytes(PARQUET)
            except OSError:
                return  # The path went away underneath it, which is the point.
            writing.set()

    thread = threading.Thread(target=straggler)
    thread.start()
    try:
        writing.wait(timeout=PROMPTLY)
        discard_destination(str(dest))
        assert not dest.exists()
    finally:
        stop.set()
        thread.join(timeout=PROMPTLY)

    assert list(tmp_path.iterdir()) == []


def test_a_destination_that_cannot_be_emptied_fails_the_read(tmp_path):
    dest = tmp_path / "orders"
    attempt = dest / "attempt"
    attempt.mkdir(parents=True)
    (attempt / "part-0.parquet").write_bytes(PARQUET)
    attempt.chmod(0o500)
    try:
        with pytest.raises(PermanentSourceError, match="earlier attempt"):
            reset_destination(str(dest))
    finally:
        attempt.chmod(0o700)


class TestVerification:
    def test_a_short_download_is_reported_as_worth_retrying(self, tmp_path):
        target = tmp_path / "part.parquet"
        target.write_bytes(PARQUET)
        with pytest.raises(TransientSourceError):
            verify_parquet(str(target), "part", expected_size=len(PARQUET) + 9)

    def test_bytes_that_are_not_parquet_are_permanent_and_diagnosed(
            self, tmp_path):
        """Snowflake alone can say why bytes arrived that are not parquet."""
        target = tmp_path / "part.parquet"
        target.write_bytes(b"\x93\x00\x11\x22")
        with pytest.raises(PermanentSourceError) as failure:
            verify_parquet(str(target), "part", diagnosis="; not encrypted")
        assert "not encrypted" in str(failure.value)


class TestFetchAll:
    def test_every_object_lands_in_the_order_it_was_listed(self, tmp_path):
        source = warehouse()
        paths = fetch_all(files(), str(tmp_path), fetch=source.fetch)
        assert [os.path.basename(path)
                for path in paths] == ["part-0.parquet", "part-1.parquet"]
        assert all(Path(path).read_bytes() == PARQUET for path in paths)

    def test_objects_are_fetched_several_at_a_time_but_never_past_the_bound(
            self, tmp_path):
        """One at a time leaves the warehouse's own bandwidth idle, and a
        bound that stopped being applied would let every object run at once.

        A barrier one party wider than the bound can only ever time out, so
        the assertion is on what could not happen rather than on timing.
        """
        together = threading.Barrier(2, timeout=10)
        too_wide = threading.Barrier(3, timeout=0.5)
        overfilled = []

        def fetch(staged, target):
            together.wait()
            try:
                too_wide.wait()
                overfilled.append(staged.path)
            except threading.BrokenBarrierError:
                pass
            with open(target, "wb") as handle:
                handle.write(PARQUET)

        fetch_all(files(4), str(tmp_path), fetch=fetch, worker_ceiling=2)
        assert not overfilled

    def test_a_failure_is_raised_as_it_came_and_cancels_what_is_left(
            self, tmp_path):
        """A pool that only stopped submitting would still drain its queue, so
        every remaining object would be paid for to reach a failure already
        known, and the caller decides what to retry from the error itself."""
        failure = OSError("connection reset")
        source = warehouse(16, raises=failure, takes=SLOWLY)
        with pytest.raises(OSError) as raised:
            fetch_all(files(16), str(tmp_path), fetch=source.fetch,
                      worker_ceiling=2)
        assert raised.value is failure
        assert len(source.fetched) < 16

    def test_a_corrupted_object_is_refused_not_kept(self, tmp_path):
        source = warehouse()
        source.bodies["stage/part-1.parquet"] = b"\x00\x01\x02\x03cipher"[:8]
        with pytest.raises(PermanentSourceError):
            fetch_all(files(), str(tmp_path), fetch=source.fetch)

    def test_nothing_to_fetch_is_not_a_failure(self, tmp_path):
        assert fetch_all((), str(tmp_path), fetch=warehouse().fetch) == ()

    def test_objects_sharing_a_name_are_refused_before_one_hides_the_other(
            self, tmp_path):
        source = warehouse()
        collide = (StagedFile("stage/day=1/part-0.parquet"),
                   StagedFile("stage/day=2/part-0.parquet"))
        with pytest.raises(PermanentSourceError, match="overwrite"):
            fetch_all(collide, str(tmp_path), fetch=source.fetch)
        assert source.fetched == []

    def test_a_failure_leaves_none_of_what_already_landed(self, tmp_path):
        dest = tmp_path / "orders"
        dest.mkdir()

        def fetch(staged, target):
            if staged.path.endswith("part-1.parquet"):
                raise OSError("connection reset")
            with open(target, "wb") as handle:
                handle.write(PARQUET)

        with pytest.raises(OSError):
            fetch_all(files(2), str(dest), fetch=fetch, worker_ceiling=1)
        assert not dest.exists()

    def test_a_cleanup_that_fails_does_not_stand_in_for_the_failure(
            self, tmp_path, monkeypatch):
        failure = OSError("connection reset")

        def refuse(*args, **kwargs):
            raise PermissionError("no permission to delete")

        monkeypatch.setattr(shutil, "rmtree", refuse)
        source = warehouse(2, raises=failure)
        with pytest.raises(OSError) as raised:
            fetch_all(files(2), str(tmp_path), fetch=source.fetch)
        assert raised.value is failure

    def test_a_late_failure_is_not_held_behind_a_fetch_still_running(
            self, tmp_path):
        """Collected in submission order, this failure is not seen until the
        blocked fetch ahead of it returns -- and everything queued behind it
        is fetched in the meantime, to reach an answer already known."""
        released = threading.Event()
        failure = OSError("connection reset")
        fetched = []

        def fetch(staged, target):
            if staged.path.endswith("part-0.parquet"):
                released.wait(timeout=10)
                return
            fetched.append(staged.path)
            raise failure

        started = time.monotonic()
        try:
            with pytest.raises(OSError) as raised:
                fetch_all(files(16), str(tmp_path), fetch=fetch,
                          worker_ceiling=2)
            assert raised.value is failure
            assert time.monotonic() - started < PROMPTLY
            assert len(fetched) < 15
        finally:
            released.set()

    def test_a_late_callback_cannot_write_into_a_retried_destination(
            self, tmp_path):
        """A failed generation returns while one callback is still running.

        The retry completes and recreates ``dest`` before that old callback
        next opens its target. The old target must name its abandoned
        generation, not the newly published retry.
        """
        dest = tmp_path / "orders"
        reset_destination(str(dest))
        old_ready = threading.Event()
        release_old = threading.Event()
        old_finished = threading.Event()
        old_targets = []
        old_errors = []
        failure = OSError("connection reset")
        old_body = b"PAR1OLD!"
        new_body = b"PAR1NEW!"

        def first_fetch(staged, target):
            if staged.path.endswith("part-1.parquet"):
                assert old_ready.wait(timeout=PROMPTLY)
                raise failure
            old_targets.append(target)
            old_ready.set()
            try:
                assert release_old.wait(timeout=10)
                with open(target, "wb") as handle:
                    handle.write(old_body)
            except OSError as error:
                old_errors.append(error)
            finally:
                old_finished.set()

        try:
            started = time.monotonic()
            with pytest.raises(OSError) as raised:
                fetch_all(files(), str(dest), fetch=first_fetch,
                          worker_ceiling=2)
            assert raised.value is failure
            assert time.monotonic() - started < PROMPTLY

            # This is the next full read attempt: it recreates the public
            # destination and publishes a new generation before the old
            # callback is allowed to open its target.
            reset_destination(str(dest))

            def retry_fetch(staged, target):
                del staged
                with open(target, "wb") as handle:
                    handle.write(new_body)

            retried = fetch_all(files(), str(dest), fetch=retry_fetch,
                                worker_ceiling=2)
            assert all(Path(path).read_bytes() == new_body for path in retried)

            release_old.set()
            assert old_finished.wait(timeout=PROMPTLY)
            assert old_errors
            assert old_targets[0] != str(dest / "part-0.parquet")
            assert all(Path(path).read_bytes() == new_body for path in retried)
            assert list(tmp_path.iterdir()) == [dest]
        finally:
            release_old.set()


class TestRemoveQuietly:
    def test_a_failed_deletion_is_recorded_without_masking_what_led_here(
            self, caplog):
        def refuse():
            raise RuntimeError("no permission to delete")

        remove_quietly(refuse, "gs://bucket/prefix")
        assert "gs://bucket/prefix" in caplog.text

    def test_a_deletion_that_works_says_nothing(self, caplog):
        remove_quietly(lambda: None, "gs://bucket/prefix")
        assert caplog.text == ""


class TestRemoveAll:
    def test_raises_after_all_removals_finish(self):
        started = threading.Event()
        finished = threading.Event()

        def slow():
            started.set()
            time.sleep(SLOWLY)
            finished.set()

        def refuse():
            assert started.wait(PROMPTLY)
            raise RuntimeError("no permission to delete")

        with pytest.raises(RuntimeError, match="no permission"):
            remove_all([slow, refuse], worker_ceiling=2)
        assert finished.is_set()

    def test_raises_permanent_over_earlier_transient(self):
        def throttled():
            raise ConnectionError("slow down")

        def refused():
            raise PermissionError("no grant")

        with pytest.raises(PermissionError):
            remove_all([throttled, refused],
                       rules=((ConnectionError, TransientSourceError), ),
                       worker_ceiling=2)
