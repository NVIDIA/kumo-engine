# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fakes import FakeSession
from kumo_s3 import session as s3

BUCKET = "kumo-test-bucket"


@pytest.fixture(autouse=True)
def clear_region_cache():
    s3.bucket_region.cache_clear()
    yield
    s3.bucket_region.cache_clear()


def _built(monkeypatch) -> dict:
    """Captures the keywords a filesystem would have been built with."""
    captured: dict = {}

    import pyarrow.fs

    def record(**kwargs):
        captured.update(kwargs)
        return "filesystem"

    monkeypatch.setattr(pyarrow.fs, "S3FileSystem", record)
    return captured


def test_the_buckets_own_region_wins_over_the_sessions(monkeypatch):
    """A cross-region bucket signed for the session's region answers with a
    bare 301 that names neither region."""
    built = _built(monkeypatch)
    monkeypatch.setattr(s3, "bucket_region", lambda bucket: "eu-west-3")

    assert s3.filesystem(BUCKET, session=FakeSession(region_name="us-west-2"),
                         default_region="us-east-1") == "filesystem"
    assert built["region"] == "eu-west-3"


def test_a_custom_endpoint_is_never_sent_to_the_aws_region_resolver(
        monkeypatch):
    """MinIO's bucket names are its own: resolving one against AWS answers for
    a different bucket, or rejects the connector outright."""
    built = _built(monkeypatch)
    asked = []
    monkeypatch.setattr(s3, "bucket_region",
                        lambda bucket: asked.append(bucket) or "us-east-1")

    s3.filesystem(BUCKET, session=FakeSession(region_name="us-west-2"),
                  endpoint_url="https://minio.internal:9000")

    assert asked == []
    assert built["region"] == "us-west-2"
    assert built["endpoint_override"] == "https://minio.internal:9000"
    assert built["force_virtual_addressing"] is True


def test_supplied_options_replace_the_derived_endpoint_handling(monkeypatch):
    built = _built(monkeypatch)
    s3.filesystem(
        BUCKET, session=FakeSession(region_name="us-west-2"),
        endpoint_url="https://minio.internal:9000", options={
            "endpoint_override": "https://other:9000",
            "scheme": "https"
        })

    assert built["endpoint_override"] == "https://other:9000"
    assert "force_virtual_addressing" not in built


def test_an_unresolvable_region_falls_back_rather_than_failing(monkeypatch):
    built = _built(monkeypatch)

    def unreachable(bucket):
        raise OSError("transient")

    monkeypatch.setattr(s3, "bucket_region", unreachable)
    s3.filesystem(BUCKET, session=FakeSession(), default_region="us-east-1")
    assert built["region"] == "us-east-1"


def test_a_buckets_region_is_resolved_once_and_then_served(monkeypatch):
    import pyarrow.fs

    asked = []
    monkeypatch.setattr(pyarrow.fs, "resolve_s3_region",
                        lambda bucket: asked.append(bucket) or "us-east-2")

    assert s3.bucket_region("my-bucket") == "us-east-2"
    assert s3.bucket_region("my-bucket") == "us-east-2"
    assert asked == ["my-bucket"]


def test_a_blip_is_not_memoized_as_the_buckets_region(monkeypatch):
    """One unreachable moment must not pin a bucket to "no region" for the
    life of the process, which is the redirect the lookup exists to avoid.
    """
    import pyarrow.fs

    attempts = []

    def flaky(bucket):
        attempts.append(bucket)
        if len(attempts) == 1:
            raise OSError("transient")
        return "eu-west-3"

    monkeypatch.setattr(pyarrow.fs, "resolve_s3_region", flaky)
    with pytest.raises(OSError):
        s3.bucket_region(BUCKET)
    assert s3.bucket_region(BUCKET) == "eu-west-3"


def test_no_resolvable_credential_leaves_the_caller_its_own_fallback():
    assert s3.filesystem(BUCKET, session=FakeSession(credentials=None)) is None


def test_a_half_resolved_credential_is_not_signed_with():
    partial = SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(
        access_key="key", secret_key="", token=None))
    assert s3.filesystem(BUCKET,
                         session=FakeSession(credentials=partial)) is None


def test_the_aws_namespace_is_a_frozen_literal():
    """It is part of every S3 graph ID ever created; re-spelling it
    re-identifies all of them."""
    assert s3.AWS_NAMESPACE == "aws-s3"


class _Unavailable(BaseHTTPRequestHandler):
    failures = 0
    seen: dict = {}
    lock = threading.Lock()

    def do_HEAD(self):
        with self.lock:
            self.seen[self.path] = self.seen.get(self.path, 0) + 1
            failing = self.seen[self.path] <= self.failures
        self.send_response(503 if failing else 200)
        self.send_header("Content-Length", "0" if failing else "8")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def unavailable():
    handler = type("Handler", (_Unavailable, ), {"seen": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield handler, {
        "endpoint_override": f"127.0.0.1:{server.server_address[1]}",
        "scheme": "http"
    }
    server.shutdown()
    thread.join()


def _filesystems(options):
    yield s3.filesystem(BUCKET, session=FakeSession(region_name="us-east-1"),
                        endpoint_url="http://unused", options=options)
    yield s3.anonymous_filesystem(BUCKET, region="us-east-1", options=options)


# These tests use a loopback server. The public container has no external network.
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_503_retried_past_three_attempts(unavailable):
    handler, options = unavailable
    handler.failures = 3
    for built in _filesystems(options):
        handler.seen.clear()
        assert built.get_file_info(f"{BUCKET}/part-0.parquet").size == 8
        assert handler.seen == {f"/{BUCKET}/part-0.parquet": 4}


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_no_lifetime_retry_cap(unavailable):
    handler, options = unavailable
    handler.failures = 1
    built = next(_filesystems(options))
    keys = [f"{BUCKET}/part-{index}.parquet" for index in range(150)]
    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(lambda key: built.get_file_info(key).size, keys))
    assert sizes == [8] * len(keys)
