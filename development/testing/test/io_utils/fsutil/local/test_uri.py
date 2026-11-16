# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from io_utils.fsutil.uri import ADLSURI, S3URI, UCURI, LocalURI, ParseURI


def test_local_uri():
    p1 = "/local/file/system/dir"
    assert ParseURI(p1) == LocalURI(path=p1)
    assert ParseURI(p1).path == p1
    assert LocalURI(path=p1).path == p1

    p2 = "/local/file/system/dir/"
    assert ParseURI(p2) == LocalURI(path=p2)

    p3 = "/local/file/system/dir/file.ext"
    assert ParseURI(p3) == LocalURI(path=p3)

    p4 = "dir"
    assert ParseURI(p4) == LocalURI(path=p4)

    p5 = "dir/"
    assert ParseURI(p5) == LocalURI(path=p5)

    p6 = "dir/file.ext"
    assert ParseURI(p6) == LocalURI(path=p6)


def test_s3_uri():
    def _make_path(scheme: str, bucket: str, path: str) -> tuple[str, S3URI]:
        return (f"{scheme}://{bucket}/{path}", S3URI(bucket=bucket, path=path))

    p1, x1 = _make_path("s3", "remote", "file/system/dir")
    assert x1.as_s3() == p1
    assert ParseURI(p1) == x1

    p2, x2 = _make_path("s3", "remote", "file/system/dir/")
    assert x2.as_s3() == p2
    assert ParseURI(p2) == x2

    p3, x3 = _make_path("s3", "remote", "file/system/dir/file.ext")
    assert x3.as_s3() == p3
    assert ParseURI(p3) == x3

    p4, x4 = _make_path("s3a", "remote", "file/system/dir")
    assert x4.as_s3a() == p4
    assert ParseURI(p4) == x4

    p5, x5 = _make_path("s3a", "remote", "file/system/dir/")
    assert x5.as_s3a() == p5
    assert ParseURI(p5) == x5

    p6, x6 = _make_path("s3a", "remote", "file/system/dir/file.ext")
    assert x6.as_s3a() == p6
    assert ParseURI(p6) == x6


def test_uc_uri():
    def _make_path(catalog: str, schema: str, volume: str,
                   path: str) -> tuple[str, str, UCURI]:
        as_path = f"/Volumes/{catalog}/{schema}/{volume}/{path}"
        return (as_path, f"dbfs:{as_path}",
                UCURI(catalog=catalog, uc_schema=schema, volume=volume,
                      path=path))

    p1, u1, x1 = _make_path("cat", "sch", "vol", "file/system/dir")
    assert x1.as_path() == p1
    assert x1.as_url() == u1
    assert ParseURI(u1) == x1

    p2, u2, x2 = _make_path("cat", "sch", "vol", "file/system/dir/")
    assert x2.as_path() == p2
    assert x2.as_url() == u2
    assert ParseURI(u2) == x2

    p3, u3, x3 = _make_path("cat", "sch", "vol", "file/system/dir/file.ext")
    assert x3.as_path() == p3
    assert x3.as_url() == u3
    assert ParseURI(u3) == x3

    p4, u4, x4 = _make_path("cat", "sch", "vol", "file/system/dir")
    assert x4.as_path() == p4
    assert x4.as_url() == u4
    assert ParseURI(u4) == x4

    p5, u5, x5 = _make_path("cat", "sch", "vol", "file/system/dir/")
    assert x5.as_path() == p5
    assert x5.as_url() == u5
    assert ParseURI(u5) == x5

    p6, u6, x6 = _make_path("cat", "sch", "vol", "file/system/dir/file.ext")
    assert x6.as_path() == p6
    assert x6.as_url() == u6
    assert ParseURI(u6) == x6


def test_adls_uri():
    base = "abfss://mycontainer@myaccount.dfs.core.windows.net"

    # Round-trip: path with file
    u1 = f"{base}/dir/file.txt"
    x1 = ADLSURI(storage_account="myaccount", container="mycontainer",
                 path="dir/file.txt")
    assert x1.as_url() == u1
    assert str(x1) == u1
    assert ParseURI(u1) == x1

    # Round-trip: directory path with trailing slash
    u2 = f"{base}/dir/"
    x2 = ADLSURI(storage_account="myaccount", container="mycontainer",
                 path="dir/")
    assert x2.as_url() == u2
    assert ParseURI(u2) == x2

    # Round-trip: container root — no trailing slash in either direction
    u3 = base
    x3 = ADLSURI(storage_account="myaccount", container="mycontainer", path="")
    assert x3.as_url() == u3
    assert ParseURI(u3) == x3
    assert ParseURI(f"{base}/") == x3  # trailing slash normalises to empty path

    # Validator: leading slash is stripped on construction
    x4 = ADLSURI(storage_account="myaccount", container="mycontainer",
                 path="/dir/file.txt")
    assert x4.path == "dir/file.txt"
    assert x4.as_url() == f"{base}/dir/file.txt"

    # as_adls() is an alias for as_url()
    assert x1.as_adls() == x1.as_url()

    # Validator: slash in storage_account or container is rejected
    with pytest.raises(ValueError):
        ADLSURI(storage_account="my/account", container="mycontainer", path="")
    with pytest.raises(ValueError):
        ADLSURI(storage_account="myaccount", container="my/container", path="")


def test_wrong_uri():
    # raise error on unknown schema
    p1 = "unk://remote/file/system/dir"
    with pytest.raises(ValueError):
        ParseURI(p1)

    # raise error on bad passed type
    with pytest.raises(ValueError):
        ParseURI(8)  # type: ignore

    url_path = "http://this.is.a.url"
    with pytest.raises(ValueError):
        ParseURI(url_path)

    url_path = "https://this.is.a.url"
    with pytest.raises(ValueError):
        ParseURI(url_path)

    url_path = "ftp://this.is.a.ftp"
    with pytest.raises(ValueError):
        ParseURI(url_path)

    url_path = "mailto:this.is.a.mail@domain.com"
    with pytest.raises(ValueError):
        ParseURI(url_path)

    url_path = "file://this.is.a.url"
    with pytest.raises(ValueError):
        ParseURI(url_path)
