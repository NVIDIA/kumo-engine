# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nightly_data", SUITE / "data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def fixture_manifest(payload=b"golden input"):
    item = dict(
        key="dataset/test.parquet",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    entry = dict(
        dataset_id="fixture",
        dataset_version="v1",
        population="tiny",
        anchor_versions=["v1"],
        source_bytes=len(payload),
        files=[item],
    )
    return dict(
        schema_version=1,
        bucket="fixture-bucket",
        region="us-west-2",
        expected_bucket_owner="123456789012",
        datasets={"fixture": entry},
    ), item


def test_manifest_has_exact_starter_inputs():
    m = data.load_manifest()
    assert set(m["datasets"]) == {
        "hi-small",
        "hi-medium",
        "hi-large",
        "li-small",
        "li-medium",
        "li-large",
    }
    aml = m["datasets"]["hi-small"]
    assert aml["source_bytes"] == 99240770
    assert len(aml["files"]) == 11
    assert any(f["key"].endswith("_bindings/v1/entities.parquet") for f in aml["files"])
    assert any(
        f["key"].endswith("account-daily/v1/anchors.parquet") for f in aml["files"]
    )


def test_local_copy_cache_and_corruption(tmp_path):
    m, item = fixture_manifest()
    source = tmp_path / "mirror"
    (source / "dataset").mkdir(parents=True)
    (source / item["key"]).write_bytes(b"golden input")
    root = tmp_path / "data"
    first = data.execute("fetch", m, "fixture", root, source_dir=source)
    assert first["status"] == "passed"
    assert first["downloaded_files"] == 1
    assert first["transferred_bytes"] == 12
    assert first["elapsed_seconds"] > 0
    (source / item["key"]).unlink()
    cached = data.execute("fetch", m, "fixture", root, source_dir=source)
    assert cached["cached_files"] == 1
    assert cached["transferred_bytes"] == 0
    assert cached["transfer_seconds_sum"] == 0
    (root / item["key"]).write_bytes(b"broken input")
    result = data.execute("fetch", m, "fixture", root, source_dir=source)
    assert result["status"] == "failed"
    assert "SHA-256 differs" in result["results"][0]["error"]
    assert (root / item["key"]).read_bytes() == b"broken input"


def test_wrong_download_never_becomes_cached_file(tmp_path):
    m, item = fixture_manifest()
    source = tmp_path / "mirror"
    (source / "dataset").mkdir(parents=True)
    (source / item["key"]).write_bytes(b"wrong")
    root = tmp_path / "data"
    result = data.execute("fetch", m, "fixture", root, source_dir=source)
    assert result["status"] == "failed"
    assert result["transferred_bytes"] == 5
    assert result["results"][0]["phase"] == "verify"
    assert not (root / item["key"]).exists()
    assert not list(root.rglob(".download-*"))


def test_s3_pins_bucket_owner_and_records_time(tmp_path):
    m, item = fixture_manifest()

    class Client:
        def get_object(self, **kwargs):
            assert kwargs == dict(
                Bucket="fixture-bucket",
                Key=item["key"],
                ExpectedBucketOwner="123456789012",
            )
            return {"Body": io.BytesIO(b"golden input")}

    result = data.execute("fetch", m, "fixture", tmp_path, client=Client())
    assert result["status"] == "passed"
    assert result["transfer_seconds_sum"] > 0
    assert result["verification_seconds_sum"] > 0
    assert result["transferred_bytes"] == 12


def test_interrupted_stream_is_removed_and_closed(tmp_path):
    m, item = fixture_manifest()

    class Body(io.BytesIO):
        def read(self, size=-1):
            raise OSError("interrupted transfer")

    body = Body(b"golden input")

    class Client:
        def get_object(self, **kwargs):
            return {"Body": body}

    result = data.execute("fetch", m, "fixture", tmp_path, client=Client())
    assert result["status"] == "failed"
    assert result["results"][0]["phase"] == "transfer"
    assert body.closed
    assert not (tmp_path / item["key"]).exists()
    assert not list(tmp_path.rglob(".download-*"))


def test_missing_file_is_failure_not_skip(tmp_path):
    m, _ = fixture_manifest()
    result = data.execute("verify", m, "fixture", tmp_path)
    assert result["status"] == "failed"
    assert result["failed_files"] == 1
    assert result["cached_files"] == 0


@pytest.mark.parametrize(
    "path", ["../escape", "/tmp/escape", "a/../b", "a//b", "a\\b", "./a", ""]
)
def test_reject_unsafe_manifest_paths(path):
    with pytest.raises(ValueError):
        data.relative_path(path)


def test_symlink_destination_and_source_are_refused(tmp_path):
    m, item = fixture_manifest()
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "data"
    root.mkdir()
    (root / "dataset").symlink_to(outside, target_is_directory=True)
    result = data.execute("fetch", m, "fixture", root, source_dir=outside)
    assert result["status"] == "failed"
    assert "Symlink" in result["results"][0]["error"]
    assert not list(outside.iterdir())


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "bad"),
        ("bytes", -1),
        ("bytes", True),
        ("bytes", None),
        ("key", "../escape"),
    ],
)
def test_reject_invalid_manifest(tmp_path, field, value):
    m, item = fixture_manifest()
    item[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        data.load_manifest(path)


def test_conflicting_shared_file_identity_is_rejected(tmp_path):
    m, _ = fixture_manifest()
    other = json.loads(json.dumps(m["datasets"]["fixture"]))
    other["files"][0]["sha256"] = "0" * 64
    m["datasets"]["other"] = other
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Conflicting"):
        data.load_manifest(path)


def test_required_files_are_not_silently_omitted_on_partial_failure(tmp_path):
    m, item = fixture_manifest()
    m["datasets"]["fixture"]["files"].append({**item, "key": "dataset/missing.parquet"})
    path = tmp_path / item["key"]
    path.parent.mkdir()
    path.write_bytes(b"golden input")
    result = data.execute("verify", m, "fixture", tmp_path, jobs=2)
    assert result["status"] == "failed"
    assert len(result["results"]) == 2
    assert result["cached_files"] == 1
    assert result["failed_files"] == 1


def test_list_has_no_data_or_aws_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert data.main(["list", "--data-dir", str(tmp_path / "absent")]) == 0
    assert not list(tmp_path.iterdir())


def test_client_setup_failure_is_saved(tmp_path, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("profile is unavailable")

    monkeypatch.setattr(data, "s3_client", unavailable)
    report = tmp_path / "result.json"
    assert (
        data.main(
            ["fetch", "--data-dir", str(tmp_path / "data"), "--report", str(report)]
        )
        == 1
    )
    saved = json.loads(report.read_text())
    assert saved["status"] == "failed"
    assert saved["phase"] == "setup"
    assert saved["results"] == []
    assert not (tmp_path / "data").exists()


def test_report_cannot_overwrite_local_source(tmp_path):
    manifest = data.load_manifest()
    source = tmp_path / "source"
    path = source / manifest["datasets"]["hi-small"]["files"][0]["key"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve")
    with pytest.raises(SystemExit):
        data.main(["fetch", "--source-dir", str(source), "--report", str(path)])
    assert path.read_bytes() == b"preserve"


def test_graph_workloads_select_v2_and_sequence_workloads_keep_v1():
    workloads = json.loads((SUITE / "workloads/aml.json").read_text())
    selections = {}
    for workload in workloads:
        for model in workload["models"]:
            key = (workload["selector"]["workload_id"], model)
            assert key not in selections
            selections[key] = workload["selector"]["anchor_version"]
            if model in {"graphsage", "combined"}:
                assert (
                    workload["execution_cutoff"]["execution_time_column"]
                    == "graph_as_of_time"
                )
                assert (
                    workload["execution_cutoff"]["prediction_time_column"]
                    == "as_of_time"
                )
    assert selections == {
        ("account-involvement-next24h", "graphsage"): "v2",
        ("account-involvement-next24h", "combined"): "v2",
        ("account-involvement-next24h", "nuformer"): "v1",
        ("masked-value-pretrain", "pragma"): "v1",
        ("next-token-pretrain", "nuformer"): "v1",
    }
    manifest = data.load_manifest()
    for name, entry in manifest["datasets"].items():
        assert entry["anchor_versions"] == ["v1", "v2"]
        for version in entry["anchor_versions"]:
            suffix = f"{name}/account-daily/{version}/anchors.parquet"
            assert sum(f["key"].endswith(suffix) for f in entry["files"]) == 1
