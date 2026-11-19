# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from test.kumo_pytorch.bindings import selections as catalog_selections

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
        "synthetic-medium-customers",
        "synthetic-medium-accounts",
        "synthetic-medium-ltv",
        "mbd-transaction-forecast",
        "mbd-transaction-category",
    }
    aml = m["datasets"]["hi-small"]
    assert aml["source_bytes"] == 99240770
    assert len(aml["files"]) == 11
    assert any(f["key"].endswith("_bindings/v1/entities.parquet") for f in aml["files"])
    assert any(
        f["key"].endswith("account-daily/v1/anchors.parquet") for f in aml["files"]
    )


@pytest.mark.parametrize("free_gib,expected", [(127, "failed"), (128, "passed")])
def test_storage_preflight_checks_the_campaign_filesystem(
    tmp_path, monkeypatch, free_gib, expected
):
    def usage(path):
        assert path == tmp_path
        return 200 * 1024**3, (200 - free_gib) * 1024**3, free_gib * 1024**3

    monkeypatch.setattr(data.shutil, "disk_usage", usage)
    report = data.storage_preflight(tmp_path, 128)
    assert report["status"] == expected
    assert report["free_bytes"] == free_gib * 1024**3
    assert report["required_free_bytes"] == 128 * 1024**3
    assert report["path"] == str(tmp_path.resolve())


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


@pytest.mark.parametrize("fault", [None, "unknown", "duplicate", "conflicting"])
def test_shared_inventory_is_resolved_and_validated(tmp_path, fault):
    manifest, item = fixture_manifest()
    manifest["file_sets"] = {"public": [item]}
    entry = manifest["datasets"]["fixture"]
    entry.update(file_sets=["public"], files=[])
    if fault == "unknown":
        entry["file_sets"] = ["missing"]
    elif fault == "duplicate":
        entry["files"] = [item]
    elif fault == "conflicting":
        manifest["datasets"]["other"] = {
            **entry,
            "file_sets": [],
            "files": [{**item, "sha256": "0" * 64}],
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    if fault:
        with pytest.raises(
            ValueError,
            match={
                "unknown": "Unknown file sets",
                "duplicate": "Duplicate key",
                "conflicting": "Conflicting identity",
            }[fault],
        ):
            data.load_manifest(path)
    else:
        assert data.load_manifest(path)["datasets"]["fixture"]["files"] == [item]


def test_account_binding_reuses_public_sources_with_distinct_roots_and_anchors():
    from test.kumo_pytorch.bindings import dataset_definition, load_bindings

    catalog = load_bindings()
    assert len([ref for ref in catalog if ref.startswith("S01")]) == 1
    binding = catalog["S01"]
    accounts = dataset_definition(binding)
    customers = dataset_definition(catalog["S02"])
    assert accounts["source_options"]["tables"] == customers["source_options"]["tables"]
    assert accounts["source_options"]["root"]["table"] == "accounts"
    assert customers["source_options"]["root"]["table"] == "customers"
    # Returned declarations are independent, including nested shared structures.
    accounts["source_options"]["root"]["table"] = "changed"
    assert dataset_definition(binding)["source_options"]["root"]["table"] == "accounts"
    manifest = data.load_manifest()
    left, right = [
        {item["key"]: item for item in manifest["datasets"][d["inventory"]]["files"]}
        for d in (accounts, customers)
    ]
    assert len(left) == len(right) == 1613
    shared = left.keys() & right.keys()
    assert len(shared) == 1611 and all(left[key] == right[key] for key in shared)
    extra = left.keys() - right.keys()
    assert len(extra) == 2 and all("weekly-account/v1/" in key for key in extra)
    grant = json.loads(
        (SUITE / "datasets/ci-synthetic-medium-read-policy.json").read_text()
    )["Statement"][1]
    assert grant["Action"] == "s3:GetObject" and grant["Effect"] == "Allow"
    assert set(grant["Resource"]) == {
        f"arn:aws:s3:::{manifest['bucket']}/{key}" for key in extra
    }
    assert binding["binding_id"] == "suite:S01:bpe-v1"
    assert binding["base_binding_id"].endswith("abbca33")


def test_combined_binding_reuses_sources_and_only_adds_ltv_anchor_access():
    from test.kumo_pytorch.bindings import (
        dataset_definition,
        load_bindings,
        profile,
    )

    catalog = load_bindings()
    binding = catalog["S04"]
    assert len([ref for ref in catalog if ref.startswith("S04")]) == 1
    definition = dataset_definition(binding)
    assert (
        definition["source_options"]
        == dataset_definition(catalog["S02"])["source_options"]
    )
    manifest = data.load_manifest()
    base, selected = [
        {row["key"]: row for row in manifest["datasets"][name]["files"]}
        for name in ("synthetic-medium-customers", "synthetic-medium-ltv")
    ]
    assert len(base) == len(selected) == 1613
    shared = base.keys() & selected.keys()
    assert len(shared) == 1611 and all(base[key] == selected[key] for key in shared)
    extra = selected.keys() - base.keys()
    assert len(extra) == 2 and all("ltv_positive_6m/v2/" in key for key in extra)
    grant = json.loads(
        (SUITE / "datasets/ci-synthetic-medium-read-policy.json").read_text()
    )["Statement"][2]
    assert grant["Action"] == "s3:GetObject" and grant["Effect"] == "Allow"
    assert set(grant["Resource"]) == {
        f"arn:aws:s3:::{manifest['bucket']}/{key}" for key in extra
    }
    assert binding["binding_id"] == "suite:S04:aggregates-v1"
    assert binding["base_binding_id"].endswith("8925fa")
    assert set(binding["campaigns"]) == {"smoke", "training", "full", "lifecycle"}
    inputs = profile(binding, "input")
    assert len(inputs["aggregates"]) == 35
    assert len(inputs["graph"]["nodes"]) == 7
    assert "F06" in binding["feature_checks"]["features"]


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
    workloads = catalog_selections()
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
        if not entry["dataset_id"].startswith("aml-data/"):
            continue
        assert entry["anchor_versions"] == ["v1", "v2"]
        for version in entry["anchor_versions"]:
            suffix = f"{name}/account-daily/{version}/anchors.parquet"
            assert sum(f["key"].endswith(suffix) for f in entry["files"]) == 1


def test_synthetic_customer_inventory_excludes_truth_and_unused_sources():
    from test.kumo_pytorch.bindings import dataset_definition, load_bindings

    binding = load_bindings()["S02"]
    definition = dataset_definition(binding)
    entry = data.load_manifest()["datasets"][definition["inventory"]]
    assert entry["dataset_id"] == "pragma-synthetic-medium"
    assert entry["population"] == "weekly-customer" and entry["anchor_versions"] == [
        "v1"
    ]
    assert len(entry["files"]) == 1613
    assert entry["source_bytes"] == 17606082087
    public = {
        f["key"].split("/")[4]
        for f in entry["files"]
        if f["key"].startswith("datasets/") and f["key"].endswith(".parquet")
    }
    assert public == {
        "accounts",
        "customers",
        "card_txns",
        "transfers",
        "logins",
        "profile_changes",
        "disputes",
        "lifelong_events",
        "plan_events",
    }
    assert not any("_truth" in f["key"] for f in entry["files"])
    assert (
        binding["binding_id"]
        == "kumo-binding-41605c06ba4a8facd8e18ee7034b4509a5098b3eb42a905069e707f78ca3566e"
    )
    assert set(binding["campaigns"]) == {"smoke", "training", "full", "lifecycle"}


def test_count_forecast_inventory_and_grant_cover_only_selected_modalities():
    import fnmatch
    from test.kumo_pytorch.bindings import (
        dataset_definition,
        load_bindings,
        task_definition,
    )

    catalog = load_bindings()
    binding = catalog["M04"]
    assert len([ref for ref in catalog if ref.startswith("M04")]) == 1
    assert binding["base_binding_id"].endswith("fa6e68e")
    assert set(binding["campaigns"]) == {"smoke", "training", "full", "lifecycle"}
    assert set(task_definition(binding)["generated_checks"]) == {"generated_count"}
    definition = dataset_definition(binding)
    entry = data.load_manifest()["datasets"][definition["inventory"]]
    keys = {item["key"] for item in entry["files"]}
    assert len(keys) == 1008 and entry["source_bytes"] == 6037620860
    assert all(
        "/fold=-1/" not in key and "/geo/" not in key and "/dialog/" not in key
        for key in keys
    )
    assert len(definition["identity_paths"]) == 5
    grant = json.loads((SUITE / "datasets/ci-mbd-read-policy.json").read_text())[
        "Statement"
    ]
    assert len(grant) == 2 and grant[0]["Action"] == "s3:GetObject"
    resources = grant[0]["Resource"]
    assert len(resources) == 13
    assert all(
        any(
            fnmatch.fnmatchcase("arn:aws:s3:::kumo-engine-arena/" + key, pattern)
            for pattern in resources
        )
        for key in keys
    )
    assert all(
        any(
            fnmatch.fnmatchcase("arn:aws:s3:::kumo-engine-arena/" + key, pattern)
            for key in keys
        )
        for pattern in resources
    )


def test_category_forecast_reuses_sources_and_limits_new_read_grant():
    from test.kumo_pytorch.bindings import (
        dataset_definition,
        load_bindings,
        task_definition,
    )

    bindings = load_bindings()
    selected = bindings["M03"]
    assert selected["base_binding_id"].endswith("29575f5")
    assert selected["binding_id"] == "suite:M03:auxiliary-labels-v1"
    assert set(selected["campaigns"]) == {"smoke", "training", "full", "lifecycle"}
    manifest = data.load_manifest()
    original, category = [
        {
            f["key"]: f
            for f in manifest["datasets"][
                dataset_definition(bindings[ref])["inventory"]
            ]["files"]
        }
        for ref in ["M04", "M03"]
    ]
    common = set(original) & set(category)
    assert len(original) == len(category) == 1008 and len(common) == 1006
    assert all(original[k] == category[k] for k in common)
    extra = set(category) - set(original)
    assert extra == {
        "anchors/mbd/v1/transaction-forecast/v1/" + name
        for name in ["anchors.parquet", "manifest.json"]
    }
    policy = json.loads((SUITE / "datasets/ci-mbd-read-policy.json").read_text())[
        "Statement"
    ][1]
    assert policy["Action"] == "s3:GetObject" and policy["Effect"] == "Allow"
    assert set(policy["Resource"]) == {
        "arn:aws:s3:::kumo-engine-arena/" + key for key in extra
    }
    assert task_definition(selected)["auxiliary_checks"] == ["aligned", "matching"]
