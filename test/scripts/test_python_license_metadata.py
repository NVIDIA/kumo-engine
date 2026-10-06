# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from zipfile import ZipFile

from scripts import check_python_license_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def write_wheel(
    path: Path,
    *,
    license_expression: str = "Apache-2.0",
    legacy_license: str | None = None,
    include_payload: bool = True,
) -> None:
    headers = [
        "Metadata-Version: 2.4",
        "Name: demo",
        "Version: 1.0.0",
        f"License-Expression: {license_expression}",
        "License-File: LICENSE",
    ]
    if legacy_license is not None:
        headers.append(f"License: {legacy_license}")
    with ZipFile(path, "w") as wheel:
        wheel.writestr("demo-1.0.0.dist-info/METADATA",
                       "\n".join(headers) + "\n\n")
        if include_payload:
            wheel.writestr("demo-1.0.0.dist-info/licenses/LICENSE",
                           b"canonical\n")


def synthetic_spec() -> check_python_license_metadata.DistributionSpec:
    return check_python_license_metadata.DistributionSpec(
        "pyproject.toml",
        "demo",
        ("LICENSE", ),
    )


def test_repository_manifest_inventory_passes() -> None:
    assert len(check_python_license_metadata.DISTRIBUTIONS) == 11
    assert check_python_license_metadata.validate_repository(
        REPOSITORY_ROOT) == []


def test_repository_expression_tracks_notice_inventory() -> None:
    inventory = check_python_license_metadata.load_json(
        REPOSITORY_ROOT / "3rdparty/notices/inventory.json")
    policy = check_python_license_metadata.load_json(
        REPOSITORY_ROOT / "3rdparty/notices/policy.json")
    inventory = copy.deepcopy(inventory)
    component = next(component for component in inventory["components"]
                     if component["id"] == "rust:foldhash@0.1.5")
    component["artifact_ids"].append("kumo-bundled-source")
    artifact = next(artifact for artifact in inventory["artifacts"]
                    if artifact["id"] == "kumo-bundled-source")
    artifact["components"].append(component["id"])
    errors = check_python_license_metadata.validate_repository(
        REPOSITORY_ROOT, inventory=inventory, policy=policy)
    assert any("Apache-2.0 AND BSD-3-Clause AND MIT AND Zlib" in error
               for error in errors)


def test_valid_wheel_metadata_and_payload(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_bytes(b"canonical\n")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path)
    assert check_python_license_metadata.validate_wheel(wheel_path, tmp_path,
                                                        synthetic_spec()) == []


def test_distribution_specific_license_expression(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_bytes(b"canonical\n")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path, license_expression="MIT")
    spec = check_python_license_metadata.DistributionSpec(
        "pyproject.toml", "demo", ("LICENSE", ), "MIT")
    assert check_python_license_metadata.validate_wheel(wheel_path, tmp_path,
                                                        spec) == []


def test_legacy_and_wrong_license_metadata_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_bytes(b"canonical\n")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path, license_expression="MIT", legacy_license="MIT")
    errors = check_python_license_metadata.validate_wheel(
        wheel_path, tmp_path, synthetic_spec())
    assert any("License-Expression" in error for error in errors)
    assert any("legacy License" in error for error in errors)


def test_missing_or_changed_license_payload_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_bytes(b"canonical\n")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path, include_payload=False)
    errors = check_python_license_metadata.validate_wheel(
        wheel_path, tmp_path, synthetic_spec())
    assert any("missing wheel payload" in error for error in errors)


def test_derived_source_requires_component_in_wheel_notice(
        tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_bytes(b"canonical\n")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path)
    with ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("demo/derived.py", "")
    spec = check_python_license_metadata.DistributionSpec(
        "pyproject.toml",
        "demo",
        ("LICENSE", ),
        notice_artifacts=("demo-bundled-source", ),
    )
    requirement = check_python_license_metadata.DerivedSourceRequirement(
        "demo/derived.py",
        "source:upstream@example",
        frozenset({"repository-source"}),
    )

    errors = check_python_license_metadata.validate_wheel(
        wheel_path,
        tmp_path,
        spec,
        derived_requirements=(requirement, ),
    )

    assert any("source:upstream@example" in error for error in errors)


def test_setuptools_floor_requires_pep639_capable_backend() -> None:
    assert check_python_license_metadata.setuptools_floor(["setuptools>=77"
                                                           ]) == 77
    assert check_python_license_metadata.setuptools_floor(
        ["setuptools>=80.10.2,<82"]) == 80
    assert check_python_license_metadata.setuptools_floor(
        ["setuptools==68.2.2"]) is None
    assert check_python_license_metadata.setuptools_floor(["setuptools"
                                                           ]) is None
