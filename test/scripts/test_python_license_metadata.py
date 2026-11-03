# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import check_python_license_metadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def write_wheel(
    path: Path,
    *,
    license_expression: str = "Apache-2.0",
    legacy_license: str | None = None,
    license_files: tuple[str, ...] = ("LICENSE", ),
    payloads: dict[str, bytes] | None = None,
) -> None:
    headers = [
        "Metadata-Version: 2.4",
        "Name: demo",
        "Version: 1.0.0",
        f"License-Expression: {license_expression}",
    ]
    headers.extend(f"License-File: {name}" for name in license_files)
    if legacy_license is not None:
        headers.append(f"License: {legacy_license}")
    with ZipFile(path, "w") as wheel:
        wheel.writestr("demo-1.0.0.dist-info/METADATA",
                       "\n".join(headers) + "\n\n")
        if payloads is None:
            payloads = {name: b"canonical\n" for name in license_files}
        for name, content in payloads.items():
            wheel.writestr(f"demo-1.0.0.dist-info/licenses/{name}", content)


def synthetic_spec(
    license_expression: str = "Apache-2.0",
    license_files: tuple[str, ...] = ("LICENSE", ),
) -> check_python_license_metadata.DistributionSpec:
    return check_python_license_metadata.DistributionSpec(
        "pyproject.toml",
        "demo",
        license_files,
        license_expression,
    )


def test_repository_manifest_inventory_passes() -> None:
    assert len(check_python_license_metadata.DISTRIBUTIONS) == 13
    assert check_python_license_metadata.validate_repository(
        REPOSITORY_ROOT) == []


def test_repository_expression_tracks_notice_inventory() -> None:
    inventory = check_python_license_metadata.load_json(
        REPOSITORY_ROOT / "3rdparty/notices/inventory.json")
    policy = check_python_license_metadata.load_json(
        REPOSITORY_ROOT / "3rdparty/notices/policy.json")
    inventory = copy.deepcopy(inventory)
    spec = next(spec for spec in check_python_license_metadata.DISTRIBUTIONS
                if "kumo-bundled-source" in spec.notice_artifacts)
    obligations = set(
        check_python_license_metadata.selected_license_expression(
            inventory,
            policy,
            artifact_ids=spec.notice_artifacts,
            base_licenses=(spec.license_expression, ),
        ).split(" AND "))
    # Any single-license Rust record that adds an obligation; DiskGraph
    # dependency changes replace the records, so none is named here.
    component = min((component for component in inventory["components"]
                     if component["ecosystem"] == "rust"
                     and " " not in component["license_expression"]
                     and component["license_expression"] not in obligations),
                    key=lambda component: component["id"])
    component["artifact_ids"].append("kumo-bundled-source")
    artifact = next(artifact for artifact in inventory["artifacts"]
                    if artifact["id"] == "kumo-bundled-source")
    artifact["components"].append(component["id"])
    errors = check_python_license_metadata.validate_repository(
        REPOSITORY_ROOT, inventory=inventory, policy=policy)
    expected = " AND ".join(
        sorted(obligations | {component["license_expression"]}))
    assert any(expected in error for error in errors)


@pytest.mark.parametrize(("license_expression", "license_files"), [
    ("Apache-2.0", ("LICENSE", )),
    ("MIT", ("LICENSE", )),
    ("Apache-2.0 AND PDDL-1.0", ("LICENSE", "THIRD-PARTY.txt")),
])
def test_valid_wheel_metadata_and_payload(
    tmp_path: Path,
    license_expression: str,
    license_files: tuple[str, ...],
) -> None:
    payloads = {name: f"canonical {name}\n".encode() for name in license_files}
    for name, content in payloads.items():
        (tmp_path / name).write_bytes(content)
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path, license_expression=license_expression,
                license_files=license_files, payloads=payloads)
    assert check_python_license_metadata.validate_wheel(
        wheel_path, tmp_path, synthetic_spec(license_expression,
                                             license_files)) == []


@pytest.mark.parametrize("expected_expression",
                         ["Apache-2.0", "Apache-2.0 AND PDDL-1.0"])
def test_legacy_and_wrong_license_metadata_are_rejected(
        tmp_path: Path, expected_expression: str) -> None:
    (tmp_path / "LICENSE").write_bytes(b"canonical\n")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path, license_expression="MIT", legacy_license="MIT")
    errors = check_python_license_metadata.validate_wheel(
        wheel_path, tmp_path, synthetic_spec(expected_expression))
    assert any("License-Expression" in error for error in errors)
    assert any("legacy License" in error for error in errors)


@pytest.mark.parametrize(("license_files", "filename"), [
    (("LICENSE", ), "LICENSE"),
    (("LICENSE", "THIRD-PARTY.txt"), "LICENSE"),
    (("LICENSE", "THIRD-PARTY.txt"), "THIRD-PARTY.txt"),
])
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_missing_or_changed_license_payload_is_rejected(
    tmp_path: Path,
    license_files: tuple[str, ...],
    filename: str,
    mutation: str,
) -> None:
    payloads = {name: f"canonical {name}\n".encode() for name in license_files}
    for name, content in payloads.items():
        (tmp_path / name).write_bytes(content)
    if mutation == "missing":
        del payloads[filename]
    else:
        payloads[filename] += b"changed\n"
    expression = ("Apache-2.0 AND PDDL-1.0"
                  if len(license_files) > 1 else "Apache-2.0")
    wheel_path = tmp_path / "demo-1.0.0-py3-none-any.whl"
    write_wheel(wheel_path, license_expression=expression,
                license_files=license_files, payloads=payloads)
    errors = check_python_license_metadata.validate_wheel(
        wheel_path, tmp_path, synthetic_spec(expression, license_files))
    expected = ("missing wheel payload" if mutation == "missing" else
                "differs from its declared source file")
    assert any(filename in error and expected in error for error in errors)


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
