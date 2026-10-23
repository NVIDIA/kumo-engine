# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate declared license metadata in Python manifests and built wheels."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Sequence
from zipfile import BadZipFile, ZipFile

if __package__:
    from .gen_third_party import (
        NoticeError,
        load_json,
        selected_license_expression,
    )
else:
    from gen_third_party import (
        NoticeError,
        load_json,
        selected_license_expression,
    )

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        raise SystemExit(
            "Python 3.10 requires the 'tomli' package to check license metadata."
        ) from None


@dataclass(frozen=True)
class DistributionSpec:
    manifest: str
    name: str
    license_files: tuple[str, ...]
    license_expression: str = "Apache-2.0"
    uses_repository_license: bool = True
    notice_artifacts: tuple[str, ...] = ()
    wheel_source_root: str | None = None


@dataclass(frozen=True)
class DerivedSourceRequirement:
    path: str
    component_id: str
    artifact_ids: frozenset[str]


DISTRIBUTIONS = (
    DistributionSpec("pyproject.toml", "kumo",
                     ("LICENSE", "KUMO-THIRD-PARTY.txt"),
                     notice_artifacts=("kumo-bundled-source", )),
    DistributionSpec("packages/kumo-api/pyproject.toml", "kumo-api",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-bigquery/pyproject.toml", "kumo-bigquery",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-core/pyproject.toml", "kumo-core",
                     ("LICENSE", "THIRD-PARTY.txt"),
                     notice_artifacts=("kumo-core-bundled-source", ),
                     wheel_source_root="packages/kumo-core/src"),
    DistributionSpec("packages/kumo-databricks/pyproject.toml",
                     "kumo-databricks", ("LICENSE", )),
    DistributionSpec("packages/kumo-adls/pyproject.toml", "kumo-adls",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-gcs/pyproject.toml", "kumo-gcs",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-pql/pyproject.toml", "kumo-pql",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-s3/pyproject.toml", "kumo-s3",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-sdk/pyproject.toml", "kumoai-studio",
                     ("LICENSE", "THIRD-PARTY.txt"),
                     notice_artifacts=("kumo-sdk-bundled-source", )),
    DistributionSpec("packages/kumo-snowflake/pyproject.toml", "kumo-snowflake",
                     ("LICENSE", )),
    DistributionSpec("packages/kumo-engine/pyproject.toml", "kumo-engine",
                     ("LICENSE", )),
    DistributionSpec("tools/glove-6b-50d/pyproject.toml", "kumo-glove-6b-50d",
                     ("LICENSE", "THIRD-PARTY.txt"),
                     notice_artifacts=("glove-6b-50d-bundled-data", )),
)

SPECIALIZED_MANIFESTS = frozenset({
    "packages/diskgraph/pyproject.toml",
})


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def tracked_pyprojects(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*pyproject.toml"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return {path.decode("utf-8") for path in result.stdout.split(b"\0") if path}


def setuptools_floor(requirements: Sequence[str]) -> int | None:
    for requirement in requirements:
        match = re.match(r"^setuptools(?:\[[^]]+\])?\s*>=\s*(\d+)", requirement)
        if match is not None:
            return int(match.group(1))
    return None


def resolve_distribution_specs(
    root: Path,
    specs: Sequence[DistributionSpec],
    *,
    inventory: dict | None = None,
    policy: dict | None = None,
) -> tuple[tuple[DistributionSpec, ...], list[str]]:
    if not any(spec.notice_artifacts for spec in specs):
        return tuple(specs), []
    try:
        if inventory is None:
            inventory = load_json(root / "3rdparty/notices/inventory.json")
        if policy is None:
            policy = load_json(root / "3rdparty/notices/policy.json")
        resolved = tuple(
            replace(
                spec,
                license_expression=selected_license_expression(
                    inventory,
                    policy,
                    artifact_ids=spec.notice_artifacts,
                    base_licenses=(spec.license_expression, ),
                ),
            ) if spec.notice_artifacts else spec for spec in specs)
    except NoticeError as error:
        return tuple(specs), [
            f"cannot derive distribution license expressions: {error}"
        ]
    return resolved, []


def validate_repository(
    root: Path,
    specs: Sequence[DistributionSpec] = DISTRIBUTIONS,
    *,
    inventory: dict | None = None,
    policy: dict | None = None,
) -> list[str]:
    specs, errors = resolve_distribution_specs(root, specs, inventory=inventory,
                                               policy=policy)
    expected_manifests = {spec.manifest for spec in specs}
    actual_manifests = tracked_pyprojects(root) - SPECIALIZED_MANIFESTS
    for missing in sorted(expected_manifests - actual_manifests):
        errors.append(
            f"{missing}: expected distribution manifest is not tracked")
    for unexpected in sorted(actual_manifests - expected_manifests):
        errors.append(
            f"{unexpected}: tracked Python distribution is missing from the licensing inventory"
        )
    canonical_path = root / "LICENSE"
    if not canonical_path.is_file():
        errors.append("LICENSE: canonical repository license is missing")
        canonical_license = None
    else:
        canonical_license = canonical_path.read_bytes()
    for spec in specs:
        manifest_path = root / spec.manifest
        if not manifest_path.is_file():
            continue
        try:
            manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            errors.append(f"{spec.manifest}: cannot parse manifest: {error}")
            continue
        project = manifest.get("project", {})
        if project.get("name") != spec.name:
            errors.append(
                f"{spec.manifest}: expected project name {spec.name!r}, found {project.get('name')!r}"
            )
        if project.get("license") != spec.license_expression:
            errors.append(
                f"{spec.manifest}: project.license must be the SPDX expression {spec.license_expression!r}"
            )
        declared_files = tuple(project.get("license-files", ()))
        if declared_files != spec.license_files:
            errors.append(
                f"{spec.manifest}: license-files must be {list(spec.license_files)!r}, found {list(declared_files)!r}"
            )
        build_requirements = manifest.get("build-system",
                                          {}).get("requires", ())
        floor = setuptools_floor(build_requirements)
        if floor is None or floor < 77:
            errors.append(
                f"{spec.manifest}: build-system must declare a setuptools>=77 floor"
            )
        project_root = manifest_path.parent
        for relative_file in spec.license_files:
            legal_path = project_root / relative_file
            if not legal_path.is_file():
                errors.append(
                    f"{spec.manifest}: declared legal file {relative_file!r} is missing"
                )
                continue
            if (relative_file == "LICENSE" and spec.uses_repository_license
                    and canonical_license is not None
                    and legal_path.read_bytes() != canonical_license):
                errors.append(
                    f"{spec.manifest}: LICENSE does not match the canonical repository LICENSE"
                )
    return errors


def metadata_version_at_least_24(value: str | None) -> bool:
    if value is None:
        return False
    try:
        major, minor = (int(part) for part in value.split(".", 1))
    except (TypeError, ValueError):
        return False
    return (major, minor) >= (2, 4)


def derived_source_requirements(
        root: Path) -> tuple[DerivedSourceRequirement, ...]:
    inventory = load_json(root / "3rdparty/notices/inventory.json")
    provenance = load_json(root / "3rdparty/notices/source-provenance.json")

    components_by_license_file: dict[str, list[dict]] = {}
    for component in inventory["components"]:
        for legal_file in component.get("files", ()):
            legal_path = legal_file.get("path")
            if isinstance(legal_path, str):
                components_by_license_file.setdefault(legal_path,
                                                      []).append(component)

    requirements: list[DerivedSourceRequirement] = []
    for source in provenance["sources"]:
        for component in components_by_license_file[source["license_file"]]:
            requirements.append(
                DerivedSourceRequirement(source["path"], component["id"],
                                         frozenset(component["artifact_ids"])))
    return tuple(requirements)


def validate_wheel(
    path: Path,
    root: Path,
    spec: DistributionSpec,
    *,
    derived_requirements: Sequence[DerivedSourceRequirement] = (),
) -> list[str]:
    errors: list[str] = []
    try:
        with ZipFile(path) as wheel:
            names = wheel.namelist()
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                return [
                    f"{path.name}: expected exactly one .dist-info/METADATA, found {len(metadata_names)}"
                ]
            metadata_name = metadata_names[0]
            metadata = BytesParser().parsebytes(wheel.read(metadata_name))
            dist_info = metadata_name.removesuffix("/METADATA")
            if not metadata_version_at_least_24(
                    metadata.get("Metadata-Version")):
                errors.append(
                    f"{path.name}: Metadata-Version must be 2.4 or newer")
            if metadata.get("License-Expression") != spec.license_expression:
                errors.append(
                    f"{path.name}: License-Expression must be {spec.license_expression}"
                )
            if metadata.get_all("License"):
                errors.append(
                    f"{path.name}: deprecated legacy License metadata must be absent"
                )
            declared_files = tuple(metadata.get_all("License-File") or ())
            if declared_files != spec.license_files:
                errors.append(
                    f"{path.name}: License-File fields must be {list(spec.license_files)!r}, found {list(declared_files)!r}"
                )
            source_root = root / spec.manifest
            for relative_file in spec.license_files:
                archive_path = f"{dist_info}/licenses/{relative_file}"
                if archive_path not in names:
                    errors.append(
                        f"{path.name}: missing wheel payload {archive_path}")
                    continue
                expected = (source_root.parent / relative_file).read_bytes()
                if wheel.read(archive_path) != expected:
                    errors.append(
                        f"{path.name}: wheel payload {archive_path} differs from its declared source file"
                    )
            wheel_source_root = PurePosixPath(
                spec.wheel_source_root
                or PurePosixPath(spec.manifest).parent.as_posix())
            wheel_members = set(names)
            notice_artifacts = set(spec.notice_artifacts)
            for requirement in derived_requirements:
                try:
                    member = PurePosixPath(requirement.path).relative_to(
                        wheel_source_root).as_posix()
                except ValueError:
                    continue
                if member not in wheel_members:
                    continue
                if notice_artifacts.isdisjoint(requirement.artifact_ids):
                    errors.append(
                        f"{path.name}: {member} requires {requirement.component_id} in this distribution's notice"
                    )
    except (BadZipFile, KeyError, OSError) as error:
        errors.append(f"{path.name}: cannot inspect wheel: {error}")
    return errors


def wheel_project_name(path: Path) -> str | None:
    try:
        with ZipFile(path) as wheel:
            metadata_names = [
                name for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                return None
            return BytesParser().parsebytes(wheel.read(
                metadata_names[0])).get("Name")
    except (BadZipFile, KeyError, OSError):
        return None


def validate_wheel_directory(
    wheel_dir: Path,
    root: Path,
    require_all: bool,
    specs: Sequence[DistributionSpec] = DISTRIBUTIONS,
) -> tuple[list[str], set[str]]:
    specs, errors = resolve_distribution_specs(root, specs)
    derived_requirements = derived_source_requirements(root)
    seen: set[str] = set()
    expected = {normalize_name(spec.name): spec for spec in specs}
    for wheel_path in sorted(wheel_dir.glob("*.whl")):
        project_name = wheel_project_name(wheel_path)
        if project_name is None:
            errors.append(
                f"{wheel_path.name}: cannot determine wheel project name")
            continue
        normalized = normalize_name(project_name)
        spec = expected.get(normalized)
        if spec is None:
            errors.append(
                f"{wheel_path.name}: unexpected distribution {project_name!r}")
            continue
        if normalized in seen:
            errors.append(
                f"{wheel_path.name}: duplicate wheel for distribution {project_name!r}"
            )
            continue
        seen.add(normalized)
        errors.extend(
            validate_wheel(wheel_path, root, spec,
                           derived_requirements=derived_requirements))
    if require_all:
        for normalized, spec in expected.items():
            if normalized not in seen:
                errors.append(
                    f"{spec.name}: required wheel is missing from {wheel_dir}")
    return errors, seen


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    errors = validate_repository(root)
    wheel_names: set[str] = set()
    if args.require_all and args.wheel_dir is None:
        errors.append("--require-all requires --wheel-dir")
    if args.wheel_dir is not None:
        wheel_errors, wheel_names = validate_wheel_directory(
            args.wheel_dir, root, args.require_all)
        errors.extend(wheel_errors)
    if errors:
        print("Python license metadata check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    message = f"Python license metadata check passed for {len(DISTRIBUTIONS)} manifests"
    if args.wheel_dir is not None:
        message += f" and {len(wheel_names)} wheels"
    print(message + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
