#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render and verify artifact-scoped third-party notices."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "3rdparty" / "notices" / "inventory.json"
DEFAULT_POLICY = REPO_ROOT / "3rdparty" / "notices" / "policy.json"
DEFAULT_SOURCE_PROVENANCE = (REPO_ROOT / "3rdparty" / "notices" /
                             "source-provenance.json")
DISKGRAPH_ROOT = REPO_ROOT / "packages" / "diskgraph"
DEFAULT_DISKGRAPH_MANIFEST = (DISKGRAPH_ROOT / "third_party" /
                              "artifact-manifest.json")
DEFAULT_DISKGRAPH_PROVENANCE = (DISKGRAPH_ROOT / "third_party" /
                                "source-provenance.json")
DEFAULT_NOTICE = REPO_ROOT / "THIRD-PARTY.txt"
# Refresh the inventory and notice after DiskGraph dependency changes.
DISKGRAPH_REFRESH_COMMANDS = (
    "cargo fetch --locked --manifest-path packages/diskgraph/Cargo.toml && "
    "python scripts/refresh_diskgraph_notice_inventory.py && "
    "python scripts/gen_third_party.py --output THIRD-PARTY.txt")
SCHEMA_VERSION = 1
BAR = "=" * 78
RULE = "-" * 78
PERSONAL_PATH_RE = re.compile(r"(?:/home/|/Users/|file://[^\s]*/home/)")
BUNDLE_CLASSIFICATIONS = {
    "kumo-first-party",
    "diskgraph-native",
    "third-party",
}
FIRST_PARTY_DISTRIBUTIONS = {
    "kumo-api",
    "kumo-bigquery",
    "kumo-core",
    "kumo-databricks",
    "kumo-gcs",
    "kumo-adls",
    "kumo-pql",
    "kumo-s3",
    "kumo-snowflake",
    "kumo-engine",
    "kumo-glove-6b-50d",
}
NATIVE_DISTRIBUTIONS = {"diskgraph"}


class NoticeError(ValueError):
    """An invalid or stale notice input."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


LEGAL_BASENAME_RE = re.compile(
    r"^(?:license|licence|copying|notice|authors?|copyright|exception)(?:$|[-_.].*)",
    re.IGNORECASE,
)
CODE_SUFFIXES = {
    ".bat",
    ".c",
    ".cc",
    ".class",
    ".cpp",
    ".cs",
    ".dll",
    ".dylib",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".ps1",
    ".py",
    ".pyc",
    ".pyd",
    ".pyi",
    ".pyo",
    ".rb",
    ".rs",
    ".sh",
    ".so",
    ".ts",
    ".tsx",
}
NATIVE_SUFFIXES = {".a", ".dll", ".dylib", ".lib", ".pyd", ".so"}


def _native_member(name: str) -> bool:
    lower = name.lower()
    return (Path(lower).suffix in NATIVE_SUFFIXES
            or re.search(r"\.so(?:\.\d+)+$", lower) is not None)


def _license_member(name: str, dist_info: str) -> bool:
    """Select packaged license, notice, attribution, and exception files."""
    if name.endswith("/"):
        return False
    lower = name.lower()
    basename = lower.rsplit("/", 1)[-1]
    prefixes = ("license", "licence", "copying", "notice", "authors",
                "copyright", "exception")
    if lower.startswith(f"{dist_info.lower()}/licenses/"):
        return True
    if lower.startswith(f"{dist_info.lower()}/") and basename.startswith(
            prefixes):
        return True
    return (LEGAL_BASENAME_RE.fullmatch(basename) is not None
            and Path(basename).suffix not in CODE_SUFFIXES)


def _metadata_license_evidence(metadata: Any) -> str:
    evidence = {
        "license_expression":
        metadata.get("License-Expression"),
        "license":
        metadata.get("License"),
        "license_classifiers":
        sorted(value for value in (metadata.get_all("Classifier") or [])
               if value.startswith("License ::")),
        "license_files":
        sorted(metadata.get_all("License-File") or []),
    }
    return sha256_bytes(
        (json.dumps(evidence, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False) + "\n").encode("utf-8"))


def wheel_record(path: Path) -> tuple[str, str, dict[str, str], str, set[str]]:
    with zipfile.ZipFile(path) as archive:
        metadata_members = sorted(name for name in archive.namelist()
                                  if name.endswith(".dist-info/METADATA"))
        if len(metadata_members) != 1:
            raise NoticeError(
                f"{path}: expected one .dist-info/METADATA, found {len(metadata_members)}"
            )
        metadata_member = metadata_members[0]
        dist_info = metadata_member.rsplit("/", 1)[0]
        metadata = Parser().parsestr(
            archive.read(metadata_member).decode("utf-8", "replace"))
        name = metadata.get("Name", "")
        version = metadata.get("Version", "")
        if not name or not version:
            raise NoticeError(f"{path}: METADATA lacks Name or Version")
        files = {
            member: sha256_bytes(archive.read(member))
            for member in sorted(archive.namelist())
            if (_license_member(member, dist_info)
                and archive.getinfo(member).file_size > 0)
        }
        native_members = {
            member
            for member in archive.namelist() if _native_member(member)
        }
        return (
            name,
            version,
            files,
            _metadata_license_evidence(metadata),
            native_members,
        )


def normalized_legal_files(files: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for path, digest in files.items():
        first, separator, remainder = path.partition("/")
        normalized_path = remainder if separator and first.endswith(
            ".dist-info") else path
        if normalized_path in normalized:
            raise NoticeError(
                f"multiple legal files normalize to {normalized_path!r}")
        normalized[normalized_path] = digest
    return normalized


def _component_classifications(path: Path) -> dict[str, str]:
    """Read the fat-wheel component manifest."""
    classifications: dict[str, str] = {}
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise NoticeError(f"cannot read {path}: {error}") from error
    for line_number, row in enumerate(rows, 1):
        if not row.strip():
            continue
        fields = row.split("\t", 2)
        if len(fields) != 3:
            raise NoticeError(
                f"{path}:{line_number}: expected wheel, classification, and origin"
            )
        filename, classification, origin = fields
        if Path(filename).name != filename or not filename.endswith(".whl"):
            raise NoticeError(
                f"{path}:{line_number}: invalid wheel filename {filename!r}")
        if classification not in BUNDLE_CLASSIFICATIONS:
            raise NoticeError(
                f"{path}:{line_number}: unknown classification {classification!r}"
            )
        if not origin:
            raise NoticeError(f"{path}:{line_number}: empty component origin")
        if filename in classifications:
            raise NoticeError(
                f"{path}:{line_number}: duplicate wheel {filename!r}")
        classifications[filename] = classification
    return classifications


def validate_wheels(
    inventory: dict[str, Any],
    wheels_dir: Path,
    components_path: Path | None = None,
    artifact_ids: Sequence[str] = (),
) -> list[str]:
    """Compare committed bundled-Python evidence with actual merge inputs."""
    errors: list[str] = []
    selected_artifacts = set(artifact_ids) or {"fat-wheel-bundled-python"}
    expected = {
        (canonical_name(component["name"]), public_version(component["version"])):
        component
        for component in inventory["components"]
        if component["ecosystem"] == "python"
        and selected_artifacts.intersection(component.get("artifact_ids", []))
    }
    wheels = sorted(wheels_dir.glob("*.whl"))
    if components_path is None:
        classifications = {wheel.name: "third-party" for wheel in wheels}
    else:
        try:
            classifications = _component_classifications(components_path)
        except NoticeError as error:
            return [str(error)]
        wheel_names = {wheel.name for wheel in wheels}
        unclassified = sorted(wheel_names - set(classifications))
        missing = sorted(set(classifications) - wheel_names)
        if unclassified:
            errors.append(
                f"unclassified wheels in merge directory: {unclassified}")
        if missing:
            errors.append(
                f"component manifest references missing wheels: {missing}")

    found: dict[tuple[str, str], tuple[Path, dict[str, str], str,
                                       set[str]]] = {}
    for wheel in wheels:
        try:
            (name, version, files, metadata_sha256,
             native_members) = wheel_record(wheel)
        except (NoticeError, zipfile.BadZipFile) as error:
            errors.append(str(error))
            continue
        key = (canonical_name(name), public_version(version))
        classification = classifications.get(wheel.name)
        if classification is None:
            continue
        if key in expected and classification != "third-party":
            errors.append(
                f"{wheel.name}: inventoried third-party wheel is classified as {classification!r}"
            )
            continue
        approved_names = {
            "kumo-first-party": FIRST_PARTY_DISTRIBUTIONS,
            "diskgraph-native": NATIVE_DISTRIBUTIONS,
        }.get(classification)
        if approved_names is not None and key[0] not in approved_names:
            errors.append(
                f"{wheel.name}: {key[0]!r} is not approved for the {classification!r} classification"
            )
            continue
        if classification != "third-party":
            continue
        if key not in expected:
            errors.append(
                f"{wheel.name}: third-party wheel is absent from the committed inventory"
            )
            continue
        if key in found:
            errors.append(
                f"duplicate bundled wheel for {name} {version}: {found[key][0].name} and {wheel.name}"
            )
        found[key] = (wheel, files, metadata_sha256, native_members)
    for key, component in expected.items():
        if key not in found:
            errors.append(
                f"bundled wheel missing for {component['name']} {component['version']}"
            )
            continue
        (wheel, actual_files, metadata_sha256, native_members) = found[key]
        recorded_files = {
            item["path"]: item["sha256"]
            for item in component["files"] if item["source"] == "python-wheel"
        }
        try:
            actual_files = normalized_legal_files(actual_files)
        except NoticeError as error:
            errors.append(f"{wheel.name}: {error}")
        if actual_files != recorded_files:
            errors.append(
                f"{wheel.name}: packaged license/notice files differ from the committed inventory"
            )
        if metadata_sha256 != component.get("wheel_metadata_sha256"):
            errors.append(
                f"{wheel.name}: license-bearing METADATA fields differ from the committed inventory"
            )
        native_audit = component.get("native_audit")
        if not isinstance(native_audit, dict):
            native_audit = {}
        recorded_native_members = {
            item.get("path")
            for item in native_audit.get("members", [])
            if isinstance(item, dict)
        }
        if native_members != recorded_native_members:
            errors.append(
                f"{wheel.name}: native member inventory differs from the committed audit"
            )
    return errors


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NoticeError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise NoticeError(f"{path}: top-level JSON value must be an object")
    if value.get("schema") != SCHEMA_VERSION:
        raise NoticeError(
            f"{path}: schema {value.get('schema')!r}; expected {SCHEMA_VERSION}"
        )
    return value


@dataclass(frozen=True)
class Expr:
    op: str
    value: str = ""
    left: "Expr | None" = None
    right: "Expr | None" = None


TOKEN_RE = re.compile(r"\s*(\(|\)|AND\b|OR\b|WITH\b|[A-Za-z0-9.+-]+)")


def _tokens(expression: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        match = TOKEN_RE.match(expression, position)
        if match is None:
            raise NoticeError(
                f"invalid SPDX expression near {expression[position:]!r}")
        tokens.append(match.group(1))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = _tokens(expression)
        self.index = 0

    def take(self, value: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise NoticeError(f"incomplete SPDX expression {self.expression!r}")
        token = self.tokens[self.index]
        if value is not None and token != value:
            raise NoticeError(
                f"expected {value!r}, got {token!r} in SPDX expression {self.expression!r}"
            )
        self.index += 1
        return token

    def parse(self) -> Expr:
        result = self.parse_or()
        if self.index != len(self.tokens):
            raise NoticeError(
                f"unexpected token {self.tokens[self.index]!r} in SPDX expression {self.expression!r}"
            )
        return result

    def parse_or(self) -> Expr:
        result = self.parse_and()
        while self.index < len(self.tokens) and self.tokens[self.index] == "OR":
            self.take("OR")
            result = Expr("or", left=result, right=self.parse_and())
        return result

    def parse_and(self) -> Expr:
        result = self.parse_with()
        while self.index < len(
                self.tokens) and self.tokens[self.index] == "AND":
            self.take("AND")
            result = Expr("and", left=result, right=self.parse_with())
        return result

    def parse_with(self) -> Expr:
        result = self.parse_atom()
        if self.index < len(self.tokens) and self.tokens[self.index] == "WITH":
            if result.op != "id":
                raise NoticeError(
                    f"WITH must follow an identifier in {self.expression!r}")
            self.take("WITH")
            exception = self.take()
            result = Expr("with", value=f"{result.value} WITH {exception}")
        return result

    def parse_atom(self) -> Expr:
        token = self.take()
        if token == "(":
            result = self.parse_or()
            self.take(")")
            return result
        if token in {"AND", "OR", "WITH", ")"}:
            raise NoticeError(
                f"unexpected token {token!r} in {self.expression!r}")
        return Expr("id", value=token)


def parse_spdx(expression: str) -> Expr:
    if not expression:
        raise NoticeError("empty SPDX expression")
    return _Parser(expression).parse()


def declared_spdx(expression: str) -> Expr:
    """Parse Cargo's legacy slash spelling as an SPDX OR expression."""
    return parse_spdx(re.sub(r"\s*/\s*", " OR ", expression))


def _choices(
    expr: Expr,
    selected_by_parent_or: bool = False,
) -> set[tuple[frozenset[str], frozenset[str]]]:
    """Return (obligations, recorded OR election) for every valid choice."""
    if expr.op in {"id", "with"}:
        leaf = frozenset({expr.value})
        election = leaf if selected_by_parent_or else frozenset()
        return {(leaf, election)}
    assert expr.left is not None and expr.right is not None
    if expr.op == "and":
        left = _choices(expr.left, selected_by_parent_or)
        right = _choices(expr.right, selected_by_parent_or)
        return {(left_obligations | right_obligations,
                 left_election | right_election)
                for left_obligations, left_election in left
                for right_obligations, right_election in right}
    if expr.op == "or":
        return _choices(expr.left, True) | _choices(expr.right, True)
    raise AssertionError(expr.op)


def expression_allowed(expression: str, elected: Sequence[str],
                       allowed: set[str]) -> bool:
    expr = parse_spdx(expression)
    elected_set = frozenset(elected)
    choices = _choices(expr)
    if "or" not in _ops(expr):
        if elected_set:
            return False
        obligations, _ = next(iter(choices))
        return all(identifier in allowed for identifier in obligations)
    matching_obligations = [
        obligations for obligations, election in choices
        if election == elected_set
    ]
    return bool(matching_obligations) and any(
        all(identifier in allowed for identifier in obligations)
        for obligations in matching_obligations)


def _ops(expr: Expr) -> set[str]:
    result = {expr.op}
    if expr.left is not None:
        result |= _ops(expr.left)
    if expr.right is not None:
        result |= _ops(expr.right)
    return result


def _identifiers(expr: Expr) -> set[str]:
    if expr.op in {"id", "with"}:
        return {expr.value}
    assert expr.left is not None and expr.right is not None
    return _identifiers(expr.left) | _identifiers(expr.right)


def _provenance_sources(path: Path) -> list[dict[str, Any]]:
    provenance = load_json(path)
    sources = provenance.get("sources")
    if not isinstance(sources, list):
        raise NoticeError(f"{path}: sources must be a list")
    if not all(isinstance(source, dict) for source in sources):
        raise NoticeError(f"{path}: every source must be an object")
    return sources


def _matched_source_files(root: Path, pattern: str) -> list[Path]:
    relative = Path(pattern)
    if relative.is_absolute() or ".." in relative.parts:
        raise NoticeError(f"{pattern!r}: source path must stay below {root}")
    glob_pattern = (f"{pattern.rstrip('/')}/*"
                    if relative.name == "**" else pattern)
    try:
        matches = sorted(path for path in root.glob(glob_pattern)
                         if path.is_file())
    except (OSError, ValueError) as error:
        raise NoticeError(
            f"{pattern!r}: invalid source path: {error}") from error
    if not matches:
        raise NoticeError(
            f"{pattern!r}: source path matches no files below {root}")
    return matches


def _source_provenance_errors(
        root: Path, inventory: dict[str, Any] | None = None) -> list[str]:
    diskgraph_root = root / "packages" / "diskgraph"
    path = diskgraph_root / "third_party" / "source-provenance.json"
    try:
        sources = _provenance_sources(path)
    except NoticeError as error:
        return [str(error)]
    errors: list[str] = []
    recorded_license_files: set[str] = set()
    recorded_sources: set[tuple[str, str]] = set()
    for index, source in enumerate(sources):
        required_strings = (
            "path",
            "upstream",
            "upstream_ref",
            "license_expression",
            "license_file",
            "header_policy",
        )
        invalid = [
            field for field in required_strings
            if not isinstance(source.get(field), str) or not source[field]
        ]
        if invalid:
            errors.append(
                f"{path}: source {index} has missing or invalid fields: {', '.join(invalid)}"
            )
            continue
        for field in ("modified", "apache_4b_notice_required"):
            if not isinstance(source.get(field), bool):
                errors.append(f"{source['path']}: {field} must be boolean")
        if source["header_policy"] not in {"retain-upstream"}:
            errors.append(
                f"{source['path']}: unsupported header_policy {source['header_policy']!r}"
            )
        identity = source.get("path", "<missing path>")
        source_key = (identity, source.get("upstream_ref", ""))
        if source_key in recorded_sources:
            errors.append(f"{identity}: duplicate source provenance record")
        recorded_sources.add(source_key)
        try:
            _matched_source_files(diskgraph_root, identity)
        except NoticeError as error:
            errors.append(str(error))
        upstream_ref = source.get("upstream_ref", "")
        if re.fullmatch(r"[0-9a-f]{40}", upstream_ref) is None:
            errors.append(
                f"{identity}: upstream_ref must be an immutable full commit SHA"
            )
        archive = source.get("upstream_archive")
        archive_sha256 = source.get("upstream_archive_sha256")
        if bool(archive) != bool(archive_sha256):
            errors.append(
                f"{identity}: upstream archive and checksum must be recorded together"
            )
        if archive_sha256 and re.fullmatch(r"[0-9a-f]{64}",
                                           archive_sha256) is None:
            errors.append(
                f"{identity}: upstream archive checksum is not a SHA-256 digest"
            )
        license_file = source.get("license_file", "")
        try:
            license_path = (diskgraph_root / license_file).resolve()
            license_path.relative_to(diskgraph_root.resolve())
        except (ValueError, OSError):
            errors.append(
                f"{identity}: license_file escapes packages/diskgraph")
            continue
        recorded_license_files.add(license_file)
        if not license_path.is_file():
            errors.append(f"{identity}: missing license_file {license_file!r}")
            continue
        if inventory is not None:
            matching_components = [
                component for component in inventory.get("components", [])
                if any(
                    file_record.get("source") == "vendored-source"
                    and file_record.get("path") == license_file
                    for file_record in component.get("files", []))
            ]
            if len(matching_components) != 1:
                errors.append(
                    f"{identity}: expected one inventory record for {license_file!r}"
                )
            else:
                component = matching_components[0]
                matching_record = next(
                    file_record for file_record in component["files"]
                    if file_record.get("source") == "vendored-source"
                    and file_record.get("path") == license_file)
                if matching_record.get("sha256") != sha256_bytes(
                        license_path.read_bytes()):
                    errors.append(
                        f"{identity}: vendored license hash disagrees with inventory"
                    )
                if component.get("license_expression") != source.get(
                        "license_expression"):
                    errors.append(
                        f"{identity}: provenance license expression disagrees with inventory"
                    )
    if inventory is not None:
        inventory_license_files = {
            file_record["path"]
            for component in inventory.get("components", [])
            for file_record in component.get("files", [])
            if file_record.get("source") == "vendored-source"
        }
        missing_provenance = sorted(inventory_license_files -
                                    recorded_license_files)
        if missing_provenance:
            errors.append(
                f"vendored license files lack source provenance: {missing_provenance}"
            )
    return errors


def _workspace_source_provenance_errors(
        root: Path, inventory: dict[str, Any] | None = None) -> list[str]:
    path = root / "3rdparty" / "notices" / "source-provenance.json"
    try:
        sources = _provenance_sources(path)
    except NoticeError as error:
        return [str(error)]
    errors: list[str] = []
    recorded_license_files: set[str] = set()
    recorded_sources: set[tuple[str, str]] = set()
    for index, source in enumerate(sources):
        required_strings = (
            "path",
            "upstream",
            "upstream_path",
            "upstream_ref",
            "license_expression",
            "license_file",
        )
        invalid = [
            field for field in required_strings
            if not isinstance(source.get(field), str) or not source[field]
        ]
        if invalid:
            errors.append(
                f"{path}: source {index} has missing or invalid fields: {', '.join(invalid)}"
            )
            continue
        if not isinstance(source.get("modified"), bool):
            errors.append(f"{source['path']}: modified must be boolean")
        identity = source.get("path", "<missing path>")
        source_key = (identity, source.get("upstream_ref", ""))
        if source_key in recorded_sources:
            errors.append(f"{identity}: duplicate source provenance record")
        recorded_sources.add(source_key)
        try:
            _matched_source_files(root, identity)
        except NoticeError as error:
            errors.append(str(error))
        upstream_ref = source.get("upstream_ref", "")
        if re.fullmatch(r"[0-9a-f]{40}", upstream_ref) is None:
            errors.append(
                f"{identity}: upstream_ref must be an immutable full commit SHA"
            )
        license_file = source.get("license_file", "")
        try:
            license_path = (root / license_file).resolve()
            license_path.relative_to(root.resolve())
        except (ValueError, OSError):
            errors.append(f"{identity}: license_file escapes the repository")
            continue
        recorded_license_files.add(license_file)
        if not license_path.is_file():
            errors.append(f"{identity}: missing license_file {license_file!r}")
            continue
        if inventory is not None:
            matching_components = [
                component for component in inventory.get("components", [])
                if any(
                    file_record.get("source") == "workspace-vendored-source"
                    and file_record.get("path") == license_file
                    for file_record in component.get("files", []))
            ]
            if len(matching_components) != 1:
                errors.append(
                    f"{identity}: expected one inventory record for {license_file!r}"
                )
            else:
                component = matching_components[0]
                matching_record = next(
                    file_record for file_record in component["files"]
                    if file_record.get("source") == "workspace-vendored-source"
                    and file_record.get("path") == license_file)
                if matching_record.get("sha256") != sha256_bytes(
                        license_path.read_bytes()):
                    errors.append(
                        f"{identity}: vendored license hash disagrees with inventory"
                    )
                if component.get("license_expression") != source.get(
                        "license_expression"):
                    errors.append(
                        f"{identity}: provenance license expression disagrees with inventory"
                    )
    if inventory is not None:
        inventory_license_files = {
            file_record["path"]
            for component in inventory.get("components", [])
            for file_record in component.get("files", [])
            if file_record.get("source") == "workspace-vendored-source"
        }
        missing_provenance = sorted(inventory_license_files -
                                    recorded_license_files)
        if missing_provenance:
            errors.append(
                f"vendored license files lack source provenance: {missing_provenance}"
            )
    return errors


def _derived_source_policy_errors(root: Path) -> list[str]:
    """Require each derived header to have the legal text it declares."""
    header_policy_path = root / "scripts" / "license_header_policy.json"
    provenance_path = root / "3rdparty" / "notices" / "source-provenance.json"
    try:
        header_policy = json.loads(header_policy_path.read_text("utf-8"))
        if header_policy.get("version") != 1:
            raise NoticeError(
                f"{header_policy_path}: unsupported policy version")
        sources = _provenance_sources(provenance_path)
    except (OSError, json.JSONDecodeError, NoticeError) as error:
        return [str(error)]

    errors: list[str] = []
    provenance_by_path: dict[str, set[str]] = {}
    for source in sources:
        path = source.get("path")
        expression = source.get("license_expression")
        if not isinstance(path, str) or not isinstance(expression, str):
            continue
        try:
            identifiers = _identifiers(parse_spdx(expression))
        except NoticeError as error:
            errors.append(f"{path}: {error}")
            continue
        provenance_by_path.setdefault(path, set()).update(identifiers)

    derived_paths: set[str] = set()
    for source in header_policy.get("derived_sources", []):
        paths = source.get("paths")
        expression = source.get("license_identifier")
        if (not isinstance(paths, list)
                or any(not isinstance(path, str) for path in paths)
                or not isinstance(expression, str)):
            errors.append("license header policy has an invalid derived source")
            continue
        try:
            identifiers = _identifiers(parse_spdx(expression)) - {"Apache-2.0"}
        except NoticeError as error:
            errors.append(f"{source.get('name', '<unnamed>')}: {error}")
            continue

        notice_file = source.get("notice_file")
        notice_text: str | None = None
        if notice_file is not None:
            if not isinstance(notice_file, str):
                errors.append(
                    f"{source.get('name', '<unnamed>')}: notice_file must be a string"
                )
            else:
                try:
                    notice_path = (root / notice_file).resolve()
                    notice_path.relative_to(root.resolve())
                    notice_text = notice_path.read_text("utf-8")
                except (OSError, ValueError) as error:
                    errors.append(
                        f"{notice_file}: invalid notice file: {error}")

        for path in paths:
            derived_paths.add(path)
            if notice_file is None:
                if path not in provenance_by_path:
                    errors.append(
                        f"{path}: derived-source policy lacks workspace provenance"
                    )
                elif identifiers != provenance_by_path[path]:
                    errors.append(
                        f"{path}: derived-source licenses disagree with workspace provenance"
                    )
                continue
            if notice_text is None:
                continue
            for identifier in identifiers:
                if f"License: {identifier}" not in notice_text:
                    errors.append(
                        f"{path}: {notice_file} lacks {identifier} attribution")
            for copyright_line in source.get("required_copyright_lines", []):
                copyright_text = copyright_line.removeprefix(
                    "SPDX-FileCopyrightText: ")
                if copyright_text not in notice_text:
                    errors.append(
                        f"{path}: {notice_file} lacks required copyright text")

    for path in sorted(provenance_by_path.keys() - derived_paths):
        errors.append(
            f"{path}: workspace provenance lacks derived-source policy")
    return errors


def _component_key(component: dict[str, Any]) -> tuple[str, str, str]:
    return (component["ecosystem"], canonical_name(component["name"]),
            component["version"])


def _exception_key(exception: dict[str, Any]) -> tuple[str, str, str]:
    return (
        exception["ecosystem"],
        canonical_name(exception["name"]),
        exception["version"],
    )


def _valid_keyed_records(
    records: Any,
    record_type: str,
    required_fields: tuple[str, ...],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        errors.append(f"{record_type} records must be a list")
        return []
    valid: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"{record_type} {index}: must be an object")
            continue
        invalid = [
            field for field in required_fields
            if not isinstance(record.get(field), str) or not record[field]
        ]
        if invalid:
            errors.append(
                f"{record_type} {index}: missing or invalid required string fields: {', '.join(invalid)}"
            )
            continue
        valid.append(record)
    return valid


def _diskgraph_manifest_errors(inventory: dict[str, Any],
                               root: Path) -> list[str]:
    errors: list[str] = []
    diskgraph_root = root / "packages" / "diskgraph"
    manifest_path = diskgraph_root / "third_party" / "artifact-manifest.json"
    if not manifest_path.is_file():
        return [f"missing resolved DiskGraph manifest: {manifest_path}"]
    try:
        manifest = load_json(manifest_path)
    except (NoticeError, OSError) as error:
        return [f"invalid resolved DiskGraph manifest: {error}"]

    lock_hash = manifest.get("cargo_lock_sha256")
    lock_path = diskgraph_root / "Cargo.lock"
    current_lock_hash = (sha256_bytes(lock_path.read_bytes())
                         if lock_path.is_file() else "absent")
    if lock_hash != current_lock_hash:
        errors.append(
            "resolved DiskGraph Cargo.lock hash disagrees with in-tree source")

    inventory_components = {
        item["id"]: item
        for item in inventory.get("components", [])
    }
    inventory_artifacts = {
        item.get("id"): item
        for item in inventory.get("artifacts", [])
    }
    manifest_artifacts = {
        item.get("id"): item
        for item in manifest.get("artifacts", [])
    }
    for artifact_id in ("diskgraph-extension", "diskgraph-ingest"):
        artifact = inventory_artifacts.get(artifact_id)
        resolved = manifest_artifacts.get(artifact_id)
        if artifact is None or resolved is None:
            errors.append(
                f"{artifact_id}: missing inventory or resolved manifest artifact"
            )
            continue
        for field in ("root", "targets", "features", "default_features",
                      "dependency_kinds"):
            if artifact.get(field) != resolved.get(field):
                errors.append(
                    f"{artifact_id}: {field} disagrees with resolved manifest")

        inventory_packages = set()
        for component_id in artifact.get("components", []):
            component = inventory_components.get(component_id)
            if component is not None and component.get("ecosystem") == "rust":
                inventory_packages.add(
                    (canonical_name(component["name"]), component["version"]))
        resolved_packages = {
            (canonical_name(package["name"]), package["version"])
            for package in resolved.get("packages", [])
            if package.get("source_kind") != "workspace-first-party"
        }
        if inventory_packages != resolved_packages:
            missing = sorted(resolved_packages - inventory_packages)[:5]
            extra = sorted(inventory_packages - resolved_packages)[:5]
            errors.append(
                f"{artifact_id}: Rust package set disagrees with resolved manifest; missing={missing}, extra={extra}; "
                f"refresh with: {DISKGRAPH_REFRESH_COMMANDS}")

        for package in resolved.get("packages", []):
            if package.get("source_kind") == "workspace-first-party":
                continue
            key = (canonical_name(package["name"]), package["version"])
            matches = [
                component for component in inventory_components.values()
                if component.get("ecosystem") == "rust" and
                (canonical_name(component["name"]), component["version"]) == key
            ]
            if len(matches) == 1 and matches[0].get(
                    "declared_expression") != package.get("license"):
                errors.append(
                    f"{artifact_id}: {key} declared license disagrees with resolved manifest"
                )
    return errors


def validate(
    inventory: dict[str, Any],
    policy: dict[str, Any],
    root: Path = REPO_ROOT,
    wheels_dir: Path | None = None,
    components_path: Path | None = None,
    artifact_ids: Sequence[str] = (),
) -> list[str]:
    errors: list[str] = []
    allowed = set(policy.get("allowed_licenses", []))
    exception_items = _valid_keyed_records(
        policy.get("exceptions", []),
        "policy exception",
        ("ecosystem", "name", "version"),
        errors,
    )
    exceptions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in exception_items:
        key = _exception_key(item)
        if key in exceptions:
            errors.append("duplicate package-and-version policy exception")
        exceptions[key] = item
    components = _valid_keyed_records(
        inventory.get("components", []),
        "inventory component",
        ("id", "ecosystem", "name", "version"),
        errors,
    )
    components = [dict(component) for component in components]
    artifacts = _valid_keyed_records(
        inventory.get("artifacts", []),
        "inventory artifact",
        ("id", "label"),
        errors,
    )
    artifacts = [dict(artifact) for artifact in artifacts]
    for component in components:
        files = component.get("files")
        if not isinstance(files, list):
            errors.append(f"{component['id']}: files must be a list")
            component["files"] = []
            continue
        component["files"] = _valid_keyed_records(
            files,
            f"{component['id']} legal file",
            ("path", "sha256", "source"),
            errors,
        )
        artifact_ids = component.get("artifact_ids")
        if (not isinstance(artifact_ids, list)
                or any(not isinstance(item, str) or not item
                       for item in artifact_ids)):
            errors.append(
                f"{component['id']}: artifact_ids must be a string list")
            component["artifact_ids"] = []
    for artifact in artifacts:
        component_ids = artifact.get("components")
        if (not isinstance(component_ids, list)
                or any(not isinstance(item, str) or not item
                       for item in component_ids)):
            errors.append(f"{artifact['id']}: components must be a string list")
            artifact["components"] = []
    texts = inventory.get("texts", {})
    if not isinstance(texts, dict):
        errors.append("inventory texts must be an object")
        texts = {}

    validated_inventory = dict(inventory)
    validated_inventory["components"] = components
    validated_inventory["artifacts"] = artifacts
    errors.extend(_diskgraph_manifest_errors(validated_inventory, root))
    errors.extend(_source_provenance_errors(root, validated_inventory))
    errors.extend(_workspace_source_provenance_errors(root,
                                                      validated_inventory))
    errors.extend(_derived_source_policy_errors(root))

    ids: set[str] = set()
    component_keys: set[tuple[str, str, str]] = set()
    used_exceptions: set[tuple[str, str, str]] = set()
    referenced_texts: set[str] = set()
    for component in components:
        identity = component.get("id", "<missing id>")
        if identity in ids:
            errors.append(f"duplicate component id: {identity}")
        ids.add(identity)
        key = _component_key(component)
        if key in component_keys:
            errors.append(f"duplicate component: {key}")
        component_keys.add(key)

        expression = component.get("license_expression", "")
        elected = component.get("elected_licenses", [])
        declared = component.get("declared_expression", "")
        if component.get("ecosystem") == "rust":
            try:
                if _choices(declared_spdx(declared)) != _choices(
                        parse_spdx(expression)):
                    errors.append(
                        f"{identity}: normalized license expression disagrees with Cargo metadata"
                    )
            except NoticeError as error:
                errors.append(f"{identity}: invalid declared license: {error}")
        try:
            permitted = expression_allowed(expression, elected, allowed)
        except NoticeError as error:
            errors.append(f"{identity}: {error}")
            permitted = False
        if not permitted:
            exception = exceptions.get(key)
            if exception is None or exception.get(
                    "license_expression") != expression:
                errors.append(
                    f"{identity}: disallowed or unelected license expression {expression!r}"
                )
            else:
                used_exceptions.add(key)
                for field in ("review_reference", "obligations", "reason"):
                    if not exception.get(field):
                        errors.append(
                            f"{identity}: policy exception lacks {field}")

        if not component.get("review_reference"):
            errors.append(f"{identity}: missing license review reference")

        if (component.get("ecosystem") == "python"
                and "fat-wheel-bundled-python" in component.get(
                    "artifact_ids", [])):
            native_audit = component.get("native_audit")
            if native_audit is not None:
                if not isinstance(native_audit, dict):
                    errors.append(f"{identity}: native_audit must be an object")
                else:
                    if not isinstance(native_audit.get("decision"),
                                      str) or not native_audit["decision"]:
                        errors.append(
                            f"{identity}: native_audit lacks a review decision")
                    members = native_audit.get("members")
                    if not isinstance(members, list) or not members:
                        errors.append(
                            f"{identity}: native_audit must list native members"
                        )
                    else:
                        member_paths: set[str] = set()
                        for member in members:
                            if not isinstance(member, dict):
                                errors.append(
                                    f"{identity}: native member record must be an object"
                                )
                                continue
                            member_path = member.get("path")
                            if (not isinstance(member_path, str)
                                    or not member_path
                                    or member_path in member_paths):
                                errors.append(
                                    f"{identity}: native member paths must be unique nonempty strings"
                                )
                            else:
                                member_paths.add(member_path)
                    embedded_components = native_audit.get(
                        "embedded_components")
                    if (not isinstance(embedded_components, list)
                            or any(not isinstance(item, str) or not item
                                   for item in embedded_components)):
                        errors.append(
                            f"{identity}: native_audit embedded_components must be a string list"
                        )

        files = component.get("files", [])
        if not files:
            errors.append(
                f"{identity}: no upstream text and no canonical backfill")
        if [item.get("path", "") for item in files
            ] != sorted(item.get("path", "") for item in files):
            errors.append(
                f"{identity}: license files are not in stable filename order")
        for item in files:
            digest = item.get("sha256", "")
            referenced_texts.add(digest)
            text = texts.get(digest)
            if text is None:
                errors.append(
                    f"{identity}: missing text {digest} for {item.get('path')}")
                continue
            if sha256_text(text) != digest:
                errors.append(
                    f"{identity}: text hash mismatch for {item.get('path')}")
            if item.get("source") == "spdx-backfill":
                provenance = item.get("provenance") or {}
                if provenance.get("version") != inventory.get(
                        "tools", {}).get("spdx_license_list_data"):
                    errors.append(
                        f"{identity}: SPDX backfill version is missing or inconsistent"
                    )
                if not provenance.get("declared_in") or not provenance.get(
                        "license_id"):
                    errors.append(
                        f"{identity}: incomplete canonical-backfill provenance")

        serialized = json.dumps(component, ensure_ascii=False)
        if PERSONAL_PATH_RE.search(serialized):
            errors.append(
                f"{identity}: personal/build-machine path in provenance")

    if referenced_texts != set(texts):
        missing = sorted(set(texts) - referenced_texts)
        errors.append(f"unreferenced license texts in inventory: {missing[:5]}")

    components_by_id = {component["id"]: component for component in components}
    embedded_component_owners: dict[str, str] = {}
    for component in components:
        native_audit = component.get("native_audit")
        if not isinstance(native_audit, dict):
            continue
        for embedded_id in native_audit.get("embedded_components", []):
            if not isinstance(embedded_id, str):
                continue
            previous = embedded_component_owners.get(embedded_id)
            if previous is not None:
                errors.append(
                    f"{embedded_id}: listed by multiple native wheel audits: {previous}, {component['id']}"
                )
            embedded_component_owners[embedded_id] = component["id"]
            embedded = components_by_id.get(embedded_id)
            if embedded is None:
                errors.append(
                    f"{component['id']}: unknown embedded component {embedded_id}"
                )
                continue
            embedded_in = embedded.get("embedded_in") or {}
            audited_member_paths = {
                item.get("path")
                for item in native_audit.get("members", [])
                if isinstance(item, dict)
            }
            if embedded.get("ecosystem") != "native":
                errors.append(
                    f"{embedded_id}: embedded component must use the native ecosystem"
                )
            if embedded_in.get("component_id") != component["id"]:
                errors.append(
                    f"{embedded_id}: embedded_in component disagrees with its native wheel audit"
                )
            if embedded_in.get("member") not in audited_member_paths:
                errors.append(
                    f"{embedded_id}: embedded_in member is absent from its parent native-member audit"
                )
    native_ids = {
        component["id"]
        for component in components if component.get("ecosystem") == "native"
    }
    unowned_native = sorted(native_ids - set(embedded_component_owners))
    if unowned_native:
        errors.append(
            f"native components outside every wheel audit: {unowned_native[:5]}"
        )

    stale_exceptions = sorted(set(exceptions) - used_exceptions)
    if stale_exceptions:
        errors.append(f"stale policy exceptions: {stale_exceptions}")

    known_artifact_ids: set[str] = set()
    recorded_scopes: dict[str, set[str]] = {}
    for artifact in artifacts:
        artifact_id = artifact.get("id", "")
        if not artifact_id or artifact_id in known_artifact_ids:
            errors.append(f"duplicate or missing artifact id: {artifact_id!r}")
        known_artifact_ids.add(artifact_id)
        artifact_components = artifact.get("components", [])
        if len(artifact_components) != len(set(artifact_components)):
            errors.append(f"{artifact_id}: duplicate component ids")
        unknown = sorted(set(artifact_components) - ids)
        if unknown:
            errors.append(
                f"{artifact_id}: unknown component ids: {unknown[:5]}")
        for component_id in artifact_components:
            recorded_scopes.setdefault(component_id, set()).add(artifact_id)

    for component in components:
        identity = component["id"]
        declared_scopes = set(component.get("artifact_ids", []))
        if declared_scopes != recorded_scopes.get(identity, set()):
            errors.append(
                f"{identity}: component artifact_ids disagree with artifact inventories"
            )
        unknown_scopes = sorted(declared_scopes - known_artifact_ids)
        if unknown_scopes:
            errors.append(
                f"{identity}: unknown artifact_ids: {unknown_scopes[:5]}")

    used_component_ids = {
        component_id
        for artifact in artifacts
        for component_id in artifact.get("components", [])
    }
    unused_components = sorted(ids - used_component_ids)
    if unused_components:
        errors.append(
            f"components outside every artifact/source scope: {unused_components[:5]}"
        )
    if wheels_dir is not None:
        errors.extend(validate_wheels(inventory, wheels_dir, components_path,
                                      artifact_ids))
    elif components_path is not None:
        errors.append("a component manifest requires a wheel directory")
    return errors


def _indent(text: str) -> Iterator[str]:
    for line in text.rstrip("\n").split("\n"):
        yield f"    {line}" if line else ""


def _components_for_artifacts(
    inventory: dict[str, Any],
    artifact_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    artifact_labels = {
        artifact["id"]: artifact["label"]
        for artifact in inventory["artifacts"]
    }
    selected_artifacts = set(artifact_ids)
    unknown_artifacts = sorted(selected_artifacts - set(artifact_labels))
    if unknown_artifacts:
        raise NoticeError(f"unknown artifact ids: {unknown_artifacts}")
    components = [
        component for component in inventory["components"]
        if not selected_artifacts
        or selected_artifacts.intersection(component["artifact_ids"])
    ]
    if selected_artifacts and not components:
        raise NoticeError("selected artifacts contain no components")
    return components, artifact_labels


def selected_license_expression(
        inventory: dict[str, Any],
        policy: dict[str, Any],
        *,
        artifact_ids: Sequence[str] = (),
        base_licenses: Sequence[str] = (),
) -> str:
    errors = validate(inventory, policy)
    if errors:
        raise NoticeError("\n".join(errors))
    components, _ = _components_for_artifacts(inventory, artifact_ids)
    obligations: set[str] = set()
    for expression in base_licenses:
        parsed = parse_spdx(expression)
        if "or" in _ops(parsed):
            raise NoticeError(
                f"base license requires an explicit election: {expression!r}")
        selected, _ = next(iter(_choices(parsed)))
        obligations.update(selected)
    for component in components:
        parsed = parse_spdx(component["license_expression"])
        choices = _choices(parsed)
        if "or" in _ops(parsed):
            election = frozenset(component.get("elected_licenses", []))
            selected = {
                choice
                for choice, recorded_election in choices
                if recorded_election == election
            }
        else:
            selected = {choice for choice, _ in choices}
        if len(selected) != 1:
            raise NoticeError(
                f"{component['id']}: license election does not resolve to one obligation set"
            )
        obligations.update(next(iter(selected)))
    return " AND ".join(sorted(obligations))


def render(
        inventory: dict[str, Any],
        policy: dict[str, Any],
        wheels_dir: Path | None = None,
        components_path: Path | None = None,
        artifact_ids: Sequence[str] = (),
) -> bytes:
    errors = validate(inventory, policy, wheels_dir=wheels_dir,
                      components_path=components_path, artifact_ids=artifact_ids)
    if errors:
        raise NoticeError("\n".join(errors))
    components, artifact_labels = _components_for_artifacts(
        inventory, artifact_ids)
    selected_artifacts = set(artifact_ids)
    text_ids = {digest: digest[:12] for digest in inventory["texts"]}
    emitted_texts: set[str] = set()
    lines = [
        BAR,
        "Kumo third-party software notices",
        BAR,
        "",
        "Third-party components redistributed in Kumo artifacts are listed below.",
        "",
        BAR,
        "THIRD-PARTY COMPONENTS",
        BAR,
        "",
    ]
    for component in sorted(components, key=_component_key):
        heading = component.get("display_name") or (
            f"{component['name']} {component['version']}")
        lines.extend([RULE, heading, RULE])
        lines.append(f"  License       : {component['license_expression']}")
        scopes = sorted(artifact_labels[item]
                        for item in component["artifact_ids"]
                        if not selected_artifacts or item in selected_artifacts)
        lines.append(f"  Distributed in: {', '.join(scopes)}")
        if component.get("copyright"):
            lines.append(f"  Copyright     : {component['copyright'][0]}")
            lines.extend(f"                  {item}"
                         for item in component["copyright"][1:])
        lines.append("")
        for file_record in component["files"]:
            digest = file_record["sha256"]
            text_id = text_ids[digest]
            lines.append(f"  Legal text {text_id}: {file_record['path']}")
            if digest in emitted_texts:
                continue
            emitted_texts.add(digest)
            lines.append("")
            lines.extend(_indent(inventory["texts"][digest]))
            lines.append("")

    rendered = "\n".join(lines).rstrip() + "\n"
    if PERSONAL_PATH_RE.search(rendered):
        raise NoticeError(
            "rendered notice contains a personal/build-machine path")
    for forbidden in ("[URL]", "[monitored email]", "NOT SHIPPED"):
        if forbidden in rendered:
            raise NoticeError(
                f"rendered notice contains forbidden placeholder {forbidden!r}")
    return rendered.encode("utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--wheels",
        type=Path,
        help="validate bundled distributions before rendering",
    )
    parser.add_argument(
        "--components",
        type=Path,
        help="classify every wheel using the merger's component manifest",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="render only components in this artifact; may be repeated",
    )
    parser.add_argument(
        "--base-license",
        action="append",
        default=[],
        help="include a first-party license in --license-expression output",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="write a generated notice")
    mode.add_argument("--check", type=Path,
                      help="byte-compare a notice with generated output")
    mode.add_argument(
        "--license-expression",
        action="store_true",
        help="print the selected artifacts' combined SPDX expression",
    )
    args = parser.parse_args(argv)
    if args.components is not None and args.wheels is None:
        parser.error("--components is only valid with --wheels")
    if args.base_license and not args.license_expression:
        parser.error("--base-license requires --license-expression")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inventory = load_json(args.inventory)
        policy = load_json(args.policy)
        if args.license_expression:
            print(
                selected_license_expression(
                    inventory,
                    policy,
                    artifact_ids=args.artifact,
                    base_licenses=args.base_license,
                ))
            return 0
        generated = render(
            inventory,
            policy,
            wheels_dir=args.wheels,
            components_path=args.components,
            artifact_ids=args.artifact,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(generated)
            print(f"wrote {args.output} ({len(generated)} bytes)")
            return 0
        actual = args.check.read_bytes() if args.check.is_file() else b""
        if actual != generated:
            command = ["scripts/gen_third_party.py"]
            for artifact in args.artifact:
                command.extend(("--artifact", artifact))
            command.extend(("--output", str(args.check)))
            print(
                f"{args.check} is stale; run: {shlex.join(command)}",
                file=sys.stderr,
            )
            return 1
        selected_artifacts = set(args.artifact)
        component_count = sum(
            1 for component in inventory["components"] if not selected_artifacts
            or selected_artifacts.intersection(component["artifact_ids"]))
        print(f"{args.check}: current ({component_count} components)")
        return 0
    except (NoticeError, OSError) as error:
        print(f"third-party notice error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
