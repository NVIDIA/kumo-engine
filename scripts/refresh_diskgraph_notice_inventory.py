#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile DiskGraph Rust components in the notice inventory.

The resolved DiskGraph artifact manifest lives in packages/diskgraph. This
tool makes the Rust components of ``3rdparty/notices/inventory.json`` match it:
components that are still shipped keep their records, components that left
both artifacts are removed, and new crate versions are derived from the legal
files in Cargo's source cache. Run ``cargo fetch --locked`` for the DiskGraph
workspace first; Cargo is consulted only when a component is new.

``gen_third_party.py --check`` validates the result. It rejects a license
outside the policy, a crate without license text when no reviewed SPDX text
can be reused, and a shipped crate missing from the inventory, such as a
vendored path crate, which this tool skips.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__:
    from .gen_third_party import (
        DEFAULT_INVENTORY,
        DEFAULT_POLICY,
        DISKGRAPH_ROOT,
        NoticeError,
        _choices,
        _ops,
        canonical_name,
        declared_spdx,
        load_json,
        sha256_text,
    )
else:
    from gen_third_party import (
        DEFAULT_INVENTORY,
        DEFAULT_POLICY,
        DISKGRAPH_ROOT,
        NoticeError,
        _choices,
        _ops,
        canonical_name,
        declared_spdx,
        load_json,
        sha256_text,
    )

DISKGRAPH_ARTIFACTS = ("diskgraph-extension", "diskgraph-ingest")
RESOLVED_ARTIFACT_FIELDS = ("root", "targets", "features", "default_features",
                            "dependency_kinds")
# Preferred elections among allowed choices of a Cargo OR expression; the
# reviewed inventory elects Apache-2.0, then MIT.
ELECTION_PREFERENCE = ("Apache-2.0", "MIT")
# Components admitted by this tool are covered by the allowed-license policy
# reviewed in OSRB-6649311 rather than by a package-specific review.
AUTOMATED_REVIEW_REFERENCE = ("OSRB-6649311 allowed-license policy; "
                              "added by DiskGraph synchronization")
LEGAL_FILE_RE = re.compile(r"(?i)^(?:licen[cs]e|copying|unlicense|notice)")
COPYRIGHT_LINE_RE = re.compile(
    r"^[\s#*/;!-]*((?:Copyright|COPYRIGHT|©)(?:\s|$).*)$")
COPYRIGHT_OWNER_RE = re.compile(
    r"^(?:Copyright|COPYRIGHT)\s+(?:(?:\([cC]\)|©)\s*)?"
    r"(?!(?:License|Convention|Notice|Holders?|Owners?|Act|Law|Statement|"
    r"Protection|AND|HOLDERS?|OWNERS?|NOTICE|LAW)\b)\[?[0-9A-Z]")
COPYRIGHT_PLACEHOLDER_RE = re.compile(
    r"\[yyyy\]|\{yyyy\}|<year>|<owner>|\[name of|\{name of|<name of|"
    r"<copyright holders?>|\[fullname\]|\{fullname\}", re.IGNORECASE)


class RefreshError(NoticeError):
    """The inventory cannot be reconciled with the resolved manifest."""


def cargo_metadata(diskgraph_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "cargo", "metadata", "--format-version", "1", "--locked",
            "--offline", "--manifest-path",
            str(diskgraph_root / "Cargo.toml")
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def normalized_expression(declared: Any) -> str:
    """Spell Cargo's legacy slash alternatives as an SPDX OR expression."""
    if not isinstance(declared, str):
        raise RefreshError("Cargo metadata declares no license expression")
    expression = re.sub(r"\s*/\s*", " OR ", declared.strip())
    declared_spdx(expression)
    return expression


def elected_licenses(expression: str, allowed: set[str]) -> list[str]:
    expr = declared_spdx(expression)
    if "or" not in _ops(expr):
        return []
    elections = {
        election
        for obligations, election in _choices(expr)
        if all(identifier in allowed for identifier in obligations)
    }
    for preferred in ELECTION_PREFERENCE:
        if frozenset({preferred}) in elections:
            return [preferred]
    if not elections:
        return []
    return sorted(min(elections, key=lambda item: (len(item), sorted(item))))


def _obligations(expression: str, elected: Sequence[str]) -> list[str]:
    election = frozenset(elected)
    for obligations, recorded in sorted(_choices(declared_spdx(expression)),
                                        key=lambda item: sorted(item[0])):
        if recorded == election:
            return sorted(obligations)
    return []


def copyright_lines(texts: Sequence[str]) -> list[str]:
    """Return ownership statements, without license prose or templates."""
    statements: list[str] = []
    for text in texts:
        for line in text.split("\n"):
            match = COPYRIGHT_LINE_RE.match(line)
            if match is None:
                continue
            statement = " ".join(match.group(1).split()).rstrip(".").rstrip()
            if COPYRIGHT_PLACEHOLDER_RE.search(statement):
                continue
            if not (statement.startswith("©")
                    or COPYRIGHT_OWNER_RE.match(statement)):
                continue
            if statement not in statements:
                statements.append(statement)
    return statements


def _legal_texts(package_dir: Path) -> list[tuple[str, str]]:
    texts = []
    for path in sorted(package_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or not LEGAL_FILE_RE.match(path.name):
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            print(f"warning: skipping non-UTF-8 legal file {path}: {error}",
                  file=sys.stderr)
            continue
        texts.append((path.name, text.replace("\r\n", "\n")))
    return texts


def _backfill_texts(inventory: dict[str, Any]) -> dict[str, str]:
    """Map SPDX identifiers to reviewed canonical texts already inventoried."""
    version = inventory.get("tools", {}).get("spdx_license_list_data")
    backfills: dict[str, str] = {}
    for component in inventory.get("components", []):
        for item in component.get("files", []):
            provenance = item.get("provenance") or {}
            if (item.get("source") == "spdx-backfill"
                    and provenance.get("version") == version
                    and provenance.get("license_id")):
                backfills.setdefault(provenance["license_id"], item["sha256"])
    return backfills


def derive_component(
    package: dict[str, Any],
    metadata_package: dict[str, Any],
    artifact_ids: list[str],
    allowed: set[str],
    inventory: dict[str, Any],
    texts: dict[str, str],
) -> dict[str, Any]:
    """Derive one Rust component from its Cargo package source."""
    manifest_path = Path(metadata_package["manifest_path"])
    package_dir = manifest_path.parent
    cache_dir = f"{package_dir.parent.name}/{package_dir.name}"
    declared = package.get("license")
    expression = normalized_expression(declared)
    elected = elected_licenses(expression, allowed)

    files: list[dict[str, Any]] = []
    legal_texts = _legal_texts(package_dir)
    for name, text in legal_texts:
        digest = sha256_text(text)
        texts.setdefault(digest, text)
        files.append({
            "path": name,
            "sha256": digest,
            "source": "cargo-package",
        })
    if files:
        expression_source = f"{cache_dir}/{files[0]['path']}"
    else:
        expression_source = "Cargo.toml: license"
        version = inventory.get("tools", {}).get("spdx_license_list_data")
        backfills = _backfill_texts(inventory)
        obligations = _obligations(expression, elected)
        missing = [item for item in obligations if item not in backfills]
        if missing:
            # Left without files, which validation rejects.
            obligations = []
        for license_id in obligations:
            files.append({
                "path": f"SPDX-{license_id}.txt",
                "provenance": {
                    "declared_in": f"{package_dir.name}/Cargo.toml",
                    "license_id": license_id,
                    "version": version,
                },
                "sha256": backfills[license_id],
                "source": "spdx-backfill",
            })
        files.sort(key=lambda item: item["path"])

    origin = (metadata_package.get("repository")
              or metadata_package.get("homepage")
              or f"https://crates.io/crates/{package['name']}")
    return {
        "artifact_ids": artifact_ids,
        "copyright": copyright_lines([text for _, text in legal_texts]),
        "declared_expression": declared,
        "ecosystem": "rust",
        "elected_licenses": elected,
        "expression_source": expression_source,
        "files": files,
        "id": f"rust:{canonical_name(package['name'])}@{package['version']}",
        "license_expression": expression,
        "name": package["name"],
        "origin": origin,
        "version": package["version"],
        "review_reference": AUTOMATED_REVIEW_REFERENCE,
    }


def refresh(
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    policy: dict[str, Any],
    metadata: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Return an inventory whose Rust components match ``manifest``."""
    allowed = set(policy.get("allowed_licenses", []))
    resolved_artifacts = {
        item.get("id"): item
        for item in manifest.get("artifacts", [])
    }
    shipped: dict[tuple[str, str], dict[str, Any]] = {}
    scopes: dict[tuple[str, str], set[str]] = {}
    for artifact_id in DISKGRAPH_ARTIFACTS:
        resolved = resolved_artifacts.get(artifact_id)
        if resolved is None:
            raise RefreshError(f"resolved manifest lacks {artifact_id}")
        for package in resolved.get("packages", []):
            if package.get("source_kind") == "workspace-first-party":
                continue
            key = (canonical_name(package["name"]), package["version"])
            shipped.setdefault(key, package)
            scopes.setdefault(key, set()).add(artifact_id)

    components = inventory.get("components", [])
    rust_positions = [
        index for index, component in enumerate(components)
        if component.get("ecosystem") == "rust"
    ]
    insert_at = rust_positions[0] if rust_positions else len(components)
    existing = {
        (canonical_name(component["name"]), component["version"]): component
        for component in components if component.get("ecosystem") == "rust"
    }

    texts = dict(inventory.get("texts", {}))
    metadata_packages: dict[tuple[str, str, str], dict[str, Any]] | None = None
    rust_components = []
    for key in sorted(shipped):
        artifact_ids = sorted(scopes[key])
        component = existing.get(key)
        if component is not None:
            component = dict(component)
            component["artifact_ids"] = artifact_ids
            rust_components.append(component)
            continue
        package = shipped[key]
        if package.get("source_kind") != "registry-or-git":
            print(
                f"skipping rust:{key[0]}@{key[1]}: "
                f"{package.get('source_kind')} source", file=sys.stderr)
            continue
        if metadata_packages is None:
            metadata_packages = {
                (item["name"], item["version"], item["source"] or ""): item
                for item in metadata()["packages"]
            }
        metadata_package = metadata_packages.get(
            (package["name"], package["version"], package["source"]))
        if metadata_package is None:
            raise RefreshError(
                f"Cargo metadata lacks {package['name']} {package['version']} "
                f"from {package['source']}")
        try:
            component = derive_component(package, metadata_package,
                                         artifact_ids, allowed, inventory,
                                         texts)
        except NoticeError as error:
            raise RefreshError(
                f"{package['name']} {package['version']}: {error}") from error
        rust_components.append(component)
    rust_components.sort(key=lambda item: item["id"])

    others = [
        component for component in components
        if component.get("ecosystem") != "rust"
    ]
    before = [
        component for index, component in enumerate(components)
        if component.get("ecosystem") != "rust" and index < insert_at
    ]
    refreshed_components = (before + rust_components + others[len(before):])

    artifacts = []
    for artifact in inventory.get("artifacts", []):
        artifact = dict(artifact)
        if artifact.get("id") in DISKGRAPH_ARTIFACTS:
            resolved = resolved_artifacts[artifact["id"]]
            for field in RESOLVED_ARTIFACT_FIELDS:
                artifact[field] = resolved.get(field)
            retained = [
                component_id for component_id in artifact["components"]
                if not component_id.startswith("rust:")
            ]
            artifact["components"] = sorted(retained + [
                component["id"] for component in rust_components
                if artifact["id"] in component["artifact_ids"]
            ])
        artifacts.append(artifact)

    referenced = {
        item["sha256"]
        for component in refreshed_components
        for item in component.get("files", [])
    }
    refreshed = dict(inventory)
    refreshed["artifacts"] = artifacts
    refreshed["components"] = refreshed_components
    refreshed["texts"] = {
        digest: text
        for digest, text in texts.items() if digest in referenced
    }
    return refreshed


def serialize(inventory: dict[str, Any]) -> bytes:
    return (json.dumps(inventory, indent=2, ensure_ascii=False) +
            "\n").encode("utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--diskgraph-root", type=Path, default=DISKGRAPH_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inventory = load_json(args.inventory)
        policy = load_json(args.policy)
        manifest = load_json(args.diskgraph_root / "third_party" /
                             "artifact-manifest.json")
        refreshed = refresh(inventory, manifest, policy,
                            lambda: cargo_metadata(args.diskgraph_root))
    except (NoticeError, OSError, subprocess.CalledProcessError) as error:
        print(f"inventory refresh error: {error}", file=sys.stderr)
        return 2
    generated = serialize(refreshed)
    if args.inventory.read_bytes() == generated:
        print(f"{args.inventory}: current")
        return 0
    args.inventory.write_bytes(generated)
    before = {item["id"] for item in inventory["components"]}
    after = {item["id"] for item in refreshed["components"]}
    print(f"wrote {args.inventory}: added {len(after - before)}, "
          f"removed {len(before - after)} Rust components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
