# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained repro files for property failures, and the regression corpus.

A repro file holds the case (rows, contract, anchors, seed), the property, the
engine, the case's position in its batch and the versions that produced it. The
corpus is a directory of repro files that each name an issue; every entry runs
under the same required-execution rules as golden cases (design §4.3–4.4).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TypedDict

from conformance.cases import REPOSITORY
from conformance.generate import CasePayload, GeneratedCase
from conformance.registry import REGISTRY_VERSION

FORMAT = "conformance-repro"
FORMAT_VERSION = 1
CORPUS = REPOSITORY / "conformance/corpus"
REPRO_DIR = Path(os.environ.get("CONFORMANCE_REPROS", REPOSITORY / "conformance-repros"))


class Versions(TypedDict):
    package: str
    diskgraph: str | None
    registry: int
    python: str


class ReproPayload(TypedDict, total=False):
    format: str
    format_version: int
    property: str
    engine: str
    seed: int
    position: int
    case: CasePayload
    failure: str
    versions: Versions
    issue: str


@dataclass(frozen=True)
class Repro:
    id: str
    property: str
    engine: str
    position: int
    case: GeneratedCase
    failure: str
    versions: Versions
    issue: str | None = None


def versions() -> Versions:
    def installed(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "package": installed("kumo-pytorch") or "unknown",
        "diskgraph": installed("diskgraph"),
        "registry": REGISTRY_VERSION,
        "python": platform.python_version(),
    }


def repro_payload(
    property_name: str, engine: str, case: GeneratedCase, position: int, failure: BaseException
) -> ReproPayload:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "property": property_name,
        "engine": engine,
        "seed": case.seed,
        "position": position,
        "case": case.to_dict(),
        "failure": f"{type(failure).__name__}: {failure}",
        "versions": versions(),
    }


def repro_id(payload: ReproPayload) -> str:
    """Stable across runs: what reproduces the failure, not how it was reported."""

    identity = {key: payload[key] for key in ("property", "engine", "position", "case")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"repro-{digest[:16]}"


def write_repro(
    property_name: str,
    engine: str,
    case: GeneratedCase,
    position: int,
    failure: BaseException,
    directory: Path | None = None,
) -> Path:
    payload = repro_payload(property_name, engine, case, position, failure)
    target = directory or REPRO_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{repro_id(payload)}.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def load_repro(path: Path, *, require_issue: bool = False) -> Repro:
    payload: ReproPayload = json.loads(path.read_text())
    required = {"format", "format_version", "property", "engine", "seed", "position", "case"}
    required |= {"failure", "versions"} | ({"issue"} if require_issue else set())
    missing = sorted(required - payload.keys())
    if missing or set(payload) - required - {"issue"}:
        raise ValueError(f"{path}: repro fields must be {sorted(required)}; missing {missing}")
    if payload["format"] != FORMAT or payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"{path}: not a v{FORMAT_VERSION} {FORMAT} file")
    if require_issue and not str(payload["issue"]).strip():
        raise ValueError(f"{path}: a corpus entry links its issue")
    case = GeneratedCase.from_dict(payload["case"])
    if case.seed != payload["seed"]:
        raise ValueError(f"{path}: seed does not match the case")
    if repro_id(payload) != path.stem:
        raise ValueError(f"{path}: file name must be {repro_id(payload)}.json")
    return Repro(
        path.stem,
        payload["property"],
        payload["engine"],
        payload["position"],
        case,
        payload["failure"],
        payload["versions"],
        payload.get("issue"),
    )


def load_corpus(directory: Path = CORPUS) -> dict[str, Repro]:
    """Every corpus entry, validated; JSON outside the expected shape fails."""

    if not directory.is_dir():
        raise FileNotFoundError(f"regression corpus directory is missing: {directory}")
    return {
        path.stem: load_repro(path, require_issue=True) for path in sorted(directory.glob("*.json"))
    }
