#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject links to company-only documentation and operational services."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

URL_PATTERN = re.compile(r"(?:https?|git\+https?|git\+ssh)://[^\s<>\"']+")
DOMAIN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.-])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
    r"(?![A-Za-z0-9.-])")
GITHUB_SSH_PATTERN = re.compile(
    r"git@github\.com\.?:kumo-ai/(?P<repository>[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
TRAILING_URL_PUNCTUATION = ").,;:]}"

PRIVATE_HOSTS = {
    "www.notion.so": "company documentation",
    "notion.so": "company documentation",
    "kumoai.grafana.net": "company monitoring",
    "logs-prod-us-central2.grafana.net": "company monitoring",
    "kumoai.cd.akuity.cloud": "company deployment console",
    "kumo.corp": "company network",
    "staging.kumoai.cloud": "company staging service",
    "uat.staging.kumoai.cloud": "company staging service",
    "staging-rfm.kumorfm.ai": "company staging service",
    "rfm2-uat.staging-rfm.kumorfm.ai": "company staging service",
    "dbc-117b701d-939a.cloud.databricks.com": "company Databricks workspace",
    "app.slack.com": "company chat workspace",
}

PRIVATE_HOST_SUFFIXES = {
    ".kumo.corp": "company network",
    ".staging.kumoai.cloud": "company staging service",
    ".staging-rfm.kumorfm.ai": "company staging service",
}

PRIVATE_TEXT_PATTERNS = (
    (re.compile(r"(?<![\w-])@global-oncall\b"), "company chat mention"),
    (re.compile(r"(?<![\w-])#ask-infra(?:-devops)?\b"), "company chat channel"),
    (re.compile(r"\borg-[0-9]+@github\.com\b"), "company GitHub login alias"),
)

PRIVATE_GITHUB_REPOSITORIES = {
    "kumo-api",
    "kumo-diskgraph",
    "kumo-docker",
    "kumo-grafana",
    "kumo-iac",
    "kumo-ml",
    "kumo-online",
    "kumo-pql",
    "kumo-rfm",
    "kumo-sdk",
    "kumo-snow-integration",
    "kumo-tests",
    "kumo-tools",
    "kumo-ui-catalog",
    "rdl-planning",
}

# Company infrastructure identifiers are matched by their format so that new
# accounts, buckets, and paths are caught without listing each one. Each rule
# names the captured value and the documented placeholders it ignores.
SNOWFLAKE_PLACEHOLDER_ACCOUNTS = frozenset(
    {"acct", "account", "myaccount", "xy12345"})

IDENTIFIER_RULES = (
    (
        re.compile(r"\barn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:"
                   r"(?P<value>[0-9]{12}):"),
        "cloud account ID",
        # AWS documentation examples and repeated-digit test values.
        frozenset({"123456789012", "111122223333"}
                  | {digit * 12
                     for digit in "0123456789"}),
    ),
    (
        re.compile(r"\b(?P<value>[0-9]{12})\.dkr\.ecr\."),
        "cloud account ID",
        frozenset({"123456789012", "111122223333"}
                  | {digit * 12
                     for digit in "0123456789"}),
    ),
    (
        re.compile(
            r"(?<![\w.-])(?P<value>[a-z0-9_-]+)(?:\.[a-z0-9-]+)*"
            r"\.snowflakecomputing\.com\b", re.IGNORECASE),
        "warehouse account",
        SNOWFLAKE_PLACEHOLDER_ACCOUNTS,
    ),
    (
        # Console links may include a region before the account locator.
        re.compile(
            r"\bapp\.snowflake\.com/(?:[a-z]{2}-[a-z]+-[0-9](?:\.[a-z]+)?/)?"
            r"(?P<value>[a-z0-9_-]+)/", re.IGNORECASE),
        "warehouse account",
        SNOWFLAKE_PLACEHOLDER_ACCOUNTS,
    ),
    (
        # Legacy account locators are two or three letters and five digits,
        # assigned to an account setting (quoted, unquoted, or typed).
        re.compile(
            r"\baccount\w*[\"']?\s*(?::\s*\w+\s*)?[:=(]\s*[\"']?"
            r"(?P<value>[a-z]{2,3}[0-9]{5})(?![a-z0-9])", re.IGNORECASE),
        "warehouse account",
        SNOWFLAKE_PLACEHOLDER_ACCOUNTS,
    ),
    (
        # A "kumo-" GCP project assigned to a project setting, including
        # prefixed keys such as billing_project_id or GOOGLE_CLOUD_PROJECT,
        # typed defaults, and unquoted values.
        re.compile(
            r"project(?:[_-]?id)?[\"']?\s*(?:[:=,(]|:\s*\w+\s*=)\s*[\"']?"
            r"(?P<value>kumo-[a-z0-9-]*[a-z0-9])(?![\w-])", re.IGNORECASE),
        "company cloud project",
        frozenset(),
    ),
    (
        re.compile(r"\bprojects/(?P<value>kumo-[a-z0-9-]*[a-z0-9])/",
                   re.IGNORECASE),
        "company cloud project",
        frozenset(),
    ),
    (
        re.compile(
            r"@(?P<value>kumo-[a-z0-9-]*[a-z0-9])\.iam\.gserviceaccount\.com",
            re.IGNORECASE),
        "company cloud project",
        frozenset(),
    ),
    (
        # Templated names such as gs://kumo-${DEPLOYMENT}/ are not buckets.
        re.compile(
            r"\b(?:s3a?|gs)://(?P<value>kumo(?:[a-z0-9.-]*[a-z0-9])?)"
            r"(?![a-z0-9.-]*[{$<%])", re.IGNORECASE),
        "company storage bucket",
        frozenset({"kumo-public-datasets", "kumo-public-datasets-bigquery"}),
    ),
    (
        re.compile(
            r"(?<![\w.~$-])(?:/home|/Users)/(?P<value>[A-Za-z][\w.-]*)/"),
        "personal home directory",
        # Service, runner, and container accounts plus test fixture names.
        frozenset({
            "bp", "ec2-user", "jovyan", "kumouser", "pred_table", "root",
            "runner", "ubuntu", "user"
        }),
    ),
)

# Identifier rules do not apply to internal build and deployment
# infrastructure, to tests (which may use company test resources), or to
# product modules whose defaults intentionally point at company resources.
# Host, repository, and chat rules still apply to these paths.
IDENTIFIER_RULE_EXEMPT_PREFIXES = (
    ".github/",
    "Makefile",
    "deploy/",
    "docker/",
    "kumo/testing/rest.py",
    "kumo/testing/snowflake.py",
    "packages/kumo-api/tools/schema_compat/config.py",
    "packages/kumo-bigquery/tests/live/test_live_conformance.py",
    "packages/kumo-databricks/tests/live/test_live_conformance.py",
    "packages/kumo-snowflake/tests/live/test_live_conformance.py",
    # Sample text in these stories appears in committed screenshot baselines,
    # which must be regenerated in the CI Playwright image before it changes.
    "packages/kumo-ui-catalog/src/stories/components/ui/Alert.stories.tsx",
    "packages/kumo-ui-catalog/src/stories/components/ui/CopyToClipboard.stories.tsx",
    "packages/kumo-ui-catalog/src/stories/components/ui/OutputCard.stories.tsx",
    "rest/devserver/",
    "scripts/",
    "service_framework/deploy_env.py",
    "test/",
    "tools/",
)

PRIVATE_GITHUB_ALLOWLIST = {
    # This test verifies that published SDK metadata does not include the URL.
    ("packages/kumo-sdk/test/test_release_artifact.py", "kumo-api"),
}

# These dependencies must be published, vendored, or replaced before release.
# The normal CI check permits only these existing path/repository counts, while
# --strict reports every occurrence so the remaining work stays visible.
KNOWN_PRIVATE_GITHUB_DEPENDENCIES = {
    (".github/workflows/sync-diskgraph.yml", "kumo-diskgraph"): 1,
}

SKIPPED_PATHS = {
    "THIRD-PARTY.txt",
    "scripts/check_public_references.py",
    "test/scripts/test_check_public_references.py",
}
SKIPPED_SUFFIXES = (
    ".lock",
    ".min.js",
    ".map",
)
# DiskGraph is mirrored from its own repository, which is the source of truth
# for these files. Fixes are made there and arrive through synchronization.
SKIPPED_PREFIXES = ("packages/diskgraph/", )


def is_skipped(relative: str) -> bool:
    return (relative in SKIPPED_PATHS or relative.endswith(SKIPPED_SUFFIXES)
            or relative.startswith(SKIPPED_PREFIXES))


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    host: str
    category: str
    repository: str = ""


def classify_host(host: str) -> str | None:
    normalized = host.lower().rstrip(".")
    if normalized in PRIVATE_HOSTS:
        return PRIVATE_HOSTS[normalized]
    for suffix, category in PRIVATE_HOST_SUFFIXES.items():
        if normalized.endswith(suffix):
            return category
    return None


def github_repository(url: str) -> str | None:
    parsed = urlsplit(url)
    parts = [part for part in parsed.path.split("/") if part]
    host = (parsed.hostname or "").lower().rstrip(".")
    if host == "github.com" and len(parts) >= 2:
        owner, repository = parts[:2]
    elif (host == "colab.research.google.com" and len(parts) >= 3
          and parts[0] == "github"):
        owner, repository = parts[1:3]
    else:
        return None
    if owner.lower() != "kumo-ai":
        return None
    return repository.split("@", maxsplit=1)[0].lower().removesuffix(".git")


def scan_text(path: str, text: str, *, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    identifier_rules = (() if path.startswith(IDENTIFIER_RULE_EXEMPT_PREFIXES)
                        else IDENTIFIER_RULES)
    for line_number, line in enumerate(text.splitlines(), start=1):
        reported_hosts: set[str] = set()
        for match in URL_PATTERN.finditer(line):
            url = match.group(0).rstrip(TRAILING_URL_PUNCTUATION)
            try:
                host = urlsplit(url).hostname
            except ValueError:
                continue
            if host is None:
                continue
            category = classify_host(host)
            repository = None
            if category is None and strict:
                repository = github_repository(url)
                if (repository in PRIVATE_GITHUB_REPOSITORIES
                        and (path, repository) not in PRIVATE_GITHUB_ALLOWLIST):
                    category = "private Kumo GitHub repository"
            if category is not None:
                reported_hosts.add(host.lower())
                findings.append(
                    Finding(
                        path=path,
                        line=line_number,
                        host=host.lower(),
                        category=category,
                        repository=repository or "",
                    ))
        for match in DOMAIN_PATTERN.finditer(line):
            host = match.group(0).lower()
            category = classify_host(host)
            if category is not None and host not in reported_hosts:
                findings.append(
                    Finding(
                        path=path,
                        line=line_number,
                        host=host,
                        category=category,
                    ))
        for pattern, category, ignored in identifier_rules:
            for match in pattern.finditer(line):
                value = match.group("value").lower()
                if value not in ignored:
                    findings.append(
                        Finding(
                            path=path,
                            line=line_number,
                            host=value,
                            category=category,
                        ))
        for pattern, category in PRIVATE_TEXT_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        path=path,
                        line=line_number,
                        host="internal collaboration reference",
                        category=category,
                    ))
        if strict:
            for match in GITHUB_SSH_PATTERN.finditer(line):
                repository = match.group("repository").lower().removesuffix(
                    ".git")
                if (repository in PRIVATE_GITHUB_REPOSITORIES
                        and (path, repository) not in PRIVATE_GITHUB_ALLOWLIST):
                    findings.append(
                        Finding(
                            path=path,
                            line=line_number,
                            host="github.com",
                            category="private Kumo GitHub repository",
                            repository=repository,
                        ))
    return findings


def repository_paths(root: Path) -> Iterable[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if is_skipped(relative):
            continue
        yield root / relative


def _resolve_commit(root: Path, reference: str, label: str) -> str:
    if not reference:
        raise ValueError("Public-reference diff references must not be empty.")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{reference}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Public-reference diff {label} is not an available commit: "
            f"{reference}")
    return result.stdout.strip()


def _commit_range_paths(
    root: Path,
    base: str,
    head: str,
) -> tuple[str, list[str]]:
    if not base or not head:
        raise ValueError("Public-reference diff references must not be empty.")
    base_commit = _resolve_commit(root, base, "base")
    head_commit = _resolve_commit(root, head, "head")
    merge_base = subprocess.run(
        ["git", "-C",
         str(root), "merge-base", base_commit, head_commit],
        check=False,
        capture_output=True,
        text=True,
    )
    if merge_base.returncode != 0:
        raise ValueError(
            "Public-reference diff commits do not share Git history.")

    changed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            # Rename detection compares file contents, which a blob-less CI
            # clone cannot fetch; a renamed file is scanned as an addition.
            "--no-renames",
            "-z",
            merge_base.stdout.strip(),
            head_commit,
            "--",
        ],
        check=True,
        capture_output=True,
    )
    paths = [
        raw_path.decode("utf-8", errors="surrogateescape")
        for raw_path in changed.stdout.split(b"\0") if raw_path
    ]
    return head_commit, paths


def _commit_blob(root: Path, commit: str, path: str) -> bytes | None:
    tree_entry = subprocess.run(
        ["git", "-C",
         str(root), "ls-tree", "-z", commit, "--", path],
        check=True,
        capture_output=True,
    ).stdout
    if not tree_entry:
        return None

    metadata, _ = tree_entry.rstrip(b"\0").split(b"\t", maxsplit=1)
    mode, object_type, object_id = metadata.split()
    if mode == b"120000" or object_type != b"blob":
        return None
    return subprocess.run(
        ["git", "-C",
         str(root), "cat-file", "blob",
         object_id.decode()],
        check=True,
        capture_output=True,
    ).stdout


def _commit_range_sources(
    root: Path,
    base: str,
    head: str,
) -> Iterable[tuple[str, bytes]]:
    head_commit, paths = _commit_range_paths(root, base, head)
    for relative in paths:
        if is_skipped(relative):
            continue
        content = _commit_blob(root, head_commit, relative)
        if content is not None:
            yield relative, content


def _repository_sources(root: Path) -> Iterable[tuple[str, bytes]]:
    for path in repository_paths(root):
        if path.is_symlink() or not path.is_file():
            continue
        yield str(path.relative_to(root)), path.read_bytes()


def scan_repository(
    root: Path,
    *,
    strict: bool = False,
    diff: tuple[str, str] | None = None,
    validate_baseline: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    sources = (_repository_sources(root)
               if diff is None else _commit_range_sources(root, *diff))
    for relative_path, content in sources:
        if b"\0" in content:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Private repository detection always runs. The normal mode suppresses
        # only the explicitly recorded dependency counts below.
        findings.extend(scan_text(relative_path, text, strict=True))

    ordered = sorted(findings)
    if strict:
        return ordered

    remaining_dependencies = KNOWN_PRIVATE_GITHUB_DEPENDENCIES.copy()
    result: list[Finding] = []
    for finding in ordered:
        dependency_key = (finding.path, finding.repository)
        if (finding.repository
                and remaining_dependencies.get(dependency_key, 0) > 0):
            remaining_dependencies[dependency_key] -= 1
        else:
            result.append(finding)
    if validate_baseline:
        stale = {
            key
            for key, count in remaining_dependencies.items() if count > 0
        }
        if stale:
            raise ValueError(
                f"Stale private-dependency baseline entries: {sorted(stale)}. "
                "Update the known-private-reference entries.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to scan",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("BASE", "COMMIT"),
        help="scan files added or modified between two commits",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also report links to Kumo repositories known to be private",
    )
    args = parser.parse_args()

    try:
        findings = scan_repository(
            args.root.resolve(),
            strict=args.strict,
            diff=tuple(args.diff) if args.diff is not None else None,
            validate_baseline=args.diff is None and not args.strict,
        )
    except ValueError as error:
        parser.error(str(error))
    if not findings:
        print("Public-reference check passed.")
        return 0

    print("Links to company-only resources were found:")
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.category} "
              f"({finding.host})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
