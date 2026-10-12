# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate license headers on tracked, comment-capable repository files."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

COPYRIGHT_PREFIX = "SPDX-FileCopyrightText: Copyright (c) "
SPDX_COPYRIGHT_PREFIX = "SPDX-FileCopyrightText: "
LICENSE_PREFIX = "SPDX-License-Identifier: "
PYTHON_ENCODING_RE = re.compile(r"^#.*coding[=:]\s*[-\w.]+")


def is_shebang(line: str) -> bool:
    """Distinguish executable shebangs from Rust inner attributes."""
    return line.startswith("#!") and not line.startswith("#![")


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("version") != 1:
        raise ValueError(f"unsupported policy version in {path}")
    return policy


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def matching_exclusions(path: str, policy: dict[str,
                                                Any]) -> list[dict[str, Any]]:
    return [
        exclusion for exclusion in policy["exclusions"]
        if fnmatch.fnmatchcase(path, exclusion["pattern"])
    ]


def matching_exclusion(path: str, policy: dict[str,
                                               Any]) -> dict[str, Any] | None:
    exclusions = matching_exclusions(path, policy)
    return exclusions[0] if exclusions else None


def matching_derived_sources(path: str,
                             policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        source for source in policy.get("derived_sources", [])
        if path in source["paths"]
    ]


def matching_license_scopes(path: str,
                            policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        scope for scope in policy.get("license_scopes", [])
        if fnmatch.fnmatchcase(path, scope["pattern"])
    ]


def styles_for(path: str, text: str, policy: dict[str, Any]) -> tuple[str, ...]:
    candidate_types = policy["candidate_types"]
    pure_path = PurePosixPath(path)
    styles: list[str] = []
    styles.extend(candidate_types["path_styles"].get(path, []))
    styles.extend(candidate_types["basenames"].get(pure_path.name, []))
    for prefix, prefix_styles in candidate_types["basename_prefixes"].items():
        if pure_path.name.startswith(prefix):
            styles.extend(prefix_styles)
    for suffix, suffix_styles in candidate_types.get("basename_suffixes",
                                                     {}).items():
        if pure_path.name.endswith(suffix):
            styles.extend(suffix_styles)
    styles.extend(candidate_types["extensions"].get(pure_path.suffix, []))
    if (candidate_types["include_shebang_files"] and not styles
            and text.splitlines() and is_shebang(text.splitlines()[0])):
        styles.append("hash")
    return tuple(dict.fromkeys(styles))


def header_start(path: str, lines: Sequence[str],
                 policy: dict[str, Any]) -> tuple[int, str | None]:
    index = 0
    if lines and is_shebang(lines[0]):
        index = 1
    pure_path = PurePosixPath(path)
    is_dockerfile = (pure_path.name == "Dockerfile"
                     or pure_path.name.startswith("Dockerfile.")
                     or pure_path.name.endswith(".Dockerfile"))
    if is_dockerfile and lines:
        syntax_pattern = re.compile(r"#\s*syntax\s*=.*", re.IGNORECASE)
        if index == 0 and syntax_pattern.fullmatch(lines[0]):
            index = 1
        elif any(
                syntax_pattern.fullmatch(line)
                for line in lines[1:policy["max_header_lines"]]):
            return index, "Docker syntax parser directive must remain on line 1"
    if index < len(lines) and re.fullmatch(r'''["']use client["'];?''',
                                           lines[index].strip()):
        index += 1
    if index < len(lines) and PYTHON_ENCODING_RE.match(lines[index]):
        index += 1
    for exception in policy["preamble_exceptions"]:
        if path != exception["path"]:
            continue
        if index != 0 or not lines or re.fullmatch(exception["line_pattern"],
                                                   lines[0]) is None:
            return index, "does not match its required line-1 preamble"
        index = 1
        break
    return index, None


def line_comment_body(lines: Sequence[str], start: int, marker: str,
                      limit: int) -> tuple[list[str], int] | None:
    pattern = re.compile(rf"^\s*{re.escape(marker)}\s?(.*)$")
    body: list[str] = []
    end = start
    for end, line in enumerate(lines[start:limit], start=start + 1):
        match = pattern.match(line)
        if match is None:
            end -= 1
            break
        body.append(match.group(1).strip())
    return (body, end) if body else None


def delimited_comment_body(lines: Sequence[str], start: int, limit: int,
                           style: str) -> tuple[list[str], int] | None:
    delimiters = {
        "block": (re.compile(r"^\s*/\*+\s?"), re.compile(r"\*/\s*$")),
        "html": (re.compile(r"^\s*<!--\s?"), None),
        "handlebars":
        (re.compile(r"^\s*\{\{!--\s?"), re.compile(r"--\}\}\s*$")),
        "helm":
        (re.compile(r"^\s*\{\{-?\s*/\*+\s?"), re.compile(r"\*/\s*-?\}\}\s*$")),
    }
    opener, closer = delimiters[style]
    selected = list(lines[start:limit])
    if not selected or opener.match(selected[0]) is None:
        return None
    selected[0] = opener.sub("", selected[0], count=1)
    closed = False
    body: list[str] = []
    end = start
    for end, line in enumerate(selected, start=start + 1):
        if style == "html" and line.rstrip().endswith("-->"):
            line = line.rstrip()[:-3]
            closed = True
        elif closer is not None and closer.search(line):
            line = closer.sub("", line, count=1)
            closed = True
        line = re.sub(r"^\s*\*\s?", "", line)
        body.append(line.strip())
        if closed:
            break
    return (body, end) if closed else None


def comment_body(lines: Sequence[str], start: int, limit: int,
                 style: str) -> tuple[list[str], int] | None:
    if style == "hash":
        return line_comment_body(lines, start, "#", limit)
    if style == "slash":
        return line_comment_body(lines, start, "//", limit)
    if style == "dash":
        return line_comment_body(lines, start, "--", limit)
    return delimited_comment_body(lines, start, limit, style)


def leading_comment_bodies(lines: Sequence[str], start: int,
                           styles: Sequence[str]) -> Iterable[list[str]]:
    """Yield comment bodies before the first source line."""
    index = start
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        for style in styles:
            parsed = comment_body(lines, index, len(lines), style)
            if parsed is not None:
                body, index = parsed
                yield body
                break
        else:
            return


def header_signature(body: Sequence[str]) -> tuple[str, str, str] | None:
    logical_lines = [line for line in body if line]
    if len(logical_lines) < 3:
        return None
    return tuple(logical_lines[:3])


def signature_count(bodies: Iterable[Sequence[str]],
                    signature: tuple[str, str, str]) -> int:
    count = 0
    for body in bodies:
        logical_lines = [line for line in body if line]
        count += sum(
            tuple(logical_lines[index:index + 3]) == signature
            for index in range(len(logical_lines) - 2))
    return count


def spdx_lines(bodies: Iterable[Sequence[str]]) -> list[str]:
    return [
        line for body in bodies for line in body
        if line.startswith((SPDX_COPYRIGHT_PREFIX, LICENSE_PREFIX))
    ]


def validate_body(body: Sequence[str], policy: dict[str, Any],
                  derived_source: dict[str, Any] | None,
                  license_identifier: str) -> str | None:
    logical_lines = [line for line in body if line]
    if not logical_lines:
        return "header comment is empty"
    position = 0
    if derived_source is not None:
        for required_line in derived_source["required_copyright_lines"]:
            if (len(logical_lines) <= position
                    or logical_lines[position] != required_line):
                return f"derived-source header must begin with upstream attribution: {required_line}"
            position += 1
    copyright_re = re.compile(
        rf"^{re.escape(COPYRIGHT_PREFIX)}\d{{4}}(?:-\d{{4}})?,? "
        rf"{re.escape(policy['copyright_holder'])}$")
    if (len(logical_lines) <= position
            or copyright_re.fullmatch(logical_lines[position]) is None):
        return "header has a missing or malformed NVIDIA SPDX copyright line"
    position += 1
    if len(logical_lines) <= position or logical_lines[position] != policy[
            "rights_statement"]:
        return "header is missing the separate rights statement"
    position += 1
    expected_license = LICENSE_PREFIX + license_identifier
    if len(logical_lines
           ) <= position or logical_lines[position] != expected_license:
        return f"header must declare {expected_license}"
    return None


def validate_text(path: str, text: str, policy: dict[str, Any]) -> str | None:
    styles = styles_for(path, text, policy)
    if not styles:
        return None
    lines = text.splitlines()
    derived_sources = matching_derived_sources(path, policy)
    if len(derived_sources) > 1:
        names = ", ".join(source["name"] for source in derived_sources)
        return f"matches multiple derived-source classifications: {names}"
    derived_source = derived_sources[0] if derived_sources else None
    license_scopes = matching_license_scopes(path, policy)
    if len(license_scopes) > 1:
        patterns = ", ".join(scope["pattern"] for scope in license_scopes)
        return f"matches multiple license scopes: {patterns}"
    if derived_source is not None and license_scopes:
        return "matches both a derived-source classification and a license scope"
    license_identifier = (derived_source["license_identifier"]
                          if derived_source is not None else
                          license_scopes[0]["license_identifier"]
                          if license_scopes else policy["license_identifier"])
    start, preamble_error = header_start(path, lines, policy)
    if preamble_error is not None:
        return preamble_error
    limit = min(len(lines), policy["max_header_lines"])
    style_errors: list[str] = []
    for style in styles:
        parsed = comment_body(lines, start, limit, style)
        if parsed is None:
            continue
        body, _ = parsed
        error = validate_body(body, policy, derived_source, license_identifier)
        if error is None:
            expected_spdx = []
            if derived_source is not None:
                expected_spdx.extend(derived_source["required_copyright_lines"])
            expected_spdx.append(
                next(line for line in body if line.startswith(COPYRIGHT_PREFIX)
                     and policy["copyright_holder"] in line))
            expected_spdx.append(LICENSE_PREFIX + license_identifier)
            actual_spdx = spdx_lines(
                leading_comment_bodies(lines, start, styles))
            if actual_spdx != expected_spdx:
                return "contains SPDX declarations outside its classified repository header"
            signature = header_signature(body)
            if signature is not None and signature_count(
                    leading_comment_bodies(lines, start, styles),
                    signature) > 1:
                return "duplicate repository license header"
            return None
        style_errors.append(error)
    if style_errors:
        return style_errors[0]
    return f"missing a valid {license_identifier} header at the required file start"


def contains_repository_copyright(text: str, policy: dict[str, Any]) -> bool:
    return any(COPYRIGHT_PREFIX in line and policy["copyright_holder"] in line
               for line in text.splitlines())


def check_paths(
    root: Path,
    policy: dict[str, Any],
    paths: Iterable[str],
) -> tuple[list[str], int, list[tuple[str, dict[str, Any]]]]:
    errors: list[str] = []
    scanned = 0
    excluded: list[tuple[str, dict[str, Any]]] = []
    used_exclusions: set[int] = set()
    encountered_derived_paths: set[str] = set()
    used_license_scopes: set[int] = set()
    for relative_path in paths:
        absolute_path = root / relative_path
        if not absolute_path.is_file():
            continue
        try:
            text = absolute_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            candidate_styles = styles_for(relative_path, "", policy)
            if candidate_styles:
                error = "candidate file is not valid UTF-8"
                exclusions = matching_exclusions(relative_path, policy)
                used_exclusions.update(map(id, exclusions))
                if len(exclusions) > 1:
                    scanned += 1
                    patterns = ", ".join(exclusion["pattern"]
                                         for exclusion in exclusions)
                    errors.append(
                        f"{relative_path}: matches multiple license-header exclusions: {patterns}"
                    )
                elif exclusions:
                    excluded.append((relative_path, exclusions[0]))
                else:
                    scanned += 1
                    errors.append(f"{relative_path}: {error}")
            continue
        if not styles_for(relative_path, text, policy):
            continue
        license_scopes = matching_license_scopes(relative_path, policy)
        used_license_scopes.update(map(id, license_scopes))
        derived_sources = matching_derived_sources(relative_path, policy)
        if derived_sources:
            encountered_derived_paths.add(relative_path)
        error = validate_text(relative_path, text, policy)
        exclusions = matching_exclusions(relative_path, policy)
        if len(exclusions) > 1:
            used_exclusions.update(map(id, exclusions))
            scanned += 1
            patterns = ", ".join(exclusion["pattern"]
                                 for exclusion in exclusions)
            errors.append(
                f"{relative_path}: matches multiple license-header exclusions: {patterns}"
            )
            continue
        if exclusions:
            used_exclusions.add(id(exclusions[0]))
            if derived_sources:
                scanned += 1
                errors.append(
                    f"{relative_path}: matches both a derived-source classification and a license-header exclusion"
                )
                continue
            if license_scopes:
                scanned += 1
                errors.append(
                    f"{relative_path}: matches both a license scope and a license-header exclusion"
                )
                continue
            if error is not None and contains_repository_copyright(
                    text, policy):
                scanned += 1
                errors.append(
                    f"{relative_path}: excluded source contains an NVIDIA SPDX copyright line outside a valid repository header"
                )
                continue
            if error is None:
                scanned += 1
                errors.append(
                    f"{relative_path}: excluded source satisfies repository header policy; remove the repository header or narrow the exclusion"
                )
                continue
            excluded.append((relative_path, exclusions[0]))
            continue
        scanned += 1
        if error is not None:
            errors.append(f"{relative_path}: {error}")
    for exclusion in policy["exclusions"]:
        if id(exclusion) not in used_exclusions:
            errors.append(
                f"license-header exclusion is stale or unnecessary: {exclusion['pattern']}"
            )
    for source in policy.get("derived_sources", []):
        for path in source["paths"]:
            if path not in encountered_derived_paths:
                errors.append(
                    f"derived-source classification is stale or not a candidate: {path}"
                )
    for scope in policy.get("license_scopes", []):
        if id(scope) not in used_license_scopes:
            errors.append(
                f"license scope is stale or matches no candidate: {scope['pattern']}"
            )
    return errors, scanned, excluded


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    policy_path = args.policy or root / "scripts/license_header_policy.json"
    policy = load_policy(policy_path)
    errors, scanned, excluded = check_paths(root, policy, tracked_files(root))
    if args.verbose:
        for path, exclusion in excluded:
            print(
                f"excluded {path}: {exclusion['kind']} ({exclusion['license']})"
            )
    if errors:
        print("License header check failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(
        f"License header check passed for {scanned} tracked candidate files ({len(excluded)} exclusions)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
