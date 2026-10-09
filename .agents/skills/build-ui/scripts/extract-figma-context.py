#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FIGMA_URL_RE = re.compile(
    r"https://(?:www\.)?figma\.com/(?:file|design|proto|slides|board|make)/[^\s<>\]\}]+",
    re.IGNORECASE,
)
TRAILING_URL_PUNCTUATION = ".,;:)]}>"
FIGMA_PATH_TYPES = {"file", "design", "proto", "slides", "board", "make"}
REDACTED_FIGMA_KEYS = {"users", "lastModifiedBy"}
TRUSTED_COMMENT_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
FIGMA_NODE_DEPTH = 6
FIGMA_FILE_DEPTH = 2
FIGMA_IMAGE_SCALE = 2


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize_node_id(value):
    if not value:
        return None
    decoded = urllib.parse.unquote(value)
    return decoded.replace("-", ":")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "figma"


def sanitize_figma_url(url):
    return url.rstrip(TRAILING_URL_PUNCTUATION)


def parse_figma_url(url):
    url = sanitize_figma_url(url)
    parsed = urllib.parse.urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2 or path_parts[0].lower() not in FIGMA_PATH_TYPES:
        return None

    query = urllib.parse.parse_qs(parsed.query)
    node_id = normalize_node_id((query.get("node-id") or [None])[-1])
    payload = {
        "url": url,
        "fileKey": path_parts[1],
        "nodeId": node_id,
    }
    if len(path_parts) >= 4 and path_parts[2].lower() == "branch":
        payload["branchKey"] = path_parts[3]
    return payload


def extract_figma_links(text):
    if not text:
        return []

    links = []
    seen = set()
    for match in FIGMA_URL_RE.finditer(text):
        parsed = parse_figma_url(match.group(0))
        if not parsed:
            continue
        key = (parsed["fileKey"], parsed.get("nodeId"), parsed["url"])
        if key in seen:
            continue
        seen.add(key)
        links.append(parsed)
    return links


def load_issue_context_from_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("issue input file must contain a JSON object")
    return data


def request_url(request, timeout):
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504
                                  } or attempt == attempts:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 8)
            time.sleep(delay)

    raise RuntimeError("unreachable retry state")


def request_json(url, token=None, headers=None):
    request_headers = {
        "Accept": "application/json",
        **(headers or {}),
    }
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=request_headers)
    return json.loads(request_url(request, timeout=30).decode("utf-8"))


def request_bytes(url):
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    return request_url(request, timeout=60)


def load_issue_context_from_github(repo, issue_number, token):
    if not token:
        raise ValueError("GitHub token is required when --input-file is absent")

    api_base = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    issue = request_json(api_base, token=token)
    comments = []
    page = 1
    while True:
        url = f"{api_base}/comments?per_page=100&page={page}"
        batch = request_json(url, token=token)
        if not batch:
            break
        comments.extend(comment for comment in batch if comment.get(
            "author_association") in TRUSTED_COMMENT_ASSOCIATIONS)
        page += 1

    return {
        "issue": {
            "title": issue.get("title") or "",
            "body": issue.get("body") or "",
            "html_url": issue.get("html_url") or "",
        },
        "comments": [{
            "body": comment.get("body") or "",
            "html_url": comment.get("html_url") or "",
        } for comment in comments],
    }


def context_text(issue_context):
    parts = []
    issue = issue_context.get("issue") or {}
    parts.extend([issue.get("title") or "", issue.get("body") or ""])
    for comment in issue_context.get("comments") or []:
        if isinstance(comment, dict):
            parts.append(comment.get("body") or "")
        elif isinstance(comment, str):
            parts.append(comment)
    return "\n\n".join(parts)


def filter_allowed_figma_links(links, allowlist_value):
    allowed = {
        item.strip()
        for item in (allowlist_value or "").split(",") if item.strip()
    }
    if not allowed:
        return links
    return [
        link for link in links
        if link.get("fileKey") in allowed or link.get("branchKey") in allowed
    ]


def figma_api_json(path, figma_token):
    url = f"https://api.figma.com/v1/{path.lstrip('/')}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Figma-Token": figma_token,
        },
    )
    return json.loads(request_url(request, timeout=60).decode("utf-8"))


def redact_figma_metadata(value):
    if isinstance(value, dict):
        return {
            key: redact_figma_metadata(child)
            for key, child in value.items() if key not in REDACTED_FIGMA_KEYS
        }
    if isinstance(value, list):
        return [redact_figma_metadata(item) for item in value]
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(redact_figma_metadata(payload), indent=2, sort_keys=True) +
        "\n", encoding="utf-8")


def figma_api_file_key(link):
    return link.get("branchKey") or link["fileKey"]


def fetch_figma_artifacts(links, output_path, figma_token):
    base_dir = output_path.parent
    nodes_dir = base_dir / "figma-nodes"
    images_dir = base_dir / "figma-images"
    fetched_nodes = []
    fetched_files = []
    fetched_images = []
    errors = []

    by_file = {}
    for link in links:
        by_file.setdefault(figma_api_file_key(link), []).append(link)

    for api_file_key, file_links in sorted(by_file.items()):
        file_keys = sorted({link["fileKey"] for link in file_links})
        branch_keys = sorted(
            {link["branchKey"]
             for link in file_links if link.get("branchKey")})
        file_key = file_keys[0]
        branch_key = branch_keys[0] if branch_keys else None
        node_ids = sorted(
            {link["nodeId"]
             for link in file_links if link.get("nodeId")})
        try:
            if node_ids:
                ids = ",".join(node_ids)
                nodes_payload = figma_api_json(
                    f"files/{api_file_key}/nodes?ids={urllib.parse.quote(ids)}&depth={FIGMA_NODE_DEPTH}",
                    figma_token,
                )
                nodes_path = nodes_dir / f"{safe_name(api_file_key)}.nodes.json"
                write_json(nodes_path, nodes_payload)
                returned_nodes = nodes_payload.get("nodes") or {}
                valid_node_ids = []
                for node_id in node_ids:
                    node = returned_nodes.get(node_id)
                    document = (node or {}).get("document") or {}
                    if not document:
                        errors.append({
                            "fileKey":
                            file_key,
                            "branchKey":
                            branch_key,
                            "apiFileKey":
                            api_file_key,
                            "nodeId":
                            node_id,
                            "stage":
                            "node-fetch",
                            "error":
                            "Figma did not return document metadata for requested node",
                        })
                        continue
                    valid_node_ids.append(node_id)
                    fetched_nodes.append({
                        "fileKey": file_key,
                        "branchKey": branch_key,
                        "apiFileKey": api_file_key,
                        "nodeId": node_id,
                        "name": document.get("name"),
                        "type": document.get("type"),
                        "metadataPath": str(nodes_path),
                    })

                if not valid_node_ids:
                    continue

                try:
                    valid_ids = ",".join(valid_node_ids)
                    images_payload = figma_api_json(
                        f"images/{api_file_key}?ids={urllib.parse.quote(valid_ids)}&format=png&scale={FIGMA_IMAGE_SCALE}",
                        figma_token,
                    )
                    returned_images = images_payload.get("images") or {}
                    for node_id in valid_node_ids:
                        image_url = returned_images.get(node_id)
                        if not image_url:
                            errors.append({
                                "fileKey":
                                file_key,
                                "branchKey":
                                branch_key,
                                "apiFileKey":
                                api_file_key,
                                "nodeId":
                                node_id,
                                "stage":
                                "image-render",
                                "error":
                                "Figma did not return a rendered image URL for requested node",
                            })
                            continue
                        try:
                            image_path = images_dir / f"{safe_name(file_key)}__{safe_name(node_id)}.png"
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            image_path.write_bytes(request_bytes(image_url))
                            fetched_images.append({
                                "fileKey": file_key,
                                "branchKey": branch_key,
                                "apiFileKey": api_file_key,
                                "nodeId": node_id,
                                "imagePath": str(image_path),
                            })
                        except Exception as error:
                            errors.append({
                                "fileKey": file_key,
                                "branchKey": branch_key,
                                "apiFileKey": api_file_key,
                                "nodeId": node_id,
                                "stage": "image-download",
                                "error": str(error),
                            })
                except urllib.error.HTTPError as error:
                    errors.append({
                        "fileKey": file_key,
                        "branchKey": branch_key,
                        "apiFileKey": api_file_key,
                        "stage": "image-render",
                        "status": error.code,
                        "reason": error.reason,
                    })
                except Exception as error:
                    errors.append({
                        "fileKey": file_key,
                        "branchKey": branch_key,
                        "apiFileKey": api_file_key,
                        "stage": "image-render",
                        "error": str(error),
                    })
            else:
                file_payload = figma_api_json(
                    f"files/{api_file_key}?depth={FIGMA_FILE_DEPTH}",
                    figma_token,
                )
                file_path = nodes_dir / f"{safe_name(api_file_key)}.file.json"
                write_json(file_path, file_payload)
                fetched_files.append({
                    "fileKey":
                    file_key,
                    "branchKey":
                    branch_key,
                    "apiFileKey":
                    api_file_key,
                    "name":
                    file_payload.get("name"),
                    "lastModified":
                    file_payload.get("lastModified"),
                    "metadataPath":
                    str(file_path),
                })
        except urllib.error.HTTPError as error:
            errors.append({
                "fileKey": file_key,
                "branchKey": branch_key,
                "apiFileKey": api_file_key,
                "status": error.code,
                "reason": error.reason,
            })
        except Exception as error:
            errors.append({
                "fileKey": file_key,
                "branchKey": branch_key,
                "apiFileKey": api_file_key,
                "error": str(error),
            })

    return fetched_files, fetched_nodes, fetched_images, errors


def build_summary(context):
    lines = [
        "# Figma Context",
        "",
        f"Status: {context['status']}",
        f"Reason: {context.get('reason') or 'none'}",
        f"Links found: {len(context.get('links') or [])}",
        f"Nodes fetched: {len(context.get('nodes') or [])}",
        f"Images fetched: {len(context.get('images') or [])}",
        "",
    ]

    if context.get("links"):
        lines.extend(["## Links", ""])
        for link in context["links"]:
            node = f" node `{link['nodeId']}`" if link.get("nodeId") else ""
            lines.append(f"- `{link['fileKey']}`{node}: {link['url']}")
        lines.append("")

    if context.get("nodes"):
        lines.extend(["## Nodes", ""])
        for node in context["nodes"]:
            lines.append(
                f"- `{node['fileKey']}` `{node['nodeId']}` {node.get('type') or 'UNKNOWN'}: {node.get('name') or 'Unnamed'}"
            )
        lines.append("")

    if context.get("errors"):
        lines.extend(["## Errors", ""])
        for error in context["errors"]:
            lines.append(f"- `{error.get('fileKey')}`: {error}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Extract Figma REST context for build-ui headless mode.")
    parser.add_argument("--repo")
    parser.add_argument("--issue-number")
    parser.add_argument("--input-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    figma_token = os.environ.get("FIGMA_ACCESS_TOKEN", "")
    github_token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
                    or "")

    context = {
        "artifactType": "figma-context",
        "generatedAt": now_iso(),
        "source": "figma-rest",
        "status": "skipped",
        "reason": None,
        "links": [],
        "files": [],
        "nodes": [],
        "images": [],
        "errors": [],
    }

    try:
        if args.skip:
            context["reason"] = "skipped by --skip-figma"
        else:
            issue_context = (load_issue_context_from_file(args.input_file)
                             if args.input_file else
                             load_issue_context_from_github(
                                 args.repo, args.issue_number, github_token))
            links = filter_allowed_figma_links(
                extract_figma_links(context_text(issue_context)),
                os.environ.get("FIGMA_FILE_ALLOWLIST", ""),
            )
            context["links"] = links
            context[
                "scope"] = "issue title, body, and trusted collaborator comments"
            if not links:
                context["status"] = "not-found"
                context["reason"] = "no Figma links found in issue context"
            elif not figma_token:
                context["status"] = "skipped"
                context[
                    "reason"] = "FIGMA_ACCESS_TOKEN is not configured; links were recorded but REST artifacts were not fetched"
            else:
                files, nodes, images, errors = fetch_figma_artifacts(
                    links, output_path, figma_token)
                context["files"] = files
                context["nodes"] = nodes
                context["images"] = images
                context["errors"] = errors
                context["status"] = "partial" if errors else "extracted"
                context["reason"] = ("some Figma REST requests failed" if errors
                                     else "Figma REST context extracted")
    except Exception as error:
        context["status"] = "failed"
        context["reason"] = str(error)
        context["errors"].append({"error": str(error)})

    summary_path = output_path.parent / "figma-summary.md"
    context["summaryPath"] = str(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(context), encoding="utf-8")
    write_json(output_path, context)

    print(f"{context['status']}:{output_path}")
    return 1 if context["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
