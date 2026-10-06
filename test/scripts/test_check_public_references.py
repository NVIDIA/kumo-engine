# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

import scripts.check_public_references as public_reference_scanner
from scripts.check_public_references import scan_repository, scan_text


def test_scan_reports_location_without_retaining_the_full_url() -> None:
    host = "kumoai." + "grafana.net"
    findings = scan_text("guide.md", f"See https://{host}/private?id=123")

    assert len(findings) == 1
    assert findings[0].path == "guide.md"
    assert findings[0].line == 1
    assert findings[0].host == host


def test_scan_allows_public_and_example_links() -> None:
    text = "\n".join([
        "https://docs.python.org/3/",
        "https://monitoring.example.test/dashboard",
        "https://github.com/kumo-ai/kumo",
    ])

    assert scan_text("README.md", text) == []


def test_scan_reports_company_host_without_url_scheme() -> None:
    host = "uat.staging." + "kumoai.cloud"

    findings = scan_text("settings.py", f"SERVICE_HOST={host}")
    assert len(findings) == 1
    assert findings[0].host == host


def test_scan_reports_company_chat_reference() -> None:
    reference = "get help in #ask-" + "infra-devops"

    findings = scan_text("README.md", reference)
    assert len(findings) == 1
    assert findings[0].category == "company chat channel"


def test_scan_skips_malformed_url_shaped_text() -> None:
    assert scan_text("example.txt", "http://[") == []


def test_strict_scan_reports_known_private_github_repository() -> None:
    url = "https://github.com/" + "kumo-ai/" + "kumo-ml.git"

    assert scan_text("setup.py", url) == []
    findings = scan_text("setup.py", url, strict=True)
    assert len(findings) == 1
    assert findings[0].category == "private Kumo GitHub repository"


def test_strict_scan_reports_scp_style_github_url() -> None:
    url = "git@github.com:" + "kumo-ai/kumo-grafana.git"

    findings = scan_text("monitoring.sh", url, strict=True)
    assert len(findings) == 1
    assert findings[0].host == "github.com"


def test_strict_scan_reports_mixed_case_github_references() -> None:
    https_url = "https://github.com/" + "Kumo-AI/Kumo-ML.GIT"
    scp_url = "git@GitHub.com:" + "Kumo-AI/Kumo-Grafana.GIT"

    assert len(scan_text("setup.py", https_url, strict=True)) == 1
    assert len(scan_text("monitoring.sh", scp_url, strict=True)) == 1


def test_strict_scan_reports_github_hosts_with_trailing_dot() -> None:
    https_url = "https://github.com./" + "kumo-ai/kumo-ml.git"
    scp_url = "git@github.com.:" + "kumo-ai/kumo-grafana.git"

    assert len(scan_text("setup.py", https_url, strict=True)) == 1
    assert len(scan_text("monitoring.sh", scp_url, strict=True)) == 1


def test_repository_scan_rejects_a_new_private_dependency(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    url = "https://github.com/" + "kumo-ai/" + "kumo-ml.git"
    (tmp_path / "README.md").write_text(url)

    findings = scan_repository(tmp_path)
    assert len(findings) == 1
    assert findings[0].repository == "kumo-ml"


def test_repository_scan_rejects_a_stale_private_dependency_allowance(
    tmp_path,
    monkeypatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(
        public_reference_scanner,
        "KNOWN_PRIVATE_GITHUB_DEPENDENCIES",
        {("README.md", "kumo-ml"): 1},
    )

    with pytest.raises(ValueError,
                       match="Stale private-dependency baseline entries"):
        scan_repository(tmp_path, validate_baseline=True)


def _git(repo, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_repository_scan_skips_the_mirrored_diskgraph_tree(tmp_path) -> None:
    _git(tmp_path, "init", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "base.md").write_text("public text")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    mirrored = tmp_path / "packages" / "diskgraph" / "README.md"
    mirrored.parent.mkdir(parents=True)
    mirrored.write_text("https://www." + "notion.so/page")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "mirror")
    head = _git(tmp_path, "rev-parse", "HEAD")

    assert scan_repository(tmp_path, strict=True) == []
    assert scan_repository(tmp_path, diff=(base, head)) == []


def test_diff_scan_reads_only_files_changed_after_base(tmp_path) -> None:
    _git(tmp_path, "init", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    private_host = "kumo." + "corp"
    (tmp_path / "unchanged.md").write_text(f"https://{private_host}/old")
    (tmp_path / "changed.md").write_text("https://example.test/public")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "changed.md").write_text(f"https://{private_host}/new")
    _git(tmp_path, "add", "changed.md")
    _git(tmp_path, "commit", "-m", "change")
    head = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "changed.md").write_text("working tree differs from head")

    findings = scan_repository(tmp_path, diff=(base, head))

    assert [(finding.path, finding.line)
            for finding in findings] == [("changed.md", 1)]


@pytest.mark.parametrize(
    ("text", "category", "value"),
    [
        ("arn:aws:iam::" + "210987654321:role/deploy", "cloud account ID",
         "210987654321"),
        ("FROM " + "210987654321.dkr.ecr.us-west-2.amazonaws.com/base",
         "cloud account ID", "210987654321"),
        ("https://ab12345." + "snowflakecomputing.com", "warehouse account",
         "ab12345"),
        ("https://app.snowflake.com/" + "us-west-2/ab12345/#/data",
         "warehouse account", "ab12345"),
        ("account='" + "ab12345'", "warehouse account", "ab12345"),
        ('"account": "' + 'abc12345"', "warehouse account", "abc12345"),
        ('project_id: "kumo-' + 'research",', "company cloud project",
         "kumo-research"),
        ('billing_project_id="kumo-' + 'research"', "company cloud project",
         "kumo-research"),
        ("GOOGLE_CLOUD_PROJECT=kumo-" + "research", "company cloud project",
         "kumo-research"),
        ('project: str = "kumo-' + 'research"', "company cloud project",
         "kumo-research"),
        ('os.environ.get("X_PROJECT", "kumo-' + 'research")',
         "company cloud project", "kumo-research"),
        ("account: ab" + "12345", "warehouse account", "ab12345"),
        ('account: str = "ab' + '12345"', "warehouse account", "ab12345"),
        ("projects/kumo-" + "research/datasets/x", "company cloud project",
         "kumo-research"),
        ("ci@kumo-" + "research.iam.gserviceaccount.com",
         "company cloud project", "kumo-research"),
        ("s3://" + "kumo-team-bucket/data.parquet", "company storage bucket",
         "kumo-team-bucket"),
        ("gs://" + "kumo/data", "company storage bucket", "kumo"),
        ("/home/" + "someone/checkout/run.sh", "personal home directory",
         "someone"),
        ("/Users/" + "someone/data", "personal home directory", "someone"),
    ],
)
def test_scan_reports_company_infrastructure_identifiers(
    text,
    category,
    value,
) -> None:
    findings = scan_text("config.py", text)

    assert [(finding.category, finding.host)
            for finding in findings] == [(category, value)]


@pytest.mark.parametrize(
    "text",
    [
        "arn:aws:iam::123456789012:role/example",
        "arn:aws:s3:::bucket and arn:aws:iam::000000000000:root",
        "111111111111.dkr.ecr.us-west-2.amazonaws.com/image",
        "https://acct.snowflakecomputing.com",
        "account='xy12345'",
        "project_id='example-project'",
        "project = 'kumopql'",
        "accountant = 12345",
        "gs://kumo-${DEPLOYMENT_NAME}/data",
        "s3://kumo-{deploy_mode}-dataplane-bucket/",
        "gs://kumo-<deployment-name>/",
        'import { Button } from "@kumo-ai/ui-catalog/button"',
        "https://{account}.snowflakecomputing.com",
        "https://app.snowflake.com/myaccount/#/data",
        "s3://kumo-public-datasets/movielens/ratings.parquet",
        "s3://customer-bucket/data and gs://example-bucket/data",
        "/home/runner/work and /home/ubuntu/.cache and ~/data and $HOME/x",
    ],
)
def test_scan_ignores_placeholder_and_public_identifiers(text) -> None:
    assert scan_text("config.py", text) == []


@pytest.mark.parametrize("path", [
    ".github/workflows/deploy.yml",
    "docker/Dockerfile.rest",
    "test/connector/test_snowflake_connector.py",
    "service_framework/deploy_env.py",
])
def test_identifier_rules_skip_exempt_paths(path) -> None:
    text = "arn:aws:iam::" + "210987654321:role/deploy"

    assert scan_text(path, text) == []
    assert len(scan_text("kumo/config.py", text)) == 1


def test_diff_scan_works_in_a_blobless_clone_without_lazy_fetch(
    tmp_path,
    monkeypatch,
) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--initial-branch=main")
    _git(origin, "config", "user.name", "Test User")
    _git(origin, "config", "user.email", "test@example.com")
    _git(origin, "config", "uploadpack.allowFilter", "true")
    (origin / "removed.md").write_text("content that only the base has\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "base")
    base = _git(origin, "rev-parse", "HEAD")
    _git(origin, "rm", "-q", "removed.md")
    (origin / "added.md").write_text("https://example.test/public\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "head")
    head = _git(origin, "rev-parse", "HEAD")

    clone = tmp_path / "clone"
    subprocess.run(
        [
            "git", "clone", "-q", "--filter=blob:none",
            origin.resolve().as_uri(),
            str(clone)
        ],
        check=True,
    )
    # CI cannot fetch file contents after checkout; fail instead of fetching.
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "1")

    assert scan_repository(clone, diff=(base, head)) == []
