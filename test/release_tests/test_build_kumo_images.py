# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which image lineup gets built, and from which ref.

Only kumo3 (v3.0+) images are built from main; v1.x/v2.x refs are rejected.
The Python builder remains shared by active Kumo 3 release scripts and private
operational callers, so its ref and explicit-application resolution stay
covered here.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.release.build_kumo_images import (
    DEFAULT_APPLICATION,
    Application,
    Platform,
    WorkflowFile,
    gather_flags,
    images,
    resolve_application,
)

# A ref that carries no version: nothing to derive from.
SHA = "0123456789abcdef0123456789abcdef01234567"
ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(
    (ROOT / ".github/workflows/_docker-kumo3-images-custom.yml").read_text())

SAAS_LINEUP = {
    # cpu, gpu and ui are built as one set by a single orchestrator workflow,
    # so a deployable ref always has all three.
    Application.KUMO3: [
        WorkflowFile.KUMO3_IMAGES,
    ],
}


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # Patch tags supported by the builder's application inference.
        ("v3.0.0", Application.KUMO3),
        ("v10.0.0", Application.KUMO3),
        # Floating minor tags / release branch names, built alongside the patch tag.
        ("v3.0", Application.KUMO3),
        # Refs that carry no version fall back to the default rather than guessing.
        (SHA, DEFAULT_APPLICATION),
        ("master", DEFAULT_APPLICATION),
        # Nonstandard tags must not be read as Kumo release refs.
        ("other-v1.0.0", DEFAULT_APPLICATION),
        # Not release-shaped, even though it starts with a v.
        ("v3-experiment", DEFAULT_APPLICATION),
    ])
def test_ref_selects_the_product_line_and_its_lineup(ref, expected):
    app = resolve_application(None, ref)
    assert app == expected
    # The mapping only matters because it selects a different set of workflows.
    # Close that loop: ref in, image lineup out.
    assert images[app][Platform.SAAS] == SAAS_LINEUP[app]


@pytest.mark.parametrize("ref", ["v1.9.4", "v2.21.35", "v2.21"])
def test_v1_and_v2_release_refs_are_rejected(ref):
    with pytest.raises(ValueError, match="only Kumo 3"):
        resolve_application(None, ref)


@pytest.mark.parametrize(
    "ref", ["v2.21.35", "v3.0.0", "v10.0.0", SHA, "master", "other-v1.0.0"])
def test_explicit_app_always_wins(ref):
    # Release callers pass `--app` explicitly. Derivation must never override
    # an explicit application selection.
    assert resolve_application(Application.KUMO3, ref) == Application.KUMO3


def test_kumo3_release_bakes_complete_default_glove_bundle():
    assert dict(
        gather_flags(
            Application.KUMO3,
            Platform.SAAS,
            WorkflowFile.KUMO3_IMAGES,
            object(),
        ))["bake_glove_level"] == "basic"


def test_acr_publish_flag_is_passed_to_the_kumo3_builder():
    class Args:
        publish_to_acr = True

    flags = dict(
        gather_flags(
            Application.KUMO3,
            Platform.SAAS,
            WorkflowFile.KUMO3_IMAGES,
            Args(),
        ))
    assert flags["publish_to_acr"] == "true"


@pytest.fixture
def image_cli(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("build_kumo_images.py", "mirror_kumo_images.py"):
        shutil.copy2(ROOT / "scripts/release" / name, scripts / name)
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "gh.log"
    stubs = {
        "git":
        """#!/bin/bash
test "$3" = show-ref && test "$6" = refs/heads/main
""",
        "gh":
        """#!/bin/bash
printf '%s\\n' "$*" >> "$IMAGE_CLI_LOG"
case "$1 $2" in
  "--version ") echo 'gh version 2.96.0' ;;
  "api user") echo 'test-maintainer' ;;
  "workflow run") echo 'https://github.com/NVIDIA/kumo-engine/actions/runs/42' ;;
  "run list") printf '[{"createdAt":"%s","databaseId":42}]' "$(date -u +%FT%TZ)" ;;
  "run watch") exit 0 ;;
  "run view") echo '{"startedAt":"2026-01-01T00:00:00Z","updatedAt":"2026-01-01T00:00:01Z"}' ;;
  *) echo "Unexpected gh call: $*" >&2; exit 1 ;;
esac
""",
    }
    for name, body in stubs.items():
        executable = tools / name
        executable.write_text(body)
        executable.chmod(0o755)

    def run(name, *args):
        result = subprocess.run(
            [sys.executable, str(scripts / name), *args],
            env=os.environ | {
                "PATH": f"{tools}:{os.defpath}",
                "IMAGE_CLI_LOG": str(log),
                "GH_REPO": "unrelated/fork",
            },
            capture_output=True,
            text=True,
        )
        return result, log.read_text().splitlines() if log.exists() else []

    return run


@pytest.mark.parametrize("extra", [[], ["--publish_to_acr"]])
def test_image_build_dispatches_public_repo_main(image_cli, extra):
    result, calls = image_cli("build_kumo_images.py", SHA, *extra, "--attach")
    assert result.returncode == 0, result.stdout + result.stderr
    dispatch, = [call for call in calls if call.startswith("workflow run ")]
    assert "--ref main" in dispatch
    scoped = [
        call for call in calls if call.startswith(("workflow run ", "run "))
    ]
    assert all("--repo NVIDIA/kumo-engine" in call for call in scoped), scoped
    assert "https://github.com/NVIDIA/kumo-engine/actions/runs/42" in result.stdout


def test_acr_publish_rejects_untrusted_workflow_branch(image_cli):
    result, calls = image_cli("build_kumo_images.py", SHA, "--publish_to_acr",
                              "--workflow_branch", "feature")
    assert result.returncode == 2
    assert "--publish_to_acr requires --workflow_branch main" in result.stderr
    assert not calls


def test_image_mirror_dispatches_public_repo_main(image_cli):
    result, calls = image_cli("mirror_kumo_images.py", "--mirror_to", "azure",
                              "--workflow", "KUMO3_IMAGES", "--tag", SHA)
    assert result.returncode == 0, result.stdout + result.stderr
    dispatches = [call for call in calls if call.startswith("workflow run ")]
    assert len(dispatches) == 3
    assert all("--ref main" in call for call in dispatches), dispatches
    scoped = [
        call for call in calls if call.startswith(("workflow run ", "run "))
    ]
    assert all("--repo NVIDIA/kumo-engine" in call for call in scoped), scoped


@pytest.mark.parametrize("ref,accepted", [
    ("refs/heads/main", True),
    ("refs/heads/master", False),
    ("refs/heads/feature", False),
    ("refs/tags/main", False),
    ("refs/heads/main-other", False),
    ("", False),
])
def test_publication_guard_accepts_only_main(ref, accepted):
    step, = WORKFLOW["jobs"]["guard"]["steps"]
    result = subprocess.run(["bash", "-e", "-c", step["run"]],
                            env=os.environ | {"WORKFLOW_REF": ref},
                            capture_output=True, text=True)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.parametrize("ref,accepted", [("refs/heads/main", True),
                                          ("refs/heads/feature", False)])
def test_acr_plan_requires_main_and_plans_complete_image_set(
        tmp_path, ref, accepted):
    aws = tmp_path / "aws"
    aws.write_text(
        "#!/bin/sh\nprintf '%s\\n' ImageNotFoundException >&2\nexit 255\n")
    aws.chmod(0o755)
    output = tmp_path / "output"
    step = next(s for s in WORKFLOW["jobs"]["plan"]["steps"]
                if s.get("id") == "plan")
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]], env=os.environ | {
            "PATH": f"{tmp_path}:{os.defpath}",
            "GITHUB_REF": ref,
            "GITHUB_OUTPUT": str(output),
            "REF": SHA,
            "IMAGES": "all",
            "ALIAS": "",
            "FORCE": "false",
            "PUBLISH_TO_ACR": "true",
        }, capture_output=True, text=True)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr
    if accepted:
        values = dict(
            line.split("=", 1) for line in output.read_text().splitlines())
        assert all(values[f"build_{image}"] == "true"
                   for image in ("cpu", "gpu", "ui"))
        assert values["selected_repos"].split() == [
            "kumo/kumo-backend-cpu", "kumo/kumo-backend-gpu", "kumo/kumo-ui"
        ]
    else:
        assert not output.exists()
