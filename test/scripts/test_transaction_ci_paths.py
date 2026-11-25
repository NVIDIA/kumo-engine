# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.transaction_ci_paths import package_changed


def commit(root: Path, path: str, text: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run([
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--no-gpg-sign",
        "-qm",
        "fixture",
    ], cwd=root, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    base = commit(tmp_path, "packages/experimental/kumo-transaction/old.py",
                  "value = 1\n")
    return tmp_path, base


def test_unrelated_diff_does_not_run_package(repository) -> None:
    root, base = repository
    head = commit(root, "README.md", "unrelated\n")
    assert not package_changed(root, base, head)


@pytest.mark.parametrize("path", [
    "packages/experimental/kumo-transaction/new.py",
    ".github/workflows/transaction-blueprint.yml",
    "scripts/check_license_headers.py",
    "setup.cfg",
])
def test_package_and_integration_policy_changes_run_checks(
        repository, path: str) -> None:
    root, base = repository
    head = commit(root, path, "changed\n")
    assert package_changed(root, base, head)


def test_moving_a_file_out_of_the_package_still_runs_checks(repository) -> None:
    root, base = repository
    (root / "packages/experimental/kumo-transaction/old.py").rename(root /
                                                                    "moved.py")
    head = commit(root, "README.md", "moved\n")
    assert package_changed(root, base, head)


def test_missing_or_invalid_revisions_fail_instead_of_skipping(
        repository) -> None:
    root, head = repository
    assert package_changed(root, "0" * 40, head)
    with pytest.raises(ValueError):
        package_changed(root, "", head)
    with pytest.raises(subprocess.CalledProcessError):
        package_changed(root, "1" * 40, head)


def test_required_status_is_not_hidden_by_event_path_filters() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / ".github/workflows/transaction-blueprint.yml"
    workflow = yaml.safe_load(path.read_text())
    triggers = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "merge_group"):
        assert not triggers[event]
    job = workflow["jobs"]["blueprint"]
    assert job["name"] == "transaction-blueprint"
    assert "if" not in job
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "uv sync --frozen --extra dev" in commands
    assert "pytest tests/unit tests/contract tests/recipe_smoke" in commands
    assert "check_python_license_metadata.py" in commands
    wheel_step = next(
        step for step in job["steps"]
        if step.get("name") == "Build and verify wheel legal metadata")
    command = shlex.split(wheel_step["run"].splitlines()[-1])
    checker = command[command.index("python") + 1]
    resolved = (root / wheel_step["working-directory"] / checker).resolve()
    assert resolved == root / "scripts/check_python_license_metadata.py"


def test_native_default_uses_verified_wheels_without_resyncing_git_sources() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/transaction-blueprint.yml").read_text())
    steps = workflow["jobs"]["blueprint"]["steps"]
    build = next(i for i, step in enumerate(steps) if step.get("name") == "Build or verify matching native wheels")
    install = next(i for i, step in enumerate(steps) if step.get("name") == "Install locked package dependencies")
    assert build < install
    assert "--extra native" in steps[install]["run"]
    assert "--extra kumo" in steps[install]["run"]
    assert "--extra baselines" in steps[install]["run"]
    assert "--no-deps --reinstall native-wheels/*/*.whl" in steps[install]["run"]
    for step in steps[install+1:]:
        for line in step.get("run", "").splitlines():
            if "uv run" in line:
                assert "uv run --no-sync" in line
