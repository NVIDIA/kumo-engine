# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / (
    ".github/workflows/kumo-engine-wheel.yml"
)
JOBS = yaml.safe_load(WORKFLOW.read_text())["jobs"]
BUILDS = JOBS["build"]["strategy"]["matrix"]["include"]
PORTAL_WORKFLOW = yaml.safe_load((WORKFLOW.parent / "kitmaker-portal.yml").read_text())
PORTAL_JOBS = PORTAL_WORKFLOW["jobs"]


def _step(job, name):
    return next(step for step in JOBS[job]['steps'] if step['name'] == name)


def _capture_command(path, capture):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n'
        f'Path({str(capture)!r}).write_text(json.dumps({{'
        "'args': sys.argv[1:], 'target': os.getenv('MACOSX_DEPLOYMENT_TARGET')"
        '}))\n')
    path.chmod(0o755)


def test_checkout_preserves_notice_bytes(tmp_path):
    """Override a Windows-style checkout policy without changing Git config."""
    source = tmp_path / 'notice.txt'
    canonical = b'Copyright example\nLicense text\n'
    subprocess.run(['git', 'init', str(tmp_path)], check=True,
                   capture_output=True)
    (tmp_path / '.gitattributes').write_bytes(
        (WORKFLOW.parents[2] / '.gitattributes').read_bytes())
    source.write_bytes(canonical)
    subprocess.run(['git', '-c', 'core.autocrlf=false', 'add', 'notice.txt'],
                   cwd=tmp_path, check=True)
    subprocess.run(['git', 'config', '--local', 'core.autocrlf', 'true'],
                   cwd=tmp_path, check=True)
    source.unlink()
    subprocess.run(['git', 'checkout-index', '--force', 'notice.txt'],
                   cwd=tmp_path, check=True)
    assert source.read_bytes() == canonical.replace(b'\n', b'\r\n')

    checkout_env = _step('build', 'Checkout repository')['env']
    source.unlink()
    subprocess.run(['git', 'checkout-index', '--force', 'notice.txt'],
                   cwd=tmp_path, env=dict(os.environ, **checkout_env),
                   check=True)
    assert source.read_bytes() == canonical
    configured = subprocess.check_output(
        ['git', 'config', '--local', 'core.autocrlf'], cwd=tmp_path, text=True)
    assert configured.strip() == 'true'


@pytest.mark.parametrize('build', BUILDS, ids=lambda build: build['platform'])
@pytest.mark.parametrize('release', [False, True])
def test_build_step_selects_native_platform_and_release_mode(
        tmp_path, build, release):
    capture = tmp_path / 'command.json'
    _capture_command(tmp_path / 'packages/kumo-engine/scripts/build_wheel.sh',
                     capture)
    uname = tmp_path / 'uname'
    uname.write_text(f'#!/bin/sh\necho {build["machine"]}\n')
    uname.chmod(0o755)
    step = _step('build', 'Build wheel')
    subprocess.run(['bash', '-e', '-c', step['run']], cwd=tmp_path, env=dict(
        os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
        WHEEL_ARCH=build['machine'], WHEEL_PLATFORM=build['platform'],
        MACOSX_DEPLOYMENT_TARGET=step['env']['MACOSX_DEPLOYMENT_TARGET'],
        IS_RELEASE=str(release).lower()), check=True)

    command = json.loads(capture.read_text())
    expected = (['--platform-tag', build['platform']]
                if build['platform'].startswith('manylinux_') else [])
    assert command['args'] == expected + (['--release'] if release else [])
    if build['platform'].startswith('macosx_'):
        assert command['target'] == '14.0'


@pytest.mark.parametrize('job,step_name,build', [
    (job, step_name, build)
    for job, step_name in [
        ('check-requirements', 'Resolve the wheel and each extra from PyPI'),
        ('kitmaker', 'Release through Kitmaker'),
    ]
    for build in BUILDS
    if build['arch'] in {row['arch'] for row in JOBS[job][
        'strategy']['matrix']['include']}
])
def test_release_steps_use_the_matching_platform_and_wheel(
    tmp_path: Path, job: str, step_name: str, build: dict[str, str]
) -> None:
    """Execute a matrix-selected command with its candidate wheel.

    Args:
        tmp_path: Isolated command and wheel fixture directory.
        job: Engine workflow job being verified.
        step_name: Command step in the engine workflow.
        build: Build matrix entry for the candidate platform.
    """
    row = next(
        row
        for row in JOBS[job]["strategy"]["matrix"]["include"]
        if row["arch"] == build["arch"]
    )
    step = (
        _step(job, step_name)
        if job != "kitmaker"
        else next(
            step for step in PORTAL_JOBS["portal"]["steps"] if step["name"] == step_name
        )
    )
    download = (
        _step(job, "Download wheel")["with"]["name"]
        if job != "kitmaker"
        else JOBS[job]["with"]["artifact"]
    )
    upload = _step("build", "Upload wheel")["with"]["name"]
    assert download == upload
    if job == "kitmaker":
        assert row["platform"] == build["platform"]
        assert JOBS[job]["uses"] == "./.github/workflows/kitmaker-portal.yml"
        assert JOBS[job]["with"]["platform"] == "${{ matrix.platform }}"
        assert JOBS[job]["with"]["upload"] is True
        assert JOBS[job]["permissions"]["id-token"] == "write"

    wheel = tmp_path / "dist" / f"kumo_engine-1.0-cp311-abi3-{build['platform']}.whl"
    wheel.parent.mkdir()
    wheel.touch()
    capture = tmp_path / "command.json"
    _capture_command(tmp_path / ("uv" if job == "kitmaker" else "python3"), capture)
    subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PATH=f"{tmp_path}:{os.environ['PATH']}",
            WHEEL_PLATFORM=row["platform"],
            RELEASE_TAG="kumo-engine-v1.0",
            REPO="NVIDIA/kumo-engine",
            PLATFORM=row["platform"],
            PROJECT="kumo-engine",
            PROJECT_ID="6903",
            UPLOAD="true",
            MACOSX_DEPLOYMENT_TARGET=step.get("env", {}).get(
                "MACOSX_DEPLOYMENT_TARGET", ""
            ),
        ),
        check=True,
    )

    command = json.loads(capture.read_text())
    args = command["args"]
    assert args[args.index("--platform") + 1] == row["platform"]
    assert str(wheel.relative_to(tmp_path)) in args
    if job == "check-requirements" and build["platform"].startswith("macosx_"):
        assert command["target"] == "14.0"


def test_all_platforms_require_validation_and_release_environment() -> None:
    """Verify candidate artifact selection and release approval gates."""
    arches = {build["arch"] for build in BUILDS}
    assert len(arches) == len(BUILDS)
    assert {
        row["arch"]
        for row in JOBS["check-requirements"]["strategy"]["matrix"]["include"]
    } == arches
    assert "windows-x86_64" in arches
    release_arches = {
        row["arch"]
        for row in JOBS["attach-to-release"]["strategy"]["matrix"]["include"]
    }
    assert {"x86_64", "arm64"} <= release_arches
    assert release_arches <= {"x86_64", "arm64", "aarch64"}
    if "aarch64" in release_arches:
        assert "platform-qualification" in JOBS["attach-to-release"]["needs"]
    for job in ["attach-to-release", "kitmaker"]:
        published = {row["arch"] for row in JOBS[job]["strategy"]["matrix"]["include"]}
        assert published == release_arches
        if job == "kitmaker":
            download = next(
                step["with"]
                for step in PORTAL_JOBS["portal"]["steps"]
                if step["name"] == "Download wheel"
            )
            assert download["name"] == "${{ inputs.artifact }}"
            artifact = JOBS[job]["with"]["artifact"]
        else:
            download = _step(job, "Download wheel")["with"]
            artifact = download["name"]
        assert "pattern" not in download
        assert artifact == _step("build", "Upload wheel")["with"]["name"]
    assert JOBS["check-requirements"]["needs"] == ["build"]
    assert set(JOBS["attach-to-release"]["needs"]) == {
        "build",
        "check-requirements",
        "wheel-smoke",
        "platform-qualification",
        "skill-zips",
    }
    assert JOBS["kitmaker"]["needs"] == ["attach-to-release"]
    portal = PORTAL_JOBS["portal"]
    assert portal["environment"] == "kumo-engine-release"
    assert portal["if"] == (
        "${{ github.event_name == 'release' "
        "|| (github.event_name == 'workflow_dispatch' "
        "&& github.ref == 'refs/heads/main') }}"
    )
    assert (
        PORTAL_WORKFLOW[True]["workflow_call"]["secrets"]["KITMAKER_API_TOKEN"][
            "required"
        ]
        is False
    )
    assert portal["env"]["KITMAKER_API_TOKEN"] == "${{ secrets.KITMAKER_API_TOKEN }}"
    glove = yaml.safe_load(
        (WORKFLOW.parent / "kumo-glove-6b-50d-wheel.yml").read_text()
    )
    for caller in [JOBS["kitmaker"], glove["jobs"]["kitmaker"]]:
        assert caller["secrets"] == {
            "KITMAKER_API_TOKEN": "${{ secrets.KITMAKER_API_TOKEN }}"
        }
    assert portal["env"]["RELEASE_TAG"] == "${{ inputs.release_tag }}"
    assert JOBS["kitmaker"]["with"]["release_tag"] == (
        "${{ github.event.release.tag_name }}"
    )
    for step in portal["steps"]:
        if step["name"] in {"Download wheel", "Release through Kitmaker"}:
            assert step["if"] == "${{ github.event_name == 'release' }}"
    assert _step("build", "Checkout repository")["with"]["persist-credentials"] is False
    for job in ["attach-to-release", "kitmaker"]:
        assert JOBS[job]["if"] == "${{ github.event_name == 'release' }}"
    smoke = yaml.safe_load(
        (WORKFLOW.parent / "kumo-engine-wheel-smoke.yml").read_text()
    )
    assert set(smoke[True]) == {"workflow_call"}


def test_publication_forwards_the_shared_prediction_tolerance():
    root = WORKFLOW.parents[2]
    workflow = yaml.safe_load(
        (root / ('.github/workflows/kumo-engine-wheel-smoke.yml')).read_text())
    action = yaml.safe_load(
        (root / ('.github/actions/engine-gpu-smoke/action.yml')).read_text())
    call = next(step['with'] for step in workflow['jobs']['smoke']['steps']
                if step.get('uses') == './.github/actions/engine-gpu-smoke')
    pr = yaml.safe_load(
        (root / '.github/workflows/kumo-engine-pr.yml').read_text())
    pr_call = next(step['with'] for job in pr['jobs'].values()
                   for step in job.get('steps', [])
                   if step.get('uses') == './.github/actions/engine-gpu-smoke')
    run = next(
        step for step in action['runs']['steps']
        if step.get('name') == 'Run the eight GPU tasks outside the checkout')
    assert call['reload-atol'] == pr_call['reload-atol'] == '0.005'
    assert run['env']['RELOAD_ATOL'] == '${{ inputs.reload-atol }}'
    assert '--reload-atol "$RELOAD_ATOL"' in run['run']


def test_cpu_qualification_uses_candidate_artifacts_on_all_pythons():
    caller = JOBS['platform-qualification']
    assert caller['needs'] == ['build']
    assert caller['uses'] == (
        './.github/workflows/kumo-engine-platform-smoke.yml')
    path = WORKFLOW.parent / 'kumo-engine-platform-smoke.yml'
    workflow = yaml.safe_load(path.read_text())
    # PyYAML's YAML 1.1 parser interprets the GitHub Actions `on` key as True.
    assert workflow[True] == {'workflow_call': None}
    job = workflow['jobs']['smoke']
    matrix = job['strategy']['matrix']
    assert set(matrix['python']) == {'3.11', '3.12', '3.13', '3.14'}
    assert len(matrix['target']) == 3
    assert {row['platform']: row['artifact'] for row in matrix['target']} == {
        'windows-native': 'windows-x86_64', 'windows-wsl': 'x86_64',
        'linux-arm64': 'aarch64',
    }
    assert {row['artifact'] for row in matrix['target']} <= {
        build['arch'] for build in BUILDS}
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    download = next(step for step in job['steps']
                    if step.get('uses', '').startswith(
                        'actions/download-artifact@'))
    assert download['with'] == {
        'name': ('kumo-engine-wheel-${{ matrix.target.artifact }}-'
                 '${{ github.sha }}'),
        'path': '${{ runner.temp }}/platform-wheel/artifacts',
    }
    # Supplying another run or token here could accidentally qualify a stale
    # artifact while publishing the current build.
    assert 'run-id' not in download['with']
    assert 'github-token' not in download['with']
