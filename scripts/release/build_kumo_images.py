#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Run all of the workflows to build kumo images
# and push to image repositories.
#
# NOTE: Must not `import kumo` from this script for portability

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from packaging.version import Version

SCRIPT_DIR = Path(__file__).parent
MIN_GH_VERSION = Version("2.28.0")
GH_RUN_DETAILS_VERSION = Version("2.87.0")


class Application(Enum):
    KUMO3 = "kumo3"
    ALL = "all"

    def expand(self) -> list['Application']:
        if self == Application.ALL:
            return [Application.KUMO3]
        return [self]

    def __str__(self):
        """
        String value for enum, required for use in argparse.
        """
        return self.value


# The application built when nothing -- neither the caller nor the ref -- says
# which product line is wanted.
DEFAULT_APPLICATION = Application.KUMO3

# A release ref: the branch `v3.0`, the floating minor tag `v2.21`, or a patch
# tag `v2.21.35`. SHAs and branch names deliberately do not match because they
# carry no version to read.
_RELEASE_REF_RE = re.compile(r"^v(\d+)\.\d+(?:\.\d+)?$")


def application_from_ref(kumo_branch: str) -> Optional[Application]:
    """
    Which product line a ref belongs to, or None when the ref does not say.

    v3.0+ is kumo3 (kumo-backend-cpu, kumo-backend-gpu, kumo-ui). v1.x/v2.x
    images are no longer built from master.

    :param kumo_branch: the ref whose contents are being built.
    :return: the application the ref implies, or None to leave it to the caller.
    :raises ValueError: for a v1.x/v2.x release ref.
    """
    match = _RELEASE_REF_RE.match(kumo_branch)
    if match is None:
        return None
    if int(match.group(1)) < 3:
        raise ValueError(f"{kumo_branch} is a v1.x/v2.x release; only Kumo 3 "
                         f"images can be built from master")
    return Application.KUMO3


def resolve_application(app: Optional[Application],
                        kumo_branch: str) -> Application:
    """
    Decide which application to build images for.

    :param app: the value of --app, or None when it was not passed.
    :param kumo_branch: the ref whose contents are being built.
    :return: an explicitly requested application, else the one the ref implies,
             else DEFAULT_APPLICATION.
    """
    if app is not None:
        return app
    derived = application_from_ref(kumo_branch)
    if derived is None:
        return DEFAULT_APPLICATION
    print(f"Derived --app={derived} from release ref {kumo_branch}. "
          f"Pass --app explicitly to override.")
    return derived


class Platform(Enum):
    SAAS = "saas"
    ALL = "all"

    def expand(self, app: Application) -> list['Platform']:
        """
        :param self: Target platform as requested by user.
        :return: List of platforms to build images for.
        """
        _all_platforms: dict[Application, list[Platform]] = {
            Application.KUMO3: [
                Platform.SAAS,
            ],
        }
        if self == Platform.ALL:
            return _all_platforms[app]
        else:
            return [self]

    def __str__(self):
        """
        String value for enum, required for use in argparse.
        """
        return self.value


class WorkflowFile(Enum):
    UI = "_docker-ui-image-custom.yml"
    # kumo3 backend-image consolidation: one orchestrator builds cpu+gpu+ui as
    # a set and publishes them together, so a deployable ref always has all
    # three.
    KUMO3_IMAGES = "_docker-kumo3-images-custom.yml"

    def __lt__(self, other: 'WorkflowFile'):
        return self.value < other.value


def check_github_cli() -> bool:
    """
    Validate that github command line tool `gh` is installed.
    Warn if the installed version is out of date.
    :return: True when gh supports returning run details from workflow dispatch.
    """
    try:
        proc = subprocess.run(["gh", "--version"], stdout=subprocess.PIPE,
                              text=True)
    except FileNotFoundError:
        print("Please install github CLI: https://cli.github.com/")
        exit(1)
    match = re.search("[0-9]+([.][0-9]+)+", proc.stdout)
    if match is None:
        print("Warning: unknown version schema")
        print(proc.stdout)
        return False

    gh_version = Version(match.group())
    if gh_version < MIN_GH_VERSION:
        print("You are running an out of date version of github cli. "
              "Please upgrade to at least v2.28.0")
        print("Current version:", match.group())

    if gh_version < GH_RUN_DETAILS_VERSION:
        print("Note: gh versions < v2.87.0 do not return workflow run URLs "
              "for `gh workflow run`; using legacy run lookup.")
        return False
    return True


def gh_whoami() -> str:
    proc = subprocess.run(["gh", "api", "user", "-q", ".login"],
                          stdout=subprocess.PIPE, text=True)
    return proc.stdout.strip()


# List of yml workflow files used to build images for each platform
images: dict[Application, dict[Platform, list[WorkflowFile]]] = {
    # kumo3 backend image consolidation: one cpu backend + one gpu backend +
    # UI, built as a coupled set by a single orchestrator workflow. Frequent
    # rebuilds (every kumo source change).
    Application.KUMO3: {
        Platform.SAAS: [
            WorkflowFile.KUMO3_IMAGES,
        ],
    },
}


# Per-workflow configuration flags
# Provided to github actions as key-value flags
def gather_flags(app: Application, platform: Platform,
                 workflow: WorkflowFile, args) -> list[tuple[str, str]]:
    flags: dict[str, str] = {}

    def _gather_from_cross_product(
        *,
        apps: list[Application],
        platforms: list[Platform],
        workflows: list[WorkflowFile],
        values: dict[str, str],
    ):
        """Update flags if all conditions are met"""
        if app in apps and platform in platforms and workflow in workflows:
            flags.update(values)

    _gather_from_cross_product(
        apps=[Application.KUMO3],
        platforms=[Platform.SAAS],
        workflows=[
            WorkflowFile.KUMO3_IMAGES,
        ],
        values={
            "bake_glove_level": "basic",
        },
    )

    if (app == Application.KUMO3 and platform == Platform.SAAS
            and workflow == WorkflowFile.KUMO3_IMAGES
            and getattr(args, "publish_to_acr", False)):
        flags["publish_to_acr"] = "true"

    return list(sorted(flags.items()))


@dataclass(order=True)
class WorkflowSpec:
    """
    Definition of a Github Actions workflow to run.
    :param workflow_name: The name of the yaml workflow file to run.
    :param workflow_branch: The branch or tag used to fetch scripts.
    :param kumo_branch: The branch to build images for; image contents.
    :param flags: Key-value flags to provide this workflow.
    """
    workflow: WorkflowFile
    workflow_branch: str
    kumo_branch: str
    flags: list[tuple[str, str]]

    def __hash__(self):
        return hash(
            (self.workflow.value, self.workflow_branch, self.kumo_branch,
             tuple(self.flags)))


@dataclass(frozen=True)
class CloudRegistry:
    name: str
    registry: str
    user_secret: str  # name of GH secret holding the registry username
    password_secret: str  # name of GH secret holding the registry password


def timestamp_as_dt(timestamp: str) -> datetime:
    dt_format = r"%Y-%m-%dT%H:%M:%SZ"
    return datetime.strptime(timestamp, dt_format)


def _latest_run(workflow_name: str, user: str) -> tuple[datetime, int]:
    command_list = [
        "gh", "run", "list", "--json=createdAt,databaseId", f"--user={user}",
        f"--workflow={workflow_name}"
    ]
    proc_list = subprocess.run(command_list, check=True, stdout=subprocess.PIPE,
                               text=True, cwd=SCRIPT_DIR)
    parsed_list = [(timestamp_as_dt(obj['createdAt']), int(obj['databaseId']))
                   for obj in json.loads(proc_list.stdout)]
    if not parsed_list:
        raise ValueError("no workflow runs found")
    return sorted(parsed_list)[-1]


def get_run_id(workflow_name: str, user: str,
               timefilter: datetime) -> Optional[int]:
    """
    Determine run id for most recent workflow run matching workflow_name.
    Timestamp must be after timefilter to be considered.
    Workaround for https://github.com/cli/cli/issues/4001
    """
    def _get_run_id_internal(workflow_name: str, user: str,
                             timefilter: datetime) -> int:
        latest = _latest_run(workflow_name, user)
        if latest[0] < timefilter:
            raise ValueError(f"no workflow run since {timefilter=} "
                             f"({latest=} is too early)")
        return latest[1]

    delay = 1.0
    for _ in range(4):
        try:
            return _get_run_id_internal(workflow_name, user, timefilter)
        except ValueError:
            time.sleep(delay)
            delay *= 1.5
    try:
        return _get_run_id_internal(workflow_name, user, timefilter)
    except ValueError as e:
        print(repr(e))


def get_run_url(run_id: int) -> str:
    return (f"https://github.com/kumo-ai/kumo/"
            f"actions/runs/{run_id}")


def get_workflow_url(spec: WorkflowSpec) -> str:
    return (f"https://github.com/kumo-ai/kumo/"
            f"actions/workflows/{spec.workflow.value}")


def _run_id_from_dispatch_stdout(stdout: str) -> Optional[int]:
    match = re.search(
        r"https://github\.com/[^/\s]+/[^/\s]+/actions/runs/([0-9]+)",
        stdout,
    )
    if match is None:
        return None
    return int(match.group(1))


def run_workflow(spec: WorkflowSpec, user: str,
                 gh_supports_run_details: bool) -> Optional[int]:
    """
    Command Github Actions to run a workflow.
    :param spec: specification of the workflow to execute.
    :return: returns workflow run_id.
    """
    flag_list = [f"{k}={v}" for k, v in spec.flags]
    prefixed_flags = [x for flag in flag_list for x in ("-f", flag)]
    # flags k:v, ... -> ["-f", "k=v", ...]

    command = [
        "gh", "workflow", "run", spec.workflow.value, "--ref",
        spec.workflow_branch, "-f", f"kumo_repo_ref={spec.kumo_branch}"
    ] + prefixed_flags
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=SCRIPT_DIR,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"gh workflow run failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise

    if gh_supports_run_details:
        run_id = _run_id_from_dispatch_stdout(proc.stdout)
        if run_id is not None:
            return run_id
        print("Warning: expected run URL in `gh workflow run` output, "
              "falling back to legacy run lookup.")

    skew = timedelta(minutes=1)
    timefilter = datetime.now() - skew

    return get_run_id(spec.workflow.value, user, timefilter)


def duration_workflow_run(run_id: int) -> timedelta:
    """
    Compute the runtime of a completed workflow
    """
    command_time = [
        "gh", "run", "view", "--json=startedAt,updatedAt",
        str(run_id)
    ]
    proc_time = subprocess.run(command_time, check=True, stdout=subprocess.PIPE,
                               text=True, cwd=SCRIPT_DIR)
    obj = json.loads(proc_time.stdout)
    dt_start = timestamp_as_dt(obj["startedAt"])
    dt_end = timestamp_as_dt(obj["updatedAt"])
    return dt_end - dt_start


def attach_workflow_run(description: str, run_id: int):
    """
    Watch a workflow run until it completes.
    Aborts the process with nonzero exit code if the run fails.
    """
    print("awaiting", description, "run", run_id)
    command_watch = ["gh", "run", "watch", "--exit-status", str(run_id)]
    proc = subprocess.run(command_watch, capture_output=True, cwd=SCRIPT_DIR)
    if proc.returncode != 0:
        print("Build Failed, see", get_run_url(run_id))
        exit(proc.returncode)
    print("-", "done in", duration_workflow_run(run_id))


def collect_workflows(args) -> list[WorkflowSpec]:
    """
    Collect the set of workflows required to satisfy the user request
    :param args: argparse object containing workflow_branch, kumo_branch,
                 target for image builds
    :return: returns the set of WorkflowSpec to execute.
    """
    workflows: set[WorkflowSpec] = set()

    def _add_workflows(app: Application, platform: Platform,
                       images: list[WorkflowFile]):
        for img in images:
            combined_flags = gather_flags(app, platform, img, args)
            workflows.add(
                WorkflowSpec(
                    workflow=img,
                    workflow_branch=args.workflow_branch,
                    kumo_branch=args.kumo_branch,
                    flags=list(sorted(combined_flags)),
                ))

    for target in args.target.expand(args.app):
        if args.app not in images or target not in images[args.app]:
            raise ValueError(f"Invalid app/platform combination "
                             f"{args.app=} {target=}")
        _add_workflows(args.app, target, images[args.app][target])

    return list(sorted(workflows))


def validate_branch_or_tag(branch_or_tag):
    branch_check = subprocess.run([
        "git", "-C", SCRIPT_DIR, "show-ref", "-q", "--verify",
        f"refs/heads/{branch_or_tag}"
    ])
    tag_check = subprocess.run([
        "git", "-C", SCRIPT_DIR, "show-ref", "-q", "--verify",
        f"refs/tags/{branch_or_tag}"
    ])
    if branch_check.returncode != 0 and tag_check.returncode != 0:
        print(f"{branch_or_tag} is not a valid branch or tag name, "
              f"so it cannot be used to trigger the github action.")
        exit(1)


# ECR image names for each workflow, as the path component after the registry
# (e.g. "kumo/kumo-ui"). Used to construct sync workflow inputs. A workflow may
# produce several images: the kumo3 orchestrator publishes the whole cpu+gpu+ui
# set from a single dispatch.
#
_ECR_IMAGE_NAMES: dict[WorkflowFile, list[str]] = {
    WorkflowFile.UI: ["kumo/kumo-ui"],
    WorkflowFile.KUMO3_IMAGES: [
        "kumo/kumo-backend-cpu",
        "kumo/kumo-backend-gpu",
        "kumo/kumo-ui",
    ],
}


def get_ecr_images_for_sync(
    workflow: WorkflowFile,
    tag: str,
) -> list[tuple[str, str]]:
    """
    Return the ECR (image_name, tag) pairs produced by a given workflow.
    image_name is the ECR path without registry (e.g. "kumo/kumo-ui").

    Returns an empty list for workflows that don't land in ECR.
    """
    if workflow not in _ECR_IMAGE_NAMES:
        return []

    return [(name, tag) for name in _ECR_IMAGE_NAMES[workflow]]


CLOUD_REGISTRIES: dict[str, CloudRegistry] = {
    "azure":
    CloudRegistry(
        name="azure",
        registry="kumoacr.azurecr.io",
        user_secret="ACR_CLIENT_ID",
        password_secret="ACR_CLIENT_SECRET",
    ),
    # TODO: add GCP (GAR) support — confirm registry URL and create GH secrets
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Build Kumo Images",
        description="Trigger github workflows to build kumo images",
        epilog=("Please remember to 'git push' "
                "if you are building a local branch."))
    parser.add_argument(
        "kumo_branch", help=("the Git SHA or tag of the Kumo "
                             "repository that you want to build."))
    parser.add_argument(
        "--workflow_branch", required=False, default=None,
        help=("the branch or tag of Kumo that will "
              "be used to fetch the scripts which "
              "generate individual container images. Defaults to master."))
    parser.add_argument(
        "--app", type=Application, choices=list(Application), required=False,
        default=None,
        help="the application for which images will be built and pushed. If "
        "omitted, it is derived from kumo_branch: a v3.0+ release ref selects "
        "kumo3 (cpu/gpu backend image consolidation) and a v1.x or v2.x one "
        f"is rejected. Anything that is not a release ref defaults to "
        f"{DEFAULT_APPLICATION}.")
    parser.add_argument(
        "--target", type=Platform, choices=list(Platform), required=False,
        default=Platform.SAAS,
        help="the platform for which images will be built and pushed.")
    parser.add_argument(
        "--mirror_to", choices=list(CLOUD_REGISTRIES), required=False,
        default=None, help=(
            "If specified, sync all built ECR images to the given cloud "
            "registry after building. Implies full attach: builds and "
            "sync workflows are both awaited."))
    parser.add_argument(
        "--publish_to_acr", action="store_true",
        help=("Also publish the complete Kumo 3 SaaS image set to the "
              "customer ACR."))
    parser.add_argument(
        "--attach", action="store_true",
        help=("watch progress and wait for all image "
              "builds to complete before exiting."))
    args = parser.parse_args()

    # Derive the application from the code ref when not set explicitly, and
    # reject v1.x/v2.x release refs.
    args.app = resolve_application(args.app, args.kumo_branch)

    if args.workflow_branch is None:
        args.workflow_branch = "master"

    if args.publish_to_acr:
        release_app = application_from_ref(args.kumo_branch)
        is_full_sha = re.fullmatch(r"[0-9a-f]{40}", args.kumo_branch)
        if args.app != Application.KUMO3 or args.target != Platform.SAAS:
            parser.error("--publish_to_acr requires --app kumo3 --target saas")
        if is_full_sha is None and release_app != Application.KUMO3:
            parser.error("--publish_to_acr requires a full Git SHA or a "
                         "Kumo 3 release ref")
        if args.workflow_branch != "master":
            parser.error("--publish_to_acr requires --workflow_branch master")

    validate_branch_or_tag(args.workflow_branch)
    gh_supports_run_details = check_github_cli()
    user = gh_whoami()

    mirror_registry = CLOUD_REGISTRIES[
        args.mirror_to] if args.mirror_to else None

    workflow_specs = collect_workflows(args)
    workflow_ids = []
    print("You can monitor the build progress at these URLs (CMD+click):")
    for spec in workflow_specs:
        run_id = run_workflow(spec, user, gh_supports_run_details)
        workflow_ids.append(run_id)
        print("-", get_run_url(run_id) if run_id else get_workflow_url(spec))

    if mirror_registry:
        if not all(run_id is not None for run_id in workflow_ids):
            print(
                "Warning: not all build run IDs could be determined. Skipping sync."
            )
            exit(1)

        image_pairs = []
        for spec in workflow_specs:
            image_pairs.extend(
                get_ecr_images_for_sync(spec.workflow, spec.kumo_branch))

        mirror_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "mirror_kumo_images.py"),
            "--mirror_to",
            args.mirror_to,
            *[arg for img, tag in image_pairs for arg in ("--image", img, tag)],
            "--await-run-ids",
            *[str(r) for r in workflow_ids],
        ]
        result = subprocess.run(mirror_cmd, cwd=SCRIPT_DIR)
        exit(result.returncode)

    elif args.attach:
        print()
        for spec, run_id in zip(workflow_specs, workflow_ids):
            if run_id:
                attach_workflow_run(spec.workflow.value, run_id)

        if not all(run_id is not None for run_id in workflow_ids):
            print("Warning: not all run IDs could be determined. "
                  "Your builds may not be complete, manual checking required. "
                  "Exiting with a failure to block dependent operations.")
            exit(1)
