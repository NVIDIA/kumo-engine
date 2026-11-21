#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Sync kumo images from ECR to a secondary cloud registry.
#
# Two modes:
#
# --workflow mode  (human-facing)
#   Resolve ECR image names from workflow name(s) + branch tag, then sync.
#   Use this when images are already in ECR and you want to re-sync without
#   rebuilding — e.g. after a transient sync failure, or to mirror a branch
#   to a new registry.
#
#   SaaS images:
#     python mirror_kumo_images.py --mirror_to azure \
#       --workflow KUMO3_IMAGES --tag my-branch
#
#   Dry-run to preview resolved image:tag pairs without triggering anything:
#     python mirror_kumo_images.py --mirror_to azure \
#       --workflow KUMO3_IMAGES --tag my-branch --dry-run
#
# --image mode  (programmatic)
#   Accept pre-resolved image name + tag pairs directly. Intended to be called
#   by build_kumo_images.py after builds complete, passing the exact ECR paths
#   it just built. Can also be used directly when you know the exact image refs.
#
#   Wait for build run IDs before syncing, then sync:
#     python mirror_kumo_images.py --mirror_to azure \
#       --image kumo/kumo-rest my-branch \
#       --image kumo/kumo my-branch \
#       --await-run-ids 12345678 12345679
#
#   Sync immediately (images already in ECR):
#     python mirror_kumo_images.py --mirror_to azure \
#       --image kumo/kumo-rest my-branch
#
# NOTE: Must not `import kumo` from this script for portability

import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Optional

from build_kumo_images import (
    _ECR_IMAGE_NAMES,
    CLOUD_REGISTRIES,
    GITHUB_REPOSITORY,
    SCRIPT_DIR,
    CloudRegistry,
    WorkflowFile,
    attach_workflow_run,
    check_github_cli,
    get_ecr_images_for_sync,
    get_run_id,
    get_run_url,
    gh_whoami,
)

SYNC_WORKFLOW = "_docker-image-sync.yml"


def trigger_sync_workflow(
    image_name: str,
    source_tag: str,
    registry: CloudRegistry,
    user: str,
) -> Optional[int]:
    """
    Trigger _docker-image-sync.yml to copy image_name:source_tag from ECR to
    the target registry. Returns the GHA run ID, or None if it cannot be determined.
    """
    flag_list = [
        f"source_image_name={image_name}",
        f"source_tag={source_tag}",
        f"target_registry={registry.registry}",
        f"target_registry_user_secret={registry.user_secret}",
        f"target_registry_password_secret={registry.password_secret}",
    ]
    prefixed_flags = [x for flag in flag_list for x in ("-f", flag)]
    command = [
        "gh", "workflow", "run", SYNC_WORKFLOW, "--repo", GITHUB_REPOSITORY,
        "--ref", "main"
    ] + prefixed_flags
    try:
        subprocess.run(command, capture_output=True, check=True, cwd=SCRIPT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"gh workflow run failed:\nstdout: {e.stdout.decode()}\nstderr: {e.stderr.decode()}")
        raise

    skew = timedelta(minutes=1)
    timefilter = datetime.now() - skew
    return get_run_id(SYNC_WORKFLOW, user, timefilter)


_SYNCABLE_WORKFLOWS = [w.name for w in _ECR_IMAGE_NAMES]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mirror Kumo Images",
        description=(
            "Sync kumo images from ECR to a secondary cloud registry.\n\n"
            "  --workflow mode: human-facing. Resolves ECR image names from workflow\n"
            "                   name(s) + branch tag. Use to re-sync without rebuilding.\n"
            "  --image mode:    programmatic. Accepts pre-resolved image:tag pairs.\n"
            "                   Called by build_kumo_images.py; can also be used directly.\n\n"
            "Run with --dry-run to preview resolved image:tag pairs without syncing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mirror_to",
        choices=list(CLOUD_REGISTRIES),
        required=True,
        help="Target cloud registry to sync images to.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--workflow",
        nargs="+",
        choices=_SYNCABLE_WORKFLOWS,
        metavar="WORKFLOW",
        help=(
            f"One or more workflow names to resolve and sync "
            f"({', '.join(_SYNCABLE_WORKFLOWS)}). "
            "Requires --tag."
        ),
    )
    mode.add_argument(
        "--image",
        nargs=2,
        action="append",
        metavar=("NAME", "TAG"),
        dest="images",
        help=(
            "Pre-resolved image name and tag to sync "
            "(e.g. --image kumo/kumo-rest my-branch). "
            "Repeat for multiple images. "
            "Optionally combine with --await-run-ids to wait for builds first."
        ),
    )

    parser.add_argument(
        "--tag",
        help="Branch or tag to sync. Required with --workflow; has no effect with --image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved image:tag pairs and exit without triggering any sync workflows.",
    )
    parser.add_argument(
        "--await-run-ids",
        nargs="+",
        type=int,
        metavar="RUN_ID",
        dest="await_run_ids",
        help=(
            "GHA build run IDs to wait for before syncing. "
            "Run IDs are printed by build_kumo_images.py when --mirror_to is set. "
            "Only valid with --image."
        ),
    )

    args = parser.parse_args()

    if args.workflow and not args.tag:
        parser.error("--tag is required when using --workflow")
    if args.images and args.tag:
        parser.error("--tag has no effect with --image (tags are already resolved)")

    if not args.dry_run:
        check_github_cli()
    user = gh_whoami() if not args.dry_run else None
    registry = CLOUD_REGISTRIES[args.mirror_to]

    # Resolve image+tag pairs
    if args.images:
        image_pairs = [tuple(pair) for pair in args.images]
    else:
        image_pairs = []
        for name in args.workflow:
            image_pairs.extend(get_ecr_images_for_sync(WorkflowFile[name], args.tag))
        if not image_pairs:
            print("No ECR images resolved for the specified workflows.")
            exit(1)

    if args.dry_run:
        print(f"Would sync to {registry.registry}:")
        for image_name, tag in image_pairs:
            print(f"  {image_name}:{tag}")
        exit(0)

    # Wait for in-progress builds before syncing
    if args.await_run_ids:
        print("Waiting for builds to complete before syncing...")
        for run_id in args.await_run_ids:
            attach_workflow_run("build", run_id)

    # Trigger all sync workflows (GHA runs them concurrently)
    sync_run_ids: list[int | None] = []
    print(f"\nTriggering image sync to {registry.registry} (CMD+click):")
    for image_name, tag in image_pairs:
        sync_run_id = trigger_sync_workflow(image_name, tag, registry, user)
        sync_run_ids.append(sync_run_id)
        label = f"sync {image_name}:{tag}"
        print("-", get_run_url(sync_run_id) if sync_run_id else label)

    # Attach to sync workflows
    print()
    for (image_name, tag), sync_run_id in zip(image_pairs, sync_run_ids):
        if sync_run_id:
            attach_workflow_run(f"sync {image_name}:{tag}", sync_run_id)

    if not all(run_id is not None for run_id in sync_run_ids):
        print(
            "Warning: not all sync run IDs could be determined. "
            "Your syncs may not be complete, manual checking required. "
            "Exiting with a failure to block dependent operations."
        )
        exit(1)
