# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import os.path as osp
import subprocess
from functools import cache
from typing import Optional

logger = logging.getLogger(__name__)


@cache
def get_kumo_version() -> str:
    r"""Returns the version of Kumo, where version is defined in alignment
    with our image naming conventions. This version can be used to specify
    an image ID."""
    try:
        # Best case scenario: running Kumo in a container, version stored
        # there:
        version = get_version_from_container()
        if version:
            logger.info(
                f"Obtained Kumo version from the container kumo_version.txt"
                f" as {version}")
            return version

        # Second best case scenario: Kumo is installed in editable mode:
        version = get_version_from_git()
        if version:
            logger.info(f"Obtained Kumo version from the git hash of the "
                        f"installation directory as {version}")
            return version
    except Exception as e:
        logger.error(f"Error when obtain Kumo version from container "
                     f"information or git installation: {e}")

    version = "latest"
    logger.warning(
        f"Could not obtain Kumo version from container information or git "
        f"installation. Using the default version `{version}`, but note that "
        f"this may cause problems downstream.")
    return version


def get_version_from_container() -> Optional[str]:
    r"""Get the kumo version (commit) for the current container, as determined
    by the kumo_version.txt file (if present)."""
    path = osp.expanduser('~/kumo_version.txt')
    if osp.exists(path):
        with open(path, 'r') as f:
            version = f.read().strip()
            if len(version) > 0:
                return version
    return None


def get_version_from_git() -> Optional[str]:
    r"""Obtain the version of Kumo based on the current git SHA (if on
    master) or the current branch name (if not on master). Requires an
    editable install, so the Kumo install path is pointed to the development
    directory."""
    kumo_install_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    if _is_git_repository(kumo_install_path):
        # Step 1: Check if we are in a branch with an exact tag. If so, we
        # should return the tagged version:
        try:
            commit_hash = subprocess.check_output(
                ['git', 'log', '-n1', "--pretty='%h'"])
            commit_hash = commit_hash.decode("utf-8").strip("\n").strip("\'")
            tag_name = subprocess.check_output([
                'git', '-C', kumo_install_path, 'describe', '--exact-match',
                '--tags', commit_hash
            ], stderr=subprocess.DEVNULL)
            tag_name = tag_name.decode("utf-8").strip("\n")
            return tag_name
        except subprocess.CalledProcessError:
            # Not on a tag:
            pass

        # Step 2: Check the branch name:
        branch_name = subprocess.check_output([
            'git', '-C', kumo_install_path, 'rev-parse', '--abbrev-ref', 'HEAD'
        ]).decode("utf-8").strip("\n")
        if branch_name != "master":
            return branch_name
        return subprocess.check_output(
            ['git', '-C', kumo_install_path, 'rev-parse',
             'HEAD']).decode("utf-8").strip("\n")
    return None


def _is_git_repository(path: str) -> bool:
    return subprocess.call(['git', '-C', path, 'status'],
                           stderr=subprocess.STDOUT,
                           stdout=open(os.devnull, 'w')) == 0
