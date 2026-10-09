# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import os.path as osp
import subprocess
from functools import cache
import json
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    entry_points,
)
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

#: Entry-point group of version providers: zero-argument callables returning
#: a version string, or `None` to defer to the next provider. Providers must
#: be cheap to import (loading one imports its module).
VERSION_PROVIDER_GROUP = "kumo_core.version_providers"


@cache
def get_kumo_version() -> str:
    r"""Returns the version of Kumo, where version is defined in alignment
    with our image naming conventions. This version can be used to specify
    an image ID.

    Version providers are asked in entry-point name order and the first one
    that returns a version wins. Distributions register providers in the
    `kumo_core.version_providers` group (the platform and kumo-engine each do);
    kumo_core only defines the mechanism. If none answers, the git checkout the
    code runs from is used, and finally the hard-coded `latest`."""
    for provider in _version_providers():
        try:
            version = provider()
        except Exception as e:
            logger.error(f"Kumo version provider {provider!r} failed: {e}")
            continue
        if version:
            logger.info(f"Obtained Kumo version from {provider!r} as "
                        f"{version}")
            return version

    try:
        # Kumo is installed in editable mode:
        version = get_version_from_git()
        if version:
            logger.info(f"Obtained Kumo version from the git hash of the "
                        f"installation directory as {version}")
            return version
    except Exception as e:
        logger.error(f"Error when obtain Kumo version from git "
                     f"installation: {e}")

    version = "latest"
    logger.warning(
        f"Could not obtain Kumo version from a version provider or git "
        f"installation. Using the default version `{version}`, but note that "
        f"this may cause problems downstream.")
    return version


def _version_providers() -> List[Callable[[], Optional[str]]]:
    r"""The providers registered by installed distributions, in entry-point
    name order (so the names decide precedence)."""
    providers: List[Callable[[], Optional[str]]] = []
    for entry_point in sorted(entry_points(group=VERSION_PROVIDER_GROUP),
                              key=lambda e: e.name):
        try:
            providers.append(entry_point.load())
        except Exception as e:
            logger.error(f"Could not load Kumo version provider "
                         f"{entry_point.name!r}: {e}")
    return providers


def installed_distribution_version(name: str) -> Optional[str]:
    r"""The version of the installed distribution `name`, or `None` if it is
    not installed or is an editable install.

    Version providers use this so that an installed release reports its
    release version while a source checkout (an editable install) defers to its
    git revision."""
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return None
    direct_url = dist.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info",
                                                 {}).get("editable"):
        return None
    return dist.version


def get_version_from_git() -> Optional[str]:
    r"""Obtain the version of Kumo based on the current git SHA (if on
    master) or the current branch name (if not on master). Requires an
    editable install, so the Kumo install path is pointed to the development
    directory."""
    kumo_install_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    if _is_site_packages(kumo_install_path):
        # A regular install, not a checkout. Without this, a virtualenv that
        # happens to live inside the user's own git repository would report
        # that repository's branch as the Kumo version.
        return None
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


def _is_site_packages(path: str) -> bool:
    return osp.basename(path) in ("site-packages", "dist-packages")


def _is_git_repository(path: str) -> bool:
    return subprocess.call(['git', '-C', path, 'status'],
                           stderr=subprocess.STDOUT,
                           stdout=open(os.devnull, 'w')) == 0
