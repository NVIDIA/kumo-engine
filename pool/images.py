# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diskgraph pool runtime image selection.

Pool pods should use the same Kumo image-version convention as the legacy
SaaS launchers: derive image tags from ``get_kumo_version()`` by default and
only pass version tags, never arbitrary image URLs, through placement.
"""
import os
import os.path as osp
from enum import Enum

from kumo_core.version import get_kumo_version

from pool.dataclasses import SnapshotActivity
from service_framework.deploy_env import DEFAULT_REGISTRY


class PoolImageFlavor(str, Enum):
    CPU = 'cpu'
    GPU = 'gpu'


def backend_image_uri(flavor: PoolImageFlavor) -> str:
    tag = os.environ.get('KUMO_IMAGE_TAG') or get_kumo_version()
    repo = ('kumo-backend-gpu'
            if flavor == PoolImageFlavor.GPU else 'kumo-backend-cpu')
    reference = f'{osp.join(DEFAULT_REGISTRY, repo)}:{tag}'
    digest = os.environ.get('KUMO_IMAGE_DIGEST')
    return f'{reference}@{digest}' if digest else reference


def daemon_image_uri() -> str:
    # Durable pool daemons run on GPU cache nodes.
    return backend_image_uri(PoolImageFlavor.GPU)


def activity_image_uri(_activity: SnapshotActivity) -> str:
    # Pool activities run on GPU hosts, including CPU-heavy preprocessing.
    return backend_image_uri(PoolImageFlavor.GPU)
