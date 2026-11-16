# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live tier: the binding against real AWS S3, read through the reader
it exists to serve.

``s3://kumo-public-datasets/`` is what the hosted SDK example asserts against,
so reading it here is the portability evidence. No fixture in a customer or
production bucket is named.
"""

from __future__ import annotations

import os

import kumo_s3
import pytest
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumoapi.typing import Dtype

PUBLIC_BUCKET = "kumo-public-datasets"
PUBLIC_ROOT = f"{PUBLIC_BUCKET}/customerltv_mini"
REGION = "us-west-2"

if not os.environ.get("KUMO_S3_LIVE"):
    pytest.skip(
        "LIVE TIER NOT RUN: nothing in this file was covered. Set "
        f"KUMO_S3_LIVE=1 with AWS credentials that can read s3://{PUBLIC_ROOT}"
        "/; AWS_PROFILE must be unset for the ambient instance role to be "
        "used.",
        allow_module_level=True,
    )


def live_filesystem():
    return kumo_s3.filesystem(PUBLIC_BUCKET, session=kumo_s3.ambient_session())


def test_the_bucket_answers_in_the_region_it_lives_in():
    """A bucket signed for the wrong region answers with a bare 301 naming
    neither."""
    assert kumo_s3.bucket_region(PUBLIC_BUCKET) == REGION


def test_the_ambient_chain_resolves_a_credential_to_sign_with():
    filesystem = live_filesystem()
    assert filesystem is not None
    assert filesystem.region == REGION


def test_the_bound_filesystem_reads_the_table_the_example_pins():
    reader = ObjectStoreSourceReader(live_filesystem(), root=PUBLIC_ROOT)
    assert reader.dtypes("customer") == {"CustomerID": Dtype.float64}
