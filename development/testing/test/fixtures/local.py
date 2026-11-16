# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile

import pytest


@pytest.fixture(scope='session')
def valid_fake_private_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


@pytest.fixture
def tempdir(request):
    with tempfile.NamedTemporaryFile(delete=False) as tempdir:
        os.unlink(tempdir.name)

        request.addfinalizer(
            lambda: shutil.rmtree(tempdir.name, ignore_errors=True))
        return tempdir
