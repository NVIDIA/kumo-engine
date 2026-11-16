# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from test.io_utils.fsutil.contract import FilesystemContract

import pytest


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def root(tmp_path):
    path = tmp_path / 'root'
    path.mkdir()
    return str(path)


class TestLocalFilesystem(FilesystemContract):
    pass
