# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access

import sys

import numpy
import pytest
import torch

from kumo.testing import withMemory

torch.set_num_threads(1)

if not sys.platform.startswith("linux"):
    pytest.skip("skipping linux-only memory tests", allow_module_level=True)


def test_good_allocation():
    @withMemory(limit=1.5e9, return_result=False)
    def allocate_1GB_limit_1GB():
        obj = numpy.arange(1000 * 1000 * 125, dtype='u8')
        return obj

    assert allocate_1GB_limit_1GB()


def test_bad_allocation():
    @withMemory(limit=1e9, return_result=False)
    def allocate_2GB_limit_1GB():
        obj = numpy.arange(1000 * 1000 * 250, dtype='u8')
        return obj

    with pytest.raises(numpy.core._exceptions._ArrayMemoryError):
        allocate_2GB_limit_1GB()


def test_no_limit():
    @withMemory(return_result=False)
    def allocate_2GB():
        obj = numpy.arange(1000 * 1000 * 250, dtype='u8')
        return obj

    assert allocate_2GB()
