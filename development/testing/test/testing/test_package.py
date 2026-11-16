# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from kumo.testing.package import withPackage


def test_with_package_skips_when_required_executable_is_missing():
    with patch('kumo.testing.package.find_spec', return_value=object()), \
            patch('kumo.testing.package.which', return_value=None):

        @withPackage('graphviz', executables=('dot', ))
        def gated_test():
            raise AssertionError('Skipped test must not execute')

    skip_mark = gated_test.pytestmark[-1]
    assert skip_mark.name == 'skipif'
    assert skip_mark.args == (True, )
    assert "executable(s) {'dot'}" in skip_mark.kwargs['reason']
