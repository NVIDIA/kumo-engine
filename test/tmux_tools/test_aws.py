# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from tmux_tools.aws import Info


@pytest.mark.parametrize('error', [
    ConnectionResetError(104, 'connection reset'),
    TimeoutError('timed out'),
])
def test_metadata_network_failure_means_host_is_not_aws(
        monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    monkeypatch.setattr('urllib.request.urlopen', Mock(side_effect=error))

    assert Info.is_on_aws() is False
