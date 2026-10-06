# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging as py_logging
from unittest.mock import patch

from common_utils.startup import log_app_startup


def test_log_app_startup_emits_structured_entry(caplog):
    with patch('common_utils.startup.get_kumo_version', return_value='v1.2.3'):
        with caplog.at_level(py_logging.INFO):
            log_app_startup()

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == py_logging.INFO
    assert 'image_version=v1.2.3' in record.getMessage()
