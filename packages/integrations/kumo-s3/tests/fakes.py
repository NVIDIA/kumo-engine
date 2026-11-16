# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeSession:
    """A boto3 session, reduced to what a filesystem is built from."""
    def __init__(self, *, region_name: str | None = None,
                 credentials: Any = 'default') -> None:
        self.region_name = region_name
        self._credentials = (SimpleNamespace(
            get_frozen_credentials=lambda: SimpleNamespace(
                access_key='key', secret_key='secret', token='token'))
                             if credentials == 'default' else credentials)

    def get_credentials(self) -> Any:
        return self._credentials
