# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2018, Encode OSS Ltd.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import http
from typing import Any, Dict, Optional


class ProjectNotSetError(Exception):
    r"""Raised when an object operation is attempted without a project
    set.
    """
    def __init__(self) -> None:
        super().__init__(
            "No project is set. Call kumoai_studio.init() to select the canonical "
            "Default project, or call kumoai_studio.set_project() before performing "
            "object operations.")


class HTTPException(Exception):
    r"""An HTTP exception, with detailed information and headers."""
    def __init__(
        self,
        status_code: int,
        detail: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        # Adapted from https://github.com/Kludex/starlette/blob/b5126b2063dbd6fa46d3e60b4a7117c99dcdbe6e/starlette/exceptions.py
        if detail is None:
            detail = http.HTTPStatus(status_code).phrase
        self.status_code = status_code
        self.detail = detail
        self.headers = headers

    def __str__(self) -> str:
        return f"{self.status_code}: {self.detail}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (f"{class_name}(status_code={self.status_code!r}, "
                f"detail={self.detail!r})")
