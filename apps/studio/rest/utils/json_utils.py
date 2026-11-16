# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import orjson
from fastapi.responses import JSONResponse


class NonStrictJSONResponse(JSONResponse):
    """Same as JSONResponse, but replaces floating point NaN and Inf with null.

    The default json encoder crashes when encountering these special floating
    point values.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)
