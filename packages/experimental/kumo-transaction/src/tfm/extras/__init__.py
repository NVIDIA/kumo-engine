# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional backends. Importing ``tfm`` must not import these packages."""

from tfm.extras.capability import extra_available, require_extra

__all__ = ["extra_available", "require_extra"]
