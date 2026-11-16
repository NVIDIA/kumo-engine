# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.utils.time import now_utc

# Re-exported, not redefined. Callers inside a Temporal workflow want
# ``statedb.config.timestamp.now_utc`` instead: this one is the wall clock and
# is not replay-safe.
__all__ = ['now_utc']
