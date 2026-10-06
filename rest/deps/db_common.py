# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from temporalio_commons.auto_rpc_exception import RetriableError


class OptimisticLockError(RetriableError):
    """Raised when optimistic lock validation fails due to concurrent writes."""
