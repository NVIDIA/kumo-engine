# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample datasets installed with kumo-engine."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def shop() -> str:
    r"""Returns the absolute path of the sample shop dataset's directory.

    `users.parquet` holds 1,000 users (`user_id`) and `orders.parquet`
    their 12,000 orders (`order_id`, `user_id`, `order_time`) over 2025.
    The data is random: each user orders at their own rate, at random
    hours. It contains no real or third-party data. Read it with
    `FileConnector(root_dir=datasets.shop())`. The files are installed
    package data: treat them as read-only.
    """
    return str(_ROOT / 'shop')
