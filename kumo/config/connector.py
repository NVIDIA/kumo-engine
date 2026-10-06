# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from io_utils import kumo_dataclass


@kumo_dataclass
class ConnectorConfig:
    name: Optional[str] = None


# Individual connector configurations are defined in the respective connector
# files.
