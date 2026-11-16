# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from io_utils import kumo_dataclass
from statedb.config.common import ColumnID


@kumo_dataclass(order=True)
class ColumnGroup:
    """Represents a list of columns that should be linked together.

    This method of data entry provides a more intuitive way for users
    of the UI to set up relationships between tables. Each column group
    contains a list of columns across multiple tables that should be mutually
    joined together. For now, we have a constraint that each column group
    may only have 1 primary key. However, in the future, you could imagine
    that a group could have multiple or no primary keys, or could support
    additional link options such as "fuzzy match".
    """

    cols: List[ColumnID]
