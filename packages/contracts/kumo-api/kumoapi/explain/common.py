# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Dict, List, Optional, Union

from pydantic.dataclasses import dataclass


@dataclass
class EntityMappings:
    r"""Stores the mappings between each entity to the category
    it belongs to, such as 'true_positive', 'random' etc.

    Args:
        mappings (Dict[str, List[str]]): The mappings of
            selection strategy/category and corresponding entity ids
            starting from zero. These ids are used as file names for
            the entity-level XAI files.
        orig_mappings (Optional[Dict[str, List[Union[str, int]]]]): Mappings of
            selection strategy/category and corresponding original
            entity primary keys. None means original IDs were not recorded.
    """
    mappings: Dict[str, List[str]] = field(default_factory=dict)
    orig_mappings: Optional[Dict[str, List[Union[str, int]]]] = None
