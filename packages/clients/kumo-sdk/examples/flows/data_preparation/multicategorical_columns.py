# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configure scalar and native-array multi-categorical columns.

The delimiter is part of the saved table definition. Once Kumo infers or a
user supplies it, every snapshot and model uses that same interpretation.
"""

import kumoai_studio


def configure_multicategorical_table(
        source_table: kumoai_studio.SourceTable) -> kumoai_studio.Table:
    """Build a table with delimited-string and native-array categories."""
    return kumoai_studio.Table(
        source_table=source_table,
        columns=[
            kumoai_studio.Column(
                name='interests',
                stype='multicategorical',
                dtype='string',
                multicategorical_delimiter='|',
            ),
            # Native arrays are already separated and need no delimiter.
            kumoai_studio.Column(
                name='product_ids',
                stype='multicategorical',
                dtype='intlist',
            ),
        ],
    )
