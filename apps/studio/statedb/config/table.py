# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validation error messages
PKEY_PRESENT_IN_COLS = (
    "Primary Key (pkey) must be present in cols, but was not "
    "found. Please make sure your primary key is valid and "
    "present in your table and try again!")

PKEY_IS_ID = ("Primary Key (pkey) must be an ID. Please change the "
              "semantic type (stype) of your column to ID "
              "and try again!")

TIME_COL_PRESENT_IN_COLS = (
    "Time Column (time_col) must be present in cols, but was "
    "not found. Please make sure your time column is valid "
    "and present in your table and try again!")

END_TIME_COL_PRESENT_IN_COLS = (
    "End Time Column (end_time_col) must be present in cols, but was "
    "not found. Please make sure your end time column is valid "
    "and present in your table and try again!")

END_TIME_COL_IN_NON_DIM_TABLE = (
    "End Time Column (end_time_col) can only be "
    "specified for dimension tables. Please remove "
    "the end time column and try again!")

END_TIME_SAME_AS_TIME_COL = (
    "End Time Column (end_time_col) cannot be the same "
    "as the time column. Please select a different "
    "column and try again!")

TIME_COL_NEQ_PKEY = ("The time column cannot be the same as "
                     "the primary key")


class TableDataVersion(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema())
