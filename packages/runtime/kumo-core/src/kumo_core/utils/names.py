# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

DATA_SOURCE_TABLE_SEPARATOR = '.'

# Fully Qualified Table Name Utils ############################################


# TODO(kiriti): Convert assertions to errors with useful message.
def fully_qualified_table_name(name: Union[str, Tuple[str, str], List[str]],
                               data_source: Optional[str] = None) -> str:
    """
    Constructs and returns a fully qualified table name in the format
    "{data_source}.{table_name}".

    Args:
        name (str, Tuple[str, str], List[str])): The name of the table.
            Supports these formats for table names and data sources:
                * name={name}, data_source={data_source}
                * name=({data_source}, {name}), data_source=None
                * name=[{data_source}, {name}], data_source=None
                * name={data_source}.{name}, data_source=None
        data_source (str, optional): The data source of the table. Should be
            provided only when using `str` format for table name.
    """
    # If it's already a fully qualified name, return it.
    if isinstance(name, str) and is_fully_qualified_name(name):
        if data_source is not None:
            # Ensure that the data_source matches the input.
            assert get_data_source(name) == data_source, (
                f"data_source mismatch. Got {data_source}, expected "
                f"{get_data_source(name)}")
        return name

    def _fully_qualified_table_name(table_name: str, data_source: str) -> str:
        return f"{data_source}{DATA_SOURCE_TABLE_SEPARATOR}{table_name}"

    if data_source is not None:
        assert isinstance(name, str)
        return _fully_qualified_table_name(name, data_source)

    if isinstance(name, (tuple, list)):
        assert len(name) == 2, (
            f"Only support {type(name)} of length 2 for fully qualified table "
            f"name. Got {name}")
        return _fully_qualified_table_name(name[1], name[0])

    # TODO(kiriti): Add a tables argument which can be matched to find the
    # data_source of the table.
    raise NotImplementedError(
        f"Fully qualified table name without data source is not implemented."
        f"Got name: {name}, data_source: None")


def is_fully_qualified_name(table_name: str) -> bool:
    """
    Checks whether a table name is fully qualified.
    """
    if DATA_SOURCE_TABLE_SEPARATOR in table_name:
        return len(table_name.split(DATA_SOURCE_TABLE_SEPARATOR)) == 2
    else:
        return False


def get_table_name(fq_table_name: str) -> str:
    r"""Returns the table name from a fully qualified table name"""
    if not is_fully_qualified_name(fq_table_name):
        return fq_table_name
    return fq_table_name.split(DATA_SOURCE_TABLE_SEPARATOR, maxsplit=1)[1]


def get_data_source(fq_table_name: str) -> str:
    r"""Returns the data_source from a fully qualified table name"""
    assert is_fully_qualified_name(fq_table_name)
    return fq_table_name.split(DATA_SOURCE_TABLE_SEPARATOR, maxsplit=1)[0]


# End Fully Qualified Table Name Utils ########################################
