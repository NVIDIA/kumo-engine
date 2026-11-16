# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, List

from kumoapi.source_table import FileType as ApiFileType

from common_utils.enum import extends_enum


@extends_enum(ApiFileType)
class FileType(str, Enum):
    CSV = "CSV"
    PARQUET = "PARQUET"

    @staticmethod
    def parse_extension(fname: str) -> 'FileType':
        fname = fname.rstrip('/').casefold()
        for typ, extension in _FILE_TYPE_EXTENSIONS.items():
            if fname.endswith(extension):
                return typ
        raise ValueError(
            f"file path must have one of the following extensions: "
            f"{FileType.valid_extensions()}")

    @staticmethod
    def valid_extensions() -> List[str]:
        return list(_FILE_TYPE_EXTENSIONS.values())

    @property
    def extension(self) -> str:
        return _FILE_TYPE_EXTENSIONS[self]


_FILE_TYPE_EXTENSIONS: Dict[FileType, str] = {
    FileType.CSV: '.csv',
    FileType.PARQUET: '.parquet',
}
