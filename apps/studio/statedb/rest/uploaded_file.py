# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.data_source import UploadFileType

from io_utils.dataclasses import kumo_dataclass
from statedb.config.common import (
    ConnectorID,
    DBEntry,
    ProjectID,
    SourceTableName,
    UploadedFileID,
)


@kumo_dataclass
class UploadedFileDBEntry(DBEntry):
    """Metadata for one completed Kumo-managed project upload."""

    id: UploadedFileID
    project_id: ProjectID
    connector_id: ConnectorID
    source_table_name: SourceTableName
    normalized_source_table_name: str
    file_type: UploadFileType
    path: str
