# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from kumoapi.data_source import DataSourceType
from kumoapi.source_table import FileType

from kumoai_studio import global_state
from kumoai_studio.connector.base import Connector
from kumoai_studio.connector.utils import delete_uploaded_table, upload_table


class FileUploadConnector(Connector):
    r"""Upload CSV or Parquet tables into one Kumo project.

    Uploads are project-owned and use a hidden Kumo-managed connector. They do
    not use an external :class:`~kumoai_studio.identity.Identity`, and an uploaded
    table is visible only in the project that received it. Create one
    ``FileUploadConnector`` for the file format being uploaded.

    .. code-block:: python

        import kumoai_studio

        uploads = kumoai_studio.FileUploadConnector(file_type="parquet")
        uploads.upload(name="users", path="/data/users.parquet")

        assert uploads.has_table("users")
        source_table = uploads["users"]

    Args:
        file_type: Uploaded table format, either ``"csv"`` or ``"parquet"``.
        project_id: Exact project that receives the upload. ``None`` snapshots
            the active project; later active-project changes do not redirect
            this connector.

    Raises:
        :class:`ValueError`: If ``file_type`` is not ``"csv"`` or
            ``"parquet"``.
        :class:`~kumoai_studio.exceptions.ProjectNotSetError`: If no project is
            active.
    """
    def __init__(
        self,
        file_type: str,
        project_id: Optional[str] = None,
    ) -> None:
        normalized = file_type.lower()
        if normalized not in {'parquet', 'csv'}:
            raise ValueError("file_type must be 'csv' or 'parquet'")
        self._file_type = normalized
        if project_id is None:
            from kumoai_studio import _require_project
            _require_project()
            assert global_state._project_id is not None
            project_id = global_state._project_id
        elif not isinstance(project_id, str) or not project_id.strip():
            raise ValueError('project_id must be a non-empty project ID')
        self._kumo_project_id = project_id
        response = global_state.client.connector_api.get_upload_connector(
            project_id=self._kumo_project_id)
        self._id = response.connector_id

    @property
    def id(self) -> str:
        """Return the selected project's Kumo-managed upload connector ID."""
        return self._id

    @property
    def name(self) -> str:
        """Return the stable display name of the managed upload connector."""
        return 'uploads'

    @property
    def source_type(self) -> DataSourceType:
        """Return the internal source type used for uploaded files."""
        return DataSourceType.S3

    @property
    def file_type(self) -> FileType:
        """Return the CSV or Parquet format accepted by this instance."""
        return (FileType.PARQUET
                if self._file_type == 'parquet' else FileType.CSV)

    def upload(
        self,
        name: str,
        path: str,
        auto_partition: bool = True,
        partition_size_mb: int = 250,
    ) -> None:
        r"""Upload a table into this connector's immutable project.

        Local input is one CSV or Parquet file no larger than 100 GiB, matching
        this connector's ``file_type``, and uploaded using multipart
        transport. Remote ``s3://``, ``gs://``, ``abfs://``,
        ``abfss://``, and ``az://`` inputs may be a single file no larger
        than 1000 MiB or a directory of same-format shards with a consistent
        schema.

        After upload, access the table as ``connector[name]``. The table and
        its data remain scoped to the project that was active during upload.

        Args:
            name: Project-local table name to create.
            path: Local path or supported remote URL to a file or sharded
                directory. Its format must match ``file_type``.
            auto_partition: Deprecated compatibility option; ignored.
            partition_size_mb: Deprecated compatibility option; ignored.

        Raises:
            :class:`ValueError`: If the format, schema, size, or path is
                invalid.

        Example:
            >>> parquet = kumoai_studio.FileUploadConnector(  # doctest: +SKIP
            ...     "parquet")
            >>> parquet.upload(  # doctest: +SKIP
            ...     name="events", path="gs://data/events/")
            >>> csv = kumoai_studio.FileUploadConnector("csv")  # doctest: +SKIP
            >>> csv.upload(  # doctest: +SKIP
            ...     name="sales", path="/data/sales.csv")
        """
        upload_table(
            path=path,
            name=name,
            file_type=self._file_type,
            auto_partition=auto_partition,
            partition_size_mb=partition_size_mb,
            project_id=self._kumo_project_id,
        )

    def delete(self, name: str) -> None:
        r"""Delete a table from this connector's immutable project.

        A same-named upload in another project is not visible and is not
        modified, even if the process-wide active project has changed.

        Args:
            name: Project-local name originally passed to :meth:`upload`.

        Example:
            >>> uploads.delete("users")  # doctest: +SKIP
        """
        delete_uploaded_table(name, self._file_type,
                              project_id=self._kumo_project_id)
