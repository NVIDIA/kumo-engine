# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from io_utils.fsutil.impl_local import LocalPresignedHelper
from io_utils.fsutil.uri import LocalURI
from rest.driver.connector_api_driver import (
    MAX_UPLOADED_FILE_SIZE,
    ConnectorApiDriver,
)
from rest.utils.file_upload import supports_backend_file_upload
from statedb.config.common import ProjectID


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True, scope="function")
def auto_cleanup_per_func():
    yield


def test_local_storage_supports_backend_file_upload(tmp_path):
    storage_root = str(tmp_path)
    assert supports_backend_file_upload(storage_root)


@pytest.mark.parametrize(
    ('file_size', 'exceeds_limit'),
    ((MAX_UPLOADED_FILE_SIZE, False), (MAX_UPLOADED_FILE_SIZE + 1, True)),
)
def test_uploaded_file_size_limit(file_size, exceeds_limit, monkeypatch):
    driver = object.__new__(ConnectorApiDriver)
    monkeypatch.setattr(
        'rest.driver.connector_api_driver.fsutil.filesize',
        lambda _: file_size,
    )

    if exceeds_limit:
        with pytest.raises(ValueError, match='100.0 GiB'):
            driver._validate_uploaded_file('table.csv', validate_data=False)
    else:
        driver._validate_uploaded_file('table.csv', validate_data=False)


def test_local_upload_part_urls_are_backend_relative():
    helper = LocalPresignedHelper(
        LocalURI(path="/tmp/kumo/temporary-uploaded-datasets/csv/table.csv"))

    upload_id, urls = helper._create_multipart_upload(
        LocalURI(path="/tmp/kumo/temporary-uploaded-datasets/csv/table.csv"), 2)

    assert set(urls) == {1, 2}
    assert urls[1].startswith("?")
    assert "temp_upload_path=" in urls[1]
    assert f"upload_id={upload_id}" in urls[1]
    assert "part_number=1" in urls[1]


def test_local_upload_writes_and_completes_parts(tmp_path, monkeypatch):
    storage_root = str(tmp_path)
    project_id = ProjectID.generate()
    temp_upload_path = str(tmp_path / "temporary-uploaded-datasets" /
                           str(project_id) / "table.csv")
    driver = object.__new__(ConnectorApiDriver)
    monkeypatch.setattr(
        "rest.driver.connector_api_driver.deploy_env",
        lambda: SimpleNamespace(data_plane_storage_root=storage_root),
    )
    monkeypatch.setattr(
        "service_framework.deploy_env_config.deploy_env",
        lambda: SimpleNamespace(data_plane_storage_root=storage_root),
    )

    upload_id = "upload-id"
    first_etag = driver._upload_file_part(project_id, temp_upload_path,
                                          upload_id, 1, b"hello ")
    second_etag = driver._upload_file_part(project_id, temp_upload_path,
                                           upload_id, 2, b"world")

    helper = LocalPresignedHelper(LocalURI(path=temp_upload_path))
    helper._complete_multipart_upload(
        LocalURI(path=temp_upload_path),
        upload_id,
        [
            SimpleNamespace(part_number=2, etag=second_etag),
            SimpleNamespace(part_number=1, etag=first_etag),
        ],
    )

    assert Path(temp_upload_path).read_bytes() == b"hello world"
    assert not Path(f"{temp_upload_path}.{upload_id}.part00001").exists()
    assert not Path(f"{temp_upload_path}.{upload_id}.part00002").exists()


@pytest.mark.asyncio
async def test_large_local_upload_part_does_not_serialize_body_for_dedupe(
    tmp_path,
    monkeypatch,
):
    storage_root = str(tmp_path)
    project_id = ProjectID.generate()
    temp_upload_path = str(tmp_path / "temporary-uploaded-datasets" /
                           str(project_id) / "table.csv")
    driver = object.__new__(ConnectorApiDriver)
    monkeypatch.setattr(
        "rest.driver.connector_api_driver.deploy_env",
        lambda: SimpleNamespace(data_plane_storage_root=storage_root),
    )

    body = b"x" * ((10 << 20) + 1)
    etag = await driver._upload_file_part.coroutine(project_id,
                                                    temp_upload_path,
                                                    "upload-id", 1, body)

    assert Path(f"{temp_upload_path}.upload-id.part00001").read_bytes() == body
    assert etag


def test_local_upload_rejects_another_projects_temporary_path(
    tmp_path,
    monkeypatch,
):
    storage_root = str(tmp_path)
    active_project_id = ProjectID.generate()
    other_project_id = ProjectID.generate()
    temp_upload_path = str(tmp_path / "temporary-uploaded-datasets" /
                           str(other_project_id) / "table.csv")
    driver = object.__new__(ConnectorApiDriver)
    monkeypatch.setattr(
        "rest.driver.connector_api_driver.deploy_env",
        lambda: SimpleNamespace(data_plane_storage_root=storage_root),
    )

    with pytest.raises(ValueError, match="active project's"):
        driver._upload_file_part(
            active_project_id,
            temp_upload_path,
            "upload-id",
            1,
            b"data",
        )
