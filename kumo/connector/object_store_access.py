# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

from kumo_core.sources.object_store_access import ObjectStoreAccess

from io_utils.fsutil.credentials import credentials_for_provider
from io_utils.fsutil.s3_connection import active_s3_connection
from io_utils.fsutil.uri import GCSURI, S3URI, LocalURI, ParseURI


def source_access(path: str) -> ObjectStoreAccess | None:
    uri = ParseURI(path)
    if isinstance(uri, S3URI):
        import boto3
        from kumo_s3.access import source_access as s3_access

        scope = credentials_for_provider('s3')
        connection = active_s3_connection()
        material = (scope.credentials or {}) if scope is not None else {}
        session = material.get('ambient_session')
        if session is not None:
            credentials = session.get_credentials()
            frozen = credentials.get_frozen_credentials(
            ) if credentials else None
            if not (frozen and frozen.access_key and frozen.secret_key
                    and frozen.token):
                raise RuntimeError(
                    'Attested ambient AWS session has no temporary credentials')
        if session is None:
            if scope is not None:
                if not (material.get('aws_access_key_id')
                        and material.get('aws_secret_access_key')):
                    raise RuntimeError('Missing explicit S3 credentials')
                session = boto3.session.Session(
                    aws_access_key_id=material['aws_access_key_id'],
                    aws_secret_access_key=material['aws_secret_access_key'],
                    aws_session_token=material.get('aws_session_token'),
                    region_name=(connection.region if connection else None)
                    or material.get('region'))
            elif connection is not None:
                session = boto3.session.Session(
                    aws_access_key_id=connection.access_key,
                    aws_secret_access_key=connection.secret_key,
                    aws_session_token=connection.session_token,
                    region_name=connection.region)
            else:
                from kumo_s3 import ambient_session
                session = ambient_session()
        return s3_access(
            uri.bucket, session=session,
            endpoint_url=connection.endpoint_url if connection else None,
            default_region=(connection.region if connection else None)
            or material.get('region'),
            options=connection.pyarrow_options() if connection else None,
            client_config=connection.boto_config if connection else None,
            verify=connection.boto_verify if connection else None,
            allow_anonymous=scope is None and connection is None)
    if isinstance(uri, GCSURI):
        from kumo_gcs.source_access import source_access as gcs_access

        scope = credentials_for_provider('gcs')
        material = (scope.credentials or {}) if scope is not None else {}
        credential = material.get('credential')
        project = material.get('project')
        if credential is None:
            if scope is not None:
                raise RuntimeError('Missing explicit GCS credentials')
            from google.auth import default
            credential, project = default(scopes=[
                'https://www.googleapis.com/auth/devstorage.read_write'
            ])
        return gcs_access(
            credentials=credential, project=project,
            requester_pays_project=material.get('requester_pays_project'))
    return None


def source_reader(path: str, *, file_format: str | None = None,
                  file_suffix: str | None = None):
    from kumo_core.sources.object_store import ObjectStoreSourceReader

    from io_utils import fsutil
    from io_utils.fsutil.uri import ADLSURI

    uri = ParseURI(path)
    if isinstance(uri, LocalURI):
        path = osp.abspath(uri.path)
    access = source_access(path)
    filesystem = access.filesystem if access else fsutil.pyarrow_filesystem(
        path)
    if isinstance(uri, (S3URI, GCSURI)):
        root = f'{uri.bucket}/{uri.path}'
    elif isinstance(uri, ADLSURI):
        root = uri.path
    else:
        root = path
    return ObjectStoreSourceReader(filesystem, root=root.rstrip('/'),
                                   file_format=file_format,
                                   file_suffix=file_suffix,
                                   fetch=access.fetch if access else None)
