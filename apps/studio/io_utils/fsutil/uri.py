# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from typing import TypeAlias, Union
from urllib.parse import ParseResult, quote, unquote, urlparse

from pydantic import BaseModel, field_validator


class LocalURI(BaseModel):
    scheme: str = 'local'
    path: str

    def as_url(self) -> str:
        return f"file:{osp.abspath(self.path)}"


# In pydantic v2 reusable field validators receive (cls, value); the leading
# ``cls`` is unused here but required by the field_validator protocol.
def _no_slash(cls, field: str) -> str:
    if "/" in field:
        raise ValueError(f"expect no / in field, got: {field}")
    return field


def _no_leading_slash(cls, field: str) -> str:
    if field.startswith("/"):
        raise ValueError(f"expected no leading / in field, got: {field}")
    return field


class S3URI(BaseModel):
    scheme: str = 's3'
    bucket: str
    path: str

    _validate_bucket = field_validator('bucket')(_no_slash)
    _validate_path = field_validator('path')(_no_leading_slash)

    def as_s3(self) -> str:
        path = quote(self.path, safe="/-._~!$&'()*+,;=:@")
        return f's3://{self.bucket}/{path}'

    def as_s3a(self) -> str:
        path = quote(self.path, safe="/-._~!$&'()*+,;=:@")
        return f's3a://{self.bucket}/{path}'

    def file_bucket_key(self) -> tuple[str, str]:
        if len(self.path) == 0 or self.path.endswith('/'):
            raise ValueError(
                f"S3 path must point to a file (no trailing slash), "
                f"but got {self.as_s3()})")
        return self.bucket, self.path

    def dir_bucket_key(self) -> tuple[str, str]:
        path = self.path
        if len(path) > 0 and not path.endswith('/'):
            path += '/'
        return self.bucket, path


class GCSURI(BaseModel):
    scheme: str = 'gs'
    bucket: str
    path: str

    _validate_bucket = field_validator('bucket')(_no_slash)
    _validate_path = field_validator('path')(_no_leading_slash)

    def as_gs(self) -> str:
        return f'gs://{osp.join(self.bucket, self.path)}'

    def file_bucket_key(self) -> tuple[str, str]:
        if len(self.path) == 0 or self.path.endswith('/'):
            raise ValueError(
                f"GCS path must point to a file (no trailing slash), "
                f"but got {self.as_gs()})")
        return self.bucket, self.path

    def dir_bucket_key(self) -> tuple[str, str]:
        path = self.path
        if len(path) > 0 and not path.endswith('/'):
            path += '/'
        return self.bucket, path


class UCURI(BaseModel):
    scheme: str = 'dbfs'
    catalog: str
    uc_schema: str
    volume: str
    path: str

    _validate_catalog = field_validator('catalog')(_no_slash)
    _validate_uc_schema = field_validator('uc_schema')(_no_slash)
    _validate_volume = field_validator('volume')(_no_slash)
    _validate_path = field_validator('path')(_no_leading_slash)

    @staticmethod
    def from_parse_result(o: ParseResult) -> 'UCURI':
        _, vol, catalog, schema, volume, path = o.path.split("/", maxsplit=5)
        if vol != "Volumes":
            raise ValueError("expect UC Path beginning with Volumes, "
                             f"got {vol}")
        return UCURI(catalog=catalog, uc_schema=schema, volume=volume,
                     path=path)

    def as_path(self) -> str:
        return (f"/Volumes/{self.catalog}/{self.uc_schema}"
                f"/{self.volume}/{self.path}")

    def as_url(self) -> str:
        return f"dbfs:{self.as_path()}"


class ADLSURI(BaseModel):
    """Azure Data Lake Storage Gen2 URI
    Format: abfss://container@storageaccount.dfs.core.windows.net/path
    """
    storage_account: str
    endpoint_suffix: str = 'dfs.core.windows.net'
    container: str
    path: str
    scheme: str = 'abfss'

    _validate_storage_account = field_validator('storage_account')(_no_slash)
    _validate_container = field_validator('container')(_no_slash)

    @field_validator('path')
    @classmethod
    def _normalize_path(cls, v: str) -> str:
        # Strip leading slashes so path is always relative. Other URI classes
        # reject leading slashes; ADLS normalises instead because urlparse
        # includes a leading slash in o.path that callers should not have to
        # strip manually before construction.
        return v.lstrip('/')

    def as_url(self) -> str:
        base = (f"abfss://{self.container}@{self.storage_account}"
                f".{self.endpoint_suffix}")
        return f"{base}/{self.path}" if self.path else base

    def as_adls(self) -> str:
        """Alias for as_url()."""
        return self.as_url()

    def as_path(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.as_url()

    @classmethod
    def from_parse_result(cls, o: ParseResult) -> 'ADLSURI':
        if "@" not in o.netloc:
            raise ValueError(
                f"Invalid ADLS URI format, expected container@storageaccount: {o.netloc}"
            )
        container, account_part = o.netloc.split("@", 1)
        if '.' not in account_part:
            raise ValueError(
                f"Invalid ADLS endpoint, expected account.endpoint: {account_part}"
            )
        storage_account, endpoint_suffix = account_part.split('.', 1)
        if not endpoint_suffix.startswith('dfs.'):
            raise ValueError(f'Invalid ADLS endpoint suffix: {endpoint_suffix}')
        return cls(
            storage_account=storage_account,
            endpoint_suffix=endpoint_suffix,
            container=container,
            path=o.path,  # _normalize_path validator handles lstrip
            scheme=o.scheme,
        )


AnyURI: TypeAlias = Union[LocalURI, S3URI, GCSURI, UCURI, ADLSURI]


def ParseURI(raw: str) -> AnyURI:
    if not isinstance(raw, str):
        raise ValueError(f"trying to parse a URI from a non-string, "
                         f"check stack trace for dispatch issues: {raw}")

    o = urlparse(raw)

    if o.scheme == "s3" or o.scheme == "s3a":
        return S3URI(bucket=o.netloc,
                     path=unquote(o.path.lstrip("/"), errors='strict'))
    elif o.scheme == "gs":
        return GCSURI(bucket=o.netloc, path=o.path.lstrip("/"))
    elif o.scheme == "dbfs":
        return UCURI.from_parse_result(o)
    elif o.scheme == "abfss" or o.scheme == "abfs":
        return ADLSURI.from_parse_result(o)
    elif o.scheme != "":
        raise ValueError(f"unknown scheme {o.scheme} for path {raw}")

    return LocalURI(path=o.path)
