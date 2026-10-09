# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from kumo_core.errors import PublicError, TrainingValidationError
from pydantic import BaseModel

from common_utils.error import PublicException
from kumo.validation import ValidationException
from rest.utils.public import (
    get_base_kumo_url,
    presign_parquet_table_data,
    wrap_driver_errors,
    wrap_rpc_error,
)
from rest.utils.routers import wrap_validation_exceptions
from temporalio_commons import RpcError


class _SamplePublicError(PublicException):
    template = "boom: {reason}"
    http_code = 422
    non_retryable = True


class _ValidatedInput(BaseModel):
    count: int


@pytest.mark.parametrize('wrapper',
                         [wrap_driver_errors, wrap_validation_exceptions])
def test_pydantic_errors_are_clean_422s(wrapper):
    with pytest.raises(HTTPException) as error:
        with wrapper():
            _ValidatedInput(count=cast(int, 'not-an-integer'))

    assert error.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert 'valid integer' in error.value.detail
    assert 'input_value' not in error.value.detail
    assert 'errors.pydantic.dev' not in error.value.detail


def test_wrap_driver_errors_custom_mapping_overrides_public_exception():
    # A route that explicitly maps a PublicException subclass via
    # custom_error_to_status must get that status as an HTTPException, instead
    # of having the exception re-raised to the app-level PublicException
    # handler (which would ignore the route's mapping and change the body
    # shape). Regression guard for the PublicException branch shadowing the
    # route's mapping.
    with pytest.raises(HTTPException) as exc_info:
        with wrap_driver_errors({_SamplePublicError: HTTPStatus.BAD_REQUEST}):
            raise _SamplePublicError(reason='x')

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST


def test_wrap_driver_errors_reraises_unmapped_public_exception():
    # Without a custom mapping, a PublicException must propagate untouched so
    # the app-level PublicException handler renders its richer body (it is not
    # converted to a bare HTTPException here).
    with pytest.raises(_SamplePublicError):
        with wrap_driver_errors():
            raise _SamplePublicError(reason='x')


def test_wrap_driver_errors_restores_core_public_error_type():
    with pytest.raises(PublicException) as exc_info:
        with wrap_driver_errors():
            raise PublicError('unsupported operation')

    assert type(exc_info.value) is PublicException
    assert exc_info.value.get_user_message() == 'unsupported operation'


def test_wrap_validation_exceptions_restores_validation_message_shape():
    with pytest.raises(HTTPException) as exc_info:
        with wrap_validation_exceptions():
            raise TrainingValidationError(
                title='GPU out-of-memory error',
                message='Reduce output_embedding_dim.',
            )

    assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert exc_info.value.detail == (
        'GPU out-of-memory error - Reduce output_embedding_dim.')


def test_wrap_driver_errors_maps_entity_not_found_to_404():
    # EntityDoesNotExistError subclasses (TableDoesNotExistError, etc.) subclass
    # ValueError. The MRO walk must resolve them to 404 NOT_FOUND (via the
    # EntityDoesNotExistError entry) rather than 422 (the ValueError entry), so
    # a missing entity is not misreported as a validation error even when
    # wrap_driver_errors is nested inside wrap_validation_exceptions.
    from statedb.utils.errors import TableDoesNotExistError

    with pytest.raises(HTTPException) as exc_info:
        with wrap_driver_errors():
            raise TableDoesNotExistError('table foo does not exist')

    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND


def test_wrap_driver_errors_passes_through_http_exception():
    # An HTTPException raised by an inner context manager (e.g. a not-found
    # mapped to 404 by a wrap_validation_exceptions nested inside) must
    # propagate untouched rather than being re-mapped (e.g. to 422). Guards the
    # not-found-status ordering when wrap_driver_errors is the outer CM.
    with pytest.raises(HTTPException) as exc_info:
        with wrap_driver_errors():
            raise HTTPException(HTTPStatus.NOT_FOUND, 'missing')

    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND


def test_wrap_rpc_error_maps_validation_exception_to_422():
    rpc_error = RpcError(
        rpc_service='PQueryRpcService',
        rpc_method='ValidatePredictiveQuery',
        error_type=ValidationException.__name__,
        message='Invalid predictive query',
        stack_trace='traceback omitted',
        non_retryable=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        with wrap_rpc_error():
            raise rpc_error

    assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert exc_info.value.detail == 'Invalid predictive query'


class TestGetBaseKumoUrl:
    """Base-URL selection for a single-tenant deployment."""
    def test_external_host_set_builds_url_from_it(self):
        config = MagicMock()
        config.external_host = 'kumo-acme.dev.kumoai.cloud'
        with patch('rest.utils.public.deploy_env', return_value=config):
            assert get_base_kumo_url() == ('https://kumo-acme.dev.kumoai.cloud')

    def test_external_host_unset_falls_back_to_localhost(self):
        config = MagicMock()
        config.external_host = None
        with (
                patch('rest.utils.public.deploy_env', return_value=config),
        ):
            assert get_base_kumo_url() == 'http://localhost'


class TestPresignParquetTableData:
    """Unit tests for presign_parquet_table_data."""
    def test_s3_directory(self):
        """S3 directory should list parquet files and presign each."""
        mock_files = [
            's3://bucket/output/part-0.parquet',
            's3://bucket/output/date=2026',
        ]
        mock_helper = MagicMock()
        mock_helper.generate_download_url.side_effect = (
            lambda p: f'https://presigned/{p}')
        with (
                patch('rest.utils.public.fsutil.get_presigned_helper',
                      return_value=mock_helper),
                patch('rest.utils.public.fsutil.isdir',
                      side_effect=lambda path: not path.endswith('.parquet')),
                patch(
                    'rest.utils.public.fsutil.listdir', side_effect=[
                        mock_files,
                        ['s3://bucket/output/date=2026/part-1.parquet']
                    ]),
        ):
            result = presign_parquet_table_data('s3://bucket/output')

        assert len(result) == 2
        assert result[0] == (
            'https://presigned/s3://bucket/output/part-0.parquet')
        assert result[1] == (
            'https://presigned/s3://bucket/output/date=2026/part-1.parquet')

    def test_gcs_directory(self):
        """GCS directory should list parquet files and presign each."""
        mock_files = [
            'gs://bucket/output/part-0.parquet',
            'gs://bucket/output/part-1.parquet',
        ]
        mock_helper = MagicMock()
        mock_helper.generate_download_url.side_effect = (
            lambda p: f'https://signed/{p}')
        with (
                patch('rest.utils.public.fsutil.get_presigned_helper',
                      return_value=mock_helper),
                patch('rest.utils.public.fsutil.isdir',
                      side_effect=lambda path: path.endswith('/output')),
                patch('rest.utils.public.fsutil.listdir',
                      return_value=mock_files),
        ):
            result = presign_parquet_table_data('gs://bucket/output')

        assert len(result) == 2
        assert result[0] == 'https://signed/gs://bucket/output/part-0.parquet'
        assert result[1] == 'https://signed/gs://bucket/output/part-1.parquet'

    def test_single_file(self):
        """Single file path should presign directly without listing."""
        mock_helper = MagicMock()
        mock_helper.generate_download_url.side_effect = (
            lambda p: f'https://presigned/{p}')
        with (
                patch('rest.utils.public.fsutil.get_presigned_helper',
                      return_value=mock_helper),
                patch('rest.utils.public.fsutil.isdir', return_value=False),
        ):
            result = presign_parquet_table_data('s3://bucket/file.parquet')

        assert result == ['https://presigned/s3://bucket/file.parquet']

    def test_local_path(self):
        """Local path should be returned via as_url (no presigning)."""
        with (
                patch('rest.utils.public.fsutil.get_presigned_helper',
                      return_value=None),
                patch('rest.utils.public.fsutil.customer_facing_path',
                      side_effect=lambda p: p),
        ):
            result = presign_parquet_table_data('/tmp/output/data.parquet')

        assert result == ['/tmp/output/data.parquet']
