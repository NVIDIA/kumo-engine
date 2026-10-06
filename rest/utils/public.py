# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from http import HTTPStatus
from typing import Dict, Generator, List, Optional, Type

from fastapi import HTTPException
from pydantic import ValidationError
from temporalio.exceptions import TimeoutError

from async_utils.sync import blocking_func
from common_utils.error import PublicException
from io_utils import fsutil, logging
from kumo.platform_errors import (
    CORE_PLATFORM_ERROR_TYPES,
    as_platform_public_exception,
)
from kumo.validation import ValidationException
from rest.deps.db_common import OptimisticLockError
from rest.utils.routers import PydanticValidationHTTPException
from service_framework.deploy_env_config import deploy_env
from statedb.utils.errors import (
    EntityDoesNotExistError,
    ExternalMembershipUnavailableError,
    ForbiddenError,
    GraphDefinitionAlreadyExists,
    UnsupportedInDiskgraphError,
)
from temporalio_commons import RpcError

log = logging.getLogger(__name__)

ERROR_TO_HTTP_STATUS = {
    KeyError: HTTPStatus.NOT_FOUND,
    # EntityDoesNotExistError subclasses ValueError; the MRO walk in
    # wrap_driver_errors matches the most-derived base first, so a missing
    # table/graph/pquery resolves to 404 rather than the 422 of the ValueError
    # entry below.
    EntityDoesNotExistError: HTTPStatus.NOT_FOUND,
    ValueError: HTTPStatus.UNPROCESSABLE_ENTITY,
    ValidationError: HTTPStatus.UNPROCESSABLE_ENTITY,
    ValidationException: HTTPStatus.UNPROCESSABLE_ENTITY,
    TimeoutError: HTTPStatus.REQUEST_TIMEOUT,
    OptimisticLockError: HTTPStatus.CONFLICT,  # 409
    ForbiddenError: HTTPStatus.FORBIDDEN,
    ExternalMembershipUnavailableError: HTTPStatus.SERVICE_UNAVAILABLE,
    GraphDefinitionAlreadyExists: HTTPStatus.UNPROCESSABLE_ENTITY,
    # Without this mapping the wrap_rpc_error fallthrough returns 500,
    # misclassifying "not yet supported in DiskGraph" as a server fault
    # instead of 422.
    UnsupportedInDiskgraphError: HTTPStatus.UNPROCESSABLE_ENTITY,
}

ERROR_TO_CUSTOM_FORMAT = {
    TimeoutError:
    ("{0}. Please try again, or contact Kumo for further assistance if the "
     "problem persists."),
    OptimisticLockError:
    ("{0}. The resource was modified by another request. Please retry your "
     "operation."),
}

ERROR_NAME_TO_CUSTOM_FORMAT = {
    k.__name__: v
    for k, v in ERROR_TO_CUSTOM_FORMAT.items()
}

ERROR_NAME_TO_HTTP_STATUS = {
    "ProjectForbiddenError": HTTPStatus.FORBIDDEN,
    "ExternalMembershipUnavailableError": HTTPStatus.SERVICE_UNAVAILABLE,
}


@contextmanager
def wrap_rpc_error(
    custom_error_to_status: Optional[Dict[Type[Exception], HTTPStatus]] = None
) -> Generator[None, None, None]:
    """Catches all RpcError, and wraps them with HTTPExceptions.

    Use this for routes whose driver is still an AutoRPC client (the
    MLJobBackend job paths). In-process metadata routes that raise native
    Python exceptions should use ``wrap_driver_errors`` instead.
    """
    custom_error_to_status = custom_error_to_status or {}
    error_to_status = {**ERROR_TO_HTTP_STATUS, **custom_error_to_status}
    error_name_to_status = {k.__name__: v for k, v in error_to_status.items()}
    try:
        yield
    except RpcError as rpc_err:
        log.error(f'RpcError occured: {rpc_err}')
        http_status = error_name_to_status.get(
            rpc_err.error_type,
            ERROR_NAME_TO_HTTP_STATUS.get(rpc_err.error_type,
                                          HTTPStatus.INTERNAL_SERVER_ERROR),
        )
        custom_format = ERROR_NAME_TO_CUSTOM_FORMAT.get(rpc_err.error_type,
                                                        None)
        message = rpc_err.message
        if custom_format:
            message = custom_format.format(message)
        raise HTTPException(http_status, message) from rpc_err


@contextmanager
def wrap_driver_errors(
    custom_error_to_status: Optional[Dict[Type[Exception], HTTPStatus]] = None
) -> Generator[None, None, None]:
    """Map in-process driver errors to HTTPExceptions.

    Routes backed by in-process metadata drivers raise native Python
    exceptions; this maps them through the shared status tables to HTTP
    codes/messages. It also maps ``RpcError`` (so it is safe for a route whose
    driver may transitively make a remote call). Routes that are purely AutoRPC
    (MLJobBackend job paths) can keep using ``wrap_rpc_error``.

    ``PublicException`` and ``OptimisticLockError`` are re-raised so their
    dedicated FastAPI handlers render the richer JSON body, unless the route
    supplies an explicit ``custom_error_to_status`` for the type (which wins).
    Pydantic validation errors become sanitized 422 responses.
    """
    custom_error_to_status = custom_error_to_status or {}
    error_to_status = {**ERROR_TO_HTTP_STATUS, **custom_error_to_status}
    error_name_to_status = {k.__name__: v for k, v in error_to_status.items()}
    try:
        yield
    except RpcError as rpc_err:
        log.error(f'RpcError occured: {rpc_err}')
        http_status = error_name_to_status.get(
            rpc_err.error_type,
            ERROR_NAME_TO_HTTP_STATUS.get(rpc_err.error_type,
                                          HTTPStatus.INTERNAL_SERVER_ERROR),
        )
        custom_format = ERROR_NAME_TO_CUSTOM_FORMAT.get(rpc_err.error_type,
                                                        None)
        message = rpc_err.message
        if custom_format:
            message = custom_format.format(message)
        raise HTTPException(http_status, message) from rpc_err
    except HTTPException:
        # Already mapped (e.g. by a nested context manager or the route
        # itself); let it propagate untouched.
        raise
    except CORE_PLATFORM_ERROR_TYPES as core_error:
        platform_error = as_platform_public_exception(core_error)
        assert platform_error is not None
        raise platform_error from None
    except ValidationError as native_err:
        raise PydanticValidationHTTPException(native_err) from native_err
    except tuple(custom_error_to_status) as custom_err:
        # A route's explicit custom_error_to_status mapping wins over the
        # PublicException/OptimisticLockError re-raise below, so a route can
        # force a specific status for a type it owns (e.g. ConnectorError, a
        # PublicException subclass, -> 422). Without this, such a type would be
        # caught by the PublicException branch and the route's mapping ignored.
        # Match by MRO so subclasses inherit the route's mapping.
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR
        for base in type(custom_err).__mro__:
            if base in custom_error_to_status:
                http_status = custom_error_to_status[base]
                break
        raise HTTPException(http_status, str(custom_err)) from custom_err
    except (PublicException, OptimisticLockError):
        # The main and public REST apps register dedicated FastAPI handlers for
        # these (richer JSON bodies); re-raise so those handlers run unchanged.
        raise
    except tuple(error_to_status) as native_err:
        # Native exception from an in-process driver with no dedicated FastAPI
        # handler (e.g. ValueError, KeyError, TimeoutError, EntityDoesNotExist).
        # Resolve the status by walking the MRO so subclasses inherit their
        # base's mapping (e.g. EntityDoesNotExistError -> 404 before its
        # ValueError base -> 422). Without this the bare exception would 500.
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR
        matched_type: Optional[Type[Exception]] = None
        for base in type(native_err).__mro__:
            if base in error_to_status:
                http_status = error_to_status[base]
                matched_type = base
                break
            if base.__name__ in ERROR_NAME_TO_HTTP_STATUS:
                http_status = ERROR_NAME_TO_HTTP_STATUS[base.__name__]
                matched_type = base
                break
        custom_format = (ERROR_TO_CUSTOM_FORMAT.get(matched_type)
                         if matched_type is not None else None)
        message = str(native_err)
        if custom_format:
            message = custom_format.format(message)
        log.error('Driver error occured: '
                  f'{type(native_err).__name__}: {message}')
        raise HTTPException(http_status, message) from native_err


@blocking_func()
def presign_parquet_table_data(path: str) -> List[str]:
    helper = fsutil.get_presigned_helper(path)
    if helper is not None and fsutil.isdir(path):
        pending, parquet_files = [path], []
        num_entries = 0
        while pending:
            for entry in fsutil.listdir(pending.pop(), max_items=201):
                num_entries += 1
                if num_entries > 200:
                    raise ValueError(
                        'Prediction-table readback exceeds 200 entries')
                if fsutil.isdir(entry):
                    pending.append(entry)
                elif entry.casefold().endswith('.parquet'):
                    parquet_files.append(entry)
        if not parquet_files:
            raise ValueError('Prediction-table readback found no Parquet data')
        return [helper.generate_download_url(f) for f in parquet_files]

    if helper is not None:
        return [helper.generate_download_url(path)]
    return [fsutil.customer_facing_path(path)]


def get_base_kumo_url() -> str:
    external_host = deploy_env().external_host
    if external_host:
        return f'https://{external_host}'
    return 'http://localhost'
