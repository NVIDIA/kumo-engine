# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from contextlib import contextmanager
from http import HTTPStatus
from typing import Callable, Generator, Optional, Sequence, Type

from fastapi import APIRouter, HTTPException, Request
from fastapi.routing import APIRoute
from pydantic import ValidationError

from async_utils.caching import fastapi_cached
from common_utils.error import PublicException
from io_utils import kumo_dataclass, logging
from kumo.platform_errors import (
    CORE_PLATFORM_ERROR_TYPES,
    as_platform_public_exception,
)
from kumo.validation import ValidationException
from statedb.utils.errors import (
    AuthenticationError,
    ExternalMembershipUnavailableError,
    ForbiddenError,
    NotSupportedError,
    UnauthorizedError,
)

logger = logging.getLogger(__name__)

ORGANIZATION_NOT_FOUND = 'Organization must exist'


class StrictQueryAPIRoute(APIRoute):
    """Reject query keys not declared by the selected FastAPI endpoint.

    FastAPI otherwise ignores unknown query keys, which can make removed
    authorization or ownership fields appear to have taken effect. Validation
    happens before body parsing and applies generically to every route using
    this route class.
    """
    def get_route_handler(self) -> Callable:
        original_handler = super().get_route_handler()

        def query_names(dependant) -> set[str]:
            names = {field.alias for field in dependant.query_params}
            for dependency in dependant.dependencies:
                names.update(query_names(dependency))
            return names

        allowed = frozenset(query_names(self.dependant))

        async def strict_handler(request: Request):
            unknown = sorted(set(request.query_params) - allowed)
            if unknown:
                names = ', '.join(unknown)
                raise HTTPException(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    f'Unexpected query parameter(s): {names}',
                )
            return await original_handler(request)

        return strict_handler


@contextmanager
def wrap_auth_exceptions() -> Generator[None, None, None]:
    """ Catches all AuthenticationErrors, and wraps them with
    ValidationHTTPExceptions. This is intended to be used to catch errors that
    are intentionally raised when when authenticating and authorizing a user.
    """
    try:
        yield
    except AuthenticationError as err:
        if isinstance(err, ExternalMembershipUnavailableError):
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, str(err))
        if isinstance(err, UnauthorizedError):
            raise HTTPException(HTTPStatus.UNAUTHORIZED, str(err))
        if isinstance(err, ForbiddenError):
            raise HTTPException(HTTPStatus.FORBIDDEN, str(err))
        if isinstance(err, NotSupportedError):
            raise HTTPException(HTTPStatus.NOT_FOUND, str(err))
        if str(err) == ORGANIZATION_NOT_FOUND:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(err))
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, str(err))


@contextmanager
def wrap_validation_exceptions(
    not_exists_error: Optional[Type[ValueError]] = None,
    failure_callback: Callable = lambda *args: None
) -> Generator[None, None, None]:
    """Map user-input validation failures to HTTP errors.

    Pydantic validation errors become sanitized 422 responses; a matching
    ``not_exists_error`` becomes a 404. Other value and validation errors
    become 422 responses.

    Args:
        not_exists_error: Error class that should become an HTTP 404.
        failure_callback: Callback invoked for mapped validation failures.
    """
    try:
        yield
    except KeyError as err:
        raise HTTPException(HTTPStatus.NOT_FOUND, str(err)) from err
    except ValidationError as err:
        failure_callback()
        raise PydanticValidationHTTPException(err) from err
    except CORE_PLATFORM_ERROR_TYPES as core_error:
        platform_error = as_platform_public_exception(core_error)
        assert platform_error is not None
        if isinstance(platform_error, ValidationException):
            raise ValidationHTTPException(
                platform_error.title_and_message(),
                platform_error,
            ) from core_error
        failure_callback()
        raise ValidationHTTPException(
            str(platform_error),
            platform_error,
        ) from core_error
    except ValueError as err:
        if not_exists_error is not None and isinstance(err, not_exists_error):
            raise HTTPException(HTTPStatus.NOT_FOUND, str(err)) from err
        failure_callback()
        raise ValidationHTTPException(str(err), err) from err
    except ValidationException as err:
        raise ValidationHTTPException(err.title_and_message(), err)
    except PublicException as err:
        failure_callback()
        raise ValidationHTTPException(str(err), err) from err


@contextmanager
def convert_500_http_error_to_503() -> Generator[None, None, None]:
    """Converts all 500 errors to 503 errors, indicating to the client that
    it is safe to retry the request. This context manager should only be used
    on read-only endpoints.
    """
    try:
        yield
    except PublicException:
        raise
    except CORE_PLATFORM_ERROR_TYPES as core_error:
        platform_error = as_platform_public_exception(core_error)
        assert platform_error is not None
        raise platform_error from None
    except HTTPException as err:
        logger.exception(err)
        if err.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
            err.status_code = HTTPStatus.SERVICE_UNAVAILABLE
        raise err from err
    except Exception as err:
        logger.exception(err)
        # Also convert all uncaught exceptions to 503 (they would normally be
        # converted automatically to 500 by the FastAPI framework.
        raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, str(err))


class PydanticValidationHTTPException(BaseException):
    """Convert a Pydantic error to a 422 with sanitized messages only."""
    def __new__(cls, error: ValidationError) -> HTTPException:
        error_messages = [err['msg'] for err in error.errors()]
        message = '\n'.join(error_messages)
        return HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, message)


class ValidationHTTPException(HTTPException):
    """Raise an exception that should result in an error message being shown to
    the user in the UI. For convenience, the error message is logged to stderr
    as well.
    """
    def __init__(self, msg: str, error: Optional[Exception] = None) -> None:
        logger.info(
            "User input validation error: %s (error_type=%s)",
            msg,
            type(error).__name__ if error is not None else "none",
        )
        return super().__init__(HTTPStatus.UNPROCESSABLE_ENTITY, msg)


@kumo_dataclass
class ValidateNameResponse:
    is_valid: bool
    error_msg: Optional[str]


def nullable_response(status_code: int = 200) -> dict:
    """
    Returns a FastAPI response configuration dict that marks the response as nullable.
    This is useful for endpoints that may return None.

    Returns a new dict on each call to avoid mutation issues.

    Args:
        status_code: HTTP status code for the response (default: 200)

    Returns:
        A dict with the nullable response schema

    Example:
        self.router.add_api_route(
            "/my_endpoint",
            self.my_handler,
            response_model=MyModel,
            methods=['GET'],
            responses=nullable_response(),
        )
    """
    return {
        status_code: {
            "content": {
                "application/json": {
                    "schema": {
                        "nullable": True
                    }
                }
            }
        }
    }


def enable_api_caching(
    router: APIRouter,
    *,
    method: str,
    ttl_sec: Optional[int] = None,
    include_paths: Optional[Sequence[str]] = None,
    exclude_paths: Optional[Sequence[str]] = None,
):
    """
    Convenient helper to apply `@fastapi_cached` decorator to multiple API
    routes under the router in batch.
    """
    for api_route in router.routes:
        if isinstance(api_route, APIRoute) and method in api_route.methods:
            path = api_route.path
            if path.startswith(router.prefix):  # strip prefix for matching
                path = path[len(router.prefix):]
            if include_paths is not None and path not in include_paths:
                continue
            if exclude_paths is not None and path in exclude_paths:
                continue

            handler_fn = api_route.dependant.call
            if handler_fn and inspect.iscoroutinefunction(handler_fn):
                logger.info('Enable API caching for %s', api_route.path)
                cache_decorator = fastapi_cached(ttl_sec=ttl_sec)
                api_route.dependant.call = cache_decorator(handler_fn)
