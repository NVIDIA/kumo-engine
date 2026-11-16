# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type, cast

from temporalio.client import WorkflowFailureError
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    ServerError,
    TimeoutError,
)
from temporalio.service import RPCError as TemporalRPCError
from temporalio.service import RPCStatusCode


# Subclass from RetriableError to declare custom error types as retriable.
class RetriableError(Exception):
    pass


# TODO(siyang): Capture the fully qualified exception type name so that
# client side can locate the actual exception class.
class RpcError(Exception):
    rpc_service: str
    rpc_method: str
    error_type: str
    message: str
    stack_trace: str
    non_retryable: bool

    def __init__(self, rpc_service, rpc_method, error_type, message,
                 stack_trace, non_retryable) -> None:
        self.rpc_service = rpc_service
        self.rpc_method = rpc_method
        self.error_type = error_type
        self.message = message
        self.stack_trace = stack_trace
        self.non_retryable = non_retryable

    def __str__(self):
        return (f'RPC {self.rpc_service}.{self.rpc_method} failed '
                f'due to {self.error_type}, message: {self.message}, \n'
                f'stack trace:\n{self.stack_trace}')


class _AutoRpcNonRetriableError(Exception):
    error_type: Type[Exception]
    error_message: str


# AutoRPC client-side implementation to convert a workflow failure error to
# AutoRPC's RPCError.
def wf_failure_to_rpc_error(service: str, method: str,
                            wf_failure: WorkflowFailureError):
    cause = wf_failure.cause
    error_type = type(cause).__name__
    message = str(cause)
    stack_trace = ''
    non_retryable = False

    # If workflow failure is caused by ActivityError, unwrap the cause
    if isinstance(cause, ActivityError):
        cause = cause.cause

    # ApplicationError wraps the exception raised from RPC handler on the
    # server side.
    if isinstance(cause, ApplicationError):
        non_retryable = cause.non_retryable
        # See `wrap_non_retriable_exception()` below, any non retriable
        # exception raised from RPC handler function will be wrapped as
        # `_AutoRpcNonRetriableError`, so we unwrap the cause again.
        if cause.type == _AutoRpcNonRetriableError.__name__:
            non_retryable = True
            cause = cast(ApplicationError, cause.cause)

        error_type = cause.type
        message = cause.message
        if cause.failure:
            stack_trace = cause.failure.stack_trace

    raise RpcError(service, method, error_type, message, stack_trace,
                   non_retryable) from cause


# Categorize if an exception raised from RPC function (service side
# implementation) could be retried.
def is_retryable(ex: BaseException):
    # Temporalio-related errors that are retriable.
    if (isinstance(ex, TimeoutError)
            or (isinstance(ex, (ServerError, ApplicationError))
                and not ex.non_retryable)):  # type: ignore
        return True

    # RuntimeError and Custom subclass of RetriableError are considered
    # as retryable.
    if isinstance(ex, (RetriableError, RuntimeError)):
        return True

    if isinstance(ex, TemporalRPCError):
        # The following gRPC status codes will be retried.
        return ex.status in [
            RPCStatusCode.ABORTED,
            RPCStatusCode.DATA_LOSS,
            RPCStatusCode.DEADLINE_EXCEEDED,
            RPCStatusCode.INTERNAL,
            RPCStatusCode.RESOURCE_EXHAUSTED,
            RPCStatusCode.UNAVAILABLE,
            RPCStatusCode.CANCELLED,
            RPCStatusCode.UNKNOWN,
        ]

    if isinstance(ex, RpcError):
        # Propagate (non-)retryable state if we catch a RPCError raised from
        # calling another AutoRPC service.
        return not ex.non_retryable

    # TODO(siyang): Monitor and add more common retriable errors here.

    return False
