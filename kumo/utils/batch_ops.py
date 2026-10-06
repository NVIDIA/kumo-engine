# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper functions for writing batch functions (eg. functions that operate over
# lists of elements). Batch functions are critical in latency-sensitive code.
# Generally, it is recommended for batch functions to have the following
# function signatures, depending on the business logic:
#
#   def always_succeed(inputs: List[T]) -> List[U]:
#   def sometimes_fail(inputs: List[T]) -> List[Optional[U]]:
#   def fail_with_error(inputs: List[T]) -> List[Union[U, Exception]]:
#
# Developers can use the functions in this package to convert between the
# various type signatures. For example:
#
#   remove_nones: List[Optional[U]] -> List[U]
#   replace_nones: List[Optional[U]] -> List[U]
#   log_errors: List[Union[U, Exception]] -> List[Optional[U]]
#   raise_errors: List[Union[U, Exception]] -> List[U]
#   insert_nones: List[U] -> List[Optional[U]]
#

import inspect
import logging
from typing import List, Optional, Tuple, TypeVar, Union

log = logging.getLogger(__name__)

T = TypeVar('T')


def remove_nones(input: List[Optional[T]]) -> Tuple[List[T], List[int]]:
    """Removes Nones from a list, and returns a list of indexes where the
    none_idxs were removed from. This is useful when calling a Batch API
    function that only accepts non-optional inputs.
    """
    result: List[T] = []
    none_idxs: List[int] = []
    for idx, element in enumerate(input):
        if element is None:
            none_idxs.append(idx)
        else:
            result.append(element)
    return result, none_idxs


def insert_nones(input: Union[List[T], List[Optional[T]]],
                 none_idxs: List[int]) -> List[Optional[T]]:
    """Insert None entries such that the output list has None entries at the
    given set of indexes. This is generally called after `remove_nones`, in
    order to restore a list to its original length after passing it to another
    batch API function.

    Args:
        none_idxs: must be sorted
    """
    result: List[Optional[T]] = []
    result.extend(input)
    for idx in none_idxs:
        result.insert(idx, None)
    return result


def replace_nones(input: List[Optional[T]], default: T) -> List[T]:
    """Insert the given default value for all items that are None.
    """
    result: List[T] = []
    for item in input:
        if item is None:
            result.append(default)
        else:
            result.append(item)
    return result


def log_errors(input: List[Union[T, Exception]]) -> List[Optional[T]]:
    """Convert all failed entries in the list to `None`. For use in batch APIs
    that return error message on failed items. Log the first 5 errors as WARN.

    `T` must not be a subtype of `Exception`
    """
    result: List[Optional[T]] = []
    errors: List[Exception] = []
    for element in input:
        if isinstance(element, Exception):
            errors.append(element)
            result.append(None)
        else:
            result.append(element)
    if len(errors) > 0:
        n_errors = len(errors)
        errors_to_log = errors[:5]
        # Print the line of code that called this function for easier debug
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            func = frame.f_back.f_code
            log.warn(
                "Encountered %s errors at %s in %s:%i: %s",
                n_errors,
                func.co_name,
                func.co_filename,
                func.co_firstlineno,
                errors_to_log,
            )
        else:
            log.warn(
                "Encountered %s errors: %s",
                n_errors,
                errors_to_log,
            )
    return result


def raise_errors(input: List[Union[T, Exception]], message: str) -> List[T]:
    """Raises an exception of any of the inputs are an exception.

    `T` must not be a subtype of `Exception`
    """
    result: List[T] = []
    for element in input:
        if isinstance(element, Exception):
            raise Exception(message) from element
        else:
            result.append(element)
    return result
