# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.jobs import ErrorDetail, ErrorDetails, ErrorType

from kumoai_studio.formatting import pretty_print_error_details


def test_error_formatting_single_error():
    result = pretty_print_error_details(
        ErrorDetails(items=[
            ErrorDetail(
                type=ErrorType.ERROR,
                description='GPU error caused by OOM',
                title='Training error',
                cta=None,
            )
        ]))
    assert result is not None
    assert 'Training error: GPU error caused by OOM\n' in result


def test_error_formatting_multiple_errors():
    result = pretty_print_error_details(
        ErrorDetails(items=[
            ErrorDetail(
                type=ErrorType.ERROR,
                description='GPU error caused by OOM',
                title='Training error',
                cta=None,
            ),
            ErrorDetail(
                type=ErrorType.ERROR,
                description='GPU error caused by Dataloader',
                title='Training error',
                cta=None,
            )
        ]))
    assert result is not None
    assert 'Training error: GPU error caused by OOM' in result
    assert 'Training error: GPU error caused by Dataloader' in result
