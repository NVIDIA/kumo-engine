# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from http import HTTPStatus
from typing import List

from fastapi import HTTPException

from rest.utils.routers import convert_500_http_error_to_503


def test_convert_500_to_503():
    # Test case no exception thrown
    with convert_500_http_error_to_503():
        x = 1
    assert x == 1

    # Test cases where exception is thrown
    @dataclass
    class TestCase:
        name: str
        exc: Exception
        expect_status: HTTPStatus
        expect_msg: str

    cases: List[TestCase] = [
        TestCase(
            name="400 Error",
            exc=HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "Invalid data"),
            expect_status=HTTPStatus.UNPROCESSABLE_ENTITY,
            expect_msg="Invalid data",
        ),
        TestCase(
            name="500 Error",
            exc=HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR,
                              "Internal server error"),
            expect_status=HTTPStatus.SERVICE_UNAVAILABLE,
            expect_msg="Internal server error",
        ),
        TestCase(
            name="Uncaught exception",
            exc=RuntimeError("Runtime error"),
            expect_status=HTTPStatus.SERVICE_UNAVAILABLE,
            expect_msg="Runtime error",
        ),
    ]

    for case in cases:
        print(f"Test case: {case.name}")

        try:
            with convert_500_http_error_to_503():
                raise case.exc
            assert False, "Test case should throw exception"
        except HTTPException as e:
            assert e.status_code == case.expect_status
            assert case.expect_msg in e.detail
        except Exception as e:
            assert False, f"Exception must be HTTPException but got {e}"
