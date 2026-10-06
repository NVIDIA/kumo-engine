// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render } from "@testing-library/react";
import { ErrorDetail, ErrorType } from "gen/v1/types";
import {
    hoverOverStatus,
    verifyTextDoesNotExist,
    verifyTextExists
} from "tests/testHelpers";

import { InputState } from "types/inputs";

import ValidationStatus, { ValidationStatusProps } from "../ValidationStatus";

const defaultProps: ValidationStatusProps = {
    state: InputState.None,
    errorInfo: []
};

describe("ValidationStatus", () => {
    const renderValidationStatus = async (props: ValidationStatusProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ValidationStatus {...props} />
                </MemoryRouter>
            );
        });
    };

    const createErrorDetail = (
        type: ErrorType,
        title: string,
        description: string
    ): ErrorDetail => ({
        type,
        title,
        description,
        cta: null
    });

    it.each([
        {
            description:
                "Should display emptyStateText when state is set as None",
            state: InputState.None,
            emptyStateText: "Empty Query",
            expectedText: "Empty Query"
        },
        {
            description:
                "Should display `Looks good!` when the state is set as Valid",
            state: InputState.Valid,
            emptyStateText: "",
            expectedText: "Looks good!"
        },
        {
            description:
                "Should display `Checking query` when the state is set as Validating",
            state: InputState.Validating,
            emptyStateText: "",
            expectedText: "Checking query"
        },
        {
            description:
                "Should display `Looks good!` when the state is set as Warning",
            state: InputState.Warning,
            emptyStateText: "",
            expectedText: "Looks good!"
        },
        {
            description:
                "Should display `Issues found` when the state is set as Invalid",
            state: InputState.InValid,
            emptyStateText: "",
            expectedText: "Issues found"
        },
        {
            description:
                "Should fall back to emptyStateText when state is ValidationError",
            state: InputState.ValidationError,
            emptyStateText: "Validation failed",
            expectedText: "Validation failed"
        }
    ])("$description", async ({ state, emptyStateText, expectedText }) => {
        await renderValidationStatus({
            ...defaultProps,
            state,
            emptyStateText
        });

        await verifyTextExists(expectedText);
    });

    it("Should display `Empty Query` by default", async () => {
        await renderValidationStatus({
            state: "" as unknown as InputState
        });

        await verifyTextExists("Empty Query");
    });

    it("Should filter error data correctly when errorInfo consists type 0 items", async () => {
        const errorInfo = [
            createErrorDetail(0 as ErrorType, "Error 1", "Error description 1"),
            createErrorDetail(
                1 as ErrorType,
                "Warning 1",
                "Warning description 1"
            ),
            createErrorDetail(0 as ErrorType, "Error 2", "Error description 2")
        ];

        await renderValidationStatus({
            ...defaultProps,
            state: InputState.InValid,
            errorInfo
        });

        await hoverOverStatus("Issues found");

        await verifyTextExists("Error 1");
        await verifyTextExists("Error 2");
        await verifyTextExists("Warning 1");
    });

    it("Should filter warning data correctly when errorInfo consists type 1 items", async () => {
        const errorInfo = [
            createErrorDetail(
                1 as ErrorType,
                "Warning 1",
                "Warning description 1"
            ),
            createErrorDetail(
                1 as ErrorType,
                "Warning 2",
                "Warning description 2"
            )
        ];

        await renderValidationStatus({
            ...defaultProps,
            state: InputState.Warning,
            errorInfo
        });

        await hoverOverStatus("Looks good!");

        await verifyTextExists("Warning 1");
        await verifyTextExists("Warning 2");
    });

    it.each([
        {
            description:
                "Should not show ValidationCard when errorInfo is undefined",
            errorInfo: undefined
        },
        {
            description:
                "Should not show ValidationCard when errorInfo is empty array",
            errorInfo: []
        }
    ])("$description", async ({ errorInfo }) => {
        await renderValidationStatus({
            ...defaultProps,
            state: InputState.InValid,
            errorInfo
        });

        await hoverOverStatus("Issues found");

        verifyTextDoesNotExist(/Error/);
        verifyTextDoesNotExist(/Warning/);
    });

    it.each([
        {
            description:
                "Should not show error ValidationCard when type 0 items does not exist under errorInfo",
            state: InputState.InValid,
            errorInfo: [
                createErrorDetail(
                    ErrorType.NUMBER_1,
                    "Warning only",
                    "Warning description"
                ),
                createErrorDetail(
                    2 as ErrorType,
                    "Info only",
                    "Info description"
                )
            ],
            hoverText: "Issues found",
            textToVerifyAbsent: /Error/
        },
        {
            description:
                "Should not show warning ValidationCard when type 1 items does not exist under errorInfo",
            state: InputState.Warning,
            errorInfo: [
                createErrorDetail(
                    0 as ErrorType,
                    "Error only",
                    "Error description"
                ),
                createErrorDetail(
                    2 as ErrorType,
                    "Info only",
                    "Info description"
                )
            ],
            hoverText: "Looks good!",
            textToVerifyAbsent: /Warning/
        }
    ])(
        "$description",
        async ({ state, errorInfo, hoverText, textToVerifyAbsent }) => {
            await renderValidationStatus({
                ...defaultProps,
                state,
                errorInfo
            });

            await hoverOverStatus(hoverText);

            verifyTextDoesNotExist(textToVerifyAbsent);
        }
    );
});
