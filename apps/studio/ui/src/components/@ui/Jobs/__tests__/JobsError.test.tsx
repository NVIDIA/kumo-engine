// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render } from "@testing-library/react";
import {
    verifySkeletonDoesNotExist,
    verifyTextDoesNotExist,
    verifyTextExists
} from "tests/testHelpers";

import { JobErrorType } from "types/warnings";

import JobsError, { JobsErrorProps } from "../JobsError";

const defaultProps: JobsErrorProps = {
    errors: [],
    fetchingErrors: false
};

describe("JobsError", () => {
    const renderJobsError = async (props: JobsErrorProps = defaultProps) => {
        await act(async () => {
            return render(
                <MemoryRouter>
                    <JobsError {...defaultProps} {...props} />
                </MemoryRouter>
            );
        });
    };

    const createMockError = (
        title: string,
        description: string,
        type: JobErrorType = JobErrorType.ERROR
    ) => ({
        type,
        title,
        description,
        cta: null
    });

    it("Should render skeleton loader when fetchingErrors is true", async () => {
        await renderJobsError({
            errors: [],
            fetchingErrors: true
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });

    it("Should not render skeleton loader when fetchingErrors is false", () => {
        renderJobsError({
            errors: [],
            fetchingErrors: false
        });

        verifySkeletonDoesNotExist();
    });

    it("Should render errors when fetchingErrors is false and errors are provided", async () => {
        const mockErrors = [
            createMockError("Error Title", "Error description")
        ];

        await renderJobsError({
            errors: mockErrors,
            fetchingErrors: false
        });

        verifyTextExists(/Error Title/);
        verifyTextExists(/Error description/);
    });

    it("Should render multiple errors when provided", async () => {
        const mockErrors = [
            createMockError("Error 1", "Desc 1"),
            createMockError("Error 2", "Desc 2")
        ];

        await renderJobsError({ errors: mockErrors, fetchingErrors: false });

        verifyTextExists(/Error 1/);
        verifyTextExists(/Desc 1/);
        verifyTextExists(/Error 2/);
        verifyTextExists(/Desc 2/);
    });

    it("Should not render errors that are not of type ERROR", async () => {
        const mockErrors = [
            createMockError("Info Title", "Info Desc", JobErrorType.INFO),
            createMockError("Error Title", "Error Desc")
        ];

        await renderJobsError({ errors: mockErrors, fetchingErrors: false });

        verifyTextDoesNotExist(/Info Title/);
        verifyTextDoesNotExist(/Info Desc/);

        verifyTextExists(/Error Title/);
        verifyTextExists(/Error Desc/);
    });

    it("Should render nothing when errors array is empty and fetchingErrors is false", () => {
        renderJobsError({ errors: [], fetchingErrors: false });

        verifySkeletonDoesNotExist();
        verifyTextDoesNotExist(/:/);
    });
});
