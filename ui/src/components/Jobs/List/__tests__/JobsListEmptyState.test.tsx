// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createQueryClient } from "tests/testUtil";

import { JOBS_LIST_EMPTY_STATE } from "utils/Constants";

import JobsListEmptyState, {
    JobsListEmptyStateProps
} from "../JobsListEmptyState";

const onResetFiltersMock = vi.fn();
const defaultProps: JobsListEmptyStateProps = {
    onResetFilters: onResetFiltersMock
};

describe("JobsListEmptyState", () => {
    const queryClient = createQueryClient();
    const renderJobsListEmptyState = async (props: JobsListEmptyStateProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <JobsListEmptyState {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render empty state message", async () => {
        await renderJobsListEmptyState(defaultProps);

        expect(
            screen.getByText(JOBS_LIST_EMPTY_STATE.TITLE)
        ).toBeInTheDocument();
    });

    it("Should render Inbox icon", async () => {
        await renderJobsListEmptyState(defaultProps);

        const icon = document.querySelector('[class*="size-9"]');
        expect(icon).toBeInTheDocument();
    });

    it("Should render reset filters button with correct text", async () => {
        await renderJobsListEmptyState(defaultProps);

        const button = screen.getByRole("button", {
            name: JOBS_LIST_EMPTY_STATE.BUTTON_TEXT
        });
        expect(button).toBeInTheDocument();
        expect(button).toHaveTextContent(JOBS_LIST_EMPTY_STATE.BUTTON_TEXT);
    });

    it("Should call onResetFilters when button is clicked", async () => {
        await renderJobsListEmptyState(defaultProps);

        const button = screen.getByRole("button", {
            name: JOBS_LIST_EMPTY_STATE.BUTTON_TEXT
        });
        await userEvent.click(button);

        await waitFor(() => {
            expect(onResetFiltersMock).toHaveBeenCalledTimes(1);
        });
    });

    it("Should call onResetFilters only once per click", async () => {
        await renderJobsListEmptyState(defaultProps);

        const button = screen.getByRole("button", {
            name: JOBS_LIST_EMPTY_STATE.BUTTON_TEXT
        });

        await userEvent.click(button);
        await waitFor(() => {
            expect(onResetFiltersMock).toHaveBeenCalledTimes(1);
        });

        await userEvent.click(button);
        await waitFor(() => {
            expect(onResetFiltersMock).toHaveBeenCalledTimes(2);
        });
    });

    it("Should have reset button with correct id", async () => {
        await renderJobsListEmptyState(defaultProps);

        const button = document.querySelector("#reset-filters-button");
        expect(button).toBeInTheDocument();
    });
});
