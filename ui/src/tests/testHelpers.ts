// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { mockAxiosInstance } from "./mockAxios";

export const getColumn = <T>(columns: T[], index: number): T => {
    const column = columns[index];
    if (!column) throw new Error(`Column at index ${index} not found`);
    return column;
};

export const verifyElementExists = (element: HTMLElement | Element | null) => {
    expect(element).toBeInTheDocument();
};

export const verifyElementDoesNotExist = (
    element: HTMLElement | Element | null
) => {
    expect(element).not.toBeInTheDocument();
};

export const verifyTextExists = async (text: string | RegExp) => {
    await waitFor(() => {
        expect(screen.getByText(text)).toBeInTheDocument();
    });
};

export const verifyTextDoesNotExist = (text: string | RegExp) => {
    expect(screen.queryByText(text)).not.toBeInTheDocument();
};

export const getSkeletonElement = () =>
    document.querySelector('[data-slot="skeleton"]');

export const verifySkeletonExists = () => {
    const skeleton = getSkeletonElement();
    expect(skeleton).toBeInTheDocument();
    expect(skeleton).toHaveClass("animate-pulse");
};

export const verifySkeletonDoesNotExist = () => {
    expect(getSkeletonElement()).not.toBeInTheDocument();
};

export const getEditIcon = (tableName: string) =>
    screen.queryByTestId(`${tableName}_edit_icon`);

export const clickEditIcon = async (tableName: string) => {
    const editIcon = await screen.findByTestId(`${tableName}_edit_icon`);
    fireEvent.click(editIcon);
};

export const verifyControlButtons = () => {
    verifyElementExists(screen.getByLabelText(/Zoom In/i));
    verifyElementExists(screen.getByLabelText(/Zoom Out/i));
    verifyElementExists(screen.getByLabelText(/Fit View/i));
    verifyElementExists(screen.getByLabelText(/Toggle Interactivity/i));
};

export const verifyButtonExists = (buttonName: string | RegExp) => {
    expect(
        screen.getByRole("button", { name: buttonName })
    ).toBeInTheDocument();
};

const noPointerCheckUser = userEvent.setup({
    delay: null,
    pointerEventsCheck: 0
});

export const clickWithoutPointerCheck = async (element: Element) => {
    await noPointerCheckUser.click(element);
};

export const hoverWithoutPointerCheck = async (element: Element) => {
    await noPointerCheckUser.hover(element);
};

export const typeWithoutDelay = async (element: Element, text: string) => {
    await noPointerCheckUser.type(element, text);
};

export const findAndClickButton = async (buttonText: string) => {
    const button = await screen.findByText(buttonText);
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
};

export const findAndClickButtonByRole = async (buttonText: string) => {
    const button = screen.getByRole("button", { name: buttonText });
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
};

export const findAndClickButtonByTestId = async (buttonText: string) => {
    const button = screen.getByTestId(buttonText);
    await waitFor(() => expect(button).toBeEnabled());
    await userEvent.click(button);
};

export const clickResetFilters = async () => {
    const resetButton = screen.getByRole("button", {
        name: /Reset Filters/i
    });
    await userEvent.click(resetButton);
};

export const getInputByPlaceholder = async (placeholder: string) => {
    const input = await screen.findByPlaceholderText(placeholder);
    expect(input).toBeInTheDocument();
    return input;
};

// Shared helper to verify the Jobs API request payload
export type VerifyJobsApiCallParams = {
    job_type?: string[];
    limit: number;
    page: number;
    status?: string[];
    tags_filter?: string[];
    search_keyword?: string;
    sort_by?: string;
    sort_desc?: boolean;
    additionalHeaders?: object;
};

export const sharedVerifyJobsApiCall = (params: {
    job_type?: string[];
    default_job_type: string[];
    limit: number;
    page: number;
    status?: string[];
    tags_filter?: string[];
    search_keyword?: string;
    sort_by?: string;
    sort_desc?: boolean;
    additionalHeaders?: object;
}) => {
    const {
        job_type,
        default_job_type,
        limit,
        page,
        status = [],
        tags_filter = [],
        search_keyword,
        sort_by,
        sort_desc = false,
        additionalHeaders
    } = params;

    const headers = {
        headers: { "Content-Type": "application/json" },
        ...(additionalHeaders || {})
    };

    expect(mockAxiosInstance.post).toHaveBeenCalledWith(
        "v0/jobs/",
        {
            job_type: job_type ?? default_job_type,
            limit,
            page,
            project_id: "project-tests",
            status,
            tags: {},
            tags_filter,
            time_range_end: null,
            time_range_start: null,
            search_keyword,
            sort_desc,
            sort_by
        },
        headers
    );
};

export const getTable = () => {
    const table = document.getElementById("output-stats-table");
    expect(table).toBeInTheDocument();
    return table!;
};

export const getDataRows = (table: HTMLElement) => {
    return table.querySelectorAll('tbody [data-slot="table-row"]');
};

export const hoverOverStatus = async (statusText: string) => {
    const statusComponent = screen.getByText(statusText);
    await userEvent.hover(statusComponent);
};

export const getAlert = () => {
    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    return alert;
};

export const expectNoAlert = () => {
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
};

export const getDialog = () => {
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    return dialog;
};

export const expectNoDialog = () => {
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
};

export const mockDevToolMode = async (mode: string) => {
    const { getReactQueryDevToolMode } = await import("utils/storage");
    (getReactQueryDevToolMode as ReturnType<typeof vi.fn>).mockReturnValue(
        mode
    );
};

export const getSwitch = () => screen.getByRole("switch");
export const getCloseButton = () =>
    screen.getByRole("button", { name: /cancel/i });

export const getColumnHeaders = () =>
    screen.getAllByRole("columnheader").map((th) => th.textContent);

export const fireEventChange = (element: HTMLElement, value: string) => {
    fireEvent.change(element, { target: { value } });
};

export const jobsListFilterOptions = [
    { value: "any", label: "Any Status" },
    { value: "QUEUED", label: "Queued" },
    { value: "RUNNING", label: "Running" },
    { value: "DONE", label: "Done" },
    { value: "FAILED", label: "Failed" },
    { value: "CANCELLED", label: "Canceled" },
    { value: "UNTRAINED", label: "Untrained" }
];
