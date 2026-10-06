// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import PaginationComponent, {
    PaginationComponentProps
} from "../EntityLevel/Pagination";

const onSelectChange = vi.fn();
const defaultProps: PaginationComponentProps = {
    currentSelection: 2,
    totalOptions: 3,
    onSelectChange,
    siblingCount: 1,
    boundaryCount: 1,
    labelText: ""
};

const labelText = "1050000000000000000000000000000000000000000";

describe("PaginationComponent", () => {
    const renderPaginationComponent = async (
        props: PaginationComponentProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PaginationComponent {...props} />
                </MemoryRouter>
            );
        });
    };

    it("Should call onSelectChange when 'Next' button is clicked", async () => {
        await renderPaginationComponent({ ...defaultProps, onSelectChange });

        const nextButton = screen.getByRole("button", { name: /Next/i });
        fireEvent.click(nextButton);

        expect(onSelectChange).toHaveBeenCalledWith(3);
    });

    it("Should call onSelectChange when 'Previous' button is clicked", async () => {
        await renderPaginationComponent({
            ...defaultProps,
            onSelectChange
        });

        const prevButton = screen.getByRole("button", { name: /Previous/i });
        fireEvent.click(prevButton);

        expect(onSelectChange).toHaveBeenCalledWith(1);
    });

    it("Should disable 'Previous' button on first page", async () => {
        await renderPaginationComponent({
            ...defaultProps,
            currentSelection: 1
        });

        const prevButton = screen.getByRole("button", { name: /Previous/i });

        expect(prevButton).toBeDisabled();
    });

    it("Should disable 'Next' button on last page", async () => {
        await renderPaginationComponent({
            ...defaultProps,
            currentSelection: 3
        });

        const nextButton = screen.getByRole("button", { name: /Next/i });

        expect(nextButton).toBeDisabled();
    });

    it("Should display the labelText between the Previous and Next buttons", async () => {
        await renderPaginationComponent({
            ...defaultProps,
            labelText: "1050"
        });

        const labelText = screen.getByText("1050");
        expect(labelText).toBeInTheDocument();
    });

    it("Should display tooltip when labelText exceeds 32 characters", async () => {
        await renderPaginationComponent({
            ...defaultProps,
            labelText: labelText
        });

        const tooltipTrigger = document.querySelector(
            '[data-slot="tooltip-trigger"]'
        );

        await userEvent.hover(tooltipTrigger!);

        await waitFor(() => {
            const tooltipContent = screen.queryByRole("tooltip");
            expect(tooltipContent).toBeInTheDocument();
        });

        const tooltipContent = screen.queryByRole("tooltip", {
            name: labelText
        });

        expect(tooltipContent).toBeInTheDocument();

        // Check for the tooltip's dynamically generated ID
        const tooltipId = document.querySelector('[id^="radix-"]');
        expect(tooltipId).toBeInTheDocument();
    });

    it("Should not display tooltip when labelText length is <= 32 characters", async () => {
        const labelText = "Short text";

        await renderPaginationComponent({
            ...defaultProps,
            labelText: labelText
        });

        const tooltipTrigger = document.querySelector(
            '[data-slot="tooltip-trigger"]'
        );

        await userEvent.hover(tooltipTrigger!);

        await waitFor(() => {
            const tooltipContent = screen.queryByRole("tooltip");
            expect(tooltipContent).not.toBeInTheDocument();
        });
    });

    it("Should not display tooltip when not hovered", async () => {
        await renderPaginationComponent({
            ...defaultProps,
            labelText: labelText
        });

        const tooltipContent = screen.queryByRole("tooltip");
        expect(tooltipContent).not.toBeInTheDocument();
    });
});
