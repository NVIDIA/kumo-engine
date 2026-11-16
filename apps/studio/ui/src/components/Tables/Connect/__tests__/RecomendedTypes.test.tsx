// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import RecommendedTypes, { RecommendedTypesProps } from "../RecomendedTypes";

const mockRecommendedSTypes = [
    "Column 'FN' to 'Categorical'",
    "Column 'Active' to 'Categorical'"
];
const onApplyRecommendedTypes = vi.fn();

const defaultProps: RecommendedTypesProps = {
    recommendedSTypes: mockRecommendedSTypes,
    onApplyRecommendedTypes
};

describe("RecommendedTypes", () => {
    const renderRecommendedTypes = async (props: RecommendedTypesProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <RecommendedTypes {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should not render alert when recommendedSTypes is empty", async () => {
        await renderRecommendedTypes({
            ...defaultProps,
            recommendedSTypes: []
        });

        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("Should not render alert when recommendedSTypes is undefined", async () => {
        await renderRecommendedTypes({
            ...defaultProps,
            recommendedSTypes: undefined as unknown as string[]
        });

        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("Should render alert with correct message when recommendedSTypes has items", async () => {
        await renderRecommendedTypes(defaultProps);

        const alert = screen.getByRole("alert");
        expect(alert).toBeInTheDocument();
        expect(alert).toHaveTextContent(
            /Based on recent sample data analysis, we recommend updating/i
        );
        expect(alert).toHaveTextContent(
            /Apply for Default.*will only apply changes to columns where you have not yet manually changed the the type/i
        );
    });

    it("Should render all three action buttons", async () => {
        await renderRecommendedTypes(defaultProps);

        expect(
            screen.getByRole("button", { name: /Show Recommended Types/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Apply For All/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Apply For Default/i })
        ).toBeInTheDocument();
    });

    it("Should render with single recommended type", async () => {
        const singleType = ["Column 'ID' to 'Numeric'"];
        await renderRecommendedTypes({
            ...defaultProps,
            recommendedSTypes: singleType
        });

        const alert = screen.getByRole("alert");
        expect(alert).toBeInTheDocument();
    });

    it("Should show recommended types when Show Recommended Types button is clicked", async () => {
        await renderRecommendedTypes(defaultProps);

        mockRecommendedSTypes.forEach((type) => {
            expect(screen.queryByText(type)).not.toBeInTheDocument();
        });

        const showButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        await userEvent.click(showButton);

        mockRecommendedSTypes.forEach((type) => {
            const warningItem = screen.getByText(type);
            expect(warningItem).toBeInTheDocument();
        });
    });

    it("Should hide recommended types when Hide Recommended Types button is clicked", async () => {
        await renderRecommendedTypes(defaultProps);

        const showButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        await userEvent.click(showButton);

        mockRecommendedSTypes.forEach((type) => {
            expect(screen.getByText(type)).toBeInTheDocument();
        });

        const hideButton = screen.getByRole("button", {
            name: /Hide Recommended Types/i
        });
        await userEvent.click(hideButton);

        mockRecommendedSTypes.forEach((type) => {
            expect(screen.queryByText(type)).not.toBeInTheDocument();
        });
    });

    it("Should toggle button text between Show and Hide", async () => {
        await renderRecommendedTypes(defaultProps);

        let toggleButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        expect(toggleButton).toBeInTheDocument();

        await userEvent.click(toggleButton);

        toggleButton = screen.getByRole("button", {
            name: /Hide Recommended Types/i
        });
        expect(toggleButton).toBeInTheDocument();

        await userEvent.click(toggleButton);

        toggleButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        expect(toggleButton).toBeInTheDocument();
    });

    it("Should handle multiple show/hide cycles correctly", async () => {
        await renderRecommendedTypes(defaultProps);

        const getToggleButton = () =>
            screen.getByRole("button", {
                name: /(?:Show|Hide) Recommended Types/i
            });

        await userEvent.click(getToggleButton());
        expect(screen.getByText(mockRecommendedSTypes[0])).toBeInTheDocument();

        await userEvent.click(getToggleButton());
        expect(
            screen.queryByText(mockRecommendedSTypes[0])
        ).not.toBeInTheDocument();

        await userEvent.click(getToggleButton());
        expect(screen.getByText(mockRecommendedSTypes[0])).toBeInTheDocument();

        await userEvent.click(getToggleButton());
        expect(
            screen.queryByText(mockRecommendedSTypes[0])
        ).not.toBeInTheDocument();
    });

    it("Should call `onApplyRecommendedTypes` with true when Apply For Default is clicked", async () => {
        await renderRecommendedTypes(defaultProps);

        const applyForDefaultButton = screen.getByRole("button", {
            name: /Apply For Default/i
        });
        await userEvent.click(applyForDefaultButton);

        await waitFor(() => {
            expect(onApplyRecommendedTypes).toHaveBeenCalledTimes(1);
            expect(onApplyRecommendedTypes).toHaveBeenCalledWith(true);
        });
    });

    it("Should call `onApplyRecommendedTypes` with false when Apply For All is clicked", async () => {
        await renderRecommendedTypes(defaultProps);

        const applyForAllButton = screen.getByRole("button", {
            name: /Apply For All/i
        });
        await userEvent.click(applyForAllButton);

        await waitFor(() => {
            expect(onApplyRecommendedTypes).toHaveBeenCalledTimes(1);
            expect(onApplyRecommendedTypes).toHaveBeenCalledWith(false);
        });
    });

    it("Should handle clicking Apply For Default multiple times", async () => {
        await renderRecommendedTypes(defaultProps);

        const applyForDefaultButton = screen.getByRole("button", {
            name: /Apply For Default/i
        });

        await userEvent.click(applyForDefaultButton);
        await userEvent.click(applyForDefaultButton);

        await waitFor(() => {
            expect(onApplyRecommendedTypes).toHaveBeenCalledTimes(2);
            expect(onApplyRecommendedTypes).toHaveBeenNthCalledWith(1, true);
            expect(onApplyRecommendedTypes).toHaveBeenNthCalledWith(2, true);
        });
    });

    it("Should handle clicking Apply For All multiple times", async () => {
        await renderRecommendedTypes(defaultProps);

        const applyForAllButton = screen.getByRole("button", {
            name: /Apply For All/i
        });

        await userEvent.click(applyForAllButton);
        await userEvent.click(applyForAllButton);

        await waitFor(() => {
            expect(onApplyRecommendedTypes).toHaveBeenCalledTimes(2);
            expect(onApplyRecommendedTypes).toHaveBeenNthCalledWith(1, false);
            expect(onApplyRecommendedTypes).toHaveBeenNthCalledWith(2, false);
        });
    });

    it("Should work when Apply For Default is clicked after showing recommendations", async () => {
        await renderRecommendedTypes(defaultProps);

        const showButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        await userEvent.click(showButton);

        const applyForDefaultButton = screen.getByRole("button", {
            name: /Apply For Default/i
        });
        await userEvent.click(applyForDefaultButton);

        await waitFor(() => {
            expect(onApplyRecommendedTypes).toHaveBeenCalledWith(true);
        });
    });

    it("Should work when Apply For All is clicked after showing recommendations", async () => {
        await renderRecommendedTypes(defaultProps);

        const showButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        await userEvent.click(showButton);

        const applyForAllButton = screen.getByRole("button", {
            name: /Apply For All/i
        });
        await userEvent.click(applyForAllButton);

        await waitFor(() => {
            expect(onApplyRecommendedTypes).toHaveBeenCalledWith(false);
        });
    });

    it("Should handle large number of recommended types", async () => {
        const manyTypes = Array.from(
            { length: 20 },
            (_, i) => `Column 'Col${i}' to 'Type${i}'`
        );
        await renderRecommendedTypes({
            ...defaultProps,
            recommendedSTypes: manyTypes
        });

        const showButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        await userEvent.click(showButton);

        manyTypes.forEach((type) => {
            expect(screen.getByText(type)).toBeInTheDocument();
        });
    });

    it("Should handle recommended types with special characters", async () => {
        const specialTypes = [
            "Column 'User's Name' to 'Text'",
            "Column \"Quoted\" to 'String'",
            "Column with (parentheses) to 'Numeric'"
        ];
        await renderRecommendedTypes({
            ...defaultProps,
            recommendedSTypes: specialTypes
        });

        const showButton = screen.getByRole("button", {
            name: /Show Recommended Types/i
        });
        await userEvent.click(showButton);

        specialTypes.forEach((type) => {
            expect(screen.getByText(type)).toBeInTheDocument();
        });
    });
});
