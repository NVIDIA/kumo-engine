// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import EntityLevel, { EntitySelectionProps } from "../EntityLevel/EntityLevel";

import Categories from "./MockResponses/EntityLevelCategories.json";

const defaultProps: EntitySelectionProps = {
    categories: Categories,
    category: "random",
    setCategory: vi.fn(),
    selectedEntityIndex: 0,
    setSelectedEntityIndex: vi.fn(),
    entityId: "265",
    originalEntityId: "3806",
    showCategories: true
};

describe("EntityLevel", () => {
    const renderEntityLevel = async (
        props: EntitySelectionProps = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <EntityLevel {...props} />
                </MemoryRouter>
            );
        });
    };

    const expectNoNavigationElements = () => {
        expect(
            screen.queryByTestId("testid-category-dropdown")
        ).not.toBeInTheDocument();
        expect(
            screen.queryByRole("button", { name: /Previous/i })
        ).not.toBeInTheDocument();
        expect(
            screen.queryByRole("button", { name: /Next/i })
        ).not.toBeInTheDocument();
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render `EntityLevel` when all the props are set", async () => {
        await renderEntityLevel();

        expect(
            screen.getByTestId("testid-category-dropdown")
        ).toBeInTheDocument();
        expect(screen.getByText(/Sort: Random/i)).toBeInTheDocument();

        expect(screen.getByText("3806")).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Previous/i })
        ).toBeDisabled();
        expect(screen.getByRole("button", { name: /Next/i })).toBeEnabled();
    });

    it("Should call `setCategory` when the category is changed", async () => {
        const setCategory = vi.fn();
        await renderEntityLevel({
            ...defaultProps,
            setCategory
        });

        const dropdown = screen.getByTestId("testid-category-dropdown");
        await userEvent.click(dropdown);

        //format category names correctly in dropdown options
        const expectedOptions = Object.keys(Categories).map((key) =>
            key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
        );

        // Wait for all options to be rendered
        await waitFor(() => {
            expectedOptions.forEach((display) => {
                expect(
                    screen.getByRole("menuitem", { name: display })
                ).toBeInTheDocument();
            });
        });

        const option = screen.getByText("Old Entity");
        await userEvent.click(option);
        expect(setCategory).toHaveBeenCalledWith(Object.keys(Categories)[6]);

        const category =
            Categories[Object.keys(Categories)[6] as keyof typeof Categories];

        expect(category?.toString()).toBe(String(category));
    });

    it("Should not display options when the `showCategories` is set as false", async () => {
        await renderEntityLevel({
            ...defaultProps,
            showCategories: false
        });

        expect(
            screen.queryByTestId("testid-category-dropdown")
        ).not.toBeInTheDocument();
    });

    it("Should call `setSelectedEntityIndex` with correct value when navigating", async () => {
        const setSelectedEntityIndex = vi.fn();
        await renderEntityLevel({
            ...defaultProps,
            setSelectedEntityIndex
        });

        const nextButton = screen.getByRole("button", { name: /Next/i });
        await userEvent.click(nextButton);

        expect(setSelectedEntityIndex).toHaveBeenCalledWith(1);
    });

    it("Should decode `entityId` when `originalEntityId` is null", async () => {
        await renderEntityLevel({
            ...defaultProps,
            originalEntityId: null,
            entityId: "265%2Fabc"
        });

        expect(screen.getByText("265/abc")).toBeInTheDocument();
    });

    it.each([
        {
            testCase: "category is undefined",
            props: { category: undefined }
        },
        {
            testCase: "categories is an empty object",
            props: { categories: {}, category: "empty_category" }
        },
        {
            testCase: "category is empty string",
            props: {
                categories: { "": ["id1", "id2"], ...Categories },
                category: ""
            }
        }
    ])("Should render null when $testCase", async ({ props }) => {
        await renderEntityLevel({
            ...defaultProps,
            ...props
        });

        expectNoNavigationElements();
    });

    it("Should display `entityId` when `originalEntityId` is null", async () => {
        await renderEntityLevel({
            ...defaultProps,
            originalEntityId: null
        });

        expect(screen.getByText("265")).toBeInTheDocument();
    });

    it("Should not render category dropdown when category is empty string", async () => {
        await renderEntityLevel({
            ...defaultProps,
            category: "",
            setCategory: vi.fn()
        });

        expect(
            screen.queryByTestId("testid-category-dropdown")
        ).not.toBeInTheDocument();
    });

    it("Should display entityIdentifier alongside entity value when provided", async () => {
        await renderEntityLevel({
            ...defaultProps,
            entityIdentifier: "driverid",
            originalEntityId: "838"
        });

        expect(screen.getByText("driverid")).toBeInTheDocument();
        expect(screen.getByText("838")).toBeInTheDocument();
    });
});
