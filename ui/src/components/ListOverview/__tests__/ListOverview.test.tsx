// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter, Route, Routes } from "react-router";

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";

import ListOverview, { ListEntity, ListOverviewProps } from "../ListOverview";

const mockSetSelectedEntity = vi.fn();

const createMockEntity = (
    id: string,
    name: string,
    createdDate: string,
    updatedDate: string
): ListEntity => ({
    id,
    name,
    timestamps: {
        created_at: createdDate,
        updated_at: updatedDate
    }
});

const defaultProps: ListOverviewProps<ListEntity> = {
    title: "Connectors",
    data: [
        createMockEntity(
            "1",
            "Connector_1",
            "2023-01-01T00:00:00Z",
            "2023-01-02T00:00:00Z"
        ),
        {
            ...createMockEntity(
                "2",
                "Connector_2",
                "2023-02-01T00:00:00Z",
                "2023-02-03T00:00:00Z"
            ),
            dateCreated: "2023-02-01T00:00:00Z"
        }
    ],
    loadingStatus: false,
    overviewComponent: null,
    entityPresent: true,
    buttons: [],
    selectedEntity: null,
    setSelectedEntity: mockSetSelectedEntity
};

describe("ListOverview", () => {
    const renderListOverview = async (props: ListOverviewProps<ListEntity>) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={["/project/project-tests/connectors/1"]}>
                    <Routes>
                        <Route
                            path="/project/:projectId/connectors/:id"
                            element={<ListOverview {...props} />}
                        />
                    </Routes>
                </MemoryRouter>
            );
        });
    };

    const findSearchInput = () => screen.getByPlaceholderText("Search");

    const findEntityByIndex = (index: number) =>
        screen.findByTestId(`entity-name-${index}`);

    const clickEntityAndWait = async (index: number) => {
        const entity = await findEntityByIndex(index);
        expect(entity).toBeInTheDocument();
        await userEvent.click(entity);
    };

    const setupWindowLocation = (pathname: string) => {
        Object.defineProperty(window, "location", {
            value: { pathname },
            writable: true
        });
    };

    beforeEach(() => {
        mockNavigateCall();
    });

    it("Should render heading with number representing number of connectors present", async () => {
        await renderListOverview(defaultProps);

        const heading = screen.getByRole("heading", { name: /connectors/i });
        expect(heading).toBeInTheDocument();

        const count = screen.getByText("2");
        expect(count).toBeInTheDocument();

        expect(heading.parentElement).toContainElement(count);
    });

    // The header count lives in the `.ps-2` badge inside the title: it shows the
    // numeric count when settled and the (text-less) spinner while loading.
    const headerCountBadge = () =>
        screen
            .getByRole("heading", { name: /connectors/i })
            .querySelector(".ps-2");

    it("Should settle the header to a count (not a perpetual spinner) when a filter empties the list", async () => {
        // A legitimately-empty, fully-loaded list (e.g. a "Kind" filter matching
        // no rows) must show its count, never hang on the header spinner.
        await renderListOverview({
            ...defaultProps,
            data: [],
            loadingStatus: false
        });

        expect(headerCountBadge()?.textContent).toBe("0");
    });

    it("Should show the header spinner (no count) only while actively loading", async () => {
        await renderListOverview({
            ...defaultProps,
            data: [],
            loadingStatus: true
        });

        // Spinner element is rendered instead of a number, so the badge carries
        // no digit but does contain a child element (the settled "0" is text-only).
        expect(headerCountBadge()?.textContent ?? "").not.toMatch(/\d/);
        expect((headerCountBadge()?.children.length ?? 0) > 0).toBe(true);
    });

    it("Should render sort dropdown with default option", async () => {
        await renderListOverview(defaultProps);

        const searchInput = findSearchInput();
        expect(searchInput).toBeInTheDocument();

        const sortDropdown = screen.getByRole("button", {
            name: /Sort by: Most recent/i
        });
        expect(sortDropdown).toBeInTheDocument();
    });

    it("Should call setSelectedEntity when an entity is clicked", async () => {
        await renderListOverview(defaultProps);

        await clickEntityAndWait(0);

        await waitFor(() =>
            expect(mockSetSelectedEntity).toHaveBeenCalledWith(
                expect.objectContaining({
                    id: "1",
                    name: "Connector_1"
                })
            )
        );
    });

    it("Should show empty state message when no data is present", async () => {
        const props = {
            ...defaultProps,
            data: undefined
        };

        await renderListOverview(props);

        const emptyText = screen.getByText(
            /Any Connectors will be listed here/i
        );
        expect(emptyText).toBeInTheDocument();
    });

    it("Should display 'Select a connector from the left panel' when none of the tables are selected from the connector list.", async () => {
        await renderListOverview(defaultProps);

        const unselectedEntity = screen.getByTestId("entity-item-not-selected");
        expect(unselectedEntity).toBeInTheDocument();
        expect(
            screen.getByText(
                "Select a connector from the left panel to view its details here."
            )
        ).toBeInTheDocument();
    });

    it("Should display 'Any Connectors will be listed here.' for invalid search term", async () => {
        await renderListOverview(defaultProps);

        const searchInput = findSearchInput();
        expect(searchInput).toBeInTheDocument();

        await userEvent.type(searchInput, "Search Connectors");
        await waitFor(() => {
            const sideView = document.querySelector('[data-slot="card"]');
            expect(sideView).toBeInTheDocument();
            expect(sideView).toHaveTextContent(
                "Any Connectors will be listed here."
            );
        });
    });

    it("Should auto-select entity based on id from route params when selectedEntity is null", async () => {
        await renderListOverview(defaultProps);

        await waitFor(() => {
            expect(mockSetSelectedEntity).toHaveBeenCalledWith(
                expect.objectContaining({
                    id: "1",
                    name: "Connector_1"
                })
            );
        });
    });

    it("Should deselect selected entity and navigates when same entity is clicked again", async () => {
        setupWindowLocation("/project/project-tests/connectors/1");

        const selectedEntity = createMockEntity(
            "1",
            "Connector_1",
            "2023-01-01",
            "2023-01-02"
        );

        await renderListOverview({
            ...defaultProps,
            selectedEntity
        });

        // The default "Most recent" sort orders by the raw created_at
        // timestamp, so the older selected Connector_1 renders at index 1
        // (Connector_2, created later, is index 0).
        await clickEntityAndWait(1);

        await waitFor(() => {
            expect(mockSetSelectedEntity).toHaveBeenCalledWith(null);
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-tests/connectors/",
                {
                    "state": null
                }
            );
        });
    });

    it("Should navigate to the project connector list when route does not match", async () => {
        setupWindowLocation("/randomroute/1");

        const selectedEntity = createMockEntity(
            "1",
            "Connector_1",
            "2023-01-01",
            "2023-01-02"
        );

        await renderListOverview({
            ...defaultProps,
            selectedEntity
        });

        // The default "Most recent" sort orders by the raw created_at
        // timestamp, so the older selected Connector_1 renders at index 1
        // (Connector_2, created later, is index 0).
        await clickEntityAndWait(1);

        await waitFor(() => {
            expect(mockSetSelectedEntity).toHaveBeenCalledWith(null);
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-tests/connectors/",
                {
                    "state": null
                }
            );
        });
    });
});
