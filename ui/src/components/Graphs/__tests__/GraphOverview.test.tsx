// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { focusManager } from "@tanstack/react-query";
import type { RenderResult } from "@testing-library/react";
import { waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { clickWithoutPointerCheck } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";

import {
    defaultGraphOverviewProps,
    resetQueryClient,
    setupGraphTests
} from "components/Graphs/__tests__/graphTestUtils";

import { setGraphsDetailResponseType } from "pages/Graphs/__tests__/MockFunction/GraphDetail.test.mock";
import { graphDetail } from "pages/Graphs/__tests__/MockResponse/GraphDetail.response";

import { formatToNumericLocalDateString } from "utils/timeConverter";

vi.mock("hooks/featureFlags");
vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-test"
}));

describe("GraphOverview", () => {
    const { graphOverviewMock, renderGraphOverview } = setupGraphTests();
    const graphId = "graph-00022000000000000000000000000000";

    beforeEach(() => {
        resetQueryClient();
        focusManager.setFocused(false);
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
        focusManager.setFocused(undefined);
        setGraphsDetailResponseType();
    });

    const renderComponent = async (
        overrides: Partial<typeof defaultGraphOverviewProps> = {}
    ) => {
        return await renderGraphOverview({
            ...defaultGraphOverviewProps,
            graphId,
            ...overrides
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/graphs/${graphId}`,
                undefined,
                { headers: undefined, params: { project_id: "project-test" } }
            );
        });
    };

    const openDropdownMenu = async (
        findByTestId: RenderResult["findByTestId"]
    ) => {
        const dropdownButton = await findByTestId("more-options-dropdown");
        await clickWithoutPointerCheck(dropdownButton);
        return dropdownButton;
    };

    const clickMenuItem = async (
        findByTestId: RenderResult["findByTestId"],
        getByRole: RenderResult["getByRole"],
        menuItemName: string
    ) => {
        await openDropdownMenu(findByTestId);
        const menuItem = getByRole("menuitem", { name: menuItemName });
        await clickWithoutPointerCheck(menuItem);
        return menuItem;
    };

    it("Should render loading skeleton for empty props", async () => {
        await renderGraphOverview();
        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });

    it("Should render nodes for each table in the graph", async () => {
        const { getByTestId } = await renderComponent();

        await checkApiCall();

        const tables = Object.keys(graphDetail.graph.tables);

        await waitFor(() => {
            tables.forEach((nodeId) => {
                expect(getByTestId(`rf__node-${nodeId}`)).toBeInTheDocument();
            });
        });
    });

    it("Should display graph creation time and tables/links summary", async () => {
        const { getByText, findByText } = await renderComponent();

        await checkApiCall();

        const graph_created = formatToNumericLocalDateString(
            graphDetail.timestamps.created_at
        );
        expect(await findByText(graph_created)).toBeInTheDocument();

        expect(
            getByText(
                (content) =>
                    content.includes("3 tables") && content.includes("2 links")
            )
        ).toBeInTheDocument();
    });

    it("Should render PK and FK for all tables in the graph snapshot", async () => {
        const { findByText, findAllByText } = await renderComponent();
        await checkApiCall();

        const tables = graphDetail.graph.tables;
        const foreignKeys = ["customer_id", "article_id"];

        for (const tableName in tables) {
            const key = tableName as keyof typeof tables;
            const pk = tables[key].pkey;

            if (pk) {
                const pkText = `PK: ${pk}`;
                expect(await findByText(pkText)).toBeInTheDocument();
            }
        }

        for (const fk of foreignKeys) {
            const fkElements = await findAllByText((text) =>
                text.startsWith(`FK: ${fk}`)
            );

            expect(fkElements.length).toBeGreaterThan(0);
            fkElements.forEach((el) => expect(el).toBeInTheDocument());
        }
    });

    it("Should render all dropdown menu items when hideMenuOptions is false", async () => {
        const { findByTestId, getByRole } = await renderComponent();

        await checkApiCall();
        await openDropdownMenu(findByTestId);

        const menuItems = [
            "Edit",
            "Delete",
            "Duplicate",
            "Re-ingest",
            "SDK Code"
        ];
        menuItems.forEach((item) => {
            const menuItem = getByRole("menuitem", { name: item });
            expect(menuItem).toBeInTheDocument();
        });
    });

    it("Should not render dropdown when hideMenuOptions is true", async () => {
        const { queryByTestId } = await renderComponent({
            hideMenuOptions: true
        });

        await checkApiCall();

        const dropdownButton = queryByTestId("more-options-dropdown");
        expect(dropdownButton).not.toBeInTheDocument();
    });

    it("Should call 'onEditClick' when Edit button is clicked", async () => {
        const { onEditClick } = graphOverviewMock;

        const { findByTestId, getByRole } = await renderComponent({
            onEditClick
        });

        await checkApiCall();
        await clickMenuItem(findByTestId, getByRole, "Edit");

        expect(onEditClick).toHaveBeenCalled();
    });

    it("Should call 'onDeleteClick' when Delete button is clicked", async () => {
        const { onDeleteClick } = graphOverviewMock;

        const { findByTestId, getByRole } = await renderComponent({
            onDeleteClick
        });

        await checkApiCall();
        await clickMenuItem(findByTestId, getByRole, "Delete");

        expect(onDeleteClick).toHaveBeenCalled();
    });

    it("Should call 'onCopyClick' when Duplicate button is clicked", async () => {
        const { onCopyClick } = graphOverviewMock;

        const { findByTestId, getByRole } = await renderComponent({
            onCopyClick
        });

        await checkApiCall();
        await clickMenuItem(findByTestId, getByRole, "Duplicate");

        expect(onCopyClick).toHaveBeenCalled();
    });

    it("Should call API again when Re-ingest button is clicked", async () => {
        const { findByTestId, getByRole, queryByText } =
            await renderComponent();

        await checkApiCall();
        await openDropdownMenu(findByTestId);

        const sdkCodeItem = queryByText(/SDK Code/i);
        expect(sdkCodeItem).toBeInTheDocument();

        const reinjestMenuItem = getByRole("menuitem", {
            name: "Re-ingest"
        });
        await clickWithoutPointerCheck(reinjestMenuItem);

        // Confirm that the API is called again after clicking Re-ingest
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v1/graphs/${graphId}/refresh`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-test" }
                }
            );
        });
    });

    it("Should render 'SDK Code' button and open SDK modal", async () => {
        const {
            getByTestId,
            getByText,
            findByTestId,
            getByRole,
            queryByTestId
        } = await renderComponent();

        await checkApiCall();
        await openDropdownMenu(findByTestId);

        const sdkCodeItem = getByText(/SDK Code/i);
        expect(sdkCodeItem).toBeInTheDocument();

        await clickWithoutPointerCheck(sdkCodeItem);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/graphs/${graphId}/sdk`,
                undefined,
                { headers: undefined, params: { project_id: "project-test" } }
            );
        });

        const modal = await findByTestId("sdk-content");
        expect(modal).toBeInTheDocument();
        expect(modal).toHaveTextContent(/View SDK Code/i);

        expect(getByRole("button", { name: "Copy" })).toBeInTheDocument();
        expect(getByRole("button", { name: /Download/i })).toBeInTheDocument();
        expect(getByTestId("close-icon")).toBeInTheDocument();

        await clickWithoutPointerCheck(getByTestId("close-icon"));

        await waitFor(() => {
            expect(queryByTestId("sdk-content")).not.toBeInTheDocument();
        });
    });

    it("Should display `Something went wrong` for API error", async () => {
        setGraphsDetailResponseType(testResponseType.ERROR);

        const { findByTestId } = await renderComponent();

        await checkApiCall();

        const alert = await findByTestId("alert-dialog-content");
        expect(alert).toBeInTheDocument();

        expect(alert).toHaveTextContent(/Something went wrong!/i);
    });

    it("Should handle invalid graphInfo structure and fallback correctly", async () => {
        setGraphsDetailResponseType(testResponseType.INVALID);

        const { findByText, queryByText } = await renderComponent();

        await checkApiCall();

        // Invalid graph data (no tables/links) — Configs falls back to emdash, not "0 Table x 0 Link"
        const configsLabel = await findByText("Configs");
        expect(configsLabel).toBeInTheDocument();
        expect(
            queryByText((c) => c.includes("table") && c.includes("link"))
        ).not.toBeInTheDocument();
    });

    it("Should render Skeleton loading state", async () => {
        setGraphsDetailResponseType(testResponseType.EMPTY);

        await renderComponent();
        await checkApiCall();
        const loadingSkeleton = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(loadingSkeleton).toBeInTheDocument();
    });

    it("Should render skeleton and hide dropdown when graphId is empty", async () => {
        const { queryByTestId } = await renderGraphOverview({
            ...defaultGraphOverviewProps,
            graphId: ""
        });

        const dropdownButton = queryByTestId("more-options-dropdown");
        expect(dropdownButton).not.toBeInTheDocument();

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });

    it("Should render GraphInformationPanel when selectedGraphInfo exists", async () => {
        const { container } = await renderComponent();

        await checkApiCall();

        await waitFor(() => {
            const reactFlowWrapper = container.querySelector(".react-flow");
            expect(reactFlowWrapper).toBeInTheDocument();
        });
    });
});
