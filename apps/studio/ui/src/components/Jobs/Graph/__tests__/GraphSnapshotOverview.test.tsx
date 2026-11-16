// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import mockGraphSnapshotResponse from "tests/fixtures/GraphSnapshot.json";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setTestGraphSnapshotResponseType } from "components/Jobs/Graph/__tests__/MockFunctions/GraphSnapshot.mock";

import { jobStatus } from "utils/Constants";

import GraphSnapshotOverview, {
    GraphSnapshotOverviewProps
} from "../GraphSnapshotOverview";

const graphSnapshotId =
    "graph-0005b000000000000000000000000000@00033000000000000000000000000000";

const defaultProps: GraphSnapshotOverviewProps = {
    id: "",
    hidePreTitle: false,
    setStatus: vi.fn(),
    parentJobSource: ""
};

describe("GraphSnapshotOverview", () => {
    const queryClient = createQueryClient();
    const renderGraphSnapshotOverview = async (
        props: GraphSnapshotOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphSnapshotOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (id = graphSnapshotId) => {
        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/snapshots/graph`,
                {
                    "graph_snapshot_id": id,
                    "graph_definiton": undefined,
                    "graph_version": undefined
                },
                {
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "params": { "project_id": "project-tests" }
                }
            );

            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/snapshots/graph/${id}/stats`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });
    };

    const testStatusScenario = async (
        responseType: string,
        expectedStatus: string,
        expectedMessage: string | RegExp | null,
        expectedJobStatus: string
    ) => {
        setTestGraphSnapshotResponseType(responseType);
        const mockSetStatus = vi.fn();

        await renderGraphSnapshotOverview({
            ...defaultProps,
            id: graphSnapshotId,
            setStatus: mockSetStatus
        });

        await checkApiCall();
        await screen.findByText("Overview");

        const headerId = document.getElementById("header-chip-id");
        expect(headerId).toHaveTextContent(expectedStatus);

        if (expectedMessage) {
            const snapOverviewId = document.getElementById(
                "graphsnapshot-overview"
            );
            expect(snapOverviewId).toHaveTextContent(expectedMessage);
        }
        expect(mockSetStatus).toHaveBeenCalledWith(expectedJobStatus);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);
        setTestGraphSnapshotResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should show Title when hidePreTitle is false and display loading skeleton", async () => {
        await renderGraphSnapshotOverview(defaultProps);

        const skeleton = document.querySelector(`[data-slot="skeleton"]`);
        expect(skeleton).toBeInTheDocument();
        expect(await screen.findByText("Overview")).toBeInTheDocument();
    });

    it("Should not show Title when hidePreTitle is true", async () => {
        await renderGraphSnapshotOverview({
            ...defaultProps,
            hidePreTitle: true
        });

        const title = screen.queryByText("OVERVIEW");
        expect(title).not.toBeInTheDocument();
    });

    it("Should render graph snapshot with correct metadata, table nodes, and keys", async () => {
        await renderGraphSnapshotOverview({
            ...defaultProps,
            id: graphSnapshotId
        });

        await checkApiCall();

        expect(await screen.findByText("Overview")).toBeInTheDocument();

        // Check Tables x Links (separate property)
        expect(
            await screen.findByText(
                (content) =>
                    content.includes("3 Tables") && content.includes("1 Link")
            )
        ).toBeInTheDocument();
        // Check Rows • Size (separate property)
        expect(
            screen.getByText(
                (content) =>
                    content.includes("1.1M Rows") && content.includes("MB")
            )
        ).toBeInTheDocument();

        expect(await screen.findByText(/Snapshotted/i)).toBeInTheDocument();

        // Check all table nodes are rendered
        const tableNames = Object.keys(
            mockGraphSnapshotResponse.definition.tables
        );
        for (const tableName of tableNames) {
            const testId = `rf__node-${tableName}`;
            const node = await screen.findByTestId(testId);
            expect(node).toBeInTheDocument();
        }

        // Check PK and FK for all tables
        const tables = mockGraphSnapshotResponse.definition.tables;
        const foreignKeys = ["ACCOUNT_ID"];

        for (const tableName in tables) {
            const key = tableName as keyof typeof tables;
            const pkText = `PK: ${tables[key].pkey}`;
            expect(await screen.findByText(pkText)).toBeInTheDocument();
        }

        for (const fk of foreignKeys) {
            const fkElements = await screen.findAllByText((text) =>
                text.startsWith(`FK: ${fk}`)
            );
            expect(fkElements.length).toBeGreaterThan(0);
            fkElements.forEach((el) => expect(el).toBeInTheDocument());
        }
    });

    it("Should display the parentJobSource value in the Source label and call setStatus with correct job status", async () => {
        const testSource = "UI";
        const mockSetStatus = vi.fn();
        await renderGraphSnapshotOverview({
            ...defaultProps,
            id: graphSnapshotId,
            parentJobSource: testSource,
            setStatus: mockSetStatus
        });

        await checkApiCall();

        expect(await screen.findByText(testSource)).toBeInTheDocument();
        expect(mockSetStatus).toHaveBeenCalledWith(jobStatus.Done);
    });

    it("Should render show icons for all tables and display column stats when clicked", async () => {
        await renderGraphSnapshotOverview({
            ...defaultProps,
            id: graphSnapshotId
        });

        await checkApiCall();

        // Verify all show icons are rendered
        const tableNames = Object.keys(
            mockGraphSnapshotResponse.definition.tables
        );
        for (const tableName of tableNames) {
            const iconTestId = `${tableName}_show_icon`;
            const showIcon = await screen.findByTestId(iconTestId);
            expect(showIcon).toBeInTheDocument();
        }

        const tableName = "FINANCE_TRANS";
        const iconTestId = `${tableName}_show_icon`;

        const showIcon = await screen.findByTestId(iconTestId);

        fireEvent.click(showIcon);

        const columnStatElements =
            await screen.findAllByText("Column Statistics");
        expect(columnStatElements.length).toBeGreaterThan(0);

        const tables = await screen.findAllByRole("table");
        const colStatsTable = tables?.find(
            (t) => t?.id === "column-stats-table"
        );
        expect(colStatsTable).toBeInTheDocument();
    });

    it("Should render graph snapshot overview for 'FAILED' status with errors", async () => {
        const tableId = "table-1e113db640484f22c29f9100ccc34844@1680056762";
        await testStatusScenario(
            testResponseType.FAILED,
            "Failed",
            new RegExp(
                `Graph snapshot failed.*${tableId}.*Ingested table empty`,
                "s"
            ),
            jobStatus.Failed
        );

        // Verify errors are displayed
        const snapOverviewId = document.getElementById(
            "graphsnapshot-overview"
        );
        expect(snapOverviewId).toHaveTextContent("Errors:");
        expect(snapOverviewId).toHaveTextContent("Ingested table empty");
    });

    it.each([
        [
            testResponseType.QUEUED,
            "Queued",
            "Graph snapshot queued",
            jobStatus.Queued
        ],
        [
            testResponseType.RUNNING,
            "Running",
            "Graph snapshot creation in progress...",
            jobStatus.Running
        ],
        [testResponseType.CANCELLED, "Canceled", null, jobStatus.Cancelled]
    ])(
        "Should render graph snapshot overview for '%s' status",
        async (
            responseType,
            expectedStatus,
            expectedMessage,
            expectedJobStatus
        ) => {
            await testStatusScenario(
                responseType,
                expectedStatus,
                expectedMessage,
                expectedJobStatus
            );
        }
    );

    it("Should show an alert when validation state triggers an invalid request", async () => {
        setTestGraphSnapshotResponseType(testResponseType.INVALID);

        await renderGraphSnapshotOverview({
            ...defaultProps,
            id: graphSnapshotId
        });

        const alert = await screen.findByRole("alert");

        expect(
            await within(alert).findByText(/Unable to fetch graph snapshots/i)
        ).toBeInTheDocument();

        expect(
            await within(alert).findByText(
                /Please provide a valid graph snapshot ID or a valid graph definition and graph version to fetch a graph snapshot./i
            )
        ).toBeInTheDocument();
    });
});
