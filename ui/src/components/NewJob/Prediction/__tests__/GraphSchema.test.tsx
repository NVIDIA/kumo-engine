// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setTestGraphSnapshotResponseType } from "components/Jobs/Graph/__tests__/MockFunctions/GraphSnapshot.mock";

import * as Converters from "utils/Converters/snapshotConverter";

import GraphSchema, { GraphSchemaProps } from "../Schema/GraphSchema";

describe("GraphSchema", () => {
    const queryClient = createQueryClient();
    const renderGraphSchema = async (props: GraphSchemaProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphSchema {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        setTestGraphSnapshotResponseType(testResponseType.ALL);
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should render `GraphSchema` when the `graphSnapshotId` is set empty", async () => {
        await renderGraphSchema({
            graphSnapshotId: ""
        });

        expect(screen.queryByTestId("rf__background")).not.toBeInTheDocument();
    });

    it("Should render `GraphSchema` when the `graphSnapshotId` is null", async () => {
        await renderGraphSchema({
            graphSnapshotId: null as unknown as string
        });

        expect(screen.queryByTestId("rf__background")).not.toBeInTheDocument();
    });

    it("Should render `GraphSchema` when `graphSnapshotId` is set", async () => {
        const graphId =
            "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460";
        await renderGraphSchema({
            graphSnapshotId: graphId
        });

        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            `v0/snapshots/graph`,
            {
                "graph_snapshot_id": `${graphId}`,
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
    });

    it("Should display Graph with and the `View` icon for valid `graphSnapshotId`", async () => {
        const graphId =
            "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460";
        await renderGraphSchema({
            graphSnapshotId: graphId
        });

        const tableNames = [
            "FINANCE_ACCOUNT",
            "FINANCE_ORDERS",
            "FINANCE_TRANS"
        ];

        await waitFor(() => {
            for (const table of tableNames) {
                // Check table node
                expect(
                    screen.getByTestId(`rf__node-${table}`)
                ).toBeInTheDocument();

                // Check show icon
                expect(
                    screen.getByTestId(`${table}_show_icon`)
                ).toBeInTheDocument();
            }
        });
    });

    it("Should display `Column Statistics` table when view is clicked", async () => {
        const graphId =
            "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460";
        await renderGraphSchema({
            graphSnapshotId: graphId
        });

        expect(screen.getByTestId("rf__background")).toBeInTheDocument();

        expect(
            await screen.findByTestId("rf__node-FINANCE_ACCOUNT")
        ).toBeInTheDocument();

        const tableName = "FINANCE_ACCOUNT";
        const viewTable = screen.getByTestId(tableName + "_show_icon");
        fireEvent.click(viewTable);

        const collapseButton = screen.getByText("Back");
        expect(collapseButton).toBeInTheDocument();

        const table = screen.getAllByRole("table");
        const colStatsTable = table?.find(
            (t) => t?.id === "column-stats-table"
        );
        expect(colStatsTable).toBeInTheDocument();
    });

    it("Should display `Something went wrong` when there is an error", async () => {
        vi.spyOn(Converters, "graphSnapshotToJSGraph").mockImplementation(
            () => {
                throw new Error("Error");
            }
        );
        await renderGraphSchema({
            graphSnapshotId: ""
        });

        expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
        expect(
            screen.getByText(/Please contact Kumo support/i)
        ).toBeInTheDocument();
    });
});
