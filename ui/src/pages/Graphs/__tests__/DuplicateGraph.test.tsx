// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import {
    findAndClickButtonByRole,
    verifyButtonExists,
    verifyTextExists
} from "tests/testHelpers";
import { createQueryClient } from "tests/testUtil";

import { setTableListResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/TableList.test.mock";

import { GraphDefinition, JSGraph } from "types/graphs";
import { FlowType } from "types/tables";

import { getProjectPath } from "utils/Constants";
import {
    formatToLongLocalDateString,
    parseDate,
    parseDateTime
} from "utils/timeConverter";

import NewGraph from "../NewGraph/NewGraph";

import { expectedGraphDetailResponse } from "./MockFunction/GraphDetail.test.mock";
import { mockNewDeuplicateGraphResponse } from "./MockResponse/NewGraphResponse";
import { executeGraphCreationFlow } from "./graphCreationHelpers";

const duplicateGraphInfo = (): JSGraph => {
    const graphDetails = expectedGraphDetailResponse();
    return {
        id: graphDetails.id,
        name: graphDetails.name_alias,
        tables: graphDetails.graph?.tables,
        groups: graphDetails.graph?.col_groups,
        totalLinks: graphDetails.num_links as number,
        totalTables: graphDetails.num_tables as number,
        created: parseDate(graphDetails.timestamps?.created_at),
        dateCreated: parseDateTime(graphDetails.timestamps?.created_at),
        updated: parseDateTime(graphDetails.timestamps?.updated_at),
        dateUpdated: formatToLongLocalDateString(
            graphDetails.timestamps?.updated_at
        ),
        table_name_to_fpid_mapping: graphDetails.table_name_to_fpid_mapping
    };
};

describe("DuplicateGraphPage", () => {
    const renderDuplicateGraphPage = async () => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/connectors/new-connector"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <NewGraph />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const pageLoadTest = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = ["v1/tables/"];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `${API}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });
        });

        verifyTextExists("Create New Graph");
        verifyButtonExists("Cancel");
        verifyButtonExists("Next");
        verifyTextExists("Link tables to create graph schema");
    };

    const duplicateGraphApiCallTest = async (
        args: GraphDefinition,
        options: {
            validateWithFailure?: boolean;
            validateWithInvalid?: boolean;
            validateWithInvalidSuccess?: boolean;
            validName?: string;
        }
    ) => {
        const clickSubmitButton = async () => {
            const saveGraph = screen.getByText("Save");
            await userEvent.click(saveGraph);
        };

        await executeGraphCreationFlow(args, clickSubmitButton, {
            validateWithFailure: options.validateWithFailure,
            validateWithInvalid: options.validateWithInvalid,
            validName: "ltv_graph_copy",
            forceRename: true
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setTableListResponseType();
        mockNavigateCall();
    });

    it("Should navigate back to Graph list page on `Cancel` click", async () => {
        mockLocationStateValue({
            flow_type: FlowType.DUPLICATE
        });
        await renderDuplicateGraphPage();
        await pageLoadTest();

        await findAndClickButtonByRole("Cancel");

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                getProjectPath("project-tests").Graphs,
                {
                    "replace": true
                }
            );
        }); //Previous page
    });

    it("Should successfully duplicate the graph and handle validation errors (failure, invalid data)", async () => {
        const graphInfo = duplicateGraphInfo();
        mockLocationStateValue({
            flow_type: FlowType.DUPLICATE,
            name: graphInfo?.name,
            jsGraph: graphInfo
        });
        await renderDuplicateGraphPage();
        await pageLoadTest();

        const graphNameInput = screen.getByPlaceholderText("name");
        await waitFor(() => {
            expect(graphNameInput).toHaveValue("ltv_graph_copy");
        });

        const remainingNodes = screen.queryAllByTestId(/^rf__node-/);
        expect(remainingNodes).toHaveLength(3);

        // Deselect the table
        await userEvent.click(await screen.findByTestId("pagination-3"));
        const checkBoxes = await screen.findAllByRole("checkbox", {
            name: /select row/i
        });

        await userEvent.click(checkBoxes[1]);
        expect(checkBoxes[1]).not.toBeChecked();

        // Verify the number of table nodes after deselecting a table
        const remainingNodes2 = screen.queryAllByTestId(/^rf__node-/);
        expect(remainingNodes2).toHaveLength(2);

        await findAndClickButtonByRole("Next");

        await duplicateGraphApiCallTest(
            mockNewDeuplicateGraphResponse as GraphDefinition,
            {
                validateWithFailure: true
            }
        );
        await duplicateGraphApiCallTest(
            mockNewDeuplicateGraphResponse as GraphDefinition,
            {
                validateWithInvalid: true
            }
        );
        await duplicateGraphApiCallTest(
            mockNewDeuplicateGraphResponse as GraphDefinition,
            {
                validateWithInvalidSuccess: true
            }
        );
        await duplicateGraphApiCallTest(
            mockNewDeuplicateGraphResponse as GraphDefinition,
            {}
        );
    });
});
