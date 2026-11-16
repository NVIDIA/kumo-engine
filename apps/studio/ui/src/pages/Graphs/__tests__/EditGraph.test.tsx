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
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setTableListResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/TableList.test.mock";

import { GraphDefinition, JSGraph } from "types/graphs";
import { FlowType } from "types/tables";

import {
    formatToLongLocalDateString,
    parseDate,
    parseDateTime
} from "utils/timeConverter";

import NewGraph from "../NewGraph/NewGraph";

import { expectedGraphDetailResponse } from "./MockFunction/GraphDetail.test.mock";
import { setGraphValidateResponseType } from "./MockFunction/GraphValidate.test.mock";
import { mockNewEditGraphResponse } from "./MockResponse/NewGraphResponse";
import { executeGraphCreationFlow } from "./graphCreationHelpers";

const editGraphInfo = (): JSGraph => {
    const graphDetails = expectedGraphDetailResponse();
    return {
        id: graphDetails.id,
        name: graphDetails.name_alias,
        tables: graphDetails.graph?.tables,
        groups: graphDetails.graph?.col_groups,
        totalLinks: graphDetails.num_links as number,
        totalTables: graphDetails.num_tables as number,
        created: parseDate(graphDetails.timestamps?.created_at),
        dateCreated: formatToLongLocalDateString(
            graphDetails.timestamps?.created_at
        ),
        updated: parseDateTime(graphDetails.timestamps?.updated_at),
        dateUpdated: formatToLongLocalDateString(
            graphDetails.timestamps?.updated_at
        ),
        table_name_to_fpid_mapping: graphDetails.table_name_to_fpid_mapping
    };
};

describe("EditGraphPage", () => {
    const renderEditGraphPage = async () => {
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

        verifyTextExists("Edit Graph");
        verifyButtonExists("Cancel");
        verifyButtonExists("Next");
        verifyTextExists("Link tables to create graph schema");
    };

    const editGraphApiCallTest = async (
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
            validateWithInvalidSuccess: options.validateWithInvalidSuccess,
            validName: "ltv_graph",
            forceRename: true
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        setTableListResponseType(); // reset to default ""
        mockNavigateCall();
    });

    afterEach(() => {
        setGraphValidateResponseType();
    });

    it("Should navigate back to Graph list page on `Cancel` click", async () => {
        const graphInfo = editGraphInfo();
        mockLocationStateValue({
            flow_type: FlowType.EDIT,
            name: graphInfo?.name,
            jsGraph: graphInfo
        });
        await renderEditGraphPage();
        await pageLoadTest();

        await findAndClickButtonByRole("Cancel");

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(-1);
        }); //Previous page
    });

    it("Should edit and save the graph with validation checks and handle errors", async () => {
        const graphInfo = editGraphInfo();
        mockLocationStateValue({
            flow_type: FlowType.EDIT,
            name: graphInfo?.name,
            jsGraph: graphInfo
        });

        await renderEditGraphPage();
        await pageLoadTest();

        const remainingNodes = screen.queryAllByTestId(/^rf__node-/);
        expect(remainingNodes).toHaveLength(3);

        // Verify that the checkbox is initially checked for articles table
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

        await editGraphApiCallTest(
            mockNewEditGraphResponse as GraphDefinition,
            {
                validateWithFailure: true
            }
        );
        await editGraphApiCallTest(
            mockNewEditGraphResponse as GraphDefinition,
            {
                validateWithInvalid: true
            }
        );
        await editGraphApiCallTest(
            mockNewEditGraphResponse as GraphDefinition,
            {
                validateWithInvalidSuccess: true
            }
        );
        await editGraphApiCallTest(
            mockNewEditGraphResponse as GraphDefinition,
            {}
        );
    });

    it("Should fire exactly one validate round-trip for a rapid burst of Save activations", async () => {
        const graphInfo = editGraphInfo();
        mockLocationStateValue({
            flow_type: FlowType.EDIT,
            name: graphInfo?.name,
            jsGraph: graphInfo
        });
        await renderEditGraphPage();
        await pageLoadTest();

        await findAndClickButtonByRole("Next");

        // The validate request never resolves, keeping the submit in flight.
        setGraphValidateResponseType(testResponseType.RUNNING);

        const saveButton = screen.getByRole("button", { name: "Save" });
        expect(saveButton).toBeEnabled();

        // Simulate queued keyboard re-activations: two extra click events
        // delivered while the first activation is still dispatching, before
        // React commits the disabled/loading re-render.
        let extraActivations = 2;
        const reactivate = () => {
            if (extraActivations > 0) {
                extraActivations -= 1;
                saveButton.dispatchEvent(
                    new MouseEvent("click", { bubbles: true, cancelable: true })
                );
            }
        };
        saveButton.addEventListener("click", reactivate);
        fireEvent.click(saveButton);
        saveButton.removeEventListener("click", reactivate);

        // Let the mutation pipeline flush before counting requests.
        await act(async () => {
            await new Promise((resolve) => setTimeout(resolve, 50));
        });

        const validateCalls = mockAxiosInstance.post.mock.calls.filter(
            (call) =>
                typeof call[0] === "string" &&
                call[0].startsWith("v1/graphs/validate")
        ).length;
        expect(validateCalls).toBe(1);

        await waitFor(() => {
            expect(saveButton).toBeDisabled();
        });
    });
});
