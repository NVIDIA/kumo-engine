// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { expectedConnectorListResponse } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";

import { ConnectorDropdownOption, InputState } from "types/inputs";
import { FlowType } from "types/tables";

import ConnectorSelection, { ConnectorSelectionProps } from "../Connector";

const connectors = expectedConnectorListResponse().map((c) => ({
    value: c.id,
    label: c.name,
    type: c.config.type,
    config: c.config
})) as ConnectorDropdownOption[];

const onConnectorChange = vi.fn();

const defaultProps: ConnectorSelectionProps = {
    connectors,
    onConnectorChange,
    nameValidationState: InputState.None,
    fileConnectors: [],
    flowType: FlowType.CREATE,
    connectorId: ""
};

describe("ConnectorSelection", () => {
    const renderConnectorSelection = async (
        props: ConnectorSelectionProps,
        initialEntries: (string | { pathname: string; state?: unknown })[] = [
            "/tables/new-table"
        ]
    ) => {
        await act(async () => {
            render(
                <MemoryRouter initialEntries={initialEntries}>
                    <ConnectorSelection {...props} />
                </MemoryRouter>
            );
        });
    };
    const openDropdown = async () => {
        const dropdown = await screen.findByTestId("testid-select-connector");
        await userEvent.click(dropdown);
        return dropdown;
    };

    const verifyConnectorList = (
        connectorList: HTMLElement,
        expectedConnectors: string[]
    ) => {
        const options = within(connectorList).getAllByRole("option");
        expect(options).toHaveLength(expectedConnectors.length);
        expectedConnectors.forEach((connector) => {
            expect(connectorList).toHaveTextContent(connector);
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should display 'No connector found.' when no connectors are available", async () => {
        await renderConnectorSelection({
            ...defaultProps,
            connectors: []
        });

        expect(screen.getByText("Connector")).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Select Connector" })
        ).toBeInTheDocument();

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeInTheDocument();

        await userEvent.click(dropdown);
        await waitFor(() => {
            expect(screen.getByText("No connector found.")).toBeInTheDocument();
        });
    });

    it.each([
        InputState.Valid,
        InputState.InValid,
        InputState.Warning,
        InputState.ValidationError,
        InputState.None
    ])(
        "Should enable connector selection when nameValidationState is %s",
        async (state) => {
            await renderConnectorSelection({
                ...defaultProps,
                nameValidationState: state
            });

            const dropdown = screen.getByTestId("testid-select-connector");
            expect(dropdown).toBeEnabled();
        }
    );

    it("Should disable connector selection when nameValidationState is `Validating`", async () => {
        await renderConnectorSelection({
            ...defaultProps,
            nameValidationState: InputState.Validating
        });

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeDisabled();
    });

    it("Should disable connector selection in EDIT flow type", async () => {
        await renderConnectorSelection({
            ...defaultProps,
            nameValidationState: InputState.Valid,
            flowType: FlowType.EDIT
        });

        expect(screen.getByText("Connector")).toBeInTheDocument();

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeDisabled();
    });

    it("Should display connector list in dropdown for `CREATE` flow type", async () => {
        await renderConnectorSelection(defaultProps);

        await openDropdown();

        const connectorsToCheck = [
            "ltv_test",
            "hnm",
            "finance",
            "test_bigquery",
            "test_databricks",
            "raw_gcs",
            "curated_adls"
        ];
        const connectorList = await screen.findByRole("listbox");
        verifyConnectorList(connectorList, connectorsToCheck);
    });

    it("Should show connector name as selected in `EDIT` flow type", async () => {
        const connectorId = "ltv_test";
        await renderConnectorSelection({
            ...defaultProps,
            connectors: [],
            flowType: FlowType.EDIT,
            connectorId
        });

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeDisabled();
        expect(dropdown).toHaveTextContent(connectorId);
    });

    it("Should show selected connector and keep dropdown disabled in `DUPLICATE` flow type", async () => {
        const connectorId = "ltv_test";
        await renderConnectorSelection({
            ...defaultProps,
            flowType: FlowType.DUPLICATE,
            connectorId
        });

        expect(screen.getByText("Connector")).toBeInTheDocument();

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeDisabled();
        expect(dropdown).toHaveTextContent(connectorId);
    });

    it("Should update and display selected connector name after selection in `CREATE` flow type", async () => {
        const onConnectorChange = vi.fn();
        const namedConnectors = [
            {
                ...connectors[0],
                value: "warehouse-connector-id",
                label: "Customer Warehouse"
            },
            {
                ...connectors[1],
                value: "lakehouse-connector-id",
                label: "Analytics Lakehouse"
            }
        ];
        await renderConnectorSelection({
            ...defaultProps,
            connectors: namedConnectors,
            onConnectorChange,
            disableDefaultSelection: true
        });

        const dropdown = await openDropdown();
        const connectorList = await screen.findByRole("listbox");
        verifyConnectorList(connectorList, [
            "Customer Warehouse",
            "Analytics Lakehouse"
        ]);
        expect(connectorList).not.toHaveTextContent("warehouse-connector-id");
        expect(connectorList).not.toHaveTextContent("lakehouse-connector-id");

        const connectorOption = within(connectorList).getByRole("option", {
            name: "Analytics Lakehouse"
        });
        await userEvent.click(connectorOption);
        await waitFor(() => {
            expect(onConnectorChange).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: "lakehouse-connector-id",
                    label: "Analytics Lakehouse"
                })
            );
            expect(connectorList).not.toBeInTheDocument();
        });
        expect(dropdown).toHaveTextContent("Analytics Lakehouse");
        expect(dropdown).not.toHaveTextContent("lakehouse-connector-id");
    });

    it("Should filter and display matching connector names on valid search", async () => {
        await renderConnectorSelection(defaultProps);

        await openDropdown();

        const search = screen.getByPlaceholderText("Search...");
        expect(search).toBeInTheDocument();
        await userEvent.type(search, "hnm");

        const connectorsToCheck = ["hnm"];
        const connectorList = await screen.findByRole("listbox");
        const options = within(connectorList).getAllByRole("option");
        expect(options).toHaveLength(connectorsToCheck.length);
        verifyConnectorList(connectorList, connectorsToCheck);
    });

    it("Should display 'No connector found.' on invalid search input", async () => {
        await renderConnectorSelection(defaultProps);

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeInTheDocument();

        await userEvent.click(dropdown);

        const search = screen.getByPlaceholderText("Search...");
        expect(search).toBeInTheDocument();
        const searchInput = "03052000%";
        await userEvent.type(search, searchInput);

        await waitFor(() => {
            expect(screen.getByText("No connector found.")).toBeInTheDocument();
        });
    });

    it("Should display 'Something went wrong' when connector data is invalid", async () => {
        const invalidConnectors = [
            { config: {}, id: undefined },
            { value: null, label: null }
        ] as unknown as ConnectorDropdownOption[];

        await renderConnectorSelection({
            ...defaultProps,
            connectors: invalidConnectors
        });

        expect(
            screen.getByText(/Something went wrong in/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(/Please contact Kumo support/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(
                /or try refreshing the page if the issue persists./i
            )
        ).toBeInTheDocument();
    });

    it("Should add connector to list if not found during EDIT flow with non-empty connectorId", async () => {
        const connectorId = "custom_connector_not_in_list";
        const onConnectorChange = vi.fn();

        await renderConnectorSelection({
            ...defaultProps,
            flowType: FlowType.EDIT,
            connectorId,
            onConnectorChange
        });

        const dropdown = screen.getByTestId("testid-select-connector");
        expect(dropdown).toBeDisabled();
        expect(dropdown).toHaveTextContent(connectorId);
    });

    it("Should set connector from location state when available and in connector list", async () => {
        const connectorIdFromLocation = connectors[0].value;
        await renderConnectorSelection(defaultProps, [
            {
                pathname: "/tables/new-table",
                state: { value: connectorIdFromLocation }
            }
        ]);

        await waitFor(() => {
            expect(onConnectorChange).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: connectorIdFromLocation,
                    label: connectorIdFromLocation
                })
            );
        });
    });

    it("Should use first connector when location state value is not in connector list", async () => {
        const connectorIdFromLocation = "non_existent_connector";
        await renderConnectorSelection(defaultProps, [
            {
                pathname: "/tables/new-table",
                state: { value: connectorIdFromLocation }
            }
        ]);

        await waitFor(() => {
            expect(onConnectorChange).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: connectors[0].value
                })
            );
        });
    });
});
