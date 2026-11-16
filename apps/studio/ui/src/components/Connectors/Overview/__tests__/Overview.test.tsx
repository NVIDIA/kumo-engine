// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";

import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { useConnectedTables } from "hooks/connector/connector";

import { JsConnector } from "types/connectors";

import Overview from "../Overview";

vi.mock("hooks/connector/connector", () => ({
    useConnectedTables: vi.fn()
}));
vi.mock("hooks/validation/validation", () => ({
    useValidationState: () => [
        null,
        null,
        vi.fn(),
        vi.fn(),
        { current: null },
        vi.fn()
    ]
}));
vi.mock("utils/navigator", () => ({
    useCustomNavigation: () => vi.fn()
}));

const connector: JsConnector = {
    id: "connector-sales",
    projectId: "project-sales",
    identityId: "identity-sales",
    name: "Sales analytics",
    description: "Least privilege",
    name_link: "/project/project-sales/connectors/connector-sales",
    created: "Jan 1, 2026",
    dateCreated: "Jan 1, 2026",
    sortCreated: "2026-01-01",
    updated: "Jan 1, 2026",
    dateUpdated: "Jan 1, 2026",
    sortUpdated: "2026-01-01",
    source: "databricks",
    config: {
        type: "databricks",
        catalog: "sales",
        schema_name: "analytics",
        warehouse_id: "warehouse"
    },
    showDelete: true,
    timestamps: {
        created_at: "2026-01-01",
        updated_at: "2026-01-01"
    }
};

describe("Connector overview", () => {
    beforeEach(() => {
        vi.mocked(useConnectedTables).mockReturnValue({
            isLoading: false,
            tables: [],
            setTables: vi.fn(),
            deleteTable: vi.fn()
        });
    });

    it("renders project-owned connector metadata and actions", () => {
        render(
            <Overview
                connectorInfo={connector}
                actionItems={[]}
                loadingStatus={false}
                isProjectView
            />
        );

        expect(screen.getByText("Connected Tables")).toBeVisible();
    });

    it("renders nothing without a selected connector", () => {
        const { container } = render(
            <Overview
                connectorInfo={null}
                actionItems={[]}
                loadingStatus={false}
            />
        );
        expect(container).toBeEmptyDOMElement();
    });

    it("truncates a long connector name while preserving its full value", () => {
        vi.mocked(useConnectedTables).mockImplementation(
            (_connectorId, _onError, setLoading) => {
                useEffect(() => setLoading(false), [setLoading]);
                return {
                    isLoading: false,
                    tables: [],
                    setTables: vi.fn(),
                    deleteTable: vi.fn()
                };
            }
        );
        const longName = "connector-" + "z".repeat(256);
        render(
            <Overview
                connectorInfo={{ ...connector, name: longName }}
                actionItems={[]}
                loadingStatus={false}
                isProjectView
            />
        );

        const headerName = screen.getByTitle(longName);
        expect(headerName).toHaveTextContent(longName);
    });
});
