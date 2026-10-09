// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { fireEvent, render, screen } from "@testing-library/react";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { vi } from "vitest";

import { PageNames } from "utils/Constants";

import EmptyEntities from "../EmptyEntities";

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-sales"
}));
vi.mock("hooks/connector/connector", () => ({
    useConnectors: () => ({
        isDataLoading: false,
        isConnectorsPresent: false
    })
}));
vi.mock("hooks/graph/graphs", () => ({
    useGraphs: () => ({ isDataLoading: false, isGraphsPresent: false })
}));
vi.mock("hooks/tables/tables", () => ({
    useTables: () => ({ isDataLoading: false, tablesPresent: false })
}));
vi.mock("hooks/jobs/list/jobList", () => ({
    useJobs: () => [[], 0, false, vi.fn(), false]
}));
describe("EmptyEntities", () => {
    it("creates the first connector inside the selected project", () => {
        mockNavigateCall();
        render(
            <MemoryRouter>
                <EmptyEntities currentPage={PageNames.Connector} />
            </MemoryRouter>
        );

        fireEvent.click(screen.getByRole("button", { name: "Add connector" }));
        expect(mockUseNavigate).toHaveBeenCalledWith(
            "/project/project-sales/connectors/new-connector"
        );
    });
});
