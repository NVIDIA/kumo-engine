// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { updateProjectsConnectorsByConnectorId } from "gen/v0/api/connectors/connectors";
import { vi } from "vitest";

import ConnectorMetadataForm from "../ConnectorMetadataForm";

const navigate = vi.fn();

vi.mock("gen/v0/api/connectors/connectors", () => ({
    updateProjectsConnectorsByConnectorId: vi.fn()
}));
vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-sales"
}));
vi.mock("utils/navigator", () => ({
    useCustomNavigation: () => navigate
}));

describe("ConnectorMetadataForm", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(updateProjectsConnectorsByConnectorId).mockResolvedValue({
            data: {}
        } as never);
    });

    it("updates project-owned connector metadata", async () => {
        render(
            <MemoryRouter
                initialEntries={[
                    {
                        pathname:
                            "/project/project-sales/connectors/edit-connector",
                        state: {
                            connectorID: "connector-sales",
                            name: "sales",
                            description: "Old description"
                        }
                    }
                ]}>
                <ConnectorMetadataForm />
            </MemoryRouter>
        );

        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "sales analytics" }
        });
        fireEvent.change(screen.getByLabelText("Description"), {
            target: { value: "Updated description" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Save" }));

        await waitFor(() =>
            expect(updateProjectsConnectorsByConnectorId).toHaveBeenCalledWith(
                "project-sales",
                "connector-sales",
                {
                    name: "sales analytics",
                    description: "Updated description"
                }
            )
        );
        expect(navigate).toHaveBeenCalledWith(
            "/project/project-sales/connectors/connector-sales",
            { replace: true }
        );
    });
});
