// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { JsConnector } from "types/connectors";

import { CopyConnectorDialog } from "../CopyConnectorDialog";

const copyConnector = vi.fn();
const navigate = vi.fn();
const onClose = vi.fn();

const projects = [
    { id: "project-sales", display_name: "Sales", status: "active" },
    { id: "project-marketing", display_name: "Marketing", status: "active" },
    { id: "project-legacy", display_name: "Legacy", status: "archived" }
];

vi.mock("hooks/connector/connector", () => ({
    useCopyConnector: () => ({ copyConnector, isCopying: false })
}));
vi.mock("hooks/project/projects", () => ({
    useProjectList: () => ({
        projectListResponse: { data: projects },
        isLoading: false
    })
}));
vi.mock("utils/navigator", () => ({
    useCustomNavigation: () => navigate
}));

const connector = {
    id: "connector-sales",
    name: "Sales analytics"
} as JsConnector;

const renderDialog = () =>
    render(
        <MemoryRouter>
            <CopyConnectorDialog
                open={true}
                onClose={onClose}
                connector={connector}
                sourceProjectId="project-sales"
            />
        </MemoryRouter>
    );

const copyButton = () => screen.getByRole("button", { name: "Copy" });

const chooseTarget = async (name: string) => {
    await userEvent.click(
        screen.getByTestId("testid-copy-connector-target-project")
    );
    const listbox = await screen.findByTestId("listbox");
    await userEvent.click(await within(listbox).findByText(name));
};

describe("CopyConnectorDialog", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("offers only other active projects and requires a target", async () => {
        renderDialog();

        expect(screen.getByLabelText("Connector name")).toHaveValue(
            "Sales analytics"
        );
        expect(copyButton()).toBeDisabled();

        await userEvent.click(
            screen.getByTestId("testid-copy-connector-target-project")
        );
        const listbox = await screen.findByTestId("listbox");
        expect(within(listbox).getByText("Marketing")).toBeInTheDocument();
        expect(within(listbox).queryByText("Sales")).not.toBeInTheDocument();
        expect(within(listbox).queryByText("Legacy")).not.toBeInTheDocument();
    });

    it("copies with the chosen target and trimmed name, then links to the target project", async () => {
        copyConnector.mockResolvedValue({
            connector: {
                id: "connector-copy",
                name: "Sales copy",
                project_id: "project-marketing"
            },
            error: null
        });
        renderDialog();

        await chooseTarget("Marketing");
        const nameInput = screen.getByLabelText("Connector name");
        await userEvent.clear(nameInput);
        await userEvent.type(nameInput, "  Sales copy  ");
        await userEvent.click(copyButton());

        expect(copyConnector).toHaveBeenCalledWith(
            "project-sales",
            "connector-sales",
            { target_project_id: "project-marketing", name: "Sales copy" }
        );
        expect(
            await screen.findByText(
                'Connector "Sales copy" was copied to "Marketing".'
            )
        ).toBeInTheDocument();

        await userEvent.click(
            screen.getByRole("button", { name: "Go to project" })
        );
        expect(onClose).toHaveBeenCalled();
        expect(navigate).toHaveBeenCalledWith(
            "/project/project-marketing/connectors/"
        );
    });

    it("shows the server error inline and keeps the form", async () => {
        copyConnector.mockResolvedValue({
            connector: null,
            error: "Connector named 'Sales analytics' already exists in this project"
        });
        renderDialog();

        await chooseTarget("Marketing");
        await userEvent.click(copyButton());

        await waitFor(() =>
            expect(screen.getByRole("alert")).toHaveTextContent(
                "already exists in this project"
            )
        );
        expect(copyButton()).toBeEnabled();
        expect(navigate).not.toHaveBeenCalled();
    });

    it("sends one copy request when Copy is activated twice while a copy is pending", async () => {
        let resolveCopy: (value: unknown) => void = () => {};
        copyConnector.mockReturnValue(
            new Promise((resolve) => {
                resolveCopy = resolve;
            })
        );
        renderDialog();

        await chooseTarget("Marketing");
        await userEvent.click(copyButton());
        await userEvent.click(copyButton());
        resolveCopy({
            connector: null,
            error: "Connector named 'Sales analytics' already exists in this project"
        });

        await waitFor(() =>
            expect(screen.getByRole("alert")).toHaveTextContent(
                "already exists"
            )
        );
        expect(copyConnector).toHaveBeenCalledTimes(1);
    });

    it("disables Copy when the name is blank", async () => {
        renderDialog();

        await chooseTarget("Marketing");
        expect(copyButton()).toBeEnabled();
        await userEvent.clear(screen.getByLabelText("Connector name"));

        expect(copyButton()).toBeDisabled();
        expect(copyConnector).not.toHaveBeenCalled();
    });
});
