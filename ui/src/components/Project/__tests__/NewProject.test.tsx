// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockLocationStateValue, mockNavigateCall } from "tests/mockRouter";
import { fireEventChange } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";

import NewProject, { NewProjectProps } from "../NewProject";

import { setCreateProjectResponseType } from "./MockFunctions/CreateProject.mock";

const toggleDialog = vi.fn();
const defaultProps: NewProjectProps = {
    groupId: "group_00000000000000000000000000000000",
    groupName: "Default",
    open: true,
    toggleDialog
};

describe("NewProject", () => {
    let queryClient: QueryClient;

    const renderNewProject = async (props: NewProjectProps = defaultProps) => {
        await act(async () => {
            render(<NewProject {...props} />, {
                wrapper: createWrapper(queryClient)
            });
        });
    };

    const getNameInput = () => screen.getByPlaceholderText("Enter name");

    const typeProjectName = async (name: string) => {
        await userEvent.type(getNameInput(), name);
    };

    const typeDescription = (text: string) => {
        fireEventChange(
            screen.getByPlaceholderText("Enter description here"),
            text
        );
    };

    const clickCreate = async () => {
        await userEvent.click(screen.getByRole("button", { name: /Create/i }));
    };

    beforeEach(() => {
        queryClient = createQueryClient();
        vi.clearAllMocks();
        mockLocationStateValue(null);
        mockNavigateCall();
        setCreateProjectResponseType();
    });

    afterEach(() => {
        queryClient.clear();
    });

    it("Should render the default placeholder when none is provided", async () => {
        await renderNewProject();

        expect(screen.getByText("Project name")).toBeInTheDocument();
        expect(getNameInput()).toBeInTheDocument();

        const input = screen.getByRole("textbox", { name: /project name/i });
        expect(input).toHaveAttribute("id", "project-name");
        expect(input).toHaveAttribute("name", "project_name");
    });

    it("Should render dialog title, description field, and buttons with Create disabled", async () => {
        await renderNewProject();

        expect(screen.getByRole("dialog")).toBeInTheDocument();
        expect(screen.getByText("Create New Project")).toBeInTheDocument();
        expect(
            screen.getByPlaceholderText("Enter description here")
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Back/i })
        ).toBeInTheDocument();
        expect(screen.getByRole("button", { name: /Create/i })).toBeDisabled();
    });

    it("Should enable Create when project name is typed and disable again after clearing", async () => {
        await renderNewProject();

        await typeProjectName("My Project");
        expect(screen.getByRole("button", { name: /Create/i })).toBeEnabled();

        await userEvent.clear(getNameInput());
        expect(screen.getByRole("button", { name: /Create/i })).toBeDisabled();
    });

    it("Should call toggleDialog(false) and not hit the API when Back is clicked", async () => {
        await renderNewProject();

        await userEvent.click(screen.getByRole("button", { name: /Back/i }));

        expect(toggleDialog).toHaveBeenCalledWith(false);
        expect(mockAxiosInstance.post).not.toHaveBeenCalled();
    });

    it.each([
        {
            description: "private project with group and description",
            props: {
                ...defaultProps,
                groupId: "group-abc",
                groupName: "Engineering"
            },
            projectName: "My Project",
            descriptionText: "A great project" as string | null,
            shareWithGroup: false,
            expectedPayload: {
                name: "My Project",
                description: "A great project",
                is_private: true
            },
            expectToggleDialog: true
        },
        {
            description: "shared project without description",
            props: defaultProps,
            projectName: "Shared Project",
            descriptionText: null,
            shareWithGroup: true,
            expectedPayload: {
                name: "Shared Project",
                description: undefined,
                is_private: false
            },
            expectToggleDialog: false
        }
    ])(
        "Should POST correct payload on Create for $description",
        async ({
            props,
            projectName,
            descriptionText,
            shareWithGroup,
            expectedPayload,
            expectToggleDialog
        }) => {
            await renderNewProject(props);

            await typeProjectName(projectName);

            if (descriptionText) {
                typeDescription(descriptionText);
            }

            if (shareWithGroup) {
                await userEvent.click(screen.getByText("Share with group"));
            }

            await clickCreate();

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    `v0/groups/${props.groupId}/projects`,
                    expectedPayload,
                    { headers: { "Content-Type": "application/json" } }
                );
            });

            if (expectToggleDialog) {
                await waitFor(() => {
                    expect(toggleDialog).toHaveBeenCalledWith(false);
                });
            }
        }
    );

    it("submits an explicit immutable ID", async () => {
        await renderNewProject();
        await typeProjectName("Fraud Detection");
        await userEvent.type(
            screen.getByRole("textbox", { name: /ID/i }),
            "fraud-prod"
        );

        await clickCreate();

        await waitFor(() =>
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/groups/${defaultProps.groupId}/projects`,
                {
                    name: "Fraud Detection",
                    slug: "fraud-prod",
                    description: undefined,
                    is_private: true
                },
                { headers: { "Content-Type": "application/json" } }
            )
        );
    });

    it("blocks a reserved derived ID and lets the user choose a valid one", async () => {
        await renderNewProject();
        await typeProjectName("Settings");

        expect(screen.getByText("Preview: settings")).toBeVisible();
        expect(screen.getByRole("alert")).toHaveTextContent("is reserved");
        expect(screen.getByRole("button", { name: /Create/i })).toBeDisabled();

        await userEvent.type(
            screen.getByRole("textbox", { name: /ID/i }),
            "project-settings"
        );
        expect(screen.getByRole("button", { name: /Create/i })).toBeEnabled();
    });

    it("shows the authoritative server conflict without choosing a suffix", async () => {
        setCreateProjectResponseType("conflict");
        await renderNewProject();
        await typeProjectName("Fraud Detection");
        await clickCreate();

        expect(
            await screen.findByText(
                "Project slug 'fraud-detection' is already in use."
            )
        ).toBeVisible();
        expect(mockAxiosInstance.post).toHaveBeenCalledTimes(1);
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
            `v0/groups/${defaultProps.groupId}/projects`,
            {
                name: "Fraud Detection",
                description: undefined,
                is_private: true
            },
            { headers: { "Content-Type": "application/json" } }
        );
        expect(getNameInput()).toHaveValue("Fraud Detection");
    });

    it.each([
        {
            scenario: "server error",
            responseType: testResponseType.ERROR,
            expectedTexts: ["Project creation failed!"]
        },
        {
            scenario: "invalid response",
            responseType: testResponseType.INVALID,
            expectedTexts: [
                "Project creation failed!",
                "Failed to create Project"
            ]
        }
    ])(
        "Should show error banner on $scenario",
        async ({ responseType, expectedTexts }) => {
            setCreateProjectResponseType(responseType);
            await renderNewProject();

            await typeProjectName("Bad Project");
            await clickCreate();

            for (const text of expectedTexts) {
                expect(await screen.findByText(text)).toBeInTheDocument();
            }
        }
    );
});
