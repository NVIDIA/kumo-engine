// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import {
    removeFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import { fireEventChange } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";

import { setMockAdminUserIsAdmin } from "pages/Admin/__tests__/MockFunctions/AdminInfo.mock";

import { storageKeys } from "utils/Constants/enums";
import { getSelectedGroup, getStoredSelectedProjectId } from "utils/storage";
import { formatToShortLocalDateString } from "utils/timeConverter";

import ProjectSettings from "../ProjectSettings";

import {
    mockProjectData,
    setArchiveProjectResponseType,
    setProjectResponseType,
    setUpdateProjectResponseType
} from "./MockFunctions/ProjectSettings.mock";

import "tests/mockAxios";

describe("ProjectSettings", () => {
    let queryClient: QueryClient;

    const renderProjectSettings = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ProjectSettings />
                </MemoryRouter>,
                { wrapper: createWrapper(queryClient) }
            );
        });
    };

    const getNameInput = () =>
        screen.getByRole("textbox", { name: /project name/i });

    const clickSave = async () => {
        await userEvent.click(screen.getByRole("button", { name: /Save/i }));
    };

    const clickCancel = async () => {
        await userEvent.click(screen.getByRole("button", { name: /Cancel/i }));
    };

    const waitForProjectLoad = async () =>
        await waitFor(() =>
            expect(getNameInput()).toHaveValue(mockProjectData.display_name)
        );

    beforeEach(() => {
        queryClient = createQueryClient();
        vi.clearAllMocks();
        mockLocationStateValue(null);
        mockNavigateCall();
        setProjectResponseType();
        setUpdateProjectResponseType();
        setArchiveProjectResponseType();
        storeToLocalStorage(storageKeys.selectedProject, mockProjectData.id);
    });

    afterEach(() => {
        queryClient.clear();
        removeFromLocalStorage(storageKeys.selectedProject);
    });

    it("Should render the page title and action buttons", async () => {
        await renderProjectSettings();

        expect(screen.getByText("Project Settings")).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Cancel/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Save/i })
        ).toBeInTheDocument();
    });

    it("Should populate project name and description from API data and Save should be disabled", async () => {
        await renderProjectSettings();

        await waitFor(() => {
            expect(getNameInput()).toHaveValue(mockProjectData.display_name);
            expect(screen.getByPlaceholderText("Description")).toHaveValue(
                mockProjectData.description
            );
        });

        expect(screen.getByRole("button", { name: /Save/i })).toBeDisabled();
    });

    it("Should enable Save after changing project name, and disable again after reverting", async () => {
        await renderProjectSettings();

        await userEvent.clear(getNameInput());
        await userEvent.type(getNameInput(), "New Project Name");
        expect(screen.getByRole("button", { name: /Save/i })).toBeEnabled();

        await userEvent.clear(getNameInput());
        await userEvent.type(getNameInput(), mockProjectData.display_name!);
        expect(screen.getByRole("button", { name: /Save/i })).toBeDisabled();
    });

    it("Should enable Save after changing description, and disable again after reverting", async () => {
        await renderProjectSettings();
        await waitForProjectLoad();

        fireEventChange(
            screen.getByPlaceholderText("Description"),
            "Updated description"
        );
        expect(screen.getByRole("button", { name: /Save/i })).toBeEnabled();

        fireEventChange(
            screen.getByPlaceholderText("Description"),
            mockProjectData.description!
        );
        expect(screen.getByRole("button", { name: /Save/i })).toBeDisabled();
    });

    it("Should enable Save when access radio changes", async () => {
        await renderProjectSettings();

        await userEvent.click(screen.getByRole("radio", { name: /private/i }));

        expect(screen.getByRole("button", { name: /Save/i })).toBeEnabled();
    });

    it.each([{ field: "group name", text: "Engineering Team" }])(
        "Should display $field in overview",
        async ({ text }) => {
            await renderProjectSettings();

            await waitFor(() =>
                expect(screen.getByText(text)).toBeInTheDocument()
            );
        }
    );

    it("Should display Created and Last updated labels with formatted dates in overview", async () => {
        await renderProjectSettings();
        await waitForProjectLoad();

        expect(screen.getByText("Created")).toBeInTheDocument();
        expect(screen.getByText("Last updated")).toBeInTheDocument();

        const createdYear = new Date(mockProjectData.timestamps!.created_at)
            .getFullYear()
            .toString();
        const updatedYear = new Date(mockProjectData.timestamps!.updated_at)
            .getFullYear()
            .toString();

        expect(
            screen.getAllByText(new RegExp(createdYear)).length
        ).toBeGreaterThanOrEqual(1);
        expect(
            screen.getAllByText(new RegExp(updatedYear)).length
        ).toBeGreaterThanOrEqual(1);
    });

    // Regression for kumo-ai/kumo#29906: this page used a one-off
    // unlabeled formatter, so it disagreed with every other page rendering
    // the same fields. It must render through the shared local formatter,
    // zone label included (the old formatter's "Jan 1, 2026, 10:00 AM"
    // shape fails both assertions in any environment timezone).
    it("Should display Created and Last updated via the shared labeled local formatter", async () => {
        await renderProjectSettings();
        await waitForProjectLoad();

        const created = formatToShortLocalDateString(
            mockProjectData.timestamps!.created_at
        );
        const updated = formatToShortLocalDateString(
            mockProjectData.timestamps!.updated_at
        );
        // e.g. "PST", "UTC", or "GMT+9" — some zone label must be present.
        expect(created).toMatch(/ [A-Z]{2,5}([+-]\d{1,2}(:\d{2})?)?$/);

        expect(screen.getByText(created)).toBeInTheDocument();
        expect(screen.getByText(updated)).toBeInTheDocument();
    });

    it("Should render two radio buttons with Share selected by default", async () => {
        await renderProjectSettings();

        expect(screen.getAllByRole("radio")).toHaveLength(2);
        expect(
            screen.getByRole("radio", { name: /share with group/i })
        ).toHaveAttribute("data-state", "checked");
        expect(screen.getByRole("radio", { name: /private/i })).toHaveAttribute(
            "data-state",
            "unchecked"
        );
    });

    it("Should call navigate(-1) when Cancel is clicked", async () => {
        await renderProjectSettings();

        await clickCancel();

        expect(mockUseNavigate).toHaveBeenCalledWith(-1);
    });

    it.each([
        {
            description: "name change only",
            clickPrivateRadio: false,
            expectedPayload: {
                name: "Updated Name",
                description: mockProjectData.description,
                is_private: false
            }
        },
        {
            description: "name change with Private radio selected",
            clickPrivateRadio: true,
            expectedPayload: {
                name: "Updated Name",
                description: mockProjectData.description,
                is_private: true
            }
        }
    ])(
        "Should PATCH correct payload on $description",
        async ({ clickPrivateRadio, expectedPayload }) => {
            await renderProjectSettings();
            await waitForProjectLoad();
            await userEvent.clear(getNameInput());
            await userEvent.type(getNameInput(), "Updated Name");

            if (clickPrivateRadio) {
                await userEvent.click(
                    screen.getByRole("radio", { name: /private/i })
                );
            }

            await clickSave();

            await waitFor(() => {
                expect(mockAxiosInstance.patch).toHaveBeenCalledWith(
                    `v0/projects/${mockProjectData.id}`,
                    expectedPayload,
                    expect.objectContaining({
                        headers: { "Content-Type": "application/json" }
                    })
                );
            });
        }
    );

    it.each([
        {
            scenario: "server error",
            responseType: testResponseType.ERROR,
            expectedTexts: ["Project update failed!"]
        },
        {
            scenario: "invalid response",
            responseType: testResponseType.INVALID,
            expectedTexts: [
                "Project update failed!",
                "Failed to update Project"
            ]
        }
    ])(
        "Should show error banner on $scenario during save",
        async ({ responseType, expectedTexts }) => {
            setUpdateProjectResponseType(responseType);
            await renderProjectSettings();

            await userEvent.clear(getNameInput());
            await userEvent.type(getNameInput(), "Changed Name");
            await clickSave();

            for (const text of expectedTexts) {
                expect(await screen.findByText(text)).toBeInTheDocument();
            }
        }
    );

    it.each([
        {
            label: "Private",
            radioIndex: 0,
            tooltipText: "Only visible to you and workspace admins"
        },
        {
            label: "Share with group",
            radioIndex: 1,
            tooltipText:
                "All members in your group can view and access this project"
        }
    ])(
        "Should show tooltip for $label radio on hover",
        async ({ radioIndex, tooltipText }) => {
            await renderProjectSettings();
            await waitForProjectLoad();

            const infoIcons = document.querySelectorAll(
                '[data-slot="tooltip-trigger"]'
            );
            const infoIcon = infoIcons[radioIndex];
            expect(infoIcon).toBeInTheDocument();

            await userEvent.hover(infoIcon!);

            await waitFor(() => {
                expect(screen.getAllByText(tooltipText).length).toBeGreaterThan(
                    0
                );
            });
        }
    );

    it("Should display the immutable project ID with a copy button", async () => {
        await renderProjectSettings();
        await waitForProjectLoad();

        expect(screen.getByText(mockProjectData.slug)).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Copy to clipboard" })
        ).toBeInTheDocument();
    });

    describe("Archive project", () => {
        const archiveUrl = `v0/projects/${mockProjectData.id}/archive`;
        const archiveRequested = () =>
            vi
                .mocked(mockAxiosInstance.post)
                .mock.calls.some(([url]) => url === archiveUrl);
        const adminInfoLoaded = () =>
            vi
                .mocked(mockAxiosInstance.get)
                .mock.calls.some(([url]) => url === "v0/admin/info");

        const openArchiveDialog = async () => {
            await userEvent.click(
                await screen.findByRole("button", { name: "Archive project" })
            );
            const dialog = await screen.findByTestId("dialog-content");
            expect(dialog).toHaveTextContent(mockProjectData.display_name!);
            return dialog;
        };

        beforeEach(() => {
            setMockAdminUserIsAdmin(false);
            useUserStore
                .getState()
                .setUserDetails({ user_id: mockProjectData.created_by });
        });

        afterEach(() => {
            setMockAdminUserIsAdmin(true);
            useUserStore.getState().setUserDetails({ user_id: "" });
            removeFromLocalStorage(storageKeys.selectedGroup);
        });

        it("Should archive after confirmation and return to the Default project and group", async () => {
            storeToLocalStorage(storageKeys.selectedGroup, "group-001");
            await renderProjectSettings();
            await waitForProjectLoad();

            const dialog = await openArchiveDialog();
            await userEvent.click(
                within(dialog).getByRole("button", { name: "Archive" })
            );

            await waitFor(() => expect(archiveRequested()).toBe(true));
            await waitFor(() => {
                expect(mockUseNavigate).toHaveBeenCalledWith(
                    "/project/default/models/?page=1&job_type=TRAINING_JOB"
                );
            });
            expect(getStoredSelectedProjectId()).toBe("default");
            expect(getSelectedGroup()).toBe(
                "group_00000000000000000000000000000000"
            );
        });

        it("Should not archive when the confirmation is cancelled", async () => {
            await renderProjectSettings();
            await waitForProjectLoad();

            const dialog = await openArchiveDialog();
            await userEvent.click(
                within(dialog).getByRole("button", { name: "Cancel" })
            );

            await waitFor(() =>
                expect(
                    screen.queryByTestId("dialog-content")
                ).not.toBeInTheDocument()
            );
            expect(archiveRequested()).toBe(false);
            expect(mockUseNavigate).not.toHaveBeenCalled();
        });

        it("Should show an error and stay on the page when archiving fails", async () => {
            setArchiveProjectResponseType(testResponseType.ERROR);
            await renderProjectSettings();
            await waitForProjectLoad();

            const dialog = await openArchiveDialog();
            await userEvent.click(
                within(dialog).getByRole("button", { name: "Archive" })
            );

            await waitFor(() =>
                expect(
                    screen.getAllByText("Project archive failed!").length
                ).toBeGreaterThan(0)
            );
            expect(mockUseNavigate).not.toHaveBeenCalled();
            expect(getStoredSelectedProjectId()).toBe(mockProjectData.id);
        });

        it("Should offer archiving to an administrator who does not own the project", async () => {
            setMockAdminUserIsAdmin(true);
            useUserStore.getState().setUserDetails({ user_id: "someone-else" });
            await renderProjectSettings();
            await waitForProjectLoad();

            expect(
                await screen.findByRole("button", { name: "Archive project" })
            ).toBeInTheDocument();
        });

        it("Should not offer archiving to a member who neither owns the project nor is an administrator", async () => {
            useUserStore.getState().setUserDetails({ user_id: "someone-else" });
            await renderProjectSettings();
            await waitForProjectLoad();
            await waitFor(() => expect(adminInfoLoaded()).toBe(true));
            await act(async () => {});

            expect(
                screen.queryByRole("button", { name: "Archive project" })
            ).not.toBeInTheDocument();
        });

        it.each([
            {
                description: "the Default project",
                selectedProject: "default",
                responseType: ""
            },
            {
                description: "an already archived project",
                selectedProject: mockProjectData.id,
                responseType: testResponseType.ARCHIVED
            }
        ])(
            "Should not offer archiving to the owner for $description",
            async ({ selectedProject, responseType }) => {
                storeToLocalStorage(
                    storageKeys.selectedProject,
                    selectedProject
                );
                setProjectResponseType(responseType);
                await renderProjectSettings();
                await waitForProjectLoad();

                expect(
                    screen.queryByRole("button", { name: "Archive project" })
                ).not.toBeInTheDocument();
                expect(
                    screen.queryByTestId("archive-project-section")
                ).not.toBeInTheDocument();
            }
        );
    });
});
