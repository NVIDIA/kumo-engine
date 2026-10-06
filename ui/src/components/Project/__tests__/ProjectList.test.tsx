// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { removeFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import { fireEventChange } from "tests/testHelpers";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";

import { setGroupsResponseType } from "components/Admin/Groups/__tests__/MockFunctions/Groups.mock";

import { storageKeys } from "utils/Constants/enums";
import { getProjectPath } from "utils/Constants/routes";
import { getStoredSelectedProjectId } from "utils/storage";

import ProjectList, { ProjectListProps } from "../ProjectList";

import {
    mockProjectsData,
    setProjectsResponseType
} from "./MockFunctions/ProjectList.mock";

import "tests/mockAxios";

const toggleModal = vi.fn();
const defaultProps: ProjectListProps = {
    open: true,
    toggleModal,
    enableClose: false
};

describe("ProjectList", () => {
    let queryClient: QueryClient;

    const renderProjectList = async (
        props: ProjectListProps = defaultProps
    ) => {
        await act(async () => {
            render(<ProjectList {...props} />, {
                wrapper: createWrapper(queryClient)
            });
        });
    };

    const openGroupDropdown = async () => {
        await userEvent.click(
            await screen.findByTestId("project-group-selector")
        );
    };

    const selectGroup = async () => {
        await openGroupDropdown();
        await userEvent.click(await screen.findByText("Engineering Team"));
    };

    beforeEach(() => {
        queryClient = createQueryClient();
        vi.clearAllMocks();
        removeFromLocalStorage(storageKeys.selectedGroup);
        removeFromLocalStorage(storageKeys.selectedProject);
        mockLocationStateValue(null);
        mockNavigateCall();
        setGroupsResponseType();
        setProjectsResponseType();
    });

    afterEach(() => {
        queryClient.clear();
    });

    it("Should render the dialog with title, group dropdown, search bar, and tabs", async () => {
        await renderProjectList();

        expect(screen.getByRole("dialog")).toBeInTheDocument();
        expect(screen.getByText("Select Project")).toBeInTheDocument();
        expect(screen.getByPlaceholderText("Search")).toBeInTheDocument();
        expect(
            screen.getByRole("tab", { name: /My projects/i })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("tab", { name: /All projects/i })
        ).toBeInTheDocument();
    });

    it("Should render table headers: Name, Owner, Last Updated when no group is selected", async () => {
        await renderProjectList();

        expect(
            screen.getByRole("button", { name: /New project/i })
        ).toBeDisabled();

        await waitFor(() => {
            expect(screen.getByText("Name")).toBeInTheDocument();
            expect(screen.getByText("Owner")).toBeInTheDocument();
            expect(screen.getByText("Last Updated")).toBeInTheDocument();
        });
    });

    it.each([
        { enableClose: false, shouldExist: false },
        { enableClose: true, shouldExist: true }
    ])(
        "Should show Cancel button: $shouldExist when enableClose is $enableClose",
        async ({ enableClose, shouldExist }) => {
            await renderProjectList({ ...defaultProps, enableClose });
            if (shouldExist) {
                expect(
                    screen.getByRole("button", { name: /Cancel/i })
                ).toBeInTheDocument();
            } else {
                expect(
                    screen.queryByRole("button", { name: /Cancel/i })
                ).not.toBeInTheDocument();
            }
        }
    );

    it("Should call toggleModal when Cancel is clicked", async () => {
        await renderProjectList({ ...defaultProps, enableClose: true });

        await userEvent.click(screen.getByRole("button", { name: /Cancel/i }));

        expect(toggleModal).toHaveBeenCalled();
    });

    it("Should show group names in dropdown after groups load", async () => {
        await renderProjectList();
        await openGroupDropdown();

        await waitFor(() => {
            expect(screen.getByText("Engineering Team")).toBeInTheDocument();
            expect(screen.getByText("Data Science Team")).toBeInTheDocument();
            expect(screen.getByText("Administrators")).toBeInTheDocument();
        });
    });

    it("Should disable project creation when the user has no groups", async () => {
        setGroupsResponseType(testResponseType.EMPTY);
        await renderProjectList();

        await waitFor(() => {
            expect(
                screen.queryByTestId("project-group-selector")
            ).not.toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: /New project/i })
            ).toBeDisabled();
        });
    });

    it.each([
        {
            description: "selected group has projects",
            projectsResponseType: "",
            shouldBeVisible: [
                mockProjectsData[0].display_name!,
                mockProjectsData[1].display_name!
            ],
            shouldBeHidden: [] as string[]
        },
        {
            description: "selected group has no projects",
            projectsResponseType: testResponseType.EMPTY,
            shouldBeVisible: [] as string[],
            shouldBeHidden: [
                mockProjectsData[0].display_name!,
                mockProjectsData[1].display_name!
            ]
        }
    ])(
        "Should render persisted projects when $description",
        async ({ projectsResponseType, shouldBeVisible, shouldBeHidden }) => {
            setProjectsResponseType(projectsResponseType);
            await renderProjectList();
            await selectGroup();

            await waitFor(() => {
                shouldBeVisible.forEach((name) =>
                    expect(screen.getByText(name)).toBeInTheDocument()
                );
                shouldBeHidden.forEach((name) =>
                    expect(screen.queryByText(name)).not.toBeInTheDocument()
                );
            });
        }
    );

    it("Should filter rows by search input", async () => {
        await renderProjectList();
        await selectGroup();

        await waitFor(() =>
            expect(
                screen.getByText(mockProjectsData[0].display_name!)
            ).toBeInTheDocument()
        );

        fireEventChange(
            screen.getByPlaceholderText("Search"),
            mockProjectsData[0].display_name!
        );

        await waitFor(() => {
            expect(
                screen.getByText(mockProjectsData[0].display_name!)
            ).toBeInTheDocument();
            expect(
                screen.queryByText(mockProjectsData[1].display_name!)
            ).not.toBeInTheDocument();
        });
    });

    it("Should select a project on a single row click", async () => {
        await renderProjectList();
        await selectGroup();

        await userEvent.click(
            await screen.findByText(mockProjectsData[0].display_name!)
        );

        await waitFor(() => {
            expect(getStoredSelectedProjectId()).toBe(mockProjectsData[0].id);
            expect(toggleModal).toHaveBeenCalledTimes(1);
            expect(mockUseNavigate).toHaveBeenCalledWith(
                expect.stringContaining(
                    getProjectPath(mockProjectsData[0].id).ModelJobsList
                )
            );
        });
    });

    // A double click is habitual here — it used to be the only way to select.
    // Selecting twice would toggle the dialog back open.
    it("Should select only once when a row is double clicked", async () => {
        await renderProjectList();
        await selectGroup();

        await userEvent.dblClick(
            await screen.findByText(mockProjectsData[0].display_name!)
        );

        await waitFor(() =>
            expect(getStoredSelectedProjectId()).toBe(mockProjectsData[0].id)
        );
        expect(toggleModal).toHaveBeenCalledTimes(1);
    });

    it("Should allow selecting again after the dialog is reopened", async () => {
        const { rerender } = render(
            <ProjectList {...defaultProps} enableClose={true} />,
            { wrapper: createWrapper(queryClient) }
        );
        await selectGroup();
        await userEvent.click(
            await screen.findByText(mockProjectsData[0].display_name!)
        );

        rerender(
            <ProjectList {...defaultProps} open={false} enableClose={true} />
        );
        rerender(
            <ProjectList {...defaultProps} open={true} enableClose={true} />
        );

        await userEvent.click(
            await screen.findByText(mockProjectsData[1].display_name!)
        );

        await waitFor(() =>
            expect(getStoredSelectedProjectId()).toBe(mockProjectsData[1].id)
        );
    });

    it("Should enable New project button and open NewProject dialog after group selection", async () => {
        await renderProjectList();
        await selectGroup();

        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: /New project/i })
            ).toBeEnabled()
        );

        await userEvent.click(
            screen.getByRole("button", { name: /New project/i })
        );

        await waitFor(() => {
            expect(screen.getByText("Create New Project")).toBeInTheDocument();
        });
    });
});
