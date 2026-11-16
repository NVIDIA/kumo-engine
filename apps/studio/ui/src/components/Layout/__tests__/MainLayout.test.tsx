// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import {
    removeFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import { createQueryClient } from "tests/testUtil";

import { setMockAdminUserIsAdmin } from "pages/Admin/__tests__/MockFunctions/AdminInfo.mock";

import { defaultUserDetails } from "services/auth";

import {
    getProjectPath,
    navRoutes,
    newDocsURL,
    SIDEBAR_VIEW_KEY,
    storageKeys
} from "utils/Constants";
import {
    getDefaultModelJobsPageURL,
    getDefaultPredictionJobsPageURL
} from "utils/jobs";

import MainLayout from "../MainLayout";

describe("MainLayout", () => {
    const projectPaths = getProjectPath("project-tests");
    // Radix dialogs set `pointer-events: none` on <body> while open; disable
    // userEvent's pointer check so clicks inside dialogs aren't rejected.
    let user: ReturnType<typeof userEvent.setup>;

    const renderMainLayout = async () => {
        await act(async () => {
            const queryClient = createQueryClient();
            render(
                <MemoryRouter initialEntries={["/"]}>
                    <QueryClientProvider client={queryClient}>
                        <MainLayout />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(`v0/admin/info`);
        });
    };

    beforeAll(() => {
        useUserStore.getState().setUserDetails({ isLoggedIn: true });
    });

    afterAll(() => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });
    });

    beforeEach(() => {
        user = userEvent.setup({ pointerEventsCheck: 0 });
        vi.clearAllMocks();
        mockNavigateCall();
        mockLocationStateValue(null);
        window.open = vi.fn();
    });

    it("Should render `MainLayout` ", async () => {
        await renderMainLayout();
        await checkApiCall();

        const sideBar = document.querySelector('[data-slot="sidebar-wrapper"]');
        expect(sideBar).toBeInTheDocument();
    });

    it("Should display Model and Prediction option on clicking the `New` button", async () => {
        await renderMainLayout();
        await checkApiCall();

        const newButton = await screen.findByRole("button", { name: "New" });
        await user.click(newButton);

        const menuItems = screen.getAllByRole("menuitem");
        expect(menuItems.length).toBeGreaterThanOrEqual(2);

        //Model
        expect(screen.getByText("Model")).toBeInTheDocument();
        expect(
            screen.getByText(
                "Train a model with a graph schema and a predictive query."
            )
        ).toBeInTheDocument();

        //Prediction
        expect(screen.getByText("Prediction")).toBeInTheDocument();
        expect(
            screen.getByText(
                "Generate predictions using a trained model and graph."
            )
        ).toBeInTheDocument();
    });

    it("Should navigate to New Model/New Prediction page when New option is clicked", async () => {
        await renderMainLayout();
        await checkApiCall();

        const pages = [
            { name: "Model", path: projectPaths.NewTrainingGraphSelect },
            { name: "Prediction", path: projectPaths.NewPredictionJob }
        ];

        for (const { name, path } of pages) {
            const newModel = screen.getByRole("button", { name: /new/i });
            await user.click(newModel);

            const modelOption = await screen.findByText(name);
            await user.click(modelOption);

            expect(mockUseNavigate).toHaveBeenCalledWith(path);
        }
        const logo = await screen.findByAltText(/logo/i);
        await user.click(logo);

        expect(mockUseNavigate).toHaveBeenCalledWith("/");
    });

    it("Should navigate to desired pages when each menu is clicked from sidemenu", async () => {
        await renderMainLayout();
        await checkApiCall();

        const pages = [
            {
                name: "Models",
                path: getDefaultModelJobsPageURL("project-tests")
            },
            {
                name: "Predictions",
                path: getDefaultPredictionJobsPageURL("project-tests")
            },
            { name: "Tables", path: projectPaths.Tables },
            { name: "Graphs", path: projectPaths.Graphs },
            { name: "Connectors", path: projectPaths.Connectors }
        ];
        for (const { name, path } of pages) {
            const button = await screen.findByRole("button", { name: name });
            await user.click(button);

            await waitFor(() => {
                expect(mockUseNavigate).toHaveBeenCalledWith(path);
            });
        }
    });

    it("Should navigate to Documentation page when `Documentation` button is clicked", async () => {
        await renderMainLayout();
        await checkApiCall();

        const tableButton = await screen.findByRole("button", {
            name: "Documentation"
        });
        await user.click(tableButton);

        expect(window.open).toHaveBeenCalledWith(newDocsURL, "_blank");
    });

    it("Should navigate to home when logo is clicked", async () => {
        useUserStore.getState().setUserDetails({ isLoggedIn: false });
        await renderMainLayout();
        await checkApiCall();

        const logo = await screen.findByAltText(/logo/i);
        await user.click(logo);

        expect(mockUseNavigate).toHaveBeenCalledWith("/");

        useUserStore.getState().setUserDetails({ isLoggedIn: true });
    });

    it("Should display first_name + last_name and navigate to /logout on logout click", async () => {
        useUserStore.getState().setUserDetails({ ...defaultUserDetails });
        await renderMainLayout();
        const userDetails = useUserStore.getState().userDetails;

        const name = await screen.findByText(
            `${userDetails.first_name} ${userDetails.last_name}`
        );
        await user.click(name);

        const logoutButton = await screen.findByRole("button", {
            name: /Log Out/i
        });
        await user.click(logoutButton);

        expect(mockUseNavigate).toHaveBeenCalledWith("/logout");
    });

    it("Should toggle sidebar closed when left panel icon is clicked", async () => {
        await renderMainLayout();
        await checkApiCall();

        const sidebar = document.querySelector('[data-slot="sidebar"]');
        expect(sidebar).toHaveAttribute("data-state", "expanded");

        const toggleButton = await screen.findByTestId("sidebar-toggle");
        expect(toggleButton).toBeInTheDocument();
        await user.click(toggleButton);

        await waitFor(() =>
            expect(
                document.querySelector('[data-slot="sidebar"]')
            ).toHaveAttribute("data-state", "collapsed")
        );
    });

    it("Should open sidebar when toggle button is clicked while sidebar is closed", async () => {
        storeToLocalStorage(SIDEBAR_VIEW_KEY, false);
        await renderMainLayout();
        await checkApiCall();

        const sidebar = document.querySelector('[data-slot="sidebar"]');
        expect(sidebar).toHaveAttribute("data-state", "collapsed");

        const toggleButton = await screen.findByTestId("sidebar-toggle");
        await user.click(toggleButton);

        await waitFor(() => {
            expect(
                document.querySelector('[data-slot="sidebar"]')
            ).toHaveAttribute("data-state", "expanded");
        });
    });

    it("Should close sidebar when toggle button is clicked while sidebar is open", async () => {
        await renderMainLayout();
        await checkApiCall();

        const sidebar = document.querySelector('[data-slot="sidebar"]');
        expect(sidebar).toHaveAttribute("data-state", "expanded");

        const toggleButton = await screen.findByTestId("sidebar-toggle");
        await user.click(toggleButton);

        await waitFor(() => {
            expect(
                document.querySelector('[data-slot="sidebar"]')
            ).toHaveAttribute("data-state", "collapsed");
        });
    });

    // Regression coverage for finding 015: the "Select Project" picker must
    // auto-open only on the empty post-login landing (`/home`), never on
    // workspace-scoped pages that function without a project (`/api-keys`,
    // `/admin/*`), and it must be dismissible on workspace-scoped pages.
    describe("Project picker auto-open (finding 015)", () => {
        beforeEach(() => {
            removeFromLocalStorage(storageKeys.selectedProject);
        });

        afterEach(() => {
            setMockAdminUserIsAdmin(true);
        });

        it("auto-opens the picker on the /home landing when no project is selected", async () => {
            mockLocationStateValue(null, navRoutes.Home);
            await renderMainLayout();
            await checkApiCall();

            expect(
                await screen.findByTestId("project-details")
            ).toBeInTheDocument();
            expect(screen.getByText("Select Project")).toBeInTheDocument();
            expect(screen.queryByText("Models")).not.toBeInTheDocument();
        });

        it("auto-opens the picker on /home/ (trailing slash) too", async () => {
            mockLocationStateValue(null, navRoutes.Home + "/");
            await renderMainLayout();
            await checkApiCall();

            expect(
                await screen.findByTestId("project-details")
            ).toBeInTheDocument();
            expect(screen.getByText("Select Project")).toBeInTheDocument();
        });

        it("lets a projectless /api-keys user dismiss the project picker", async () => {
            mockLocationStateValue(null, navRoutes.ApiKeys);
            await renderMainLayout();
            await checkApiCall();

            // Wait until MainLayout renders past its loading state before
            // asserting the picker and navigation state, so these are not
            // loading-state false negatives.
            await screen.findByTestId("main-layout");
            expect(
                screen.queryByTestId("project-details")
            ).not.toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Models" })
            ).toBeInTheDocument();

            await user.click(screen.getByTestId("project-switcher-button"));

            const dialog = await screen.findByTestId("project-details");
            await user.click(
                within(dialog).getByRole("button", { name: /cancel/i })
            );

            await waitFor(() =>
                expect(
                    screen.queryByTestId("project-details")
                ).not.toBeInTheDocument()
            );
        });

        it("falls back to /home when a project-scoped module is clicked from project-less /api-keys", async () => {
            mockLocationStateValue(null, navRoutes.ApiKeys);
            await renderMainLayout();
            await checkApiCall();

            await user.click(
                await screen.findByRole("button", { name: "Models" })
            );

            expect(mockUseNavigate).toHaveBeenCalledWith(navRoutes.Home);
        });

        it("keeps the main navigation visible on /api-keys/ with a trailing slash", async () => {
            mockLocationStateValue(null, navRoutes.ApiKeys + "/");
            await renderMainLayout();
            await checkApiCall();

            expect(
                await screen.findByRole("button", { name: "Models" })
            ).toBeInTheDocument();
        });

        it("keeps the main navigation hidden on /project/new without a selected project", async () => {
            mockLocationStateValue(null, navRoutes.NewProject);
            await renderMainLayout();
            await checkApiCall();

            await screen.findByTestId("main-layout");
            expect(
                screen.queryByRole("button", { name: "Models" })
            ).not.toBeInTheDocument();
        });

        it("does not auto-open the picker on /admin/members", async () => {
            mockLocationStateValue(null, navRoutes.AdminMembers);
            await renderMainLayout();
            await checkApiCall();

            await screen.findByTestId("main-layout");
            expect(
                screen.queryByTestId("project-details")
            ).not.toBeInTheDocument();
        });

        it("hides the Cancel button for a first-time user with no stored project", async () => {
            setMockAdminUserIsAdmin(false);
            mockLocationStateValue(null, navRoutes.Home);
            await renderMainLayout();
            await checkApiCall();

            const dialog = await screen.findByTestId("project-details");
            expect(
                within(dialog).queryByRole("button", { name: /cancel/i })
            ).not.toBeInTheDocument();
        });
    });
});
