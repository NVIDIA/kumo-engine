// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import { useLocation, useNavigate } from "react-router";

import {
    AdminPanelSettings,
    Cards,
    Cube,
    EnhancedEncryption,
    ImportContacts,
    Loupe,
    NetworkNode,
    Power,
    Settings,
    SideBarData,
    SoundSampler
} from "@kumo-ai/kumo-ui-catalog";
import { ProjectDBEntry } from "gen/v0/types";
import { useUserDetails } from "stores/userStore";

import { useProjectId } from "hooks/project/projects";
import { useSidebarLinksPrefetch } from "hooks/routes/route";

import { getProjectPath, navRoutes, newDocsURL } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";
import {
    getDefaultModelJobsPageURL,
    getDefaultPredictionJobsPageURL
} from "utils/jobs";

interface UseSidebarDataProps {
    hideMenu: boolean;
    projectInfo?: ProjectDBEntry | null;
    isAdmin?: boolean;
}

const computeSidebarContext = (
    projectInfo: ProjectDBEntry | null | undefined,
    userDetails: ReturnType<typeof useUserDetails>,
    navigationProjectId: ReturnType<typeof useProjectId>
) => {
    const isProjectOwner =
        !!projectInfo?.id && projectInfo.created_by === userDetails?.user_id;
    const projectContextId = projectInfo?.id ?? navigationProjectId;

    return {
        isProjectOwner,
        adminRoute: navRoutes.AdminMembers,
        projectPaths: projectContextId
            ? getProjectPath(projectContextId)
            : null,
        defaultModelListRoute: getDefaultModelJobsPageURL(projectContextId),
        defaultPredictionListRoute:
            getDefaultPredictionJobsPageURL(projectContextId)
    };
};

/**
 * Custom hook to generate sidebar navigation data.
 * Memoizes the sidebar configuration to prevent unnecessary re-renders.
 */
export const useSidebarData = ({
    hideMenu,
    projectInfo,
    isAdmin
}: UseSidebarDataProps): SideBarData => {
    const navigate = useNavigate();
    const location = useLocation();
    const handleSidebarLinkHover = useSidebarLinksPrefetch();
    const userDetails = useUserDetails();
    const navigationProjectId = useProjectId();
    const {
        isProjectOwner,
        adminRoute,
        projectPaths,
        defaultModelListRoute,
        defaultPredictionListRoute
    } = computeSidebarContext(projectInfo, userDetails, navigationProjectId);

    const sideBarData = useMemo<SideBarData>(() => {
        // Helper to create navigation item with common patterns
        const createNavItem = (
            title: string,
            route: string,
            icon: typeof Cube,
            options?: {
                prefetchKey?: string;
                sliceRoute?: boolean;
                externalLink?: boolean;
            }
        ) => {
            const { prefetchKey, externalLink = false } = options || {};
            const isActive =
                location.pathname === route ||
                location.pathname?.startsWith(route) ||
                location.pathname
                    ?.split("/?")[0]
                    ?.startsWith(route.split("/?")[0]) ||
                (location.pathname + "/")?.startsWith(route);

            return {
                title,
                icon,
                show: true,
                isActive,
                to: route,
                onClick: externalLink
                    ? () => window.open(route, "_blank")
                    : () => navigate(route),
                ...(prefetchKey && {
                    onMouseEnter: () => handleSidebarLinkHover(prefetchKey),
                    onFocus: () => handleSidebarLinkHover(prefetchKey)
                })
            };
        };

        const buildUserProfile = () => ({
            name:
                isValid(userDetails?.first_name) &&
                isValid(userDetails?.last_name)
                    ? `${userDetails?.first_name} ${userDetails?.last_name}`
                    : userDetails?.name,
            email: userDetails?.email as string,
            avatar: userDetails?.picture,
            show: true,
            onLogoutClick: () => {
                navigate("/logout");
            }
        });

        const buildSettingsItems = () => [
            {
                title: "Project Settings",
                icon: Settings,
                show: isProjectOwner,
                isActive: location.pathname?.startsWith(
                    projectPaths?.ProjectSettings ?? navRoutes.Home
                ),
                to: projectPaths?.ProjectSettings ?? navRoutes.Home,
                onClick: () => {
                    navigate(projectPaths?.ProjectSettings ?? navRoutes.Home);
                }
            },
            {
                title: "API Keys",
                icon: EnhancedEncryption,
                show: true,
                to: navRoutes.ApiKeys,
                isActive: location.pathname?.startsWith(navRoutes.ApiKeys),
                onClick: () => {
                    navigate(navRoutes.ApiKeys);
                }
            },
            {
                title: "Admin",
                icon: AdminPanelSettings,
                show: isAdmin,
                isActive: location.pathname?.startsWith(navRoutes.Admin),
                to: adminRoute,
                onClick: () => {
                    navigate(adminRoute);
                }
            }
        ];

        return {
            hideTrigger: hideMenu,
            userProfile: buildUserProfile(),
            navMainMenuGroup: hideMenu
                ? []
                : [
                      {
                          items: [
                              createNavItem(
                                  "Models",
                                  defaultModelListRoute,
                                  Cube,
                                  {
                                      prefetchKey: "Models",
                                      sliceRoute: true
                                  }
                              ),
                              createNavItem(
                                  "Predictions",
                                  defaultPredictionListRoute,
                                  SoundSampler,
                                  {
                                      prefetchKey: "Predictions",
                                      sliceRoute: true
                                  }
                              )
                          ]
                      },
                      {
                          items: [
                              createNavItem(
                                  "Graphs",
                                  projectPaths?.Graphs ?? navRoutes.Home,
                                  NetworkNode,
                                  {
                                      prefetchKey: "Graphs"
                                  }
                              ),
                              createNavItem(
                                  "Tables",
                                  projectPaths?.Tables ?? navRoutes.Home,
                                  Cards,
                                  { prefetchKey: "Tables" }
                              ),
                              createNavItem(
                                  "Connectors",
                                  projectPaths?.Connectors ?? navRoutes.Home,
                                  Power,
                                  {
                                      prefetchKey: "Connectors"
                                  }
                              )
                          ]
                      },
                      {
                          items: buildSettingsItems(),
                          hideSeparator: true
                      }
                  ],
            navFooterMenu: hideMenu
                ? []
                : [
                      {
                          ...createNavItem(
                              "Documentation",
                              newDocsURL,
                              ImportContacts,
                              { externalLink: true }
                          ),
                          isActive: false,
                          show: true,
                          className: "pb-6"
                      }
                  ],
            navQuickStart: hideMenu
                ? []
                : [
                      {
                          title: "New",
                          icon: Loupe,
                          childs: [
                              {
                                  title: "Model",
                                  description:
                                      "Train a model with a graph schema and a predictive query.",
                                  icon: Cube,
                                  to:
                                      projectPaths?.NewTrainingGraphSelect ??
                                      navRoutes.Home,
                                  onClick: () => {
                                      navigate(
                                          projectPaths?.NewTrainingGraphSelect ??
                                              navRoutes.Home
                                      );
                                  }
                              },
                              {
                                  title: "Prediction",
                                  description:
                                      "Generate predictions using a trained model and graph.",
                                  icon: SoundSampler,
                                  to:
                                      projectPaths?.NewPredictionJob ??
                                      navRoutes.Home,
                                  onClick: () => {
                                      navigate(
                                          projectPaths?.NewPredictionJob ??
                                              navRoutes.Home
                                      );
                                  },
                                  onMouseEnter: () => {
                                      handleSidebarLinkHover("NewPrediction");
                                  },
                                  onFocus: () => {
                                      handleSidebarLinkHover("NewPrediction");
                                  }
                              }
                          ]
                      }
                  ],
            onLogoClick: () => {
                navigate("/");
            },
            onLogoMouseEnter: () => {
                handleSidebarLinkHover("Connectors");
            },
            onLogoFocus: () => {
                handleSidebarLinkHover("Connectors");
            }
        };
    }, [
        hideMenu,
        userDetails?.first_name,
        userDetails?.last_name,
        userDetails?.name,
        userDetails?.email,
        userDetails?.picture,
        defaultModelListRoute,
        defaultPredictionListRoute,
        projectPaths?.NewTrainingGraphSelect,
        projectPaths?.NewPredictionJob,
        projectPaths?.Graphs,
        projectPaths?.Tables,
        projectPaths?.Connectors,
        projectPaths?.ProjectSettings,
        isAdmin,
        location.pathname,
        isProjectOwner,
        adminRoute,
        navigate,
        handleSidebarLinkHover
    ]);

    return sideBarData;
};
