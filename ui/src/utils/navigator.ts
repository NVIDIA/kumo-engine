// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import {
    NavigateFunction,
    NavigateOptions,
    To,
    useNavigate
} from "react-router";

import {
    getFromLocalStorage,
    removeFromLocalStorage
} from "@kumo-ai/kumo-ui-catalog";

import { isValid } from "./Helpers/typeGuards";
import { getProjectPath, navRoutes, storageKeys } from "./Constants";
import { getSelectedGroup } from "./storage";

/**
 * Decide and navigate to landing page after login success
 *
 * If any path exists in `local-storage` then navigate to that
 * particular page else navigate to home page
 */
export const navigateToLandingPageAfterLoginSuccess = (
    navigate: NavigateFunction
) => {
    const path = getFromLocalStorage(storageKeys.navPath);
    if (isValid(path)) {
        navigate(path as To);
    } else if (isValid(getFromLocalStorage(storageKeys.selectedProject))) {
        const projectId = getFromLocalStorage(
            storageKeys.selectedProject
        ) as string;
        navigate(getProjectPath(projectId).ModelJobsList);
    } else if (isValid(getSelectedGroup())) {
        navigate(navRoutes.Home);
    } else {
        navigate(navRoutes.Home);
    }
    // Clearing stored navigation path after navigation
    removeFromLocalStorage(storageKeys.navPath);
};

export const useCustomNavigation = () => {
    const navigate = useNavigate();
    //Added navRule to handle Cntrl+Mouse_clicks and cmd+Mouse_clicks
    // (that opens page in a new Tab)in addition to Mouse_clicks.
    // [metaKey represents the cmd button]
    const navRule = (
        path: string | -1,
        options?: NavigateOptions | null,
        event?: React.MouseEvent<HTMLAnchorElement> | MouseEvent
    ) => {
        // if path ===-1 naviagte back to previous page
        if (path === -1) {
            navigate(-1);
        } else if (event && (event.ctrlKey || event.metaKey)) {
            window.open(path, "_blank");
        } else {
            if (options) {
                navigate(path, options);
            } else {
                navigate(path);
            }
        }
    };
    return navRule;
};
