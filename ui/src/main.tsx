// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 *  Main application module. Loaded through src/entry.ts (the index.html
 *  entry), which imports the Trusted-Types policy bootstrap before this
 *  module.
 */
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";

import {
    getFromLocalStorage,
    storeToLocalStorage,
    Toaster
} from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { useUserStore } from "stores/userStore";

import ReactQueryDevTool from "components/Debug/ReactQueryDevTool";
import { useValidationSchemaToastListener } from "components/Validation/ValidationSchemaToast";

import { initDirectConnect } from "hooks/auth/directConnectAuth";

import DirectConnectLogout from "pages/Login/DirectConnectLogout";

import { resolveAuthRoutes } from "routes/authRouteOverrides";

import { defaultUserDetails } from "services/auth";
import { setDirectConnectUnauthorizedHandler } from "services/axios_client";
import tokenService from "services/token_service";

import { isAuthDisabled, loadAuthClientConfig } from "utils/authRuntimeConfig";
import { storageKeys } from "utils/Constants";
import { queryClient } from "utils/queryClient";
import reportWebVitals from "utils/reportWebVitals";
import { inDebugMode } from "utils/storage";

import "@kumo-ai/kumo-ui-catalog/kumo-ui-catalog.css";
import "main.css";

import "utils/trustedSecurityPolicies";

const CI_SCREENSHOT_AUTH_KEY = "__ciScreenshotAuth";

/** To remove console logs from Production build */
if (process.env.NODE_ENV === "production" && !inDebugMode()) {
    console.log = () => {};

    console.error = () => {};

    console.debug = () => {};

    console.trace = () => {};
}

/**
 *  React DOM gets the root element from `ui/index.html`
 *  and renders the react UI.
 *
 */

const root = ReactDOM.createRoot(
    document.getElementById("root") as HTMLElement
);

// Runtime accessibility checks (development only). Logs axe-core violations to
// the browser console while developing so a11y regressions surface early. The
// whole block is tree-shaken out of production builds via import.meta.env.DEV.
// The MODE guard excludes the Vitest environment (jsdom has no
// requestIdleCallback, which @axe-core/react relies on).
if (import.meta.env.DEV && import.meta.env.MODE !== "test") {
    void Promise.all([
        import("react"),
        import("react-dom"),
        import("@axe-core/react")
    ]).then(([react, reactDom, axe]) => {
        axe.default(react.default, reactDom.default, /* debounce ms */ 1000);
    });
}

interface AuthFlags {
    authDisabled: boolean;
    ciScreenshotAuthEnabled: boolean;
}

const getAuthFlags = (): AuthFlags => {
    const authDisabled = isAuthDisabled();
    const ciScreenshotAuthEnabled =
        getFromLocalStorage<boolean>(CI_SCREENSHOT_AUTH_KEY) === true;

    return {
        authDisabled,
        ciScreenshotAuthEnabled
    };
};

const isDirectConnectLogoutPath = (flags: AuthFlags) =>
    !flags.authDisabled && window.location.pathname === "/logout";

setDirectConnectUnauthorizedHandler(() => {
    useUserStore.getState().clearUserDetails();
});

const initializeBypassAuthState = (flags: AuthFlags) => {
    // When auth is disabled, set up a logged-in session immediately so
    // PrivateRoute and other auth-gated components work.
    if (flags.authDisabled) {
        tokenService.setToken("AUTH_DISABLED", "AUTH_DISABLED");
        storeToLocalStorage(storageKeys.userDetails, defaultUserDetails);
        useUserStore.getState().setUserDetails(defaultUserDetails);
    } else if (flags.ciScreenshotAuthEnabled) {
        const ciAccessToken = getFromLocalStorage<string>(
            storageKeys.accessToken,
            true
        );
        const ciUserDetails = getFromLocalStorage<typeof defaultUserDetails>(
            storageKeys.userDetails
        );

        if (ciAccessToken) {
            tokenService.setSSOFlow();
            tokenService.setToken(ciAccessToken);
            useUserStore.getState().setUserDetails({
                ...defaultUserDetails,
                ...ciUserDetails,
                isLoggedIn: true
            });
        }
    }
};

export const Base = ({
    router
}: {
    router: ReturnType<typeof createBrowserRouter>;
}) => {
    useValidationSchemaToastListener();

    return (
        <div className="main-content bg-grey">
            <RouterProvider router={router} />
            <Toaster position="bottom-right" richColors />
            <ReactQueryDevTool />
        </div>
    );
};

const renderRoot = async () => {
    const { routes } = await import("routes/Routes");
    const flags = getAuthFlags();
    const router = createBrowserRouter(
        resolveAuthRoutes(routes, {
            authDisabled: flags.authDisabled
        })
    );

    root.render(
        <QueryClientProvider client={queryClient}>
            <Base router={router} />
        </QueryClientProvider>
    );
};

const renderRootWithErrorHandling = () =>
    renderRoot().catch((err) => {
        console.error("[main] render failed", err);
    });

const renderDirectConnectLogout = () => {
    root.render(
        <QueryClientProvider client={queryClient}>
            <div className="main-content bg-grey">
                <DirectConnectLogout />
                <Toaster position="bottom-right" richColors />
            </div>
        </QueryClientProvider>
    );
};

const bootstrap = async () => {
    await loadAuthClientConfig();
    const flags = getAuthFlags();
    initializeBypassAuthState(flags);

    if (isDirectConnectLogoutPath(flags)) {
        renderDirectConnectLogout();
    } else if (!flags.authDisabled && !flags.ciScreenshotAuthEnabled) {
        initDirectConnect().finally(() => renderRootWithErrorHandling());
    } else {
        renderRootWithErrorHandling();
    }
};

bootstrap().catch((err) => {
    console.error("[main] bootstrap failed", err);
    renderRootWithErrorHandling();
});

/**
 *
 *  To measure any of the supported metrics, you only need to pass a function
 *  into the reportWebVitals function reportWebVitals(console.log);
 *
 *  Example:
 *  ```function sendToAnalytics({ id, name, value }) {
 *         //Send email implementation
 *     }
 *
 *     reportWebVitals(sendToAnalytics); ```
 *
 *  This function is fired when the final values for any of the metrics
 *  have finished calculating on the page. You can use it to log any of the
 *  results to the console or send to a particular endpoint.
 *
 */
reportWebVitals();
