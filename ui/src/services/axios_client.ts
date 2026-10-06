// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import axios, { AxiosInstance } from "axios";

import { resolveApiBaseUrl } from "utils/apiBaseUrl";
import { isAuthDisabled } from "utils/authRuntimeConfig";
import {
    publicPaths,
    storageKeys,
    UNEXPECTED_RESPONSE_MESSAGE
} from "utils/Constants";
import { isSessionLoggedIn } from "utils/session";
import { getProfileMode } from "utils/storage";

import tokenService from "./token_service";

const authEndpoint = "v0/authentication/";
let directConnectUnauthorizedHandler: (() => void) | null = null;

export const setDirectConnectUnauthorizedHandler = (
    handler: (() => void) | null
) => {
    directConnectUnauthorizedHandler = handler;
};

export const NETWORK_ERROR_MESSAGE =
    "Could not connect to the Kumo API server. Please " +
    "check your internet connection and disable any ad-blockers. " +
    "If the issue persists, please try again in a few minutes as " +
    "there may be ongoing system maintenance.";

// Detect raw HTML error pages emitted by an upstream proxy (nginx, load
// balancer). nginx gateway pages (502/503/504) start with a bare `<html>`,
// not `<!DOCTYPE`, which is exactly why the raw page leaked (#30031), so match
// both. Kept narrow (full-page markers only) so genuine `<...` error text at
// non-proxy statuses is preserved.
const HTML_PATTERN = /^\s*<(?:!DOCTYPE|html)/i;

/* ----- Axios Client Configuration ----- */

export function getBaseURL() {
    const MODE = process.env.NODE_ENV;
    const WINDOW_ENV = (
        window as unknown as {
            _env_: {
                REACT_APP_BASE_URL_DEV: string;
                REACT_APP_BASE_URL_PROD: string;
            };
        }
    )._env_;
    return resolveApiBaseUrl({
        apiTarget: process.env.API_TARGET,
        mode: MODE,
        // Prefer the build-time variable, then the runtime env.js value.
        developmentBase:
            process.env.REACT_APP_BASE_URL_DEV ||
            WINDOW_ENV?.REACT_APP_BASE_URL_DEV,
        productionBase: WINDOW_ENV?.REACT_APP_BASE_URL_PROD
    });
}

/** Create Axios client instance pointing at the REST API backend
 *
 * Args :
 *
 *    *baseUrl* - Rest API end point
 *
 *    *header*  -  Adding comman header
 */
const axiosClient: AxiosInstance = axios.create({
    baseURL: getBaseURL(),
    headers: {
        "Content-type": "application/json"
    }
});

/**
 * This method can be used to update the `axios instance config`
 * before making any REST API calls
 *
 * Keeps cookie credentials enabled for Direct Connect session auth. When
 * auth is disabled there is no session cookie to send, and the backend
 * refuses credentialed CORS (rest/app.py sets allow_credentials only when
 * auth is enabled), so credentialed cross-origin requests would be blocked
 * by the browser.
 */
axiosClient?.interceptors?.request?.use(
    (config) => {
        config.withCredentials = !isAuthDisabled();
        const profileMode = getProfileMode();
        if (profileMode === "ALL" || config?.url?.startsWith(profileMode)) {
            config.params = { ...config.params, "profile": "true" };
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

/**
 * This method can be used to update the `axios instance config` or `response`
 * before sending response/error to service classes
 *
 * Redirects expired Direct Connect sessions back to login.
 *
 */
axiosClient?.interceptors?.response?.use(
    (res) => {
        return res;
    },
    async (err) => {
        if (axios.isCancel(err)) {
            // Return same error it is for cancel request
            return Promise.reject(err);
        }
        const originalConfig = err.config;
        if (
            originalConfig &&
            !originalConfig.url.includes(authEndpoint) &&
            err.response
        ) {
            if (err.response.status === 401 && !originalConfig._retry) {
                originalConfig._retry = true; //Setting _retry to true to avoid infinite loop

                const currentPath = window.location.pathname;
                const isLoggedIn = isSessionLoggedIn();
                if (isLoggedIn && !publicPaths.includes(currentPath)) {
                    // Storing current location path is to navigate the same page after
                    // login success
                    storeToLocalStorage(
                        storageKeys.navPath,
                        currentPath + window.location.search
                    );
                }
                directConnectUnauthorizedHandler?.();
                tokenService.clear();
                window.location.assign("/login");
                return Promise.reject(err);
            } else if (err.response.status === 403 && !originalConfig._retry) {
                //Invalid token, user should re-login again.
                // Dispatching `logout` event via eventBus to redirect current page to
                // login page
                // TODO(frg100): Think about adding this back in
                // eventBus.dispatch("logout");
            }
        }
        if (err.message === "Network Error") {
            err.message = NETWORK_ERROR_MESSAGE;
        }
        // Sanitize raw HTML error pages from the proxy (e.g. an nginx 502/503/504
        // page) at this single chokepoint so no axios consumer surfaces raw
        // markup; use the shared UNEXPECTED_RESPONSE_MESSAGE (#30031).
        if (
            err.response?.data &&
            typeof err.response.data === "string" &&
            HTML_PATTERN.test(err.response.data)
        ) {
            err.response.data = UNEXPECTED_RESPONSE_MESSAGE;
        }
        return Promise.reject(err);
    }
);

export default axiosClient;
