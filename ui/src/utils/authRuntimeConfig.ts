// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GetAuthenticationClientConfigResponse } from "gen/v0/schemas/authentication";
import type {
    ClientAuthConfigResponse,
    ClientAuthConfigResponseDirectConnectLoginMethod
} from "gen/v0/types";

import { resolveApiBaseUrl } from "./apiBaseUrl";

export type DirectConnectLoginMethod = "oidc" | "password";

export interface AuthClientConfig extends Omit<
    ClientAuthConfigResponse,
    "direct_connect_login_method"
> {
    auth_disabled: boolean;
    direct_connect_login_method?: ClientAuthConfigResponseDirectConnectLoginMethod;
}

interface WindowAuthEnv {
    AUTH_DISABLED?: boolean | string;
    REACT_APP_BASE_URL_DEV?: string;
    REACT_APP_BASE_URL_PROD?: string;
}

let authClientConfig: AuthClientConfig | null = null;

const windowEnv = (): WindowAuthEnv =>
    (window as unknown as { _env_?: WindowAuthEnv })._env_ || {};

const normalizeLoginMethod = (
    value?: string | null
): DirectConnectLoginMethod =>
    value?.toLowerCase() === "password" ? "password" : "oidc";

const truthyValue = (value?: boolean | string) =>
    value === true || String(value).toLowerCase() === "true";

export const getAuthClientConfig = () => authClientConfig;

export const setAuthClientConfig = (config: AuthClientConfig | null) => {
    authClientConfig = config
        ? {
              auth_disabled: Boolean(config.auth_disabled),
              direct_connect_login_method:
                  config.direct_connect_login_method == null
                      ? null
                      : normalizeLoginMethod(config.direct_connect_login_method)
          }
        : null;
};

export const isAuthDisabled = () => {
    const runtimeConfig = getAuthClientConfig();
    if (runtimeConfig) return runtimeConfig.auth_disabled;
    return truthyValue(windowEnv().AUTH_DISABLED);
};

const getRuntimeConfigBaseURL = () => {
    const env = windowEnv();
    return (
        resolveApiBaseUrl({
            apiTarget: process.env.API_TARGET,
            mode: process.env.NODE_ENV,
            developmentBase:
                process.env.REACT_APP_BASE_URL_DEV ||
                env.REACT_APP_BASE_URL_DEV,
            productionBase: env.REACT_APP_BASE_URL_PROD
        }) || ""
    );
};

export const loadAuthClientConfig = async () => {
    if (typeof fetch !== "function") return null;
    if (process.env.NODE_ENV === "test" && !process.env.API_TARGET) {
        return null;
    }

    try {
        const runtimeBaseURL = getRuntimeConfigBaseURL().replace(/\/$/, "");
        const response = await fetch(
            `${runtimeBaseURL}/v0/authentication/client-config`,
            { credentials: "include" }
        );
        if (!response.ok) return null;
        setAuthClientConfig(
            GetAuthenticationClientConfigResponse.parse(
                (await response.json()) as ClientAuthConfigResponse
            )
        );
        return getAuthClientConfig();
    } catch {
        return null;
    }
};
