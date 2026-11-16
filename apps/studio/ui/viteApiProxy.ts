// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type ProxyOptions } from "vite";

const API_PREFIXES = ["/v0", "/v1"] as const;
const FORWARDED_HEADERS = [
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-forwarded-proto"
] as const;

type ProxyEnvironment = Record<string, string | undefined>;

const enabled = (value: string | undefined) =>
    ["1", "true", "yes"].includes((value || "").trim().toLowerCase());

const browserOrigin = (env: ProxyEnvironment) => {
    const port = Number(env.REACT_APP_PORT || 3000);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
        throw new Error("REACT_APP_PORT must be an integer from 1 to 65535");
    }
    return `http://localhost:${port}`;
};

const validateSameOriginBrowserBase = (env: ProxyEnvironment) => {
    const origin = browserOrigin(env);
    for (const name of ["API_TARGET", "REACT_APP_BASE_URL_DEV"] as const) {
        const value = env[name]?.trim();
        if (value && value !== "/") {
            throw new Error(
                `${name} cannot override the local UI API origin. ` +
                    `Use the relative base "/" so the browser remains on ` +
                    `${origin}; the OIDC callback is ` +
                    `${origin}/v0/auth/callback.`
            );
        }
    }
};

const validatedProxyTarget = (value: string | undefined) => {
    if (!value?.trim()) {
        throw new Error(
            "API_PROXY_TARGET is required when UI_LOCAL enables the " +
                "same-origin development proxy"
        );
    }

    let target: URL;
    try {
        target = new URL(value);
    } catch {
        throw new Error("API_PROXY_TARGET must be a valid HTTP(S) origin");
    }
    if (
        !["http:", "https:"].includes(target.protocol) ||
        Boolean(target.username || target.password) ||
        target.pathname !== "/" ||
        Boolean(target.search || target.hash)
    ) {
        throw new Error(
            "API_PROXY_TARGET must be an HTTP(S) origin without credentials, " +
                "a path, a query, or a fragment"
        );
    }
    return target.origin;
};

const configureProxy: NonNullable<ProxyOptions["configure"]> = (proxy) => {
    proxy.on("proxyReq", (proxyRequest) => {
        // The browser-facing Host header is intentionally retained by
        // changeOrigin=false. Do not let callers smuggle a second origin or
        // client address through headers that Direct Connect reads.
        for (const header of FORWARDED_HEADERS) {
            proxyRequest.removeHeader(header);
        }
    });
};

/** Build the sole local browser-to-REST topology.
 *
 * Deployed builds omit UI_LOCAL and continue to rely on their deployment
 * gateway. Local UI builds must use a relative browser API base and an
 * explicit, server-side upstream origin.
 */
export const createLocalApiProxy = (
    env: ProxyEnvironment = process.env
): Record<string, ProxyOptions> | undefined => {
    if (!enabled(env.UI_LOCAL)) return undefined;

    validateSameOriginBrowserBase(env);
    const target = validatedProxyTarget(env.API_PROXY_TARGET);
    return Object.fromEntries(
        API_PREFIXES.map((prefix) => [
            prefix,
            {
                target,
                changeOrigin: false,
                configure: configureProxy
            }
        ])
    );
};
