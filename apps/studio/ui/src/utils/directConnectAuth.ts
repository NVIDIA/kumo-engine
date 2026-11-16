// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DirectConnectLoginMethod } from "utils/authRuntimeConfig";
import { getAuthClientConfig } from "utils/authRuntimeConfig";

export interface DirectConnectPasswordLoginResponse {
    status: "ok" | "password_reset_required";
    reset_token?: string | null;
}

const DIRECT_CONNECT_PASSWORD_MIN_LENGTH = 12;
const DIRECT_CONNECT_PASSWORD_DENIED_TERMS = [
    "password",
    "temporary",
    "admin",
    "kumo"
];
const DIRECT_CONNECT_PASSWORD_TOO_WEAK = "Password is too easy to guess";

export const validateDirectConnectPassword = (
    password: string,
    userEmail = ""
): string | null => {
    if (!password) return "Password is required";
    if (password.length < DIRECT_CONNECT_PASSWORD_MIN_LENGTH) {
        return `Password must be at least ${DIRECT_CONNECT_PASSWORD_MIN_LENGTH} characters`;
    }
    if (new Set(password).size === 1) return DIRECT_CONNECT_PASSWORD_TOO_WEAK;
    if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) {
        return "Password must include at least one letter and one number";
    }

    const normalizedPassword = password.toLowerCase();
    if (
        DIRECT_CONNECT_PASSWORD_DENIED_TERMS.some((term) =>
            normalizedPassword.includes(term)
        )
    ) {
        return DIRECT_CONNECT_PASSWORD_TOO_WEAK;
    }

    const emailLocalPart = userEmail.trim().toLowerCase().split("@", 1)[0];
    if (
        emailLocalPart.length >= 3 &&
        normalizedPassword.includes(emailLocalPart)
    ) {
        return "Password must not contain your email address";
    }
    return null;
};

export const getDirectConnectLoginMethod = (): DirectConnectLoginMethod =>
    getAuthClientConfig()?.direct_connect_login_method || "oidc";

export const isDirectConnectPasswordLogin = () =>
    getDirectConnectLoginMethod() === "password";

export type { DirectConnectLoginMethod };

export interface DirectConnectMe {
    user_id: string;
    email: string;
    display_name: string;
    role: "admin" | "user";
    permissions: string[];
    kumo_group_ids: string[];
    project_ids: string[];
}
