// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    getFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { useUserStore } from "stores/userStore";

import { defaultUserDetails } from "services/auth";
import { getBaseURL } from "services/axios_client";
import tokenService from "services/token_service";

import { storageKeys } from "utils/Constants";
import {
    DirectConnectMe,
    DirectConnectPasswordLoginResponse
} from "utils/directConnectAuth";
import { getDirectConnectReturnTo } from "utils/directConnectReturnPath";
import {
    getFromSessionStorage,
    removeFromSessionStorage,
    storeToSessionStorage
} from "utils/storage";

const restUrl = (path: string) => {
    const baseUrl = getBaseURL() || window.location.origin;
    return `${baseUrl.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
};

const directConnectUserDetails = (me: DirectConnectMe) => {
    const displayName = me.display_name || me.email;
    const [firstName, ...rest] = displayName.split(" ");
    return {
        ...defaultUserDetails,
        first_name: firstName || "",
        last_name: rest.join(" "),
        email: me.email,
        isLoggedIn: true,
        user_id: me.user_id,
        name: displayName
    };
};

export const isDirectConnectLoggedOut = () =>
    getFromSessionStorage<string>(storageKeys.directConnectLoggedOut) === "1";

export const clearDirectConnectLoggedOut = () => {
    removeFromSessionStorage(storageKeys.directConnectLoggedOut);
};

export const initDirectConnect = async (): Promise<boolean> => {
    try {
        const response = await fetch(restUrl("v0/auth/me"), {
            credentials: "include"
        });
        if (!response.ok) {
            useUserStore.getState().clearUserDetails();
            tokenService.clear();
            return false;
        }
        const me = (await response.json()) as DirectConnectMe;
        const userDetails = directConnectUserDetails(me);
        tokenService.setSSOFlow();
        tokenService.setToken("DIRECT_CONNECT");
        storeToLocalStorage(storageKeys.userDetails, userDetails);
        useUserStore.getState().setUserDetails(userDetails);
        return true;
    } catch {
        useUserStore.getState().clearUserDetails();
        tokenService.clear();
        return false;
    }
};

const postDirectConnectJson = async <T>(
    path: string,
    body: unknown
): Promise<T> => {
    const response = await fetch(restUrl(path), {
        method: "POST",
        credentials: "include",
        headers: {
            "content-type": "application/json"
        },
        body: JSON.stringify(body)
    });
    if (!response.ok) {
        throw new Error(await response.text());
    }
    return (await response.json()) as T;
};

export const loginDirectConnectWithPassword = async (
    email: string,
    password: string
): Promise<DirectConnectPasswordLoginResponse> =>
    postDirectConnectJson<DirectConnectPasswordLoginResponse>(
        "v0/auth/password/login",
        { email, password }
    );

export const completeDirectConnectPasswordReset = async (
    resetToken: string,
    newPassword: string
): Promise<DirectConnectPasswordLoginResponse> =>
    postDirectConnectJson<DirectConnectPasswordLoginResponse>(
        "v0/auth/password/complete-reset",
        { reset_token: resetToken, new_password: newPassword }
    );

export const loginDirectConnect = () => {
    clearDirectConnectLoggedOut();
    const returnTo = getDirectConnectReturnTo(
        getFromLocalStorage<string>(storageKeys.navPath),
        window.location.pathname + window.location.search
    );
    window.location.assign(
        restUrl(`v0/auth/login?return_to=${encodeURIComponent(returnTo)}`)
    );
};

export const logoutDirectConnect = async (): Promise<boolean> => {
    let response: Response;
    try {
        response = await fetch(restUrl("v0/auth/logout"), {
            method: "POST",
            credentials: "include"
        });
    } catch {
        return false;
    }
    if (!response.ok) {
        return false;
    }
    useUserStore.getState().clearUserDetails();
    tokenService.clear();
    storeToSessionStorage(storageKeys.directConnectLoggedOut, "1");
    window.location.assign("/login?logged_out=1");
    return true;
};
