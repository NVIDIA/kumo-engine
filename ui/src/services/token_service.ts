// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    getFromLocalStorage,
    removeFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";

import { storageKeys } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";

/**
 *  Token service class to handle the state of
 *  access_token and refresh_token.
 */
class TokenService {
    ssoFlow: boolean;
    accessToken: string;
    constructor() {
        this.accessToken = getFromLocalStorage(
            storageKeys.accessToken,
            true
        ) as string;
        this.ssoFlow = false;
    }

    setToken(accessToken: string, refreshToken?: string) {
        this.accessToken = accessToken;
        if (!this.ssoFlow) {
            storeToLocalStorage(storageKeys.accessToken, accessToken, true);
            storeToLocalStorage(storageKeys.refreshToken, refreshToken, true);
        } else {
            storeToLocalStorage(storageKeys.accessToken, accessToken, true);
        }
    }

    setSSOFlow() {
        this.ssoFlow = true;
    }

    getSSOFlow() {
        return (
            this.ssoFlow ||
            (getFromLocalStorage(storageKeys.accessToken, true) &&
                !getFromLocalStorage(storageKeys.refreshToken, true))
        );
    }

    clear() {
        this.accessToken = "";
        removeFromLocalStorage(storageKeys.refreshToken);
        removeFromLocalStorage(storageKeys.accessToken);
    }

    getAccessToken() {
        if (this.ssoFlow) {
            return this.accessToken;
        }
        const refreshToken = getFromLocalStorage(
            storageKeys.refreshToken,
            true
        );
        //Checking refresh token is because to handle logout
        //If user logs out from one tab then it should reflect to all other tabs
        if (isValid(refreshToken)) {
            return this.accessToken;
        }
        return this.accessToken;
    }

    getRefreshToken() {
        const token = getFromLocalStorage<string>(
            storageKeys.refreshToken,
            true
        );
        let refreshToken = "";
        if (isValid(token)) {
            refreshToken = token as string;
        }
        return refreshToken;
    }
}

const tokenService = new TokenService();
export default tokenService;
