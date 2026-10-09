// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    clearSessionStorage,
    getFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { AdminResponse } from "gen/v0/types";
import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

import { storageKeys } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";
import { setSessionLoggedIn } from "utils/session";
import { clearLocalStorage } from "utils/storage";

export interface UserDetails {
    first_name: string;
    last_name: string;
    email: string | undefined;
    isLoggedIn: boolean;
    picture: string | undefined;
    user_id: string | undefined;
    name: string;
}

interface UserState {
    userDetails: UserDetails;
    adminInfo: AdminResponse | null;
}

interface UserActions {
    initializeUser: () => void;
    setUserDetails: (details: Partial<UserDetails>) => void;
    clearUserDetails: () => void;
    setAdminInfo: (info: AdminResponse | null) => void;
    isAdmin: () => boolean;
}

const getDefaultUserDetails = (): UserDetails => ({
    first_name: "",
    last_name: "",
    email: "",
    isLoggedIn: false,
    picture: "",
    user_id: "",
    name: ""
});

const loadUserDetailsFromStorage = (): UserDetails => {
    const details = getFromLocalStorage(storageKeys.userDetails) as {
        [key: string]: string;
    };

    if (isValid(details)) {
        const isLoggedIn =
            getFromLocalStorage(storageKeys.accessToken, true) !== "";

        return {
            first_name: details["first_name"] ?? "",
            last_name: details["last_name"] ?? "",
            email: details["email"] ?? "",
            picture: details["picture"] ?? "",
            user_id: details["user_id"] ?? "",
            name: details["name"] ?? "",
            isLoggedIn
        };
    }
    return getDefaultUserDetails();
};

export const useUserStore = create<UserState & UserActions>()(
    devtools(
        immer((set, get) => ({
            userDetails: getDefaultUserDetails(),
            adminInfo: null,
            initializeUser: () => {
                const details = loadUserDetailsFromStorage();
                set({ userDetails: details });
            },
            setAdminInfo: (info: AdminResponse | null) => {
                set((state) => {
                    state.adminInfo =
                        info === null
                            ? null
                            : {
                                  ...state.adminInfo,
                                  ...info
                              };
                });
            },

            setUserDetails: (details: Partial<UserDetails>) => {
                set((state) => {
                    state.userDetails = {
                        ...state.userDetails,
                        ...details
                    };
                });

                // Persist to localStorage
                storeToLocalStorage(storageKeys.userDetails, get().userDetails);
            },
            clearUserDetails: () => {
                set({ userDetails: getDefaultUserDetails(), adminInfo: null });
                clearSessionStorage();
                clearLocalStorage();
            },
            isAdmin: () => get().adminInfo?.user_is_customer_admin === true
        })),
        {
            name: "UserStore"
        }
    )
);

// Wire the session seam (utils/session) so layer-restricted code (services/,
// utils/) can read the session without importing this store. Mirror
// isLoggedIn on every change.
setSessionLoggedIn(useUserStore.getState().userDetails.isLoggedIn);
useUserStore.subscribe((state) =>
    setSessionLoggedIn(state.userDetails.isLoggedIn)
);

export const useUserDetails = () => useUserStore((state) => state.userDetails);

const resolveUserDisplayName = (details: UserDetails): string => {
    const full = `${details.first_name} ${details.last_name}`.trim();
    return full || details.name || "";
};

export const getCurrentUserDisplayName = (): string =>
    resolveUserDisplayName(useUserStore.getState().userDetails);

export const useIsLoggedIn = () =>
    useUserStore((state) => state.userDetails.isLoggedIn);
