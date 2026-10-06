// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getConstantsFlags } from "gen/v0/api/constants/constants";
import { DebugInfo, FeatureFlags } from "gen/v0/types";
import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

import { setUiSchemaValidationEnabled } from "utils/uiSchemaValidation";

import { useUserStore } from "./userStore";

export type DebugAppInfo = DebugInfo & {
    feature_flags: FeatureFlags;
};

const defaultData: DebugAppInfo = {
    db_path: "",
    release_version: "",
    temporal_url: "",
    job_queue_status: "",
    feature_flags: {
        ENABLE_COLUMN_ANALYSIS: true,
        MAX_TABLES_IN_A_GRAPH: 12,
        IS_DATABRICKS_DEPLOY: false,
        ENABLE_UI_SCHEMA_VALIDATION: false
    }
};

interface DebugAppState {
    data: DebugAppInfo;
    configFetched: boolean;
}
interface DebugAppActions {
    fetchData: () => Promise<void>;
    reset: () => void;
}

export const useDebugStore = create<DebugAppState & DebugAppActions>()(
    devtools(
        immer((set, get, store) => ({
            data: defaultData,
            configFetched: false,

            fetchData: async () => {
                const isLoggedIn =
                    useUserStore.getState().userDetails.isLoggedIn;
                if (!isLoggedIn) {
                    return;
                }

                try {
                    const flagsResponse = await getConstantsFlags();

                    if (flagsResponse?.data) {
                        set((state) => ({
                            data: {
                                ...state.data,
                                ...flagsResponse.data
                            }
                        }));
                    }

                    set({ configFetched: true });
                } catch (err) {
                    console.error("Failed fetching debug info", err);
                    set({ configFetched: true });
                }
            },
            reset: () => {
                set(store.getInitialState());
            }
        })),
        {
            name: "DebugStore"
        }
    )
);

useDebugStore.subscribe((state) => {
    setUiSchemaValidationEnabled(
        state.data.feature_flags.ENABLE_UI_SCHEMA_VALIDATION ?? false
    );
});
