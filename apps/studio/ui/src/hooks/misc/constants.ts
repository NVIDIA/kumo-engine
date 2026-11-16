// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useGetConstantsFlags } from "gen/v0/api/constants/constants";
import { DebugInfo } from "gen/v0/types";

import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useQueryOptions } from "./queryOptions";

export const useDebugInfo = () => {
    const { getQueryOptions } = useQueryOptions();
    const { data, isFetching, refetch } = useGetConstantsFlags(
        getQueryOptions({
            queryKey: [ReactQueryKeys.DebugConfig],
            staleTime: 0
        })
    );

    const getProfileOutputDir = (debugInfo: DebugInfo) => {
        const dbPath = debugInfo?.db_path;
        if (dbPath && debugInfo.release_version) {
            const dataPlane = dbPath.replace(/\/db\/[^/]+$/, "");
            return `${dataPlane}/profiling/${debugInfo.release_version}/`;
        }
        return "";
    };

    const { debugInfo, profileOutputDir } =
        data && data.data
            ? {
                  debugInfo: data.data,
                  profileOutputDir: getProfileOutputDir(data?.data)
              }
            : {
                  debugInfo: null,
                  profileOutputDir: ""
              };

    return { debugInfo, profileOutputDir, isLoading: isFetching, refetch };
};
