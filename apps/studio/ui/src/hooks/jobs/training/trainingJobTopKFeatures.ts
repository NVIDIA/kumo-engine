// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import { useQueries } from "@tanstack/react-query";
import { AxiosResponse } from "axios";
import {
    getTrainingJobsCategoryAggrFeatureByOp,
    useGetTrainingJobsTopkFeatureByTopk
} from "gen/v0/api/training-jobs/training-jobs";
import { Feature } from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";

import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useTrainingJobTopKFeatures = ({
    jobId,
    entityId,
    category
}: {
    jobId?: string;
    entityId?: string;
    category: string;
}): {
    features: Feature[] | undefined;
    global: { [key: string]: Feature | null };
    isLoading: boolean;
    isError: boolean;
} => {
    const { getQueryOptionsByEntityId } = useQueryOptions();

    const topKQueryOptions = getQueryOptionsByEntityId(
        ReactQueryKeys.TrainingJobsTopkFeatureByTopk,
        jobId,
        entityId || "",
        !!jobId && !!entityId
    );

    const {
        data: topKResponse,
        isLoading: isLoadingTopK,
        isError: isErrorTopK
    } = useGetTrainingJobsTopkFeatureByTopk(jobId || "", entityId || "", 3, {
        ...topKQueryOptions,
        refetchOnMount: false,
        refetchOnWindowFocus: false
    });

    const features = topKResponse?.data as Feature[] | undefined;

    const globalQueryOptions = useMemo(() => {
        if (!features || !jobId) return [];

        return features.map((feature) => {
            const op =
                feature?.stype === "numerical" ? "mean" : "most_frequent";

            return {
                queryKey: [
                    ReactQueryKeys.TrainingJobsCategoryAggrFeatureByOp,
                    jobId,
                    category,
                    feature?.table_name,
                    feature?.name,
                    op
                ],
                queryFn: () =>
                    getTrainingJobsCategoryAggrFeatureByOp(
                        jobId,
                        category,
                        feature?.table_name || "",
                        feature?.name || "",
                        op
                    ),
                enabled: !!jobId && !!feature?.table_name && !!feature?.name,
                retry: false,
                refetchOnMount: false,
                refetchOnWindowFocus: false
            };
        });
    }, [features, jobId, category]);

    // Fetch all global values in parallel
    const globalQueries = useQueries({
        queries: globalQueryOptions
    });

    // Extract stable values from globalQueries
    const globalQueriesData = globalQueries.map((query) => query?.data);
    const isLoadingGlobal = globalQueries.some((query) => query?.isLoading);
    const isErrorGlobal = globalQueries.some((query) => query?.isError);

    const global = useMemo(() => {
        const globalMap: { [key: string]: Feature | null } = {};

        if (!features) return globalMap;

        features.forEach((feature, index) => {
            const key = `${feature?.table_name}.${feature?.name}`;
            const queryData = globalQueriesData[index];

            if (queryData) {
                const response = queryData as AxiosResponse<Feature>;
                if (response?.data) {
                    globalMap[key] = response.data;
                }
            }
        });

        return globalMap;
    }, [features, globalQueriesData]);

    return {
        features,
        global,
        isLoading: isLoadingTopK || isLoadingGlobal,
        isError: isErrorTopK || isErrorGlobal
    };
};
