// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useGetTrainingJobsModelPlansById } from "gen/v0/api/training-jobs/training-jobs";
import { useGetV1PredictiveQueriesModelPlan } from "gen/v1/api/predictive-queries/predictive-queries";
import {
    GetV1PredictiveQueriesModelPlanParams,
    KumoapiGraphGraphDefinition as GraphDefinition,
    QueryDialect,
    RunMode,
    SuggestModelPlanRequest
} from "gen/v1/types";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { JSGraph } from "types/graphs";

import { jobStatus } from "utils/Constants";
import { filterGroupsForSingleColumn } from "utils/graph";
import { ReactQueryKeys } from "utils/reactQueryKeys";

type RequestIdentity = Record<never, never>;

interface DefaultModelPlanResult {
    isLoading: boolean;
    modelPlanYaml: string | null;
    taskType: string | null;
    fetchModelPlan: () => Promise<void>;
}

export const modelPlanGraphDefinition = (graph: JSGraph): GraphDefinition => ({
    tables: graph.tables ?? {},
    col_groups: filterGroupsForSingleColumn(graph.groups ?? [])
});

export const useGetModelPlans = (
    trainingJobId: string,
    status: jobStatus,
    bestExperimentIndex: number
) => {
    const { getModelPlansQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();
    const {
        data: trainingmodelPlansResponse,
        isError: isTrainingModelPlansError,
        error: trainingModelPlansError
    } = useGetTrainingJobsModelPlansById(
        trainingJobId,
        { project_id: projectId ?? "" },
        getModelPlansQueryOptionsById(
            ReactQueryKeys.TrainingJobsModelPlansById,
            trainingJobId,
            status,
            bestExperimentIndex
        )
    );

    useEffect(() => {
        if (isTrainingModelPlansError && trainingModelPlansError) {
            console.error("Unable to get model plans");
        }
    }, [isTrainingModelPlansError, trainingModelPlansError]);

    const data = trainingmodelPlansResponse?.data;

    const plans = data?.model_plans ?? [];
    const options = data?.advanced_options ?? [];

    return { plans, options };
};

/**
 * Fetch the default model plan for the current run mode, graph, query, and
 * project.
 *
 * `fetchModelPlan` reads those hook inputs directly. The plan and task type
 * are published, and loading and error state are updated, only for the latest
 * request while its inputs remain current; superseded responses are ignored.
 */
export const useDefaultModelPlan = (
    runMode: RunMode | null | undefined,
    graph: JSGraph | null | undefined,
    queryText: string | null | undefined,
    triggerError: (error: CustomError) => void
): DefaultModelPlanResult => {
    const projectId = useProjectId();
    const params = useMemo<GetV1PredictiveQueriesModelPlanParams>(
        () => ({
            dialect: QueryDialect.PQLANG_V2,
            project_id: projectId ?? ""
        }),
        [projectId]
    );

    const { mutateAsync } = useGetV1PredictiveQueriesModelPlan();
    const requestKey = useMemo(
        () => [runMode, graph, queryText, params] as const,
        [runMode, graph, queryText, params]
    );
    const currentRequestKey = useRef(requestKey);
    currentRequestKey.current = requestKey;
    const latestRequest = useRef<RequestIdentity | null>(null);
    const [result, setResult] = useState<{
        request: RequestIdentity;
        key: RequestIdentity;
        isLoading: boolean;
        modelPlanYaml: string | null;
        taskType: string | null;
    } | null>(null);

    const fetchModelPlan = useCallback(async () => {
        if (runMode && graph && queryText) {
            const request = {};
            latestRequest.current = request;
            setResult({
                request,
                key: requestKey,
                isLoading: true,
                modelPlanYaml: null,
                taskType: null
            });
            try {
                const requestBody: SuggestModelPlanRequest = {
                    run_mode: runMode,
                    graph_id: "",
                    query_string: queryText,
                    graph_definition: modelPlanGraphDefinition(graph)
                };
                const response = await mutateAsync({
                    data: requestBody,
                    params
                });
                if (
                    request === latestRequest.current &&
                    requestKey === currentRequestKey.current
                ) {
                    setResult({
                        request,
                        key: requestKey,
                        isLoading: false,
                        modelPlanYaml:
                            response.data?.suggested_model_plan_yaml ?? null,
                        taskType:
                            response.data?.suggested_model_plan?.task_type ??
                            null
                    });
                }
            } catch (error) {
                if (
                    request === latestRequest.current &&
                    requestKey === currentRequestKey.current
                ) {
                    setResult(null);
                    // orval v8 defaults TError to void; double-cast is intentional
                    triggerError(error as unknown as CustomError);
                }
            }
        }
    }, [
        mutateAsync,
        runMode,
        graph,
        queryText,
        params,
        requestKey,
        triggerError
    ]);
    const currentResult = result?.key === requestKey ? result : null;

    return {
        isLoading: currentResult?.isLoading ?? false,
        modelPlanYaml: currentResult?.modelPlanYaml ?? null,
        taskType: currentResult?.taskType ?? null,
        fetchModelPlan
    };
};
