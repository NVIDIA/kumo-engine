// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import { useGetTrainingJobsWarningsById } from "gen/v0/api/training-jobs/training-jobs";
import { ErrWarnDetails, GetTrainingJobsWarningsById200 } from "gen/v0/types";
import {
    useGetV1PredictiveQueriesById,
    useGetV1PredictiveQueriesMetadataById,
    useValidateV1PredictiveQueries
} from "gen/v1/api/predictive-queries/predictive-queries";
import { ErrorDetail, PQueryResponse } from "gen/v1/types";
import { useCacheStaleTime } from "stores/cacheStore";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { CreateTrainingJobArgs } from "types/jobs";
import {
    PredictiveQueryMetadata,
    QueryType,
    TaskType
} from "types/predictiveQueries";
import { HandleErrorFunc } from "types/validation";
import { CTAInfo } from "types/warnings";

import { CtaSourceName } from "utils/Constants";
import { getCta } from "utils/ctaFunctions";
import { parseErrorMessage } from "utils/errorHandler";
import { filterGroupsForSingleColumn } from "utils/graph";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useQueryDefinition = (
    queryId: string | null,
    queryYamlFromPage?: string | null
) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const [queryYaml, setQueryYaml] = useState<string>("");

    const { data: resourceResponse } = useGetV1PredictiveQueriesById<{
        data: PQueryResponse;
    }>(
        queryId as string,
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.PredictiveQueriesResourceById,
            queryId,
            Boolean(queryId && !queryYamlFromPage && projectId)
        )
    );

    useEffect(() => {
        if (resourceResponse?.data?.query_string) {
            setQueryYaml(resourceResponse.data.query_string);
        }
    }, [resourceResponse]);

    return { queryYaml, setQueryYaml };
};

export const useQueryMetaData = (queryId: string | null) => {
    const { long } = useCacheStaleTime();
    const projectId = useProjectId() ?? "";

    const {
        data: response,
        isError,
        error
    } = useGetV1PredictiveQueriesMetadataById<{
        data: PredictiveQueryMetadata;
    }>(
        queryId as string,
        {
            project_id: projectId
        },
        {
            query: {
                enabled: !!queryId && Boolean(projectId),
                staleTime: long
            }
        }
    );
    const taskType = response?.data?.task_type ?? TaskType.NONE;
    const queryType = response?.data?.query_type ?? QueryType.STATIC;

    return { taskType, queryType, isError, error };
};

// Training-job ids are minted as `trainingjob-<hex>` (see
// `TrainingJobID.is_valid_id` in `statedb/config/common.py`).
const TRAINING_JOB_ID_PREFIX = "trainingjob-";

export const useGetQueryWarnings = (
    id: string | null,
    handleError: HandleErrorFunc,
    trainingStatus?: string | null
): [CTAInfo | null, () => void, ErrWarnDetails[]] => {
    const { queryOption } = useQueryOptions();
    const queryClient = useQueryClient();
    const projectId = useProjectId() ?? "";
    const [ctaMsgs, setCtaMsgs] = useState<CTAInfo | null>(null);
    const [warnings, setWarnings] = useState<ErrWarnDetails[]>([]);
    // `POST /v0/training_jobs/{id}/warnings` only resolves training-job ids;
    // any other job-id type (e.g. a `gen-traintable-job-*` id) is a
    // guaranteed 404, so treat it as "no id" and skip the request
    // (kumo-ai/kumo#30119).
    const trainingJobId =
        id && id.startsWith(TRAINING_JOB_ID_PREFIX) ? id : null;
    const warningsQueryOptions = queryOption.trainingJobsWarningsById(
        trainingJobId,
        trainingStatus
    );

    const { data: response } = useGetTrainingJobsWarningsById<{
        data: GetTrainingJobsWarningsById200;
    }>(
        trainingJobId || "",
        { project_id: projectId },
        {
            query: {
                ...warningsQueryOptions.query,
                enabled:
                    warningsQueryOptions.query.enabled && Boolean(projectId)
            }
        },
        queryClient
    );
    const hideWarnings = () => {
        setCtaMsgs((prev) =>
            prev
                ? {
                      ...prev,
                      showCTAWarnings: false
                  }
                : null
        );
    };

    useEffect(() => {
        if (response?.data && trainingJobId) {
            try {
                const [success, cta] = getCta(
                    CtaSourceName.TrainingJob,
                    trainingJobId,
                    handleError,
                    response.data
                );
                if (success) {
                    setCtaMsgs(cta);
                    if (cta.warnings && cta.warnings.length > 0) {
                        setWarnings([...cta.warnings]);
                    }
                }
            } catch (err) {
                handleError(err as CustomError);
            }
        }
    }, [response, trainingJobId, handleError]);
    return [ctaMsgs, hideWarnings, warnings];
};

type ValidationResult = {
    error: ErrorDetail[];
    warnings: ErrorDetail[];
};

export const useQueryValidation = () => {
    const projectId = useProjectId();
    const { mutateAsync: validateQuery } = useValidateV1PredictiveQueries();

    const validateQueryWithArgs = useCallback(
        async (
            createJobArgs: CreateTrainingJobArgs
        ): Promise<ValidationResult> => {
            let error: ErrorDetail[] = [];
            let warnings: ErrorDetail[] = [];

            try {
                const groups = createJobArgs.graph?.groups ?? [];
                const col_groups = filterGroupsForSingleColumn(groups);
                const response = await validateQuery({
                    data: {
                        pquery_resource: {
                            name: "test_pq_" + new Date().getTime(), // Generating random name for validation
                            query_string: createJobArgs?.query_yaml,
                            graph: {
                                tables: createJobArgs.graph?.tables ?? {},
                                col_groups: col_groups
                            }
                        },
                        model_plan: createJobArgs?.model_planner_yaml
                    },
                    params: { project_id: projectId ?? "" }
                });
                if (response?.data?.items) {
                    error = response?.data?.items
                        ?.map((e) => {
                            if (e?.type === 0) {
                                // 0 means errors
                                return {
                                    title: e?.title ?? "",
                                    description: e?.description ?? "",
                                    type: e?.type ?? "",
                                    cta: e?.cta ?? null
                                };
                            } else return null;
                        })
                        ?.filter((e) => e !== null);
                    warnings = response?.data?.items
                        ?.map((e) => {
                            if (e?.type === 1) {
                                //  1 means warnigs
                                return {
                                    title: e?.title ?? "",
                                    description: e?.description ?? "",
                                    type: e?.type ?? "",
                                    cta: e?.cta ?? null
                                };
                            } else return null;
                        })
                        ?.filter((e) => e !== null);
                }
            } catch (err) {
                // We want to handle errors differently for formatting
                const errorText = `Unexpected validation error: ${parseErrorMessage(
                    err as CustomError
                )}. Please file a bug.`;
                error = [
                    {
                        title: "Error in Predictive Query / Model Planner",
                        description: errorText,
                        type: 0,
                        cta: null
                    }
                ];
            }

            return { error, warnings };
        },
        [projectId, validateQuery]
    );

    return { validateQuery: validateQueryWithArgs };
};
