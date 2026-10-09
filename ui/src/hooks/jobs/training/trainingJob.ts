// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCancelTrainingJobsById,
    useCreateTrainingJobsTrain,
    useGetTrainingJobsWorkflowById
} from "gen/v0/api/training-jobs/training-jobs";
import {
    JobType,
    KumoapiModelPlanTrainingTableGenerationPlan,
    StatedbConfigCommonJobStatus as JobStatus,
    TrainingWorkflowResponse
} from "gen/v0/types";
import { ModelPlan } from "gen/v1/types";
import yaml from "js-yaml";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { AxiosResponse, CustomError } from "types/common";
import { CreateTrainingJobArgs } from "types/jobs";
import { PredictiveQueryInfo } from "types/predictiveQueries";
import { HandleErrorFunc } from "types/validation";

import {
    jobStatus,
    JobStatusType,
    SUPPORT_CONTACT_MESSAGE
} from "utils/Constants";
import { parseTrainingWorkflowResponse } from "utils/Converters/predictiveQueriesConverter";
import { parseErrorMessage } from "utils/errorHandler";
import { filterGroupsForSingleColumn } from "utils/graph";
import {
    invalidateJobQueries,
    invalidateJobsList,
    invalidateTrainingTags
} from "utils/queryInvalidation";

export const useCreateTrainingJob = (
    triggerAlert: (title: string, description: string) => void
) => {
    const queryClient = useQueryClient();
    const { mutateAsync: createTrainingJobMutation } =
        useCreateTrainingJobsTrain({}, queryClient);
    const projectId = useProjectId();

    const createTrainingJob = useCallback(
        async (
            createJobArgs: CreateTrainingJobArgs,
            jobTags: Record<string, string>
        ): Promise<{ jobId: string | null; error: string }> => {
            let jobId: string | null = null;
            let error: string = "";

            try {
                const createArgs = { ...createJobArgs };
                // Transform request body for Orval client
                const modelPlanYaml = yaml.load(createArgs?.model_planner_yaml);
                const groups = createJobArgs.graph?.groups ?? [];
                const col_groups = filterGroupsForSingleColumn(groups);

                const response = await createTrainingJobMutation({
                    data: {
                        predictive_query: createArgs?.query_yaml,
                        graph_definition: {
                            tables: createArgs.graph?.tables ?? {},
                            col_groups: col_groups
                        },
                        training_table_gen_plan:
                            (
                                modelPlanYaml as {
                                    training_table_generation: KumoapiModelPlanTrainingTableGenerationPlan;
                                }
                            )?.training_table_generation ?? "",
                        model_plan: modelPlanYaml as ModelPlan,
                        job_tags: jobTags,
                        user: createArgs.user,
                        project_id: projectId ?? "",
                        ...(createArgs.base_graph_name
                            ? { base_graph_name: createArgs.base_graph_name }
                            : {})
                    }
                });

                if (response?.data) {
                    jobId = response.data;
                    invalidateJobsList(queryClient);
                    invalidateTrainingTags(queryClient);
                } else {
                    const errorText = `Unexpected validation error: ${SUPPORT_CONTACT_MESSAGE}`;
                    error = errorText;
                    triggerAlert("Request failed!", errorText);
                }
            } catch (e) {
                const errorText = `Unexpected validation error: ${parseErrorMessage(
                    e as CustomError
                )}. `;
                error = errorText;
                triggerAlert("Request failed!", errorText);
            }

            return { jobId, error };
        },
        [createTrainingJobMutation, queryClient, triggerAlert, projectId]
    );

    return { createTrainingJob };
};

interface UseCancelTrainingJobProps {
    jobId: string;
    onSuccess?: () => void;
    handleError: (error: CustomError, title: string) => void;
}

export const useCancelTrainingJob = ({
    jobId,
    onSuccess,
    handleError
}: UseCancelTrainingJobProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectId();

    const { mutate: cancelTraining } = useCancelTrainingJobsById(
        {
            mutation: {
                onSuccess: () => {
                    invalidateJobQueries(
                        queryClient,
                        JobType.TRAINING_JOB,
                        jobId
                    );
                    invalidateJobsList(queryClient);
                    onSuccess?.(); // Calls setActiveTab("") After success
                },
                onError: (error) => {
                    if (error) {
                        const title = "Unable to Cancel Training. ";
                        handleError(error as CustomError, title);
                    }
                }
            }
        },
        queryClient
    );

    return {
        cancelTraining: ({ id }: { id: string }) => {
            return cancelTraining({
                id,
                params: { project_id: projectId ?? "" }
            });
        }
    };
};

export const useTrainingJobWorkflow = ({
    id,
    setStatus,
    handleError,
    setPageLoading,
    setIsArchived,
    refetchTrigger
}: {
    id: string;
    setStatus?: ((status: jobStatus) => void) | undefined;
    handleError: HandleErrorFunc;
    setPageLoading: (v: boolean) => void;
    setIsArchived?: React.Dispatch<React.SetStateAction<boolean>>;
    refetchTrigger?: number;
}) => {
    const [queryInfo, setQueryInfo] = useState<PredictiveQueryInfo | null>(
        null
    );
    const [trainingStatus, setTrainingStatus] = useState<JobStatus | null>(
        null
    );
    const [prevStatus, setPrevStatus] = useState<JobStatusType>(jobStatus.None);
    const queryClient = useQueryClient();
    const pollQueuingCountRef = useRef(10);
    const hasErrorOccurredRef = useRef(false);

    useEffect(() => {
        if (id) {
            pollQueuingCountRef.current = 10;
            hasErrorOccurredRef.current = false;
        }
    }, [id]);

    // staleTime (cache invalidation) depends on job status
    const { queryOption } = useQueryOptions();
    const projectId = useProjectId();

    const shouldPoll = useCallback((status: string | undefined) => {
        if (hasErrorOccurredRef.current) return false;
        if (!status) return false;

        if (
            status === jobStatus.Done ||
            status === jobStatus.Failed ||
            status === jobStatus.Cancelled
        ) {
            return false;
        }

        if (status === jobStatus.Queued && pollQueuingCountRef.current <= 0) {
            return false;
        }

        return true;
    }, []);

    const getRefetchInterval = useCallback(
        (data: TrainingWorkflowResponse | undefined) => {
            const status = data?.training_job?.execution?.status;

            if (!shouldPoll(status)) {
                return false;
            }
            if (
                status === jobStatus.Queued &&
                pollQueuingCountRef.current > 0
            ) {
                pollQueuingCountRef.current--;
                return 5000;
            }

            if (status === jobStatus.Running) {
                return 15000;
            }

            return 15000;
        },
        [shouldPoll]
    );

    const {
        data: workflowResponse,
        isLoading,
        error,
        isError,
        refetch
    } = useGetTrainingJobsWorkflowById(
        id,
        { project_id: projectId ?? "" },
        queryOption.trainingJobsWorkflowById(
            id,
            trainingStatus as jobStatus,
            (query) => {
                const resp = query?.state
                    ?.data as AxiosResponse<TrainingWorkflowResponse>;
                return getRefetchInterval(resp?.data);
            }
        )
    );

    useEffect(() => {
        if (refetchTrigger && refetchTrigger > 0 && workflowResponse) {
            refetch();
        }
    }, [refetchTrigger, refetch, workflowResponse]);

    useEffect(() => {
        if (isLoading) {
            setPageLoading(true);
            return;
        }

        if (isError) {
            invalidateJobsList(queryClient);
            setPageLoading(false);
            hasErrorOccurredRef.current = true;
            handleError(error as CustomError);
            return;
        }

        if (workflowResponse?.data) {
            const parsedData = parseTrainingWorkflowResponse(
                workflowResponse?.data
            );
            setQueryInfo(parsedData);
            if (setIsArchived) setIsArchived(!!parsedData.archived);

            const status =
                workflowResponse?.data?.training_job?.execution?.status;
            if (status) {
                if (prevStatus !== jobStatus.None && prevStatus !== status) {
                    invalidateJobsList(queryClient);
                }
                if (
                    prevStatus === jobStatus.None &&
                    (status === jobStatus.Queued ||
                        status === jobStatus.Running)
                ) {
                    // Invalidate specific job cache on status change
                    // fresh data is fetched and UI updates immediately
                    // for all transitions: Queued->Running, Running->Done, etc.
                    invalidateJobQueries(queryClient, JobType.TRAINING_JOB, id);
                }

                setPrevStatus(status);
                setTrainingStatus(status);
                if (setStatus) setStatus(status as jobStatus);
            }
        } else {
            setQueryInfo(null);
            setTrainingStatus(null);
        }

        setPageLoading(false);
    }, [
        workflowResponse,
        isLoading,
        isError,
        error,
        handleError,
        setPageLoading,
        setStatus,
        setIsArchived,
        prevStatus,
        queryClient,
        id
    ]);

    return {
        queryInfo: queryInfo,
        isLoading,
        isError,
        error,
        refetch,
        workflowResponse
    };
};
