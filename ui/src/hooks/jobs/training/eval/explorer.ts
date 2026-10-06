// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import {
    useGetEvalAvailableXaiEntitiesById,
    useGetEvalEntityHistoricalItemsByEntityId,
    useGetEvalEntityOutputByEntityId
} from "gen/v0/api/evaluation-statistics/evaluation-statistics";
import {
    useGetTrainingJobsAvailableXaiEntitiesById,
    useGetTrainingJobsEntityHistoricalItemsByEntityId,
    useGetTrainingJobsEntityOutputByEntityId,
    useGetTrainingJobsSubgraphByEntityId
} from "gen/v0/api/training-jobs/training-jobs";
import {
    useGetV1BatchPredictionJobsAvailableXaiEntitiesById,
    useGetV1BatchPredictionJobsEntityHistoricalItemsByEntityId,
    useGetV1BatchPredictionJobsEntityOutputByEntityId,
    useGetV1BatchPredictionJobsSubgraphByEntityId
} from "gen/v1/api/batch-prediction-job/batch-prediction-job";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";
import { ReactQueryKeys } from "utils/reactQueryKeys";

const stringifyMappings = (
    mappings?: { [key: string]: Array<string | number> } | null
): { [key: string]: string[] } =>
    getKeys(mappings ?? {}).reduce(
        (acc, key) => {
            acc[key] = (mappings?.[key] ?? []).map(String);
            return acc;
        },
        {} as { [key: string]: string[] }
    );

/**
 * hook for fetching entity categories with the provided query ID.
 * @param queryId - The ID of the query.
 * @returns An object containing the fetched categories and a loading state.
 */

export const useEntityCategories = (
    queryId: string,
    status: string,
    handleError: (error: Error) => void,
    trainingJobId?: string
) => {
    const [categories, setCategories] = useState<{ [key: string]: string[] }>(
        {}
    );
    const [entityIdCategories, setEntityIdCategories] = useState<{
        [key: string]: string[];
    }>({});
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";

    const {
        data: trainingJobsAvailableXaiResponse,
        isLoading: trainingXaiLoading,
        error: trainingError
    } = useGetTrainingJobsAvailableXaiEntitiesById(
        trainingJobId ?? "",
        getQueryOptionsById(
            ReactQueryKeys.TrainingJobsAvailableXaiEntitiesById,
            trainingJobId,
            Boolean(projectId && trainingJobId && status === jobStatus.Done)
        )
    );

    const {
        data: evalAvailableXaiResponse,
        isLoading: evalAXaiLoading,
        error: evalError
    } = useGetEvalAvailableXaiEntitiesById(
        queryId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.EvalAvailableXaiEntitiesById,
            queryId,
            Boolean(
                projectId &&
                !trainingJobId &&
                queryId &&
                status === jobStatus.Done
            )
        )
    );

    const error = (trainingJobId ? trainingError : evalError) as Error | null;

    useEffect(() => {
        if (error) {
            handleError(error);
        }
    }, [error, handleError]);

    const data = trainingJobId
        ? trainingJobsAvailableXaiResponse?.data
        : evalAvailableXaiResponse?.data;

    useEffect(() => {
        if (data) {
            setCategories(data.mappings ?? {});
            setEntityIdCategories(stringifyMappings(data.orig_mappings));
        }
    }, [data]);

    const isLoading = trainingJobId ? trainingXaiLoading : evalAXaiLoading;

    // Filter out categories with no entities
    const filteredCategories = getKeys(categories)?.reduce(
        (acc, key) => {
            if (categories[key].length > 0) {
                acc[key] = categories[key];
            }
            return acc;
        },
        {} as { [key: string]: string[] }
    );

    const filteredEntityIdCategories = getKeys(entityIdCategories)?.reduce(
        (acc, key) => {
            if (entityIdCategories[key].length > 0) {
                acc[key] = entityIdCategories[key];
            }
            return acc;
        },
        {} as { [key: string]: string[] }
    );

    return { filteredCategories, filteredEntityIdCategories, isLoading };
};

/** Per-query enabled flags for the entity-data explorer (training vs eval, +link-pred). */
function explorerEnabledFlags(
    trainingJobId: string | undefined,
    queryId: string,
    selectedEntity: string,
    isDone: boolean,
    taskType: TaskType,
    hasProject: boolean
) {
    const isLinkPrediction =
        taskType === TaskType.LINK_PREDICTION ||
        taskType === TaskType.TEMPORAL_LINK_PREDICTION ||
        taskType === TaskType.STATIC_LINK_PREDICTION;
    const hasEntity = !!selectedEntity;
    const training = hasProject && !!trainingJobId && hasEntity && isDone;
    const evalEnabled =
        hasProject && !trainingJobId && !!queryId && hasEntity && isDone;
    return {
        training,
        evalEnabled,
        trainingLink: training && isLinkPrediction,
        evalLink: evalEnabled && isLinkPrediction
    };
}

export const useEntityData = ({
    queryId,
    selectedEntity,
    taskType,
    status,
    trainingJobId
}: {
    queryId: string;
    selectedEntity: string;
    taskType: TaskType;
    status: string;
    trainingJobId?: string;
}) => {
    const isDone = status === jobStatus.Done;
    const projectId = useProjectId() ?? "";
    const enabled = explorerEnabledFlags(
        trainingJobId,
        queryId,
        selectedEntity,
        isDone,
        taskType,
        Boolean(projectId)
    );
    const { getQueryOptionsByEntityId } = useQueryOptions();

    const {
        data: trainingJobsEntityOutputResponse,
        isLoading: trainingJobsEntityOutputLoading,
        error: trainingJobsEntityOutputError,
        isError: isTrainingJobsEntityOutputError
    } = useGetTrainingJobsEntityOutputByEntityId(
        trainingJobId ?? "",
        selectedEntity,
        getQueryOptionsByEntityId(
            ReactQueryKeys.TrainingJobsEntityOutputByEntityId,
            trainingJobId,
            selectedEntity,
            enabled.training
        )
    );

    const {
        data: evalEntityOutputResponse,
        isLoading: evalEntityOutputLoading,
        error: evalEntityOutputError,
        isError: isEvalEntityOutputError
    } = useGetEvalEntityOutputByEntityId(
        queryId ?? "",
        selectedEntity,
        { project_id: projectId },
        getQueryOptionsByEntityId(
            ReactQueryKeys.EvalEntityOutputByEntityId,
            queryId,
            selectedEntity,
            enabled.evalEnabled
        )
    );

    const {
        data: trainingSubgraph,
        isLoading: trainingSubgraphIsLoading,
        isError: isTrainingSubgraphError,
        error: trainingSubgraphError
    } = useGetTrainingJobsSubgraphByEntityId(
        trainingJobId ?? "",
        selectedEntity,
        getQueryOptionsByEntityId(
            ReactQueryKeys.TrainingJobsSubgraphByEntityId,
            trainingJobId,
            selectedEntity,
            enabled.training
        )
    );

    //Historical Items

    const {
        data: trainingHistoricalItems,
        isLoading: trainingHistoricalItemsIsLoading,
        isError: isTrainingHistoricalItemsError,
        error: trainingHistoricalItemsError
    } = useGetTrainingJobsEntityHistoricalItemsByEntityId(
        trainingJobId ?? "",
        selectedEntity,
        getQueryOptionsByEntityId(
            ReactQueryKeys.TrainingJobsEntityHistoricalItemsByEntityId,
            trainingJobId,
            selectedEntity,
            enabled.trainingLink
        )
    );

    const {
        data: evalHistoricalItems,
        isLoading: evalHistoricalItemsIsLoading,
        isError: isEvalHistoricalItemsError,
        error: evalHistoricalItemsError
    } = useGetEvalEntityHistoricalItemsByEntityId(
        queryId ?? "",
        selectedEntity,
        { project_id: projectId },
        getQueryOptionsByEntityId(
            ReactQueryKeys.EvalEntityHistoricalItemsByEntityId,
            queryId,
            selectedEntity,
            enabled.evalLink
        )
    );

    const entityOutput =
        trainingJobsEntityOutputResponse?.data ||
        evalEntityOutputResponse?.data;
    const isLoading = [
        trainingJobsEntityOutputLoading,
        evalEntityOutputLoading,
        trainingSubgraphIsLoading,
        trainingHistoricalItemsIsLoading,
        evalHistoricalItemsIsLoading
    ].some(Boolean);
    const isError = [
        isTrainingJobsEntityOutputError,
        isEvalEntityOutputError,
        isTrainingSubgraphError,
        isTrainingHistoricalItemsError,
        isEvalHistoricalItemsError
    ].some(Boolean);
    const error = [
        trainingJobsEntityOutputError,
        evalEntityOutputError,
        trainingSubgraphError,
        trainingHistoricalItemsError,
        evalHistoricalItemsError
    ].find(Boolean);

    const subgraph = trainingSubgraph?.data;
    const historicalItems =
        trainingHistoricalItems?.data || evalHistoricalItems?.data;

    return {
        entityOutput,
        historicalItems,
        subgraph,
        isLoading,
        isError,
        error
    };
};

/**
 * Batch Prediction Job Xai
 * Custom hook for fetching entity categories with the provided job ID.
 * @param jobId - The ID of the job.
 * @returns An object containing the fetched categories and a loading state.
 */

export const useBpEntityCategories = (
    jobId: string,
    status: string | undefined
) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const enabled = Boolean(projectId && jobId && status === jobStatus.Done);
    const { data, isLoading } =
        useGetV1BatchPredictionJobsAvailableXaiEntitiesById(
            jobId,
            { project_id: projectId },
            getQueryOptionsById(
                ReactQueryKeys.BpAvailableXaiEntitiesById,
                jobId,
                enabled
            )
        );

    const categories = data?.data?.mappings || {};
    const entityIdCategories = stringifyMappings(data?.data?.orig_mappings);

    // Filter out categories with no entities
    const filteredCategories = getKeys(categories)?.reduce(
        (acc, key) => {
            if (categories[key].length > 0) {
                acc[key] = categories[key];
            }
            return acc;
        },
        {} as { [key: string]: string[] }
    );

    const filteredEntityIdCategories = getKeys(entityIdCategories)?.reduce(
        (acc, key) => {
            if (entityIdCategories[key].length > 0) {
                acc[key] = entityIdCategories[key];
            }
            return acc;
        },
        {} as { [key: string]: string[] }
    );

    return {
        filteredEntityIdCategories,
        filteredCategories,
        isLoading: !enabled || isLoading
    };
};

/**
 * Custom hook to fetch entity data based on the provided job ID and selected entity.
 * @param jobId - The ID of the batch Prediction Job.
 * @param selectedEntity - The selected entity.
 * @returns An object containing the entity output, historical items, and loading state.
 */
export const useBpEntityData = (
    jobId: string,
    selectedEntity: string,
    taskType: TaskType,
    status: string | undefined
) => {
    const projectId = useProjectId() ?? "";
    const enabled =
        Boolean(projectId) && !!selectedEntity && status === jobStatus.Done;
    const { getQueryOptionsByEntityId } = useQueryOptions();

    // Fetch entity output
    const { data: entityOutputData, isLoading: isEntityOutputLoading } =
        useGetV1BatchPredictionJobsEntityOutputByEntityId(
            jobId,
            selectedEntity,
            { project_id: projectId },
            getQueryOptionsByEntityId(
                ReactQueryKeys.BpEntityOutputByEntityId,
                jobId,
                selectedEntity,
                enabled
            )
        );

    // Fetch subgraph
    const { data: subgraphData, isLoading: isSubgraphLoading } =
        useGetV1BatchPredictionJobsSubgraphByEntityId(
            jobId,
            selectedEntity,
            { project_id: projectId },
            getQueryOptionsByEntityId(
                ReactQueryKeys.BpSubgraphByEntityId,
                jobId,
                selectedEntity,
                enabled
            )
        );

    // Fetch historical items only for link prediction tasks
    const shouldFetchHistorical =
        taskType === TaskType.LINK_PREDICTION ||
        taskType === TaskType.TEMPORAL_LINK_PREDICTION ||
        taskType === TaskType.STATIC_LINK_PREDICTION;

    const { data: historicalData, isLoading: isHistoricalLoading } =
        useGetV1BatchPredictionJobsEntityHistoricalItemsByEntityId(
            jobId,
            selectedEntity,
            { project_id: projectId },
            getQueryOptionsByEntityId(
                ReactQueryKeys.HistoricalItemsByEntityId,
                jobId,
                selectedEntity,
                enabled && shouldFetchHistorical
            )
        );

    const isLoading =
        !enabled ||
        isEntityOutputLoading ||
        isSubgraphLoading ||
        (shouldFetchHistorical && isHistoricalLoading);
    return {
        entityOutput: entityOutputData?.data || null,
        historicalItems: shouldFetchHistorical
            ? historicalData?.data || null
            : null,
        subgraph: subgraphData?.data || null,
        isLoading: isLoading
    };
};
