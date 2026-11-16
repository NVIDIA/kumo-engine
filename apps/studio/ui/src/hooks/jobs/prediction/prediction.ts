// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { useWithDebounce } from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import {
    useGetGenpredtableJobsAnchorTimeById,
    useGetGenpredtableJobsErrorsById
} from "gen/v0/api/jobs/jobs";
import {
    ColStatsResponse,
    ErrWarnDetails,
    JobType,
    KumoapiCommonJobStatus,
    OutputPresignedUrlResponse
} from "gen/v0/types";
import {
    useCancelV1BatchPredictionJobsById,
    useCreateV1BatchPredictionJobs,
    useGetV1BatchPredictionJobsById,
    useGetV1BatchPredictionJobsDownloadBpById,
    useGetV1BatchPredictionJobsPredictionAnchorTimeById,
    useGetV1BatchPredictionJobsPredictionStatsById,
    useGetV1BatchPredictionJobsWarnings
} from "gen/v1/api/batch-prediction-job/batch-prediction-job";
import {
    useCreateV1PredictiveQueriesInsertFilterOverridesByTrainingJobId,
    useGetV1PredictiveQueriesAllowedBpQueryOverridesByTrainingJobId
} from "gen/v1/api/predictive-queries/predictive-queries";
import {
    BatchPredictionInternalRequestOutputsItem,
    BatchPredictionJobAnchorTime,
    BatchPredictionJobStats,
    EditableQuery,
    ErrorDetail,
    KumoapiJobsADLSPredictionOutput as ADLSPredictionOutput,
    KumoapiJobsBigQueryPredictionOutput as BigQueryPredictionOutput,
    KumoapiJobsDatabricksPredictionOutput as DatabricksPredictionOutput,
    KumoapiJobsGCSPredictionOutput as GCSPredictionOutput,
    KumoapiJobsPredictionArtifactType as PredictionArtifactType,
    KumoapiJobsS3PredictionOutput as S3PredictionOutput,
    KumoapiJobsSnowflakePredictionOutput as SnowflakePredictionOutput
} from "gen/v1/types";

import {
    BatchPredictionJobOutputJs,
    UnlinkedBatchPredictionJobData
} from "types/batchPredictions";
import { CustomError, HighLightText, ValidationResponse } from "types/common";
import { ConnectorDropdownOption, InputState } from "types/inputs";
import { PredictionJobErrorType } from "types/predictionJob";
import { ExtendedColStatsResponse } from "types/snapshots";
import { ColumnStatsAndSampleRows } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { jobStatus, JobStatusType, outputStorage } from "utils/Constants";
import { parseUnlinkedPredictionJobs } from "utils/Converters/batchPredictionsConverter";
import { getFiltersFromBpOverride } from "utils/Converters/predictiveQueriesConverter";
import { parseErrorMessage } from "utils/errorHandler";
import { isValid, Nullable } from "utils/Helpers/typeGuards";
import {
    filePathForRootedObjectStoreConnector,
    isADLSConnectorConfig,
    isGCSConnectorConfig,
    isRootedObjectStoreConnector,
    isS3ConnectorConfig
} from "utils/objectStoreTablePath";
import {
    invalidateJobQueries,
    invalidateJobsList,
    invalidatePredictionTags
} from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { getScopedRoutes } from "utils/scopedRoutes";

import { useQueryOptions } from "../../misc/queryOptions";
import { useProjectId } from "../../project/projects";

export const useBatchPredictionSummary = (
    bpId: string,
    handleError: HandleErrorFunc,
    setLoading: (isLoading: boolean) => void,
    refetchTrigger?: number,
    setIsArchived?: (isArchived: boolean) => void
) => {
    const [batchPredictionInfo, setBatchPredictionInfo] =
        useState<UnlinkedBatchPredictionJobData | null>(null);
    const [bpJobStatus, setJobStatus] = useState<JobStatusType>(jobStatus.None);
    const { queryOption } = useQueryOptions();

    const [queryId, setQueryId] = useState<string | null>(null);
    const [prevStatus, setPrevStatus] = useState<JobStatusType>(jobStatus.None);
    const queryClient = useQueryClient();
    const projectId = useProjectId();

    const {
        data: response,
        isLoading,
        error,
        isError
    } = useGetV1BatchPredictionJobsById(
        bpId,
        { project_id: projectId ?? "" },
        queryOption.batchPredictionJobsById(bpId, refetchTrigger)
    );

    useEffect(() => {
        if (isLoading) return;

        if (isError) {
            invalidateJobsList(queryClient);
            setLoading(false);
            handleError(error as CustomError);
            return;
        }

        const data = response?.data;
        if (!data) {
            setLoading(false);
            handleError(
                {
                    message: "Empty response from Batch Prediction API"
                } as CustomError,
                "Something went wrong!"
            );
            return;
        }

        // Success
        setLoading(false);
        const status = data.execution?.status;
        if (prevStatus !== jobStatus.None && prevStatus !== status) {
            invalidateJobsList(queryClient);
        }
        if (
            prevStatus === jobStatus.None &&
            (status === jobStatus.Queued || status === jobStatus.Running)
        ) {
            // Invalidate specific job cache on status change
            // fresh data is fetched and UI updates immediately
            // for all transitions: Queued->Running, Running->Done, etc.
            invalidateJobQueries(
                queryClient,
                JobType.BATCH_PREDICTION_JOB,
                bpId as string
            );
        }

        setPrevStatus(status);
        setJobStatus(status);

        const [bpInfo] = parseUnlinkedPredictionJobs([data], projectId);
        setBatchPredictionInfo(bpInfo);
        if (setIsArchived) {
            setIsArchived(!!bpInfo.archived);
        }
        const qid = data.predictive_query_link?.id;
        if (qid) setQueryId(qid);
    }, [
        response?.data,
        isLoading,
        isError,
        error,
        handleError,
        setLoading,
        setIsArchived,
        queryClient,
        prevStatus,
        bpId,
        projectId
    ]);

    return {
        isDataLoading: isLoading,
        batchPredictionInfo,
        bpJobStatus,
        queryId,
        graphId: batchPredictionInfo?.graph_link?.id || null
    };
};

export const useBatchPredictionDownloadBp = (
    bpId: string,
    handleError: HandleErrorFunc,
    status: JobStatusType | undefined,
    downloadSupported: boolean,
    setLoading?: (isLoading: boolean) => void
) => {
    const [s3PresignedUrl, setS3PresignedUrl] = useState<string>("");

    const queryClient = useQueryClient();
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();
    const {
        data: response,
        isLoading,
        error,
        isError
    } = useGetV1BatchPredictionJobsDownloadBpById<{
        data: OutputPresignedUrlResponse;
    }>(
        bpId,
        { project_id: projectId ?? "" },
        getQueryOptionsById(
            ReactQueryKeys.BatchPredictionJobsDownloadBpById,
            bpId,
            downloadSupported && !!bpId && status === jobStatus.Done
        ),
        queryClient
    );

    useEffect(() => {
        setLoading?.(false);

        if (isError) {
            handleError(error as CustomError);
            return;
        }

        if (response?.data) {
            setS3PresignedUrl(response.data.presigned_url);
        }
    }, [isError, error, response, setLoading, handleError]);

    return {
        isDataLoading: isLoading,
        s3PresignedUrl,
        isError,
        error
    };
};

export const useBatchPredictionAnchorTime = (
    bpId: string,
    source: string,
    handleError: HandleErrorFunc,
    setLoading: (loading: boolean) => void,
    generatePredTableJobId: string
) => {
    const queryClient = useQueryClient();
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();

    const uiQuery =
        useGetV1BatchPredictionJobsPredictionAnchorTimeById<BatchPredictionJobAnchorTime>(
            bpId,
            { project_id: projectId ?? "" },
            getQueryOptionsById(
                ReactQueryKeys.BatchPredictionJobsPredictionAnchorTimeById,
                bpId,
                source === "UI" && !!bpId
            ),
            queryClient
        );

    const sdkQuery = useGetGenpredtableJobsAnchorTimeById<{
        data: string | null;
    }>(
        generatePredTableJobId,
        { project_id: projectId ?? "" },
        getQueryOptionsById(
            ReactQueryKeys.PredtableJobsAnchorTimeById,
            generatePredTableJobId,
            source === "SDK" && !!generatePredTableJobId
        ),
        queryClient
    );

    // Choose active query
    const activeQuery = source === "UI" ? uiQuery : sdkQuery;

    const { data: response, isLoading, error, isError, refetch } = activeQuery;

    useEffect(() => {
        let retries = 10;

        const poll = async () => {
            if (retries <= 0 || response) return;

            try {
                await refetch();
            } catch (err) {
                console.warn("Anchor time fetch error:", err);
            }

            retries--;
            if (!response && retries > 0) {
                setTimeout(poll, 10000);
            }
        };

        if (
            (source === "UI" && bpId) ||
            (source === "SDK" && generatePredTableJobId)
        ) {
            poll();
        }

        return () => {
            retries = 0;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [bpId, generatePredTableJobId, source]);

    useEffect(() => {
        if (isError) {
            setLoading(false);
            handleError(error as CustomError);
        }
    }, [isError, error, handleError, setLoading]);

    // stop loading when data arrives
    useEffect(() => {
        setLoading(isLoading);
    }, [isLoading, setLoading]);

    let normalizedAnchorTime: string | null = null;

    if (source === "UI" && response && "anchor_time" in response) {
        normalizedAnchorTime = response.anchor_time ?? null;
    } else if (source === "SDK" && response && "data" in response) {
        normalizedAnchorTime = response.data ?? null;
    }

    return {
        isDataLoading: isLoading,
        anchorTime: normalizedAnchorTime
    };
};

interface UseAllowedBpQueryOverridesProps {
    selectedJobId: string;
    setEntityFilter: (value: string) => void;
    setTargetFilter: (value: string) => void;
}
export const useAllowedBpQueryOverrides = ({
    selectedJobId,
    setEntityFilter,
    setTargetFilter
}: UseAllowedBpQueryOverridesProps) => {
    const queryClient = useQueryClient();
    const { getQueryOptionsById } = useQueryOptions();

    const {
        data: response,
        isError,
        error
    } = useGetV1PredictiveQueriesAllowedBpQueryOverridesByTrainingJobId<{
        data: EditableQuery;
    }>(
        selectedJobId,
        getQueryOptionsById(
            ReactQueryKeys.BpQueryOverridesByTrainingJobId,
            selectedJobId
        ),
        queryClient
    );

    useEffect(() => {
        if (!response?.data) return;

        const filters = getFiltersFromBpOverride(response.data);

        if (filters?.entity_filter_override) {
            setEntityFilter(filters.entity_filter_override);
        }
        if (filters?.target_filter_override) {
            setTargetFilter(filters.target_filter_override);
        }
    }, [response?.data, setEntityFilter, setTargetFilter]);

    useEffect(() => {
        if (isError) {
            console.error(
                "Error in getAllowedBpQueryOverride",
                (error as CustomError)?.message
            );
        }
    }, [isError, error]);

    const filterArgs = response?.data
        ? getFiltersFromBpOverride(response.data)
        : undefined;

    return {
        entityFilter: filterArgs?.entity_filter_override ?? "",
        targetFilter: filterArgs?.target_filter_override ?? ""
    };
};
interface UsePredictionFilterOverridesProps {
    selectedTrainingJobId?: string;
    entityFilter: string;
    targetFilter: string;
    setFilterErrors: (errors: ErrorDetail[]) => void;
    setFilterValidState: (state: InputState) => void;
    setHighLightQuery: React.Dispatch<React.SetStateAction<HighLightText>>;
}

export const usePredictionFilterOverrides = ({
    selectedTrainingJobId,
    entityFilter,
    targetFilter,
    setFilterErrors,
    setFilterValidState,
    setHighLightQuery
}: UsePredictionFilterOverridesProps) => {
    const [highLightQueryText, setHighLightQueryText] = useState<string>("");

    const { mutate } =
        useCreateV1PredictiveQueriesInsertFilterOverridesByTrainingJobId({
            mutation: {
                onSuccess: (response) => {
                    if (!response?.data) return;

                    const { text, editable_fields, validation_response } =
                        response.data;

                    const highlight: HighLightText = {
                        text,
                        position:
                            editable_fields
                                ?.map((field) => {
                                    const pos = field.bold_text_position;
                                    return Array.isArray(pos) &&
                                        pos.length === 2 &&
                                        typeof pos[0] === "number" &&
                                        typeof pos[1] === "number"
                                        ? { start: pos[0], end: pos[1] }
                                        : null;
                                })
                                .filter(
                                    (
                                        item
                                    ): item is { start: number; end: number } =>
                                        item !== null
                                ) ?? []
                    };

                    setHighLightQuery(highlight);
                    setHighLightQueryText(text);

                    if (validation_response) {
                        const v = validation_response as ValidationResponse;
                        const errors = v.errors ?? [];
                        const warnings = v.warnings ?? [];

                        const errorsList: ErrorDetail[] = [];

                        if (errors.length > 0) {
                            errorsList.push({
                                title: "Prediction Filter is invalid.",
                                description: errors
                                    .map((e) => e.message)
                                    .join(", "),
                                type: PredictionJobErrorType.ERROR,
                                cta: null
                            });
                        }

                        if (warnings.length > 0) {
                            errorsList.push({
                                title: "Prediction Filter has warnings.",
                                description: warnings
                                    .map((w) => w.message)
                                    .join(", "),
                                type: PredictionJobErrorType.WARNING,
                                cta: null
                            });
                        }

                        setFilterErrors(errorsList);

                        setFilterValidState(
                            errors.length > 0
                                ? InputState.InValid
                                : warnings.length > 0
                                  ? InputState.Warning
                                  : InputState.None
                        );
                    }
                },
                onError: (error: CustomError) => {
                    setFilterErrors([
                        {
                            title: "Prediction Filter insertion error.",
                            description: parseErrorMessage(error),
                            type: PredictionJobErrorType.ERROR,
                            cta: null
                        }
                    ]);
                    setFilterValidState(InputState.InValid);
                }
            }
        });

    const triggerValidation = useCallback(() => {
        if (!selectedTrainingJobId || !isValid(selectedTrainingJobId)) return;

        setFilterValidState(InputState.Validating);

        mutate({
            trainingJobId: selectedTrainingJobId,
            data: {
                entity_filter: entityFilter,
                target_filter: targetFilter
            }
        });
    }, [
        selectedTrainingJobId,
        entityFilter,
        targetFilter,
        setFilterValidState,
        mutate
    ]);

    const debouncedTriggerValidation = useWithDebounce(triggerValidation, 1);

    useEffect(() => {
        debouncedTriggerValidation();
    }, [debouncedTriggerValidation]);

    return { highLightQueryText };
};

export const useGetBpColumnStatsAndSampleRows = (
    bpId: string | null,
    handleError: (err: Error) => void
): {
    columnStatsAndSampleRows: ColumnStatsAndSampleRows | null;
    transformedStatsAndSampleRows: ColumnStatsAndSampleRows | null;
} => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();
    const { data, isLoading, error, isError } =
        useGetV1BatchPredictionJobsPredictionStatsById<{
            data: BatchPredictionJobStats;
        }>(
            bpId ?? "",
            { project_id: projectId ?? "" },
            getQueryOptionsById(
                ReactQueryKeys.BatchPredictionJobsPredictionStatsById,
                bpId
            )
        );
    let columnStatsAndSampleRows: ColumnStatsAndSampleRows | null = null;
    let transformedStatsAndSampleRows: ColumnStatsAndSampleRows | null = null;

    const buildStats = (
        stats: Nullable<ColStatsResponse>
    ): ColumnStatsAndSampleRows => {
        if (!stats) return { loadingStats: true };
        return stats.statistics_available
            ? { columnStats: stats, loadingStats: false, gotError: false }
            : { loadingStats: false };
    };
    if (isError) {
        columnStatsAndSampleRows = transformedStatsAndSampleRows = {
            columnStats: {} as ExtendedColStatsResponse,
            loadingStats: false,
            gotError: true
        };
    } else if (isLoading) {
        columnStatsAndSampleRows = transformedStatsAndSampleRows = {
            loadingStats: true
        };
    } else if (data) {
        const { raw_predictions, transformed_predictions } = data.data;
        columnStatsAndSampleRows = buildStats(raw_predictions);
        transformedStatsAndSampleRows = buildStats(transformed_predictions);
    }

    useEffect(() => {
        if (isError) {
            handleError(error as CustomError);
        }
    }, [error, isError, handleError]);

    return { columnStatsAndSampleRows, transformedStatsAndSampleRows };
};

export const useCreateBatchPredictionJob = (
    handleError: (error: CustomError, title?: string) => void,
    setError: React.Dispatch<React.SetStateAction<ErrorDetail[]>>,
    setLoading: React.Dispatch<React.SetStateAction<boolean>>,
    setValidationState: React.Dispatch<React.SetStateAction<InputState>>,
    projectId?: string | null
) => {
    const queryClient = useQueryClient();
    const navigate = useNavigate();

    const triggerAlert = useCallback(
        (title: string, description: string) => {
            setLoading(false);
            setValidationState(InputState.ValidationError);
            const valError: ErrorDetail[] = [];
            const error: CustomError = {
                response: { data: description },
                name: "",
                message: description
            };
            valError.push({ title, description, type: 0, cta: null });
            setError(valError);
            handleError(error, title);
        },
        [handleError, setError, setLoading, setValidationState]
    );

    const mutation = useCreateV1BatchPredictionJobs({
        mutation: {
            onSuccess: (resp) => {
                if (resp?.data?.id) {
                    invalidateJobsList(queryClient);
                    invalidatePredictionTags(queryClient);
                    const predictionJobBasePath =
                        getScopedRoutes(projectId).PredictionJob;
                    navigate(
                        `${predictionJobBasePath}${resp.data.id}?job_type=${JobType.BATCH_PREDICTION_JOB}`
                    );
                } else {
                    triggerAlert(
                        "Request failed!",
                        "Unexpected validation error. Please contact support."
                    );
                }
            },
            onError: (error: CustomError) => {
                setValidationState(InputState.ValidationError);
                triggerAlert(
                    "Request failed!",
                    `Unexpected validation error: ${parseErrorMessage(error)}`
                );
            }
        }
    });

    return mutation;
};

const createS3PredictionOutput = (
    output: BatchPredictionJobOutputJs,
    connector: ConnectorDropdownOption,
    connectorId: string
): S3PredictionOutput | null => {
    if (!isS3ConnectorConfig(connector.config)) return null;
    let directory = output.directory;
    if (isValid(directory) && !directory.endsWith("/")) directory += "/";
    const filePath = output.table_name
        ? directory + output.table_name + output.extension
        : directory;
    try {
        return {
            storage_type: "S3",
            artifact_type: output.artifact_type as PredictionArtifactType,
            file_path: filePathForRootedObjectStoreConnector(
                filePath,
                connector.config
            ),
            connector_id: connectorId,
            extra_fields: []
        };
    } catch {
        return null;
    }
};

const createCloudFilePredictionOutput = (
    destination: outputStorage.GCS | outputStorage.ADLS,
    output: BatchPredictionJobOutputJs,
    connector: ConnectorDropdownOption,
    connectorId: string
): GCSPredictionOutput | ADLSPredictionOutput | null => {
    const config = connector.config;
    if (
        (destination === outputStorage.GCS && !isGCSConnectorConfig(config)) ||
        (destination === outputStorage.ADLS && !isADLSConnectorConfig(config))
    ) {
        return null;
    }
    if (!isGCSConnectorConfig(config) && !isADLSConnectorConfig(config)) {
        return null;
    }
    if (!isRootedObjectStoreConnector(config)) return null;
    const directory = output.directory?.replace(/\/+$/, "");
    const filePath = directory
        ? `${directory}/${output.table_name}${output.extension}`
        : `${output.table_name}${output.extension}`;
    try {
        const common = {
            artifact_type: output.artifact_type as PredictionArtifactType,
            file_path: filePathForRootedObjectStoreConnector(filePath, config),
            connector_id: connectorId,
            extra_fields: []
        };
        return destination === outputStorage.GCS
            ? ({ ...common, storage_type: "GCS" } satisfies GCSPredictionOutput)
            : ({
                  ...common,
                  storage_type: "ADLS"
              } satisfies ADLSPredictionOutput);
    } catch {
        return null;
    }
};

const createPredictionOutput = (
    destination: string,
    output: BatchPredictionJobOutputJs,
    connector: ConnectorDropdownOption
): BatchPredictionInternalRequestOutputsItem | null => {
    if (!connector.value) return null;
    const connectorId = String(connector.value);
    switch (destination) {
        case outputStorage.S3:
            return createS3PredictionOutput(output, connector, connectorId);
        case outputStorage.GCS:
        case outputStorage.ADLS:
            return createCloudFilePredictionOutput(
                destination,
                output,
                connector,
                connectorId
            );
        case outputStorage.Snowflake:
            return {
                storage_type: "SNOWFLAKE",
                artifact_type: output.artifact_type as PredictionArtifactType,
                connector_id: connectorId,
                table_name: output.table_name,
                extra_fields: []
            } satisfies SnowflakePredictionOutput;
        case outputStorage.Bigquery:
            return {
                storage_type: "BIGQUERY",
                artifact_type: output.artifact_type as PredictionArtifactType,
                connector_id: connectorId,
                table_name: output.table_name,
                extra_fields: []
            } satisfies BigQueryPredictionOutput;
        case outputStorage.Databricks:
            return {
                storage_type: "DATABRICKS",
                artifact_type: output.artifact_type as PredictionArtifactType,
                connector_id: connectorId,
                table_name: output.table_name,
                extra_fields: []
            } satisfies DatabricksPredictionOutput;
        default:
            return null;
    }
};

export const usePredictionOutputs = (
    destination: string,
    outputs: Record<string, BatchPredictionJobOutputJs>,
    selectedConnector?: ConnectorDropdownOption | null
): BatchPredictionInternalRequestOutputsItem[] =>
    useMemo(() => {
        if (!selectedConnector) return [];
        return Object.values(outputs).flatMap((output) => {
            const item = createPredictionOutput(
                destination,
                output,
                selectedConnector
            );
            return item ? [item] : [];
        });
    }, [destination, outputs, selectedConnector]);

interface UseCancelPredictionJobProps {
    bpId: string;
    onSuccess?: () => void;
    handleError: (error: CustomError, title: string) => void;
}
export const useCancelPredictionJob = ({
    bpId,
    onSuccess,
    handleError
}: UseCancelPredictionJobProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectId();

    const { mutate: cancelPrediction } = useCancelV1BatchPredictionJobsById(
        {
            mutation: {
                onSuccess: () => {
                    invalidateJobQueries(
                        queryClient,
                        JobType.BATCH_PREDICTION_JOB,
                        bpId
                    );
                    invalidateJobsList(queryClient);
                    onSuccess?.();
                },
                onError: (error) => {
                    if (error) {
                        const title = "Unable to Cancel Prediction Job. ";
                        handleError(error as CustomError, title);
                    }
                }
            }
        },
        queryClient
    );

    return {
        cancelPrediction: ({ id }: { id: string }) => {
            return cancelPrediction({
                id,
                params: { project_id: projectId ?? "" }
            });
        }
    };
};

export const useGenPredErrorsById = (
    id: string,
    status: KumoapiCommonJobStatus | undefined
) => {
    const queryClient = useQueryClient();
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();

    const { data, isLoading } = useGetGenpredtableJobsErrorsById(
        id,
        { project_id: projectId ?? "" },
        getQueryOptionsById(
            ReactQueryKeys.GenPredErrorsById + "-" + status,
            id,
            Boolean(id)
        ),
        queryClient
    );
    const { warnings, errors } = useMemo(() => {
        return {
            errors: data?.data?.items?.filter((t) => `${t.type}` !== `1`) ?? [],
            warnings:
                (data?.data?.items
                    ?.filter((t) => `${t.type}` === `1`)
                    .map((t) => ({
                        title: t.title,
                        description: t.description,
                        is_dismissible: true,
                        is_error: false
                    })) as ErrWarnDetails[]) ?? []
        };
    }, [data?.data?.items]);

    return {
        errors,
        warnings,
        fetchingErrors: isLoading
    };
};

export const usePredictionWarnings = (
    id: string,
    status: KumoapiCommonJobStatus | undefined
) => {
    const queryClient = useQueryClient();
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();

    const { data, isLoading } = useGetV1BatchPredictionJobsWarnings(
        { ids: [id] },
        { project_id: projectId ?? "" },
        getQueryOptionsById(
            ReactQueryKeys.PredictionJobWarnings + "-" + status,
            id,
            Boolean(id && status !== jobStatus.Cancelled && isValid(status))
        ),
        queryClient
    );

    return {
        fetchingErrors: isLoading,
        errors: data?.data?.[id]?.errors ?? [],
        warnings: data?.data?.[id]?.warnings ?? []
    };
};
