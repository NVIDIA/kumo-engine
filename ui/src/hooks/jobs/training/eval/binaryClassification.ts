// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";

import { useGetEvalBinaryClassificationById } from "gen/v0/api/evaluation-statistics/evaluation-statistics";
import { useGetTrainingJobsBinaryClassificationById } from "gen/v0/api/training-jobs/training-jobs";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";

import { errorStatusCode } from "utils/Constants";
import { parseErrorMessage } from "utils/errorHandler";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useGetEvalStats = (
    trainingJobId?: string,
    queryId?: string,
    handleError?: (err: CustomError, msg: string, isWarning?: boolean) => void,
    setLoading?: (v: boolean) => void
) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";

    const {
        data: trainingEvalResponse,
        error: trainingEvalError,
        isError: isTrainingEvalError,
        isLoading: isTrainingEvalLoading
    } = useGetTrainingJobsBinaryClassificationById(
        trainingJobId ?? "",
        getQueryOptionsById(
            ReactQueryKeys.TrainingJobsBinaryClassificationById,
            trainingJobId,
            Boolean(projectId && trainingJobId)
        )
    );
    const {
        data: queryEvalResponse,
        error: queryEvalError,
        isError: isQueryEvalError,
        isLoading: isQueryEvalLoading
    } = useGetEvalBinaryClassificationById(
        queryId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.EvalBinaryClassificationById,
            queryId,
            Boolean(projectId && queryId && !trainingJobId)
        )
    );

    // Handle training job errors
    useEffect(() => {
        if (isTrainingEvalError && trainingEvalError && handleError) {
            const errData = parseErrorMessage(trainingEvalError as CustomError);
            const err: CustomError = {
                response: {
                    data: errData,
                    status: (trainingEvalError as CustomError)?.response?.status
                },
                name: "",
                message: errData
            };
            if (errData?.includes("eval results not available")) {
                if (err?.response) {
                    err.response.status =
                        errorStatusCode.UnProcessableEntityError;
                }
                setLoading?.(false);
            }
            setLoading?.(false);
            handleError(
                err,
                "Failed to get Training Job Evaluation Stats!",
                true
            );
        }
    }, [isTrainingEvalError, trainingEvalError, setLoading, handleError]);

    useEffect(() => {
        if (isQueryEvalError && queryEvalError && handleError) {
            const errData = parseErrorMessage(queryEvalError as CustomError);
            const err: CustomError = {
                response: {
                    data: errData,
                    status: (queryEvalError as CustomError)?.response?.status
                },
                name: "",
                message: errData
            };
            if (errData?.includes("eval results not available")) {
                if (err?.response) {
                    err.response.status =
                        errorStatusCode.UnProcessableEntityError;
                }
                setLoading?.(false);
            }
            setLoading?.(false);
            handleError(err, "Failed to get Query Evaluation Stats!", true);
        }
    }, [isQueryEvalError, queryEvalError, handleError, setLoading]);

    return {
        evaluationResponse:
            trainingEvalResponse?.data || queryEvalResponse?.data,
        isDataLoading: isTrainingEvalLoading || isQueryEvalLoading
    };
};
