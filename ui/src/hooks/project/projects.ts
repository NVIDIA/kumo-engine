// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from "react";
import { useParams } from "react-router";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCreateGroupsProjectsByGroupId,
    useGetGroupsProjectsByGroupId
} from "gen/v0/api/groups/groups";
import {
    useGetProjects,
    useGetProjectsByProjectId,
    useUpdateProjectsByProjectId
} from "gen/v0/api/projects/projects";
import type {
    CreateProjectInGroupRequest,
    UpdateProjectRequest
} from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import { parseErrorMessage, prepareCustomError } from "utils/errorHandler";
import { queryClient } from "utils/queryClient";
import { invalidateProjectInfo } from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { getStoredSelectedProjectId } from "utils/storage";

export const useProjectList = () => {
    const { getQueryOptions } = useQueryOptions();

    const {
        data: projectListResponse,
        isLoading,
        error,
        isError
    } = useGetProjects(
        getQueryOptions({
            queryKey: [ReactQueryKeys.ProjectList]
        })
    );
    return { projectListResponse, isLoading, error, isError };
};

export const useProject = (
    projectId: string | null,
    handleError?: HandleErrorFunc
) => {
    const { getQueryOptions } = useQueryOptions();

    const {
        data: projectResponse,
        isLoading,
        error,
        isError
    } = useGetProjectsByProjectId(
        projectId as string,
        getQueryOptions({
            queryKey: [ReactQueryKeys.ProjectList, projectId],
            enabled: !!projectId
        })
    );

    useEffect(() => {
        if (isError && error) {
            handleError?.(error as CustomError);
        }
    }, [error, handleError, isError]);
    return {
        projectInfo: projectResponse?.data || null,
        isLoading,
        error,
        isError
    };
};

export const useProjectListByGroup = (
    groupId: string,
    handleError: HandleErrorFunc
) => {
    const { getQueryOptions } = useQueryOptions();

    const {
        data: projectListResponse,
        isLoading,
        error,
        isError
    } = useGetGroupsProjectsByGroupId(
        groupId,
        getQueryOptions({
            queryKey: [ReactQueryKeys.ProjectList, groupId],
            enabled: !!groupId
        })
    );
    useEffect(() => {
        if (isError && error) {
            handleError(error as CustomError);
        }
    }, [error, handleError, isError]);
    return { data: projectListResponse?.data ?? [], isLoading, error, isError };
};

export const useUpdateProject = (
    handleError?: HandleErrorFunc,
    setLoading?: (loading: boolean) => void
) => {
    const { mutateAsync: updateProjectById } = useUpdateProjectsByProjectId();
    const updateProject = useCallback(
        async (
            projectId: string,
            config: UpdateProjectRequest
        ): Promise<{
            projectId: string | null;
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let newProjectId: string | null = null;
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading?.(true);
                const response = await updateProjectById({
                    projectId,
                    data: config
                });

                if (response?.data) {
                    queryClient.setQueryData(
                        [ReactQueryKeys.ProjectList, projectId],
                        response
                    );
                    newProjectId = response.data.id;
                }
                invalidateProjectInfo(queryClient, projectId);
                setLoading?.(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError?.(err, "Project update failed!");
                setLoading?.(false);
            }

            return { projectId: newProjectId, error, statusCode };
        },
        [setLoading, updateProjectById, handleError]
    );

    return { updateProject };
};

export const useCreateProject = (
    handleError: HandleErrorFunc,
    setLoading: (value: boolean) => void
) => {
    const { mutateAsync: createProject } = useCreateGroupsProjectsByGroupId();
    const queryClient = useQueryClient();

    const createNewProject = useCallback(
        async (
            groupId: string,
            config: CreateProjectInGroupRequest,
            onSuccess: (projectId: string) => void
        ): Promise<{
            projectId: string | null;
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let projectId: string | null = null;
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading(true);
                const response = await createProject({
                    groupId,
                    data: config
                });
                if (response?.data) {
                    projectId =
                        typeof response.data === "string"
                            ? response.data
                            : (response.data as unknown as { id: string }).id;
                    queryClient.invalidateQueries({
                        queryKey: [ReactQueryKeys.ProjectList, groupId]
                    });
                    onSuccess(projectId);
                }

                setLoading(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError?.(err, "Project creation failed!");
                setLoading(false);
            }

            return { projectId, error, statusCode };
        },
        [setLoading, createProject, queryClient, handleError]
    );

    return { createProject: createNewProject };
};

export const useProjectId = () => {
    const { projectId } = useParams<{ projectId?: string }>();
    const storedProjectID = getStoredSelectedProjectId();
    if (projectId) return projectId;
    if (storedProjectID) return storedProjectID;
    return null;
};
