// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCreateGroups,
    useGetGroups,
    useUpdateGroupsByGroupId
} from "gen/v0/api/groups/groups";
import { CreateKumoGroupRequest } from "gen/v0/types/createKumoGroupRequest";
import { UpdateKumoGroupRequest } from "gen/v0/types/updateKumoGroupRequest";

import { useQueryOptions } from "hooks/misc/queryOptions";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import { parseErrorMessage, prepareCustomError } from "utils/errorHandler";
import {
    invalidateGroupsList,
    invalidateUserGroupsList
} from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useUserGroups = () => {
    const { getQueryOptions } = useQueryOptions();

    const {
        data: groups,
        isLoading,
        error,
        isError
    } = useGetGroups(
        getQueryOptions({
            queryKey: [ReactQueryKeys.UserGroups]
        })
    );

    const rawData = groups?.data ?? [];
    const groupsDict: { [key: string]: string } = Array.isArray(rawData)
        ? Object.fromEntries(rawData.map((g) => [g.id, g.display_name]))
        : {};

    return { groups: groupsDict, isLoading, error, isError };
};

export const useKumoGroups = (handleError?: HandleErrorFunc) => {
    const { getQueryOptions } = useQueryOptions();

    const {
        data: groups,
        isLoading,
        error,
        isError,
        refetch
    } = useGetGroups(
        getQueryOptions({
            queryKey: [ReactQueryKeys.Groups]
        })
    );

    useEffect(() => {
        if (isError && error) {
            handleError?.(error as CustomError);
        }
    }, [error, handleError, isError]);
    return { groups: groups?.data ?? [], isLoading, error, isError, refetch };
};

export const useCreateKumoGroup = (
    handleError?: HandleErrorFunc,
    setLoading?: (value: boolean) => void
) => {
    const queryClient = useQueryClient();
    const { mutateAsync: createGroup } = useCreateGroups();

    const createNewGroup = useCallback(
        async (
            config: CreateKumoGroupRequest,
            onSuccess: (groupId: string) => void
        ): Promise<{
            groupId: string | null;
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let groupId: string | null = null;
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading?.(true);
                const response = await createGroup({ data: config });
                if (response?.data) {
                    groupId = response.data.id;
                    invalidateGroupsList(queryClient);
                    invalidateUserGroupsList(queryClient);
                    onSuccess(groupId ?? "");
                }

                setLoading?.(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError?.(err, "Group creation failed!");
                setLoading?.(false);
            }

            return { groupId, error, statusCode };
        },
        [createGroup, handleError, queryClient, setLoading]
    );

    return { createGroup: createNewGroup };
};

export const useUpdateKumoGroup = (handleError?: HandleErrorFunc) => {
    const queryClient = useQueryClient();
    const { mutateAsync: updateGroupById } = useUpdateGroupsByGroupId();
    const [loading, setLoading] = useState(false);
    const updateGroup = useCallback(
        async (
            groupId: string,
            config: UpdateKumoGroupRequest,
            onSuccess: () => void
        ): Promise<{
            groupId: string | null;
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let newGroupId: string | null = null;
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading(true);
                const response = await updateGroupById({
                    groupId,
                    data: config
                });

                if (response?.data) {
                    newGroupId = response.data.id as string;
                    invalidateGroupsList(queryClient);
                    invalidateUserGroupsList(queryClient);
                }
                setLoading(false);
                onSuccess();
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError?.(err, "Group update failed!");
                setLoading(false);
            }

            return { groupId: newGroupId, error, statusCode };
        },
        [handleError, queryClient, updateGroupById]
    );

    return { updateGroup, loading };
};
