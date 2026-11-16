// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from "react";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
    useGetAdminUsers,
    useUpdateAdminUsersByUserId
} from "gen/v0/api/admin/admin";
import type {
    DirectConnectUserResponse,
    DirectConnectUserUpdateRequest
} from "gen/v0/types";

import axiosClient from "services/axios_client";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { ReactQueryKeys } from "utils/reactQueryKeys";

interface DirectConnectUserUpdate extends DirectConnectUserUpdateRequest {
    user_id: string;
}

interface DirectConnectUserCreate {
    email: string;
    display_name?: string;
    role: DirectConnectUserResponse["role"];
    disabled?: boolean;
    password?: string;
}

interface DirectConnectUserPasswordUpdate {
    user_id: string;
    password: string;
}

export const useDirectConnectUsers = (
    enabled: boolean,
    handleError?: HandleErrorFunc
) => {
    const queryClient = useQueryClient();
    const invalidateDirectConnectUsers = useCallback(() => {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.AdminMembers, "directConnect"]
        });
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.AdminMembers]
        });
        // A role change can affect the current user's own admin status
        // (self-demotion), which AdminRouteGuard and the sidebar read from
        // the AdminInfo query; refetch it so they cannot act on stale state.
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.AdminInfo]
        });
    }, [queryClient]);

    const {
        data: usersResponse,
        isLoading,
        isError,
        error
    } = useGetAdminUsers({
        query: {
            queryKey: [ReactQueryKeys.AdminMembers, "directConnect"],
            enabled,
            retry: false
        }
    });

    const { mutateAsync: updateAdminUser, isPending: isUpdating } =
        useUpdateAdminUsersByUserId({
            mutation: {
                onSuccess: invalidateDirectConnectUsers
            }
        });

    const setPasswordRequest = useCallback(
        async ({ user_id, password }: DirectConnectUserPasswordUpdate) => {
            const response = await axiosClient.post<DirectConnectUserResponse>(
                `v0/admin/users/${encodeURIComponent(user_id)}/password`,
                { password }
            );
            return response.data;
        },
        []
    );

    const { mutateAsync: createUser, isPending: isCreating } = useMutation({
        mutationFn: async (payload: DirectConnectUserCreate) => {
            const response = await axiosClient.post<DirectConnectUserResponse>(
                "v0/admin/users",
                payload
            );
            return response.data;
        },
        onSuccess: invalidateDirectConnectUsers
    });

    const { mutateAsync: updatePassword, isPending: isPasswordUpdating } =
        useMutation({
            mutationFn: setPasswordRequest,
            onSuccess: () => {
                queryClient.invalidateQueries({
                    queryKey: [ReactQueryKeys.AdminMembers, "directConnect"]
                });
            }
        });

    useEffect(() => {
        if (isError && error) {
            if (handleError) handleError(error as CustomError);
            else console.error(error);
        }
    }, [error, handleError, isError]);

    const updateUsers = useCallback(
        async (updates: DirectConnectUserUpdate[]) => {
            for (const update of updates) {
                const { user_id, ...payload } = update;
                await updateAdminUser({
                    userId: encodeURIComponent(user_id),
                    data: payload
                });
            }
        },
        [updateAdminUser]
    );

    const setUserPassword = useCallback(
        async (userId: string, password: string) => {
            await updatePassword({ user_id: userId, password });
        },
        [updatePassword]
    );

    return {
        users: usersResponse?.data ?? [],
        isLoading,
        createUser,
        updateUsers,
        setUserPassword,
        isUpdating: isUpdating || isCreating || isPasswordUpdating
    };
};
