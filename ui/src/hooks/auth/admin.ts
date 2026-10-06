// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from "react";

import { useQueryClient } from "@tanstack/react-query";
import { useGetAdminInfo, useGetAdminMembers } from "gen/v0/api/admin/admin";
import { useUserStore } from "stores/userStore";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useQueryOptions } from "../misc/queryOptions";

const useAdminInfo = (handleError?: HandleErrorFunc) => {
    const { getQueryOptions } = useQueryOptions();
    const queryClient = useQueryClient();
    const setAdminInfo = useUserStore((state) => state.setAdminInfo);

    const {
        data: adminInfo,
        isLoading,
        isError,
        error
    } = useGetAdminInfo(
        getQueryOptions({
            queryKey: [ReactQueryKeys.AdminInfo],
            retry: false
        })
    );

    useEffect(() => {
        if (isError && error) {
            if (handleError) handleError(error as CustomError);
            else console.error(error);
        }
    }, [error, handleError, isError]);

    useEffect(() => {
        if (adminInfo?.data) {
            setAdminInfo(adminInfo.data);
        }
    }, [adminInfo?.data, setAdminInfo]);

    const clearAdminInfoCache = useCallback(() => {
        queryClient.invalidateQueries({ queryKey: [ReactQueryKeys.AdminInfo] });
    }, [queryClient]);

    return {
        isAdmin: adminInfo?.data?.user_is_customer_admin ?? false,
        adminInfo: adminInfo ? adminInfo.data : null,
        isLoading,
        isError,
        error,
        clearAdminInfoCache
    };
};

const useAdminMembers = (handleError?: HandleErrorFunc) => {
    const { getQueryOptions } = useQueryOptions();
    const queryClient = useQueryClient();

    const {
        data: members,
        isLoading,
        isError,
        error
    } = useGetAdminMembers(
        getQueryOptions({
            queryKey: [ReactQueryKeys.AdminMembers]
        })
    );

    useEffect(() => {
        if (isError && error) {
            if (handleError) handleError(error as CustomError);
            else console.error(error);
        }
    }, [error, handleError, isError]);

    const clearAdminMembersCache = useCallback(() => {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.AdminMembers]
        });
    }, [queryClient]);

    return {
        members: members ? members.data : null,
        isLoading,
        clearAdminMembersCache
    };
};

export { useAdminInfo, useAdminMembers };
