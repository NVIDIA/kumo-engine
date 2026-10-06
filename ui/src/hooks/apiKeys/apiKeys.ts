// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCreateApiKeys,
    useCreateApiKeysAdmin,
    useDeleteApiKeysAdminById,
    useDeleteApiKeysById,
    useGetApiKeys,
    useGetApiKeysAdmin
} from "gen/v0/api/api-keys/api-keys";
import {
    APIKeySummary,
    APIKeyType,
    CreateAdminAPIKeyRequest,
    CreateAPIKeyResponse,
    CreateUserAPIKeyRequest,
    GetApiKeysAdminParams
} from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import { parseErrorMessage, prepareCustomError } from "utils/errorHandler";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export type CreateKeysFunc = (
    config: CreateAdminAPIKeyRequest | CreateUserAPIKeyRequest,
    onSuccess: (data: CreateAPIKeyResponse) => void
) => Promise<{ error: string; statusCode: errorStatusCode }>;

export type DeleteKeysFunc = (
    id: string,
    onSuccess: () => void
) => Promise<{ error: string; statusCode: errorStatusCode }>;

export const useApiKeys = (
    apiKeyType: APIKeyType,
    searchStr: string,
    isAdmin: boolean,
    handleError: HandleErrorFunc,
    setLoading: (value: boolean) => void
): {
    apiKeys: APIKeySummary[];
    isLoading: boolean;
    createKeys: CreateKeysFunc;
    deleteKeys: DeleteKeysFunc;
} => {
    const {
        apiKeys: adminKeyData,
        isLoading: adminKeyLoading,
        clearApiKeysCache: clearApiKeysCacheAdmin
    } = useAdminApiKeys(
        apiKeyType,
        searchStr,
        /*enabled*/ isAdmin,
        handleError
    );

    const { createKeys: createAdminKey } = useCreateAdminKeys(
        setLoading,
        clearApiKeysCacheAdmin
    );

    const { deleteKeys: deleteAdminKey } = useDeleteAdminKeys(
        setLoading,
        handleError,
        clearApiKeysCacheAdmin
    );

    const {
        apiKeys: userKeyData,
        isLoading: userKeyLoading,
        clearApiKeysCache: clearApiKeysCacheUser
    } = useUserApiKeys(/*enabled*/ !isAdmin, handleError);

    const { createKeys: createUserKey } = useCreateAPIKeys(
        setLoading,
        clearApiKeysCacheUser
    );

    const { deleteKeys: deleteUserKey } = useDeleteAPIKeys(
        setLoading,
        handleError,
        clearApiKeysCacheUser
    );

    return isAdmin
        ? {
              apiKeys: adminKeyData,
              isLoading: adminKeyLoading,
              createKeys: createAdminKey as CreateKeysFunc,
              deleteKeys: deleteAdminKey
          }
        : {
              apiKeys: userKeyData,
              isLoading: userKeyLoading,
              createKeys: createUserKey as CreateKeysFunc,
              deleteKeys: deleteUserKey
          };
};

export const useAdminApiKeys = (
    apiKeyType: APIKeyType,
    searchStr: string,
    enabled: boolean,
    handleError?: HandleErrorFunc
) => {
    const { getQueryOptions } = useQueryOptions();
    const queryClient = useQueryClient();

    const { data, isLoading, isError, error } = useGetApiKeysAdmin(
        { type: apiKeyType, search: searchStr } as GetApiKeysAdminParams,
        getQueryOptions({
            queryKey: [ReactQueryKeys.ApiKeys, apiKeyType, searchStr],
            retry: false,
            enabled: enabled
        })
    );

    useEffect(() => {
        if (isError && error) {
            handleError?.(error as CustomError);
        }
    }, [error, handleError, isError]);

    const clearApiKeysCache = useCallback(() => {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.ApiKeys]
        });
    }, [queryClient]);

    return {
        apiKeys: data ? data.data : [],
        isLoading,
        clearApiKeysCache
    };
};

export const useCreateAdminKeys = (
    setLoading: (value: boolean) => void,
    clearCache: () => void
) => {
    const { mutateAsync: create } = useCreateApiKeysAdmin();

    const createKeys = useCallback(
        async (
            config: CreateAdminAPIKeyRequest,
            onSuccess: (data: CreateAPIKeyResponse) => void
        ): Promise<{
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading(true);
                const response = await create({ data: config });
                if (response?.data) {
                    onSuccess(response?.data);
                    clearCache();
                }

                setLoading(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                setLoading(false);
            }

            return { error, statusCode };
        },
        [clearCache, create, setLoading]
    );

    return { createKeys };
};

export const useUserApiKeys = (
    enabled: boolean,
    handleError?: HandleErrorFunc
) => {
    const { getQueryOptions } = useQueryOptions();
    const queryClient = useQueryClient();

    const { data, isLoading, isError, error } = useGetApiKeys(
        getQueryOptions({
            queryKey: [ReactQueryKeys.UserApiKeys],
            retry: false,
            enabled: enabled
        })
    );

    useEffect(() => {
        if (isError && error) {
            handleError?.(error as CustomError);
        }
    }, [error, handleError, isError]);

    const clearApiKeysCache = useCallback(() => {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.UserApiKeys]
        });
    }, [queryClient]);

    return {
        apiKeys: data ? data.data : [],
        isLoading,
        clearApiKeysCache
    };
};

export const useCreateAPIKeys = (
    setLoading: (value: boolean) => void,
    clearCache: () => void
) => {
    const { mutateAsync: create } = useCreateApiKeys();

    const createKeys = useCallback(
        async (
            config: CreateUserAPIKeyRequest,
            onSuccess: (data: CreateAPIKeyResponse) => void
        ): Promise<{
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading(true);
                const response = await create({ data: config });
                if (response?.data) {
                    onSuccess(response?.data);
                    clearCache();
                }

                setLoading(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                setLoading(false);
            }

            return { error, statusCode };
        },
        [clearCache, create, setLoading]
    );

    return { createKeys };
};

export const useDeleteAdminKeys = (
    setLoading: (value: boolean) => void,
    handleError: HandleErrorFunc,
    clearCache: () => void
) => {
    const { mutateAsync: remove } = useDeleteApiKeysAdminById();

    const deleteKeys = useCallback(
        async (
            id: string,
            onSuccess: () => void
        ): Promise<{
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading(true);
                await remove({ id });
                onSuccess();
                clearCache();
                setLoading(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError(err, "API Key deletion failed!");
                setLoading(false);
            }

            return { error, statusCode };
        },
        [clearCache, remove, handleError, setLoading]
    );

    return { deleteKeys };
};

export const useDeleteAPIKeys = (
    setLoading: (value: boolean) => void,
    handleError: HandleErrorFunc,
    clearCache: () => void
) => {
    const { mutateAsync: remove } = useDeleteApiKeysById();

    const deleteKeys = useCallback(
        async (
            id: string,
            onSuccess: () => void
        ): Promise<{
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            try {
                setLoading(true);
                await remove({ id });
                onSuccess();
                clearCache();
                setLoading(false);
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError(err, "API Key deletion failed!");
                setLoading(false);
            }

            return { error, statusCode };
        },
        [clearCache, remove, handleError, setLoading]
    );

    return { deleteKeys };
};
