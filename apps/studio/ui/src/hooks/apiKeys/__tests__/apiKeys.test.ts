// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { APIKeyType, CreateAdminAPIKeyRequest } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";

import {
    useAdminApiKeys,
    useCreateAdminKeys,
    useCreateAPIKeys,
    useDeleteAdminKeys,
    useDeleteAPIKeys,
    useUserApiKeys
} from "../apiKeys";

import {
    mockAdminApiKeysList,
    mockCreateAdminApiKeyResponse,
    setAdminApiKeysResponseType,
    setCreateAdminApiKeyResponseType,
    setDeleteAdminApiKeyResponseType
} from "./MockFunctions/AdminApiKeys.mock";
import {
    mockCreateUserApiKeyResponse,
    mockUserApiKeysList,
    setCreateUserApiKeyResponseType,
    setDeleteUserApiKeyResponseType,
    setUserApiKeysResponseType
} from "./MockFunctions/UserApiKeys.mock";

import "tests/mockAxios";

let queryClient: QueryClient;

const handleError = vi.fn();
const setLoading = vi.fn();

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setAdminApiKeysResponseType();
    setCreateAdminApiKeyResponseType();
    setDeleteAdminApiKeyResponseType();
    setUserApiKeysResponseType();
    setCreateUserApiKeyResponseType();
    setDeleteUserApiKeyResponseType();
});

afterEach(() => {
    queryClient.clear();
});

it("Phase 1 contract: generated API key types exclude legacy", () => {
    expect(Object.values(APIKeyType)).not.toContain("legacy");
});

describe("useAdminApiKeys", () => {
    const renderUseAdminApiKeys = (
        apiKeyType: APIKeyType = APIKeyType.ADMIN,
        searchStr: string = "",
        enabled: boolean = true
    ) => {
        return renderHook(
            () => useAdminApiKeys(apiKeyType, searchStr, enabled, handleError),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should return loading state initially and admin api keys after loading", async () => {
        const { result } = renderUseAdminApiKeys();

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.apiKeys).toHaveLength(
            mockAdminApiKeysList.length
        );
    });

    it("Should return empty array and not load when disabled", () => {
        const { result } = renderUseAdminApiKeys(
            APIKeyType.ADMIN,
            "",
            /*enabled*/ false
        );

        expect(result.current.isLoading).toBe(false);
        expect(result.current.apiKeys).toEqual([]);
    });

    it("Should call handleError when API fails", async () => {
        setAdminApiKeysResponseType(testResponseType.ERROR);

        renderUseAdminApiKeys();

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should expose clearApiKeysCache as a function", () => {
        const { result } = renderUseAdminApiKeys();

        expect(typeof result.current.clearApiKeysCache).toBe("function");
    });
});

describe("useUserApiKeys", () => {
    const renderUseUserApiKeys = (enabled: boolean = true) => {
        return renderHook(() => useUserApiKeys(enabled, handleError), {
            wrapper: createWrapper(queryClient)
        });
    };

    it("Should return loading state initially and user api keys after loading", async () => {
        const { result } = renderUseUserApiKeys();

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.apiKeys).toHaveLength(mockUserApiKeysList.length);
    });

    it("Should return empty array and not load when disabled", () => {
        const { result } = renderUseUserApiKeys(/*enabled*/ false);

        expect(result.current.isLoading).toBe(false);
        expect(result.current.apiKeys).toEqual([]);
    });

    it("Should call handleError when API fails", async () => {
        setUserApiKeysResponseType(testResponseType.ERROR);

        renderUseUserApiKeys();

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should expose clearApiKeysCache as a function", () => {
        const { result } = renderUseUserApiKeys();

        expect(typeof result.current.clearApiKeysCache).toBe("function");
    });
});

describe("useCreateAdminKeys", () => {
    const renderUseCreateAdminKeys = () => {
        const clearCache = vi.fn();
        const hookResult = renderHook(
            () => useCreateAdminKeys(setLoading, clearCache),
            { wrapper: createWrapper(queryClient) }
        );
        return { ...hookResult, clearCache };
    };

    const executeCreate = async (
        { result }: ReturnType<typeof renderUseCreateAdminKeys>,
        onSuccess = vi.fn()
    ) => {
        let response!: { error: string; statusCode: number };
        await act(async () => {
            response = await result.current.createKeys(
                {
                    name: "Test Key",
                    type: APIKeyType.ADMIN
                } as CreateAdminAPIKeyRequest,
                onSuccess
            );
        });
        return response;
    };

    it("Should call onSuccess with response data and clear cache on success", async () => {
        const onSuccess = vi.fn();
        const hookResult = renderUseCreateAdminKeys();

        await executeCreate(hookResult, onSuccess);

        expect(onSuccess).toHaveBeenCalledWith(mockCreateAdminApiKeyResponse);
        expect(hookResult.clearCache).toHaveBeenCalled();
    });

    it("Should call setLoading(true) then setLoading(false) on success", async () => {
        await executeCreate(renderUseCreateAdminKeys());

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(setLoading).toHaveBeenNthCalledWith(2, false);
    });

    it("Should return error string and not call onSuccess on failure", async () => {
        setCreateAdminApiKeyResponseType(testResponseType.ERROR);

        const onSuccess = vi.fn();
        const response = await executeCreate(
            renderUseCreateAdminKeys(),
            onSuccess
        );

        expect(onSuccess).not.toHaveBeenCalled();
        expect(response.error).toBeTruthy();
    });
});

describe("useCreateAPIKeys", () => {
    const renderUseCreateAPIKeys = () => {
        const clearCache = vi.fn();
        const hookResult = renderHook(
            () => useCreateAPIKeys(setLoading, clearCache),
            { wrapper: createWrapper(queryClient) }
        );
        return { ...hookResult, clearCache };
    };

    const executeCreate = async (
        { result }: ReturnType<typeof renderUseCreateAPIKeys>,
        onSuccess = vi.fn()
    ) => {
        let response!: { error: string; statusCode: number };
        await act(async () => {
            response = await result.current.createKeys(
                { name: "Test Key" },
                onSuccess
            );
        });
        return response;
    };

    it("Should call onSuccess with response data and clear cache on success", async () => {
        const onSuccess = vi.fn();
        const hookResult = renderUseCreateAPIKeys();

        await executeCreate(hookResult, onSuccess);

        expect(onSuccess).toHaveBeenCalledWith(mockCreateUserApiKeyResponse);
        expect(hookResult.clearCache).toHaveBeenCalled();
    });

    it("Should call setLoading(true) then setLoading(false) on success", async () => {
        await executeCreate(renderUseCreateAPIKeys());

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(setLoading).toHaveBeenNthCalledWith(2, false);
    });

    it("Should return error string and not call onSuccess on failure", async () => {
        setCreateUserApiKeyResponseType(testResponseType.ERROR);

        const onSuccess = vi.fn();
        const response = await executeCreate(
            renderUseCreateAPIKeys(),
            onSuccess
        );

        expect(onSuccess).not.toHaveBeenCalled();
        expect(response.error).toBeTruthy();
    });
});

describe("useDeleteAdminKeys", () => {
    const renderUseDeleteAdminKeys = () => {
        const clearCache = vi.fn();
        const hookResult = renderHook(
            () => useDeleteAdminKeys(setLoading, handleError, clearCache),
            { wrapper: createWrapper(queryClient) }
        );
        return { ...hookResult, clearCache };
    };

    const executeDelete = async (
        { result }: ReturnType<typeof renderUseDeleteAdminKeys>,
        onSuccess = vi.fn()
    ) => {
        await act(async () => {
            await result.current.deleteKeys("admin-key-001", onSuccess);
        });
    };

    it("Should call onSuccess and clear cache on success", async () => {
        const onSuccess = vi.fn();
        const hookResult = renderUseDeleteAdminKeys();

        await executeDelete(hookResult, onSuccess);

        expect(onSuccess).toHaveBeenCalled();
        expect(hookResult.clearCache).toHaveBeenCalled();
    });

    it("Should call setLoading(true) then setLoading(false) on success", async () => {
        await executeDelete(renderUseDeleteAdminKeys());

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(setLoading).toHaveBeenNthCalledWith(2, false);
    });

    it("Should call handleError and not call onSuccess on failure", async () => {
        setDeleteAdminApiKeyResponseType(testResponseType.ERROR);

        const onSuccess = vi.fn();
        await executeDelete(renderUseDeleteAdminKeys(), onSuccess);

        expect(onSuccess).not.toHaveBeenCalled();
        expect(handleError).toHaveBeenCalled();
    });
});

describe("useDeleteAPIKeys", () => {
    const renderUseDeleteAPIKeys = () => {
        const clearCache = vi.fn();
        const hookResult = renderHook(
            () => useDeleteAPIKeys(setLoading, handleError, clearCache),
            { wrapper: createWrapper(queryClient) }
        );
        return { ...hookResult, clearCache };
    };

    const executeDelete = async (
        { result }: ReturnType<typeof renderUseDeleteAPIKeys>,
        onSuccess = vi.fn()
    ) => {
        await act(async () => {
            await result.current.deleteKeys("user-key-001", onSuccess);
        });
    };

    it("Should call onSuccess and clear cache on success", async () => {
        const onSuccess = vi.fn();
        const hookResult = renderUseDeleteAPIKeys();

        await executeDelete(hookResult, onSuccess);

        expect(onSuccess).toHaveBeenCalled();
        expect(hookResult.clearCache).toHaveBeenCalled();
    });

    it("Should call setLoading(true) then setLoading(false) on success", async () => {
        await executeDelete(renderUseDeleteAPIKeys());

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(setLoading).toHaveBeenNthCalledWith(2, false);
    });

    it("Should call handleError and not call onSuccess on failure", async () => {
        setDeleteUserApiKeyResponseType(testResponseType.ERROR);

        const onSuccess = vi.fn();
        await executeDelete(renderUseDeleteAPIKeys(), onSuccess);

        expect(onSuccess).not.toHaveBeenCalled();
        expect(handleError).toHaveBeenCalled();
    });
});
