// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    getFromLocalStorage,
    removeFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { vi } from "vitest";

import {
    connectorListWithDistinctNamesResponse,
    setConnectorListResponseType,
    setCustomConnectorListResponse
} from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { setConnectorTablesResponseType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorTables.mock";
import { setDeleteConnectorTestType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/DeleteConnector.mock";
import { setConTestResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/ConnectorResponse.test.mock";

import { getProjectPath, storageKeys } from "utils/Constants";

import {
    useConnectedTables,
    useConnectorDetails,
    useConnectorDropdownOptions,
    useConnectors,
    useGetColumns,
    useGetSampleRows
} from "../connector";

import "tests/mockAxios";

vi.mock("hooks/utils/featureFlags", () => ({
    default: vi.fn(() => ({
        IS_DATABRICKS_DEPLOY: false
    }))
}));

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-tests"
}));

let queryClient: QueryClient;

const handleError = vi.fn();
const setLoading = vi.fn();

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    mockNavigateCall();
    setConnectorListResponseType();
    setCustomConnectorListResponse(null);
    setConnectorTablesResponseType();
    setDeleteConnectorTestType();
});

afterEach(() => {
    queryClient.clear();
    removeFromLocalStorage(storageKeys.lastUsedConn);
});

describe("useConnectors", () => {
    const renderUseConnectors = (
        handleErrorFn?: (err: Error, title?: string) => void,
        setPageLoadingFn?: (v: boolean) => void,
        enabled: boolean = true
    ) => {
        return renderHook(
            () => useConnectors(handleErrorFn, setPageLoadingFn, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should return loading state initially and connector list after loading", async () => {
        setConnectorListResponseType(testResponseType.ALL);

        const { result } = renderUseConnectors();

        expect(result.current.isDataLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isDataLoading).toBe(false);
        });

        expect(result.current.connectorList.length).toBeGreaterThan(0);
    });

    it("Should call handleError when API fails", async () => {
        setConnectorListResponseType(testResponseType.ERROR);

        renderUseConnectors(handleError, setLoading);

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should call setPageLoading(false) when loading completes", async () => {
        setConnectorListResponseType(testResponseType.ALL);

        renderUseConnectors(handleError, setLoading);

        await waitFor(() => {
            expect(setLoading).toHaveBeenCalledWith(false);
        });
    });

    it.each([
        { property: "isConnectorsPresent", description: "connectors exist" },
        {
            property: "isDBConnectorsPresent",
            description: "DB connectors exist"
        }
    ])(
        "Should set $property to true when $description",
        async ({ property }) => {
            setConnectorListResponseType(testResponseType.ALL);

            const { result } = renderUseConnectors();

            await waitFor(() => {
                expect(
                    result.current[property as keyof typeof result.current]
                ).toBe(true);
            });
        }
    );

    it("Should call setPageLoading when local tables check errors", async () => {
        setConnectorTablesResponseType(testResponseType.ERROR);

        renderUseConnectors(handleError, setLoading);

        await waitFor(() => {
            expect(setLoading).toHaveBeenCalledWith(false);
        });
    });

    it("Should list only identity-backed customer connectors", async () => {
        setConnectorListResponseType(testResponseType.ALL);

        const { result } = renderUseConnectors();

        await waitFor(() => {
            expect(result.current.isDataLoading).toBe(false);
        });

        expect(
            result.current.connectorList.every(
                (connector) =>
                    Boolean(connector.config.type) &&
                    Boolean(connector.identityId)
            )
        ).toBe(true);
    });

    describe("deleteConnector", () => {
        const setupDeleteConnector = async () => {
            setConnectorListResponseType(testResponseType.ALL);

            const { result } = renderUseConnectors(handleError);
            const clearSelected = vi.fn();

            await waitFor(() => {
                expect(result.current.isDataLoading).toBe(false);
            });

            const connectorToDelete = result.current.connectorList[0];

            const executeDelete = async () => {
                await act(async () => {
                    result.current.deleteConnector(
                        connectorToDelete.id,
                        connectorToDelete.name || "",
                        clearSelected
                    );
                });
            };

            return {
                result,
                connectorToDelete,
                clearSelected,
                executeDelete
            };
        };

        it("Should clean up connector state and navigate on success", async () => {
            const { result, connectorToDelete, clearSelected, executeDelete } =
                await setupDeleteConnector();

            await executeDelete();

            await waitFor(() => {
                expect(clearSelected).toHaveBeenCalledWith(null);
                expect(
                    result.current.connectorList.some(
                        ({ id }) => id === connectorToDelete.id
                    )
                ).toBe(false);
                expect(mockUseNavigate).toHaveBeenCalledWith(
                    getProjectPath("project-tests").Connectors,
                    { replace: true }
                );
            });
        });

        it("Should preserve connector state and call handleError on failure", async () => {
            setDeleteConnectorTestType(testResponseType.ERROR);

            const { result, connectorToDelete, clearSelected, executeDelete } =
                await setupDeleteConnector();
            const connectorIdsBeforeDelete = result.current.connectorList.map(
                ({ id }) => id
            );
            storeToLocalStorage(storageKeys.lastUsedConn, {
                value: connectorToDelete.id,
                type: "s3"
            });

            await executeDelete();

            await waitFor(() => {
                expect(handleError).toHaveBeenCalledWith(
                    expect.anything(),
                    `Unable to delete connector ${connectorToDelete.name}`
                );
            });
            expect(clearSelected).not.toHaveBeenCalled();
            expect(result.current.connectorList.map(({ id }) => id)).toEqual(
                connectorIdsBeforeDelete
            );
            expect(
                getFromLocalStorage(storageKeys.lastUsedConn)
            ).not.toBeNull();
            expect(mockUseNavigate).not.toHaveBeenCalled();
        });

        it("Should remove lastUsedConn from localStorage when deleting the last used connector", async () => {
            const { result, executeDelete } = await setupDeleteConnector();
            const connectorToDelete = result.current.connectorList[0];

            storeToLocalStorage(storageKeys.lastUsedConn, {
                value: connectorToDelete.id,
                type: "s3"
            });

            await executeDelete();

            await waitFor(() => {
                expect(
                    getFromLocalStorage(storageKeys.lastUsedConn)
                ).toBeNull();
            });
        });
    });
});

describe("useConnectorDropdownOptions", () => {
    const renderUseConnectorDropdownOptions = (
        handleErrorFn: (err: Error) => void = handleError
    ) => {
        return renderHook(() => useConnectorDropdownOptions(handleErrorFn), {
            wrapper: createWrapper(queryClient)
        });
    };

    it("Should return empty connectors array initially", () => {
        const { result } = renderUseConnectorDropdownOptions();

        expect(result.current.connectors).toEqual([]);
    });

    it("Should return connector dropdown options after loading", async () => {
        setConnectorListResponseType(testResponseType.ALL);

        const { result } = renderUseConnectorDropdownOptions();

        await waitFor(() => {
            expect(result.current.connectors.length).toBeGreaterThan(0);
        });

        const firstConnector = result.current.connectors[0];
        expect(firstConnector).toHaveProperty("value");
        expect(firstConnector).toHaveProperty("label");
        expect(firstConnector).toHaveProperty("type");
        expect(firstConnector).toHaveProperty("config");
    });

    it("Should use connector names as labels and IDs as values", async () => {
        setCustomConnectorListResponse(
            connectorListWithDistinctNamesResponse()
        );

        const { result } = renderUseConnectorDropdownOptions();

        await waitFor(() => {
            expect(result.current.connectors).toHaveLength(2);
        });
        expect(result.current.connectors).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    value: "warehouse-connector-id",
                    label: "Customer Warehouse"
                }),
                expect.objectContaining({
                    value: "lakehouse-connector-id",
                    label: "Analytics Lakehouse"
                })
            ])
        );
    });

    it("Should filter out connectors with null id", async () => {
        setConnectorListResponseType(testResponseType.ALL);

        const { result } = renderUseConnectorDropdownOptions();

        await waitFor(() => {
            expect(result.current.connectors.length).toBeGreaterThan(0);
        });

        const hasNullId = result.current.connectors.some(
            (c) => c.value === null
        );
        expect(hasNullId).toBe(false);
    });
});

describe("useConnectedTables", () => {
    const renderUseConnectedTables = (
        connectorId: string,
        handleErrorFn: (err: Error, title?: string) => void = handleError,
        setLoadingFn: React.Dispatch<React.SetStateAction<boolean>> = setLoading
    ) => {
        return renderHook(
            () => useConnectedTables(connectorId, handleErrorFn, setLoadingFn),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should set loading to false when connectorId is empty", async () => {
        renderUseConnectedTables("");

        await waitFor(() => {
            expect(setLoading).toHaveBeenCalledWith(false);
        });
    });

    it("Should return tables for a valid connectorId", async () => {
        const { result } = renderUseConnectedTables("ltv_test");

        await waitFor(() => {
            expect(result.current.tables.length).toBeGreaterThan(0);
        });
    });

    it("Should call handleError when API fails", async () => {
        setConnectorTablesResponseType(testResponseType.ERROR);

        renderUseConnectedTables("invalid_connector");

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it.each(["deleteTable", "setTables"])(
        "Should provide %s function",
        (fnName) => {
            const { result } = renderUseConnectedTables("ltv_test");

            expect(
                typeof result.current[fnName as keyof typeof result.current]
            ).toBe("function");
        }
    );

    it("Should call setConnectedTables callback when provided", async () => {
        const setConnectedTables = vi.fn();

        renderHook(
            () =>
                useConnectedTables(
                    "ltv_test",
                    handleError,
                    setLoading,
                    setConnectedTables
                ),
            { wrapper: createWrapper(queryClient) }
        );

        await waitFor(() => {
            expect(setConnectedTables).toHaveBeenCalled();
        });
    });
});

describe("useConnectorDetails", () => {
    const renderUseConnectorDetails = (
        connectorId: string,
        handleErrorFn: (err: Error, title?: string) => void = handleError
    ) => {
        return renderHook(
            () => useConnectorDetails(connectorId, handleErrorFn),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it.each([
        { connectorId: "ltv_test", description: "S3 connector" },
        { connectorId: "hnm", description: "snowflake connector" }
    ])("Should fetch details for $description", async ({ connectorId }) => {
        const { result } = renderUseConnectorDetails(connectorId);

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(mockAxiosInstance.get).toHaveBeenCalledWith(
            `v0/projects/project-tests/connectors/${connectorId}`
        );
        expect(result.current.connectorResponse).toBeDefined();
        expect(result.current.connectorResponse).toHaveProperty("id");
        expect(result.current.connectorResponse).toHaveProperty("config");
    });

    it("Should call handleError when connector fetch fails", async () => {
        setConTestResponseType(testResponseType.ERROR);

        renderUseConnectorDetails("invalid_connector");

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });

        setConTestResponseType();
    });
});

describe("useGetColumns", () => {
    const renderUseGetColumns = (
        config: {
            connectorId?: string;
            params?: { source_table_name: string; query?: string };
        },
        handleErrorFn: (err: Error) => void = handleError,
        setLoadingFn: (v: boolean) => void = setLoading,
        enabled: boolean = true
    ) => {
        return renderHook(
            () => useGetColumns(config, handleErrorFn, setLoadingFn, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it.each([
        {
            description: "named columns for connector by ID",
            config: {
                connectorId: "ltv_test",
                params: { source_table_name: "transaction" }
            }
        }
    ])("Should fetch $description", async ({ config }) => {
        const { result } = renderUseGetColumns(
            config,
            handleError,
            setLoading,
            true
        );

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });
    });

    it("Should not fetch when disabled", () => {
        const { result } = renderUseGetColumns(
            {
                connectorId: "ltv_test",
                params: { source_table_name: "transaction" }
            },
            handleError,
            setLoading,
            false
        );

        expect(result.current.columns).toBeUndefined();
    });

    it("Should key the SQL discovery cache by the query", async () => {
        const query = "SELECT id FROM customers";
        renderUseGetColumns(
            {
                connectorId: "ltv_test",
                params: { source_table_name: "", query }
            },
            handleError,
            setLoading,
            true
        );

        // The query rides in the request params, which the generated hook
        // includes in its `[url, params]` cache key — so a SQL discovery never
        // collides with a raw table of the same (empty) name.
        await waitFor(() => {
            const keys = queryClient
                .getQueryCache()
                .getAll()
                .map((q) => q.queryKey);
            expect(
                keys.some(
                    (key) =>
                        Array.isArray(key) &&
                        key.some(
                            (part) =>
                                typeof part === "object" &&
                                part !== null &&
                                (part as { query?: string }).query === query
                        )
                )
            ).toBe(true);
        });
    });
});

describe("useGetSampleRows", () => {
    const renderUseGetSampleRows = (
        config: {
            connectorId?: string;
            params: {
                source_table_name?: string;
                num_rows?: number;
                response_token?: string;
                query?: string;
            };
        },
        handleErrorFn: (err: Error, title?: string) => void = handleError,
        enabled: boolean = true
    ) => {
        return renderHook(
            () => useGetSampleRows(config, handleErrorFn, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it.each([
        {
            description: "named connector",
            config: {
                connectorId: "ltv_test",
                params: { source_table_name: "transaction", num_rows: 10 }
            }
        }
    ])("Should fetch sample rows for $description", async ({ config }) => {
        const { result } = renderUseGetSampleRows(config, handleError, true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });
    });

    it("Should not fetch when disabled", () => {
        const { result } = renderUseGetSampleRows(
            {
                connectorId: "ltv_test",
                params: { source_table_name: "transaction", num_rows: 10 }
            },
            handleError,
            false
        );

        expect(result.current.columnsWithRows).toBeNull();
        expect(result.current.isLoading).toBe(false);
    });

    it("Should key the SQL sample cache by the query", async () => {
        const query = "SELECT id FROM customers";
        renderUseGetSampleRows(
            {
                connectorId: "ltv_test",
                params: { source_table_name: "", num_rows: 10, query }
            },
            handleError,
            true
        );

        // The query rides in the request params and thus the generated
        // `[url, params]` cache key, keeping SQL samples from colliding with a
        // raw table's cached rows under the same name.
        await waitFor(() => {
            const keys = queryClient
                .getQueryCache()
                .getAll()
                .map((q) => q.queryKey);
            expect(
                keys.some(
                    (key) =>
                        Array.isArray(key) &&
                        key.some(
                            (part) =>
                                typeof part === "object" &&
                                part !== null &&
                                (part as { query?: string }).query === query
                        )
                )
            ).toBe(true);
        });
    });
});
