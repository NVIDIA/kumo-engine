// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { vi } from "vitest";

import { setGraphsDetailResponseType } from "pages/Graphs/__tests__/MockFunction/GraphDetail.test.mock";
import { setGraphsListResponseType } from "pages/Graphs/__tests__/MockFunction/GraphsList.test.mock";
import {
    mockGraphSuggestLinksResponse,
    setGraphSuggestLinksResponseType
} from "pages/Graphs/__tests__/MockFunction/GraphSuggestLinks.mock";
import { graphDetail } from "pages/Graphs/__tests__/MockResponse/GraphDetail.response";

import {
    ColumnGroup,
    GraphFlowInfo,
    GraphListResponse,
    GraphTables
} from "types/graphs";

import { storageKeys } from "utils/Constants";

import { useGetGraphInfo, useGraphs, useSuggestLinks } from "../graphs";

import "tests/mockAxios";

vi.mock("utils/navigator", () => ({
    useCustomNavigation: vi.fn(() => vi.fn())
}));

let queryClient: QueryClient;

const handleError = vi.fn();
const setLoading = vi.fn();

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    storeToLocalStorage(storageKeys.selectedProject, "project-tests");

    setGraphsListResponseType();
    setGraphsDetailResponseType();
    setGraphSuggestLinksResponseType();
});

afterEach(() => {
    queryClient.clear();
});

describe("useGraphs", () => {
    const renderUseGraphs = (
        handleErrorFn?: (err: Error, title?: string) => void,
        setPageLoadingFn?: (isLoading: boolean) => void,
        setSelectedGraphFn?: (response: GraphListResponse | null) => void,
        enabled: boolean = true
    ) => {
        return renderHook(
            () =>
                useGraphs({
                    enabled,
                    projectId: "project-tests",
                    handleError: handleErrorFn,
                    setPageLoading: setPageLoadingFn,
                    setSelectedGraph: setSelectedGraphFn
                }),
            { wrapper: createWrapper(queryClient) }
        );
    };

    const waitForGraphsLoaded = async (
        result: ReturnType<typeof renderUseGraphs>["result"]
    ) => {
        await waitFor(() => {
            expect(result.current.isDataLoading).toBe(false);
        });
    };

    it("Should return loading state initially and graph list after loading", async () => {
        const { result } = renderUseGraphs();

        expect(result.current.isDataLoading).toBe(true);

        await waitForGraphsLoaded(result);

        expect(result.current.graphList.length).toBeGreaterThan(0);
        expect(result.current.isGraphsPresent).toBe(true);

        result.current.graphList.forEach((graph) => {
            expect(graph.id).toBeTruthy();
        });
    });

    it("Should transform graph list response with name, id, timestamps and dateCreated", async () => {
        const { result } = renderUseGraphs();

        await waitForGraphsLoaded(result);

        const firstGraph = result.current.graphList[0];
        const expectedProperties = ["name", "id", "timestamps", "dateCreated"];

        expectedProperties.forEach((property) => {
            expect(firstGraph).toHaveProperty(property);
        });
    });

    it("Should call handleError when API fails", async () => {
        setGraphsListResponseType(testResponseType.ERROR);

        renderUseGraphs(handleError, setLoading);

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it.each([
        {
            description: "when loading completes",
            responseType: undefined
        },
        {
            description: "when error occurs",
            responseType: testResponseType.ERROR
        },
        {
            description: "when graph list is empty",
            responseType: testResponseType.EMPTY
        }
    ])(
        "Should call setPageLoading(false) $description",
        async ({ responseType }) => {
            if (responseType) {
                setGraphsListResponseType(responseType);
            }

            renderUseGraphs(handleError, setLoading);

            await waitFor(() => {
                expect(setLoading).toHaveBeenCalledWith(false);
            });
        }
    );

    it("Should set isGraphsPresent to false when graph list is empty", async () => {
        setGraphsListResponseType(testResponseType.EMPTY);

        const { result } = renderUseGraphs(handleError, setLoading);

        await waitForGraphsLoaded(result);

        expect(result.current.isGraphsPresent).toBe(false);
    });

    describe("deleteGraph", () => {
        const setupDeleteGraph = async () => {
            const setSelectedGraph = vi.fn();

            const { result } = renderUseGraphs(
                handleError,
                setLoading,
                setSelectedGraph
            );

            await waitForGraphsLoaded(result);

            const initialLength = result.current.graphList.length;
            const graphToDelete = {
                name: result.current.graphList[0].name,
                tables: {},
                groups: [],
                totalLinks: 0,
                totalTables: 0
            };

            await act(async () => {
                result.current.deleteGraph(graphToDelete);
            });

            return { result, setSelectedGraph, initialLength };
        };

        it("Should remove deleted graph from graphList", async () => {
            const { result, initialLength } = await setupDeleteGraph();

            expect(result.current.graphList.length).toBe(initialLength - 1);
        });

        it("Should call setSelectedGraph(null) after delete", async () => {
            const { setSelectedGraph } = await setupDeleteGraph();

            expect(setSelectedGraph).toHaveBeenCalledWith(null);
        });

        it("Should call setPageLoading during delete operation", async () => {
            await setupDeleteGraph();

            expect(setLoading).toHaveBeenCalledWith(true);
            expect(setLoading).toHaveBeenCalledWith(false);
        });
    });
});

describe("useSuggestLinks", () => {
    const mockGraphFlowInfo: GraphFlowInfo = {
        name: graphDetail.name_alias,
        id: graphDetail.id,
        description: "",
        tables: graphDetail.graph.tables as GraphTables,
        selected_tables: Object.keys(graphDetail.graph.tables),
        selected_table_ids: Object.values(
            graphDetail.table_name_to_fpid_mapping
        ),
        flow_type: "CREATE"
    };

    const createSingleTableInfo = (): GraphFlowInfo => {
        const firstTableName = Object.keys(
            graphDetail.graph.tables
        )[0] as keyof typeof graphDetail.graph.tables;
        const firstTable = graphDetail.graph.tables[firstTableName];
        const firstTableId = (
            graphDetail.table_name_to_fpid_mapping as Record<string, string>
        )[firstTableName];

        return {
            name: "single_table_graph",
            id: "graph-single-id",
            description: "",
            tables: { [firstTableName]: firstTable } as GraphTables,
            selected_tables: [firstTableName],
            selected_table_ids: [firstTableId],
            flow_type: "CREATE"
        };
    };

    const renderUseSuggestLinks = (
        graphFlowInfo?: GraphFlowInfo,
        allTables?: GraphTables,
        groups?: ColumnGroup[],
        setGroupsFn?: (g: ColumnGroup[]) => void,
        handleErrorFn?: (err: Error, title?: string) => void,
        setLoadingFn?: (load: boolean) => void
    ) => {
        const info = graphFlowInfo ?? mockGraphFlowInfo;
        return renderHook(
            () =>
                useSuggestLinks(
                    info,
                    allTables ?? info.tables,
                    groups ?? [],
                    setGroupsFn ?? vi.fn(),
                    handleErrorFn ?? handleError,
                    setLoadingFn ?? setLoading
                ),
            { wrapper: createWrapper(queryClient) }
        );
    };

    const callSuggestLinksAndWait = async (
        result: ReturnType<typeof renderUseSuggestLinks>["result"]
    ) => {
        await act(async () => {
            result.current.suggestLinks();
        });

        await waitFor(() => {
            expect(setLoading).toHaveBeenCalledWith(false);
        });
    };

    it("Should provide suggestLinks function", () => {
        const { result } = renderUseSuggestLinks();

        expect(typeof result.current.suggestLinks).toBe("function");
    });

    it("Should call setLoading(true) when suggestLinks is called", async () => {
        const { result } = renderUseSuggestLinks();

        await act(async () => {
            result.current.suggestLinks();
        });

        expect(setLoading).toHaveBeenCalledWith(true);
    });

    it("Should not make API call when only one table exists", async () => {
        const singleTableInfo = createSingleTableInfo();
        const setGroups = vi.fn();

        const { result } = renderUseSuggestLinks(
            singleTableInfo,
            singleTableInfo.tables,
            [],
            setGroups
        );

        await callSuggestLinksAndWait(result);

        expect(setGroups).not.toHaveBeenCalled();
    });

    it("Should call setGroups with suggested links on success", async () => {
        const setGroups = vi.fn();

        const { result } = renderUseSuggestLinks(
            mockGraphFlowInfo,
            mockGraphFlowInfo.tables,
            [],
            setGroups
        );

        await callSuggestLinksAndWait(result);

        expect(setGroups).toHaveBeenCalled();
        expect(setGroups).toHaveBeenCalledWith(expect.any(Array));
    });

    it("ignores suggested links after the originating graph changes", async () => {
        const originalPost = mockAxiosInstance.post.getMockImplementation()!;
        const suggestedLinks = Promise.withResolvers<unknown>();
        mockAxiosInstance.post.mockImplementation((url, payload, options) => {
            if (url === "v1/graphs/suggest_links") {
                return suggestedLinks.promise as ReturnType<
                    typeof originalPost
                >;
            }
            return originalPost(url, payload, options);
        });
        const setGroups = vi.fn();
        const initialGroups: ColumnGroup[] = [];

        try {
            const { result, rerender } = renderHook(
                ({
                    graphFlowInfo,
                    groups
                }: {
                    graphFlowInfo: GraphFlowInfo;
                    groups: ColumnGroup[];
                }) =>
                    useSuggestLinks(
                        graphFlowInfo,
                        graphFlowInfo.tables,
                        groups,
                        setGroups,
                        handleError,
                        setLoading
                    ),
                {
                    initialProps: {
                        graphFlowInfo: mockGraphFlowInfo,
                        groups: initialGroups
                    },
                    wrapper: createWrapper(queryClient)
                }
            );

            act(() => {
                result.current.suggestLinks();
            });
            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v1/graphs/suggest_links",
                    expect.anything(),
                    expect.anything()
                );
            });

            rerender({
                graphFlowInfo: {
                    ...mockGraphFlowInfo,
                    name: "edited-while-suggesting"
                },
                groups: initialGroups
            });
            await act(async () => {
                suggestedLinks.resolve(await mockGraphSuggestLinksResponse());
            });

            expect(setGroups).not.toHaveBeenCalled();
            expect(setLoading).toHaveBeenLastCalledWith(false);
        } finally {
            mockAxiosInstance.post.mockImplementation(originalPost);
        }
    });

    it("ignores an older suggestion request for the current graph", async () => {
        const originalPost = mockAxiosInstance.post.getMockImplementation()!;
        const firstRequest = Promise.withResolvers<unknown>();
        const secondRequest = Promise.withResolvers<unknown>();
        const requests = [firstRequest, secondRequest];
        mockAxiosInstance.post.mockImplementation((url, payload, options) => {
            if (url === "v1/graphs/suggest_links") {
                return requests.shift()?.promise as ReturnType<
                    typeof originalPost
                >;
            }
            return originalPost(url, payload, options);
        });
        const setGroups = vi.fn();

        try {
            const { result } = renderUseSuggestLinks(
                mockGraphFlowInfo,
                mockGraphFlowInfo.tables,
                [],
                setGroups
            );

            act(() => {
                result.current.suggestLinks();
                result.current.suggestLinks();
            });
            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledTimes(2);
            });

            await act(async () => {
                firstRequest.resolve(await mockGraphSuggestLinksResponse());
            });
            expect(setGroups).not.toHaveBeenCalled();

            await act(async () => {
                secondRequest.resolve(await mockGraphSuggestLinksResponse());
            });
            await waitFor(() => {
                expect(setGroups).toHaveBeenCalledTimes(1);
            });
        } finally {
            mockAxiosInstance.post.mockImplementation(originalPost);
        }
    });
});

describe("useGetGraphInfo", () => {
    const graphId = "graph-4973156df602a3f4346186713bd6782d";

    const renderUseGetGraphInfo = (
        graphId: string | null,
        handleErrorFn: (err: Error) => void = handleError,
        setLoadingFn?: (load: boolean) => void,
        enabled: boolean = true
    ) => {
        return renderHook(
            () =>
                useGetGraphInfo(graphId, handleErrorFn, setLoadingFn, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    const waitForGraphInfoLoaded = async (
        result: ReturnType<typeof renderUseGetGraphInfo>["result"]
    ) => {
        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });
    };

    it("Should return loading state initially and selectedGraphInfo after loading", async () => {
        const { result } = renderUseGetGraphInfo(graphId);

        expect(result.current.isLoading).toBe(true);

        await waitForGraphInfoLoaded(result);

        expect(result.current.selectedGraphInfo).not.toBeNull();
    });

    it("Should return selectedGraphInfo with expected properties", async () => {
        const { result } = renderUseGetGraphInfo(graphId);

        await waitForGraphInfoLoaded(result);

        const graphInfo = result.current.selectedGraphInfo;
        const expectedProperties = [
            "tables",
            "groups",
            "totalLinks",
            "totalTables"
        ];

        expectedProperties.forEach((property) => {
            expect(graphInfo).toHaveProperty(property);
        });
    });

    it("Should call handleError when API fails", async () => {
        setGraphsDetailResponseType(testResponseType.ERROR);

        renderUseGetGraphInfo(graphId, handleError, setLoading);

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should call setLoading(false) when loading completes", async () => {
        const { result } = renderUseGetGraphInfo(
            graphId,
            handleError,
            setLoading
        );

        await waitForGraphInfoLoaded(result);

        expect(setLoading).toHaveBeenCalledWith(false);
    });

    it.each([
        {
            description: "graphId is null",
            graphId: null,
            enabled: true
        },
        {
            description: "enabled is false",
            graphId: graphId,
            enabled: false
        }
    ])("Should not fetch when $description", ({ graphId, enabled }) => {
        const { result } = renderUseGetGraphInfo(
            graphId,
            handleError,
            setLoading,
            enabled
        );

        expect(result.current.selectedGraphInfo).toBeNull();
    });

    it("Should provide fetchGraphInfo function", () => {
        const { result } = renderUseGetGraphInfo(graphId);

        expect(typeof result.current.fetchGraphInfo).toBe("function");
    });
});
