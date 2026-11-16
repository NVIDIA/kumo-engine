// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useParams } from "react-router";

import {
    removeFromLocalStorage,
    storeToLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";

import { setCreateProjectResponseType } from "components/Project/__tests__/MockFunctions/CreateProject.mock";
import {
    mockProjectsData,
    setProjectsResponseType
} from "components/Project/__tests__/MockFunctions/ProjectList.mock";
import {
    mockProjectData,
    setProjectResponseType,
    setUpdateProjectResponseType
} from "components/Project/__tests__/MockFunctions/ProjectSettings.mock";

import { storageKeys } from "utils/Constants/enums";

import {
    useCreateProject,
    useProject,
    useProjectId,
    useProjectList,
    useProjectListByGroup,
    useUpdateProject
} from "../projects";

import "tests/mockAxios";

vi.mock("react-router", async () => {
    const actual = await vi.importActual("react-router");
    return {
        ...actual,
        useParams: vi.fn()
    };
});

let queryClient: QueryClient;
const handleError = vi.fn();
const setLoading = vi.fn();

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setProjectResponseType();
    setUpdateProjectResponseType();
    setProjectsResponseType();
    setCreateProjectResponseType();
    vi.mocked(useParams).mockReturnValue({});
    removeFromLocalStorage(storageKeys.selectedProject);
});

afterEach(() => {
    queryClient.clear();
});

describe("useProjectList", () => {
    const renderUseProjectList = () =>
        renderHook(() => useProjectList(), {
            wrapper: createWrapper(queryClient)
        });

    it("Should return loading state initially and data after loading", async () => {
        const { result } = renderUseProjectList();

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.projectListResponse).toBeDefined();
    });
});

describe("useProject", () => {
    const renderUseProject = (projectId = "project_abc123def456") =>
        renderHook(() => useProject(projectId, handleError), {
            wrapper: createWrapper(queryClient)
        });

    it("Should return loading state initially and projectInfo after loading", async () => {
        const { result } = renderUseProject();

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.projectInfo).toEqual(mockProjectData);
    });

    it("Should return null and not load when projectId is empty", () => {
        const { result } = renderUseProject("");

        expect(result.current.isLoading).toBe(false);
        expect(result.current.projectInfo).toBeNull();
    });

    it("Should call handleError when API fails", async () => {
        setProjectResponseType(testResponseType.ERROR);

        renderUseProject();

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });
});

describe("useProjectListByGroup", () => {
    const renderUseProjectListByGroup = (groupId = "group-001") =>
        renderHook(() => useProjectListByGroup(groupId, handleError), {
            wrapper: createWrapper(queryClient)
        });

    it("Should return loading state initially and data after loading", async () => {
        const { result } = renderUseProjectListByGroup();

        expect(result.current.isLoading).toBe(true);

        await waitFor(() => {
            expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.data).toHaveLength(mockProjectsData.length);
    });

    it("Should return empty array and not load when groupId is empty", () => {
        const { result } = renderUseProjectListByGroup("");

        expect(result.current.isLoading).toBe(false);
        expect(result.current.data).toEqual([]);
    });

    it("Should call handleError when API fails", async () => {
        setProjectsResponseType(testResponseType.ERROR);

        renderUseProjectListByGroup();

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });
});

describe("useUpdateProject", () => {
    const renderUseUpdateProject = () =>
        renderHook(() => useUpdateProject(handleError, setLoading), {
            wrapper: createWrapper(queryClient)
        });

    const executeUpdate = async ({
        result
    }: ReturnType<typeof renderUseUpdateProject>) => {
        let response!: {
            projectId: string | null;
            error: string;
            statusCode: number;
        };
        await act(async () => {
            response = await result.current.updateProject(
                "project_abc123def456",
                { name: "Updated Project" }
            );
        });
        return response;
    };

    it("Should return projectId on success", async () => {
        const response = await executeUpdate(renderUseUpdateProject());

        expect(response.projectId).toBe(mockProjectData.id);
    });

    it("Should call setLoading(true) then setLoading(false) on success", async () => {
        await executeUpdate(renderUseUpdateProject());

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(setLoading).toHaveBeenNthCalledWith(2, false);
    });

    it("Should call handleError and setLoading(false) on failure", async () => {
        setUpdateProjectResponseType(testResponseType.ERROR);

        await executeUpdate(renderUseUpdateProject());

        expect(handleError).toHaveBeenCalled();
        expect(setLoading).toHaveBeenLastCalledWith(false);
    });
});

describe("useCreateProject", () => {
    const renderUseCreateProject = () =>
        renderHook(() => useCreateProject(handleError, setLoading), {
            wrapper: createWrapper(queryClient)
        });

    const executeCreate = async (
        { result }: ReturnType<typeof renderUseCreateProject>,
        onSuccess = vi.fn()
    ) => {
        let response!: {
            projectId: string | null;
            error: string;
            statusCode: number;
        };
        await act(async () => {
            response = await result.current.createProject(
                "group-001",
                { name: "New Project" },
                onSuccess
            );
        });
        return response;
    };

    it("Should call onSuccess with projectId on success", async () => {
        const onSuccess = vi.fn();
        const response = await executeCreate(
            renderUseCreateProject(),
            onSuccess
        );

        expect(onSuccess).toHaveBeenCalledWith("new-project-123");
        expect(response.projectId).toBe("new-project-123");
    });

    it("Should call setLoading(true) then setLoading(false) on success", async () => {
        await executeCreate(renderUseCreateProject());

        expect(setLoading).toHaveBeenNthCalledWith(1, true);
        expect(setLoading).toHaveBeenNthCalledWith(2, false);
    });

    it("Should call handleError and not call onSuccess on failure", async () => {
        setCreateProjectResponseType(testResponseType.ERROR);

        const onSuccess = vi.fn();
        await executeCreate(renderUseCreateProject(), onSuccess);

        expect(onSuccess).not.toHaveBeenCalled();
        expect(handleError).toHaveBeenCalled();
    });
});

describe("useProjectId", () => {
    it("Should return projectId from route params", () => {
        vi.mocked(useParams).mockReturnValue({ projectId: "route-project-id" });

        const { result } = renderHook(() => useProjectId());

        expect(result.current).toBe("route-project-id");
    });

    it("Should fall back to localStorage when projectId is not in params", () => {
        vi.mocked(useParams).mockReturnValue({});
        storeToLocalStorage(storageKeys.selectedProject, "stored-project-id");

        const { result } = renderHook(() => useProjectId());

        expect(result.current).toBe("stored-project-id");
    });

    it("Should return null when no projectId in params or localStorage", () => {
        vi.mocked(useParams).mockReturnValue({});

        const { result } = renderHook(() => useProjectId());

        expect(result.current).toBeNull();
    });
});
