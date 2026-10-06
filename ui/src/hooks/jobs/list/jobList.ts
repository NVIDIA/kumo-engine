// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";

import {
    SortingState,
    useDebounce,
    useWindowDimensions,
    useWindowFocus
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import {
    getJobs,
    useCreateJobsArchiveByJobId,
    useGetJobs,
    useGetJobsActiveJobs,
    useGetJobsPredictionTags,
    useGetJobsTrainingTags
} from "gen/v0/api/jobs/jobs";
import {
    GenericJob,
    GenericJobSource,
    JobTags,
    JobType,
    ListJobsQuery
} from "gen/v0/types";
import { useGetV1PredictiveQueriesMany } from "gen/v1/api/predictive-queries/predictive-queries";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { ConfirmButtonVariant, CustomError } from "types/common";
import { JobOverviewType, JobSource, TagType } from "types/jobs";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";
import { getKeys, getValues } from "utils/Helpers/objectHelpers";
import { getIdHash } from "utils/Helpers/stringHelpers";
import {
    calculateSkeletonRows,
    getCachedJobId,
    getJobOverviewType,
    getJobOverviewTypeByPathname,
    getJobsListPageSize,
    getJobsListQueryParams,
    getJobsTagsOptions,
    setJobsListQueryParams,
    TAG_KEY_ARCHIVED
} from "utils/jobs";
import { invalidateJobQueries } from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { projectQueryParams } from "utils/routeProject";
import { getScopedRoutes } from "utils/scopedRoutes";

export const useJobs = (
    query: ListJobsQuery,
    enabled: boolean = true
): [Array<GenericJob>, number, boolean, () => void, boolean] => {
    const currentProjectId = useProjectId();
    const resolvedQuery = query.project_id
        ? query
        : { ...query, ...(projectQueryParams(currentProjectId) ?? {}) };
    const cacheID = getCachedJobId(resolvedQuery);
    const { getQueryOptions } = useQueryOptions();
    const {
        data: jobsResponse,
        isLoading,
        error,
        isError,
        refetch: refreshJobs
    } = useGetJobs(
        resolvedQuery,
        getQueryOptions({
            queryKey: [ReactQueryKeys.JobList, cacheID],
            enabled
        })
    );

    useEffect(() => {
        if (isError && error) {
            console.log("TODO: handle error");
        }
    }, [error, isError]);

    const jobs = jobsResponse?.data?.jobs ?? [];
    const jobsPresent = jobs.length > 0;

    return [
        jobs,
        jobsResponse?.data?.total_num_items ?? 0,
        isLoading,
        refreshJobs,
        jobsPresent
    ];
};

export const useTrainingJobs = (
    query: ListJobsQuery
): [Array<GenericJob & PredictiveQueryInfo>, number, boolean] => {
    const cacheID = getCachedJobId(query);
    const { getQueryOptions } = useQueryOptions();

    const { data: jobsResponse, isLoading } = useGetJobs(
        query,
        getQueryOptions({ queryKey: [ReactQueryKeys.JobList, cacheID] })
    );

    const { queryYamlResponse, taskType } = useGetQueryYaml(
        jobsResponse?.data?.jobs ?? []
    );

    const modifiedResponse = (jobsResponse?.data?.jobs ?? [])?.map((job) => {
        const jobId = job.job_id;
        return {
            ...job,
            source:
                job?.source ?? (JobSource.NONE as unknown as GenericJobSource),
            queryYaml: queryYamlResponse[jobId] ?? "",
            taskType: taskType[jobId] ?? null
        };
    });

    return [
        modifiedResponse as Array<GenericJob & PredictiveQueryInfo>,
        jobsResponse?.data?.total_num_items ?? 0,
        isLoading
    ];
};

export const useGetQueryYaml = (jobs: GenericJob[]) => {
    const projectId = useProjectId() ?? "";
    const queryYaml: { [key: string]: string } = {};
    const taskType: { [key: string]: TaskType } = {};
    const requestBody: { [key: string]: string } = jobs?.reduce<
        Record<string, string>
    >((req, job) => {
        if (job?.job_id && job?.metadata?.tags?.pquery_id) {
            req[job.job_id] = job?.metadata?.tags?.pquery_id;
        }
        return req;
    }, {});
    const jobIds = getKeys(requestBody ?? {})
        .sort()
        .join("_");

    const { getQueryOptions } = useQueryOptions();
    const { data: queryYamlResponse } = useGetV1PredictiveQueriesMany(
        requestBody,
        { project_id: projectId },
        getQueryOptions({
            queryKey: [ReactQueryKeys.QueryYAML, jobIds],
            enabled: getKeys(requestBody ?? {}).length > 0 && Boolean(projectId)
        })
    );

    if (queryYamlResponse) {
        const data = queryYamlResponse.data as {
            [key: string]: [string, TaskType];
        };
        for (const key in data) {
            queryYaml[key] = data[key][0];
            taskType[key] = data[key][1];
        }
    }

    return {
        queryYamlResponse: queryYaml,
        taskType
    };
};

export const useGetActiveJobs = (
    jobs: GenericJob[],
    enabled: boolean,
    projectId?: string | null
) => {
    const isWindowFocused = useWindowFocus();

    const previousDataRef = useRef({}); // store previous data
    const lastRefetchTimeRef = useRef<number>(Date.now());

    const job_ids: string[] = jobs?.map((job) => job?.job_id);
    const { data: activeJobsResponse, refetch } = useGetJobsActiveJobs(
        { project_id: projectId ?? "" },
        job_ids,
        {
            query: {
                enabled: enabled && Boolean(projectId),
                queryKey: [ReactQueryKeys.ActiveJobs, job_ids]
            }
        }
    );

    useEffect(() => {
        previousDataRef.current = activeJobsResponse?.data ?? {};
    }, [activeJobsResponse?.data]);

    useEffect(() => {
        const pollInterval = isWindowFocused ? 30000 : 900000;
        const intervalId = setInterval(() => {
            refetch();
            lastRefetchTimeRef.current = Date.now();
        }, pollInterval);
        return () => {
            const timeSinceLastRefetch =
                Date.now() - lastRefetchTimeRef.current;
            if (timeSinceLastRefetch >= 30000) {
                refetch();
                lastRefetchTimeRef.current = Date.now();
            }
            clearInterval(intervalId);
        };
    }, [isWindowFocused, refetch]);

    return {
        activeJobsResponse: (enabled
            ? (activeJobsResponse?.data ?? {})
            : {}) as {
            [job_id: string]: jobStatus | GenericJob;
        },
        previousActiveJobsResponse: (enabled
            ? (previousDataRef?.current ?? {})
            : {}) as {
            [job_id: string]: jobStatus | GenericJob;
        }
    };
};

export const useJobsTags = (
    tagsType: TagType
): [Record<string, JobTags>, boolean] => {
    const isTrainingTag = tagsType === TagType.TRAINING_JOB;
    const projectId = useProjectId() ?? "";
    const { getQueryOptions } = useQueryOptions();
    const { data: trainingTagResponse, isLoading: isTrainingTagLoading } =
        useGetJobsTrainingTags(
            { project_id: projectId },
            getQueryOptions({
                queryKey: [ReactQueryKeys.TrainingTags],
                enabled: isTrainingTag && Boolean(projectId)
            })
        );

    const { data: predictionTagResponse, isLoading: isPredictionTagLoading } =
        useGetJobsPredictionTags(
            { project_id: projectId },
            getQueryOptions({
                queryKey: [ReactQueryKeys.PredictionTags],
                enabled: !isTrainingTag && Boolean(projectId)
            })
        );

    const tags: Record<string, JobTags> = isTrainingTag
        ? (trainingTagResponse?.data ?? {})
        : (predictionTagResponse?.data ?? {});

    const loading = isTrainingTag
        ? isTrainingTagLoading
        : isPredictionTagLoading;

    return [tags, loading];
};

export const usePrefetchNextPages = (
    currentPageLoading: boolean,
    currentPageQuery: ListJobsQuery
) => {
    const queryClient = useQueryClient();
    const STAGGERED_DELAY_MS = 2500;

    // Track which pages have been prefetched
    const [pagePlus1Done, setPagePlus1Done] = useState(false);
    const [pagePlus2Done, setPagePlus2Done] = useState(false);

    // Reset prefetch state when query changes (page navigation, filter changes, etc.)
    const currentCacheId = getCachedJobId(currentPageQuery);
    const prevCacheIdRef = useRef(currentCacheId);

    useEffect(() => {
        if (prevCacheIdRef.current !== currentCacheId) {
            prevCacheIdRef.current = currentCacheId;
            setPagePlus1Done(false);
            setPagePlus2Done(false);
        }
    }, [currentCacheId]);

    // Prefetch page + 1
    useEffect(() => {
        if (pagePlus1Done) return;

        const prefetchPage = () => {
            const nextPageQuery = {
                ...currentPageQuery,
                page: (currentPageQuery.page || 0) + 1
            };
            const cacheId = getCachedJobId(nextPageQuery);

            queryClient
                .prefetchQuery({
                    queryKey: [ReactQueryKeys.JobList, cacheId],
                    queryFn: () => getJobs(nextPageQuery)
                })
                .then(() => setPagePlus1Done(true));
        };

        if (currentPageLoading) {
            const timeout = setTimeout(prefetchPage, STAGGERED_DELAY_MS);
            return () => clearTimeout(timeout);
        } else {
            prefetchPage();
        }
    }, [currentPageQuery, currentPageLoading, pagePlus1Done, queryClient]);

    // Prefetch page + 2
    useEffect(() => {
        if (!pagePlus1Done || pagePlus2Done) return;

        const timeout = setTimeout(() => {
            const nextPageQuery = {
                ...currentPageQuery,
                page: (currentPageQuery.page || 0) + 2
            };
            const cacheId = getCachedJobId(nextPageQuery);

            queryClient
                .prefetchQuery({
                    queryKey: [ReactQueryKeys.JobList, cacheId],
                    queryFn: () => getJobs(nextPageQuery)
                })
                .then(() => setPagePlus2Done(true));
        }, STAGGERED_DELAY_MS);

        return () => clearTimeout(timeout);
    }, [currentPageQuery, pagePlus1Done, pagePlus2Done, queryClient]);
};

export const useJobsListPageSize = () => {
    const dim = useWindowDimensions();
    const defaultPageSize = getJobsListPageSize(dim?.height || 0);
    const debouncedPageSize = useDebounce(defaultPageSize, 1000);

    return debouncedPageSize;
};

/**
 * Merges actively polled jobs with the current paginated job list.
 *
 * The activeJobsResponse contains real-time status updates for all active jobs
 * (Running, Queued, etc.) from polling. This hook filters those active jobs
 * to match the current URL filters (job_type and status) and merges them with
 * the paginated results, ensuring no duplicates.
 *
 * Note: Status filtering is important here because activeJobsResponse contains
 * ALL active jobs regardless of the user's status filter. Without filtering,
 * selecting "Running" status would incorrectly show Queued jobs as well.
 */
export const useMergedActiveJobs = (
    activeJobsResponse: { [job_id: string]: jobStatus | GenericJob },
    currentJobs: GenericJob[],
    pageSearch: string
): GenericJob[] => {
    const mergedJobs = useMemo(() => {
        const genericJobs = getValues(activeJobsResponse)?.filter(
            (job) => typeof job === "object"
        ) as GenericJob[];

        const searchParams = new URLSearchParams(pageSearch);
        const statusFilter = searchParams.getAll("status");

        const filtered = genericJobs.filter((job) => {
            const matchesJobType = pageSearch?.includes(job.job_type);
            const matchesStatus =
                statusFilter.length === 0 ||
                statusFilter.includes(job.metadata?.status as string);
            return matchesJobType && matchesStatus;
        });

        const updatedJobs = [...filtered, ...currentJobs].filter(
            (job, index, self) =>
                index === self.findIndex((t) => t.job_id === job.job_id)
        );

        return updatedJobs?.sort();
    }, [activeJobsResponse, currentJobs, pageSearch]);

    return mergedJobs;
};

export const useJobsListTableData = (
    query: ListJobsQuery,
    totalNumJobs: number,
    customColumnsVisibility?: { [key: string]: boolean }
) => {
    const [sorting, setSorting] = useState<SortingState>([
        { id: query?.sort_by as string, desc: query?.sort_desc as boolean }
    ]);

    useEffect(() => {
        setSorting([
            { id: query?.sort_by as string, desc: query?.sort_desc as boolean }
        ]);
    }, [query?.sort_by, query?.sort_desc]);

    const numSkeletonRow = useMemo(() => {
        const currentPage = query?.page ?? 0;
        const currentLimit = query?.limit ?? 5;
        return calculateSkeletonRows(currentPage, currentLimit, totalNumJobs);
    }, [query?.page, query?.limit, totalNumJobs]);

    const columnsVisibility = useMemo(
        () => ({
            test_metric: query.job_type?.includes(JobType.TRAINING_JOB)
                ? true
                : false,
            task_type: customColumnsVisibility?.task_type ?? false,
            source: customColumnsVisibility?.source ?? false
        }),
        [query.job_type, customColumnsVisibility]
    );

    return {
        sorting,
        setSorting,
        numSkeletonRow,
        columnsVisibility
    };
};

export const useJobsListUrlState = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const projectId = useProjectId();
    const pageSize = useJobsListPageSize();

    const params = useMemo(() => {
        return Object.fromEntries(
            new URLSearchParams(location?.search).entries()
        );
    }, [location]);

    const query: ListJobsQuery = useMemo(() => {
        return getJobsListQueryParams(params, pageSize, projectId ?? "");
    }, [params, pageSize, projectId]);

    const setQuery = useCallback(
        (newQuery: ListJobsQuery) => {
            const params = new URLSearchParams(
                setJobsListQueryParams(newQuery, pageSize)
            );
            navigate({
                pathname: window.location.pathname,
                search: params.toString()
            });
        },
        [navigate, pageSize]
    );

    return { query, setQuery };
};

export const useJobOverviewNavigation = (
    jobs: unknown[],
    explicitProjectId?: string | null
) => {
    const navigate = useNavigate();
    const routeProjectId = useProjectId();
    const projectId = explicitProjectId ?? routeProjectId;
    const scopedRoutes = getScopedRoutes(projectId);

    const defaultPrevPage =
        (window?.location?.pathname?.includes("models")
            ? scopedRoutes.ModelJobsList
            : scopedRoutes.PredictionJobsList) + window?.location?.search;

    const [[jobId, jobOverviewType], setJobOverview] = useState<
        [string | null, JobOverviewType | null]
    >([null, null]);
    const [prevPage, setPrevPage] = useState<string>(defaultPrevPage);

    useEffect(() => {
        const pathName = window?.location?.pathname;

        if (jobs?.length > 0) {
            let jobId: string | null = null;
            const [page, jobOverviewType] = getJobOverviewTypeByPathname(
                pathName,
                window?.location?.search,
                projectId
            );
            const pagePath =
                page?.replaceAll(window?.location?.search, "") ?? "";

            if (page) {
                try {
                    jobId = pathName.replaceAll(pagePath, "")?.split("/")[0];
                    if (jobId) {
                        setJobOverview([jobId, jobOverviewType]);
                    }
                } catch (e) {
                    console.warn("Error parsing job ID from pathname:", e);
                }
            } else {
                setPrevPage(
                    window?.location?.pathname + window?.location?.search
                );
            }
        }
    }, [jobs, projectId]);

    const openJobDetails = (jobID: string | null, jobType: JobType | null) => {
        const [page, jobOverviewType] = getJobOverviewType(
            jobID,
            jobType,
            window?.location?.search,
            projectId
        );
        navigate(page ? page : prevPage, { replace: true });
        setJobOverview([jobID, jobOverviewType]);
    };

    const closeJobDetails = () => {
        navigate(prevPage, { replace: true });
        setJobOverview([null, null]);
    };

    return {
        jobId,
        jobOverviewType,
        prevPage,
        openJobDetails,
        closeJobDetails,
        setJobOverview
    };
};

export interface UseManageJobParams {
    jobId: string;
    jobType: JobType;
    jobName?: string;
    onArchive?: () => void;
    onRecover?: () => void;
    handleError: (error: CustomError, title?: string) => void;
    clearErrors: () => void;
    dialogRef: React.MutableRefObject<{
        toggleDialog: (
            title?: string,
            message?: string,
            options?: object | null,
            enableSkipButton?: boolean,
            confirmButtonText?: string,
            confirmButtonVariant?: ConfirmButtonVariant,
            cancelButtonText?: string
        ) => void;
    } | null>;
    setPageLoading: (loading: boolean) => void;
}

type JobActionType = "archive" | "recover" | null;

export const useManageJob = ({
    jobId,
    jobType,
    jobName,
    onArchive,
    onRecover,
    handleError,
    clearErrors,
    dialogRef,
    setPageLoading
}: UseManageJobParams) => {
    const [dialogType, setDialogType] = useState<JobActionType>(null);
    const [isArchived, setIsArchived] = useState(false);
    const { mutate: mutateArchived } = useCreateJobsArchiveByJobId();
    const queryClient = useQueryClient();
    const projectId = useProjectId();

    const onClickArchive = () => {
        setDialogType("archive");
        const displayName = getIdHash(jobName ?? jobId);
        dialogRef?.current?.toggleDialog(
            "Archive Job",
            `Are you sure you want to archive Job "${displayName}"? You'll still be able to restore it later if needed.`,
            {},
            false,
            "Archive",
            "destructive"
        );
    };

    const onClickRecover = () => {
        setDialogType("recover");
        const displayName = getIdHash(jobName ?? jobId);
        dialogRef?.current?.toggleDialog(
            "Restore Job",
            `Do you want to restore Job "${displayName}"?`,
            {},
            false,
            "Restore",
            "outline"
        );
    };

    const onConfirm = () => {
        dialogRef?.current?.toggleDialog();
        clearErrors();

        const shouldDelete = dialogType === "archive";

        setPageLoading(true);

        mutateArchived(
            {
                jobId,
                params: {
                    archived: shouldDelete,
                    project_id: projectId ?? ""
                }
            },
            {
                onSuccess: () => {
                    setIsArchived(shouldDelete);
                    invalidateJobQueries(queryClient, jobType, jobId);
                    if (shouldDelete) {
                        onArchive?.();
                    } else {
                        onRecover?.();
                    }
                },
                onError: (error) => {
                    const title = `Unable to ${dialogType} job. `;
                    handleError(error as CustomError, title);
                },
                onSettled: () => {
                    setPageLoading(false);
                }
            }
        );

        setDialogType(null);
    };

    return {
        onClickArchive,
        onClickRecover,
        onConfirm,
        setIsArchived,
        isDeleteDialog: dialogType === "archive",
        isRecoverDialog: dialogType === "recover",
        isManagingJob: dialogType !== null,
        isArchived
    };
};

export const useJobSearch = ({
    query,
    setQuery
}: {
    query: ListJobsQuery;
    setQuery: (newQuery: ListJobsQuery) => void;
}) => {
    const projectId = useProjectId();
    const [searchKey, setSearchValue] = useState(query?.search_keyword);
    const debouncedSearchValue = useDebounce(searchKey, 1000); // 1 seconds debounce
    const pageRef = useRef(window.location.pathname);
    const pageSize = useJobsListPageSize();

    // Update query when debounced value changes
    useEffect(() => {
        const prevPath = pageRef.current;
        pageRef.current = window.location.pathname;
        if (
            debouncedSearchValue !== undefined &&
            prevPath === window.location.pathname
        ) {
            const params = Object.fromEntries(
                new URLSearchParams(window.location?.search).entries()
            );

            const query: ListJobsQuery = getJobsListQueryParams(
                params,
                pageSize,
                projectId ?? ""
            );
            setQuery({
                ...query,
                search_keyword: debouncedSearchValue,
                page: 1 // Reset to first page on search
            });
            return;
        }
    }, [debouncedSearchValue, pageSize, projectId, setQuery]);

    useEffect(() => {
        if (query.search_keyword === undefined) {
            setSearchValue(undefined);
        }
    }, [query.search_keyword]);

    return { searchKey, setSearchValue };
};

export const useJobsTagsFilter = ({
    query,
    setQuery,
    type
}: {
    query: ListJobsQuery;
    setQuery: (newQuery: ListJobsQuery) => void;
    type: string;
}) => {
    const tagsToFetch =
        type === "Models" ? TagType.TRAINING_JOB : TagType.BATCH_PREDICTION_JOB;

    const [tags, tagsLoading] = useJobsTags(tagsToFetch);
    useEffect(() => {
        // Clean up tags_filter to remove tags that no longer exist
        const currentTagsFilter = query.tags_filter ?? [];
        // Don't run cleanup if tags are still loading or if no tags are selected
        if (currentTagsFilter.length === 0 || !tags || tagsLoading) return;

        const validTagValues = [
            ...getJobsTagsOptions(tags).map((opt) => opt.value)
        ];
        const filteredTags = currentTagsFilter.filter(
            (tag) => tag === TAG_KEY_ARCHIVED || validTagValues.includes(tag)
        );
        // Only update if there are invalid tags
        if (filteredTags.length !== currentTagsFilter.length) {
            setQuery({
                ...query,
                tags_filter: filteredTags
            });
        }
    }, [tags, tagsLoading, query.tags_filter, query, setQuery]);

    const tagsOptions = useMemo(() => [...getJobsTagsOptions(tags)], [tags]);
    const tagValues = tagsOptions.map((opt) => opt.value);
    const selectedTags = (query.tags_filter ?? []).filter(
        (tag) => tag !== TAG_KEY_ARCHIVED && tagValues.includes(tag)
    );

    return { tags, tagsOptions, selectedTags };
};
