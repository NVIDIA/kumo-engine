// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GenericJob,
    JobSortField,
    JobTags,
    JobType,
    KumoapiCommonJobStatus,
    ListJobsQuery,
    StatedbConfigCommonJobStatus as JobStatus
} from "gen/v0/types";
import { RunMode } from "gen/v1/types";

import { JSGraph } from "types/graphs";
import { SelectOption } from "types/inputs";
import { CreateTrainingJobArgs, JobOverviewType } from "types/jobs";
import { PredictiveQueryInfo } from "types/predictiveQueries";

import { isNullish } from "utils/Helpers/typeGuards";

import { PROTECTED_TAG_KEYS } from "./Constants/config";
import { getValues } from "./Helpers/objectHelpers";
import { toTitleCase } from "./Helpers/stringHelpers";
import {
    getProjectPath,
    jobStatus,
    JobStatusType,
    navRoutes
} from "./Constants";

export const StatusColors: Record<jobStatus, string> = {
    [jobStatus.Cancelled]: "var(--kumo-grey-950)",
    [jobStatus.Done]: "var(--color-green-600)",
    [jobStatus.Failed]: "var(--color-red-700)",
    [jobStatus.Queued]: "var(--color-yellow-700)",
    [jobStatus.Running]: "var(--color-violet-700)",
    [jobStatus.Untrained]: "var(--color-fuchsia-700)",
    [jobStatus.None]: "var(--kumo-grey-500)"
};

type JobRoutes = {
    ModelJobsList: string;
    PredictionJobsList: string;
    TrainingJob: string;
    PredictionJob: string;
    TrainTableJob: string;
    PredictionTableJob: string;
};

const getJobRoutes = (projectId?: string | null): JobRoutes =>
    getProjectPath(projectId);

const matchJobRouteByPath = (
    pathName: string,
    routeSet: JobRoutes
): [string | null, JobOverviewType | null] => {
    if (pathName?.startsWith(routeSet.TrainingJob)) {
        return [routeSet.TrainingJob, JobOverviewType.TRAINING_JOB];
    }
    if (pathName?.startsWith(routeSet.PredictionJob)) {
        return [routeSet.PredictionJob, JobOverviewType.BATCH_PREDICTION_JOB];
    }
    if (pathName?.startsWith(routeSet.TrainTableJob)) {
        return [routeSet.TrainTableJob, JobOverviewType.TRAIN_TABLE_JOB];
    }
    if (pathName?.startsWith(routeSet.PredictionTableJob)) {
        return [
            routeSet.PredictionTableJob,
            JobOverviewType.PREDICTION_TABLE_JOB
        ];
    }

    return [null, null];
};

export const cleanedJobType = (jobType: string) => {
    return toTitleCase(
        jobType
            ?.toString()
            ?.replaceAll("_", " ")
            ?.toLowerCase()
            ?.replaceAll(" job", " ")
            ?.replaceAll("generate", "")
            ?.trim()
    );
};

export const getDefaultModelJobsPageURL = (
    projectId?: string | null
): string => {
    if (!projectId) return navRoutes.Home;
    return `${getProjectPath(projectId).ModelJobsList}?page=1&job_type=${JobType.TRAINING_JOB}`;
};

export const getDefaultPredictionJobsPageURL = (
    projectId?: string | null
): string => {
    if (!projectId) return navRoutes.Home;
    return `${getProjectPath(projectId).PredictionJobsList}?page=1&job_type=${JobType.BATCH_PREDICTION_JOB}`;
};

export const getJobOverviewType = (
    jobId: string | null,
    jobType: JobType | null,
    pathQuery: string,
    projectId?: string | null
): [string | null, JobOverviewType | null] => {
    const routeSet = getJobRoutes(projectId);
    let page: string | null = null;
    let jobOverviewType: JobOverviewType | null = null;
    if (
        jobId &&
        jobType &&
        [JobType.TRAINING_JOB as string].includes(jobType)
    ) {
        page = routeSet.TrainingJob + jobId;
        jobOverviewType = JobOverviewType.TRAINING_JOB;
    }
    if (
        jobId &&
        jobType &&
        [JobType.BATCH_PREDICTION_JOB as string].includes(jobType)
    ) {
        page = routeSet.PredictionJob + jobId;
        jobOverviewType = JobOverviewType.BATCH_PREDICTION_JOB;
    }
    if (
        jobId &&
        jobType &&
        [JobType.GENERATE_TRAIN_TABLE_JOB as string].includes(jobType)
    ) {
        page = routeSet.TrainTableJob + jobId;
        jobOverviewType = JobOverviewType.TRAIN_TABLE_JOB;
    }
    if (
        jobId &&
        jobType &&
        [JobType.GENERATE_PREDICTION_TABLE_JOB as string].includes(jobType)
    ) {
        page = routeSet.PredictionTableJob + jobId;
        jobOverviewType = JobOverviewType.PREDICTION_TABLE_JOB;
    }

    if (page && pathQuery) {
        page += pathQuery;
    }

    return [page, jobOverviewType];
};

export const getJobOverviewTypeByPathname = (
    pathName: string,
    pathQuery: string,
    projectId?: string | null
): [string | null, JobOverviewType | null] => {
    if (!projectId) return [null, null];
    const routeSets = [getJobRoutes(projectId)];

    for (const routeSet of routeSets) {
        const [page, jobOverviewType] = matchJobRouteByPath(pathName, routeSet);
        if (page && jobOverviewType) {
            return [page + (pathQuery ?? ""), jobOverviewType];
        }
    }
    return [null, null];
};
export const TAG_KEY_ARCHIVED = "archived";

export const getJobsTagsOptions = (
    tags: Record<string, JobTags>
): SelectOption[] => {
    const uniqueTags = new Set<string>();

    Object.values(tags).forEach((tag) => {
        Object.entries(tag?.tags ?? {}).forEach(([key, value]) => {
            // Skip system/ID tag keys (pquery_name, pquery_id, parent_job,
            // model_id, archived, ...) so job/pquery IDs and other internal
            // tags don't bloat the filter dropdown (kumo-ai/kumo#29898). Only
            // user-defined tags remain as filter options.
            if (PROTECTED_TAG_KEYS.has(key)) {
                return;
            }
            uniqueTags.add(value as string);
        });
    });

    return [...uniqueTags].map((value) => ({ value, label: value }));
};

export type BadgeVariant =
    | "done"
    | "failed"
    | "queued"
    | "running"
    | "cancelled"
    | "default"
    | "pink";

export function getColorVariant(
    status?: JobStatusType | KumoapiCommonJobStatus
): BadgeVariant {
    if (!status) {
        return "default";
    }
    switch (status as jobStatus) {
        case jobStatus.Done:
            return "done";
        case jobStatus.Failed:
            return "failed";
        case jobStatus.Queued:
            return "queued";
        case jobStatus.Running:
        case jobStatus.Untrained:
            return "running";
        case jobStatus.Cancelled:
        case jobStatus.None:
            return "cancelled";
        default:
            return "default";
    }
}

/**
 * Calculate page size based on window height.
 * Used by both loaders and hooks for consistent query keys.
 */
export const getJobsListPageSize = (windowHeight?: number) => {
    const height = windowHeight ?? window.innerHeight;
    // Actual rendered DataTable row height. Must match the real row height or
    // the computed page size fetches too many rows and the table overflows the
    // viewport (the bug was assuming 42px when rows render at 44px).
    const rowHeight = 44;
    // Non-row chrome above and below the data rows (page padding + title +
    // filter row + spacer + table header + pagination footer ≈ 237px). This is
    // constant across resolutions, so a single offset (not a >1200px step) both
    // avoids overflow and fills the available height — the slight excess over
    // 237 leaves a few px of safety margin without wasting a full row.
    const headerOffset = 240;
    const pageHeight = height - headerOffset;
    return Math.floor(pageHeight / rowHeight);
};

/**
 * Returns true when `pathname` addresses the "models" or "predictions"
 * jobs-list route, matching the segment whether or not the URL carries a
 * trailing slash. The list page derives its job-type default from this check,
 * so a trailing-slash-only match (e.g. `pathname.includes("/predictions/")`)
 * silently drops the type filter on the canonical no-slash form
 * `/project/<id>/predictions`, making the page list every job type
 * (issue #29832).
 */
export const isJobsListPageForSegment = (
    pathname: string,
    segment: "models" | "predictions"
): boolean => new RegExp(`/${segment}(/|$)`).test(pathname);

/**
 * Returns true when the jobs list addressed by `pathname` + `search` is a
 * predictions view: either the URL is the predictions jobs-list route, or the
 * query string filters to a prediction job type.
 */
export const isPredictionsJobsPage = (
    pathname: string,
    search: string
): boolean =>
    isJobsListPageForSegment(pathname, "predictions") ||
    search.includes(JobType.BATCH_PREDICTION_JOB) ||
    search.includes(JobType.GENERATE_PREDICTION_TABLE_JOB);

export const getJobsListQueryParams = (
    params: {
        [k: string]: string;
    },
    pageSize: number,
    projectId: string
): ListJobsQuery => {
    const pathName = window?.location?.pathname ?? "";

    const hasAnyJobType = Object?.values(JobType)?.some((jt) =>
        window?.location?.search?.includes(jt)
    );

    let rawTypes = [];
    if (isJobsListPageForSegment(pathName, "models") && !hasAnyJobType) {
        rawTypes = [JobType.TRAINING_JOB];
    } else if (
        isJobsListPageForSegment(pathName, "predictions") &&
        !hasAnyJobType
    ) {
        rawTypes = [JobType.BATCH_PREDICTION_JOB];
    } else {
        rawTypes = params.job_type?.split("#").filter(Boolean) || [];
    }

    return {
        project_id: projectId,
        limit: params?.limit ? Number(params.limit) : pageSize,
        page: params?.page ? Number(params.page) - 1 : 0,
        time_range_start: params?.time_range_start
            ? (new Date(params.time_range_start) as unknown as string)
            : null,
        time_range_end: params?.time_range_end
            ? (new Date(params.time_range_end) as unknown as string)
            : null,
        status: params?.status ? (params.status.split("#") as jobStatus[]) : [],
        job_type: rawTypes.length > 0 ? (rawTypes as JobType[]) : [],
        tags: {},
        tags_filter: params?.tags_filter
            ? (params.tags_filter.split("#") as string[])
            : [],
        search_keyword: params?.search_keyword,
        sort_by: params?.sort_by ? (params.sort_by as JobSortField) : undefined,
        sort_desc: params?.sort_desc === "true"
    } as ListJobsQuery;
};

export const setJobsListQueryParams = (
    newQuery: ListJobsQuery,
    pageSize: number
): Record<string, string> => {
    const rawParams: Record<string, string> = {};

    if (newQuery) {
        if (newQuery.limit && newQuery.limit !== pageSize) {
            rawParams.limit = newQuery.limit.toString();
        }
        if (newQuery.page) {
            rawParams.page = newQuery.page.toString();
        }
        if (newQuery.time_range_start) {
            rawParams.time_range_start = newQuery.time_range_start; //.toISOString() as string;
        }
        if (newQuery.time_range_end) {
            rawParams.time_range_end = newQuery.time_range_end; //.toISOString() as string;
        }
        if (newQuery.job_type && newQuery.job_type.length > 0) {
            rawParams.job_type = newQuery.job_type.join("#");
        }
        if (newQuery.status && newQuery.status.length > 0) {
            rawParams.status = newQuery.status.join("#");
        }
        if (newQuery.tags_filter && newQuery.tags_filter.length > 0) {
            rawParams.tags_filter = newQuery.tags_filter.join("#");
        }
        if (newQuery.search_keyword) {
            rawParams.search_keyword = newQuery.search_keyword;
        }
        if (newQuery.sort_by) {
            rawParams.sort_by = newQuery.sort_by;
        }
        if (!isNullish(newQuery.sort_desc)) {
            rawParams.sort_desc = newQuery.sort_desc.toString();
        }
    }

    return rawParams;
};

export const getTrainingJobsSelectListQueryParams = (
    projectId: string
): ListJobsQuery => {
    // Fit the model picker to the viewport (row height 44; offset covers the
    // page title/description, search row, table header, pagination and page
    // padding) so it fills the available height with model rows instead of a
    // fixed 10-row block that leaves a large gap on tall screens. The picker's
    // wrapper stretches a full page's rows to absorb the sub-row remainder.
    const rowHeight = 44;
    const headerOffset = 285;
    const pageSize = Math.max(
        1,
        Math.floor((window.innerHeight - headerOffset) / rowHeight)
    );

    return {
        project_id: projectId,
        limit: pageSize,
        page: 0,
        time_range_start: undefined,
        time_range_end: undefined,
        status: ["DONE"],
        job_type: ["TRAINING_JOB"],
        sort_by: undefined,
        sort_desc: undefined,
        tags: {},
        search_keyword: ""
    };
};

export const calculateSkeletonRows = (
    currentPage: number,
    currentLimit: number,
    totalNumJobs: number
): number => {
    const isLastPage = (currentPage + 1) * currentLimit >= totalNumJobs;
    return isLastPage
        ? totalNumJobs % currentLimit || (totalNumJobs === 0 ? 0 : currentLimit)
        : currentLimit;
};

export const mergeActiveJobsWithCurrent = (
    activeJobsResponse: { [job_id: string]: jobStatus | GenericJob },
    currentJobs: GenericJob[],
    pageSearch: string
): GenericJob[] => {
    const genericJobs = getValues(activeJobsResponse)?.filter(
        (job) => typeof job === "object"
    ) as GenericJob[];

    const filtered = genericJobs.filter((job) =>
        pageSearch?.includes(job.job_type)
    );

    const updatedJobs = [...filtered, ...currentJobs].filter(
        (job, index, self) =>
            index === self.findIndex((t) => t.job_id === job.job_id)
    );

    return updatedJobs?.sort();
};

export const isAnyJobFilterApplied = (query: ListJobsQuery): boolean => {
    return (
        (query?.search_keyword && query?.search_keyword !== "") ||
        (query?.tags_filter && query?.tags_filter?.length > 0) ||
        (query?.status &&
            query?.status?.length > 0 &&
            !query?.status?.includes(JobStatus.RUNNING)) ||
        (query?.page !== undefined && query?.page > 0)
    );
};

export const getCachedJobId = (query: ListJobsQuery) => {
    const cacheID = [
        query.limit,
        query.job_type?.sort().join("_") ?? "",
        query.page,
        query.search_keyword,
        query.sort_by ?? "",
        query.sort_desc ? "true" : "false",
        query.status?.sort().join("_") ?? "",
        query.tags_filter?.sort().join("_") ?? "",
        query.project_id ?? ""
    ]
        .filter((a) => a !== "")
        .join("_");
    return cacheID;
};

export const PredictionTableStatusTitle: Record<
    KumoapiCommonJobStatus | jobStatus,
    string
> = {
    [jobStatus.Cancelled]: "Prediction table generation was cancelled",
    [jobStatus.Done]: "Prediction table generated",
    [jobStatus.Failed]: "Prediction table generation failed",
    [jobStatus.Queued]: "This job is queued",
    [jobStatus.Running]: "Generating prediction table...",
    [jobStatus.Untrained]: "This job is queued",
    [jobStatus.None]: "",
    [KumoapiCommonJobStatus.NOT_STARTED]: "Job has not started",
    [KumoapiCommonJobStatus.MISSING]: "Job status missing"
};

export const PredictionStatusTitle: Record<
    KumoapiCommonJobStatus | jobStatus,
    string
> = {
    [jobStatus.Cancelled]: "Prediction job was cancelled.",
    [jobStatus.Done]: "Prediction is complete.",
    [jobStatus.Failed]: "Prediction failed with the following errors",
    [jobStatus.Queued]: "This prediction job is queued.",
    [jobStatus.Running]: "Precition in progress..",
    [jobStatus.Untrained]: "This prediction job is queued.",
    [jobStatus.None]: "",
    [KumoapiCommonJobStatus.NOT_STARTED]: "Prediction has not started",
    [KumoapiCommonJobStatus.MISSING]: "Prediction status missing"
};

export const TrainingStatusTitle: Record<jobStatus, string> = {
    [jobStatus.Cancelled]: "This training job was cancelled by the user",
    [jobStatus.Done]: "Training complete",
    [jobStatus.Failed]: "Training job failed",
    [jobStatus.Queued]: "This training job is queued",
    [jobStatus.Running]: "Training in progress...",
    [jobStatus.Untrained]: "This training job is untrained",
    [jobStatus.None]: ""
};
export const TrainingTableStatusTitle: Record<
    KumoapiCommonJobStatus | jobStatus,
    string
> = {
    [jobStatus.Cancelled]: "Training table generation was cancelled",
    [jobStatus.Done]: "Training table generated",
    [jobStatus.Failed]: "Training table generation failed",
    [jobStatus.Queued]: "This job is queued",
    [jobStatus.Running]: "Generating training table...",
    [jobStatus.Untrained]: "This job is queued",
    [jobStatus.None]: "",
    [KumoapiCommonJobStatus.NOT_STARTED]: "Job has not started",
    [KumoapiCommonJobStatus.MISSING]: "Job status missing"
};

export function getJobSource(source: string | null | undefined): string {
    return !source ? "" : ["SDK", "UI"].includes(source) ? source : "";
}

export function getCacheTime(
    status: jobStatus,
    short: number,
    long: number
): number {
    return status === jobStatus.Running || status === jobStatus.Queued
        ? short
        : long;
}

export function assembleCopyTrainingJobArgs(
    queryInfo: PredictiveQueryInfo | null
): CreateTrainingJobArgs {
    const graphDef = queryInfo?.graphDefinition;

    const graphId = (queryInfo?.graphSnapshotId || "")?.split?.("@")[0];

    const jsGraph: JSGraph | null = graphDef
        ? {
              id: graphId,
              name: queryInfo.graphName || "",
              tables: graphDef.tables || {},
              groups: graphDef.col_groups || [],
              totalLinks: 0,
              totalTables: Object.keys(graphDef.tables || {}).length,
              isActive: true
          }
        : null;

    return {
        name: `${queryInfo?.queryName || ""}_copy`,
        graph_id: graphId,
        query_yaml: queryInfo?.queryYaml?.trim() || "",
        query_dialect: queryInfo?.query_dialect?.trim() || "",
        model_planner_yaml: queryInfo?.modelPlannerYaml || "",
        run_mode: queryInfo?.runMode || RunMode.FAST,
        description: queryInfo?.description || "",
        graph_name: queryInfo?.graphName || "",
        graph: jsGraph
    };
}
