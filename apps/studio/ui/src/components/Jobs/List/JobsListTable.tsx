// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo } from "react";

import { SortingState, useIsMounted } from "@kumo-ai/kumo-ui-catalog";
import {
    GenericJob,
    JobType,
    ListJobsQuery,
    ListJobsQuerySortBy
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { JobListColumns } from "components/Jobs/List/JobListColumns";

import {
    useGetActiveJobs,
    useGetQueryYaml,
    useJobOverviewNavigation,
    useJobsListTableData,
    useMergedActiveJobs
} from "hooks/jobs/list/jobList";
import { useBodyScrollLock } from "hooks/utils/useBodyScrollLock";

import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { isAnyJobFilterApplied } from "utils/jobs";

import JobOverview from "../Overview/JobOverview";

import JobsListDataTable from "./JobsListDataTable";
import JobsListEmptyState from "./JobsListEmptyState";

export interface JobsListTableProps {
    query: ListJobsQuery;
    setQuery: (newQuery: ListJobsQuery) => void;
    jobs: GenericJob[];
    refetchJobs: () => void;
    totalNumJobs: number;
    loading: boolean;
    jobTypeOptions: { value: JobType; label: string }[];
    onArchive?: () => void;
    onRecover?: () => void;
    hideBodyWhenEmpty?: boolean;
    projectId?: string | null;
    customColumnsVisibility?: { [key: string]: boolean };
}

const JobsListTable = ({
    query,
    setQuery,
    jobs,
    refetchJobs,
    totalNumJobs,
    loading,
    jobTypeOptions,
    onArchive,
    onRecover,
    hideBodyWhenEmpty = false,
    projectId,
    customColumnsVisibility
}: JobsListTableProps) => {
    const isMounted = useIsMounted();

    const { queryYamlResponse, taskType } = useGetQueryYaml(jobs);

    const isAnyFilterApplied = isAnyJobFilterApplied(query);

    const { activeJobsResponse, previousActiveJobsResponse } = useGetActiveJobs(
        jobs,
        !isAnyFilterApplied,
        projectId
    );

    const updatedActiveJobs = useMergedActiveJobs(
        activeJobsResponse,
        jobs,
        window?.location?.search
    );

    const { jobId, jobOverviewType, openJobDetails } = useJobOverviewNavigation(
        jobs,
        projectId ?? query.project_id
    );

    useBodyScrollLock(isValid(jobId) && isValid(jobOverviewType));

    const { sorting, setSorting, numSkeletonRow, columnsVisibility } =
        useJobsListTableData(query, totalNumJobs, customColumnsVisibility);

    const columns = useMemo(
        () => JobListColumns(queryYamlResponse, taskType, activeJobsResponse),
        [queryYamlResponse, taskType, activeJobsResponse]
    );

    useEffect(() => {
        if (
            JSON.stringify(previousActiveJobsResponse) !==
                JSON.stringify(activeJobsResponse) &&
            getKeys(activeJobsResponse)?.length === 0
        ) {
            refetchJobs();
        }
    }, [previousActiveJobsResponse, activeJobsResponse, refetchJobs]);

    const setCurrentPage = useCallback(
        (newPage: number) => {
            setQuery({
                ...query,
                page: newPage + 1
            });
        },
        [query, setQuery]
    );

    const handleSort = useCallback(
        (sortBy: SortingState) => {
            const newSortBy = sortBy[0]?.id;
            const newSortDesc = sortBy[0]?.desc;

            if (
                query.sort_by !== newSortBy ||
                query.sort_desc !== newSortDesc
            ) {
                setQuery({
                    ...query,
                    sort_by: newSortBy as ListJobsQuerySortBy,
                    sort_desc: newSortDesc,
                    page: 1
                });
            }
        },
        [query, setQuery]
    );

    useEffect(() => {
        handleSort(sorting);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sorting]);

    const resetFilter = useCallback(() => {
        setQuery({
            ...query,
            search_keyword: undefined,
            status: [],
            page: 1,
            sort_by: undefined,
            sort_desc: undefined,
            tags_filter: [],
            job_type: [jobTypeOptions[0]?.value]
        });
    }, [query, setQuery, jobTypeOptions]);

    const handleJobSelect = useCallback(
        (rowData: object) => {
            const selectedJob = rowData as GenericJob;
            openJobDetails(
                selectedJob?.job_id,
                selectedJob?.job_type as JobType
            );
        },
        [openJobDetails]
    );

    if (!loading && updatedActiveJobs?.length === 0 && !hideBodyWhenEmpty) {
        return <JobsListEmptyState onResetFilters={resetFilter} />;
    }

    return (
        <>
            <JobsListDataTable
                data={updatedActiveJobs}
                columns={columns}
                loading={loading}
                pageCount={Math.ceil(totalNumJobs / (query?.limit ?? 1))}
                defaultPageSize={query.limit ?? 5}
                currentPage={query.page || 0}
                numSkeletonRow={numSkeletonRow}
                columnsVisibility={columnsVisibility}
                sorting={sorting}
                onRowClick={handleJobSelect}
                onPageChange={setCurrentPage}
                onSortingChange={setSorting}
                hideBodyWhenEmpty={hideBodyWhenEmpty}
            />
            {jobId && jobOverviewType && (
                <JobOverview
                    jobId={isMounted?.current === true ? jobId : null}
                    jobOverviewType={jobOverviewType}
                    onClose={(id: string | null, type: JobType | null) => {
                        openJobDetails(id, type);
                    }}
                    onArchive={() => {
                        openJobDetails(null, null);
                        onArchive?.();
                    }}
                    onRecover={onRecover}
                />
            )}
        </>
    );
};

export default WithErrorBoundary(JobsListTable);
