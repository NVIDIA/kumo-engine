// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";
import Spacer from "react-spacer";

import {
    getFromLocalStorage,
    storeToLocalStorage,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { JobType } from "gen/v0/types";

import EmptyEntities from "components/Empty/EmptyEntities";
import JobsListFilter from "components/Jobs/List/JobsListFilter";
import JobsListTable from "components/Jobs/List/JobsListTable";

import {
    useJobs,
    useJobsListUrlState,
    usePrefetchNextPages
} from "hooks/jobs/list/jobList";
import { useProjectId } from "hooks/project/projects";

import { PageNames } from "utils/Constants";
import { isAnyJobFilterApplied, isPredictionsJobsPage } from "utils/jobs";
import { ReactQueryKeys } from "utils/reactQueryKeys";

/**
 * Jobs List Component
 *
 * Fetches list of all jobs and renders them in a table
 */
interface JobsListProps {
    title: string;
    jobTypeOptions: { value: JobType; label: string }[];
}

const JOBS_LIST_COLUMN_VISIBILITY_KEY = "jobsListColumnVisibility";
const DEFAULT_COLUMN_VISIBILITY: { [key: string]: boolean } = {
    task_type: false,
    source: false
};

const JobsList = ({ title, jobTypeOptions }: JobsListProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectId();

    const { query, setQuery } = useJobsListUrlState();

    const [customColumns, setCustomColumnsVisibility] = useState<{
        [key: string]: boolean;
    }>(() => {
        const stored = getFromLocalStorage<{ [key: string]: boolean }>(
            JOBS_LIST_COLUMN_VISIBILITY_KEY
        );
        return stored ?? { ...DEFAULT_COLUMN_VISIBILITY };
    });

    useEffect(() => {
        storeToLocalStorage(JOBS_LIST_COLUMN_VISIBILITY_KEY, customColumns);
    }, [customColumns]);

    const handleToggleColumn = useCallback((col: string) => {
        setCustomColumnsVisibility((prev) => ({
            ...prev,
            [col]: !prev[col]
        }));
    }, []);

    const handleResetColumns = useCallback(() => {
        setCustomColumnsVisibility({ ...DEFAULT_COLUMN_VISIBILITY });
    }, []);

    const queryWithProject = { ...query, project_id: projectId ?? "" };

    const [jobs, totalJobs, loading, refetchJobs] = useJobs(queryWithProject);

    usePrefetchNextPages(loading, queryWithProject);

    const isPredictionPage = isPredictionsJobsPage(
        window?.location?.pathname ?? "",
        window?.location?.search ?? ""
    );

    const anyFilterApplied =
        isAnyJobFilterApplied(query) ||
        (query?.status && query?.status?.length > 0);

    const handleCacheInvalidation = () => {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.JobList]
        });
    };

    if (!loading && totalJobs === 0 && !anyFilterApplied) {
        return (
            <EmptyEntities
                title={title}
                currentPage={
                    isPredictionPage ? PageNames.Predictions : PageNames.Jobs
                }
            />
        );
    }

    return (
        <>
            <Title
                data-testid={
                    isPredictionPage
                        ? "predictions-list-page-heading"
                        : "models-list-page-heading"
                }
                variant="primary-heading"
                className="leading-none mb-6 !font-bold text-3xl">
                {title}
            </Title>
            <JobsListFilter
                query={query}
                setQuery={setQuery}
                type={title}
                jobTypeOptions={jobTypeOptions}
                projectId={projectId}
                showCreateButton={Boolean(projectId)}
                customColumns={customColumns}
                onToggleColumn={handleToggleColumn}
                onResetColumns={handleResetColumns}
            />
            <Spacer height={16} />
            <JobsListTable
                query={query}
                setQuery={setQuery}
                refetchJobs={refetchJobs}
                totalNumJobs={totalJobs}
                jobs={jobs}
                loading={loading}
                jobTypeOptions={jobTypeOptions}
                onArchive={handleCacheInvalidation}
                onRecover={handleCacheInvalidation}
                projectId={projectId}
                customColumnsVisibility={customColumns}
            />
        </>
    );
};

export default JobsList;
