// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import {
    ColumnDef,
    DataTable,
    EmptyTable,
    SearchBar,
    SortingState,
    Stack,
    useDebounce
} from "@kumo-ai/kumo-ui-catalog";
import { GenericJob, JobSortField, ListJobsQuery } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import EmptyEntities from "components/Empty/EmptyEntities";
import { TrainingJobSelectListColumns } from "components/NewJob/Prediction/Select/TrainingJobSelectListColumns";

import { useTrainingJobs } from "hooks/jobs/list/jobList";

import { PredictiveQueryInfo } from "types/predictiveQueries";

import { PageNames } from "utils/Constants";
import { getDataTableFillClasses } from "utils/Helpers/dataTableFill";
import { isValid } from "utils/Helpers/typeGuards";
import { getTrainingJobsSelectListQueryParams } from "utils/jobs";

export interface TrainingJobSelectListProps {
    setNext: React.Dispatch<
        React.SetStateAction<(GenericJob & PredictiveQueryInfo) | null>
    >;
    projectId?: string;
    onEmptyChange?: (isEmpty: boolean) => void;
}

const TrainingJobSelectList: React.FC<TrainingJobSelectListProps> = ({
    setNext,
    projectId,
    onEmptyChange
}) => {
    const [query, setQuery] = useState<ListJobsQuery>({
        ...getTrainingJobsSelectListQueryParams(projectId ?? "")
    });
    const [searchValue, setSearchValue] = useState("");
    const debouncedSearchValue = useDebounce(searchValue, 1000);
    const [jobs, totalNumJobs, loading] = useTrainingJobs(query);
    const [sorting, setSorting] = useState<SortingState>([]);

    const columns: ColumnDef<GenericJob & PredictiveQueryInfo>[] =
        TrainingJobSelectListColumns();

    const handleSort = useCallback(
        (sortBy: SortingState) => {
            const newSortBy = sortBy[0]?.id;
            const newSortDesc = sortBy[0]?.desc;

            // Only update if the sort parameters actually changed
            if (
                query.sort_by !== newSortBy ||
                query.sort_desc !== newSortDesc
            ) {
                const updatedQuery: ListJobsQuery = {
                    ...query,
                    sort_by: newSortBy as JobSortField,
                    sort_desc: newSortDesc,
                    page: 0 // Reset to first page when sort changes
                };
                setQuery(updatedQuery);
            }
        },
        [query, setQuery]
    );

    useEffect(() => {
        handleSort(sorting);
    }, [handleSort, sorting]);

    useEffect(() => {
        if (query.search_keyword?.trim() !== debouncedSearchValue?.trim()) {
            setQuery({
                ...query,
                search_keyword: debouncedSearchValue,
                page: 0
            });
        }
    }, [debouncedSearchValue, query]);

    const setCurrentPage = useCallback(
        (newPage: number) => {
            setQuery((prevQuery) => ({
                ...prevQuery,
                page: newPage
            }));
        },
        [setQuery]
    );

    const handleTrainingJobSelect = (rowData: object) => {
        const trainingJob = rowData as GenericJob & PredictiveQueryInfo;
        setNext(trainingJob);
    };

    const emptyTableComponent = () => (
        <EmptyTable
            id={"empty-jobs-table"}
            showLoading={loading}
            enableCard={false}>
            <Stack />
            <p className="text-center text-title-secondary">
                {"No jobs found"}
            </p>
        </EmptyTable>
    );

    const isEmpty =
        !loading &&
        jobs?.length === 0 &&
        isValid(debouncedSearchValue) === false &&
        isValid(query.search_keyword) === false;

    useEffect(() => {
        onEmptyChange?.(isEmpty);
    }, [isEmpty, onEmptyChange]);

    if (isEmpty) {
        return <EmptyEntities currentPage={PageNames.Predictions} />;
    }

    return (
        <div
            data-testid="new-prediction-training-job-select-list"
            className="flex min-h-0 flex-1 flex-col">
            <div className="flex flex-row items-center gap-2">
                <SearchBar
                    className="input-group-merge input-group-reverse flex w-74 h-9 py-2 px-4 items-center"
                    placeholder="Search"
                    value={searchValue}
                    variant="secondary"
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        const newValue = e?.target?.value || "";
                        setSearchValue(newValue);
                    }}
                />
            </div>
            <div className="h-4" />
            {loading === false && jobs?.length === 0 ? (
                emptyTableComponent()
            ) : (
                // Fill the remaining viewport height with model rows when the
                // page is full; a short/last page stays content-sized.
                <div
                    className={getDataTableFillClasses(
                        jobs.length >= query.limit!
                    )}>
                    <DataTable
                        id="allJobs"
                        columns={columns}
                        data={jobs}
                        isLoading={loading}
                        pageCount={Math.ceil(
                            totalNumJobs / (query?.limit ?? 1)
                        )}
                        defaultPageSize={query.limit}
                        enableGlobalFilter={false}
                        onRowClick={handleTrainingJobSelect}
                        manualPagination={true}
                        manualPageIndex={query.page || 0}
                        handleManualPageChange={setCurrentPage}
                        manualSorting={true}
                        manualSortingState={sorting}
                        setManualSortingState={setSorting}
                        manualFiltering={true}
                        showTopRoundedBorder={true}
                    />
                </div>
            )}
        </div>
    );
};
export default WithErrorBoundary(TrainingJobSelectList);
