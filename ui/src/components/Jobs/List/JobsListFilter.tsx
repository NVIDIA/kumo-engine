// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";

import {
    Button,
    ChevronDown,
    ChevronUp,
    Cube,
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    Filter,
    SearchBar,
    Select,
    SoundSampler
} from "@kumo-ai/kumo-ui-catalog";
import type {
    JobType,
    ListJobsQuery,
    StatedbConfigCommonJobStatus as JobStatus
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useJobSearch, useJobsTagsFilter } from "hooks/jobs/list/jobList";

import {
    getProjectPath,
    jobStatus,
    ModelTitle,
    PredictionTitle
} from "utils/Constants";
import { getValues, isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { TAG_KEY_ARCHIVED } from "utils/jobs";
import { useCustomNavigation } from "utils/navigator";

interface CreateButtonProps {
    type: string;
    navigate: ReturnType<typeof useCustomNavigation>;
    projectId?: string | null;
}

const CreateButton = ({ type, navigate, projectId }: CreateButtonProps) => {
    const projectPaths = getProjectPath(projectId);
    if (type === PredictionTitle) {
        return (
            <Button
                id="create-prediction"
                onClick={() => navigate(projectPaths.NewPredictionJob)}>
                <SoundSampler className="size-4" /> Generate predictions
            </Button>
        );
    }
    if (type === ModelTitle) {
        return (
            <Button
                id="create-models"
                onClick={() => navigate(projectPaths.NewTrainingGraphSelect)}>
                <Cube className="size-4" /> New model
            </Button>
        );
    }
    return null;
};

export interface JobsListFilterProps {
    query: ListJobsQuery;
    setQuery: (newQuery: ListJobsQuery) => void;
    type: string;
    jobTypeOptions?: { value: JobType; label: string }[];
    showCreateButton?: boolean;
    projectId?: string | null;
    customColumns: { [key: string]: boolean };
    onToggleColumn: (col: string) => void;
    onResetColumns: () => void;
}

const JobsListFilter = ({
    query,
    setQuery,
    type,
    showCreateButton = true,
    projectId,
    customColumns,
    onToggleColumn,
    onResetColumns
}: JobsListFilterProps) => {
    const navigate = useCustomNavigation();
    // Update query when debounced value changes
    const { searchKey, setSearchValue } = useJobSearch({ query, setQuery });

    const { tags, tagsOptions, selectedTags } = useJobsTagsFilter({
        query,
        setQuery,
        type
    });

    const showsArchived = (query.tags_filter ?? []).includes(TAG_KEY_ARCHIVED);
    const [isStatusMenuOpen, setIsStatusMenuOpen] = useState(false);
    const [isColumnsMenuOpen, setIsColumnsMenuOpen] = useState(false);

    const handleResetFilters = useCallback(() => {
        setSearchValue("");
        setQuery({
            ...query,
            search_keyword: undefined,
            status: [],
            sort_by: undefined,
            sort_desc: undefined,
            tags_filter: [],
            page: 1
        });
        onResetColumns();
    }, [setSearchValue, setQuery, query, onResetColumns]);

    return (
        <div className="flex flex-row items-center gap-2">
            <SearchBar
                className="input-group-merge input-group-reverse flex w-74 h-9 py-2 px-4 items-center"
                placeholder="Search"
                value={searchKey ?? ""}
                variant="secondary"
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                    const newValue = e?.target?.value || "";
                    setSearchValue(newValue);
                }}
            />

            <DropdownMenu
                open={isStatusMenuOpen}
                onOpenChange={setIsStatusMenuOpen}>
                <DropdownMenuTrigger asChild>
                    <div className="inline-block">
                        <Button
                            variant="light-outline"
                            className="text-sm w-40 justify-between">
                            <span className="flex items-center gap-2">
                                Status
                                {query?.status && query?.status?.length > 0 && (
                                    <span className="flex items-center justify-center w-6 h-6 bg-surface-secondary rounded">
                                        {query?.status?.length}
                                    </span>
                                )}
                            </span>
                            {isStatusMenuOpen ? <ChevronUp /> : <ChevronDown />}
                        </Button>
                    </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-40">
                    {getValues(jobStatus)
                        .filter((status) => isValid(status))
                        .map((status) => {
                            const isSelected = (query.status ?? []).includes(
                                status
                            );
                            const label =
                                status === "CANCELLED"
                                    ? "Canceled"
                                    : capitalizeFirstLetter(
                                          status?.toLowerCase()
                                      );
                            return (
                                <DropdownMenuItem
                                    key={status}
                                    className="focus:bg-kumo-grey-50 hover:bg-kumo-grey-50"
                                    onClick={(e: React.MouseEvent) => {
                                        e.preventDefault();
                                        const currentStatus =
                                            query.status ?? [];
                                        const newStatus = isSelected
                                            ? currentStatus.filter(
                                                  (s) => s !== status
                                              )
                                            : [...currentStatus, status];

                                        setQuery({
                                            ...query,
                                            status: newStatus as JobStatus[],
                                            page: 1
                                        });
                                    }}>
                                    <span
                                        className={
                                            "flex items-center gap-2 " +
                                            (isSelected
                                                ? ""
                                                : "text-title-secondary")
                                        }>
                                        <input
                                            type="checkbox"
                                            checked={isSelected}
                                            readOnly
                                            style={{
                                                accentColor: isSelected
                                                    ? "#292D36"
                                                    : ""
                                            }}
                                            className={"cursor-pointer"}
                                        />
                                        {label}
                                    </span>
                                </DropdownMenuItem>
                            );
                        })}
                </DropdownMenuContent>
            </DropdownMenu>

            {isObjectNotEmpty(tags) && (
                <Select
                    popoverTriggerClassName="flex w-40 h-9 py-2 px-4 items-center gap-2"
                    placeholder={`Tags ${selectedTags.length > 0 ? selectedTags.length : ""}`}
                    variant="secondary"
                    size="sm"
                    search={true}
                    showSelectedCount={false}
                    multiModeText={
                        <>
                            Tags
                            {selectedTags.length > 0 && (
                                <span className="flex items-center justify-center w-6 h-6 bg-surface-secondary rounded">
                                    {selectedTags.length}
                                </span>
                            )}
                        </>
                    }
                    popoverClassName="w-125"
                    selectedValues={selectedTags}
                    options={tagsOptions}
                    onValuesChange={(newValue: string[]) => {
                        if (showsArchived) {
                            newValue.push(TAG_KEY_ARCHIVED);
                        }

                        setQuery({
                            ...query,
                            tags_filter: newValue,
                            page: 1
                        });
                    }}
                />
            )}

            <DropdownMenu
                open={isColumnsMenuOpen}
                onOpenChange={setIsColumnsMenuOpen}>
                <DropdownMenuTrigger asChild>
                    <div className="inline-block">
                        <Button variant="light-outline" className="text-sm">
                            <span className="flex items-center gap-2">
                                Columns
                                {(customColumns.task_type ||
                                    customColumns.source) && (
                                    <span className="flex items-center justify-center w-6 h-6 bg-surface-secondary rounded">
                                        {Number(customColumns.task_type) +
                                            Number(customColumns.source)}
                                    </span>
                                )}
                            </span>
                            {isColumnsMenuOpen ? (
                                <ChevronUp />
                            ) : (
                                <ChevronDown />
                            )}
                        </Button>
                    </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-40">
                    {[
                        { key: "task_type", label: "Task type" },
                        { key: "source", label: "Source" }
                    ].map(({ key, label }) => {
                        const isChecked = customColumns[key];
                        return (
                            <DropdownMenuItem
                                key={key}
                                className="focus:bg-kumo-grey-50 hover:bg-kumo-grey-50"
                                onClick={(e: React.MouseEvent) => {
                                    e.preventDefault();
                                    onToggleColumn(key);
                                }}>
                                <span
                                    className={
                                        "flex items-center gap-2 " +
                                        (isChecked
                                            ? ""
                                            : "text-title-secondary")
                                    }>
                                    <input
                                        type="checkbox"
                                        checked={isChecked}
                                        readOnly
                                        style={{
                                            accentColor: isChecked
                                                ? "#292D36"
                                                : ""
                                        }}
                                        className="cursor-pointer"
                                    />
                                    {label}
                                </span>
                            </DropdownMenuItem>
                        );
                    })}
                </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <div className="inline-block">
                        <Button variant="light-outline" className="text-sm">
                            <Filter />
                            <span className="flex items-center gap-2">
                                Filter
                                {showsArchived && (
                                    <span className="flex items-center justify-center w-6 h-6 bg-surface-secondary rounded">
                                        1
                                    </span>
                                )}
                            </span>
                        </Button>
                    </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-40">
                    <DropdownMenuItem
                        className="focus:bg-kumo-grey-50 hover:bg-kumo-grey-50"
                        onClick={(e: React.MouseEvent) => {
                            e.preventDefault();
                            const newTagsFilter = showsArchived
                                ? selectedTags
                                : [...selectedTags, TAG_KEY_ARCHIVED];

                            setQuery({
                                ...query,
                                tags_filter: newTagsFilter,
                                page: 1
                            });
                        }}>
                        <span className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showsArchived}
                                readOnly
                                className="cursor-pointer"
                                style={{
                                    accentColor: showsArchived ? "#292D36" : ""
                                }}
                            />
                            Show Archived
                        </span>
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
            <Button
                onClick={handleResetFilters}
                id="Reset-Job-Status"
                variant="light-outline"
                className="text-sm">
                Reset filters
            </Button>
            {showCreateButton && (
                <div className="ml-auto">
                    <CreateButton
                        type={type}
                        navigate={navigate}
                        projectId={projectId ?? query.project_id}
                    />
                </div>
            )}
        </div>
    );
};

export default WithErrorBoundary(JobsListFilter);
