// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState
} from "react";
import { useLocation, useParams } from "react-router";

import {
    Description,
    EmptyTable,
    ListView,
    NetworkNode,
    Power,
    SearchBar,
    Select,
    Spinner,
    SwapVert,
    Title,
    Window
} from "@kumo-ai/kumo-ui-catalog";
import type { ResourceTimestamps } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useProjectId } from "hooks/project/projects";
import useFluidCardHeight from "hooks/utils/useFluidCardHeight";

import { SelectOption } from "types/inputs";

import { getProjectPath } from "utils/Constants";
import { isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";

export interface ListEntity {
    id: string;
    name: string;
    dateCreated?: string;
    timestamps: ResourceTimestamps;
    source_table_name?: string;
    connector_id?: string;
}

export interface ListOverviewProps<T> {
    title: string;
    data?: T[];
    loadingStatus?: boolean;
    overviewComponent?: React.ReactNode | null;
    entityPresent: boolean;
    buttons: React.ReactNode[];
    selectedEntity: T | null;
    setSelectedEntity: (entity: T | null) => void;
    validationComponent?: React.ReactNode | null;
    onEntityView?: (isVisible: boolean) => void;
    isGraphsPage?: boolean;
    // Optional entity-specific filter rendered alongside search/sort (e.g. the
    // table list's Raw/SQL kind filter). Filtering the `data` stays the caller's
    // job so this component remains entity-agnostic.
    filterControl?: React.ReactNode;
}

const ListOverview = <T extends ListEntity>({
    title,
    data = [],
    loadingStatus,
    overviewComponent = null,
    entityPresent,
    buttons,
    selectedEntity,
    setSelectedEntity,
    validationComponent = null,
    onEntityView,
    isGraphsPage,
    filterControl
}: ListOverviewProps<T>) => {
    const selectedProject = useProjectId();
    const [searchStr, setSearchStr] = useState<string>("");
    const [selectedVisible, setSelectedVisible] = useState<boolean>(false);

    const [sortValue, setSortValue] = useState<string[]>(["date-desc"]);

    const { id } = useParams();
    const navigate = useCustomNavigation();
    const location = useLocation();
    const initialSelectionDoneRef = useRef<string | null>(null);

    useEffect(() => {
        if (id && data?.length > 0 && initialSelectionDoneRef.current !== id) {
            const entity = data?.find((c) => c.name === id || c.id === id);
            if (entity) {
                initialSelectionDoneRef.current = id;
                setSelectedEntity(entity);
                setSelectedVisible(true);
            }
        } else if (!id) {
            initialSelectionDoneRef.current = null;
        }
    }, [id, data, setSelectedEntity, setSelectedVisible]);

    const sortOptions: SelectOption[] = [
        { value: "name-asc", label: "Alphabetical (A–Z)" },
        { value: "name-desc", label: "Alphabetical (Z–A)" },
        { value: "date-asc", label: "Least recent" },
        { value: "date-desc", label: "Most recent" }
    ];

    const handleSortChange = useCallback((values: string[]) => {
        const selected = values[0];
        setSortValue([selected]);
    }, []);

    const filteredData: T[] = useMemo(() => {
        const searchFiltered = data?.filter((item) =>
            item?.name?.toLowerCase()?.includes(searchStr?.toLowerCase())
        );
        // Date sorting works on the raw ISO timestamp, never the formatted
        // display string: ISO parsing is engine-independent and unaffected
        // by the viewer's timezone.
        const createdAtEpoch = (item: T) =>
            new Date(item?.timestamps?.created_at as string).getTime();
        const sorted = !searchFiltered
            ? []
            : [...searchFiltered].sort((a, b) => {
                  switch (sortValue[0]) {
                      case "name-asc":
                          return a.name.localeCompare(b.name);
                      case "name-desc":
                          return b.name.localeCompare(a.name);
                      case "date-asc":
                          return createdAtEpoch(a) - createdAtEpoch(b);
                      case "date-desc":
                          return createdAtEpoch(b) - createdAtEpoch(a);
                      default:
                          return 0;
                  }
              });
        return sorted;
    }, [data, searchStr, sortValue]);

    const [selectString, depString] = useMemo(() => {
        if (title === "Connectors") return ["connector", "connectors"];
        if (title === "tables") return ["table", "connectors"];
        else if (title === "graphs") return ["graph", "tables"];
        else return [];
    }, [title]);
    const listOverviewTestId = selectString
        ? `${selectString}-list-page`
        : "list-overview-page";

    const handleCellClick = useCallback(
        (item: T) => {
            if (!selectedProject) return;
            const projectPaths = getProjectPath(selectedProject);
            const navRoute =
                title === "Connectors"
                    ? projectPaths.Connectors
                    : title === "tables"
                      ? projectPaths.Tables
                      : projectPaths.Graphs;
            setSelectedEntity(null);
            setSelectedVisible(false);
            if (selectedEntity?.id === item?.id) {
                if (id) navigate(navRoute, { state: location.state });
            } else if (isObjectNotEmpty(item)) {
                if (isGraphsPage) {
                    onEntityView?.(true);
                }
                if (selectedEntity?.id !== item?.id) {
                    setSelectedEntity(item);
                    setSelectedVisible(true);
                    if (isValid(item?.name)) {
                        navigate(navRoute + item?.name, {
                            state: location.state
                        });
                    }
                }
            }
        },
        [
            selectedProject,
            title,
            setSelectedEntity,
            selectedEntity?.id,
            id,
            navigate,
            location.state,
            isGraphsPage,
            onEntityView
        ]
    );

    // Measure the bottom of the header (title + search/sort row) and fill the
    // remaining viewport height so the list resizes with the screen resolution
    // instead of overflowing past the bottom of the viewport. bottomPadding is
    // the page's bottom padding (24px) plus the search row's bottom margin
    // (16px), which is excluded from the measured header's border box. Fall back
    // to a viewport-based estimate for the very first paint (before the header
    // is measured); the measured value takes over on the next tick.
    const { headerRef, cardHeight } = useFluidCardHeight<HTMLDivElement>({
        bottomPadding: 40
    });
    const pageHeight = cardHeight ?? window.innerHeight - 110;

    const iconClass = "w-6 h-6 fill-icon-secondary";
    const entity_icon =
        title === "Connectors" ? (
            <Power className={iconClass} />
        ) : title === "tables" ? (
            <Window className={iconClass} />
        ) : (
            <NetworkNode className={iconClass} />
        );

    const emptyTableComponent = (
        <div
            className="m-1 me-1 p-0 overflow-auto max-w-62.5"
            data-testid="side-view">
            {validationComponent}
            <EmptyTable
                id={"empty-table-component"}
                showLoading={loadingStatus}
                className="mb-1 border-none w-fit-content items-center shadow-none">
                <div
                    className="flex flex-col justify-center items-center"
                    // eslint-disable-next-line no-restricted-syntax -- runtime height
                    style={{
                        height: pageHeight - 100
                    }}>
                    <div className="flex flex-col items-center text-center">
                        {entity_icon}
                        <Description
                            size="sm"
                            variant="secondary"
                            className="!mb-0">
                            {`Any ${title} will be listed here.`}
                            {entityPresent &&
                                ` First register ${depString} to create ${title}.`}
                        </Description>
                    </div>
                </div>
            </EmptyTable>
        </div>
    );

    return (
        <div
            data-testid={listOverviewTestId}
            data-screenshot-ready={
                loadingStatus ? undefined : listOverviewTestId
            }>
            <div ref={headerRef}>
                <Title
                    data-testid={`${listOverviewTestId}-heading`}
                    variant="primary-heading"
                    className="mb-6 flex !font-bold text-3xl">
                    {capitalizeFirstLetter(title)}
                    <div className="ps-2 text-title-secondary font-normal">
                        {/* Spinner reflects the actual fetch, not emptiness: a
                            legitimately-empty view (e.g. a "Kind" filter that
                            matches no rows) must settle to its count, never hang
                            on a perpetual spinner. */}
                        {loadingStatus ? (
                            <Spinner size={28} />
                        ) : (
                            (data?.length ?? 0).toString()
                        )}
                    </div>
                </Title>
                <div className="flex flex-row items-center gap-2 my-0 mb-4">
                    <SearchBar
                        className="input-group-merge input-group-reverse flex w-74 h-9 py-2 px-4 items-center"
                        placeholder="Search"
                        value={searchStr}
                        variant="secondary"
                        onChange={(e: { target: { value: string } }) => {
                            const newValue = e?.target?.value || "";
                            setSearchStr(newValue);
                        }}
                    />
                    <Select
                        id="list-sort-select"
                        options={sortOptions}
                        sortOptions={false}
                        selectedValues={sortValue}
                        onValuesChange={handleSortChange}
                        search={false}
                        mode="single"
                        placeholder={
                            <span className="inline-flex items-center gap-1">
                                <SwapVert />
                                Sort
                            </span>
                        }
                        variant="secondary"
                        size="sm"
                        popoverTriggerClassName="flex w-60 h-9 py-2 px-4 items-center gap-2"
                        isInValid={false}
                        selectedOptionPrefix="Sort by:"
                    />
                    {filterControl}
                    <div className="ml-auto !cursor-pointer hover:rounded-sm hover:bg-transparent">
                        {buttons.map((button, i) => (
                            <React.Fragment key={"button" + i}>
                                {button}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            </div>

            <ListView
                id={id}
                searchFiltered={filteredData}
                loadingStatus={loadingStatus}
                overviewComponent={
                    <>
                        {validationComponent}
                        {overviewComponent}
                    </>
                }
                selectedEntity={selectedEntity}
                handleCellClick={handleCellClick}
                emptyTableComponent={emptyTableComponent}
                selectedVisible={selectedVisible}
                selectString={selectString}
                pageHeight={pageHeight}
            />
        </div>
    );
};

export default WithErrorBoundary(ListOverview);
