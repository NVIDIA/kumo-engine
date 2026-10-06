// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";
import Spacer from "react-spacer";

import {
    ArrowRange,
    Button,
    CalendarBasic,
    Cards,
    ChevronLeft,
    CloseSmall,
    cn,
    CopyToClipBoard,
    Cube,
    Description,
    EditCalendar,
    GridCol,
    GridRow,
    HoverCard,
    HoverCardContent,
    HoverCardTrigger,
    Label,
    Power,
    PropertyCard,
    Skeleton,
    Symbol,
    TextLink,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    useIsMounted
} from "@kumo-ai/kumo-ui-catalog";
import { S3ConnectorResourceConfig, S3SourceTable } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import StatisticsTable from "components/@ui/TablePlus/StatisticsTable";
import { ConnectorItemsSection } from "components/Tables/Overview/TableOverview";

import { useConnectorDetails } from "hooks/connector/connector";
import { useProjectId } from "hooks/project/projects";
import {
    useGetSnapshotsColumnStatsInfo,
    useGetSnapshotsTableInfo
} from "hooks/snapshot/tables";
import { useColumnStatsInfo, useGetTableInfo } from "hooks/tables/tables";
import { useValidationState } from "hooks/validation/validation";

import { CustomError, PropertyItem } from "types/common";
import {
    ExtendedColStatsResponse,
    ExtendedColumnSummary
} from "types/snapshots";
import { ColumnArgs, DataSourceType, JSTable } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { getProjectPath } from "utils/Constants";
import { getSourceNameFromTableSource } from "utils/Converters/tablesConverter";
import { capitalizeFirstLetter, checkDash } from "utils/Helpers/stringHelpers";
import { isValid, tableIdIsSnapshot } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import {
    displayPathForS3Connector,
    sourcePathFromS3Source
} from "utils/objectStoreTablePath";

// Helper to truncate text with ellipsis
const truncateText = (
    text: string | undefined | null,
    maxLength = 20
): string => {
    if (!text) return "";
    return text.length > maxLength
        ? `${text.substring(0, maxLength)}...`
        : text;
};

// Helper to create PropertyItem with HoverCard pattern
interface PropertyHoverCardOptions {
    icon: React.ReactNode;
    triggerContent: React.ReactNode;
    hoverContent: React.ReactNode;
    width?: string;
}

const createPropertyHoverCard = ({
    icon,
    triggerContent,
    hoverContent,
    width = "fit-content"
}: PropertyHoverCardOptions): {
    icon: React.ReactNode;
    label: React.ReactNode;
} => ({
    icon,
    label: (
        <HoverCard>
            <HoverCardTrigger className="cursor-pointer">
                {triggerContent}
            </HoverCardTrigger>
            <HoverCardContent arrow={false}>
                {/* eslint-disable-next-line no-restricted-syntax */}
                <div style={{ width }}>{hoverContent}</div>
            </HoverCardContent>
        </HoverCard>
    )
});

/**
 * View that shows the columns of a table
 *
 * @param tables List of tables
 * @param encoders Map of table columns to corresponding encoders
 * @param selectedTable ID of the table that's currently selected
 * @param setSelectedTable Function to set a new table that should be selected
 * @param width Chart width
 * @returns ```<TableView />```
 */
export interface TableViewProps {
    tables: JSTable[];
    selectedTable: string;
    navigateToGraph: () => void;
    open?: boolean;
    tableIds?: { [key: string]: string };
}

function TableViewContent({
    tables,
    selectedTable,
    navigateToGraph,
    tableIds = {}
}: TableViewProps) {
    const isMounted = useIsMounted();
    const navigate = useCustomNavigation();
    const projectId = useProjectId();
    const tableData = tables?.filter(({ id }) => id === selectedTable)?.[0];
    const tableId = tableData?.id;
    const tableName = tableData?.name_alias;

    // Id includes `@` means it's GraphSnapshot else check `tableIds` for id.
    // use `id` if `tableIds` doesn't have any id
    const newId = tableId?.includes("@")
        ? tableId
        : (tableIds[tableId] ?? tableId);
    const isSnapshotId = tableIdIsSnapshot(newId);
    const hasValidTableId = isValid(newId);

    const [columnStats, setColumnStats] = useState<ExtendedColStatsResponse>(
        {} as ExtendedColStatsResponse
    );
    const [fetchingStats, setFetchingStats] = useState<boolean>(false);

    const [loading, setLoading] = useState(true);
    const [, , handleError] = useValidationState(setLoading);

    const [selectedTableInfo, setSelectedTableInfo] = useState<JSTable>(
        {} as JSTable
    );

    const propertyList: PropertyItem[] = [];

    const { selectedTableInfo: tableInfo, loading: tableLoading } =
        useGetTableInfo(
            newId as string,
            handleError,
            hasValidTableId && !isSnapshotId
        );

    const { snapshotTableInfo, loading: snapshotLoading } =
        useGetSnapshotsTableInfo(
            newId as string,
            handleError,
            hasValidTableId && isSnapshotId
        );

    useEffect(() => {
        const table = isSnapshotId ? snapshotTableInfo : tableInfo;
        const loadingStatus =
            isSnapshotId === true ? snapshotLoading : tableLoading;
        if (!table && loadingStatus) {
            setLoading(true);
        }
        if (table && !loadingStatus) {
            if (loading) setLoading(false);
            setSelectedTableInfo(table);
        }
    }, [
        isSnapshotId,
        loading,
        snapshotLoading,
        snapshotTableInfo,
        tableInfo,
        tableLoading
    ]);

    const { connectorResponse: connectorConfig } = useConnectorDetails(
        selectedTableInfo?.connectorId as string,
        handleError
    );

    const timestampItems = [
        {
            label: "Last edited",
            value:
                !selectedTableInfo?.updated ||
                selectedTableInfo?.updated === "-"
                    ? null
                    : selectedTableInfo.updated,
            type: "lastEdited"
        },
        {
            label: "Created",
            value:
                !selectedTableInfo?.dateCreated ||
                selectedTableInfo?.dateCreated === "-"
                    ? null
                    : selectedTableInfo.dateCreated,
            type: "created"
        }
    ];

    const timestampComponent = (
        <div>
            {timestampItems.map((item, index) => {
                let IconComponent;
                switch (item?.type) {
                    case "lastEdited":
                        IconComponent = (
                            <EditCalendar className="size-4 fill-white" />
                        );
                        break;
                    case "created":
                        IconComponent = (
                            <CalendarBasic className="size-4 fill-white" />
                        );
                        break;
                    default:
                        break;
                }

                return (
                    <GridRow key={index}>
                        {item?.value ? (
                            <GridCol
                                xs={12}
                                sm={12}
                                className={cn(
                                    "flex items-center",
                                    index !== timestampItems.length - 1 &&
                                        "mb-2"
                                )}>
                                {IconComponent}
                                <span className="ps-1">
                                    {item?.label}
                                    <label className="ps-1">
                                        {item?.value}
                                    </label>
                                </span>
                            </GridCol>
                        ) : null}
                    </GridRow>
                );
            })}
        </div>
    );

    let connectorID: string | null = selectedTableInfo?.connectorId;
    const connectorExists = selectedTableInfo?.connectorId !== "";
    const isS3 =
        connectorConfig?.source?.toUpperCase() === DataSourceType.S3 ||
        (selectedTableInfo as JSTable)?.source_table?.data_source_type ===
            DataSourceType.S3;

    const s3Config = connectorConfig?.config as
        | S3ConnectorResourceConfig
        | undefined;
    let s3Path =
        isS3 && s3Config
            ? displayPathForS3Connector({ ...s3Config, type: "s3" })
            : "";

    if (!connectorExists && isS3) {
        const table = selectedTableInfo as JSTable;
        connectorID = table?.s3Connector?.id as string;
        const s3Table = table.source_table as S3SourceTable;

        if (s3Table?.source) {
            s3Path = sourcePathFromS3Source(s3Table.source);
        }
    }
    const connectorPath =
        projectId && connectorID
            ? `${getProjectPath(projectId).Connectors}${connectorID}`
            : "";

    const connectorItems = [
        {
            label: "Source Table",
            value: getSourceNameFromTableSource(
                (selectedTableInfo as JSTable)?.source_table
            ),
            tooltip: null,
            icon: <Cards className="size-4" />
        },
        ...ConnectorItemsSection(selectedTableInfo, connectorConfig)
    ];

    const connectorTooltipComponent = (
        <div>
            <GridRow className="text-sm font-medium leading-4">
                <GridCol
                    xs={12}
                    sm={12}
                    className="text-title-secondary flex items-center">
                    <Power className="size-5" />
                    {"CONNECTOR"}
                </GridCol>
            </GridRow>
            <label className="ps-2 text-title-primary text-sm! font-medium leading-6!">
                {isS3 && !connectorExists ? (
                    connectorID
                ) : (
                    <TextLink
                        id={`connecter-${connectorID}`}
                        pageLink={connectorPath}
                        onClick={(e) => {
                            e?.preventDefault(); // To avoid entire page load
                            navigate(connectorPath, undefined, e);
                        }}
                        title={connectorID}
                    />
                )}
            </label>
            <Spacer height={8} />

            {connectorItems.map((item, index) => {
                return (
                    <GridRow key={index}>
                        {item?.value ? (
                            <GridCol
                                xs={12}
                                sm={12}
                                className={cn(
                                    "flex items-center",
                                    index !== connectorItems.length - 1 &&
                                        "mb-2"
                                )}>
                                {item.icon}
                                <span className="ps-1 text-sm font-medium leading-4 text-title-secondary!">
                                    <span className="text-title-secondary text-sm font-medium leading-4">
                                        {item?.label}
                                    </span>
                                    <label className="ps-1 pe-2 text-title-primary text-sm leading-4 font-medium">
                                        {item?.label === "Location" &&
                                        s3Path ? (
                                            <Tooltip id={"con-location"}>
                                                <TooltipTrigger>
                                                    <span>{item?.value}</span>
                                                </TooltipTrigger>
                                                <TooltipContent
                                                    arrow={false}
                                                    sideOffset={5}>
                                                    {item?.tooltip}
                                                </TooltipContent>
                                            </Tooltip>
                                        ) : (
                                            <>{item?.value}</>
                                        )}
                                    </label>
                                    {item?.label === "Location" && s3Path && (
                                        <CopyToClipBoard
                                            copiedText={s3Path as string}
                                            iconSize={16}
                                            tooltipText={"Copied"}
                                            className="align-middle"
                                        />
                                    )}
                                </span>
                            </GridCol>
                        ) : null}
                    </GridRow>
                );
            })}
        </div>
    );

    if (isValid(selectedTableInfo?.created)) {
        propertyList.push({
            icon: <CalendarBasic className="size-4.5" />,
            label: (
                <Tooltip>
                    <TooltipTrigger>
                        <span
                            className="ml-1 cursor-pointer"
                            data-testid="Created">
                            {selectedTableInfo?.created}
                        </span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} side="bottom" sideOffset={5}>
                        {timestampComponent}
                    </TooltipContent>
                </Tooltip>
            )
        });
    }

    if (
        isValid(selectedTableInfo?.num_rows) &&
        isValid(selectedTableInfo?.num_cols) &&
        isValid(selectedTableInfo?.size)
    ) {
        propertyList.push({
            icon: <Cube className="size-4.5" />,
            label: (
                <div className="flex items-center">
                    {selectedTableInfo.num_cols} Columns{" "}
                    {<CloseSmall className="size-4.5" />}
                    {selectedTableInfo.num_rows} Rows
                    {<Symbol type="dot" className="inline-block p-1" />}
                    {selectedTableInfo.size?.toString()}
                </div>
            )
        });
    }

    if (
        selectedTableInfo?.connectorId === "" &&
        (selectedTableInfo as JSTable)?.source_table?.data_source_type === "S3"
    ) {
        propertyList.push(
            createPropertyHoverCard({
                icon: <Power className="size-5" />,
                triggerContent: truncateText(
                    (selectedTableInfo as JSTable)?.s3Connector?.id
                ),
                hoverContent: connectorTooltipComponent
            })
        );
    } else if (isValid(connectorID)) {
        propertyList.push(
            createPropertyHoverCard({
                icon: <Power className="size-5" />,
                triggerContent: truncateText(selectedTableInfo?.connectorId),
                hoverContent: connectorTooltipComponent
            })
        );
    }

    if (isValid(selectedTableInfo?.timeRange)) {
        propertyList.push({
            icon: <ArrowRange className="size-4.5" />,
            label: checkDash(selectedTableInfo?.timeRange as string)
        });
    }

    const handleGetColStatsError = useCallback<HandleErrorFunc>(
        (err: CustomError, title?: string, isWarning?: boolean) => {
            const columns = tableData?.columns ? [...tableData.columns] : [];
            columns?.forEach((c) => {
                if (c?.stype) c.stype = capitalizeFirstLetter(c?.stype);
            });
            const data: ExtendedColStatsResponse = {
                approx: true,
                rows_used: 0,
                statistics_available: false,
                columns: columns as ExtendedColumnSummary[]
            };
            // To show column without stats
            setColumnStats(data);
            setFetchingStats(false);
            handleError(err, title, isWarning);
        },
        [handleError, tableData?.columns]
    );

    const { columnStats: tableColStats, loading: loadingTableStats } =
        useColumnStatsInfo(
            isSnapshotId === false && isValid(newId) ? newId : null,
            handleGetColStatsError,
            isMounted,
            isSnapshotId === false && isValid(newId)
        );
    const { columnStats: snapshotColStats, loading: loadingSnapshotStats } =
        useGetSnapshotsColumnStatsInfo(
            isSnapshotId === true && isValid(newId) ? newId : null,
            handleGetColStatsError,
            isMounted,
            isSnapshotId === true && isValid(newId)
        );

    const loadingStats = isSnapshotId
        ? loadingSnapshotStats
        : loadingTableStats;
    const colStats = isSnapshotId ? snapshotColStats : tableColStats;

    useEffect(() => {
        setFetchingStats(loadingStats);
        if (newId && loadingStats === false && colStats) {
            const data = { ...colStats };
            const filterDataIndex: number[] = [];
            data?.columns?.forEach((element, i) => {
                // Creating sample rows in column object level
                if (element?.sample_rows) {
                    const samples = element?.sample_rows as string[];
                    let nonEmpty: string | undefined | null = samples?.find(
                        (d) => d !== null
                    );
                    if (nonEmpty === undefined) {
                        //extra secure
                        nonEmpty = null;
                    }
                    element.sample_rows =
                        nonEmpty === null ? nonEmpty : samples;
                    element.nonEmpty = nonEmpty;
                } else if (!data.statistics_available) {
                    element.sample_rows = [];
                    element.isSampleAvailable = true;
                }
                // Table data might have been edited. Update the col info from
                // table data ( only the ones that we show) into colstats columns
                // if required.
                const col: ColumnArgs | undefined = tableData?.columns?.find(
                    (c) => c?.name === element?.name
                );
                if (!col) {
                    filterDataIndex.push(i);
                } else if (
                    col &&
                    col.stype?.toUpperCase() !== element?.stype?.toUpperCase()
                ) {
                    element.stype = capitalizeFirstLetter(col.stype);
                }
            });
            // Filter to show only selected columns
            data.columns = data.columns.filter(
                (_, index) => !filterDataIndex.includes(index)
            );

            setColumnStats(data as ExtendedColStatsResponse);
        }
    }, [
        tableColStats,
        newId,
        tableData?.columns,
        loadingStats,
        colStats,
        isSnapshotId
    ]);
    if (!open) {
        return null;
    }

    return (
        <div
            data-testid="graph-table-view-panel"
            className={cn("w-full p-4", isSnapshotId && "font-inter")}>
            <GridRow className="flex justify-between items-center">
                <GridCol>
                    <Label className="text-xl font-semibold">
                        {!loading ? (
                            tableName
                        ) : (
                            <Skeleton className="w-60 h-6" />
                        )}
                    </Label>
                </GridCol>
                <GridCol>
                    <Button
                        data-testid="graph-table-view-back"
                        id="back"
                        onClick={navigateToGraph}
                        variant="light-outline"
                        size="sm">
                        {" "}
                        <>
                            <ChevronLeft /> Back
                        </>
                    </Button>
                </GridCol>
            </GridRow>
            <Spacer height={8} />
            {!loading ? (
                <PropertyCard propertyList={propertyList} id={"Propery-card"} />
            ) : (
                <Skeleton className="w-200 h-8" />
            )}
            <Spacer height={16} />

            <Label
                data-testid="graph-table-view-column-statistics-heading"
                className="!mb-0 !text-base">
                Column Statistics
            </Label>
            <Description
                variant="secondary"
                size="sm"
                className="pt-1 pb-0 mb-3 leading-5">
                The following stats are computed from a sample of table data.
            </Description>
            <StatisticsTable
                loading={fetchingStats}
                search={false}
                columnStats={columnStats as ExtendedColStatsResponse}
                primaryKey={tableData?.pkey}
                timeCol={tableData?.timeCol}
                endTimeCol={tableData?.endTimeCol}
                hiddenColumns={["statistics.counts.total"]}
                isSnapShot={isSnapshotId}
            />
        </div>
    );
}

function TableView(props: TableViewProps) {
    if (props.open === false) {
        return null;
    }
    return <TableViewContent {...props} />;
}

export default WithErrorBoundary(TableView);
