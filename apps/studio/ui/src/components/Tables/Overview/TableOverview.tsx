// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";

import {
    Badge,
    CalendarBasic,
    Card,
    Catalog,
    CodeIcon,
    Database,
    Folder,
    FolderOpen,
    GridCol,
    GridRow,
    HashTag,
    HoverCard,
    HoverCardContent,
    HoverCardTrigger,
    Schema,
    SectionHeader,
    Skeleton,
    Warehouse
} from "@kumo-ai/kumo-ui-catalog";
import { OverviewDropdown } from "@kumo-ai/kumo-ui-catalog";
import { PropertiesOverviewCard } from "@kumo-ai/kumo-ui-catalog";
import {
    BigQueryConnectorResourceConfig,
    DatabricksConnectorResourceConfig,
    SnowflakeConnectorResourceConfig
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import StatisticsTable from "components/@ui/TablePlus/StatisticsTable";
import SQLCodeEditor from "components/CodeEditor/SQLCodeEditor";
import { useTableProperties } from "components/Overview/JobProperties";
import { SDKCodeModal } from "components/SDKCodeModal/SDKCodeModal";

import { useProjectId } from "hooks/project/projects";

import { MenuItemType } from "types/common";
import { JsConnector } from "types/connectors";
import { SDKObjectType } from "types/constants";
import { ExtendedColStatsResponse } from "types/snapshots";
import { DataSourceType, JSTable, TableKind } from "types/tables";

import { ENABLE_CUSTOM_SQL_TABLES } from "utils/Constants";
import { getConnectorIDFromTableSource } from "utils/Converters/tablesConverter";
import { getFormattedObjectStorePath } from "utils/Helpers/stringHelpers";
import {
    displayPathForObjectStoreConnector,
    isObjectStoreConnectorConfig
} from "utils/objectStoreTablePath";

interface PropertyHoverCardProps {
    testId: string;
    trigger: React.ReactNode;
    content: React.ReactNode;
    width?: string;
}

export const PropertyHoverCard: React.FC<PropertyHoverCardProps> = ({
    testId,
    trigger,
    content,
    width = "fit-content"
}) => (
    <HoverCard>
        <HoverCardTrigger asChild>
            <span data-testid={testId} className="cursor-pointer">
                {trigger}
            </span>
        </HoverCardTrigger>
        {/* eslint-disable-next-line no-restricted-syntax */}
        <HoverCardContent arrow={false} style={{ width }}>
            {content}
        </HoverCardContent>
    </HoverCard>
);

export interface TableOverviewProps {
    actionItems: MenuItemType[];
    columnStats: ExtendedColStatsResponse | null;
    fetchingSamples: boolean;
    tableInfo: JSTable | null;
    loadingStatus: boolean;
    connectorConfig?: JsConnector;
    isProjectUploadConnector?: boolean;
}

const StatsCardWrapper = ({ children }: { children: React.ReactNode }) => (
    <Card className="p-4 rounded-lg border-none h-full mt-0.5">{children}</Card>
);

const TableOverview: React.FC<TableOverviewProps> = ({
    actionItems,
    columnStats,
    fetchingSamples,
    tableInfo,
    loadingStatus,
    connectorConfig,
    isProjectUploadConnector = false
}) => {
    const projectId = useProjectId();
    const [isModalOpen, setIsModalOpen] = useState(false);

    const connectorID =
        tableInfo?.connectorId ||
        getConnectorIDFromTableSource(tableInfo?.source_table ?? null) ||
        null;
    const objectStoreConfig = isObjectStoreConnectorConfig(
        connectorConfig?.config
    )
        ? connectorConfig.config
        : undefined;
    const objectStorePath = objectStoreConfig
        ? displayPathForObjectStoreConnector(objectStoreConfig)
        : "";

    const overviewProperties = useTableProperties({
        info: tableInfo,
        connectorConfig: connectorConfig,
        projectId: projectId,
        connectorID,
        objectStorePath,
        isProjectUploadConnector
    });

    // Gated behind the feature flag so the detail panel keeps its prior layout
    // until Custom SQL Tables ships. Recognize a SQL table by its CUSTOM_SQL
    // source as well as `table_kind`: the table GET response doesn't yet stamp
    // `table_kind`, but its `source_table` carries the defining SELECT, so keying
    // off the source keeps the read-only SQL surface working with today's
    // backend (physical tables have no CUSTOM_SQL source, so they're unaffected).
    const isSqlTable =
        ENABLE_CUSTOM_SQL_TABLES &&
        (tableInfo?.table_kind === TableKind.Sql ||
            (tableInfo?.source_table?.data_source_type as string) ===
                "CUSTOM_SQL");

    return (
        <>
            <GridRow className="flex justify-between items-center">
                <GridCol className="flex items-center gap-2">
                    <label>
                        {!loadingStatus ? (
                            <SectionHeader
                                title={tableInfo?.name_alias as string}
                                titleSize="xl"
                                titleLetterSpacing={0.1}
                            />
                        ) : (
                            <Skeleton className="w-60 h-7" />
                        )}
                    </label>
                    {!loadingStatus && isSqlTable && (
                        <Badge size="sm" variant="pink">
                            SQL
                        </Badge>
                    )}
                </GridCol>
                {loadingStatus || !projectId ? null : (
                    <GridCol>
                        <OverviewDropdown
                            id="table-overview-dropdown"
                            actionItems={[
                                ...actionItems,
                                {
                                    content: (
                                        <>
                                            <CodeIcon className="size-4" />{" "}
                                            {" SDK Code "}
                                        </>
                                    ),
                                    disabled: false,
                                    onSelect: () => setIsModalOpen(true),
                                    tooltipText: ""
                                }
                            ]}
                        />
                    </GridCol>
                )}
            </GridRow>

            {!loadingStatus ? (
                <PropertiesOverviewCard
                    properties={overviewProperties}
                    cardClassName={"mt-2 mb-2 p-4"}
                    hideTitle
                />
            ) : (
                <Skeleton className="h-7" />
            )}
            {!loadingStatus && isSqlTable && tableInfo?.viewSql && (
                <StatsCardWrapper>
                    <SectionHeader
                        title="SQL Definition"
                        titleSize="md"
                        description={
                            tableInfo?.updated
                                ? `Last updated ${tableInfo.updated}`
                                : "The read-only SELECT that defines this table."
                        }
                    />
                    <div className="relative mt-2">
                        <SQLCodeEditor
                            value={tableInfo.viewSql}
                            onChange={() => {}}
                            editable={false}
                            showCopyButton
                            lineNumbers={false}
                            height="auto"
                        />
                    </div>
                </StatsCardWrapper>
            )}
            <StatsCardWrapper>
                <SectionHeader
                    title="Column Statistics"
                    titleSize="md"
                    description="The following stats are computed from a sample of table data."
                />
                <StatisticsTable
                    loading={fetchingSamples || loadingStatus}
                    search={false}
                    columnStats={columnStats}
                    primaryKey={tableInfo?.pkey}
                    timeCol={tableInfo?.timeCol}
                    endTimeCol={tableInfo?.endTimeCol}
                    hiddenColumns={["statistics.counts.total"]}
                />
            </StatsCardWrapper>
            {isModalOpen && (
                <SDKCodeModal
                    id={
                        tableInfo?.id !== undefined
                            ? "NA.NA-" + tableInfo.id
                            : ""
                    }
                    open={isModalOpen}
                    toggleModel={() => setIsModalOpen(false)}
                    objectTypeToGenScript={SDKObjectType.Table}
                />
            )}
        </>
    );
};

export default WithErrorBoundary(TableOverview);

export const ConnectorItemsSection = (
    tableInfo: JSTable | null,
    connectorConfig?: JsConnector
) => {
    const objectStoreConfig = isObjectStoreConnectorConfig(
        connectorConfig?.config
    )
        ? connectorConfig.config
        : null;
    const connectorLocationOrg = objectStoreConfig
        ? displayPathForObjectStoreConnector(objectStoreConfig)
        : null;
    const connectorLocation: string | null = connectorLocationOrg
        ? getFormattedObjectStorePath(connectorLocationOrg)
        : null;

    const connectorCreated =
        !connectorConfig?.created || connectorConfig?.created === "-"
            ? null
            : connectorConfig?.created;
    const source = tableInfo?.source_table?.data_source_type;
    const connectorItems = [
        {
            label: "Created",
            value: connectorCreated,
            tooltip: null,
            icon: <CalendarBasic className="size-5" />
        },
        {
            label: "Path",
            value: connectorLocation,
            tooltip: connectorLocationOrg,
            icon: <FolderOpen className="size-5" />
        }
    ];
    if (source === DataSourceType.SnowFlake) {
        const config =
            connectorConfig?.config as SnowflakeConnectorResourceConfig;
        connectorItems.push({
            label: "Warehouse",
            value: config?.warehouse ?? null,
            tooltip: null,
            icon: <Warehouse className="size-5" />
        });
        connectorItems.push({
            label: "Schema Name",
            value: config?.schema_name ?? null,
            tooltip: null,
            icon: <Schema className="size-5" />
        });
        connectorItems.push({
            label: "Database",
            value: config?.database ?? null,
            tooltip: null,
            icon: <Database className="size-5" />
        });
    }
    if (source === DataSourceType.BigQuery) {
        const config =
            connectorConfig?.config as BigQueryConnectorResourceConfig;
        connectorItems.push({
            label: "Project",
            value: config?.project_id ?? null,
            tooltip: null,
            icon: <Folder className="size-5" />
        });
        connectorItems.push({
            label: "Dataset",
            value: config?.dataset_id ?? null,
            tooltip: null,
            icon: <Database className="size-5" />
        });
    }
    if (source === DataSourceType.Databricks) {
        const config =
            connectorConfig?.config as DatabricksConnectorResourceConfig;
        connectorItems.push({
            label: "Cluster Id",
            value: config?.cluster_id ?? null,
            tooltip: null,
            icon: <HashTag className="size-5" />
        });
        connectorItems.push({
            label: "Warehouse",
            value: config?.warehouse_id ?? null,
            tooltip: null,
            icon: <Warehouse className="size-5" />
        });
        connectorItems.push({
            label: "Schema",
            value: config?.schema_name ?? null,
            tooltip: null,
            icon: <Schema className="size-5" />
        });
        connectorItems.push({
            label: "Catalog",
            value: config?.catalog ?? null,
            tooltip: null,
            icon: <Catalog className="size-5" />
        });
    }
    if ((source as string) === "CUSTOM_SQL") {
        // A Custom SQL table's source_table carries no per-warehouse fields, so
        // the physical-source branches above don't fire. Its connector's
        // schema/catalog are only unqualified-reference defaults — the SELECT may
        // read anything the credential can access — so surfacing them would
        // misleadingly imply a scope limit; we intentionally omit them. The
        // Warehouse is the actual compute the query runs on, so we surface it,
        // keyed off the SQL source_table's server-owned `connector_type` (its
        // `data_source_type` is CUSTOM_SQL, so the physical switch can't classify
        // the underlying warehouse).
        const connectorType = (
            tableInfo?.source_table as { connector_type?: string } | undefined
        )?.connector_type?.toLowerCase();
        let warehouse: string | null = null;
        if (connectorType === "databricks") {
            warehouse =
                (connectorConfig?.config as DatabricksConnectorResourceConfig)
                    ?.warehouse_id ?? null;
        } else if (connectorType === "snowflake") {
            warehouse =
                (connectorConfig?.config as SnowflakeConnectorResourceConfig)
                    ?.warehouse ?? null;
        }
        connectorItems.push({
            label: "Warehouse",
            value: warehouse,
            tooltip: null,
            icon: <Warehouse className="size-5" />
        });
    }
    return connectorItems;
};
