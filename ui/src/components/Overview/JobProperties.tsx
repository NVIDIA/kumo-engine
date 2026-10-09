// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// eslint-disable-next-line simple-import-sort/imports
import { useMemo } from "react";

import {
    AccountBox,
    AccountTree,
    AvgPace,
    AvgTime,
    Badge,
    BookSpark,
    cn,
    CopyToClipBoard,
    GridCol,
    GridRow,
    OverviewProperties,
    Symbol,
    TextLink,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { GraphSnapshotResponse } from "gen/v0/types";

import DownloadPredictionIconButton from "components/Download/DownloadPredictionIconButton";

import {
    PredictionTableGeneration,
    UnlinkedBatchPredictionJobData
} from "types/batchPredictions";
import { JsConnector } from "types/connectors";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";
import { TrainTableGeneration } from "types/training_job";

import {
    getProjectPath,
    jobStatus,
    JobStatusType,
    warehouseSource
} from "utils/Constants";
import { bytesToSize, parseNumbers } from "utils/Helpers/numberHelpers";
import { getKeys } from "utils/Helpers/objectHelpers";
import {
    capitalizeFirstLetter,
    checkDash,
    getIdHash,
    toPascalCaseWithSpace
} from "utils/Helpers/stringHelpers";
import { getColorVariant } from "utils/jobs";
import {
    displayPathForObjectStoreConnector,
    isObjectStoreConnectorConfig
} from "utils/objectStoreTablePath";
import {
    formatToShortLocalDateString,
    formatToLongLocalDateString,
    isValidTimestamp
} from "utils/timeConverter";
import { JSTable } from "types/tables";
import { isValid } from "utils/Helpers/typeGuards";
import {
    ConnectorItemsSection,
    PropertyHoverCard
} from "components/Tables/Overview/TableOverview";
import { useCustomNavigation } from "utils/navigator";

export const usePredictionJobsProperties = (
    info: UnlinkedBatchPredictionJobData | null,
    status: JobStatusType,
    runTime: string,
    source: string
) => {
    const properties = useMemo(() => {
        if (!info) {
            return [];
        }

        const tmpList: OverviewProperties[] = [];
        tmpList.push({
            name: "Status",
            value: (
                <Badge
                    size="md"
                    variant={getColorVariant(status)}
                    id={"header-chip-id"}
                    data-testid={"header-status-chip-id"}>
                    {status === "CANCELLED"
                        ? "Canceled"
                        : capitalizeFirstLetter(status?.toLowerCase())}
                </Badge>
            )
        });
        tmpList.push({
            name: "Created",
            value: (
                <Tooltip id={"Created-tooltip"}>
                    <TooltipTrigger>
                        <span>{info?.dateCreated}</span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        side={"bottom"}
                        sideOffset={5}>
                        {`${formatToLongLocalDateString(info?.created)}`}
                    </TooltipContent>
                </Tooltip>
            )
        });
        if (info.user?.trim())
            tmpList.push({
                icon: <AccountBox className="size-4.5 fill-black" />,
                name: "User",
                value: info.user
            });
        if (status === jobStatus.Done)
            tmpList.push({
                icon: <AvgPace className="size-4.5 fill-black" />,
                name: "Total runtime",
                value: runTime
            });

        tmpList.push({
            icon: <BookSpark className="size-4.5 fill-black" />,
            name: "Task type",
            value:
                info?.taskType !== null ? (
                    toPascalCaseWithSpace(info?.taskType)
                ) : (
                    <Symbol type="emdash" />
                )
        });
        tmpList.push({
            icon: <AccountTree className="size-4.5 fill-black" />,
            name: "Source",
            value: source ? (
                <>
                    {source} {info?.kumoVersion && ` (${info?.kumoVersion})`}
                </>
            ) : (
                <Symbol type="emdash" />
            )
        });
        return tmpList;
    }, [info, runTime, source, status]);
    return properties;
};

export const usePredictionJobSettingProperties = (
    info: UnlinkedBatchPredictionJobData | null,
    anchorTime?: string | null,
    s3PresignedUrl?: string,
    urlFetchError?: boolean,
    downloadSupported?: boolean
) => {
    const outputItem = info?.outputs?.find(
        (item) =>
            item.artifact_type === "PREDICTIONS" ||
            item.artifact_type === "EMBEDDINGS"
    );

    const storageType = outputItem?.storage_type;
    const connectorId =
        outputItem && "connector_id" in outputItem
            ? outputItem.connector_id
            : undefined;

    const properties = useMemo(() => {
        if (!info) {
            return [];
        }

        const tmpList: OverviewProperties[] = [];

        tmpList.push({
            name: "Destination",
            value:
                !storageType && downloadSupported ? (
                    <div className="flex items-center gap-2">
                        <span>Local</span>
                        <DownloadPredictionIconButton
                            bpStatus={info.execution?.status}
                            jobCreatedTime={info.dateCreated}
                            s3Url={s3PresignedUrl || ""}
                            urlFetchError={urlFetchError || false}
                        />
                    </div>
                ) : storageType ? (
                    storageType
                ) : (
                    "Local"
                )
        });
        if (isValidTimestamp(anchorTime))
            tmpList.push({
                name: "Anchor time",
                value: anchorTime
            });
        if (info.numParallelWorkersLimit)
            tmpList.push({
                name: "Parallel workers",
                value: info.numParallelWorkersLimit
            });

        tmpList.push({
            name: "Explanations",
            value: info.explanations ? "On" : "Off"
        });
        if (connectorId)
            tmpList.push({
                name: "Connector",
                value: connectorId
            });

        if (info.threshold) {
            tmpList.push({
                name: "Threshold",
                value: info.threshold
            });
        }
        return tmpList;
    }, [
        anchorTime,
        connectorId,
        info,
        storageType,
        s3PresignedUrl,
        urlFetchError,
        downloadSupported
    ]);
    return properties;
};

export const useTrainingJobProperties = ({
    queryInfo,
    jobId,
    status,
    source
}: {
    queryInfo: PredictiveQueryInfo | null;
    jobId: string | null;
    status?: JobStatusType;
    source: string;
}) => {
    const properties = useMemo(() => {
        if (!queryInfo) {
            return [];
        }
        const tmpList: OverviewProperties[] = [];

        tmpList.push({
            name: "Status",
            value: (
                <Badge
                    size="md"
                    variant={getColorVariant(status)}
                    id={"header-chip-id"}
                    data-testid={"header-status-chip-id"}>
                    {status === "CANCELLED"
                        ? "Canceled"
                        : capitalizeFirstLetter(status?.toLowerCase())}
                </Badge>
            )
        });

        if (jobId) {
            tmpList.push({
                name: "Training job ID",
                value: (
                    <Tooltip id={"id-tooltip"}>
                        <TooltipTrigger>
                            <span>{getIdHash(jobId)}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {jobId}
                        </TooltipContent>
                    </Tooltip>
                ),
                copyText: jobId
            });
        }

        tmpList.push({
            name: "Created",
            value: (
                <Tooltip id={"Created-tooltip"}>
                    <TooltipTrigger>
                        <span>{queryInfo?.dateCreated as string}</span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        side={"bottom"}
                        sideOffset={5}>
                        {`${formatToLongLocalDateString(queryInfo?.created)}`}
                    </TooltipContent>
                </Tooltip>
            )
        });
        if (queryInfo.user?.trim())
            tmpList.push({
                icon: <AccountBox className="size-4.5 fill-black" />,
                name: "User",
                value: queryInfo.user
            });
        if (queryInfo.trainingStatus === jobStatus.Done)
            tmpList.push({
                icon: <AvgPace className="size-4.5 fill-black" />,
                name: "Total runtime",
                value: queryInfo.estTimings?.value
            });
        tmpList.push({
            icon: <BookSpark className="size-4.5 fill-black" />,
            name: "Task type",
            value:
                queryInfo.taskType !== null ? (
                    toPascalCaseWithSpace(queryInfo.taskType)
                ) : (
                    <Symbol type="emdash" />
                )
        });
        tmpList.push({
            icon: <AccountTree className="size-4.5 fill-black" />,
            name: "Source",
            value: source ? (
                <>
                    {source}{" "}
                    {queryInfo?.kumoVersion && ` (${queryInfo?.kumoVersion})`}
                </>
            ) : (
                <Symbol type="emdash" />
            )
        });
        return tmpList;
    }, [jobId, queryInfo, source, status]);
    return properties;
};

export const usePredictionTableGenProperties = ({
    id,
    info,
    source,
    status,
    taskType,
    user,
    anchorTime,
    numEntities
}: {
    id?: string | null;
    info: PredictionTableGeneration | null;
    source: string;
    status: JobStatusType;
    taskType: TaskType;
    user: string | null;
    anchorTime?: string | null;
    numEntities: string | null;
}) => {
    const properties = useMemo(() => {
        if (!info) {
            return [];
        }

        const tmpList: OverviewProperties[] = [];

        tmpList.push({
            name: "Status",
            value: (
                <Badge
                    size="md"
                    variant={getColorVariant(status)}
                    id={"header-chip-id"}
                    data-testid={"header-status-chip-id"}>
                    {status === "CANCELLED"
                        ? "Canceled"
                        : capitalizeFirstLetter(status?.toLowerCase())}
                </Badge>
            )
        });

        if (id) {
            tmpList.push({
                name: "Prediction table ID",
                value: (
                    <Tooltip id={"id-tooltip"}>
                        <TooltipTrigger>
                            <span>{getIdHash(id)}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {id}
                        </TooltipContent>
                    </Tooltip>
                ),
                copyText: id
            });
        }

        tmpList.push({
            name: "Created",
            value: (
                <Tooltip id={"Created-tooltip"}>
                    <TooltipTrigger>
                        <span>{info?.dateCreated}</span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        side={"bottom"}
                        sideOffset={5}>
                        {`${formatToLongLocalDateString(info?.createdAt)}`}
                    </TooltipContent>
                </Tooltip>
            )
        });

        if (user?.trim())
            tmpList.push({
                icon: <AccountBox className="size-4.5 fill-black" />,
                name: "User",
                value: user
            });
        if (status === jobStatus.Done)
            tmpList.push({
                icon: <AvgPace className="size-4.5 fill-black" />,
                name: "Total runtime",
                value: info?.runTime
            });
        if (anchorTime) {
            tmpList.push({
                icon: <AvgTime className="size-4.5 fill-black" />,
                name: "Prediction anchor time",
                value: anchorTime
            });
        }
        tmpList.push({
            icon: <BookSpark className="size-4.5 fill-black" />,
            name: "Task type",
            value:
                taskType !== null ? (
                    toPascalCaseWithSpace(taskType)
                ) : (
                    <Symbol type="emdash" />
                )
        });
        tmpList.push({
            icon: <AccountTree className="size-4.5 fill-black" />,
            name: "Source",
            value: source ? <>{source} </> : <Symbol type="emdash" />
        });
        if (numEntities !== null && numEntities !== undefined)
            tmpList.push({
                name: "Total entities",
                value: (
                    <div className="flex w-full justify-center">
                        {numEntities}
                    </div>
                )
            });
        return tmpList;
    }, [anchorTime, id, info, numEntities, source, status, taskType, user]);

    return properties;
};

export const useTrainingTableGenProperties = ({
    info,
    jobId,
    status,
    user,
    taskType,
    source
}: {
    info: TrainTableGeneration | null;
    jobId: string | null;
    status: JobStatusType;
    user: string | null;
    taskType: TaskType;
    source: string;
}) => {
    const properties = useMemo(() => {
        if (!info) {
            return [];
        }
        const tmpList: OverviewProperties[] = [];

        tmpList.push({
            name: "Status",
            value: (
                <Badge
                    size="md"
                    variant={getColorVariant(status)}
                    id={"header-chip-id"}
                    data-testid={"header-status-chip-id"}>
                    {status === "CANCELLED"
                        ? "Canceled"
                        : capitalizeFirstLetter(status?.toLowerCase())}
                </Badge>
            )
        });

        if (jobId)
            tmpList.push({
                name: "Training table ID",
                value: (
                    <Tooltip id={"id-tooltip"}>
                        <TooltipTrigger>
                            <span>{getIdHash(jobId)}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {jobId}
                        </TooltipContent>
                    </Tooltip>
                ),
                copyText: jobId
            });

        tmpList.push({
            name: "Created",
            value: `${info?.dateCreated}`,
            tooltip: `${formatToLongLocalDateString(info?.created)}`
        });
        if (user?.trim())
            tmpList.push({
                icon: <AccountBox className="size-4.5 fill-black" />,
                name: "User",
                value: user
            });
        if (status === jobStatus.Done)
            tmpList.push({
                icon: <AvgPace className="size-4.5 fill-black" />,
                name: "Total runtime",
                value: info?.runTime
            });
        tmpList.push({
            icon: <BookSpark className="size-4.5 fill-black" />,
            name: "Task type",
            value:
                taskType !== null ? (
                    toPascalCaseWithSpace(taskType)
                ) : (
                    <Symbol type="emdash" />
                )
        });
        tmpList.push({
            icon: <AccountTree className="size-4.5 fill-black" />,
            name: "Source",
            value: source ? (
                <>
                    {source}{" "}
                    {/* {queryInfo?.kumoVersion && ` (${queryInfo?.kumoVersion})`} */}
                </>
            ) : (
                <Symbol type="emdash" />
            )
        });
        return tmpList;
    }, [info, jobId, source, status, taskType, user]);
    return properties;
};

export const useGraphSnapshotProperties = ({
    graphSnapshot,
    id,
    status,
    startTime,
    isLoading,
    source,
    version,
    totalRows,
    totalSizeBytes
}: {
    graphSnapshot: GraphSnapshotResponse | null;
    id: string;
    status: JobStatusType;
    startTime?: string | null;
    isLoading?: boolean;
    source?: string;
    version?: string | null;
    totalRows: number | null;
    totalSizeBytes: number | null;
}) => {
    const { noOfTables, noOfLinks } = useMemo(() => {
        let noOfTables = 0,
            noOfLinks = 0;
        try {
            noOfTables =
                getKeys(graphSnapshot?.snapshot_resource?.table_ids)?.length ??
                0;
            noOfLinks = graphSnapshot?.definition?.col_groups?.length ?? 0;
        } catch (e) {
            console.warn(e);
        }
        return { noOfTables, noOfLinks };
    }, [graphSnapshot]);

    const properties = useMemo(() => {
        if (!graphSnapshot) {
            return [];
        }
        const tmpList: OverviewProperties[] = [];

        tmpList.push({
            name: "Status",
            value: (
                <Badge
                    size="md"
                    variant={getColorVariant(status)}
                    id={"header-chip-id"}
                    data-testid={"header-status-chip-id"}>
                    {status === "CANCELLED"
                        ? "Canceled"
                        : capitalizeFirstLetter(status?.toLowerCase())}
                </Badge>
            )
        });

        if (id)
            tmpList.push({
                name: "Snapshot ID",
                value: (
                    <Tooltip id={"GraphSnapshotId-tooltip"}>
                        <TooltipTrigger>
                            <span>{getIdHash(id)}</span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {id}
                        </TooltipContent>
                    </Tooltip>
                ),
                copyText: id
            });

        if (startTime)
            tmpList.push({
                name: "Snapshotted",
                value: (
                    <Tooltip id={"Snapshotted-at-tooltip"}>
                        <TooltipTrigger>
                            <span>
                                {checkDash(
                                    formatToShortLocalDateString(
                                        startTime as string
                                    )
                                )}
                            </span>
                        </TooltipTrigger>
                        <TooltipContent
                            arrow={false}
                            side={"bottom"}
                            sideOffset={5}>
                            {`${formatToLongLocalDateString(startTime)}`}
                        </TooltipContent>
                    </Tooltip>
                )
            });

        const graphName = graphSnapshot?.snapshot_resource?.graph_name;
        if (graphName) {
            tmpList.push({
                name: "Graph",
                value: graphName
            });
        }

        if (graphSnapshot && !isLoading) {
            tmpList.push({
                name: "Configs",
                value:
                    noOfTables && noOfLinks ? (
                        `${noOfTables} Table${noOfTables > 1 ? "s" : ""} x ${noOfLinks} Link${noOfLinks > 1 ? "s" : ""}`
                    ) : (
                        <Symbol type="emdash" />
                    )
            });
        }

        if (graphSnapshot && !isLoading) {
            tmpList.push({
                name: "Graph size",
                value:
                    totalRows !== null || totalSizeBytes !== null ? (
                        <>
                            {totalRows !== null
                                ? `${parseNumbers(totalRows, { millionSystem: true })} Rows`
                                : ""}
                            {totalRows !== null && totalSizeBytes !== null ? (
                                <Symbol
                                    type="dot"
                                    className="inline-block p-1"
                                />
                            ) : (
                                ""
                            )}
                            {totalSizeBytes !== null
                                ? bytesToSize(totalSizeBytes)
                                : ""}
                        </>
                    ) : (
                        <Symbol type="emdash" />
                    )
            });
        }

        tmpList.push({
            icon: <AccountTree className="size-4.5 fill-black" />,
            name: "Source",
            value: source ? (
                <>
                    {source}
                    {version && ` (${version})`}
                </>
            ) : (
                <Symbol type="emdash" />
            )
        });
        return tmpList;
    }, [
        graphSnapshot,
        id,
        isLoading,
        noOfLinks,
        noOfTables,
        source,
        startTime,
        status,
        totalRows,
        totalSizeBytes,
        version
    ]);

    return properties;
};

export const useConnectorProperties = ({
    info
}: {
    info?: JsConnector | null;
    isProjectView?: boolean;
}) => {
    let tmpList: OverviewProperties[] = [];

    const configs = info?.config;
    if (!configs) {
        return [];
    }
    if (isObjectStoreConnectorConfig(configs)) {
        tmpList = [
            {
                name: "Path",
                value: displayPathForObjectStoreConnector(configs)
            }
        ];
    }

    if (configs.type === warehouseSource.SnowFlake) {
        tmpList = [
            {
                value: configs.database,
                name: "Database"
            },
            {
                value: configs.warehouse,
                name: "Warehouse"
            },
            {
                value: configs.schema_name,
                name: "Schema"
            }
        ];
    }

    if (configs.type === warehouseSource.BigQuery) {
        tmpList = [
            {
                value: configs.project_id,
                name: "Project"
            },
            {
                value: configs.dataset_id,
                name: "Dataset"
            }
        ];
    }

    if (configs.type === warehouseSource.Databricks) {
        tmpList = [];
        if (configs.cluster_id) {
            tmpList.push({ value: configs.cluster_id, name: "Cluster ID" });
        } else if (configs.warehouse_id) {
            tmpList.push({ value: configs.warehouse_id, name: "Warehouse" });
        }
        tmpList.push(
            {
                value: configs.catalog ?? "",
                name: "Catalog"
            },
            {
                value: configs.schema_name,
                name: "Schema"
            }
        );
    }

    tmpList.unshift({
        value: info?.dateCreated ?? "",
        name: "Created"
    });
    return tmpList;
};

// An object-store connector path is labelled "Path" in the project-scoped view and
// "Location" in the non-project view (see ConnectorItemsSection). Both carry the
// full S3 path, so both get the hover-to-reveal tooltip + copy-to-clipboard.
const isObjectStorePathLabel = (label?: string | null) =>
    label === "Path" || label === "Location";

export const useTableProperties = ({
    info,
    connectorConfig,
    projectId,
    connectorID,
    objectStorePath,
    isProjectUploadConnector = false
}: {
    info: JSTable | null;
    connectorConfig?: JsConnector;
    projectId: string | null;
    connectorID: string | null;
    objectStorePath?: string;
    isProjectUploadConnector?: boolean;
}) => {
    const navigate = useCustomNavigation();
    const overviewProperties = useMemo<OverviewProperties[]>(() => {
        const props: OverviewProperties[] = [];
        if (isValid(info?.created)) {
            props.push({
                name: "Created",
                value: info?.created as string,
                tooltip:
                    !info?.created || info?.created === "-"
                        ? null
                        : info.dateCreated
            });
        }

        const connItems = ConnectorItemsSection(info, connectorConfig);
        const connTooltip = (
            <div>
                {connItems.map((item, index) => (
                    <GridRow key={`connector-item-${index}`}>
                        {item?.value ? (
                            <GridCol
                                xs={12}
                                sm={12}
                                className={cn(
                                    "flex items-center gap-2",
                                    index !== connItems.length - 1 && "mb-2"
                                )}>
                                <span className="text-title-secondary text-sm font-medium leading-4">
                                    {item?.label}
                                </span>
                                <label className="text-title-primary text-sm font-medium leading-4">
                                    {isObjectStorePathLabel(item?.label) &&
                                    objectStorePath ? (
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
                                {isObjectStorePathLabel(item?.label) &&
                                    objectStorePath && (
                                        <CopyToClipBoard
                                            copiedText={objectStorePath}
                                            iconSize={16}
                                            tooltipText={"Copied"}
                                            className="align-middle"
                                        />
                                    )}
                            </GridCol>
                        ) : null}
                    </GridRow>
                ))}
            </div>
        );
        const id =
            info?.connectorId === "" &&
            info?.source_table?.data_source_type === "S3"
                ? connectorID
                : info?.connectorId;
        const name =
            info?.connectorId === "" &&
            info?.source_table?.data_source_type === "S3"
                ? connectorID
                : connectorConfig?.name ||
                  info?.connectorId ||
                  connectorConfig?.description; // Description is the fallback display name.
        props.push({
            name: "Connector",
            value: isProjectUploadConnector ? (
                <span data-testid="table-connector">Local Upload</span>
            ) : (
                <PropertyHoverCard
                    testId="table-connector"
                    trigger={
                        <TextLink
                            className="underline"
                            id={`connecter-${id}`}
                            pageLink={getProjectPath(projectId).Connectors + id}
                            onClick={(e) => {
                                e?.preventDefault();
                                navigate(
                                    getProjectPath(projectId).Connectors + id,
                                    undefined,
                                    e
                                );
                            }}
                            title={`${name?.substring(0, 20)}${
                                (name as string)?.length > 20 ? "..." : ""
                            }`}
                        />
                    }
                    content={connTooltip}
                />
            )
        });

        if (
            isValid(info?.num_rows) &&
            isValid(info?.num_cols) &&
            isValid(info?.size)
        ) {
            props.push({
                name: "Dimension",
                value: `${info?.num_cols} columns × ${info?.num_rows} rows · ${info?.size?.toString()}`
            });
        }
        if (isValid(info?.timeRange)) {
            props.push({
                name: "Time Range",
                value: checkDash(info?.timeRange as string)
            });
        }
        return props;
    }, [
        info,
        connectorConfig,
        projectId,
        connectorID,
        objectStorePath,
        navigate,
        isProjectUploadConnector
    ]);
    return overviewProperties;
};
