// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import {
    Card,
    CardContent,
    PropertiesOverviewCard,
    StatusCard,
    StatusIcon
} from "@kumo-ai/kumo-ui-catalog";
import {
    ErrWarnDetails,
    GraphSnapshotResource,
    GraphSnapshotResponse
} from "gen/v0/types";

import GraphInformationPanel from "components/Graphs/InformationPanel/GraphInformationPanel";
import GraphLinkHealth from "components/Graphs/LinkHealth/GraphLinkHealth";
import ColumnSummaryWithStats from "components/Jobs/Graph/ColumnSummaryWithStats";
import { jobStatusIconMap } from "components/Jobs/jobStatusIcons";
import { useGraphSnapshotProperties } from "components/Overview/JobProperties";
import Validation from "components/Validation/Validation";

import {
    useGraphEdges,
    useGraphSnapshot,
    useGraphSnapshotStats
} from "hooks/snapshot/graphSnapshot";
import { useValidationState } from "hooks/validation/validation";

import {
    ColumnGroup,
    EdgeStats,
    GraphFlowInfo,
    GraphTables
} from "types/graphs";
import { ColumnStatsdownOption } from "types/inputs";
import { PredictiveQueryInfo } from "types/predictiveQueries";

import { jobStatus, JobStatusType } from "utils/Constants";
import {
    graphSnapshotToGraphInformationTables,
    graphSnapshotToJSGraph
} from "utils/Converters/snapshotConverter";
import { defFlowInfo } from "utils/graph";
import { copyObject } from "utils/Helpers/objectHelpers";
import { getColorVariant } from "utils/jobs";

const StatusSnapshotTitle: Record<jobStatus, string> = {
    [jobStatus.Cancelled]: "Graph snapshot cancelled",
    [jobStatus.Done]: "Graph snapshot created",
    [jobStatus.Failed]: "Graph snapshot failed",
    [jobStatus.Queued]: "Graph snapshot queued",
    [jobStatus.Running]: "Graph snapshot creation in progress...",
    [jobStatus.Untrained]: "",
    [jobStatus.None]: ""
};

const GraphSnapshotErrorDetails = (
    stages: NonNullable<GraphSnapshotResource["stages"]>
): string | undefined => {
    let out = "";
    const stage = stages?.INGEST;

    if (stage?.status) {
        if (stage.error) {
            out += `${stage.error}`;
        }
    }
    return out.length > 0 ? out : undefined;
};

/**
 * Graph Snapshot Overview component to display SDK Graph Snapshot.
 * Used from JobsSlider.tsx and 'Related Jobs' sections of PredictionJobOverview.tsx,
 * PredictionTableGenOverview.tsx, TrainingJobOverview.tsx and TrainingTableGenOverview.tsx
 *
 */
export interface GraphSnapshotOverviewProps {
    id: string;
    hidePreTitle?: boolean;
    setStatus?: (status: jobStatus) => void;
    parentJobSource?: string;
    queryInfo?: PredictiveQueryInfo;
    setWarnings?: (warnings: ErrWarnDetails[]) => void;
}

const GraphSnapshotOverview = ({
    id,
    hidePreTitle = false,
    parentJobSource,
    queryInfo,
    setStatus,
    setWarnings
}: GraphSnapshotOverviewProps) => {
    const [isLoading, setLoading] = useState<boolean>(true);

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [tableOption, setTableOption] = useState<ColumnStatsdownOption[]>([]);

    const { graphSnapshot, stages } = useGraphSnapshot(
        id as string,
        queryInfo?.graphDefinition,
        queryInfo?.graphVersion,
        setLoading,
        handleError,
        setStatus,
        setWarnings
    );
    const { graphEdges } = useGraphEdges(id, graphSnapshot);

    // graphFlowInfo ,allTables, groups are defined here just to pass it to
    // GraphInformationPanel , but is not used.
    const [groups, setGroups] = useState<Array<ColumnGroup>>([]);
    const [graphFlowInfo, setGraphFlowInfo] = useState<GraphFlowInfo>(
        copyObject(defFlowInfo)
    );
    // List of tables with individual table details
    const [allTables, setAllTables] = useState<GraphTables>({});

    useEffect(() => {
        if (tableOption?.length < 1 && graphSnapshot?.snapshot_resource) {
            const tableNames = Object.entries(
                graphSnapshot?.snapshot_resource?.table_ids ?? {}
            );
            if (tableNames?.length > 0) {
                const options: ColumnStatsdownOption[] = tableNames?.map(
                    ([tableName, tableId]) => ({
                        value: tableId,
                        label: tableName
                    })
                );
                if (graphSnapshot?.definition?.tables) {
                    const defTables = graphSnapshot.definition.tables;
                    options.forEach((o) => {
                        if (o?.label && defTables[o?.label]) {
                            o.pkey = defTables[o?.label].pkey;
                            o.timeCol = defTables[o.label].time_col;
                            o.endTimeCol = defTables[o.label].end_time_col;
                        }
                    });
                }
                setTableOption(options);
            }
        }
    }, [graphSnapshot, tableOption?.length]);

    const graphInfo = graphSnapshotToJSGraph(
        id as string,
        graphSnapshot as GraphSnapshotResponse
    );
    const tables = graphSnapshotToGraphInformationTables(
        graphSnapshot as GraphSnapshotResponse
    );
    const { totalRows, totalSizeBytes } = useGraphSnapshotStats(id);

    const source: string = parentJobSource ? parentJobSource : "";

    const status = stages?.INGEST?.status;

    const properties = useGraphSnapshotProperties({
        graphSnapshot,
        id,
        status: status as JobStatusType,
        startTime: stages?.INGEST?.start_time,
        isLoading,
        source,
        version: queryInfo?.kumoVersion,
        totalRows,
        totalSizeBytes
    });

    return (
        <div className="gap-6 p-6 flex flex-col">
            <PropertiesOverviewCard
                properties={properties}
                hideTitle={hidePreTitle}
                cardClassName="p-0"
            />

            {status &&
                status !== jobStatus.Done &&
                status !== jobStatus.Cancelled && (
                    <StatusCard
                        title={StatusSnapshotTitle[status as jobStatus]}
                        titleIcon={
                            <StatusIcon
                                status={status as jobStatus}
                                iconMap={jobStatusIconMap}
                            />
                        }
                        id={"graphsnapshot-overview"}
                        variant={getColorVariant(status)}
                        enableDivider={false}
                        contentClassName="break-all"
                        contentChild={GraphSnapshotErrorDetails(
                            stages
                        )}></StatusCard>
                )}

            <Validation
                pageLoading={false}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            {graphInfo && (
                <Card className="py-0 !py-0">
                    <CardContent>
                        <GraphInformationPanel
                            graphInfo={graphInfo}
                            tables={tables}
                            weakGraphEdges={
                                graphEdges?.filter(
                                    (edge) => edge?.weak_edge
                                ) as EdgeStats[]
                            }
                            gfc={graphFlowInfo}
                            setGfc={setGraphFlowInfo}
                            allTables={allTables}
                            setAllTables={setAllTables}
                            groups={groups}
                            setGroups={setGroups}
                            tableIds={graphInfo.table_name_to_fpid_mapping}
                            suggestLinks={() => {}}
                        />
                    </CardContent>
                </Card>
            )}
            {graphEdges && (
                <GraphLinkHealth
                    summaryParams={{
                        graphsEdges: graphEdges,
                        message:
                            graphEdges?.length === 0 ? "Fetching" : "Success"
                    }}
                    handleError={handleError}
                />
            )}
            {graphInfo && <ColumnSummaryWithStats tables={tableOption} />}
        </div>
    );
};
export default GraphSnapshotOverview;
