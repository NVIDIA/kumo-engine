// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GraphResourceResponse } from "gen/v1/types";

import { ColumnGroup, GraphTables, JSGraph } from "types/graphs";
import {
    JSTable,
    SourceTableData,
    TableDetails,
    TableID,
    TableName
} from "types/tables";

import { tableLocationFromSourceTable } from "utils/Converters/snapshotConverter";
import { getConnectorIDFromTableSource } from "utils/Converters/tablesConverter";
import { getKeys, isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString,
    parseDate
} from "utils/timeConverter";

type FpidMapping = GraphResourceResponse["table_name_to_fpid_mapping"];

/** Prefix bare fpids with "NA.NA-"; mutates and returns the mapping ({} if empty). */
function normalizeFpidMapping(mapping: FpidMapping): FpidMapping {
    if (!isObjectNotEmpty(mapping)) {
        return {};
    }
    getKeys(mapping).forEach((t) => {
        if (mapping[t] && !mapping[t].includes("-")) {
            mapping[t] = "NA.NA-" + mapping[t];
        }
    });
    return mapping ?? {};
}

/** Created/updated timestamps, formatted per the fullTimeStamp flag. */
function assignGraphTimestamps(
    graphInfo: JSGraph,
    json: GraphResourceResponse,
    fullTimeStamp: boolean
): void {
    const ts = json?.timestamps;
    graphInfo.timestamps = ts;
    if (fullTimeStamp) {
        graphInfo.created = formatToShortLocalDateString(ts?.created_at);
        graphInfo.dateCreated = formatToLongLocalDateString(ts?.created_at);
        graphInfo.updated = formatToLongLocalDateString(ts?.updated_at);
        graphInfo.dateUpdated = formatToLongLocalDateString(ts?.updated_at);
    } else {
        graphInfo.created = parseDate(ts?.created_at);
        graphInfo.updated = parseDate(ts?.updated_at);
    }
}

export function jsGraphFromGraphResponse(
    graphResponseJson: GraphResourceResponse,
    fullTimeStamp: boolean = false
): JSGraph {
    const graphInfo: JSGraph = {} as JSGraph;
    const graph = graphResponseJson?.graph;
    graphInfo.id = graphResponseJson?.id;
    graphInfo.name = graphResponseJson?.name_alias ?? undefined;
    graphInfo.tables = graph?.tables as GraphTables;
    graphInfo.groups = graph?.col_groups as ColumnGroup[];
    graphInfo.totalLinks = graphResponseJson?.num_links
        ? graphResponseJson?.num_links
        : 0;
    graphInfo.totalTables = graphResponseJson?.num_tables
        ? graphResponseJson?.num_tables
        : 0;
    graphInfo.table_name_to_fpid_mapping = normalizeFpidMapping(
        graphResponseJson.table_name_to_fpid_mapping
    );
    assignGraphTimestamps(graphInfo, graphResponseJson, fullTimeStamp);
    return graphInfo;
}

export function tablesFromGraphResponse(graphInfo: JSGraph | null) {
    const tableArray: Array<JSTable> = [];
    if (graphInfo && isObjectNotEmpty(graphInfo?.tables)) {
        const tableNames = getKeys(graphInfo?.tables);
        tableNames?.forEach((t) => {
            if (graphInfo?.tables) {
                const cols = graphInfo.tables[t].cols.map((c) => {
                    return {
                        id: c.name,
                        name: c.name,
                        value: (c.stype as string) === "string" ? "value" : 1,
                        stype: c.stype,
                        dtype: c.dtype
                    };
                });
                const connectorId = getConnectorIDFromTableSource(
                    graphInfo.tables[t].source_table as SourceTableData
                );

                const elem: JSTable = {
                    name_alias: t as TableName,
                    table: graphInfo.tables[t] as TableDetails,
                    source_table: graphInfo.tables[t]
                        .source_table as SourceTableData,
                    s3Connector: { id: connectorId, name: connectorId },
                    id: t as TableID,
                    description: "",
                    sourceTable: "",
                    connector: connectorId,
                    connectorId: connectorId,
                    table_location:
                        tableLocationFromSourceTable(
                            graphInfo.tables[t].source_table as SourceTableData
                        ) +
                        "/" +
                        connectorId,
                    name_link: "",
                    graphs_original: [],
                    graphs: [],
                    sortUpdated: "",
                    sortCreated: "",

                    columns: cols,
                    num_cols: cols.length,
                    pkey: graphInfo.tables[t].pkey,
                    timeCol: graphInfo.tables[t].time_col,
                    endTimeCol: graphInfo.tables[t].end_time_col,
                    //Not available in response. To be fixed
                    earliestFact: "-",
                    latestFact: "-"
                };
                tableArray.push(elem);
            }
        });
    }
    return tableArray;
}
