// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Location } from "react-router";

import {
    ColumnKey,
    KumoapiGraphGraphDefinition as GraphDefinition,
    KumoapiGraphGraphDefinitionTables as GraphDefinitionTables
} from "gen/v1/types";

import {
    ColumnGroup,
    GraphFlowInfo,
    GraphLinkSetupState,
    GraphTables,
    JSGraph
} from "types/graphs";
import { FlowType } from "types/tables";

import { copyObject, getKeys, isObjectNotEmpty } from "./Helpers/objectHelpers";
import { isValid } from "./Helpers/typeGuards";

// Function to get pkeys from a group of columns. The tables/cols featured
// in cols should be available in allTables. ( for now its just one for a valid group)
export const getpKeys = (cols: Array<ColumnKey>, allTables: GraphTables) => {
    const pkeys = [];

    for (let i = 0; i < cols?.length; i++) {
        const tName = cols[i]?.table_name;
        const pKey = allTables[tName]?.pkey;
        const pkeyData = isValid(pKey) && pKey === cols[i]?.col_name;
        if (pkeyData) pkeys.push(cols[i]);
    }
    return pkeys;
};

// Function to add new pkey columns if any, and remove invalid groups/cols
// ( invalid because user might have removed tables, or columns of a table)
export const updateGraphsGroups = (
    config: GraphFlowInfo,
    allTables: GraphTables,
    groups: Array<ColumnGroup>,
    setGroups: (g: Array<ColumnGroup>) => void
) => {
    const grps: Array<ColumnGroup> = [...groups];
    pruneInvalidGroups(grps, config, allTables);

    const newGroups = buildMissingPkeyGroups(grps, config);
    if (newGroups.length > 0) {
        // Append new group to the existing group.
        grps.push(...newGroups);
    }

    for (let i = 0; i < grps?.length; i++) {
        grps[i].updated = true;
    }

    setGroups(grps);
};

/**
 * Whether a group should be removed: true if it has not exactly one pkey, or
 * its pkey column's table/column was removed. Also splices out columns whose
 * table/column no longer exists (non-pkey case). Mutates currentGroup.columns.
 */
function shouldDeleteGroup(
    currentGroup: ColumnGroup,
    config: GraphFlowInfo,
    allTables: GraphTables
): boolean {
    const pkeys = getpKeys(currentGroup.columns, allTables);
    if (pkeys?.length !== 1) {
        // Pkey either not present or more than 1 - Remove this group.
        return true;
    }
    for (let j = currentGroup?.columns?.length - 1; j >= 0; j--) {
        const tableId = currentGroup.columns[j]?.table_name;
        const colName = currentGroup.columns[j]?.col_name;
        const table = config?.tables[tableId];
        const colExists =
            table !== undefined &&
            table?.cols?.find((c) => c?.name === colName) !== undefined;
        if (colExists) {
            continue;
        }
        // Table or column no longer present — remove the group if it was the
        // pkey, otherwise just drop this column.
        const wasPkey =
            table === undefined
                ? allTables[tableId]?.pkey === colName
                : table?.pkey === colName;
        if (wasPkey) {
            return true;
        }
        currentGroup?.columns?.splice(j, 1);
    }
    return false;
}

/**
 * Step 1: remove groups whose table/col is no longer selected. Iterate
 * backwards to facilitate deleting elements. Mutates grps.
 */
function pruneInvalidGroups(
    grps: Array<ColumnGroup>,
    config: GraphFlowInfo,
    allTables: GraphTables
): void {
    for (let i = grps?.length - 1; i >= 0; i--) {
        if (shouldDeleteGroup(grps[i], config, allTables)) {
            grps.splice(i, 1);
        }
    }
}

/** Step 2: build a new group for each table pkey missing from existing groups. */
function buildMissingPkeyGroups(
    grps: Array<ColumnGroup>,
    config: GraphFlowInfo
): Array<ColumnGroup> {
    const newGroups: Array<ColumnGroup> = [];
    for (const [tabName, table] of Object.entries(config?.tables)) {
        if (
            !isValid(tabName) ||
            !isObjectNotEmpty(table) ||
            !isValid(table?.pkey)
        ) {
            continue;
        }
        // Check if this pkey is present in any of the groups.
        let groupWithPkey = {};
        grps?.some((g) => {
            groupWithPkey = g?.columns?.find(
                (c) => c?.table_name === tabName && c?.col_name === table.pkey
            ) as ColumnKey;
            return groupWithPkey !== undefined;
        });

        if (
            groupWithPkey === undefined ||
            getKeys(groupWithPkey).length === 0
        ) {
            newGroups.push({
                columns: [
                    { table_name: tabName, col_name: table.pkey as string }
                ] as ColumnKey[],
                state: "new"
            });
        }
    }
    return newGroups;
}

export function createGraphBody(
    graphConfig: GraphFlowInfo,
    groups: Array<ColumnGroup>
): GraphDefinition {
    const body: GraphDefinition = {} as GraphDefinition;
    const col_groups: Array<ColumnGroup> = filterGroupsForSingleColumn(groups);
    body.col_groups = col_groups;
    body.tables = graphConfig?.tables as GraphDefinitionTables;
    return body;
}

export const filterGroupsForSingleColumn = (groups: ColumnGroup[]) => {
    const col_groups: Array<ColumnGroup> = [];

    for (let i = 0; i < groups?.length; i++) {
        if (groups[i]?.columns?.length > 1) {
            col_groups.push({
                columns: groups[i].columns as Array<ColumnKey>
            });
        }
    }
    return col_groups;
};
export const defFlowInfo: GraphFlowInfo = {
    name: "",
    id: "",
    description: "",
    tables: {},
    selected_tables: [],
    selected_table_ids: [],
    flow_type: FlowType.CREATE,
    links_state: GraphLinkSetupState.SUGGEST_LINKS
};

export const getDefaultGraphGroups = (graph?: JSGraph): ColumnGroup[] => {
    if (graph && graph?.groups) {
        const tmpGroups: ColumnGroup[] = graph?.groups;
        tmpGroups.forEach((g) => {
            g.updated = false;
            g.state = "saved";
        });
        return tmpGroups;
    }
    return [];
};

function resolveGraphFromLocation(location?: Location): {
    graph: JSGraph | null;
    name: string;
    flowType: FlowType;
} {
    const state = location?.state;
    let graph: JSGraph | null = null;
    let name: string = "";
    let flowType: FlowType = state?.flow_type ?? FlowType.CREATE;
    if (state?.flow_type !== FlowType.CREATE && state?.jsGraph) {
        graph = state.jsGraph;
        name = (
            state?.flow_type === FlowType.DUPLICATE
                ? graph?.name?.concat("_copy")
                : graph?.name
        ) as string;
    } else if (state?.copytrainingargs) {
        graph = state?.copytrainingargs?.graph as JSGraph;
        flowType = FlowType.DUPLICATE;
        name = graph?.id as string;
    }
    return { graph, name, flowType };
}

export const getDefaultGraphInfo = (location?: Location): GraphFlowInfo => {
    const { graph, name, flowType } = resolveGraphFromLocation(location);
    if (graph) {
        return {
            ...copyObject(defFlowInfo),
            "name": name as string,
            "id": graph?.id as string,
            "tables": graph?.tables ?? {},
            "selected_tables": [...getKeys(graph?.tables)],
            "flow_type": flowType
        };
    }

    return { ...copyObject(defFlowInfo) };
};
