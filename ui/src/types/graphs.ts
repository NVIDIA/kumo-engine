// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrWarnResponse, ResourceTimestamps } from "gen/v0/types";
import {
    ColumnKey,
    KumoapiGraphGraphDefinitionTables as GraphDefinitionTables,
    ListResponse
} from "gen/v1/types";

import type { GraphID } from "./ids";
import { DropdownOption } from "./inputs";
import { TableDefinition, TableName } from "./tables";

export enum GraphLinkSetupState {
    SUGGEST_LINKS = "suggest_links",
    EDIT_LINKS = "edit_links",
    UPDATED_LINKS = "update_links"
}

export type GraphFlowInfo = {
    name: string;
    id: GraphID;
    description: string;
    tables: GraphDefinitionTables;
    selected_tables: Array<string | null>;
    selected_table_ids: Array<string>;
    flow_type: string;
    links_state?: GraphLinkSetupState;
};

export type GraphListResponse = Omit<
    ListResponse,
    | "timestamps"
    | "id"
    | "source_table_name"
    | "connector_id"
    | "num_tables"
    | "num_links"
> & {
    timestamps: ResourceTimestamps;
    id: string;
    source_table_name?: string;
    connector_id?: string;
    dateCreated?: string;
    num_tables?: number;
    num_links?: number;
};

export type ColumnGroup = {
    columns: Array<ColumnKey>;
    state?: "new" | "saved" | "confirmation required";
    updated?: boolean;
};

export type GraphTables = GraphDefinitionTables;

export type Graph = {
    tables: GraphDefinitionTables;
    col_groups: Array<ColumnGroup>;
};

export type GraphResponse = {
    id?: GraphID;
    name_alias?: string;
    graph?: Graph;
    num_links?: number;
    num_tables?: number;
    timestamps: ResourceTimestamps;
    table_name_to_fpid_mapping: { [key: string]: string };
};

export type EdgeStats = {
    col_a: string;
    col_a_table_name: string;
    col_a_column_name: string;
    col_a_table_id: string;

    col_b: string;
    col_b_table_name: string;
    col_b_column_name: string;
    col_b_table_id: string;

    link_type: string;
    weak_edge: boolean;

    matches_a_in_b: string | number;
    matches_a_in_b_original: number;
    percent_a_in_b: string;
    weak_a_in_b?: boolean;

    matches_b_in_a: string | number;
    matches_b_in_a_original: number;
    percent_b_in_a: string;
    weak_b_in_a?: boolean;
};

export type JSGraph = {
    id?: GraphID;
    name?: string;
    tables?: GraphDefinitionTables;
    groups?: Array<ColumnGroup>;
    totalLinks: number;
    totalTables: number;
    /** Raw ISO timestamps from the API; created/updated below are pre-formatted. */
    timestamps?: ResourceTimestamps;
    created?: string;
    updated?: string;
    dateCreated?: string;
    dateUpdated?: string;
    isActive?: boolean;
    table_name_to_fpid_mapping?: { [key: string]: string };
} & Partial<ErrWarnResponse>;

export type GraphLinkGroup = {
    columnOptions: Array<GraphLinkColumnOption>;
    selectedTable: DropdownOption;
    selectedColumn: DropdownOption;
    invalid: boolean;
};

export type GraphLinkColumnOption = {
    value: string;
    label: string;
    sortIndex: number;
};

export type GraphDefinition = {
    tables: Record<TableName, TableDefinition>;
    col_groups: Array<ColumnGroup>;
};

export type ValidateGraphResponse = {
    warnings: [
        {
            title: "string";
            message: "string";
        }
    ];
    errors: [
        {
            loc: ["string", 0];
            message: "string";
            type: "string";
        }
    ];
    info_items: [
        {
            title: "string";
            message: "string";
            id: "string";
        }
    ];
};

// ─── Neighbourhood graph / subgraphs-sampling types ──────────────────────────
// Used by useNeighborhoodGraph to represent the schema edges and hop-level
// sampling configuration parsed from the model plan YAML.

type DirectedEdge = {
    sourceTable: string;
    sourceColumn: string;
    targetTable: string;
    targetColumn: string;
};

export type EdgeBudget = {
    edge: DirectedEdge;
    budget: number;
};

export type HopEdgeConfig = {
    hop: number;
    defaultBudget: number;
    edgeBudgets: EdgeBudget[];
};

export type NeighborSamplingInfo = {
    hops: HopEdgeConfig[];
    hasData: boolean;
};

export type SchemaEdge = {
    table1: string;
    col1: string;
    table2: string;
    col2: string;
};

// ─────────────────────────────────────────────────────────────────────────────

export const NA_VALUES = [
    "N/A",
    "NOT SUPPORTED",
    "CALCULATING",
    "CALCULATING..."
];
