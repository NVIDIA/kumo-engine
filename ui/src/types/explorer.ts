// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PredictedNodeType } from "@kumo-ai/kumo-ui-catalog";
import { KumoapiTaskTaskType, NodeOutputData } from "gen/v0/types";

import { ColumnArgs } from "./tables";

export type NodeData = {
    id: string;
    cols: ColumnArgs[];
    links?: LinkData[];
    nodeID?: string;
    id_name?: string;
    pkey?: string;
    pkey_name?: string;
    fkey?: string;
    fkey_name?: string;
};

type LinkData = {
    table_name: string;
    fkey_name: string;
    nodes: NodeOutputData[];
};

export type LinkPredictionEntityOutput = {
    task_type: KumoapiTaskTaskType;
    prediction: NodeOutputData[];
    target: NodeOutputData[];
};

export type RegressionEntityOutput = {
    task_type: KumoapiTaskTaskType;
    prediction: number;
    target: number;
};

export type ClassificationEntityOutput = {
    task_type: KumoapiTaskTaskType;
    prediction: { [key: string]: number };
    target: { [key: string]: number };
};

export type GraphNodeData = {
    cols: (ColumnArgs & XaiColumnArgs)[];
    tableName: string;
    idName: string;
    value: string;
    totalScore: number;
    hopLevel?: number;
    aggregatedCount?: number;
    hasPrimaryKey?: boolean;
    /** TLP classification — drives shape + emerald GT ring. Populated when
     *  the node corresponds to a predicted/GT/historical item in TLP tasks. */
    nodeType?: PredictedNodeType;
};
export type GraphNode = {
    id: string;
    nodeId: string;
    position?: { x: number; y: number };
    data: GraphNodeData;
};

export type GraphEdge = {
    id: string;
    source: string;
    target: string;
    type?: string;
};

type XaiData = {
    globalAvgScore?: number;
    labels?: (string | number)[];
    predictions?: (string | number)[];
};

export type XaiColumnArgs = {
    // For Graph level Node Table View
    xai_global_avg?: number;
    xai_prediction?: number;
    isXaiUpdated?: boolean;
};

export type GraphXaiScores = { [key: string]: XaiData };

export type VisualSpecAxis = {
    col_keys: string[];
    get_value_fn: (v: (string | number)[]) => string | number;
};

export type SubgraphEdge = { [key: string]: (string | number)[] };
