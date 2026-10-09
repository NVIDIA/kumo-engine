// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import {
    useGetTrainingJobsCategoryNodePredRfmByCategory,
    useGetTrainingJobsNodePredRfmByEntityId
} from "gen/v0/api/training-jobs/training-jobs";
import { NodePredRFM } from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";

import { SubgraphSummary } from "types/training_job";

import { getKeys, getValues } from "utils/Helpers/objectHelpers";
import { ReactQueryKeys } from "utils/reactQueryKeys";

type Rfm = NodePredRFM | undefined;

/** Default any unset current/global field to the "<MISSING>" sentinel. */
function withMissingDefaults(d: SubgraphSummary): SubgraphSummary {
    if (!d["current"]) d["current"] = "<MISSING>";
    if (!d["global"]) d["global"] = "<MISSING>";
    return d;
}

function buildCountRow(node: Rfm, global: Rfm): SubgraphSummary {
    const d: SubgraphSummary = {
        property: "Count",
        current: undefined,
        global: undefined
    };
    if (node?.count && getValues(node.count).length > 0) {
        d["property"] = getKeys(node.count)[0] + " count";
        d["current"] = getValues(node.count)[0];
    }
    if (global?.count && getValues(global.count).length > 0) {
        d["global"] = getValues(global.count)[0];
    }
    return withMissingDefaults(d);
}

function buildRecentFactRow(node: Rfm, global: Rfm): SubgraphSummary {
    const d: SubgraphSummary = {
        property: "Most recent fact",
        current: undefined,
        global: undefined
    };
    if (
        node?.most_recent_rel_time &&
        getValues(node.most_recent_rel_time).length > 0
    ) {
        d["current"] = getValues(node.most_recent_rel_time)[0].join(" ");
    }
    if (
        global?.most_recent_rel_time &&
        getValues(global.most_recent_rel_time).length > 0
    ) {
        d["global"] = getValues(global.most_recent_rel_time)[0].join(" ");
    }
    return withMissingDefaults(d);
}

function buildMonetaryRow(): SubgraphSummary {
    return {
        property: "Monetary value",
        current: "<MISSING>",
        global: "<MISSING>"
    };
}

function buildUniqueCountRow(node: Rfm, global: Rfm): SubgraphSummary {
    const d: SubgraphSummary = {
        property: "Unique product count",
        current: undefined,
        global: undefined
    };
    if (node?.unique_counts && getValues(node.unique_counts).length > 0) {
        const countValue = getValues(node.unique_counts)[0];
        d["current"] = countValue[getKeys(countValue)[0]];
    }
    if (global?.unique_counts && getValues(global.unique_counts).length > 0) {
        const countValue = getValues(global.unique_counts)[0];
        d["global"] = countValue[getKeys(countValue)[0]];
    }
    return withMissingDefaults(d);
}

/** Build the current-vs-global subgraph summary rows, dropping all-missing rows. */
function buildSubgraphSummary(node: Rfm, global: Rfm): SubgraphSummary[] {
    return [
        buildCountRow(node, global),
        buildRecentFactRow(node, global),
        buildMonetaryRow(),
        buildUniqueCountRow(node, global)
    ].filter(
        (d) => d["current"] !== "<MISSING>" || d["global"] !== "<MISSING>"
    );
}

export const useTrainingJobNodePredRfm = (jobId: string, entityId: string) => {
    const { getQueryOptionsByEntityId, getQueryOptionsById } =
        useQueryOptions();

    const nodePredRfmQuery = useGetTrainingJobsNodePredRfmByEntityId(
        jobId ?? "",
        entityId,
        getQueryOptionsByEntityId(
            ReactQueryKeys.TrainingJobsNodePredRfmByEntityId,
            jobId,
            entityId,
            !!(jobId && entityId)
        )
    );

    const globalNodePredRfmQuery =
        useGetTrainingJobsCategoryNodePredRfmByCategory(
            jobId ?? "",
            "random",
            getQueryOptionsById(
                ReactQueryKeys.TrainingJobsCategoryNodePredRfmByCategory,
                jobId,
                !!jobId
            )
        );

    const nodePredRfm = useMemo(
        () => nodePredRfmQuery?.data?.data,
        [nodePredRfmQuery.data]
    );

    const globalNodePredRfm = useMemo(
        () => globalNodePredRfmQuery?.data?.data,
        [globalNodePredRfmQuery.data]
    );
    const subgraphSummary = useMemo(
        () => buildSubgraphSummary(nodePredRfm, globalNodePredRfm),
        [globalNodePredRfm, nodePredRfm]
    );

    return {
        nodePredRfm,
        globalNodePredRfm,
        isNodePredRfmLoading:
            nodePredRfmQuery.isLoading || globalNodePredRfmQuery.isLoading,
        subgraphSummary
    };
};
