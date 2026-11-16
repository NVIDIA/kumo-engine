// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const CategoryAggrFeature = {
    "table_name": "FINANCE_ACCOUNT",
    "name": "FREQUENCY",
    "value": "POPLATEK MESICNE",
    "stype": "categorical",
    "score": 0.03082957160549995
};

export const CategoryNodePredRf = {
    "entity_table_name": "FINANCE_ACCOUNT",
    "count": {
        "FINANCE_TRANS": 107.0,
        "FINANCE_ORDERS": 1.32
    },
    "avg_rel_time": {
        "FINANCE_TRANS": [300.892342796429, "days"]
    },
    "most_recent_rel_time": {
        "FINANCE_TRANS": [1.66, "days"]
    },
    "unique_counts": {
        "FINANCE_TRANS": {
            "FINANCE_ORDERS": 9.4
        }
    }
};

export const TopKFeature = [
    {
        "table_name": "FINANCE_ACCOUNT",
        "name": "DATE",
        "value": "1996-04-21T00:00:00",
        "stype": "timestamp",
        "score": 1.0
    },
    {
        "table_name": "FINANCE_ACCOUNT",
        "name": "DISTRICT_ID",
        "value": 33,
        "stype": "categorical",
        "score": 0.04876631125807762
    },
    {
        "table_name": "FINANCE_ACCOUNT",
        "name": "FREQUENCY",
        "value": "POPLATEK MESICNE",
        "stype": "categorical",
        "score": 0.007253148127347231
    }
];

export const NodePredRfm = {
    "entity_table_name": "FINANCE_ACCOUNT",
    "count": {
        "FINANCE_TRANS": 123.0,
        "FINANCE_ORDERS": 1.0
    },
    "avg_rel_time": {
        "FINANCE_TRANS": [351.3983739837398, "days"]
    },
    "most_recent_rel_time": {
        "FINANCE_TRANS": [1.0, "days"]
    },
    "unique_counts": {
        "FINANCE_TRANS": {
            "FINANCE_ORDERS": 7
        }
    }
};

export const SelectedNode = {
    "cols": [
        {
            "table_name": "FINANCE_ACCOUNT",
            "name": "DISTRICT_ID",
            "value": 74,
            "stype": "categorical",
            "score": 0.0636536255478859,
            "grad_score": 0.0636536255478859
        },
        {
            "table_name": "FINANCE_ACCOUNT",
            "name": "FREQUENCY",
            "value": "POPLATEK MESICNE",
            "stype": "categorical",
            "score": 0.2849496603012085,
            "grad_score": 0.2849496603012085
        },
        {
            "table_name": "FINANCE_ACCOUNT",
            "name": "DATE",
            "value": "1996-05-14T00:00:00",
            "stype": "timestamp",
            "score": 1,
            "grad_score": 1
        }
    ],
    "tableName": "FINANCE_ACCOUNT",
    "idName": "ACCOUNT_ID",
    "value": 136,
    "totalScore": 1.3486032858490944
};

export const SubgraphSummary = [
    {
        "property": "FINANCE_TRANS Count",
        "current": 123,
        "global": 107
    },
    {
        "property": "Most Recent Fact",
        "current": "1 days",
        "global": "1.66 days"
    }
];

export const TopKFeaturePartial = [
    {
        "table_name": "FINANCE_ACCOUNT",
        "name": "DATE",
        "value": null,
        "stype": "timestamp",
        "score": 1.0
    },
    {
        "table_name": "FINANCE_ACCOUNT",
        "name": "DISTRICT_ID",
        "value": 33,
        "stype": "categorical",
        "score": 0.04876631125807762
    },
    {
        "table_name": "FINANCE_ACCOUNT",
        "name": null,
        "stype": "categorical",
        "score": 0.007253148127347231
    }
];

export const MissingSubgraphSummary = [
    {
        "property": "FINANCE_TRANS Count",
        "current": "<MISSING>",
        "global": "<MISSING>"
    },
    {
        "property": "Most Recent Fact",
        "current": "1 days",
        "global": "1.66 days"
    }
];
