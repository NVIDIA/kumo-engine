// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const MetricsData = [
    {
        "name": "Kumo Model",
        "f1@10": "0.11",
        "f1@100": "0.020",
        "map@1": "0.13",
        "recall@1": "0.13",
        "recall@100": "1",
        "map@100": "0.21",
        "precision@1": "0.13",
        "precision@10": "0.063",
        "precision@100": "0.010",
        "f1@1": "0.13",
        "map@10": "0.20",
        "recall@10": "0.63",
        "locality_score": "0.13",
        "color_f1@10": "rgba(18, 158, 30, 0.6111111111111112)",
        "color_map@10": "rgba(18, 158, 30, 0.4444444444444445)",
        "color_precision@10": "rgba(18, 158, 30, 0.63)",
        "color_recall@10": "rgba(18, 158, 30, 0.63)"
    },
    {
        "name": "Global Frequency",
        "f1@10": "0",
        "f1@100": "0",
        "map@1": "0",
        "recall@1": "0",
        "recall@100": "0",
        "map@100": "0",
        "precision@1": "0",
        "precision@10": "0",
        "precision@100": "0",
        "f1@1": "0",
        "map@10": "0",
        "recall@10": "0",
        "tipText":
            "Predicting all entities with the most popular ranked list of target values in all previous splits",
        "color_f1@10": "rgba(255, 255, 255, 0)",
        "color_map@10": "rgba(255, 255, 255, 0)",
        "color_precision@10": "rgba(255, 255, 255, 0)",
        "color_recall@10": "rgba(255, 255, 255, 0)"
    },
    {
        "name": "Per Entity Frequency",
        "f1@10": "0.18",
        "f1@100": "0.020",
        "map@1": "0.19",
        "recall@1": "0.19",
        "recall@100": "1",
        "map@100": "0.45",
        "precision@1": "0.19",
        "precision@10": "0.10",
        "precision@100": "0.010",
        "f1@1": "0.19",
        "map@10": "0.45",
        "recall@10": "1",
        "tipText":
            "Predicting each entities with the most popular ranked list of target values for this entity in all previous splits",
        "color_f1@10": "rgba(18, 158, 30, 1)",
        "color_map@10": "rgba(18, 158, 30, 1)",
        "color_precision@10": "rgba(18, 158, 30, 1)",
        "color_recall@10": "rgba(18, 158, 30, 1)"
    }
];

export const ColumnData = [
    {
        "header": "Metrics",
        "id": "name",
        "accessorKey": "name"
    },
    {
        "header": "F1@10",
        "id": "f1@10",
        "accessorKey": "f1@10"
    },
    {
        "header": "MAP@10",
        "id": "map@10",
        "accessorKey": "map@10"
    },
    {
        "header": "PRECISION@10",
        "id": "precision@10",
        "accessorKey": "precision@10"
    },
    {
        "header": "RECALL@10",
        "id": "recall@10",
        "accessorkey": "recall@10"
    }
];
