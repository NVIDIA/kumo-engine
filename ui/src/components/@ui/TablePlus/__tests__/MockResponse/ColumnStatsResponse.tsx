// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExtendedColStatsResponse } from "types/snapshots";

export const column_stats: ExtendedColStatsResponse = {
    "statistics_available": true,
    "columns": [
        {
            "name": "CustomerID",
            "stype": "ID",
            "dtype": "Float64",
            "sample_rows": [
                "16685.0",
                "15970.0",
                "16875.0",
                "15550.0",
                "13596.0",
                "17812.0",
                "14114.0",
                "15750.0",
                "15145.0",
                "13267.0"
            ],
            "statistics": {
                "counts": {
                    "total": 100,
                    "missing_percentage": 0,
                    "num_unique": 100,
                    "invalid_percentage": 0
                },
                "summary_stats": [],
                "distribution_points": [],
                "distribution_category": [],
                "distribution_date": [],
                "bp_stats": {},
                "approx": false
            }
        }
    ],
    "approx": false,
    "rows_used": 100
};

export const sampleRowsWithInvalidValue = {
    "statistics_available": true,
    "columns": [
        {
            "name": "CustomerID",
            "stype": "ID",
            "dtype": "Float64",
            "sample_rows": [
                null,
                "15970.0",
                "16875.0",
                "15550.0",
                "13596.0",
                null,
                "14114.0",
                "15750.0",
                "15145.0"
            ],
            "statistics": {
                "counts": {
                    "total": 100,
                    "missing_percentage": 0,
                    "num_unique": 100,
                    "invalid_percentage": 0
                },
                "summary_stats": [],
                "distribution_points": null,
                "distribution_category": null,
                "distribution_date": null,
                "bp_stats": null,
                "approx": false
            }
        }
    ],
    "approx": false,
    "rows_used": 100
};
