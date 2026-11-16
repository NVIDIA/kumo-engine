// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColStatsResponse, ColumnSummary } from "gen/v0/types";

export type ExtendedColumnSummary = Omit<ColumnSummary, "sample_rows"> & {
    sample_rows: (string | null)[] | null;
    nonEmpty?: string | null;
    isSampleAvailable?: boolean;
};

export type ExtendedColStatsResponse = Omit<ColStatsResponse, "columns"> & {
    columns: ExtendedColumnSummary[];
};
