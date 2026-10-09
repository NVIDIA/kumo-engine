// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { GraphSnapshotStatsResponse } from "gen/v0/types";
import { createMockResponse } from "tests/testUtil";

export const mockGraphSnapshotTotalRows = (): Promise<
    AxiosResponse<GraphSnapshotStatsResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            total_rows: 1067291,
            total_size_bytes: 982056960
        }) as AxiosResponse<GraphSnapshotStatsResponse>
    );
};
