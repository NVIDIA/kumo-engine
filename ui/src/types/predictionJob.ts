// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JobTagsTags } from "gen/v0/types";
import { BatchPredictionInternalRequest } from "gen/v1/types";

type JobTags = JobTagsTags;

export type CreatePredictionJobArgs = BatchPredictionInternalRequest & {
    job_tags: JobTags;
    graph_snapshot_id?: string | null;
    pred_table_job_id?: string | null;
};
export enum PredictionJobErrorType {
    ERROR = 0,
    WARNING = 1
}
