// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GenericJob,
    JobType,
    StatedbConfigCommonJobStatus as JobStatus
} from "gen/v0/types";

export const mockCreateGenericJob = (
    overrides?: Partial<GenericJob>
): GenericJob => {
    const jobType = overrides?.job_type ?? JobType.TRAINING_JOB;
    const status = overrides?.execution?.status ?? JobStatus.RUNNING;
    const timestamp = new Date().toISOString();

    return {
        job_id: "mock-job-id",
        job_type: jobType,
        execution: {
            status: status,
            log: {
                stage_name: "mock-stage",
                last_updated_at: timestamp
            },
            start_time: timestamp,
            last_updated_at: timestamp
        },
        metadata: {
            created_at: timestamp,
            start_time: timestamp,
            end_time: null,
            job_type: jobType,
            status: status
        },
        ...overrides,
        project_id:
            overrides?.project_id ?? "project_00000000000000000000000000000001"
    };
};
