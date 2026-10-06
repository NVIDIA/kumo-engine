// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { Skeleton } from "@kumo-ai/kumo-ui-catalog";
import type { ErrorDetail } from "gen/v0/types";

import { JobErrorType } from "types/warnings";

import { adjustError } from "utils/errorHandler";

export type JobsErrorProps = {
    errors: ErrorDetail[];
    fetchingErrors: boolean;
};

const JobsError: React.FC<JobsErrorProps> = ({ errors, fetchingErrors }) => {
    if (fetchingErrors) {
        return <Skeleton className="w-240 h-8" />;
    }
    return (
        errors
            ?.filter((err) => err?.type === JobErrorType.ERROR)
            .map((err, i) => (
                <div key={"job-err-list" + i}>
                    <span className="font-bold">
                        {err.title}
                        {" : "}
                    </span>
                    {adjustError(err.description)
                        .split("\n")
                        .map((line, index) => (
                            <React.Fragment key={"err_desc" + index}>
                                <span>{line}</span>
                                <br />
                            </React.Fragment>
                        ))}
                </div>
            )) ?? null
    );
};

export default JobsError;
