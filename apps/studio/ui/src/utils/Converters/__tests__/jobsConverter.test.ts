// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GenerateTrainTableResponse } from "gen/v0/types";

import {
    formatToNumericLocalDateString,
    formatToShortLocalDateString,
    jobRunTimeFromJobResponse
} from "utils/timeConverter";

import {
    parsePredictionTableGenResponse,
    parseTrainTableGenResponse
} from "../jobsConverter";

import {
    predictionTableGenmockResponse,
    trainTableGenmockResponse
} from "./MockResponse/mockResponses";

describe("parseTrainTableGenResponse", () => {
    it("Should correctly parse the train table generation response", () => {
        const result = parseTrainTableGenResponse(trainTableGenmockResponse);

        expect(result.jobId).toBe(trainTableGenmockResponse.resource.job_id);
        expect(result.config).toEqual(
            trainTableGenmockResponse.resource.config
        );
        expect(result.created).toBe(
            trainTableGenmockResponse?.resource?.created_at
        );
        expect(result.createdAt).toBe(
            trainTableGenmockResponse?.resource?.created_at
        );
        expect(result.dateCreated).toBe(
            formatToShortLocalDateString(
                trainTableGenmockResponse?.resource?.created_at
            )
        );
        expect(result.source).toBe("UI");
        expect(result.tags).toEqual(trainTableGenmockResponse.resource.tags);
        expect(result.jobStatusReport).toEqual(
            trainTableGenmockResponse.resource.job_status_report
        );
        expect(result.graphSnapshotId).toBe(
            trainTableGenmockResponse?.resource?.config?.graph_snapshot_id
        );
        expect(result.status).toBe("DONE");
    });

    it("Should return empty runTime if start or end time is missing", () => {
        const responseWithoutEndTime: GenerateTrainTableResponse = {
            resource: {
                ...trainTableGenmockResponse.resource,
                job_status_report: {
                    ...trainTableGenmockResponse.resource.job_status_report,
                    end_time: undefined
                }
            }
        };

        const result = parseTrainTableGenResponse(responseWithoutEndTime);
        expect(result.runTime).toBe("");
    });

    it("Should compute runTime if start_time and end_time are present", () => {
        const responseWithTimes: GenerateTrainTableResponse = {
            resource: {
                ...trainTableGenmockResponse.resource,
                job_status_report: {
                    ...trainTableGenmockResponse.resource.job_status_report,
                    start_time: new Date(
                        "2025-06-04T20:34:29.757558+00:00"
                    ).toISOString(),
                    end_time: new Date(
                        "2025-06-04T20:46:54.026512+00:00"
                    ).toISOString()
                }
            }
        };

        const result = parseTrainTableGenResponse(responseWithTimes);

        expect(result.runTime).toBe(
            jobRunTimeFromJobResponse(
                responseWithTimes?.resource?.job_status_report?.start_time,
                responseWithTimes?.resource?.job_status_report?.end_time
            )
        );
    });
});

describe("parsePredictionTableGenResponse", () => {
    it("Should parses a valid prediction response correctly", () => {
        const result = parsePredictionTableGenResponse(
            predictionTableGenmockResponse
        );

        expect(result.jobId).toBe(
            predictionTableGenmockResponse.resource.job_id
        );
        expect(result.status).toBe("DONE");
        expect(result.config).toEqual(
            predictionTableGenmockResponse.resource.config
        );
        expect(result.created).toBe(
            formatToNumericLocalDateString(
                predictionTableGenmockResponse?.resource?.created_at
            )
        );
        expect(result.created).toBe(
            formatToNumericLocalDateString(
                predictionTableGenmockResponse?.resource?.created_at
            )
        );
        expect(result.dateCreated).toBe(
            formatToShortLocalDateString(
                predictionTableGenmockResponse?.resource?.created_at
            )
        );
        expect(result.source).toBe("UI");
        expect(result.tags).toEqual(
            predictionTableGenmockResponse.resource.tags
        );
        expect(result.jobStatusReport).toEqual(
            predictionTableGenmockResponse.resource.job_status_report
        );
        expect(result.graphSnapshotId).toBe(
            predictionTableGenmockResponse?.resource?.config?.graph_snapshot_id
        );
    });

    it("Should returns empty runTime if end_time is missing", () => {
        const responseWithoutEnd = {
            resource: {
                ...predictionTableGenmockResponse.resource,
                job_status_report: {
                    ...predictionTableGenmockResponse.resource
                        .job_status_report,
                    end_time: undefined
                }
            }
        };

        const result = parsePredictionTableGenResponse(responseWithoutEnd);
        expect(result.runTime).toBe("");
    });
    it("Should compute runTime if start_time and end_time are present", () => {
        const responseWithTimes = {
            resource: {
                ...predictionTableGenmockResponse.resource,
                job_status_report: {
                    ...predictionTableGenmockResponse.resource
                        .job_status_report,
                    start_time: "2025-06-05T04:36:27.551303+00:00",
                    end_time: "2025-06-05T04:44:31.118493+00:00"
                }
            }
        };

        const result = parsePredictionTableGenResponse(responseWithTimes);

        expect(result.runTime).toBe(
            jobRunTimeFromJobResponse(
                responseWithTimes.resource.job_status_report.start_time,
                responseWithTimes.resource.job_status_report.end_time
            )
        );
    });
});
