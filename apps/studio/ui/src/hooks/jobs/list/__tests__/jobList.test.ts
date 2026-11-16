// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook, waitFor } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import modelTagsJson from "pages/Jobs/__tests__/MockResponses/ModelTags.test.json";
import predictionTagsJson from "pages/Jobs/__tests__/MockResponses/PredictionTags.test.json";
import getManyJsonRaw from "pages/NewJob/Prediction/__tests__/MockResponse/GetManyResponse.json";

const getManyJson = getManyJsonRaw as Record<string, string[]>;

import { TagType } from "types/jobs";

import { jobStatus } from "utils/Constants";

import { useGetQueryYaml, useJobsTags, useMergedActiveJobs } from "../jobList";

const JobType = {
    TRAINING_JOB: "TRAINING_JOB",
    BATCH_PREDICTION_JOB: "BATCH_PREDICTION_JOB"
} as const;

const makeJob = (
    job_id: string,
    job_type: string,
    status = jobStatus.Running
) => ({ job_id, job_type, metadata: { status } }) as never;

const makeJobWithPquery = (job_id: string, pquery_id: string) =>
    ({
        job_id,
        job_type: JobType.TRAINING_JOB,
        metadata: { tags: { pquery_id } }
    }) as never;

const getIds = (jobs: { job_id: string }[]) => jobs.map((j) => j.job_id);

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

describe("useMergedActiveJobs", () => {
    it("Should return currentJobs when activeJobsResponse is empty", () => {
        const currentJobs = [makeJob("job-1", JobType.TRAINING_JOB)];
        const { result } = renderWithQueryClient(() =>
            useMergedActiveJobs(
                {},
                currentJobs,
                `job_type=${JobType.TRAINING_JOB}`
            )
        );

        expect(result.current).toEqual(currentJobs);
    });

    it("Should return empty array when activeJobsResponse and currentJobs are empty", () => {
        const { result } = renderWithQueryClient(() =>
            useMergedActiveJobs({}, [], `job_type=${JobType.TRAINING_JOB}`)
        );

        expect(result.current).toEqual([]);
    });

    it.each([
        [JobType.TRAINING_JOB, `job_type=${JobType.TRAINING_JOB}`, true],
        [
            JobType.TRAINING_JOB,
            `job_type=${JobType.BATCH_PREDICTION_JOB}`,
            false
        ]
    ])(
        "Should include active job of type '%s' when pageSearch is '%s': %s",
        (jobType, pageSearch, shouldInclude) => {
            const activeJob = makeJob("active-1", jobType);
            const { result } = renderWithQueryClient(() =>
                useMergedActiveJobs({ "active-1": activeJob }, [], pageSearch)
            );

            expect(getIds(result.current).includes("active-1")).toBe(
                shouldInclude
            );
        }
    );

    it("Should include all active jobs when no status filter is set", () => {
        const runningJob = makeJob(
            "job-1",
            JobType.TRAINING_JOB,
            jobStatus.Running
        );
        const queuedJob = makeJob(
            "job-2",
            JobType.TRAINING_JOB,
            jobStatus.Queued
        );
        const { result } = renderWithQueryClient(() =>
            useMergedActiveJobs(
                { "job-1": runningJob, "job-2": queuedJob },
                [],
                `job_type=${JobType.TRAINING_JOB}`
            )
        );

        expect(getIds(result.current)).toEqual(
            expect.arrayContaining(["job-1", "job-2"])
        );
    });

    it.each([
        [jobStatus.Running, jobStatus.Running, true],
        [jobStatus.Running, jobStatus.Queued, false]
    ])(
        "Should filter active job with status '%s' by statusFilter '%s': %s",
        (jobStatusValue, filterStatus, shouldInclude) => {
            const activeJob = makeJob(
                "job-1",
                JobType.TRAINING_JOB,
                jobStatusValue
            );
            const pageSearch = `job_type=${JobType.TRAINING_JOB}&status=${filterStatus}`;
            const { result } = renderWithQueryClient(() =>
                useMergedActiveJobs({ "job-1": activeJob }, [], pageSearch)
            );

            expect(getIds(result.current).includes("job-1")).toBe(
                shouldInclude
            );
        }
    );

    it("Should deduplicate by job_id, keeping active job over current job", () => {
        const activeJob = makeJob(
            "job-1",
            JobType.TRAINING_JOB,
            jobStatus.Running
        );
        const currentJob = makeJob(
            "job-1",
            JobType.TRAINING_JOB,
            jobStatus.Done
        );
        const { result } = renderWithQueryClient(() =>
            useMergedActiveJobs(
                { "job-1": activeJob },
                [currentJob],
                `job_type=${JobType.TRAINING_JOB}`
            )
        );

        expect(result.current).toHaveLength(1);
        expect(result.current[0].metadata?.status).toBe(jobStatus.Running);
    });

    it("Should merge active jobs with non-overlapping current jobs", () => {
        const activeJob = makeJob(
            "job-1",
            JobType.TRAINING_JOB,
            jobStatus.Running
        );
        const currentJob = makeJob(
            "job-2",
            JobType.TRAINING_JOB,
            jobStatus.Done
        );
        const { result } = renderWithQueryClient(() =>
            useMergedActiveJobs(
                { "job-1": activeJob },
                [currentJob],
                `job_type=${JobType.TRAINING_JOB}`
            )
        );

        expect(getIds(result.current)).toEqual(
            expect.arrayContaining(["job-1", "job-2"])
        );
    });

    it("Should exclude string status values from activeJobsResponse", () => {
        const activeJob = makeJob(
            "job-1",
            JobType.TRAINING_JOB,
            jobStatus.Running
        );
        const { result } = renderWithQueryClient(() =>
            useMergedActiveJobs(
                {
                    "job-1": activeJob,
                    "job-2": jobStatus.Running // string value, not a GenericJob object
                },
                [],
                `job_type=${JobType.TRAINING_JOB}`
            )
        );

        const ids = getIds(result.current);
        expect(ids.includes("job-2")).toBe(false);
        expect(ids.includes("job-1")).toBe(true);
    });
});

describe("useGetQueryYaml", () => {
    it.each([
        ["empty jobs array", []],
        ["jobs without pquery_id", [makeJob("job-1", JobType.TRAINING_JOB)]]
    ])("Should return empty results when %s", (_, jobs) => {
        const { result } = renderWithQueryClient(() => useGetQueryYaml(jobs));

        expect(result.current.queryYamlResponse).toEqual({});
        expect(result.current.taskType).toEqual({});
    });

    it("Should map queryYaml and taskType from API response", async () => {
        const firstJobId = Object.keys(getManyJson)[0];
        const jobs = [makeJobWithPquery(firstJobId, "pq-1")];
        const { result } = renderWithQueryClient(() => useGetQueryYaml(jobs));

        await waitFor(() =>
            expect(result.current.queryYamlResponse[firstJobId]).toBeTruthy()
        );

        expect(result.current.queryYamlResponse[firstJobId]).toBe(
            getManyJson[firstJobId][0]
        );
        expect(result.current.taskType[firstJobId]).toBe(
            getManyJson[firstJobId][1]
        );
    });
});

describe("useJobsTags", () => {
    it.each([
        [TagType.TRAINING_JOB, modelTagsJson],
        [TagType.BATCH_PREDICTION_JOB, predictionTagsJson]
    ])(
        "Should return correct tags for tagsType '%s'",
        async (tagsType, expectedTags) => {
            const { result } = renderWithQueryClient(() =>
                useJobsTags(tagsType)
            );

            await waitFor(() => expect(result.current[1]).toBe(false));
            expect(result.current[0]).toEqual(expectedTags);
        }
    );

    it.each([[TagType.TRAINING_JOB], [TagType.BATCH_PREDICTION_JOB]])(
        "Should return empty object before data loads for tagsType '%s'",
        (tagsType) => {
            const { result } = renderWithQueryClient(() =>
                useJobsTags(tagsType)
            );

            expect(result.current[0]).toEqual({});
        }
    );
});
