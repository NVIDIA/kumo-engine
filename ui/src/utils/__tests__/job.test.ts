// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GenericJob,
    JobTags,
    JobType,
    StatedbConfigCommonJobStatus as JobStatus
} from "gen/v0/types";
import { afterEach, describe, expect, it } from "vitest";

import { mockCreateGenericJob } from "components/Jobs/List/__tests__/MockFunction/JobsListTable.test.mock";

import { JobOverviewType } from "types/jobs";

import { jobStatus } from "utils/Constants";
import {
    calculateSkeletonRows,
    cleanedJobType,
    getJobOverviewType,
    getJobOverviewTypeByPathname,
    getJobsListQueryParams,
    getJobsTagsOptions,
    isAnyJobFilterApplied,
    isJobsListPageForSegment,
    isPredictionsJobsPage,
    mergeActiveJobsWithCurrent
} from "utils/jobs";

describe("cleanedJobType", () => {
    it.each([
        {
            description:
                "Should replace underscores with spaces and title-case the result",
            input: "DATA_SCIENCE_JOB",
            expected: "Data Science"
        },
        {
            description: 'Should remove the word "generate" and "job"',
            input: "GENERATE_REPORT_JOB",
            expected: "Report"
        },
        {
            description: "Should lowercase the input before processing",
            input: "MaChine_Learning_JOB",
            expected: "Machine Learning"
        },
        {
            description: "Should trim extra spaces after replacements",
            input: "  GENERATE_data_job  ",
            expected: "Data"
        },
        {
            description:
                "Should return an empty string if input becomes empty after cleanup",
            input: "GENERATE_JOB",
            expected: ""
        },
        {
            description: "Should handle empty string input gracefully",
            input: "",
            expected: ""
        }
    ])("$description", ({ input, expected }) => {
        const result = cleanedJobType(input);
        expect(result).toBe(expected);
    });

    it("Should handle null/undefined input without throwing", () => {
        expect(() =>
            cleanedJobType(undefined as unknown as string)
        ).not.toThrow();
        expect(() => cleanedJobType(null as unknown as string)).not.toThrow();
    });
});

describe("getJobOverviewType", () => {
    const jobId = "12345";
    const query = "?debug=true";

    it.each([
        {
            description: "Should returncorrect info for TRAINING_JOB",
            jobType: JobType.TRAINING_JOB,
            expectedPage:
                "/project/project-test/models/training/" + jobId + query,
            expectedType: JobOverviewType.TRAINING_JOB
        },
        {
            description: "Should returncorrect info for BATCH_PREDICTION_JOB",
            jobType: JobType.BATCH_PREDICTION_JOB,
            expectedPage:
                "/project/project-test/predictions/prediction/" + jobId + query,
            expectedType: JobOverviewType.BATCH_PREDICTION_JOB
        },
        {
            description:
                "Should returncorrect info for GENERATE_TRAIN_TABLE_JOB",
            jobType: JobType.GENERATE_TRAIN_TABLE_JOB,
            expectedPage:
                "/project/project-test/models/trainingtable/" + jobId + query,
            expectedType: JobOverviewType.TRAIN_TABLE_JOB
        },
        {
            description:
                "Should returncorrect info for GENERATE_PREDICTION_TABLE_JOB",
            jobType: JobType.GENERATE_PREDICTION_TABLE_JOB,
            expectedPage:
                "/project/project-test/predictions/predictiontable/" +
                jobId +
                query,
            expectedType: JobOverviewType.PREDICTION_TABLE_JOB
        }
    ])("$description", ({ jobType, expectedPage, expectedType }) => {
        const [page, type] = getJobOverviewType(
            jobId,
            jobType,
            query,
            "project-test"
        );
        expect(page).toBe(expectedPage);
        expect(type).toBe(expectedType);
    });

    it.each([
        {
            description: "Should return[null, null] if jobId is null",
            testJobId: null,
            testJobType: JobType.TRAINING_JOB
        },
        {
            description: "Should return[null, null] if jobType is null",
            testJobId: jobId,
            testJobType: null
        },
        {
            description: "Should return[null, null] if jobType is unrecognized",
            testJobId: jobId,
            testJobType: "UNKNOWN_JOB" as JobType
        }
    ])("$description", ({ testJobId, testJobType }) => {
        const [page, type] = getJobOverviewType(
            testJobId,
            testJobType,
            query,
            "project-test"
        );
        expect(page).toBeNull();
        expect(type).toBeNull();
    });

    it("Should returnpage without query if pathQuery is empty", () => {
        const [page, type] = getJobOverviewType(
            jobId,
            JobType.TRAINING_JOB,
            "",
            "project-test"
        );
        expect(page).toBe("/project/project-test/models/training/" + jobId);
        expect(type).toBe(JobOverviewType.TRAINING_JOB);
    });
});

describe("getJobOverviewTypeByPathname", () => {
    const query = "?debug=true";

    it.each([
        {
            description: "Should return TrainingJob info",
            pathname: "/project/project-test/models/training/" + "123",
            expectedPage: "/project/project-test/models/training/" + query,
            expectedType: JobOverviewType.TRAINING_JOB
        },
        {
            description: "Should return PredictionJob info",
            pathname: "/project/project-test/predictions/prediction/" + "456",
            expectedPage:
                "/project/project-test/predictions/prediction/" + query,
            expectedType: JobOverviewType.BATCH_PREDICTION_JOB
        },
        {
            description: "Should return TrainTableJob info",
            pathname: "/project/project-test/models/trainingtable/" + "789",
            expectedPage: "/project/project-test/models/trainingtable/" + query,
            expectedType: JobOverviewType.TRAIN_TABLE_JOB
        },
        {
            description: "Should return PredictionTableJob info",
            pathname:
                "/project/project-test/predictions/predictiontable/" + "abc",
            expectedPage:
                "/project/project-test/predictions/predictiontable/" + query,
            expectedType: JobOverviewType.PREDICTION_TABLE_JOB
        }
    ])("$description", ({ pathname, expectedPage, expectedType }) => {
        const [page, type] = getJobOverviewTypeByPathname(
            pathname,
            query,
            "project-test"
        );
        expect(page).toBe(expectedPage);
        expect(type).toBe(expectedType);
    });

    it("Should return [null, null] for unknown path", () => {
        const [page, type] = getJobOverviewTypeByPathname(
            "/unknown/path",
            query,
            "project-test"
        );
        expect(page).toBeNull();
        expect(type).toBeNull();
    });

    it("Should return page without query if query is empty", () => {
        const [page, type] = getJobOverviewTypeByPathname(
            "/project/project-test/models/training/" + "id",
            "",
            "project-test"
        );
        expect(page).toBe("/project/project-test/models/training/");
        expect(type).toBe(JobOverviewType.TRAINING_JOB);
    });
});

describe("getJobsTagsOptions", () => {
    it("Should keep only user tags, excluding all system/ID tags", () => {
        // Regression for kumo-ai/kumo#29898: job/pquery IDs and other system
        // tags (pquery_name, pquery_id, parent_job, ...) must not bloat the
        // filter dropdown; only user-defined tags are offered.
        const tags = {
            "gen-traintable-job-0fd23d01f6064a58ac4aa2a6ba142209": {
                "timestamps": {
                    "created_at": "2025-04-24T08:55:38.613327+00:00",
                    "updated_at": "2025-04-24T08:55:38.613327+00:00"
                },
                "tags": {
                    "pquery_name": "financial_test",
                    "pquery_id": "pquery-9a60522917eaab2e7bc7f3c1a4306477",
                    "parent_job":
                        "trainingjob-0fd23d01f6064a58ac4aa2a6ba142209",
                    "team": "growth"
                }
            },
            "gen-traintable-job-1db2edcc9f30420898ce79b9161a4c56": {
                "timestamps": {
                    "created_at": "2025-04-03T05:29:33.349961+00:00",
                    "updated_at": "2025-04-03T05:29:33.349961+00:00"
                },
                "tags": {
                    "pquery_name": "churn_model",
                    "pquery_id": "pquery-5ac1a5d72856a9ba1f0f235ee11df1c1",
                    "parent_job":
                        "trainingjob-1db2edcc9f30420898ce79b9161a4c56",
                    "team": "ml"
                }
            }
        } as unknown as Record<string, JobTags>;

        const result = getJobsTagsOptions(tags);
        // Only the user "team" tag values remain.
        expect(result).toEqual(
            expect.arrayContaining([
                { value: "growth", label: "growth" },
                { value: "ml", label: "ml" }
            ])
        );
        // System tags (pquery_name, pquery_id, parent_job) are excluded.
        const values = result.map((option) => option.value);
        expect(values).not.toContain("financial_test");
        expect(values).not.toContain("pquery-9a60522917eaab2e7bc7f3c1a4306477");
        expect(values).not.toContain(
            "trainingjob-0fd23d01f6064a58ac4aa2a6ba142209"
        );
        expect(result).toHaveLength(2);
    });

    it("Should exclude every system/ID tag key from the options", () => {
        const tags = {
            job1: {
                tags: {
                    pquery_name: "financial_test",
                    pquery_id: "pquery-abc",
                    parent_job: "trainingjob-abc",
                    model_id: "model-abc",
                    base_train_job_id: "trainingjob-base",
                    archived: "true",
                    job_status: "DONE"
                }
            }
        } as unknown as Record<string, JobTags>;
        expect(getJobsTagsOptions(tags)).toEqual([]);
    });

    it("Should return an empty array if tags are empty", () => {
        const result = getJobsTagsOptions({});
        expect(result).toEqual([]);
    });

    it("Should handle jobs with empty tag objects", () => {
        const tags = {
            job1: { tags: {} },
            job2: { tags: {} }
        } as unknown as Record<string, JobTags>;
        const result = getJobsTagsOptions(tags);
        expect(result).toEqual([]);
    });

    it("Should handle tags with duplicate values across keys", () => {
        const tags = {
            job1: { tags: { region: "us", env: "us" } },
            job2: { tags: { stage: "us" } }
        } as unknown as Record<string, JobTags>;
        const result = getJobsTagsOptions(tags);
        expect(result).toEqual([{ value: "us", label: "us" }]);
        expect(result).toHaveLength(1);
    });
});

describe("calculateSkeletonRows", () => {
    it.each([
        {
            description: "Should return full limit for middle pages",
            page: 1,
            limit: 10,
            totalNumJobs: 100,
            expected: 10
        },
        {
            description: "Should return full limit for middle pages (case 2)",
            page: 5,
            limit: 20,
            totalNumJobs: 200,
            expected: 20
        },
        {
            description: "Should return remaining rows for last page",
            page: 9,
            limit: 10,
            totalNumJobs: 95,
            expected: 5
        },
        {
            description: "Should return remaining rows for last page (case 2)",
            page: 4,
            limit: 10,
            totalNumJobs: 45,
            expected: 5
        },
        {
            description: "Should return 0 for empty list",
            page: 0,
            limit: 10,
            totalNumJobs: 0,
            expected: 0
        },
        {
            description:
                "Should return full limit when totalNumJobs equals full pages",
            page: 9,
            limit: 10,
            totalNumJobs: 100,
            expected: 10
        },
        {
            description: "Should handle page 0 with partial results",
            page: 0,
            limit: 10,
            totalNumJobs: 5,
            expected: 5
        },
        {
            description: "Should handle page 0 with full results",
            page: 0,
            limit: 10,
            totalNumJobs: 10,
            expected: 10
        },
        {
            description:
                "Should return limit when remainder is 0 but not empty",
            page: 2,
            limit: 10,
            totalNumJobs: 30,
            expected: 10
        }
    ])("$description", ({ page, limit, totalNumJobs, expected }) => {
        expect(calculateSkeletonRows(page, limit, totalNumJobs)).toBe(expected);
    });
});

describe("mergeActiveJobsWithCurrent", () => {
    const mockJob1: GenericJob = mockCreateGenericJob({
        job_id: "job1",
        job_type: JobType.TRAINING_JOB
    });

    const mockJob2: GenericJob = mockCreateGenericJob({
        job_id: "job2",
        job_type: JobType.BATCH_PREDICTION_JOB
    });

    it("Should merge active jobs with current jobs", () => {
        const activeJobsResponse = {
            job1: mockJob1
        };
        const currentJobs = [mockJob2];
        const pageSearch =
            "?job_type=TRAINING_JOB&job_type=BATCH_PREDICTION_JOB";

        const result = mergeActiveJobsWithCurrent(
            activeJobsResponse,
            currentJobs,
            pageSearch
        );

        expect(result.length).toBe(2);
        expect(result.find((j) => j.job_id === "job1")).toBeDefined();
        expect(result.find((j) => j.job_id === "job2")).toBeDefined();
    });

    it("Should filter jobs by page type", () => {
        const activeJobsResponse = {
            job1: mockJob1,
            job2: mockJob2
        };
        const currentJobs: GenericJob[] = [];
        const pageSearch = "?job_type=TRAINING_JOB";

        const result = mergeActiveJobsWithCurrent(
            activeJobsResponse,
            currentJobs,
            pageSearch
        );

        expect(result.length).toBe(1);
        expect(result[0].job_id).toBe("job1");
        expect(result[0].job_type).toBe(JobType.TRAINING_JOB);
    });

    it("Should remove duplicate jobs", () => {
        const activeJobsResponse = {
            job1: mockJob1
        };
        const currentJobs = [mockJob1, mockJob2];
        const pageSearch =
            "?job_type=TRAINING_JOB&job_type=BATCH_PREDICTION_JOB";

        const result = mergeActiveJobsWithCurrent(
            activeJobsResponse,
            currentJobs,
            pageSearch
        );

        expect(result.length).toBe(2);
        const job1Count = result.filter((j) => j.job_id === "job1").length;
        expect(job1Count).toBe(1);
    });

    it("Should handle empty active jobs", () => {
        const activeJobsResponse = {};
        const currentJobs = [mockJob1, mockJob2];
        const pageSearch =
            "?job_type=TRAINING_JOB&job_type=BATCH_PREDICTION_JOB";

        const result = mergeActiveJobsWithCurrent(
            activeJobsResponse,
            currentJobs,
            pageSearch
        );

        expect(result.length).toBe(2);
    });

    it("Should handle empty current jobs", () => {
        const activeJobsResponse = {
            job1: mockJob1,
            job2: mockJob2
        };
        const currentJobs: GenericJob[] = [];
        const pageSearch =
            "?job_type=TRAINING_JOB&job_type=BATCH_PREDICTION_JOB";

        const result = mergeActiveJobsWithCurrent(
            activeJobsResponse,
            currentJobs,
            pageSearch
        );

        expect(result.length).toBe(2);
    });

    it("Should filter out non-object job status values", () => {
        const activeJobsResponse = {
            job1: jobStatus.Running,
            job2: mockJob2
        };
        const currentJobs: GenericJob[] = [];
        const pageSearch = "?job_type=BATCH_PREDICTION_JOB";

        const result = mergeActiveJobsWithCurrent(
            activeJobsResponse,
            currentJobs,
            pageSearch
        );

        expect(result.length).toBe(1);
        expect(result[0].job_id).toBe("job2");
    });
});

describe("isAnyJobFilterApplied", () => {
    it.each([
        {
            description: "Should return false when no filters applied",
            query: {
                search_keyword: "",
                tags_filter: [],
                status: [],
                page: 0
            },
            expected: false
        },
        {
            description: "Should return true when search keyword is applied",
            query: {
                search_keyword: "test",
                tags_filter: [],
                status: [],
                page: 0
            },
            expected: true
        },
        {
            description: "Should return true when tags filter is applied",
            query: {
                search_keyword: "",
                tags_filter: ["tag1"],
                status: [],
                page: 0
            },
            expected: true
        },
        {
            description: "Should return true when status filter is applied",
            query: {
                search_keyword: "",
                tags_filter: [],
                status: ["DONE"] as JobStatus[],
                page: 0
            },
            expected: true
        },
        {
            description: "Should return true when multiple filters applied",
            query: {
                search_keyword: "test",
                tags_filter: ["tag1"],
                status: ["RUNNING"] as JobStatus[],
                page: 2
            },
            expected: true
        },
        {
            description: "Should handle null search_keyword",
            query: {
                search_keyword: null,
                tags_filter: [],
                status: [],
                page: 0
            },
            expected: false
        },
        {
            description: "Should handle undefined values",
            query: {
                search_keyword: undefined,
                tags_filter: undefined,
                status: undefined,
                page: undefined
            },
            expected: false
        }
    ])("$description", ({ query, expected }) => {
        expect(
            isAnyJobFilterApplied({
                project_id: "project_00000000000000000000000000000001",
                ...query
            })
        ).toBe(expected);
    });
});

describe("isJobsListPageForSegment", () => {
    it.each([
        {
            pathname: "/project/default/predictions",
            segment: "predictions" as const,
            expected: true
        },
        {
            pathname: "/project/default/predictions/",
            segment: "predictions" as const,
            expected: true
        },
        {
            pathname: "/project/default/predictions/prediction/123",
            segment: "predictions" as const,
            expected: true
        },
        {
            pathname: "/project/default/models",
            segment: "models" as const,
            expected: true
        },
        {
            pathname: "/project/default/models/",
            segment: "models" as const,
            expected: true
        },
        // A different page must not match either segment.
        {
            pathname: "/project/default/graphs",
            segment: "predictions" as const,
            expected: false
        },
        // The models list is not the predictions list.
        {
            pathname: "/project/default/models",
            segment: "predictions" as const,
            expected: false
        },
        // A longer segment that merely starts with the name must not match.
        {
            pathname: "/project/default/predictions-archive",
            segment: "predictions" as const,
            expected: false
        }
    ])(
        "'$pathname' matches '$segment' => $expected",
        ({ pathname, segment, expected }) => {
            expect(isJobsListPageForSegment(pathname, segment)).toBe(expected);
        }
    );
});

describe("isPredictionsJobsPage", () => {
    it.each([
        {
            description: "bare no-slash predictions path",
            pathname: "/project/default/predictions",
            search: "",
            expected: true
        },
        {
            description: "trailing-slash predictions path",
            pathname: "/project/default/predictions/",
            search: "",
            expected: true
        },
        {
            description: "models path with a batch-prediction job_type filter",
            pathname: "/project/default/models",
            search: "?job_type=" + JobType.BATCH_PREDICTION_JOB,
            expected: true
        },
        {
            description: "models path with a prediction-table job_type filter",
            pathname: "/project/default/models",
            search: "?job_type=" + JobType.GENERATE_PREDICTION_TABLE_JOB,
            expected: true
        },
        {
            description: "plain models path",
            pathname: "/project/default/models",
            search: "",
            expected: false
        },
        {
            description: "non-jobs path",
            pathname: "/project/default/graphs",
            search: "",
            expected: false
        }
    ])("$description => $expected", ({ pathname, search, expected }) => {
        expect(isPredictionsJobsPage(pathname, search)).toBe(expected);
    });
});

describe("getJobsListQueryParams job_type default (#29832)", () => {
    const PAGE_SIZE = 25;
    const PROJECT = "default";
    const originalLocation = window.location;

    const setLocation = (pathname: string, search = "") => {
        Object.defineProperty(window, "location", {
            value: { pathname, search },
            writable: true,
            configurable: true
        });
    };

    afterEach(() => {
        Object.defineProperty(window, "location", {
            value: originalLocation,
            writable: true,
            configurable: true
        });
    });

    it.each([
        {
            description:
                "bare /predictions path defaults to BATCH_PREDICTION_JOB",
            pathname: "/project/default/predictions",
            search: "",
            params: {},
            expected: [JobType.BATCH_PREDICTION_JOB]
        },
        {
            description: "bare /models path defaults to TRAINING_JOB",
            pathname: "/project/default/models",
            search: "",
            params: {},
            expected: [JobType.TRAINING_JOB]
        },
        {
            description:
                "trailing-slash /predictions/ path keeps BATCH_PREDICTION_JOB",
            pathname: "/project/default/predictions/",
            search: "",
            params: {},
            expected: [JobType.BATCH_PREDICTION_JOB]
        },
        {
            description: "trailing-slash /models/ path keeps TRAINING_JOB",
            pathname: "/project/default/models/",
            search: "",
            params: {},
            expected: [JobType.TRAINING_JOB]
        }
    ])("$description", ({ pathname, search, params, expected }) => {
        setLocation(pathname, search);
        const result = getJobsListQueryParams(params, PAGE_SIZE, PROJECT);
        expect(result.job_type).toEqual(expected);
    });

    it("respects an explicit job_type in the query string over the page default", () => {
        setLocation(
            "/project/default/predictions",
            "?job_type=" + JobType.TRAINING_JOB
        );
        const result = getJobsListQueryParams(
            { job_type: JobType.TRAINING_JOB },
            PAGE_SIZE,
            PROJECT
        );
        expect(result.job_type).toEqual([JobType.TRAINING_JOB]);
    });

    it("applies no type filter on a non-jobs list path", () => {
        setLocation("/project/default/graphs");
        const result = getJobsListQueryParams({}, PAGE_SIZE, PROJECT);
        expect(result.job_type).toEqual([]);
    });
});
