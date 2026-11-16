// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { GenericJob, ListJobsQuery, ListJobsResponse } from "gen/v0/types";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse } from "tests/testUtil";

import { getValues } from "utils/Helpers/objectHelpers";
import { capitalizeFirstLetter, getIdHash } from "utils/Helpers/stringHelpers";
import { cleanedJobType } from "utils/jobs";
import { jobRunTimeFromJobResponse, parseDateTime } from "utils/timeConverter";

import mockGetJobsList from "../MockResponses/JobsList.test.response.json";
let currentTestResponseType = "";

export const expectedModelListTableHeader =
    "Job IDStatusStartedUserDurationTags";

export const expectedPredictionListTableHeader =
    "Job IDStatusStartedUserDurationTags";

export const mockJobsListResponse = (
    payload: ListJobsQuery
): Promise<AxiosResponse<ListJobsResponse>> => {
    if (currentTestResponseType === testResponseType.ERROR) {
        return mockErrorJobsListResponse();
    } else if (currentTestResponseType === testResponseType.EMPTY) {
        return mockEmptyJobsListResponse();
    } else if (currentTestResponseType === testResponseType.INVALID) {
        return mockInvalidJobsListResponse();
    }
    return mockDefaultJobsListResponse(payload);
};

export const setJobsTestResponseType = (responseType: string = "") => {
    currentTestResponseType = responseType;
};

const mockDefaultJobsListResponse = (
    payload: ListJobsQuery
): Promise<AxiosResponse<ListJobsResponse>> => {
    const {
        limit,
        page,
        status,
        search_keyword,
        tags_filter,
        job_type,
        sort_by,
        sort_desc
    } = payload;
    const paginatedData = filterJobs(
        page,
        limit,
        status,
        search_keyword ?? "",
        tags_filter,
        job_type,
        false,
        sort_by ?? undefined,
        sort_desc ?? false
    );
    return Promise.resolve(
        createMockResponse({
            jobs: paginatedData,
            total_num_items: 50
        }) as unknown as AxiosResponse<ListJobsResponse>
    );
};

export const mockEmptyJobsListResponse = (): Promise<
    AxiosResponse<ListJobsResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            jobs: [],
            total_num_items: 0
        }) as unknown as AxiosResponse<ListJobsResponse>
    );
};

export const mockInvalidJobsListResponse = (): Promise<
    AxiosResponse<ListJobsResponse>
> => {
    return Promise.resolve(
        createMockResponse({
            jobs: [[]],
            total_num_items: 0
        }) as unknown as AxiosResponse<ListJobsResponse>
    );
};

export const mockErrorJobsListResponse = () => {
    return Promise.reject(
        createMockResponse(new Error("Internal Server Error"), 500)
    );
};

export const expectedJobsListResponse = (
    page: number = 0,
    limit: number = 10,
    status: string[] = [],
    search_keyword: string = "",
    tagsFilter: string[] = [],
    jobType: string[] = [],
    sort_by?: string,
    sort_desc?: boolean
) => {
    return filterJobs(
        page,
        limit,
        status,
        search_keyword,
        tagsFilter,
        jobType,
        true,
        sort_by,
        sort_desc
    );
};

const flattenToExpectedString = (obj: GenericJob) => {
    const jobID = getIdHash(obj?.job_id);
    const jobType = cleanedJobType(obj?.metadata?.job_type);
    const started = parseDateTime(
        obj?.metadata?.start_time ?? obj.metadata.created_at
    );
    const status =
        obj?.metadata?.status === "CANCELLED"
            ? "Cancelled"
            : capitalizeFirstLetter(obj?.metadata?.status?.toLowerCase());
    const source =
        obj?.source === null || obj?.source === undefined || !obj?.source
            ? "\u2014"
            : obj?.source;
    const duration =
        obj?.metadata?.start_time && obj?.metadata?.end_time
            ? jobRunTimeFromJobResponse(
                  obj?.metadata?.start_time,
                  obj?.metadata?.end_time
              )
            : "\u2014";
    const tags = getValues(obj?.metadata?.tags).join("");

    return jobID + jobType + started + status + source + duration + tags;
};

const filterJobs = (
    page: number = 0,
    limit: number = 10,
    status: string[] = [],
    search_keyword: string = "",
    tagsFilter: string[] = [],
    jobType: string[] = [],
    flatten: boolean = false,
    sort_by?: string,
    sort_desc: boolean = false
) => {
    let result: string[] | GenericJob[] = [];
    let filteredData: GenericJob[] =
        status.length > 0
            ? (mockGetJobsList.jobs.filter((item) =>
                  item.metadata.status.includes(status[0])
              ) as unknown as GenericJob[])
            : (mockGetJobsList.jobs as unknown as GenericJob[]);

    if (jobType?.length > 0) {
        filteredData = filteredData.filter((item) => {
            return jobType.includes(item.job_type);
        }) as unknown as GenericJob[];
    }
    if (tagsFilter?.length > 0) {
        filteredData = filteredData.filter((item) => {
            const data = Object.values(item.metadata.tags ?? {});
            return tagsFilter.some((value) => data.includes(value));
        }) as unknown as GenericJob[];
    }

    if (search_keyword) {
        filteredData = filteredData.filter(
            (item) =>
                item.job_id.includes(search_keyword) ||
                item.job_type.includes(search_keyword)
        ) as unknown as GenericJob[];
    }

    if (sort_by) {
        filteredData = [...filteredData].sort((a, b) => {
            let comparison = 0;

            switch (sort_by) {
                case "job_id":
                    comparison = (a.job_id || "").localeCompare(b.job_id || "");
                    break;

                case "metadata.job_type":
                    comparison = (a.job_type || "").localeCompare(
                        b.job_type || ""
                    );
                    break;

                case "metadata.status":
                    comparison = (a.metadata?.status || "").localeCompare(
                        b.metadata?.status || ""
                    );
                    break;

                case "source":
                    comparison = (a.source || "").localeCompare(b.source || "");
                    break;

                case "metadata.start_time":
                    {
                        const timeA = new Date(
                            a.metadata?.start_time || 0
                        ).getTime();
                        const timeB = new Date(
                            b.metadata?.start_time || 0
                        ).getTime();
                        comparison = timeA - timeB;
                    }
                    break;

                case "metadata.tags":
                    {
                        const tagsA = Object.values(
                            a.metadata?.tags || {}
                        ).join("");
                        const tagsB = Object.values(
                            b.metadata?.tags || {}
                        ).join("");
                        comparison = tagsA.localeCompare(tagsB);
                    }
                    break;

                case "duration":
                    {
                        const durationA =
                            a.metadata?.end_time && a.metadata?.start_time
                                ? new Date(a.metadata.end_time).getTime() -
                                  new Date(a.metadata.start_time).getTime()
                                : 0;
                        const durationB =
                            b.metadata?.end_time && b.metadata?.start_time
                                ? new Date(b.metadata.end_time).getTime() -
                                  new Date(b.metadata.start_time).getTime()
                                : 0;
                        comparison = durationA - durationB;
                    }
                    break;

                case "test_metric":
                    {
                        const metricA = a.test_metric?.val ?? 0;
                        const metricB = b.test_metric?.val ?? 0;
                        comparison = metricA - metricB;
                    }
                    break;

                default:
                    comparison = 0;
            }

            return sort_desc ? -comparison : comparison;
        });
    }

    if (filteredData?.length > limit) {
        const start = page * limit;
        const end =
            filteredData?.length > start + limit
                ? start + limit
                : filteredData?.length - start;
        const paginatedData = filteredData.slice(
            start,
            end
        ) as unknown as GenericJob[];

        result = flatten
            ? paginatedData.map((obj) => {
                  return flattenToExpectedString(obj);
              })
            : paginatedData;
    } else if (filteredData?.length < limit) {
        result = flatten
            ? filteredData.map((obj) => {
                  return flattenToExpectedString(obj);
              })
            : filteredData;
    }
    return result;
};
