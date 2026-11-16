// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useRef } from "react";

import { ScalarChartData } from "@kumo-ai/kumo-ui-catalog";
import { useQueries } from "@tanstack/react-query";
import { useGetTensorboardTagsByJobId } from "gen/v0/api/tensorboard/tensorboard";
import { GetTensorboardManyScalars200 } from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";

import { getKeys, getValues } from "utils/Helpers/objectHelpers";
import { filterTags, isValidTag } from "utils/Helpers/tagHelpers";

const splits = ["train", "val"];

type ChartData = { [key: string]: ScalarChartData };

/** Sequentially renumber trail names from 0, preserving the version-/trial- prefix. */
function buildNormalizedTrails(trails: string[]): string[] {
    return trails.map((_, index) => {
        const sampleTrail = trails[0] || "";
        if (sampleTrail.includes("version-")) {
            return `version-${index}`;
        }
        if (sampleTrail.includes("trial-")) {
            return `trial-${index}`;
        }
        const prefix = sampleTrail.split("-")[0] || "experiment";
        return `${prefix}-${index}`;
    });
}

const trailKeyOf = (trailMapping: Map<string, string>, trail: string): string =>
    (trailMapping.get(trail) || "0").split("-").at(-1) || "0";

/** Initialize the tag -> split -> trail nested chart structure with empty series. */
function initChartData(
    tagsOptions: string[],
    trails: string[],
    trailMapping: Map<string, string>
): ChartData {
    const chartData: ChartData = {};
    for (const tag of tagsOptions) {
        chartData[tag] = {};
        for (const split of splits) {
            chartData[tag][split] = {};
            for (const trail of trails) {
                chartData[tag][split][trailKeyOf(trailMapping, trail)] = {
                    xData: [],
                    yData: []
                };
            }
        }
    }
    return chartData;
}

/** Fill in one trail's scalar series into the chart structure (mutates chartData). */
function populateTrailScalars(
    chartData: ChartData,
    data: GetTensorboardManyScalars200,
    trail: string,
    trailMapping: Map<string, string>
): void {
    const scalarKeys = getKeys(data);
    const hasScalars = scalarKeys.some(
        (key) => data[key].scalars && data[key].scalars.length > 0
    );
    if (!hasScalars) {
        return;
    }
    for (const key of scalarKeys) {
        const [split] = key.split("_");
        const tag = key.replace(`${split}_`, "");
        if (
            !(
                isValidTag(key) &&
                split &&
                tag &&
                data[key].scalars &&
                data[key].scalars.length > 0 &&
                chartData?.[tag]?.[split]
            )
        ) {
            continue;
        }
        const scalarData = data[key].scalars;
        // If step starts with 0, add one to all steps in the array - #11670.
        if (scalarData && scalarData.length > 0 && scalarData[0].step === 0) {
            scalarData.forEach((elem, i, arr) => {
                arr[i].step += 1;
            });
        }
        chartData[tag][split][trailKeyOf(trailMapping, trail)] = {
            xData: scalarData.map((s) => s.step),
            yData: scalarData.map((s) =>
                typeof s.value === "number" ? s.value : null
            )
        };
    }
}

export function useTensorBoardStats(jobId: string, trainingStatus: string) {
    const { queryOption } = useQueryOptions();
    const previosScalarResp = useRef<{
        [key: string]: GetTensorboardManyScalars200;
    }>({});

    const { data: tagsResponse, isLoading: tagsLoading } =
        useGetTensorboardTagsByJobId(
            jobId,
            undefined,
            queryOption.tensorboardTagsByJobId(jobId, trainingStatus)
        );

    // Extract tags, trails from the /scalar/tags API response
    // `tagsId` - To use as dependancy for scalar API as
    // scalar should be called once the tags api returns response
    // even though there is no changes tags or trials
    // `tagsOptions` - To use as drop-down option
    const { tags, trails, tagsOptions, tagsId } = useMemo(() => {
        if (tagsResponse && tagsResponse?.data) {
            const allTrails = getKeys(tagsResponse?.data?.trial_tags ?? {});

            // Filter trails that have valid tags (non-empty tag arrays)
            const trails = allTrails.filter((trail) => {
                const trialTags = tagsResponse?.data?.trial_tags[trail];
                const tagValues = getValues(trialTags ?? {});
                const validTags = tagValues.filter((tag: string) =>
                    isValidTag(tag)
                );
                return validTags.length > 0;
            });

            const uniqueTags = filterTags(
                trails,
                tagsResponse?.data?.trial_tags
            );
            const options = [
                ...new Set(
                    uniqueTags.map((t) =>
                        t.replace("train_", "").replace("val_", "")
                    )
                )
            ];

            return {
                tags: uniqueTags,
                trails: trails,
                tagsOptions: options,
                tagsId: uniqueTags.join("-") + new Date().getTime()
            };
        }
        return { tags: [], trails: [], tagsOptions: [], tagsId: "" };
    }, [tagsResponse]);

    // Trigger scalar API calls for each trails using useQueries
    const scalarsResponses = useQueries({
        queries: queryOption.tensorboardManyScalars(
            jobId,
            trails,
            tags,
            tagsId,
            trainingStatus
        )
    });

    const scalarDataList = scalarsResponses?.map((q) => q?.data);

    const isLoadingScalar = scalarsResponses.every((val) => val.isLoading);

    const { chartData } = useMemo(() => {
        const normalizedTrails = buildNormalizedTrails(trails);
        const trailMapping = new Map(
            trails.map((trial, index) => [trial, normalizedTrails[index]])
        );
        const chartData = initChartData(tagsOptions, trails, trailMapping);
        for (const [index, trail] of trails.entries()) {
            const data =
                scalarDataList && scalarDataList[index]
                    ? scalarDataList[index]?.data
                    : previosScalarResp.current[trail];
            if (scalarDataList && scalarDataList[index]?.data) {
                previosScalarResp.current[trail] = scalarDataList[index].data;
            }
            if (data) {
                populateTrailScalars(chartData, data, trail, trailMapping);
            }
        }
        return { chartData: chartData };
    }, [scalarDataList, tagsOptions, trails]);

    return {
        tags: tagsOptions,
        loading: tagsLoading || isLoadingScalar,
        chartStats: chartData
    };
}
