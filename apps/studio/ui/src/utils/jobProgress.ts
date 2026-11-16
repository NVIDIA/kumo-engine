// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TrainingJobProgress } from "gen/v0/types";

import { parseDurationToSeconds } from "utils/timeConverter";

/**
 * Rough percent-complete for a running job: the server-reported
 * `elapsed_time` over the `start_time` / `estimated_end_time` estimate
 * window. Numerator and denominator both come from server-side clocks,
 * so the client's own clock never skews the ratio.
 *
 * On the wire `elapsed_time` is an ISO-8601 duration string (e.g.
 * "PT27M45.5S"); `Number()` on it yields NaN and froze the bar at 0
 * (kumo#29935), so it must go through `parseDurationToSeconds`.
 */
export function estimateJobProgressPercent(
    progress:
        | Pick<
              TrainingJobProgress,
              "start_time" | "estimated_end_time" | "elapsed_time"
          >
        | null
        | undefined
): number {
    if (!progress?.start_time || !progress.estimated_end_time) return 0;

    const start = new Date(progress.start_time).getTime();
    const end = new Date(progress.estimated_end_time).getTime();
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return 0;
    }

    const elapsed = parseDurationToSeconds(progress.elapsed_time);
    if (elapsed === null) return 0;

    const percent = (elapsed / ((end - start) / 1000)) * 100;
    if (percent <= 0) return 0;
    return percent < 100 ? percent : 98;
}
