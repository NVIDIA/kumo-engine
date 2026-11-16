// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const cancelAPICall: string = "Request was cancelled";

export const jobErrTitle: string =
    "This job has failed. Please contact Kumo for more information.";

export const SUPPORT_CONTACT_MESSAGE: string = "Please contact Kumo support";

/**
 * Generic, safe message shown in place of a raw markup response body
 * (e.g. a proxy-generated nginx HTML error page) so raw `<html>...` is never
 * surfaced to the user. Shared by the axios response interceptor and the
 * parseErrorMessage markup guard so both layers speak with one voice - #30031.
 */
export const UNEXPECTED_RESPONSE_MESSAGE: string =
    "The server returned an unexpected response. Please retry. If the " +
    "problem persists, contact Kumo Support.";

export const predictionWindowSplitPopoverMessage =
    "Prediction window is the target tables time ranges " +
    "used to generate labels for a train table timeframe " +
    "following target aggregation in predictive query syntax. " +
    "e.g. if the target aggregation in predictive query " +
    "syntax is SUM(col, 30, 60), and anchor date for one " +
    "train table timeframe is 2023-01-01 then the " +
    'prediction window is "2023-02-01 - 2023-03-01".';

export const predictionWindowNonSplitPopoverMessage =
    "Prediction window for split table is the set of target table " +
    "time ranges used to generate labels for all train table " +
    "timeframes constituing a train/val/test set. " +
    "e.g. if the target aggregation in predictive query syntax is " +
    "SUM(col, 30, 60), and anchor date for the first train table " +
    "timeframe is 2023-01-01, and that train set contains the first " +
    "three consecutive timeframes, then the prediction window for " +
    'the first 3 train table timeframes are "2023-02-01 - 2023-03-01", ' +
    '"2023-03-01 - 2023-04-01" and "2023-04-01 - 2023-05-01". ' +
    "the prediction window for the train set is then " +
    '"2023-02-01 - 2023-05-01".';

export const anchorDateMessage =
    'Anchor date is the "day 0" for model prediction ' +
    "on each prediction horizon following target " +
    "aggregation in predictive query syntax. " +
    "e.g. if the target aggregation in predictive query " +
    "syntax is SUM(col, 30, 60, days), and anchor date for one " +
    "train table timeframe is 2023-01-01 then the " +
    "model prediction is made for labels between " +
    "2023-01-31 and 2023-03-02.";

export const JOBS_LIST_EMPTY_STATE = {
    TITLE: "No matches for your selected filters.",
    BUTTON_TEXT: "Reset filters"
} as const;
