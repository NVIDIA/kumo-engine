// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JobType } from "gen/v0/types";

export const numSampleRows: number = 10;

export const warningsToShow: number = 3;

export const graphEdgeThreshold: number = 20.0;

// Toggle for the Custom SQL Tables authoring surface. On by default in this
// branch. Read at load time from the runtime `env-config.js` (`window._env_`),
// the same mechanism AUTH_DISABLED uses (see utils/authRuntimeConfig
// `truthyValue`), so a deployment can turn it OFF without a UI rebuild by setting
// `window._env_.ENABLE_CUSTOM_SQL_TABLES = "false"`. When the key is unset —
// including when `window`/`_env_` is unavailable (tests, SSR) — it defaults to
// `true`. Kept an exported const so existing vitest mocks that override it keep
// working.
const runtimeEnv = (): { ENABLE_CUSTOM_SQL_TABLES?: boolean | string } =>
    typeof window === "undefined"
        ? {}
        : ((
              window as unknown as {
                  _env_?: { ENABLE_CUSTOM_SQL_TABLES?: boolean | string };
              }
          )._env_ ?? {});

const truthyValue = (value?: boolean | string): boolean =>
    value === true || String(value).toLowerCase() === "true";

const rawCustomSqlFlag = runtimeEnv().ENABLE_CUSTOM_SQL_TABLES;
export const ENABLE_CUSTOM_SQL_TABLES: boolean =
    rawCustomSqlFlag === undefined || rawCustomSqlFlag === null
        ? true
        : truthyValue(rawCustomSqlFlag);

// Zone label for data-domain (wire/UTC) renderings only, e.g. parseDateTime
// and anchor-time headers. Resource metadata renders in the viewer's local
// timezone via the formatters in utils/timeConverter.ts.
export const timeZone: string = "UTC";

export const splitKey = "<split_date>";

export const SIDEBAR_VIEW_KEY = "sidebarViewOpen";

export const ADD_KUMO_ENTITY_VALUE = "<add_kumo_entity>";

export const MODEL_JOB_TYPE_OPTIONS = [
    { value: JobType.TRAINING_JOB, label: "Training" },
    { value: JobType.GENERATE_TRAIN_TABLE_JOB, label: "Training Table" }
];

export const PREDICTION_JOB_TYPE_OPTIONS = [
    { value: JobType.BATCH_PREDICTION_JOB, label: "Batch Prediction" },
    { value: JobType.GENERATE_PREDICTION_TABLE_JOB, label: "Prediction Table" }
];

export const ModelTitle = "Models";

export const PredictionTitle = "Predictions";

export const PROTECTED_TAG_KEYS = new Set([
    "pquery_name",
    "pquery_id",
    "base_train_job_id",
    "model_id",
    "parent_job",
    "archived",
    "limit",
    "job_status",
    "status",
    "skip_cache",
    "cache_ttl"
]);
