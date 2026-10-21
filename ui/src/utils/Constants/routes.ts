// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JobType } from "gen/v0/types";

export enum navRoutes {
    Home = "/home",
    Admin = "/admin",
    NewProject = "/project/new",
    ProjectBase = "/project/:projectId",
    AdminGroups = "/admin/groups",
    AdminGroupEdit = "/admin/groups/:groupId",
    AdminIdentitySettings = "/admin/identity-settings",
    AdminAddIdentity = "/admin/identity-settings/add",
    AdminMembers = "/admin/members",
    ApiKeys = "/api-keys"
}

export enum docURLs {
    QuickStart = "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/kumo-studio-code-guide/installation",
    BatchPrediction = "batch-prediction",
    SnowflakeSite = "https://docs.snowflake.com/en/user-guide/gen-conn-config",
    BigQuerySite = "https://cloud.google.com/bigquery/docs",
    DatabricksSite = "https://docs.databricks.com/aws/en",
    EncoderOverrides = "reference/column_processing#encoder-overrides",
    Metrics = "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/legacy-ui-references/metrics",
    NeighborSampling = "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/legacy-ui-references/neighbor-sampling",
    ModelArchitecture = "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/train-model/model-settings/model-planner#architecture-search",
    CreateDateSelection = "column-preprocessing#create-date-column",
    PkeySelection = "column-preprocessing#primary-key-column"
}

export const defaultModelsPageURL =
    navRoutes.Home + "?page=1&job_type=" + JobType.TRAINING_JOB;

/**
 * Returns project-scoped absolute paths for a given projectId.
 * Use these for navigation within a project context.
 */
export const getProjectPath = (projectId?: string | null) => {
    if (!projectId) {
        throw new Error("Project resource navigation requires projectId");
    }
    return {
        Connectors: `/project/${projectId}/connectors/`,
        NewConnector: `/project/${projectId}/connectors/new-connector`,
        EditConnector: `/project/${projectId}/connectors/edit-connector`,
        Tables: `/project/${projectId}/tables/`,
        NewTable: `/project/${projectId}/tables/new-table`,
        EditTable: `/project/${projectId}/tables/edit-table`,
        Graphs: `/project/${projectId}/graphs/`,
        NewGraph: `/project/${projectId}/graphs/new-graph`,
        EditGraph: `/project/${projectId}/graphs/edit-graph`,
        ModelJobsList: `/project/${projectId}/models/`,
        PredictionJobsList: `/project/${projectId}/predictions/`,
        TrainingJob: `/project/${projectId}/models/training/`,
        PredictionJob: `/project/${projectId}/predictions/prediction/`,
        TrainTableJob: `/project/${projectId}/models/trainingtable/`,
        PredictionTableJob: `/project/${projectId}/predictions/predictiontable/`,
        NewTrainingJob: `/project/${projectId}/newjob/training/`,
        NewTrainingGraphSelect: `/project/${projectId}/newjob/training/graph-select/`,
        NewPredictionJob: `/project/${projectId}/newjob/prediction/`,
        ProjectSettings: `/project/${projectId}/settings`
    };
};

export const publicPaths: string[] = ["", "login", "logout"];

export const newDocsURL = "https://kumo.ai/docs/";
