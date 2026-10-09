// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "fs";
import { ListJobsQuery } from "gen/v0/types";
import { InsertFilterOverridesRequest } from "gen/v1/types";
import path from "path";
import { mockDebugInfoResponse } from "stores/__tests__/MockFunctions/DebugInfo.mock";

import { mockEditGroupResponse } from "components/Admin/Groups/__tests__/MockFunctions/EditGroup.mock";
import { mockGroupsResponse } from "components/Admin/Groups/__tests__/MockFunctions/Groups.mock";
import { mockCreateGroupResponse } from "components/Admin/Groups/__tests__/MockFunctions/NewGroup.mock";
import { mockBPAvailableXaiResponse } from "components/BatchPredictions/__tests__/MockFunctions/BPAvailableXai.mock";
import { mockBPXaiResponse } from "components/BatchPredictions/__tests__/MockFunctions/BPXai.mock";
import { mockPredictionStatsResponse } from "components/BatchPredictions/__tests__/MockFunctions/PredictionStats.mock";
import { mockGraphSnapshotResponse } from "components/Jobs/Graph/__tests__/MockFunctions/GraphSnapshot.mock";
import { mockGraphSnapshotTotalRows } from "components/Jobs/Graph/__tests__/MockFunctions/GraphSnapshotTotalRows.mock";
import { mockArchiveResponse } from "components/Jobs/Overview/__tests__/MockFunctions/Archive.mock";
import { mockAddTagResponse } from "components/Jobs/Prediction/__tests__/MockFunctions/AddTag.mock";
import { mockBatchPredictionJobResponse } from "components/Jobs/Prediction/__tests__/MockFunctions/BatchPredictionResponse.mock";
import { mockCancelBatchPredictionResponse } from "components/Jobs/Prediction/__tests__/MockFunctions/CancelBatchPrediction.mock";
import { mockDeleteTagResponse } from "components/Jobs/Prediction/__tests__/MockFunctions/DeleteTag.mock";
import { mockGetBpJobWarningResponse } from "components/Jobs/Prediction/__tests__/MockFunctions/GetWarningsResponse.mock";
import { mockPredictiveQueryResoureResponse } from "components/Jobs/Prediction/__tests__/MockFunctions/PredictiveQueryResource.mock";
import { mockCancelGenPredictionTableResponse } from "components/Jobs/PredictionTable/__tests__/MockFunctions/CancelGenPredictionTable.mock";
import { mockPredAnchorTimeResponse } from "components/Jobs/PredictionTable/__tests__/MockFunctions/PredAnchorTime.mock";
import {
    mockGenPredictionTableJobResponse,
    mockPredictionTotalEntities
} from "components/Jobs/PredictionTable/__tests__/MockFunctions/PredictionTableGeneration.mock";
import { mockTrainingAvailXaiResponse } from "components/Jobs/Training/__tests__/MockFunctions/AvailableXai.mock";
import { mockCancelTraining } from "components/Jobs/Training/__tests__/MockFunctions/CancelTraining.mock";
import { mockTrainingJobModelPlanResponse } from "components/Jobs/Training/__tests__/MockFunctions/ModelPlan.mock";
import {
    mockEmptyResponse,
    mockTrainingWarningResponse
} from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWarning.mock";
import {
    mockEncodersDefaultResponse,
    mockPredictiveQueryEncodersResponse,
    mockTrainingWorkflowResponse
} from "components/Jobs/Training/__tests__/MockFunctions/TrainingJobWorkflow.mock";
import { mockTrainingXaiResponse } from "components/Jobs/Training/__tests__/MockFunctions/Xai.mock";
import { mockCancelTrainingTable } from "components/Jobs/TrainingTable/__tests__/MockFunctions/CancelTrainingTable.mock";
import { mockGenerationStatsResponse } from "components/Jobs/TrainingTable/__tests__/MockFunctions/GenerationStats.mock";
import { mockTrainingTableGenMetaDataResponse } from "components/Jobs/TrainingTable/__tests__/MockFunctions/PredictiveQueryMetaDeta.mock";
import { mockSplitStatsResponse } from "components/Jobs/TrainingTable/__tests__/MockFunctions/SplitStats.mock";
import { mockTrainingTableGenResponse } from "components/Jobs/TrainingTable/__tests__/MockFunctions/TrainingTableGeneration.mock";
import { mockPredictionTableSchemaResponses } from "components/NewJob/Prediction/__tests__/MockFunctions/PredictionTableSchema.mock";
import { mockSampleForOutputResponses } from "components/NewJob/Prediction/__tests__/MockFunctions/SampleForOutput.mock";
import { mockBinaryClassificationResponses } from "components/PredictiveQueries/Evaluation/BinaryClassification/__tests__/MockFunctions/BinaryClassification.mock";
import { mockHistogramResponses } from "components/PredictiveQueries/Evaluation/Histogram/__tests__/MockFunctions/Histogram.mock";
import { mockRegressionHeatMapResponses } from "components/PredictiveQueries/Evaluation/RegressionHeatMap/__tests__/MockFunctions/RegressionHeatMap.mock";
import { mockCategoryAggrFeatureResponse } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/CategoryAggrFeature.mock";
import { mockCategoryNodePredRfResponseResponse } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/CategoryNodePredRf.mock";
import { mockEntityOutputResponse } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/EntityOutput.mock";
import { mockNodePredRfmResponseResponse } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/NodePredRfm.mock";
import { mockSubgraphResponse } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/Subgraph.mock";
import { mockTopKFeatureResponse } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/TopKFeature.mock";
import { mockManyScalarsResponse } from "components/PredictiveQueries/Training/__tests__/MockFunctions/ManyScalar.mock";
import { mockTrialTagsResponse } from "components/PredictiveQueries/Training/__tests__/MockFunctions/TrialTags.mock";
import { mockCreateProjectResponse } from "components/Project/__tests__/MockFunctions/CreateProject.mock";
import { mockProjectsByGroupResponse } from "components/Project/__tests__/MockFunctions/ProjectList.mock";
import {
    mockGetProjectResponse,
    mockUpdateProjectResponse
} from "components/Project/__tests__/MockFunctions/ProjectSettings.mock";
import { mockSDKGraphCodeModalResponses } from "components/SDKCodeModal/__tests__/MockFunctions/SDKGraphCodeModal.mock";
import { mockSDKPredictionCodeModalResponses } from "components/SDKCodeModal/__tests__/MockFunctions/SDKPredictionCodeModal.mock";
import { mockSDKTableCodeModalResponses } from "components/SDKCodeModal/__tests__/MockFunctions/SDKTableCodeModal.mock";
import { mockSDKTrainingCodeModalResponses } from "components/SDKCodeModal/__tests__/MockFunctions/SDKTrainingCodeModal.mock";
import { mockColumnResponse } from "components/Tables/Connect/__tests__/MockFunction/ColumnResponse.test.mock";
import { mockCompleteFileUpload } from "components/Tables/Connect/__tests__/MockFunction/CompleteFileUpload.mock";
import { mockCreateTableResponse } from "components/Tables/Connect/__tests__/MockFunction/CreateTable.mock";
import { mockDeleteTable } from "components/Tables/Connect/__tests__/MockFunction/DeleteTable.mock";
import { mockDeleteUploadedFile } from "components/Tables/Connect/__tests__/MockFunction/DeleteUploadedFile.mock";
import { mockInferMetadataResponse } from "components/Tables/Connect/__tests__/MockFunction/InferMetadata.mock";
import { mockSampleRowsResponse } from "components/Tables/Connect/__tests__/MockFunction/SampleRowsResponse.test.mock";
import { mockSourceTablesResponse } from "components/Tables/Connect/__tests__/MockFunction/SourceTables.mock";
import { mockStartFileUpload } from "components/Tables/Connect/__tests__/MockFunction/StartFileUpload.mock";
import { mockTableNameValidationResponse } from "components/Tables/Connect/__tests__/MockFunction/TableName.mock";
import { mockValidateSqlTableResponseFn } from "components/Tables/Connect/__tests__/MockFunction/ValidateSqlTable.test.mock";
import { mockValidateTableResponse } from "components/Tables/Connect/__tests__/MockFunction/ValidateTable.mock";

import {
    mockAdminApiKeysResponse,
    mockCreateAdminApiKeyApiResponse,
    mockDeleteAdminApiKeyResponse
} from "hooks/apiKeys/__tests__/MockFunctions/AdminApiKeys.mock";
import {
    mockCreateUserApiKeyApiResponse,
    mockDeleteUserApiKeyResponse,
    mockUserApiKeysResponse
} from "hooks/apiKeys/__tests__/MockFunctions/UserApiKeys.mock";

import { mockAdminInfoResponse } from "pages/Admin/__tests__/MockFunctions/AdminInfo.mock";
import { mockOrgMembersInfoResponse } from "pages/Admin/__tests__/MockFunctions/OrgMembers.mock";
import { mockConnectorListResponse } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { mockConnectorTablesResponse } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorTables.mock";
import { mockDeleteConnector } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/DeleteConnector.mock";
import { mockCreateConnectorResponse } from "pages/Connectors/NewConnector/__tests__/MockFunctions/CreateConnector.mock";
import { mockCreateGraphResponse } from "pages/Graphs/__tests__/MockFunction/GraphCreate.test.mock";
import { mockGraphDetailResponse } from "pages/Graphs/__tests__/MockFunction/GraphDetail.test.mock";
import { mockGraphNameValidationResponse } from "pages/Graphs/__tests__/MockFunction/GraphName.mock";
import {
    mockDeleteGraphResponse,
    mockEditGraphResponse,
    mockGraphsListResponse
} from "pages/Graphs/__tests__/MockFunction/GraphsList.test.mock";
import { mockGraphSuggestLinksResponse } from "pages/Graphs/__tests__/MockFunction/GraphSuggestLinks.mock";
import { mockValidateGraphResponse } from "pages/Graphs/__tests__/MockFunction/GraphValidate.test.mock";
import { mockActiveJobsResponse } from "pages/Jobs/__tests__/MockFunctions/ActiveJobs.mock";
import { mockJobsListResponse } from "pages/Jobs/__tests__/MockFunctions/JobList.mock";
import { mockModelTagsResponse } from "pages/Jobs/__tests__/MockFunctions/ModelTags.mock";
import { mockPredictionTagsResponse } from "pages/Jobs/__tests__/MockFunctions/PredictionTags.mock";
import { mockBatchPredictionJobsResponse } from "pages/NewJob/Prediction/__tests__/MockFunction/BatchPredictionJobs.mock";
import { mockBpOverrideFilterResponse } from "pages/NewJob/Prediction/__tests__/MockFunction/BpQueryOverrides.mock";
import { mockModelGetManyResponse } from "pages/NewJob/Prediction/__tests__/MockFunction/GetMany";
import { mockInsertFilterOverridesResponse } from "pages/NewJob/Prediction/__tests__/MockFunction/InsertFilterOverrides.mock";
import { mockOutputSchemaResponse } from "pages/NewJob/Prediction/__tests__/MockFunction/OutputSchema.mock";
import { mockModelPlanResponses } from "pages/NewJob/Training/__tests__/MockFunctions/ModelPlan.mock";
import { mockTrainResponse } from "pages/NewJob/Training/__tests__/MockFunctions/TrainPayload.mock";
import { mockValidatePayloadResponse } from "pages/NewJob/Training/__tests__/MockFunctions/ValidatePayload.mock";
import { mockColStatsResponse } from "pages/Tables/TablesList/__tests__/MockFunction/Colstats.test.mock";
import { mockConnectorResponse } from "pages/Tables/TablesList/__tests__/MockFunction/ConnectorResponse.test.mock";
import { mockTableDetailResponse } from "pages/Tables/TablesList/__tests__/MockFunction/TableDetail.test.mock";
import { mockTableListResponse } from "pages/Tables/TablesList/__tests__/MockFunction/TableList.test.mock";

import { Graph } from "types/graphs";

import { isNullish } from "utils/Helpers/typeGuards";

import { createMockResponse } from "./testUtil";

const mockDirectConnectUsers = [
    {
        user_id: "direct_connect|alice",
        email: "alice@example.com",
        display_name: "Alice Example",
        role: "admin",
        disabled: false
    },
    {
        user_id: "direct_connect|bob",
        email: "bob@example.com",
        display_name: "Bob Example",
        role: "user",
        disabled: false
    }
];

const mockExternalGroupMappings = [
    {
        id: "external-mapping-1",
        group_id: "group-001",
        identity_provider_id: "corporate-oidc",
        external_group_id: "engineering"
    },
    {
        id: "external-mapping-2",
        group_id: "group-001",
        identity_provider_id: "corporate-oidc",
        external_group_id: "data-science"
    }
];

const mockDirectGroupMembers = [
    {
        id: "group-member-1",
        principal_id: "test@121254452",
        principal_type: "user",
        source: "manual"
    },
    {
        id: "group-member-2",
        principal_id: "training-automation",
        principal_type: "service_account",
        source: "manual"
    }
];

export const mockAxiosInstance = {
    interceptors: {
        request: { use: vi.fn(), eject: vi.fn() },
        response: { use: vi.fn(), eject: vi.fn() }
    },
    post: vi.fn((url, payload, options) => {
        if (url === "v0/admin/users") {
            return Promise.resolve(
                createMockResponse({
                    user_id: "direct_connect|charlie",
                    email: payload.email,
                    display_name: payload.display_name || payload.email,
                    role: payload.role,
                    disabled: payload.disabled ?? false
                })
            );
        }
        if (url.startsWith("v0/admin/users/") && url.endsWith("/password")) {
            return Promise.resolve(
                createMockResponse(mockDirectConnectUsers[0])
            );
        }
        if (url.startsWith(`v0/authentication/`) && url.endsWith(`/refresh`)) {
            return Promise.resolve({
                status: 200,
                data: {
                    access_token: "new-access-token",
                    refresh_token: "new-refresh-token"
                }
            });
        }
        if (url.startsWith(`v0/jobs/`) && url.endsWith(`/archive`)) {
            return mockArchiveResponse();
        }
        if (url === "v0/jobs/") {
            return mockJobsListResponse(payload as ListJobsQuery);
        }
        if (url === "v0/jobs/get_active_jobs") {
            return mockActiveJobsResponse();
        }
        if (url === "v0/jobs/add_tag") {
            return mockAddTagResponse();
        }
        if (url === "v0/jobs/delete_tag") {
            return mockDeleteTagResponse();
        }
        if (url === "v1/predictive_queries/get_many") {
            return mockModelGetManyResponse();
        }
        if (
            url.startsWith("v1/batch_prediction_jobs/samples_for_output_schema")
        ) {
            return mockSampleForOutputResponses();
        }
        if (/^v0\/projects\/[^/]+\/connectors\/$/.test(url)) {
            return mockCreateConnectorResponse();
        }
        if (url === "v1/tables/validate") {
            return mockValidateTableResponse();
        }
        if (url === "v1/tables/resolve_sql") {
            return mockValidateSqlTableResponseFn();
        }
        if (url === "v1/tables/infer_metadata") {
            return mockInferMetadataResponse(payload);
        }
        if (
            url.startsWith("v0/projects/") &&
            url.endsWith("/connectors/uploads/start")
        ) {
            return mockStartFileUpload();
        }
        if (
            url.startsWith("v0/projects/") &&
            url.endsWith("/connectors/uploads/complete")
        ) {
            return mockCompleteFileUpload();
        }
        if (
            url.startsWith("v0/projects/") &&
            url.endsWith("/connectors/uploads/delete")
        ) {
            return mockDeleteUploadedFile();
        }
        if (url.startsWith("v1/tables/") && url.endsWith("/refresh")) {
            return Promise.resolve({ status: 200 });
        }
        if (url === "v1/predictive_queries/validate") {
            return mockValidatePayloadResponse();
        }
        if (url === "v1/predictive_queries/model_plan") {
            return mockModelPlanResponses();
        }
        if (url == "v0/training_jobs/train") {
            return mockTrainResponse();
        }
        if (
            url === "v1/tables/" &&
            options?.params?.name_alias !== undefined &&
            options?.params?.force_rename !== undefined
        ) {
            return mockCreateTableResponse();
        }
        if (url.startsWith(`v1/graphs/validate`)) {
            return mockValidateGraphResponse();
        }
        if (url.startsWith("v1/graphs/") && url.endsWith("/refresh")) {
            return Promise.resolve({ status: 200 });
        }

        if (url === "v1/graphs/" && options?.params?.force_rename === false) {
            return mockCreateGraphResponse();
        }
        if (
            url === "v1/graphs/" &&
            options?.params?.force_rename === true &&
            options?.params?.name_alias === undefined
        ) {
            //Graph delete (no name_alias sent — backend stores null)
            return mockDeleteGraphResponse(payload as Graph);
        }
        if (url === "v1/graphs/" && options?.params?.force_rename === true) {
            //Graph edit
            return mockEditGraphResponse(payload as Graph);
        }
        if (url === "v1/graphs/suggest_links") {
            //Graph suggest links
            return mockGraphSuggestLinksResponse();
        }
        if (url.startsWith(`v0/snapshots/graph`)) {
            return mockGraphSnapshotResponse();
        }

        if (url === "v1/batch_prediction_jobs/get_warnings") {
            return mockGetBpJobWarningResponse();
        }
        if (url === "v1/batch_prediction_jobs/output_schema") {
            return mockOutputSchemaResponse();
        }
        if (
            url.startsWith(`v1/predictive_queries/`) &&
            url.endsWith(`/insert_filter_overrides`)
        ) {
            return mockInsertFilterOverridesResponse(
                payload as InsertFilterOverridesRequest
            );
        }
        if (url === "v1/batch_prediction_jobs/") {
            return mockBatchPredictionJobsResponse();
        }
        if (url.startsWith(`v0/training_jobs`) && url.endsWith("/warnings")) {
            const id = url
                .split("v0/training_jobs/")
                .pop()
                .replace("/warnings", "");
            return mockTrainingWarningResponse(id);
        }
        if (url.startsWith(`v0/training_jobs`) && url.endsWith(`/cancel`)) {
            return mockCancelTraining();
        }
        if (
            url.startsWith(`v0/genpredtable_jobs/`) &&
            url.endsWith(`/cancel`)
        ) {
            return mockCancelGenPredictionTableResponse();
        }
        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith(`/cancel`)
        ) {
            return mockCancelBatchPredictionResponse();
        }
        if (
            url.startsWith(`v0/tensorboard/`) &&
            url.endsWith(`/plugin/scalars/many_scalars`)
        ) {
            return mockManyScalarsResponse();
        }
        if (url.startsWith("v0/log/log_ui_error")) {
            return Promise.resolve();
        }
        if (url === "v0/groups") {
            return mockCreateGroupResponse();
        }
        if (/^v0\/groups\/[^/]+\/external-mappings$/.test(url)) {
            return Promise.resolve(
                createMockResponse({
                    id: "external-mapping-created",
                    group_id: url.split("/")[2],
                    ...payload
                })
            );
        }
        if (/^v0\/groups\/[^/]+\/members$/.test(url)) {
            return Promise.resolve(
                createMockResponse({
                    id: "group-member-created",
                    ...payload
                })
            );
        }
        if (url.startsWith("v0/groups/") && url.endsWith("/projects")) {
            return mockCreateProjectResponse();
        }
        if (url === "v0/api-keys/admin") {
            return mockCreateAdminApiKeyApiResponse();
        }
        if (url === "v0/api-keys") {
            return mockCreateUserApiKeyApiResponse();
        }
        return Promise.reject(new Error(`UnSupported request: ${url}`));
    }),
    get: vi.fn((url, data, options) => {
        if (url === "v0/api-keys/admin") {
            return mockAdminApiKeysResponse();
        }
        if (url === "v0/api-keys") {
            return mockUserApiKeysResponse();
        }
        if (url === "v0/admin/info") {
            return mockAdminInfoResponse();
        }
        if (url === "v0/admin/members") {
            return mockOrgMembersInfoResponse();
        }
        if (url === "v0/admin/users") {
            return Promise.resolve(createMockResponse(mockDirectConnectUsers));
        }
        if (url === "v0/groups") {
            return mockGroupsResponse();
        }
        if (/^v0\/groups\/[^/]+$/.test(url)) {
            const group = {
                id: "group-001",
                slug: "engineering-team",
                display_name: "Engineering Team",
                description: "All engineers in the organization",
                members: mockDirectGroupMembers,
                external_group_mappings: mockExternalGroupMappings.map(
                    (mapping) => ({
                        id: mapping.id,
                        identity_provider_id: mapping.identity_provider_id,
                        external_group_id: mapping.external_group_id
                    })
                )
            };
            return Promise.resolve(createMockResponse(group));
        }
        if (/^v0\/groups\/[^/]+\/external-mappings$/.test(url)) {
            return Promise.resolve(
                createMockResponse(mockExternalGroupMappings)
            );
        }
        if (/^v0\/groups\/[^/]+\/members$/.test(url)) {
            return Promise.resolve(createMockResponse([]));
        }
        if (/^v0\/groups\/[^/]+\/all-authenticated-users$/.test(url)) {
            return Promise.resolve(createMockResponse(false));
        }
        if (url.startsWith("v0/groups/") && url.endsWith("/projects")) {
            return mockProjectsByGroupResponse();
        }
        if (url === "v0/projects/") {
            return mockProjectsByGroupResponse();
        }
        if (/^v0\/projects\/[^/]+\/connectors\/$/.test(url)) {
            return mockConnectorListResponse();
        }
        if (/^v0\/projects\/[^/]+\/connectors\/uploads\/connector$/.test(url)) {
            return Promise.resolve(
                createMockResponse({ connector_id: "project_upload_connector" })
            );
        }
        if (
            /^v0\/projects\/[^/]+\/connectors\/[^/]+\/source-tables$/.test(url)
        ) {
            return mockSourceTablesResponse();
        }
        if (
            /^v0\/projects\/[^/]+\/connectors\/[^/]+\/source-table-config$/.test(
                url
            )
        ) {
            const parts = url.split("/");
            const connectorId = parts[4];
            const sourceTableName = options?.params?.table_name as string;
            const dataSourceType = connectorId.includes("gcs")
                ? "GCS"
                : connectorId.includes("adls")
                  ? "ADLS"
                  : "S3";
            return Promise.resolve(
                createMockResponse({
                    source_table: {
                        connector_id: connectorId,
                        ...(dataSourceType === "S3"
                            ? {
                                  source: sourceTableName.startsWith("s3://")
                                      ? {
                                            mode: "uri",
                                            uri: sourceTableName
                                        }
                                      : {
                                            mode: "relative",
                                            path: sourceTableName
                                        }
                              }
                            : { source_table_name: sourceTableName }),
                        file_type:
                            sourceTableName.endsWith(".csv") ||
                            sourceTableName.endsWith("directory-dataset")
                                ? "CSV"
                                : "PARQUET",
                        data_source_type: dataSourceType
                    }
                })
            );
        }
        if (/^v0\/projects\/[^/]+\/connectors\/[^/]+\/columns$/.test(url)) {
            return mockColumnResponse(options?.params?.table_name);
        }
        if (/^v0\/projects\/[^/]+\/connectors\/[^/]+\/sample$/.test(url)) {
            return mockSampleRowsResponse(options?.params?.table_name);
        }
        if (/^v0\/projects\/[^/]+\/connectors\/[^/]+$/.test(url)) {
            return mockConnectorResponse();
        }
        if (/^v0\/projects\/[^/]+$/.test(url)) {
            return mockGetProjectResponse(url.split("/").at(-1));
        }
        if (url === "v1/tables/") {
            return mockTableListResponse();
        }
        if (url.startsWith(`v1/tables/`) && url.endsWith(`/col_stats`)) {
            const parts = url.split("/");
            const tableId = parts[2];

            return mockColStatsResponse(tableId);
        }
        if (
            url.startsWith(`v0/snapshots/table/`) &&
            url.endsWith(`/col_stats`)
        ) {
            const parts = url.split("/");
            const tableId = parts[3];

            return mockColStatsResponse(tableId);
        }
        if (url.startsWith("v1/tables/") && url.endsWith("/sdk")) {
            return mockSDKTableCodeModalResponses();
        }
        if (url.startsWith("v1/tables/") && url.endsWith("list_tables")) {
            return mockConnectorTablesResponse();
        }
        if (url.startsWith("v1/tables/") && url.endsWith("/validate_name")) {
            const name = url
                .replace("v1/tables/", "")
                .replace("/validate_name", "")
                .replace("%20", " ");
            return mockTableNameValidationResponse(name);
        }
        if (url.startsWith(`v1/tables/`)) {
            const id = url.replace("v1/tables/", "");

            return mockTableDetailResponse(id);
        }

        if (url.startsWith("v1/graphs/utils/validate_name")) {
            //Graph name validation
            const { params } = options;
            const name = params.name;
            return mockGraphNameValidationResponse(name);
        }
        if (
            url.startsWith(`v0/connectors/`) &&
            url.includes(`/source_tables_v2`) &&
            !isNullish(options?.params?.response_token)
        ) {
            //  Maybe needed if we need to pass connector wise response
            //  const conName = url.replace("v0/connectors/", "").split("/")[0];
            return mockSourceTablesResponse();
        }

        if (url.startsWith(`v0/connectors/`)) {
            return mockConnectorResponse();
        }

        if (url === "v0/constants/debug_app" || url === "v0/constants/flags") {
            return mockDebugInfoResponse();
        }

        if (url === "v1/graphs/") {
            return mockGraphsListResponse();
        }
        if (url.startsWith(`v1/graphs/`) && url.endsWith(`/sdk`)) {
            return mockSDKGraphCodeModalResponses();
        }
        if (url.startsWith(`v1/graphs/`)) {
            return mockGraphDetailResponse();
        }
        if (url.startsWith(`v0/jobs/training_tags`)) {
            return mockModelTagsResponse();
        }
        if (url.startsWith(`v0/jobs/prediction_tags`)) {
            return mockPredictionTagsResponse();
        }
        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith(`/available_xai_entities`)
        ) {
            return mockBPAvailableXaiResponse();
        }
        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith("/xai")
        ) {
            return mockBPXaiResponse();
        }
        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith("/prediction_stats")
        ) {
            return mockPredictionStatsResponse();
        }
        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.includes("/subgraph/")
        ) {
            return mockSubgraphResponse();
        }

        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith(`/prediction_table_schema`)
        ) {
            return mockPredictionTableSchemaResponses();
        }

        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith(`/sdk`)
        ) {
            return mockSDKPredictionCodeModalResponses();
        }
        if (
            url.startsWith(`v1/batch_prediction_jobs/`) &&
            url.endsWith("/prediction_anchor_time")
        ) {
            return Promise.resolve({
                status: 200,
                data: { anchor_time: "2025-12-16T00:00:00" }
            });
        }

        if (url.startsWith(`v1/batch_prediction_jobs/`)) {
            return mockBatchPredictionJobResponse();
        }
        if (
            url.startsWith(`v0/predictive_queries/`) &&
            url.endsWith(`/resource`)
        ) {
            return mockPredictiveQueryResoureResponse();
        }

        if (url.startsWith(`v0/training_jobs`) && url.endsWith("/workflow")) {
            const id = url
                .split("v0/training_jobs/")
                .pop()
                .replace("/workflow", "");
            return mockTrainingWorkflowResponse(id);
        }
        if (url.startsWith(`v0/training_jobs`) && url.endsWith("/encoders")) {
            return mockEncodersDefaultResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.endsWith("/model_plans")
        ) {
            return mockTrainingJobModelPlanResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.endsWith("/eval/available_xai_entities")
        ) {
            return mockTrainingAvailXaiResponse();
        }
        if (url.startsWith(`v0/training_jobs`) && url.endsWith("/eval/xai")) {
            return mockTrainingXaiResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.includes(`eval/entity_output`)
        ) {
            return mockEntityOutputResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.includes(`eval/subgraph`)
        ) {
            return mockSubgraphResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.includes(`eval/topk_feature`)
        ) {
            return mockTopKFeatureResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.includes(`eval/category_node_pred_rfm`)
        ) {
            return mockCategoryNodePredRfResponseResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.includes(`eval/node_pred_rfm`)
        ) {
            return mockNodePredRfmResponseResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.includes(`eval/category_aggr_feature`)
        ) {
            return mockCategoryAggrFeatureResponse();
        }

        if (url.startsWith(`v0/training_jobs/`) && url.endsWith(`/sdk`)) {
            return mockSDKTrainingCodeModalResponses();
        }
        if (
            url.startsWith(`v0/training_jobs/`) &&
            url.endsWith(`/get_train_table_split_stats`)
        ) {
            return mockSplitStatsResponse();
        }
        if (
            url.startsWith(`v0/training_jobs/`) &&
            url.endsWith(`eval/binary_classification`)
        ) {
            return mockBinaryClassificationResponses();
        }
        if (
            url.startsWith(`v0/training_jobs/`) &&
            url.endsWith(`eval/regression_heatmap`)
        ) {
            const parts = url.split("/");
            const trainingJobId = parts[2];

            return mockRegressionHeatMapResponses(trainingJobId);
        }
        if (
            url.startsWith(`v0/training_jobs`) &&
            url.endsWith(`eval/regression_histogram`)
        ) {
            return mockHistogramResponses();
        }
        if (
            url.startsWith(`v0/tensorboard/`) &&
            url.endsWith(`/plugin/scalars/tags`)
        ) {
            return mockTrialTagsResponse();
        }
        if (
            url.startsWith(`v0/training_jobs`) ||
            url.startsWith(`v0/tensorboard/`) /*  &&
            url.endsWith("/eval/available_xai_entities") */
        ) {
            /* const id = url
                .split("v0/training_jobs/")
                .pop()
                .replace("/workflow", ""); */
            return mockEmptyResponse();
        }

        if (
            url.startsWith(`v1/predictive_queries/`) &&
            url.endsWith(`/get_allowed_bp_query_overrides`)
        ) {
            return mockBpOverrideFilterResponse();
        }
        if (
            url.startsWith(`v1/predictive_queries/`) &&
            url.endsWith(`/metadata`)
        ) {
            return mockTrainingTableGenMetaDataResponse();
        }
        if (
            url.startsWith(`v1/predictive_queries/`) &&
            url.endsWith(`/encoders`)
        ) {
            return mockPredictiveQueryEncodersResponse();
        }
        if (
            url.startsWith(`v1/predictive_queries/`) &&
            url.endsWith(`/download_holdout`)
        ) {
            // Holdout download is exercised only as an error path in unit
            // tests; mirror the previously unmocked v0 behavior by rejecting.
            return Promise.reject(new Error(`UnSupported request: ${url}`));
        }
        if (url.startsWith(`v1/predictive_queries/`)) {
            return mockPredictiveQueryResoureResponse();
        }
        if (url.startsWith(`v0/snapshots/graph/`) && url.endsWith("stats")) {
            return mockGraphSnapshotTotalRows();
        }
        if (
            url.startsWith(`v0/genpredtable_jobs/`) &&
            url.endsWith("anchor_time")
        ) {
            return mockPredAnchorTimeResponse();
        }
        if (
            url.startsWith(`v0/genpredtable_jobs/`) &&
            url.endsWith("entities")
        ) {
            return mockPredictionTotalEntities();
        }
        if (url.startsWith(`v0/genpredtable_jobs/`)) {
            const id = url.split("v0/genpredtable_jobs/").pop();
            return mockGenPredictionTableJobResponse(id);
        }
        if (
            url.startsWith(`v0/gentraintable_jobs/gen-traintable-job-`) &&
            url.endsWith(`/generation_stats`)
        ) {
            return mockGenerationStatsResponse();
        }
        if (
            url.startsWith(`v0/gentraintable_jobs/`) &&
            url.endsWith(`/cancel`)
        ) {
            return mockCancelTrainingTable();
        }

        if (url.startsWith(`v0/gentraintable_jobs/gen-traintable-job-`)) {
            const id = url.split("v0/gentraintable_jobs/").pop();
            return mockTrainingTableGenResponse(id);
        }
        if (
            url.startsWith(`v0/predictive_queries/`) &&
            url.endsWith(`/metadata`)
        ) {
            return mockTrainingTableGenMetaDataResponse();
        }
        if (
            url.startsWith(`v0/predictive_queries/`) &&
            url.endsWith(`/encoders`)
        ) {
            return mockEncodersDefaultResponse();
        }
        return Promise.reject(new Error(`UnSupported request: ${url}`));
    }),
    put: vi.fn((url, payload, options) => {
        if (url.startsWith("http")) {
            options?.onUploadProgress?.({ loaded: 1, total: 1 });
            return new Promise((resolve) => {
                setTimeout(() => {
                    resolve(createMockResponse({ etag: "mock-etag" }));
                }, 25);
            });
        }
        if (/^v1\/groups\/[^/]+\/external-mappings$/.test(url)) {
            const groupId = url.split("/")[2];
            return Promise.resolve(
                createMockResponse({
                    mappings: payload.mappings.map(
                        (
                            mapping: {
                                identity_provider_id: string;
                                external_group_id: string;
                            },
                            index: number
                        ) => ({
                            id: `external-mapping-replaced-${index}`,
                            group_id: groupId,
                            ...mapping
                        })
                    )
                })
            );
        }
        if (/^v1\/groups\/[^/]+\/members$/.test(url)) {
            const groupId = url.split("/")[2];
            return Promise.resolve(
                createMockResponse({
                    members: payload.members.map(
                        (
                            member: {
                                principal_id: string;
                                principal_type: string;
                            },
                            index: number
                        ) => ({
                            id: `group-member-replaced-${index}`,
                            group_id: groupId,
                            source: "manual",
                            ...member
                        })
                    )
                })
            );
        }
        return Promise.reject(new Error(`UnSupported request: ${url}`));
    }),
    patch: vi.fn((url) => {
        if (url.startsWith("v0/admin/users/")) {
            return Promise.resolve(
                createMockResponse(mockDirectConnectUsers[0])
            );
        }
        if (url.startsWith("v0/groups/")) {
            return mockEditGroupResponse();
        }
        if (/^v0\/projects\/[^/]+\/connectors\/[^/]+$/.test(url)) {
            return mockCreateConnectorResponse();
        }
        if (/^v0\/projects\/[^/]+$/.test(url)) {
            return mockUpdateProjectResponse();
        }
        return Promise.reject(new Error(`UnSupported request: ${url}`));
    }),
    delete: vi.fn((url) => {
        if (/^v0\/groups\/[^/]+\/external-mappings\/[^/]+$/.test(url)) {
            return Promise.resolve(createMockResponse({}));
        }
        if (/^v0\/groups\/[^/]+\/members\/.+$/.test(url)) {
            return Promise.resolve(createMockResponse({}));
        }
        if (url.startsWith("v0/api-keys/admin/")) {
            return mockDeleteAdminApiKeyResponse();
        }
        if (url.startsWith("v0/api-keys/")) {
            return mockDeleteUserApiKeyResponse();
        }
        if (/^v0\/projects\/[^/]+\/connectors\/[^/]+$/.test(url)) {
            return mockDeleteConnector();
        }
        if (url.startsWith("v1/graphs/")) {
            return mockDeleteGraphResponse({} as Graph);
        }
        if (url.startsWith("v1/tables/")) {
            return mockDeleteTable();
        }
        return Promise.reject(new Error(`UnSupported request: ${url}`));
    }),
    create: vi.fn(() => mockAxiosInstance)
};

vi.doMock("axios", async (importOriginal) => {
    const actual = await importOriginal<typeof import("axios")>();
    const createMockedAxiosInstance = vi.fn(
        (...args: Parameters<typeof actual.default.create>) => {
            const instance = actual.default.create(...args);
            const originalPut = instance.put.bind(instance);
            const createConfig = args[0] as
                | {
                      onUploadProgress?: (progressEvent: unknown) => void;
                  }
                | undefined;

            instance.put = vi.fn(
                (...putArgs: Parameters<typeof instance.put>) => {
                    const [url, _payload, requestConfig] = putArgs;
                    if (typeof url === "string" && url.startsWith("http")) {
                        const uploadConfig = (requestConfig ?? createConfig) as
                            | {
                                  onUploadProgress?: (
                                      progressEvent: unknown
                                  ) => void;
                              }
                            | undefined;
                        return mockAxiosInstance.put(
                            url,
                            _payload,
                            uploadConfig
                        );
                    }
                    return originalPut(...putArgs);
                }
            ) as typeof instance.put;

            return instance;
        }
    );
    const mockedAxios = Object.assign(
        ((...args: Parameters<typeof actual.default>) =>
            actual.default(...args)) as typeof actual.default,
        actual.default,
        {
            create: createMockedAxiosInstance
        }
    );

    return {
        ...actual,
        default: mockedAxios,
        create: createMockedAxiosInstance
    };
});

vi.mock("services/axios_client", () => ({
    default: mockAxiosInstance,
    getBaseURL: vi.fn(() => "/"),
    setDirectConnectUnauthorizedHandler: vi.fn(),
    create: vi.fn(() => mockAxiosInstance),
    cancelRequest: vi.fn()
}));

type HttpMethod = "get" | "post" | "put" | "patch" | "delete";

const customClient = vi.fn(({ url, method, data, params, headers }) => {
    const axiosMethod = (method as string).toLowerCase() as HttpMethod;
    if (axiosMethod in mockAxiosInstance) {
        const fn = mockAxiosInstance[axiosMethod] as (
            ...args: unknown[]
        ) => unknown;

        const apiUrl = url.startsWith(`/`) ? url.slice(1) : url;

        if (data || params) {
            const options: Record<string, unknown> = { headers };
            if (params !== undefined) {
                options.params = params;
            }
            return fn(apiUrl, data, options);
        }
        return fn(apiUrl);
    }

    return Promise.reject(new Error(`Unsupported method: ${method}`));
});

vi.mock("services/orval_axios_client", () => ({
    customClient: customClient
}));

function wrapMockMethod(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    mockAxiosInstance: any,
    method: HttpMethod,
    writeToFile: boolean = false
) {
    const LOG_FILE = path.resolve("mock_log.txt");

    const original = mockAxiosInstance[method];

    mockAxiosInstance[method] = vi.fn(async (...args) => {
        const [url, payload, options] = args;
        const header = `${"=".repeat(100)}\n${method.toUpperCase()}: ${url}`;
        const body = [
            `Payload: ${JSON.stringify(payload, null, 2)}`,
            `Options: ${JSON.stringify(options, null, 2)}`
        ].join("\n");

        const response = original(...args);
        let result;
        try {
            result = await response;
        } catch (err) {
            result = {
                error: err instanceof Error ? err.message : String(err)
            };
        }

        const log = [
            header,
            body,
            `Mock Response: ${JSON.stringify(result, null, 2)}`,
            "\n\n"
        ].join("\n");

        if (writeToFile) {
            try {
                fs.appendFileSync(LOG_FILE, log);
            } catch (err) {
                console.error("Failed to write log file:", err);
            }
        }

        return response;
    });
}

(["get", "post", "put", "patch", "delete"] as HttpMethod[]).forEach((method) =>
    wrapMockMethod(mockAxiosInstance, method)
);
