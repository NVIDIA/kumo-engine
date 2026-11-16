// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { DebugInfo, FeatureFlags } from "gen/v0/types";
import { deploymentType } from "tests/testType";
import { createMockResponse } from "tests/testUtil";

import { copyObject } from "utils/Helpers/objectHelpers";

import { debug_info } from "../MockResponse/DebugInfo";

let currentDeploymentType = deploymentType.SAAS;

export const mockDebugInfoResponse = (): Promise<AxiosResponse<DebugInfo>> => {
    return debugInfoResponse();
};

export const updateFeatureFlags = (
    featureFlags: FeatureFlags
): FeatureFlags => {
    switch (currentDeploymentType) {
        case deploymentType.SAAS:
            return {
                ...featureFlags,
                IS_DATABRICKS_DEPLOY: false
            };
        case deploymentType.Databricks:
            return {
                ...featureFlags,
                IS_DATABRICKS_DEPLOY: true
            };
        default:
            return featureFlags;
    }
};

let debugInfoOverride: Partial<DebugInfo> | null = null;

const debugInfoResponse = (): Promise<AxiosResponse<DebugInfo>> => {
    const mockResp = {
        ...debug_info,
        ...(debugInfoOverride || {}),
        ...{
            feature_flags: updateFeatureFlags(
                copyObject(debug_info.feature_flags)
            )
        }
    };
    return Promise.resolve(
        createMockResponse(mockResp) as unknown as AxiosResponse<DebugInfo>
    );
};

export const setDebugInfoOverride = (override: Partial<DebugInfo> | null) => {
    debugInfoOverride = override;
};

export const setDeploymentType = (type: deploymentType) => {
    currentDeploymentType = type;
};

export const getDeplomentType = () => {
    return currentDeploymentType;
};
