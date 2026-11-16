// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDebugStore } from "stores/debugStore";
import { deploymentType } from "tests/testType";

import {
    setDeploymentType,
    updateFeatureFlags
} from "./MockFunctions/DebugInfo.mock";
import { debug_info } from "./MockResponse/DebugInfo";

export const setDeploymentTypeInStore = (type: deploymentType) => {
    setDeploymentType(type);

    useDebugStore.setState((state) => ({
        ...state,
        data: {
            ...debug_info,
            feature_flags: updateFeatureFlags(debug_info.feature_flags)
        },
        configFetched: true
    }));
};
