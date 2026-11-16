// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FeatureFlags } from "gen/v0/types";
import { useDebugStore } from "stores/debugStore";

const useFeatureFlags = (): FeatureFlags => {
    const { data } = useDebugStore();
    return data.feature_flags;
};

export default useFeatureFlags;
