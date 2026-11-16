// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

type ApiBaseUrlOptions = {
    apiTarget?: string;
    developmentBase?: string;
    mode?: string;
    productionBase?: string;
    testBase?: string;
};

/** Resolve the browser REST base without coupling callers to build tooling. */
export const resolveApiBaseUrl = ({
    apiTarget,
    developmentBase,
    mode,
    productionBase,
    testBase = "http://localhost:10001"
}: ApiBaseUrlOptions) => {
    if (apiTarget) return apiTarget;
    if (mode === "test") return testBase;
    if (mode === "development") return developmentBase;
    return productionBase;
};
