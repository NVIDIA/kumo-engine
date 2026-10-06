// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { IdentityResponse } from "gen/v0/types";

export const isAmbientS3Identity = (
    identity: IdentityResponse | undefined
): boolean =>
    identity?.config.type === "s3" &&
    identity.config.authentication.mode === "ambient";

export const isAmbientGoogleIdentity = (
    identity: IdentityResponse | undefined
): boolean =>
    (identity?.config.type === "bigquery" || identity?.config.type === "gcs") &&
    identity?.config.authentication.mode === "ambient";
