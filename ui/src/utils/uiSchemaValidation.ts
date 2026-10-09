// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Mirror of debugStore's ENABLE_UI_SCHEMA_VALIDATION feature flag.
 *
 * The orval axios client (services/) gates Zod response validation on this
 * flag but must not import the store. debugStore keeps it in sync (see the
 * subscribe in stores/debugStore). Non-reactive read is sufficient — it's
 * checked per-request at call time.
 */
let enabled = false;

export const setUiSchemaValidationEnabled = (value: boolean): void => {
    enabled = value;
};

export const isUiSchemaValidationEnabled = (): boolean => enabled;
