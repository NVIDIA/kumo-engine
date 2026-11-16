// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared contract for the schema-validation-error event. Lives in utils/ so the
 * emitter (orval axios client, services/) and the listener (ValidationSchemaToast,
 * components/) both depend on it without depending on each other.
 */
export const VALIDATION_SCHEMA_ERROR_EVENT = "validation-schema-error";

export interface ValidationSchemaErrorEvent {
    method: string;
    url: string;
    apiResponse: string;
    validationErrors: string;
    apiPayload?: string;
}
