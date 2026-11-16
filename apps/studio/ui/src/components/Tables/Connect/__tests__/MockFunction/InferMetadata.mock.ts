// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import {
    Column,
    TableMetadataRequest,
    TableMetadataResponse
} from "gen/v1/types";
import { createMockResponse } from "tests/testUtil";

// Columns returned by the mocked stats-based inference endpoint
// (`POST /v1/tables/infer_metadata`). `null` (the default) answers with an
// empty column list, which leaves the wizard's stats-free preview stypes
// untouched, so suites that do not opt in are unaffected.
let currentInferredCols: Column[] | null = null;
let respondWithError = false;
let heldRequests: HeldInferMetadataRequest[] | null = null;

export interface HeldInferMetadataRequest {
    payload?: TableMetadataRequest;
    release: (cols?: Column[] | null) => void;
}

export const setInferMetadataResponseCols = (cols: Column[] | null = null) => {
    currentInferredCols = cols;
    respondWithError = false;
    heldRequests = null;
};

export const setInferMetadataResponseError = () => {
    respondWithError = true;
};

// Switches the mock to manual resolution: each request is parked in the
// returned list until the test releases it with the columns of its choice,
// so orderings (samples before stats, responses across source switches) can
// be exercised deterministically.
export const holdInferMetadataResponses = (): HeldInferMetadataRequest[] => {
    heldRequests = [];
    return heldRequests;
};

const buildInferMetadataResponse = (
    cols: Column[] | null,
    payload?: TableMetadataRequest
): AxiosResponse<TableMetadataResponse> =>
    createMockResponse({
        cols: structuredClone(cols ?? []),
        source_table: payload?.source_table,
        pkey: null,
        time_col: null,
        end_time_col: null
    }) as unknown as AxiosResponse<TableMetadataResponse>;

export const mockInferMetadataResponse = (
    payload?: TableMetadataRequest
): Promise<AxiosResponse<TableMetadataResponse>> => {
    if (respondWithError) {
        return Promise.reject(
            new Error("Stats-based metadata inference unavailable")
        );
    }
    if (heldRequests) {
        const held = heldRequests;
        return new Promise((resolve) => {
            held.push({
                payload,
                release: (cols: Column[] | null = null) =>
                    resolve(buildInferMetadataResponse(cols, payload))
            });
        });
    }
    return Promise.resolve(
        buildInferMetadataResponse(currentInferredCols, payload)
    );
};
