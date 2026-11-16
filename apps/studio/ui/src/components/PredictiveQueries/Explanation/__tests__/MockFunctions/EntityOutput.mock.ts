// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AxiosResponse } from "axios";
import { createMockResponse } from "tests/testUtil";

export const mockEntityOutputResponse = (): Promise<AxiosResponse<unknown>> => {
    return Promise.resolve(
        createMockResponse({
            task_type: "binary_classification"
        }) as AxiosResponse<unknown>
    );
};
