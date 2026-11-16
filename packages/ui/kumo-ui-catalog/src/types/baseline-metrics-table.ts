// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type Metric = {
    [key: string]: number | string | undefined;
    name?: string;
    val?: number;
};
