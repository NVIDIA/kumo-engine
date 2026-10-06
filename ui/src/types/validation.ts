// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CustomError } from "./common";

export type BannerData = {
    title: string;
    subheader?: string;
    warnings: Array<string>;
    isWarning?: boolean;
};
export type AlertData = {
    title: string;
    description: string;
};
export type HandleErrorFunc = (
    err: CustomError,
    title?: string,
    isWarning?: boolean
) => void;
