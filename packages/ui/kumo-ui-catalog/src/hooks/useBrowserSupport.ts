// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const useIsBrowserSupported = () => {
    const userAgent = navigator?.userAgent;
    if (userAgent?.indexOf("Chrome") > -1) {
        return true;
    }
    return false;
};
