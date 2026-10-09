// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect } from "react";

import { Bouncingball } from "@kumo-ai/kumo-ui-catalog";

import { clearLocalStorage } from "utils/storage";

const ClearState = () => {
    useEffect(() => {
        clearLocalStorage();
        setTimeout(() => {
            window.location.href = window.location.origin;
        }, 1 * 1000);
    });

    return <Bouncingball overlay={false} className="h-25" />;
};

export default ClearState;
