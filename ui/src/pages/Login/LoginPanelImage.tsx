// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@kumo-ai/kumo-ui-catalog";

import loginPanelImage from "images/login-panel-final.webp";

const LoginPanelImage = () => (
    <div className={cn("relative !h-64 overflow-hidden sm:!h-80 lg:!h-full")}>
        <img
            src={loginPanelImage}
            alt=""
            aria-hidden="true"
            className="absolute inset-0 h-full w-full object-cover object-center"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/10 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 !px-6 pb-8 sm:!px-10 lg:!px-14 lg:pb-16">
            <p className="mb-2 text-3xl font-bold leading-tight tracking-tight text-title-white">
                Predictive AI on your data
            </p>
            <p className="text-sm font-medium text-title-white/80">
                Accelerated by{" "}
                <span className="font-semibold text-brand-green">NVIDIA</span>
            </p>
        </div>
    </div>
);

export default LoginPanelImage;
