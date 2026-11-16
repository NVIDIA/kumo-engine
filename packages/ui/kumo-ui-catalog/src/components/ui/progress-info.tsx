// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

export interface ProgressInfoProps {
    info: Array<{
        row1: React.ReactNode | string;
        row2: React.ReactNode | string;
    }>;
}

export const ProgressInfo: React.FC<ProgressInfoProps> = ({ info }) => {
    return (
        <div
            className="font-inter flex gap-8"
            data-testid="progress-info-container">
            {info?.map((value, index) => (
                <div
                    key={`progress-info-${index}`}
                    className="flex flex-col gap-2">
                    <div className="flex items-center gap-1">{value.row1}</div>
                    {value.row2}
                </div>
            ))}
        </div>
    );
};
