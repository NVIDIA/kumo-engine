// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

type SymbolProps = {
    type: "emdash" | "dot";
    className?: string;
};

export const Symbol: React.FC<SymbolProps> = ({ type, className = "" }) => {
    return type === "emdash" ? (
        <div className={`text-slate-500 ${className}`}>&mdash;</div>
    ) : (
        <div className={`text-slate-500 ${className}`}>&bull;</div>
    );
};
