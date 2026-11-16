// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Database } from "@/icons";

export const ConnectorChip = ({ type }: { type: string }) => {
    let className = "";
    let label = "";
    switch (type) {
        case "databricks":
            className = "bg-orange-100 text-orange-700 [&_svg]:fill-orange-700";
            label = "Databricks";
            break;
        case "snowflake":
            className = "bg-cyan-100 text-cyan-700 [&_svg]:fill-cyan-700";
            label = "Snowflake";
            break;
        case "file":
        case "s3":
            className = "bg-red-100 text-red-700 [&_svg]:fill-red-700";
            label = "S3";
            break;
        case "bigquery":
            className = "bg-blue-100 text-blue-700 [&_svg]:fill-blue-700";
            label = "BigQuery";
            break;
        case "gcs":
            className = "bg-green-100 text-green-700 [&_svg]:fill-green-700";
            label = "GCS";
            break;
        case "adls":
            className = "bg-violet-100 text-violet-700 [&_svg]:fill-violet-700";
            label = "ADLS";
            break;
        default:
            return null;
    }

    return (
        <div
            className={
                "flex w-fit items-center gap-2 rounded-sm p-1 px-2 leading-5 " +
                className
            }>
            <Database className="size-5" /> {label}
        </div>
    );
};
