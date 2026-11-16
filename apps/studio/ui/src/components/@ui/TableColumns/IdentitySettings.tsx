// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    ColumnDef,
    Row,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import type { IdentityResponse } from "gen/v0/types";

import {
    isAmbientGoogleIdentity,
    isAmbientS3Identity
} from "utils/identityUtils";

const PROVIDER_LABELS: Record<string, string> = {
    s3: "S3",
    databricks: "Databricks",
    snowflake: "Snowflake",
    bigquery: "BigQuery",
    gcs: "GCS",
    adls: "ADLS"
};

const identityAuthenticationLabel = (identity: IdentityResponse): string => {
    if (isAmbientS3Identity(identity))
        return "Deployment-managed ambient identity";
    if (isAmbientGoogleIdentity(identity)) return "Ambient Google credentials";
    const config = identity.config;
    if (config.type === "bigquery" || config.type === "gcs") {
        if ("service_account_email" in config.authentication)
            return "Workload identity";
        return identity.credential_status.configured
            ? "Service-account key configured"
            : "Service-account key missing";
    }
    return identity.credential_status.configured
        ? "Stored material configured"
        : "Workload identity";
};

export const identitySettingsColumns = (
    onManage: (identity: IdentityResponse) => void,
    onDelete: (id: string) => void
): ColumnDef<IdentityResponse>[] => [
    {
        id: "identity",
        header: "Identity",
        accessorKey: "display_name",
        enableSorting: true,
        cell: ({ row }: { row: Row<IdentityResponse> }) => {
            const identity = row.original;
            return (
                <div>
                    <div className="font-medium text-title-primary">
                        {identity.display_name}
                    </div>
                    {identity.description && (
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div className="mt-1.5 max-w-sm truncate text-sm text-title-secondary">
                                    {identity.description}
                                </div>
                            </TooltipTrigger>
                            <TooltipContent arrow={false}>
                                {identity.description}
                            </TooltipContent>
                        </Tooltip>
                    )}
                </div>
            );
        }
    },
    {
        id: "provider",
        header: "Provider",
        accessorKey: "config.type",
        enableSorting: true,
        cell: ({ row }: { row: Row<IdentityResponse> }) => {
            const type = row.original.config.type ?? "";
            return <span>{PROVIDER_LABELS[type] ?? type}</span>;
        }
    },
    {
        id: "authentication",
        header: "Authentication",
        accessorKey: "credential_status.configured",
        enableSorting: false,
        cell: ({ row }: { row: Row<IdentityResponse> }) =>
            identityAuthenticationLabel(row.original)
    },
    {
        id: "actions",
        header: "Actions",
        enableSorting: false,
        accessorKey: "id",
        cell: ({ row }: { row: Row<IdentityResponse> }) => {
            const identity = row.original;
            return (
                <div
                    className="flex gap-2"
                    onClick={(e) => e.stopPropagation()}>
                    <Button
                        variant="outline"
                        onClick={() => onManage(identity)}>
                        Manage
                    </Button>
                    {!isAmbientS3Identity(identity) && (
                        <Button
                            variant="outline"
                            onClick={() => onDelete(identity.id)}>
                            Delete
                        </Button>
                    )}
                </div>
            );
        }
    }
];
