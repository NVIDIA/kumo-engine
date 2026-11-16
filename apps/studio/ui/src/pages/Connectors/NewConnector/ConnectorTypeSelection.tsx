// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Check, cn } from "@kumo-ai/kumo-ui-catalog";

import AzureLogo from "images/azure-logo.png";
import BigQueryLogo from "images/bigquerylogo.png";
import DatabricksLogo from "images/databrickslogo.png";
import GCSLogo from "images/gcslogo.png";
import S3Logo from "images/s3logo.png";
import SnowflakeLogo from "images/snowflakelogo.png";

export type ConnectorType =
    | "databricks"
    | "snowflake"
    | "bigquery"
    | "s3"
    | "gcs"
    | "adls";

type ConnectorTypeOption = {
    value: ConnectorType;
    label: string;
    icon: string;
    iconClassName?: string;
    testId: string;
};

export const CONNECTOR_TYPES: ConnectorTypeOption[] = [
    {
        value: "databricks",
        label: "Databricks",
        icon: DatabricksLogo,
        testId: "connector-type-databricks"
    },
    {
        value: "snowflake",
        label: "Snowflake",
        icon: SnowflakeLogo,
        testId: "connector-type-snowflake"
    },
    {
        value: "bigquery",
        label: "BigQuery",
        icon: BigQueryLogo,
        testId: "connector-type-bigquery"
    },
    {
        value: "s3",
        label: "Amazon S3",
        icon: S3Logo,
        testId: "connector-type-s3"
    },
    {
        value: "gcs",
        label: "Google Cloud Storage",
        icon: GCSLogo,
        testId: "connector-type-gcs"
    },
    {
        value: "adls",
        label: "Azure Data Lake Storage",
        icon: AzureLogo,
        iconClassName: "object-cover object-left",
        testId: "connector-type-adls"
    }
];

interface ConnectorTypeSelectionProps {
    value: ConnectorType | "";
    onChange: (value: ConnectorType) => void;
    disabled?: boolean;
}

const ConnectorTypeSelection = ({
    value,
    onChange,
    disabled = false
}: ConnectorTypeSelectionProps) => (
    <fieldset disabled={disabled} data-testid="connector-type-selection">
        {/* The visible heading is the section header rendered by the parent
            form; the legend stays for the fieldset's accessible name. */}
        <legend className="sr-only">Connector type</legend>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {CONNECTOR_TYPES.map((option) => {
                const selected = value === option.value;
                return (
                    <label
                        key={option.value}
                        className={cn(
                            "relative flex min-h-20 items-center gap-3 rounded-sm border px-4 py-3 focus-within:ring-2 focus-within:ring-border-primary",
                            disabled
                                ? "cursor-not-allowed bg-surface-grey text-title-secondary opacity-60"
                                : "cursor-pointer hover:border-border-primary hover:bg-surface-grey",
                            selected
                                ? "border-border-primary bg-surface-grey"
                                : "border-border-secondary bg-white"
                        )}>
                        <input
                            type="radio"
                            name="connector-type"
                            value={option.value}
                            checked={selected}
                            disabled={disabled}
                            data-testid={option.testId}
                            className="sr-only"
                            onChange={() => onChange(option.value)}
                        />
                        <img
                            src={option.icon}
                            alt=""
                            className={cn(
                                "size-9 shrink-0",
                                option.iconClassName ?? "object-contain"
                            )}
                        />
                        <span className="font-medium text-title-primary">
                            {option.label}
                        </span>
                        {selected && (
                            <span className="ml-auto flex items-center gap-1 text-sm font-medium text-title-primary">
                                <Check className="size-4" aria-hidden="true" />
                                Selected
                            </span>
                        )}
                    </label>
                );
            })}
        </div>
    </fieldset>
);

export default ConnectorTypeSelection;
