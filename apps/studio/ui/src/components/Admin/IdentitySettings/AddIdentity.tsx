// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";

import type { FileUploadItem } from "@kumo-ai/kumo-ui-catalog";
import {
    Alert,
    AlertDescription,
    Button,
    Checkbox,
    Label,
    RadioGroup,
    RadioGroupItem,
    Select,
    Spinner,
    TextArea,
    TextInput,
    Title,
    Upload
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { createIdentities } from "gen/v0/api/identities/identities";
import {
    AzureCloud,
    CreateIdentityArgs,
    SnowflakeAuthenticationKind
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { ResourceSlugField } from "components/ResourceSlug";

import { useKumoGroups } from "hooks/groups/groups";
import useFeatureFlags from "hooks/utils/featureFlags";

import { CustomError } from "types/common";

import { navRoutes } from "utils/Constants";
import { parseErrorMessage } from "utils/errorHandler";
import { parseGoogleServiceAccount } from "utils/googleServiceAccount";
import { invalidateIdentityLists } from "utils/queryInvalidation";
import { resourceSlugError } from "utils/resourceSlug";

type IdentityKind =
    | "databricks"
    | "snowflake"
    | "bigquery"
    | "s3"
    | "gcs"
    | "adls";

type GoogleAuthenticationMode =
    | "workload_identity"
    | "service_account_key"
    | "ambient";

type S3AuthenticationMode = "assume_role" | "access_key";

const googleKeyFields = [
    "project_id",
    "private_key_id",
    "private_key",
    "client_email",
    "client_id",
    "auth_uri",
    "token_uri"
] as const;

const requiredFields: Record<IdentityKind, string[]> = {
    databricks: ["display_name", "host", "client_id", "client_secret"],
    snowflake: ["display_name", "account", "username"],
    bigquery: ["display_name"],
    s3: ["display_name"],
    gcs: ["display_name"],
    adls: ["display_name", "tenant_id", "client_id", "cloud"]
};

const providerDefaults: Record<IdentityKind, Record<string, string>> = {
    databricks: {},
    snowflake: {},
    bigquery: {},
    s3: { partition: "aws" },
    gcs: {},
    adls: { cloud: "public" }
};

type IdentityPayloadCommon = Omit<CreateIdentityArgs, "config" | "credentials">;

const snowflakeCredentials = (
    authentication: SnowflakeAuthenticationKind,
    values: Record<string, string>
) =>
    authentication === "key_pair"
        ? {
              kind: "snowflake_key_pair" as const,
              private_key: values.private_key,
              private_key_passphrase: values.private_key_passphrase || null
          }
        : {
              kind: "snowflake_password" as const,
              password: values.password
          };

const googleAuthentication = (
    mode: GoogleAuthenticationMode,
    values: Record<string, string>
) => {
    if (mode === "ambient") {
        // Ambient: Kumo authenticates with its own GCP Workload Identity
        // service account. No fields, no credentials - the customer grants that
        // service account IAM access to their bucket/dataset.
        return { mode: "ambient" as const };
    }
    if (mode === "workload_identity") {
        return {
            mode: "workload_identity" as const,
            service_account_email: values.service_account_email.trim()
        };
    }
    return {
        mode: "service_account_key" as const,
        project_id: values.project_id.trim(),
        client_email: values.client_email.trim(),
        client_id: values.client_id.trim(),
        auth_uri: values.auth_uri.trim(),
        token_uri: values.token_uri.trim(),
        universe_domain: values.universe_domain?.trim() || "googleapis.com"
    };
};

const googleCredentials = (
    mode: GoogleAuthenticationMode,
    values: Record<string, string>
) =>
    mode === "service_account_key"
        ? {
              kind: "google_service_account_key" as const,
              private_key_id: values.private_key_id.trim(),
              private_key: values.private_key
          }
        : undefined;

const s3Credentials = (
    mode: S3AuthenticationMode,
    values: Record<string, string>
) => {
    if (mode === "access_key") {
        return {
            kind: "s3_access_key" as const,
            access_key_id: values.access_key_id,
            secret_access_key: values.secret_access_key,
            session_token: values.session_token || null,
            ca_certificate: values.ca_certificate || null
        };
    }
    return values.external_id
        ? {
              kind: "s3_assume_role" as const,
              external_id: values.external_id
          }
        : undefined;
};

const buildIdentityPayload = (
    identityKind: IdentityKind,
    common: IdentityPayloadCommon,
    values: Record<string, string>,
    authentication: SnowflakeAuthenticationKind,
    googleAuthenticationMode: GoogleAuthenticationMode,
    s3AuthenticationMode: S3AuthenticationMode
): CreateIdentityArgs => {
    switch (identityKind) {
        case "databricks":
            return {
                ...common,
                config: {
                    type: "databricks",
                    host: values.host.trim(),
                    client_id: values.client_id.trim()
                },
                credentials: {
                    kind: "databricks_oauth",
                    client_secret: values.client_secret
                }
            };
        case "snowflake":
            return {
                ...common,
                config: {
                    type: "snowflake",
                    account: values.account.trim(),
                    username: values.username.trim(),
                    authentication
                },
                credentials: snowflakeCredentials(authentication, values)
            };
        case "bigquery":
            return {
                ...common,
                config: {
                    type: "bigquery",
                    authentication: googleAuthentication(
                        googleAuthenticationMode,
                        values
                    )
                },
                credentials: googleCredentials(googleAuthenticationMode, values)
            };
        case "s3":
            if (s3AuthenticationMode === "access_key") {
                return {
                    ...common,
                    config: {
                        type: "s3",
                        authentication: { mode: "access_key" },
                        connection: {
                            endpoint_url: values.endpoint_url.trim(),
                            region: values.region.trim(),
                            addressing_style:
                                (values.addressing_style as
                                    | "auto"
                                    | "path"
                                    | "virtual") || "path"
                        }
                    },
                    credentials: s3Credentials(s3AuthenticationMode, values)
                };
            }
            return {
                ...common,
                config: {
                    type: "s3",
                    authentication: {
                        mode: "assume_role",
                        partition: values.partition.trim(),
                        role_arn: values.role_arn.trim()
                    }
                },
                credentials: s3Credentials(s3AuthenticationMode, values)
            };
        case "gcs":
            return {
                ...common,
                config: {
                    type: "gcs",
                    authentication: googleAuthentication(
                        googleAuthenticationMode,
                        values
                    )
                },
                credentials: googleCredentials(googleAuthenticationMode, values)
            };
        case "adls":
            return {
                ...common,
                config: {
                    type: "adls",
                    tenant_id: values.tenant_id.trim(),
                    client_id: values.client_id.trim(),
                    cloud: values.cloud as AzureCloud
                }
            };
    }
};

type GoogleAuthenticationOption = {
    value: GoogleAuthenticationMode;
    title: string;
    description: ReactNode;
    available: boolean;
    // Why this deployment cannot offer the option. Rendered next to the option
    // it explains so a disabled control is never left unexplained.
    unavailableReason?: string;
};

const googleAuthenticationOptions = ({
    identityKind,
    workloadExchangeAvailable,
    ambientAvailable,
    developerCredentials
}: Pick<
    GoogleAuthenticationFormProps,
    | "identityKind"
    | "workloadExchangeAvailable"
    | "ambientAvailable"
    | "developerCredentials"
>): GoogleAuthenticationOption[] => {
    // Neither keyless mode is a product setting: ambient is on when the deploy
    // config carries an `ambient_gcp` allowlist, and workload exchange is on
    // when KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED is set. Name the actual switch so
    // the reader knows what to change rather than who to ask.
    return [
        {
            value: "ambient",
            title: developerCredentials
                ? "Ambient local Google credentials"
                : "Use Kumo's GCP identity",
            description: developerCredentials
                ? "Uses your local gcloud credentials. Nothing is uploaded."
                : `Kumo uses its own Google service account. You grant it read access to your ${
                      identityKind === "bigquery" ? "dataset" : "bucket"
                  }.`,
            available: ambientAvailable,
            unavailableReason:
                "Enable by adding an approved service account to ambient_gcp in your deployment config."
        },
        {
            value: "workload_identity",
            title: "Workload identity (impersonation)",
            description:
                "Kumo borrows a service account you nominate. Nothing is uploaded or stored.",
            available: workloadExchangeAvailable,
            unavailableReason:
                "Enable by setting KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED in your deployment config."
        },
        {
            value: "service_account_key",
            title: "Service account JSON key",
            // "Prefer the options above" is only advice when one of them can
            // actually be picked; on a deployment offering neither it reads as
            // a dead end. Point at rotation instead.
            description: `You upload a key file from Google. It never expires, so ${
                ambientAvailable || workloadExchangeAvailable
                    ? "prefer the options above"
                    : "rotate it periodically"
            }.`,
            available: true
        }
    ];
};

type GoogleAuthenticationFormProps = {
    identityKind: IdentityKind;
    mode: GoogleAuthenticationMode;
    workloadExchangeAvailable: boolean;
    ambientAvailable: boolean;
    developerCredentials: boolean;
    serviceAccountEmail: string;
    values: Record<string, string>;
    fileError: string;
    keyFile: FileUploadItem | undefined;
    rowClassName: string;
    labelClassName: string;
    onModeChange: (mode: GoogleAuthenticationMode) => void;
    onKeyFileChange: (file: FileUploadItem | undefined) => void;
    renderInput: (name: string, label: string, secret?: boolean) => ReactNode;
};

const GoogleAuthenticationForm = ({
    identityKind,
    mode,
    workloadExchangeAvailable,
    ambientAvailable,
    developerCredentials,
    serviceAccountEmail,
    values,
    fileError,
    keyFile,
    rowClassName,
    labelClassName,
    onModeChange,
    onKeyFileChange,
    renderInput
}: GoogleAuthenticationFormProps) => {
    const options = googleAuthenticationOptions({
        identityKind,
        workloadExchangeAvailable,
        ambientAvailable,
        developerCredentials
    });
    // "Recommended" marks the best keyless option this deployment can actually
    // offer. It is never applied to a disabled option, and never to the key
    // upload - recommending the only long-lived credential would be wrong.
    const recommended = options.find(
        (option) => option.available && option.value !== "service_account_key"
    )?.value;

    const ambientDetail = developerCredentials ? (
        <>
            <p className="font-medium">Local development credentials</p>
            <p className="mt-1">
                Kumo uses Google user ADC or the active gcloud profile on this
                devserver host. If credentials are missing or expired, run:
            </p>
            <pre className="mt-1 overflow-x-auto rounded bg-surface-primary p-2 text-xs">
                gcloud auth login
            </pre>
            <p className="mt-2 text-title-secondary">
                Access is evaluated as your Google user. This development
                identity is not portable to a production deployment.
            </p>
        </>
    ) : (
        <>
            <p className="font-medium">
                Grant Kumo&apos;s service account access
            </p>
            <p className="mt-1">
                Service account:{" "}
                <code className="break-all">
                    {serviceAccountEmail || "(set at deployment)"}
                </code>
            </p>
            <p className="mt-2">
                Run one IAM binding on your resource, then create the identity:
            </p>
            <pre className="mt-1 overflow-x-auto rounded bg-surface-primary p-2 text-xs">
                {identityKind === "bigquery"
                    ? `gcloud projects add-iam-policy-binding YOUR_PROJECT \\\n  --member=serviceAccount:${
                          serviceAccountEmail || "KUMO_SERVICE_ACCOUNT"
                      } \\\n  --role=roles/bigquery.dataViewer`
                    : `gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \\\n  --member=serviceAccount:${
                          serviceAccountEmail || "KUMO_SERVICE_ACCOUNT"
                      } \\\n  --role=roles/storage.objectViewer`}
            </pre>
            <p className="mt-2 text-title-secondary">
                No key or secret is stored. Kumo reads your data with its own
                Workload Identity credentials.
            </p>
        </>
    );

    // Each option owns the fields it needs, so the controls read as part of the
    // choice rather than as a separate step below the group.
    const detailFor = (option: GoogleAuthenticationOption): ReactNode => {
        if (option.value === "ambient") {
            return (
                <div
                    className="rounded bg-surface-grey p-3 text-sm"
                    aria-label="Ambient GCP grant instructions">
                    {ambientDetail}
                </div>
            );
        }
        if (option.value === "workload_identity") {
            return renderInput(
                "service_account_email",
                "Target service-account email"
            );
        }
        return (
            <>
                <Upload
                    multiple={false}
                    accept="application/json,.json"
                    status={fileError ? "error" : undefined}
                    // `null`, not `undefined`: undefined hands state back to the
                    // component, which would resurface the removed file.
                    value={keyFile ?? null}
                    onValueChange={onKeyFileChange}
                    attributes={{
                        UploadInputElement: {
                            id: "google-service-account-file",
                            "aria-label": "Service-account JSON file"
                        }
                    }}
                />
                {fileError && (
                    <p className="mt-2 text-sm text-destructive">{fileError}</p>
                )}
                {values.client_email && values.project_id && (
                    // The identity detail page shows a saved key as a
                    // definition list, so a key reads the same either side of
                    // the save.
                    <dl
                        className="mt-3 grid gap-y-2 rounded bg-surface-grey p-3 text-sm"
                        aria-label="Parsed service-account details">
                        <div>
                            <dt className="text-title-secondary">
                                Service account
                            </dt>
                            <dd className="break-all">{values.client_email}</dd>
                        </div>
                        <div>
                            <dt className="text-title-secondary">Project</dt>
                            <dd className="break-all">{values.project_id}</dd>
                        </div>
                    </dl>
                )}
                <Alert variant="info" className="mt-3" enableClose={false}>
                    <AlertDescription>
                        Your private key is never shown back to you. Kumo checks
                        the file, then encrypts it before storing it.
                    </AlertDescription>
                </Alert>
            </>
        );
    };

    return (
        <div className={`${rowClassName} !items-start`}>
            <span className={labelClassName}>Authentication</span>
            <RadioGroup
                aria-label="Authentication"
                className="gap-3"
                value={mode}
                onValueChange={(next) =>
                    onModeChange(next as GoogleAuthenticationMode)
                }>
                {options.map((option) => {
                    const id = `google-authentication-${option.value}`;
                    const showReason =
                        !option.available && Boolean(option.unavailableReason);
                    return (
                        <div key={option.value}>
                            <div className="flex items-start gap-2">
                                <RadioGroupItem
                                    className="mt-0.5"
                                    value={option.value}
                                    id={id}
                                    disabled={!option.available}
                                    // Everything inside the label would
                                    // otherwise be read as the radio's name, so
                                    // a screen reader announces all three
                                    // sentences on every arrow key. Name it
                                    // from the title and demote the rest to a
                                    // description, which is announced after a
                                    // pause and can be skipped.
                                    aria-labelledby={`${id}-title`}
                                    aria-describedby={
                                        showReason
                                            ? `${id}-description ${id}-reason`
                                            : `${id}-description`
                                    }
                                />
                                <label
                                    className={`flex flex-col ${
                                        option.available
                                            ? "cursor-pointer"
                                            : "cursor-not-allowed"
                                    }`}
                                    htmlFor={id}>
                                    <span
                                        id={`${id}-title`}
                                        className="font-medium">
                                        {option.title}
                                        {recommended === option.value &&
                                            " (Recommended)"}
                                    </span>
                                    <span
                                        id={`${id}-description`}
                                        className="text-sm text-title-secondary">
                                        {option.description}
                                    </span>
                                    {showReason && (
                                        <span
                                            id={`${id}-reason`}
                                            className="mt-1 text-sm text-title-secondary">
                                            {option.unavailableReason}
                                        </span>
                                    )}
                                </label>
                            </div>
                            {mode === option.value && (
                                <div className="ml-6 mt-3 max-w-xl">
                                    {detailFor(option)}
                                </div>
                            )}
                        </div>
                    );
                })}
            </RadioGroup>
        </div>
    );
};

const providerOptions = [
    { value: "databricks", label: "Databricks" },
    { value: "snowflake", label: "Snowflake" },
    { value: "bigquery", label: "BigQuery" },
    { value: "s3", label: "Amazon S3" },
    { value: "gcs", label: "Google Cloud Storage" },
    { value: "adls", label: "Azure Data Lake Storage" }
];

const snowflakeAuthOptions = [
    { value: "password", label: "Password" },
    { value: "key_pair", label: "Key pair" }
];

type SnowflakeFieldsProps = {
    values: Record<string, string>;
    authentication: SnowflakeAuthenticationKind;
    rowClassName: string;
    labelClassName: string;
    onChangeSnowflakeAuth: (kind: SnowflakeAuthenticationKind) => void;
    onChangeValue: (name: string, value: string) => void;
};

const SnowflakeFields = ({
    values,
    authentication,
    rowClassName,
    labelClassName,
    onChangeSnowflakeAuth,
    onChangeValue
}: SnowflakeFieldsProps) => (
    <>
        <div className={rowClassName}>
            <Label htmlFor="account" className={labelClassName}>
                Account
            </Label>
            <TextInput
                id="account"
                name="account"
                type="text"
                value={values.account ?? ""}
                onChange={(e) => onChangeValue("account", e.target.value)}
            />
        </div>
        <div className={rowClassName}>
            <Label htmlFor="username" className={labelClassName}>
                Service-account username
            </Label>
            <TextInput
                id="username"
                name="username"
                type="text"
                value={values.username ?? ""}
                onChange={(e) => onChangeValue("username", e.target.value)}
            />
        </div>
        <div className={rowClassName}>
            <Label
                htmlFor="snowflake-authentication"
                className={labelClassName}>
                Authentication
            </Label>
            <Select
                id="snowflake-authentication"
                options={snowflakeAuthOptions}
                selectedValues={[authentication]}
                onValuesChange={(vals) =>
                    onChangeSnowflakeAuth(
                        vals[0] as SnowflakeAuthenticationKind
                    )
                }
                mode="single"
                search={false}
            />
        </div>
        {authentication === "password" ? (
            <div className={rowClassName}>
                <Label htmlFor="password" className={labelClassName}>
                    Password
                </Label>
                <TextInput
                    id="password"
                    name="password"
                    type="password"
                    value={values.password ?? ""}
                    onChange={(e) => onChangeValue("password", e.target.value)}
                />
            </div>
        ) : (
            <>
                <div className={rowClassName}>
                    <Label htmlFor="private_key" className={labelClassName}>
                        Private key
                    </Label>
                    <TextInput
                        id="private_key"
                        name="private_key"
                        type="password"
                        value={values.private_key ?? ""}
                        onChange={(e) =>
                            onChangeValue("private_key", e.target.value)
                        }
                    />
                </div>
                <div className={rowClassName}>
                    <Label
                        htmlFor="private_key_passphrase"
                        className={labelClassName}>
                        Passphrase (optional)
                    </Label>
                    <TextInput
                        id="private_key_passphrase"
                        name="private_key_passphrase"
                        type="password"
                        value={values.private_key_passphrase ?? ""}
                        onChange={(e) =>
                            onChangeValue(
                                "private_key_passphrase",
                                e.target.value
                            )
                        }
                    />
                </div>
            </>
        )}
    </>
);

type ProviderFieldsProps = {
    identityKind: IdentityKind;
    values: Record<string, string>;
    authentication: SnowflakeAuthenticationKind;
    googleAuthenticationForm: ReactNode;
    s3AuthenticationMode: S3AuthenticationMode;
    rowClassName: string;
    labelClassName: string;
    onChangeProvider: (kind: IdentityKind) => void;
    onChangeSnowflakeAuth: (kind: SnowflakeAuthenticationKind) => void;
    onChangeS3Auth: (mode: S3AuthenticationMode) => void;
    onChangeValue: (name: string, value: string) => void;
};

// Provider-specific branches are intentionally kept together so changing the
// selected provider cannot leave credential controls from another provider.
/* eslint-disable complexity -- each provider contributes one explicit branch */
const ProviderFields = ({
    identityKind,
    values,
    authentication,
    googleAuthenticationForm,
    s3AuthenticationMode,
    rowClassName,
    labelClassName,
    onChangeProvider,
    onChangeSnowflakeAuth,
    onChangeS3Auth,
    onChangeValue
}: ProviderFieldsProps) => (
    <>
        <div className={rowClassName}>
            <Label htmlFor="identity-provider" className={labelClassName}>
                Provider
            </Label>
            <Select
                id="identity-provider"
                options={providerOptions}
                selectedValues={[identityKind]}
                onValuesChange={(vals) =>
                    onChangeProvider(vals[0] as IdentityKind)
                }
                mode="single"
                search={false}
            />
        </div>
        {identityKind === "databricks" && (
            <>
                <div className={rowClassName}>
                    <Label htmlFor="host" className={labelClassName}>
                        Workspace host
                    </Label>
                    <TextInput
                        id="host"
                        name="host"
                        type="text"
                        value={values.host ?? ""}
                        onChange={(e) => onChangeValue("host", e.target.value)}
                    />
                </div>
                <div className={rowClassName}>
                    <Label htmlFor="client_id" className={labelClassName}>
                        OAuth client ID
                    </Label>
                    <TextInput
                        id="client_id"
                        name="client_id"
                        type="text"
                        value={values.client_id ?? ""}
                        onChange={(e) =>
                            onChangeValue("client_id", e.target.value)
                        }
                    />
                </div>
                <div className={rowClassName}>
                    <Label htmlFor="client_secret" className={labelClassName}>
                        OAuth client secret
                    </Label>
                    <TextInput
                        id="client_secret"
                        name="client_secret"
                        type="password"
                        value={values.client_secret ?? ""}
                        onChange={(e) =>
                            onChangeValue("client_secret", e.target.value)
                        }
                    />
                </div>
            </>
        )}
        {identityKind === "snowflake" && (
            <SnowflakeFields
                values={values}
                authentication={authentication}
                rowClassName={rowClassName}
                labelClassName={labelClassName}
                onChangeSnowflakeAuth={onChangeSnowflakeAuth}
                onChangeValue={onChangeValue}
            />
        )}
        {(identityKind === "bigquery" || identityKind === "gcs") &&
            googleAuthenticationForm}
        {identityKind === "s3" && (
            <>
                <div className={rowClassName}>
                    <Label
                        htmlFor="s3-authentication"
                        className={labelClassName}>
                        Authentication
                    </Label>
                    <Select
                        id="s3-authentication"
                        options={[
                            { value: "assume_role", label: "AWS role" },
                            {
                                value: "access_key",
                                label: "S3-compatible access key"
                            }
                        ]}
                        selectedValues={[s3AuthenticationMode]}
                        onValuesChange={(vals) =>
                            onChangeS3Auth(vals[0] as S3AuthenticationMode)
                        }
                        mode="single"
                        search={false}
                    />
                </div>
                {s3AuthenticationMode === "assume_role" ? (
                    <>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="partition"
                                className={labelClassName}>
                                AWS partition
                            </Label>
                            <TextInput
                                id="partition"
                                name="partition"
                                type="text"
                                value={values.partition ?? ""}
                                onChange={(e) =>
                                    onChangeValue("partition", e.target.value)
                                }
                            />
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="role_arn"
                                className={labelClassName}>
                                Role ARN
                            </Label>
                            <TextInput
                                id="role_arn"
                                name="role_arn"
                                type="text"
                                value={values.role_arn ?? ""}
                                onChange={(e) =>
                                    onChangeValue("role_arn", e.target.value)
                                }
                            />
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="external_id"
                                className={labelClassName}>
                                External ID (optional)
                            </Label>
                            <TextInput
                                id="external_id"
                                name="external_id"
                                type="password"
                                value={values.external_id ?? ""}
                                onChange={(e) =>
                                    onChangeValue("external_id", e.target.value)
                                }
                            />
                        </div>
                    </>
                ) : (
                    <>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="endpoint_url"
                                className={labelClassName}>
                                S3 API endpoint
                            </Label>
                            <TextInput
                                id="endpoint_url"
                                name="endpoint_url"
                                type="text"
                                placeholder="https://minio.example.com:9000"
                                value={values.endpoint_url ?? ""}
                                onChange={(e) =>
                                    onChangeValue(
                                        "endpoint_url",
                                        e.target.value
                                    )
                                }
                            />
                        </div>
                        <p className="text-sm text-title-secondary">
                            Use your storage service&apos;s S3 API endpoint, not
                            its web console.
                        </p>
                        <div className={rowClassName}>
                            <Label htmlFor="region" className={labelClassName}>
                                Region
                            </Label>
                            <TextInput
                                id="region"
                                name="region"
                                type="text"
                                value={values.region ?? ""}
                                onChange={(e) =>
                                    onChangeValue("region", e.target.value)
                                }
                            />
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="addressing_style"
                                className={labelClassName}>
                                Addressing style
                            </Label>
                            <Select
                                id="addressing_style"
                                options={[
                                    { value: "path", label: "Path" },
                                    {
                                        value: "virtual",
                                        label: "Virtual hosted"
                                    },
                                    { value: "auto", label: "Automatic" }
                                ]}
                                selectedValues={[
                                    values.addressing_style || "path"
                                ]}
                                onValuesChange={(vals) =>
                                    onChangeValue("addressing_style", vals[0])
                                }
                                mode="single"
                                search={false}
                            />
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="access_key_id"
                                className={labelClassName}>
                                Access key ID
                            </Label>
                            <TextInput
                                id="access_key_id"
                                name="access_key_id"
                                type="password"
                                value={values.access_key_id ?? ""}
                                onChange={(e) =>
                                    onChangeValue(
                                        "access_key_id",
                                        e.target.value
                                    )
                                }
                            />
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="secret_access_key"
                                className={labelClassName}>
                                Secret access key
                            </Label>
                            <TextInput
                                id="secret_access_key"
                                name="secret_access_key"
                                type="password"
                                value={values.secret_access_key ?? ""}
                                onChange={(e) =>
                                    onChangeValue(
                                        "secret_access_key",
                                        e.target.value
                                    )
                                }
                            />
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="session_token"
                                className={labelClassName}>
                                Session token (optional)
                            </Label>
                            <TextInput
                                id="session_token"
                                name="session_token"
                                type="password"
                                value={values.session_token ?? ""}
                                onChange={(e) =>
                                    onChangeValue(
                                        "session_token",
                                        e.target.value
                                    )
                                }
                            />
                        </div>
                        <div className={`${rowClassName} !items-start`}>
                            <Label
                                htmlFor="ca_certificate"
                                className={labelClassName}>
                                CA certificate (optional)
                            </Label>
                            <TextArea
                                id="ca_certificate"
                                name="ca_certificate"
                                rows={4}
                                value={values.ca_certificate ?? ""}
                                onChange={(e) =>
                                    onChangeValue(
                                        "ca_certificate",
                                        e.target.value
                                    )
                                }
                            />
                        </div>
                    </>
                )}
            </>
        )}
        {identityKind === "adls" && (
            <>
                <div className={rowClassName}>
                    <Label htmlFor="tenant_id" className={labelClassName}>
                        Tenant ID
                    </Label>
                    <TextInput
                        id="tenant_id"
                        name="tenant_id"
                        type="text"
                        value={values.tenant_id ?? ""}
                        onChange={(e) =>
                            onChangeValue("tenant_id", e.target.value)
                        }
                    />
                </div>
                <div className={rowClassName}>
                    <Label htmlFor="client_id" className={labelClassName}>
                        Client ID
                    </Label>
                    <TextInput
                        id="client_id"
                        name="client_id"
                        type="text"
                        value={values.client_id ?? ""}
                        onChange={(e) =>
                            onChangeValue("client_id", e.target.value)
                        }
                    />
                </div>
                <div className={rowClassName}>
                    <Label htmlFor="cloud" className={labelClassName}>
                        Azure cloud
                    </Label>
                    <TextInput
                        id="cloud"
                        name="cloud"
                        type="text"
                        value={values.cloud ?? ""}
                        onChange={(e) => onChangeValue("cloud", e.target.value)}
                    />
                </div>
            </>
        )}
    </>
);
/* eslint-enable complexity */

const AddIdentity = () => {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const {
        groups,
        isLoading: areGroupsLoading,
        isError: didGroupsLoadFail,
        refetch: refetchGroups
    } = useKumoGroups();
    const featureFlags = useFeatureFlags();
    const workloadExchangeAvailable = Boolean(
        featureFlags.GCP_WORKLOAD_EXCHANGE_ENABLED
    );
    const ambientGcpAvailable = Boolean(featureFlags.GCP_AMBIENT_ENABLED);
    const ambientGcpDeveloperCredentials = Boolean(
        featureFlags.GCP_AMBIENT_DEVELOPER_CREDENTIALS
    );
    const ambientGcpServiceAccountEmail =
        featureFlags.GCP_AMBIENT_SERVICE_ACCOUNT_EMAIL ?? "";
    // Prefer ambient when the deployment allows it, then workload-identity
    // impersonation, then the uploaded service-account key as the last resort.
    const defaultGoogleAuthenticationMode: GoogleAuthenticationMode =
        ambientGcpAvailable
            ? "ambient"
            : workloadExchangeAvailable
              ? "workload_identity"
              : "service_account_key";
    const [identityKind, setIdentityKind] =
        useState<IdentityKind>("databricks");
    const [values, setValues] = useState<Record<string, string>>(
        providerDefaults.databricks
    );
    const [groupIds, setGroupIds] = useState<string[]>([]);
    const [slug, setSlug] = useState("");
    const [error, setError] = useState("");
    const [saving, setSaving] = useState(false);
    const [googleAuthenticationMode, setGoogleAuthenticationMode] =
        useState<GoogleAuthenticationMode>(defaultGoogleAuthenticationMode);
    const [googleFileError, setGoogleFileError] = useState("");
    const [googleKeyFile, setGoogleKeyFile] = useState<FileUploadItem>();
    const [s3AuthenticationMode, setS3AuthenticationMode] =
        useState<S3AuthenticationMode>("assume_role");

    // A single-tenant deployment has exactly one Kumo group. Default a new
    // identity into it (once) so it is immediately usable: an identity bound
    // to no group is inert and fails automatic connector selection because no
    // identity is assigned to the project's group. With multiple groups, keep
    // the deliberate least-privilege opt-in. The ref guard applies the default
    // only once so the user can still deselect.
    const didDefaultGroup = useRef(false);
    useEffect(() => {
        if (didDefaultGroup.current || groups.length !== 1) return;
        const onlyGroupId = groups[0].id;
        if (!onlyGroupId) return;
        didDefaultGroup.current = true;
        setGroupIds((current) =>
            current.length === 0 ? [onlyGroupId] : current
        );
    }, [groups]);

    const setValue = (name: string, value: string) =>
        setValues((current) => ({ ...current, [name]: value }));

    const changeProvider = (next: IdentityKind) => {
        setIdentityKind(next);
        setValues((current) => ({
            display_name: current.display_name ?? "",
            description: current.description ?? "",
            ...providerDefaults[next]
        }));
        setGoogleAuthenticationMode(defaultGoogleAuthenticationMode);
        setS3AuthenticationMode("assume_role");
        setGoogleFileError("");
        setGoogleKeyFile(undefined);
    };

    const changeGoogleAuthentication = (next: GoogleAuthenticationMode) => {
        setGoogleAuthenticationMode(next);
        setGoogleFileError("");
        setGoogleKeyFile(undefined);
        setValues((current) => {
            const safe = { ...current };
            for (const field of [
                ...googleKeyFields,
                "universe_domain",
                "service_account_email"
            ]) {
                delete safe[field];
            }
            return safe;
        });
    };

    // The upload control is controlled so the file card can settle on a real
    // outcome: the catalog stamps every new file as "uploading" at 0 bytes, so
    // an uncontrolled Upload would spin forever. Removing the file clears the
    // parsed key with it, so a removed file can never be submitted as though it
    // were still attached.
    const changeGoogleKeyFile = (file: FileUploadItem | undefined) => {
        setGoogleFileError("");
        setGoogleKeyFile(file);
        setValues((current) => {
            const safe = { ...current };
            for (const field of [...googleKeyFields, "universe_domain"]) {
                delete safe[field];
            }
            return safe;
        });
        if (!file) return;

        // The card's default failure label is "Upload failed", which misreads
        // here: nothing is uploaded at this point, the file is read and checked
        // in the browser.
        const settle = (
            status: FileUploadItem["status"],
            errorMessage?: string
        ) =>
            setGoogleKeyFile((current) =>
                current?.id === file.id
                    ? { ...current, status, errorMessage }
                    : current
            );

        void file.file
            .text()
            .then((text) => {
                // Parse before setValues: a state updater runs during render,
                // so a throw inside it escapes this catch and trips the error
                // boundary instead of showing the field-level message.
                const parsed = parseGoogleServiceAccount(text);
                setValues((current) => ({ ...current, ...parsed }));
                settle("success");
            })
            .catch(() => {
                setGoogleFileError(
                    "Select a complete Google service-account JSON key."
                );
                settle("error", "Not a valid key file");
            });
    };

    const changeSnowflakeAuthentication = (
        next: SnowflakeAuthenticationKind
    ) => {
        setValues((current) => {
            const providerValues = { ...current };
            delete providerValues.password;
            delete providerValues.private_key;
            delete providerValues.private_key_passphrase;
            return { ...providerValues, authentication: next };
        });
    };

    const changeS3Authentication = (next: S3AuthenticationMode) => {
        setS3AuthenticationMode(next);
        setValues((current) => {
            const safe = { ...current };
            for (const field of [
                "partition",
                "role_arn",
                "external_id",
                "endpoint_url",
                "region",
                "addressing_style",
                "access_key_id",
                "secret_access_key",
                "session_token",
                "ca_certificate"
            ]) {
                delete safe[field];
            }
            return next === "assume_role"
                ? { ...safe, ...providerDefaults.s3 }
                : safe;
        });
    };

    const authentication = (values.authentication ??
        "password") as SnowflakeAuthenticationKind;
    const selectedGroupIds = useMemo(
        () =>
            groupIds.filter((groupId) =>
                groups.some((group) => group.id === groupId)
            ),
        [groupIds, groups]
    );
    const canSubmit = useMemo(() => {
        const fields = [...requiredFields[identityKind]];
        if (identityKind === "snowflake") {
            fields.push(
                authentication === "key_pair" ? "private_key" : "password"
            );
        }
        if (identityKind === "bigquery" || identityKind === "gcs") {
            if (googleAuthenticationMode === "workload_identity") {
                fields.push("service_account_email");
            } else if (googleAuthenticationMode === "service_account_key") {
                fields.push(...googleKeyFields);
            }
            // "ambient" needs no extra fields - only display_name.
        }
        if (identityKind === "s3") {
            fields.push(
                ...(s3AuthenticationMode === "access_key"
                    ? [
                          "endpoint_url",
                          "region",
                          "access_key_id",
                          "secret_access_key"
                      ]
                    : ["partition", "role_arn"])
            );
        }
        return (
            !areGroupsLoading &&
            !didGroupsLoadFail &&
            selectedGroupIds.length > 0 &&
            fields.every((field) => Boolean(values[field]?.trim()))
        );
    }, [
        areGroupsLoading,
        authentication,
        didGroupsLoadFail,
        googleAuthenticationMode,
        identityKind,
        selectedGroupIds,
        s3AuthenticationMode,
        values
    ]);
    const slugError = resourceSlugError(slug, values.display_name ?? "");

    const payload = (): CreateIdentityArgs => {
        const common = {
            display_name: values.display_name.trim(),
            ...(slug ? { slug } : {}),
            description: values.description?.trim() || undefined,
            group_ids: selectedGroupIds
        };
        return buildIdentityPayload(
            identityKind,
            common,
            values,
            authentication,
            googleAuthenticationMode,
            s3AuthenticationMode
        );
    };

    const submit = async () => {
        if (!canSubmit) return;
        setSaving(true);
        setError("");
        try {
            await createIdentities(payload());
            // Invalidate the cached identity lists so the newly created
            // identity shows up immediately, without a manual page refresh.
            invalidateIdentityLists(queryClient);
            navigate(navRoutes.AdminIdentitySettings);
        } catch (caught) {
            setError(parseErrorMessage(caught as CustomError));
            setSaving(false);
        }
    };

    const input = (name: string, label: string, secret = false) => (
        <TextInput
            id={name}
            name={name}
            label={label}
            type={secret ? "password" : "text"}
            value={values[name] ?? ""}
            onChange={(event) => setValue(name, event.target.value)}
        />
    );

    const labelClassName = "text-sm font-medium text-title-secondary leading-5";
    const rowClassName =
        "grid grid-cols-[var(--form-label-col)_1fr] items-center gap-2";

    const googleAuthenticationForm = (
        <GoogleAuthenticationForm
            identityKind={identityKind}
            mode={googleAuthenticationMode}
            workloadExchangeAvailable={workloadExchangeAvailable}
            ambientAvailable={ambientGcpAvailable}
            developerCredentials={ambientGcpDeveloperCredentials}
            serviceAccountEmail={ambientGcpServiceAccountEmail}
            values={values}
            fileError={googleFileError}
            rowClassName={rowClassName}
            labelClassName={labelClassName}
            onModeChange={changeGoogleAuthentication}
            keyFile={googleKeyFile}
            onKeyFileChange={changeGoogleKeyFile}
            renderInput={input}
        />
    );

    let groupSelection: ReactNode;
    if (areGroupsLoading) {
        groupSelection = (
            <div
                aria-live="polite"
                className="flex items-center gap-2 text-sm text-title-secondary">
                <Spinner />
                <span>Loading Kumo groups…</span>
            </div>
        );
    } else if (didGroupsLoadFail) {
        groupSelection = (
            <div
                role="alert"
                className="flex items-center justify-between gap-4 rounded bg-surface-grey p-3 text-sm text-destructive">
                <span>
                    Kumo groups couldn’t be loaded. Try again to continue.
                </span>
                <Button variant="outline" onClick={() => void refetchGroups()}>
                    Try again
                </Button>
            </div>
        );
    } else if (groups.length === 0) {
        groupSelection = (
            <div
                role="status"
                className="flex items-center justify-between gap-4 rounded bg-surface-grey p-3 text-sm text-title-secondary">
                <span>
                    No Kumo groups are available. Create a group before adding
                    an identity.
                </span>
                <Button
                    variant="outline"
                    onClick={() => navigate(navRoutes.AdminGroups)}>
                    Manage groups
                </Button>
            </div>
        );
    } else {
        groupSelection = (
            <div className="grid grid-cols-2 gap-2">
                {groups.map((group) => (
                    <label
                        key={group.id ?? group.display_name}
                        className="flex items-center gap-2 cursor-pointer">
                        <Checkbox
                            checked={groupIds.includes(group.id ?? "")}
                            onCheckedChange={(checked) =>
                                group.id &&
                                setGroupIds((current) =>
                                    checked
                                        ? [...current, group.id as string]
                                        : current.filter(
                                              (id) => id !== group.id
                                          )
                                )
                            }
                        />
                        <span className="text-base font-medium leading-5 text-title-primary">
                            {group.display_name}
                        </span>
                    </label>
                ))}
            </div>
        );
    }

    return (
        <>
            <div className="flex items-center justify-between">
                <Title
                    variant="primary-heading"
                    className="font-bold text-3xl text-title-primary">
                    Add Identity
                </Title>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        size="sm"
                        className="rounded-sm px-4 py-2 text-base h-9 leading-5"
                        onClick={() =>
                            navigate(navRoutes.AdminIdentitySettings)
                        }>
                        Cancel
                    </Button>
                    <Button
                        variant="primary"
                        size="sm"
                        className="rounded-sm px-4 py-2 text-base h-9 leading-5"
                        disabled={saving || !canSubmit || Boolean(slugError)}
                        onClick={() => void submit()}>
                        Add
                    </Button>
                </div>
            </div>

            <div className="max-w-3xl [&_input:not([type=radio],[type=checkbox],[type=file])]:text-base [&_input:not([type=radio],[type=checkbox],[type=file])]:font-medium [&_input:not([type=radio],[type=checkbox],[type=file])]:leading-5 [&_input:not([type=radio],[type=checkbox],[type=file])]:text-title-primary [&_textarea]:text-base [&_textarea]:font-medium [&_textarea]:leading-5 [&_textarea]:text-title-primary">
                <p className="mt-2 text-title-secondary">
                    Configure a least-privilege external principal, then make it
                    available to each Kumo group that may use it.
                </p>

                {error && <p className="mt-4 text-destructive">{error}</p>}

                <section className="mt-6 flex flex-col gap-4">
                    <Title
                        variant="section-header"
                        className="font-bold text-title-primary">
                        Identity details
                    </Title>
                    <div className={rowClassName}>
                        <Label
                            htmlFor="display_name"
                            className={labelClassName}>
                            Identity name
                        </Label>
                        <TextInput
                            id="display_name"
                            name="display_name"
                            type="text"
                            value={values.display_name ?? ""}
                            onChange={(event) =>
                                setValue("display_name", event.target.value)
                            }
                        />
                    </div>
                    <div className={`${rowClassName} !items-start`}>
                        <Label
                            htmlFor="description"
                            className={`${labelClassName} pt-2 !flex-col !items-start`}>
                            Description{" "}
                            <span className="font-normal">(optional)</span>
                        </Label>
                        <TextArea
                            id="description"
                            name="description"
                            rows={2}
                            value={values.description ?? ""}
                            onChange={(event) =>
                                setValue("description", event.target.value)
                            }
                            className="resize-none overflow-y-auto max-h-[var(--form-description-max-h)]"
                        />
                    </div>
                    <ResourceSlugField
                        id="new-identity-slug"
                        displayName={values.display_name ?? ""}
                        value={slug}
                        onChange={setSlug}
                        horizontal
                        labelClassName={labelClassName}
                        hidePreviewHint
                    />
                    <ProviderFields
                        identityKind={identityKind}
                        values={values}
                        authentication={authentication}
                        googleAuthenticationForm={googleAuthenticationForm}
                        s3AuthenticationMode={s3AuthenticationMode}
                        rowClassName={rowClassName}
                        labelClassName={labelClassName}
                        onChangeProvider={changeProvider}
                        onChangeSnowflakeAuth={changeSnowflakeAuthentication}
                        onChangeS3Auth={changeS3Authentication}
                        onChangeValue={setValue}
                    />
                </section>

                <section className="mt-8 flex flex-col gap-4">
                    <Title
                        variant="section-header"
                        className="font-bold text-title-primary">
                        Kumo groups
                    </Title>
                    <p className="text-sm text-title-secondary">
                        Projects owned by a selected group may create connectors
                        using this identity.
                    </p>
                    {groupSelection}
                </section>
            </div>
        </>
    );
};

export default WithErrorBoundary(AddIdentity);
