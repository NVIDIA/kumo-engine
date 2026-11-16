// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useMemo, useRef, useState } from "react";
import Spacer from "react-spacer";

import {
    Bouncingball,
    Button,
    cn,
    Description,
    Info,
    Label,
    Select,
    TextInput,
    Title,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { createProjectsConnectorsByProjectId } from "gen/v0/api/connectors/connectors";
import {
    useGetIdentities,
    useGetProjectsIdentitiesByProjectId
} from "gen/v0/api/identities/identities";
import {
    BigQueryIdentityConfigAuthentication,
    CreateConnectorArgs,
    CreateConnectorArgsConfig,
    GCSIdentityConfigAuthentication,
    IdentityResponse,
    SnowflakeAuthenticationKind
} from "gen/v0/types";

import { useAdminInfo } from "hooks/auth/admin";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";

import { getProjectPath, navRoutes } from "utils/Constants";
import { parseErrorMessage } from "utils/errorHandler";
import { isAmbientS3Identity } from "utils/identityUtils";
import { useCustomNavigation } from "utils/navigator";
import {
    adlsTargetForBaseLocation,
    gcsTargetForBaseLocation,
    s3TargetForBaseLocation
} from "utils/objectStoreTablePath";
import { invalidateConnectorList } from "utils/queryInvalidation";

import ConnectorTypeSelection, {
    ConnectorType
} from "./ConnectorTypeSelection";

const IDENTITY_PROVIDER_LABEL: Record<ConnectorType, string> = {
    databricks: "Databricks",
    snowflake: "Snowflake",
    bigquery: "BigQuery",
    s3: "S3",
    gcs: "GCS",
    adls: "ADLS"
};

// Shared form layout, matching Add identity and Edit Group. TextInput emits
// label, input and error as bare siblings, so each lands directly in this
// grid; the error is nudged to column 2 to sit under the field, not the label.
const labelClassName = "text-sm font-medium leading-5";
const rowClassName =
    "grid grid-cols-[var(--form-label-col)_1fr] items-center gap-2 [&>[id^=error]]:col-start-2";
const sectionClassName = "mt-8 flex max-w-3xl flex-col gap-4";
const sectionTitleClassName = "font-bold text-title-primary";
// Read-only rows share the field label column so values line up with the
// inputs below them; the value matches the type the inputs render.
const detailLabelClassName =
    "text-sm font-medium leading-5 text-title-secondary";
const detailValueClassName =
    "font-medium text-base leading-5 text-title-primary";

// "(optional)" and friends are secondary to the field name, so they drop to
// their own line at normal weight. Matches Add identity and the slug field.
const LABEL_QUALIFIER = /\s*(\([^)]*\))$/;

const fieldLabel = (id: string, label: string): ReactNode => {
    const qualifier = LABEL_QUALIFIER.exec(label);
    // Without a qualifier the catalog styles the plain string itself, which
    // also keeps the input's accessible name intact.
    if (!qualifier) return label;
    return (
        <Label
            htmlFor={id}
            className={cn(
                labelClassName,
                "text-title-secondary pt-2 flex-col items-start"
            )}>
            {label.slice(0, qualifier.index)}{" "}
            <span className="font-normal">{qualifier[1]}</span>
        </Label>
    );
};

const fieldRowClassName = (label: string) =>
    cn(rowClassName, LABEL_QUALIFIER.test(label) && "items-start");

const hasValue = (value?: string) => Boolean(value?.trim());
const trimmedOr = <T,>(value: string | undefined, fallback: T) =>
    value?.trim() || fallback;

const s3BaseLocationError = (baseLocation = ""): string => {
    if (!baseLocation) return "";
    try {
        s3TargetForBaseLocation(baseLocation);
        return "";
    } catch (error) {
        return error instanceof Error
            ? error.message
            : "Enter a valid S3 base location.";
    }
};

const targetLocationError = (
    connectorType: "gcs" | "adls",
    baseLocation = ""
): string => {
    if (!baseLocation) return "";
    try {
        if (connectorType === "gcs") {
            gcsTargetForBaseLocation(baseLocation);
        } else {
            adlsTargetForBaseLocation(baseLocation);
        }
        return "";
    } catch (error) {
        return error instanceof Error
            ? error.message
            : `Enter a valid ${connectorType.toUpperCase()} base location.`;
    }
};

interface IdentityDetail {
    label: string;
    value: string;
    /** Names the auth mode rather than a value unique to this identity. */
    isMode?: boolean;
}

// Same wording as Add identity's Authentication dropdown.
const SNOWFLAKE_AUTHENTICATION_LABEL: Record<
    SnowflakeAuthenticationKind,
    string
> = {
    password: "Password",
    key_pair: "Key pair",
    oauth_token_file: "OAuth token file"
};

// `mode` is optional in the schema for the two credentialed modes, so the
// impersonation target's presence is what tells those apart.
const googleIdentityDetails = (
    authentication:
        | BigQueryIdentityConfigAuthentication
        | GCSIdentityConfigAuthentication
): IdentityDetail[] => {
    if (authentication.mode === "ambient")
        return [
            {
                label: "Authentication",
                value: "Kumo's GCP identity",
                isMode: true
            }
        ];
    if ("service_account_email" in authentication)
        return [
            {
                label: "Authentication",
                value: "Workload identity (impersonation)",
                isMode: true
            },
            {
                label: "Target service-account email",
                value: authentication.service_account_email
            }
        ];
    return [
        {
            label: "Authentication",
            value: "Service account JSON key",
            isMode: true
        },
        { label: "Service account", value: authentication.client_email },
        { label: "Project", value: authentication.project_id }
    ];
};

// Labels are taken verbatim from Add identity so the same value is called the
// same thing on both pages.
const identityDetails = (identity: IdentityResponse): IdentityDetail[] => {
    const config = identity.config;
    switch (config.type) {
        case "databricks":
            return [
                { label: "Workspace host", value: config.host ?? "\u2014" },
                {
                    label: "OAuth client ID",
                    value: config.client_id ?? "\u2014"
                }
            ];
        case "snowflake":
            // Authentication is part of Snowflake's identity_key, so two
            // identities can share an account and username and differ only
            // here — without it they would describe themselves identically.
            return [
                { label: "Account", value: config.account },
                {
                    label: "Service-account username",
                    value: config.username ?? "\u2014"
                },
                {
                    label: "Authentication",
                    value: SNOWFLAKE_AUTHENTICATION_LABEL[
                        config.authentication
                    ],
                    isMode: true
                }
            ];
        case "bigquery":
        case "gcs":
            return googleIdentityDetails(config.authentication);
        case "s3": {
            const authentication = config.authentication;
            if (authentication.mode === "ambient") {
                return [
                    {
                        label: "Authentication",
                        value: "Verified ambient AWS workload identity",
                        isMode: true
                    }
                ];
            }
            if (authentication.mode === "access_key") {
                return [
                    {
                        label: "Authentication",
                        value: "S3-compatible access key",
                        isMode: true
                    },
                    {
                        label: "S3 API endpoint",
                        value: config.connection?.endpoint_url ?? "AWS default"
                    },
                    {
                        label: "Region",
                        value: config.connection?.region ?? "AWS default"
                    }
                ];
            }
            return [
                { label: "Role ARN", value: authentication.role_arn },
                {
                    label: "AWS partition",
                    value: authentication.partition
                }
            ];
        }
        case "adls":
            return [
                { label: "Tenant ID", value: config.tenant_id },
                { label: "Client ID", value: config.client_id },
                { label: "Azure cloud", value: config.cloud }
            ];
        default:
            return [];
    }
};

// The option has room for one secondary line, and display_name is free text
// an admin may not have made descriptive. So it spends that line on a field
// that varies between identities rather than the mode, which repeats across
// them and already appears in the rows below. Ambient has nothing else.
const identitySummary = (identity: IdentityResponse): string | undefined => {
    const details = identityDetails(identity);
    return (details.find((detail) => !detail.isMode) ?? details[0])?.value;
};

const isTargetComplete = (
    connectorType: ConnectorType | "",
    values: Record<string, string>
) => {
    if (!connectorType || !hasValue(values.name)) return false;
    switch (connectorType) {
        case "databricks":
            return (
                hasValue(values.warehouse_id) &&
                hasValue(values.catalog) &&
                hasValue(values.schema_name)
            );
        case "snowflake":
            return (
                hasValue(values.warehouse) &&
                hasValue(values.database) &&
                hasValue(values.schema_name)
            );
        case "bigquery":
            return (
                hasValue(values.project_id) &&
                hasValue(values.dataset_id) &&
                hasValue(values.location || "US")
            );
        case "s3":
            return !s3BaseLocationError(values.s3_base_location);
        case "gcs":
            return !targetLocationError("gcs", values.gcs_base_location);
        case "adls":
            return !targetLocationError("adls", values.adls_base_location);
    }
};

const configFor = (
    connectorType: ConnectorType,
    values: Record<string, string>
): CreateConnectorArgsConfig => {
    switch (connectorType) {
        case "databricks":
            return {
                type: "databricks",
                warehouse_id: values.warehouse_id.trim(),
                catalog: values.catalog.trim(),
                schema_name: values.schema_name.trim()
            };
        case "snowflake":
            return {
                type: "snowflake",
                role: trimmedOr(values.role, null),
                warehouse: values.warehouse.trim(),
                database: values.database.trim(),
                schema_name: values.schema_name.trim()
            };
        case "bigquery":
            return {
                type: "bigquery",
                project_id: values.project_id.trim(),
                dataset_id: values.dataset_id.trim(),
                location: trimmedOr(values.location, "US"),
                billing_project_id: trimmedOr(values.billing_project_id, null)
            };
        case "s3":
            return {
                type: "s3",
                target: s3TargetForBaseLocation(
                    values.s3_base_location ?? "",
                    values.region
                )
            };
        case "gcs":
            return {
                type: "gcs",
                target: gcsTargetForBaseLocation(
                    values.gcs_base_location ?? ""
                ),
                requester_pays_project: trimmedOr(
                    values.requester_pays_project,
                    null
                )
            };
        case "adls":
            return {
                type: "adls",
                target: adlsTargetForBaseLocation(
                    values.adls_base_location ?? ""
                )
            };
        default:
            throw new Error(`Unsupported connector type: ${connectorType}`);
    }
};

interface TargetFieldsProps {
    connectorType: ConnectorType;
    values: Record<string, string>;
    setValue: (name: string, value: string) => void;
}

const TargetFields = ({
    connectorType,
    values,
    setValue
}: TargetFieldsProps) => {
    // `id` differs from `name` where the field predates the snake_case value
    // keys, so callers can override it without restating the label.
    const input = (
        name: string,
        label: string,
        extra?: { id?: string; placeholder?: string; errorMsg?: string }
    ) => {
        const id = extra?.id ?? name;
        return (
            <div className={fieldRowClassName(label)}>
                <TextInput
                    id={id}
                    name={name}
                    label={fieldLabel(id, label)}
                    labelVariant="secondary"
                    labelClassname={labelClassName}
                    placeholder={extra?.placeholder}
                    isInvalid={Boolean(extra?.errorMsg)}
                    errorMsg={extra?.errorMsg}
                    value={values[name] ?? ""}
                    onChange={(event) => setValue(name, event.target.value)}
                />
            </div>
        );
    };

    let providerFields;
    switch (connectorType) {
        case "databricks":
            providerFields = (
                <>
                    {input("warehouse_id", "SQL warehouse ID")}
                    {input("catalog", "Catalog")}
                    {input("schema_name", "Schema")}
                </>
            );
            break;
        case "snowflake":
            providerFields = (
                <>
                    {input("role", "Role (optional)")}
                    {input("warehouse", "Warehouse")}
                    {input("database", "Database")}
                    {input("schema_name", "Schema")}
                </>
            );
            break;
        case "bigquery":
            providerFields = (
                <>
                    {input("project_id", "Data project ID")}
                    {input("dataset_id", "Dataset")}
                    {input("location", "Location (defaults to US)")}
                    {input(
                        "billing_project_id",
                        "Billing project ID (optional)"
                    )}
                </>
            );
            break;
        case "s3": {
            const baseLocation = values.s3_base_location ?? "";
            const baseLocationError = s3BaseLocationError(baseLocation);
            providerFields = (
                <>
                    {input("s3_base_location", "S3 base location (optional)", {
                        id: "s3-base-location",
                        placeholder: "bucket/prefix or s3://bucket/prefix",
                        errorMsg: baseLocationError
                    })}
                    {hasValue(baseLocation) && (
                        <>{input("region", "Region (optional)")}</>
                    )}
                    <p className="text-sm text-title-secondary">
                        Leave this blank for source-only any-URI mode. It cannot
                        list tables or write outputs; enter a full s3:// URI for
                        each source table. With a base location, relative paths
                        stay within that location.
                    </p>
                </>
            );
            break;
        }
        case "gcs": {
            const baseLocation = values.gcs_base_location ?? "";
            const baseLocationError = targetLocationError("gcs", baseLocation);
            providerFields = (
                <>
                    {input(
                        "gcs_base_location",
                        "GCS base location (optional)",
                        {
                            id: "gcs-base-location",
                            placeholder: "bucket/prefix or gs://bucket/prefix",
                            errorMsg: baseLocationError
                        }
                    )}
                    {input(
                        "requester_pays_project",
                        "Requester-pays project (optional)"
                    )}
                    <p className="text-sm text-title-secondary">
                        Leave this blank for source-only any-URI mode. It cannot
                        list tables or write outputs; enter a full gs:// URI for
                        each source table. With a base location, relative paths
                        stay within that bucket and prefix.
                    </p>
                </>
            );
            break;
        }
        case "adls": {
            const baseLocation = values.adls_base_location ?? "";
            const baseLocationError = targetLocationError("adls", baseLocation);
            providerFields = (
                <>
                    {input(
                        "adls_base_location",
                        "ADLS base location (optional)",
                        {
                            id: "adls-base-location",
                            placeholder:
                                "abfss://filesystem@account.dfs.core.windows.net/prefix",
                            errorMsg: baseLocationError
                        }
                    )}
                    <p className="text-sm text-title-secondary">
                        Leave this blank for source-only any-URI mode. It cannot
                        list tables or write outputs; enter a full abfss:// URI
                        for each source table. With a base location, relative
                        paths stay within that filesystem and prefix.
                    </p>
                </>
            );
            break;
        }
    }

    return (
        <>
            {input("name", "Connector name")}
            {input("description", "Description (optional)")}
            {providerFields}
        </>
    );
};

const useUnassignedIdentityState = (
    connectorType: ConnectorType | "",
    identities: IdentityResponse[],
    compatibleCount: number,
    queryState: { isLoading: boolean; isError: boolean }
) => {
    const { isAdmin } = useAdminInfo();
    const listIsEmpty =
        Boolean(connectorType) &&
        !queryState.isLoading &&
        !queryState.isError &&
        compatibleCount === 0;
    // Admins get the full identity list so the empty state can distinguish
    // "none exist" from "exist but not assigned to this project's group";
    // the endpoint is admin-only, so it stays disabled for everyone else.
    const shouldLoadUnassignedIdentities = isAdmin && listIsEmpty;
    const {
        data,
        isLoading: isUnassignedLoading,
        isError: isUnassignedError
    } = useGetIdentities({
        query: { enabled: shouldLoadUnassignedIdentities }
    });
    const count = useMemo(() => {
        if (!shouldLoadUnassignedIdentities) return 0;
        const visibleIds = new Set(identities.map((item) => item.id));
        return (data?.data ?? []).filter(
            (item) =>
                item.config.type === connectorType &&
                !isAmbientS3Identity(item) &&
                !visibleIds.has(item.id)
        ).length;
    }, [connectorType, data?.data, identities, shouldLoadUnassignedIdentities]);
    return {
        count,
        isLoading: shouldLoadUnassignedIdentities && isUnassignedLoading,
        isError: shouldLoadUnassignedIdentities && isUnassignedError
    };
};

interface IdentityListStatusProps {
    connectorType: ConnectorType;
    isLoading: boolean;
    isError: boolean;
    hasIdentityData: boolean;
    compatibleCount: number;
    unassignedCount: number;
    isUnassignedLoading: boolean;
    isUnassignedError: boolean;
    onRetry: () => void;
    onOpenIdentitySettings: () => void;
}

const IdentityListStatus = ({
    connectorType,
    isLoading,
    isError,
    hasIdentityData,
    compatibleCount,
    unassignedCount,
    isUnassignedLoading,
    isUnassignedError,
    onRetry,
    onOpenIdentitySettings
}: IdentityListStatusProps) => {
    // Only the initial error leaves the picker empty. A refetch failure that
    // still has cached identities keeps the list usable, so it stays silent.
    if (isError && !hasIdentityData) {
        return (
            <div
                role="alert"
                className="flex max-w-xl items-center justify-between gap-4 rounded bg-surface-grey p-3 text-sm text-destructive">
                <span>
                    Identities for this project could not be loaded. Try again,
                    or contact an administrator if the problem persists.
                </span>
                <Button variant="outline" onClick={onRetry}>
                    Try again
                </Button>
            </div>
        );
    }
    if (isLoading || compatibleCount > 0) return null;
    if (isUnassignedError) {
        return (
            <p role="alert" className="text-sm text-destructive">
                Unassigned identities could not be loaded. Refresh the page, or
                contact an administrator if the problem persists.
            </p>
        );
    }
    if (isUnassignedLoading) return null;
    if (unassignedCount > 0) {
        return (
            <p role="status" className="text-sm text-title-secondary">
                {unassignedCount === 1
                    ? `1 ${IDENTITY_PROVIDER_LABEL[connectorType]} identity exists but is not assigned to this project's group.`
                    : `${unassignedCount} ${IDENTITY_PROVIDER_LABEL[connectorType]} identities exist but none is assigned to this project's group.`}{" "}
                Assign groups in{" "}
                <button
                    type="button"
                    className="font-medium underline"
                    onClick={onOpenIdentitySettings}>
                    Identity settings
                </button>
                .
            </p>
        );
    }
    return (
        <p className="text-sm text-title-secondary">
            {`No ${IDENTITY_PROVIDER_LABEL[connectorType]} identities are available for this project. Ask an administrator to add project access before creating this connector.`}
        </p>
    );
};

const IdentityConnectorForm = () => {
    const navigate = useCustomNavigation();
    const queryClient = useQueryClient();
    const projectId = useProjectId() ?? "";
    const { data, isLoading, isError, refetch } =
        useGetProjectsIdentitiesByProjectId(projectId, {
            query: { enabled: Boolean(projectId) }
        });
    const identities = useMemo(() => data?.data ?? [], [data?.data]);
    const [connectorType, setConnectorType] = useState<ConnectorType | "">("");
    const [identityId, setIdentityId] = useState("");
    const [values, setValues] = useState<Record<string, string>>({});
    const [saving, setSaving] = useState(false);
    const submitting = useRef(false);
    const [error, setError] = useState("");
    const compatibleIdentities = useMemo(
        () => identities.filter((item) => item.config.type === connectorType),
        [connectorType, identities]
    );
    const unassignedIdentityState = useUnassignedIdentityState(
        connectorType,
        identities,
        compatibleIdentities.length,
        { isLoading, isError }
    );
    const identityOptions = useMemo(
        () =>
            compatibleIdentities.map((item) => ({
                value: item.id,
                label: item.display_name,
                labelDescription: identitySummary(item)
            })),
        [compatibleIdentities]
    );
    // Every connector stores exactly one identity, so an unmade or stale
    // choice is the only thing this has to reject; an empty, still-loading
    // or failed list simply has nothing to match.
    const selectedIdentity = useMemo(
        () => compatibleIdentities.find((item) => item.id === identityId),
        [compatibleIdentities, identityId]
    );
    const canSubmit =
        Boolean(projectId) &&
        Boolean(selectedIdentity) &&
        isTargetComplete(connectorType, values);
    const projectRoutes = projectId ? getProjectPath(projectId) : undefined;

    const selectConnectorType = (nextType: ConnectorType) => {
        setConnectorType(nextType);
        setIdentityId("");
        setError("");
        setValues(
            ["s3", "gcs", "adls"].includes(nextType) ? { name: nextType } : {}
        );
    };

    const setValue = (name: string, value: string) =>
        setValues((current) => ({ ...current, [name]: value }));

    const submit = async () => {
        // A queued second activation can land before the disabled re-render
        // commits, so the ref rejects it synchronously.
        if (
            submitting.current ||
            !connectorType ||
            !projectId ||
            !canSubmit ||
            !projectRoutes
        )
            return;
        submitting.current = true;
        setSaving(true);
        setError("");
        try {
            const args: CreateConnectorArgs = {
                identity_id: identityId,
                name: values.name.trim(),
                description: values.description?.trim() || undefined,
                config: configFor(connectorType, values)
            };
            const response = await createProjectsConnectorsByProjectId(
                projectId,
                args
            );
            // Invalidate the cached connector list so the newly created
            // connector shows up immediately, without a manual page refresh.
            invalidateConnectorList(queryClient, projectId);
            navigate(projectRoutes.Connectors + response.data.id);
        } catch (caught) {
            setError(parseErrorMessage(caught as CustomError));
            setSaving(false);
        } finally {
            submitting.current = false;
        }
    };

    return (
        <>
            {/* Same full-screen loading overlay Add table and Add graph use
                while their create request is in flight. */}
            {saving && <Bouncingball overlay={true} />}
            <div
                data-screenshot-ready="new-connector-page"
                className="flex items-center justify-between">
                <Title
                    variant="page-heading"
                    className="text-3xl font-bold"
                    data-testid="new-connector-page-heading">
                    Add Connector
                </Title>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        onClick={() =>
                            projectRoutes && navigate(projectRoutes.Connectors)
                        }>
                        Cancel
                    </Button>
                    <Button
                        disabled={saving || !canSubmit}
                        onClick={() => void submit()}>
                        Add
                    </Button>
                </div>
            </div>
            <div className="max-w-5xl [&_input:not([type=radio])]:text-base [&_input:not([type=radio])]:font-medium [&_input:not([type=radio])]:leading-5 [&_input:not([type=radio])]:text-title-primary">
                <Description
                    size="md"
                    className="leading-5 mt-2 text-title-secondary">
                    Choose a connector type to configure its identity and target
                </Description>
                <Spacer height={24} />
                {!projectId && (
                    <p className="text-destructive">
                        Select a project before creating a connector.
                    </p>
                )}
                {error && <p className="text-destructive">{error}</p>}
                <section className="flex flex-col gap-4">
                    <Title
                        variant="section-header"
                        className={sectionTitleClassName}>
                        Connector type
                    </Title>
                    <ConnectorTypeSelection
                        value={connectorType}
                        disabled={!projectId}
                        onChange={selectConnectorType}
                    />
                </section>
                {connectorType && (
                    <section className={sectionClassName}>
                        <div className="flex items-center gap-1">
                            <Title
                                variant="section-header"
                                className={sectionTitleClassName}>
                                Identity
                            </Title>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <button
                                        type="button"
                                        aria-label="What is an identity"
                                        className="rounded-sm text-title-secondary focus-visible:ring-2 focus-visible:ring-border-primary">
                                        <Info
                                            className="size-4"
                                            aria-hidden="true"
                                        />
                                    </button>
                                </TooltipTrigger>
                                <TooltipContent
                                    side="right"
                                    sideOffset={5}
                                    arrow={false}
                                    className="max-w-sm">
                                    The credentials Kumo uses to reach your data
                                    source. Administrators create identities and
                                    assign them to groups; you can choose any
                                    identity assigned to this project&apos;s
                                    group.
                                </TooltipContent>
                            </Tooltip>
                        </div>
                        <div className={rowClassName}>
                            <Label
                                htmlFor="connector-identity"
                                className={cn(
                                    labelClassName,
                                    "text-title-secondary"
                                )}>
                                Identity
                            </Label>
                            <Select
                                id="connector-identity"
                                options={identityOptions}
                                // Always an array: passing undefined puts the
                                // catalog Select back on its own internal
                                // state, which still holds the previous
                                // connector type's identity.
                                selectedValues={
                                    selectedIdentity
                                        ? [selectedIdentity.id]
                                        : []
                                }
                                onValuesChange={(vals) =>
                                    setIdentityId(vals[0] ?? "")
                                }
                                mode="single"
                                search={false}
                                placeholder="Select an identity"
                                disabled={
                                    isLoading ||
                                    !projectId ||
                                    identityOptions.length === 0
                                }
                            />
                        </div>
                        {selectedIdentity && (
                            // Same dt/dd treatment Identity settings uses for
                            // identity metadata, in this page's label column so
                            // it lines up with the fields below.
                            <dl
                                aria-label="Selected identity details"
                                className="flex flex-col gap-4">
                                {identityDetails(selectedIdentity).map(
                                    (detail) => (
                                        <div
                                            key={detail.label}
                                            className={rowClassName}>
                                            <dt
                                                className={
                                                    detailLabelClassName
                                                }>
                                                {detail.label}
                                            </dt>
                                            <dd
                                                className={
                                                    detailValueClassName
                                                }>
                                                {detail.value}
                                            </dd>
                                        </div>
                                    )
                                )}
                            </dl>
                        )}
                        <IdentityListStatus
                            connectorType={connectorType}
                            isLoading={isLoading}
                            isError={isError}
                            hasIdentityData={data !== undefined}
                            onRetry={() => void refetch()}
                            compatibleCount={compatibleIdentities.length}
                            unassignedCount={unassignedIdentityState.count}
                            isUnassignedLoading={
                                unassignedIdentityState.isLoading
                            }
                            isUnassignedError={unassignedIdentityState.isError}
                            onOpenIdentitySettings={() =>
                                navigate(navRoutes.AdminIdentitySettings)
                            }
                        />
                    </section>
                )}
                {connectorType && (
                    <section className={sectionClassName}>
                        <Title
                            variant="section-header"
                            className={sectionTitleClassName}>
                            Connector details
                        </Title>
                        <TargetFields
                            connectorType={connectorType}
                            values={values}
                            setValue={setValue}
                        />
                    </section>
                )}
            </div>
        </>
    );
};

export default IdentityConnectorForm;
