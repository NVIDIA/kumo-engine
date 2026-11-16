// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";

import {
    AddCircle,
    Button,
    DataTable,
    Label,
    TextArea,
    TextInput,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import {
    createIdentitiesTestByIdentityId,
    deleteIdentitiesByIdentityId,
    getIdentitiesConnectorsByIdentityId,
    getIdentitiesGroupsByIdentityId,
    updateIdentitiesByIdentityId,
    updateIdentitiesCredentialsByIdentityId,
    updateIdentitiesGroupsByIdentityId,
    useGetIdentities
} from "gen/v0/api/identities/identities";
import {
    BigQueryIdentityConfig,
    GCSIdentityConfig,
    GoogleServiceAccountKeyAuthentication,
    IdentityConnectorUsage,
    IdentityResponse,
    RotateIdentityCredentialsArgs
} from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { identitySettingsColumns } from "components/@ui/TableColumns/IdentitySettings";
import {
    AdminActionRow,
    AdminCollection,
    AdminPage,
    AdminSection
} from "components/Admin/AdminPage";
import { DeleteIdentityDialog } from "components/Admin/IdentitySettings/DeleteIdentityDialog";
import { ImmutableResourceSlug } from "components/ResourceSlug";

import { useKumoGroups } from "hooks/groups/groups";

import { navRoutes } from "utils/Constants";
import { parseGoogleServiceAccount } from "utils/googleServiceAccount";
import {
    isAmbientGoogleIdentity,
    isAmbientS3Identity
} from "utils/identityUtils";
import { queryClient } from "utils/queryClient";
import { invalidateIdentityLists } from "utils/queryInvalidation";

type Draft = {
    displayName: string;
    description: string;
    groupIds: string[];
    credentials: Record<string, string>;
};

const emptyDraft: Draft = {
    displayName: "",
    description: "",
    groupIds: [],
    credentials: {}
};

// The delete endpoint returns 409 when connectors still reference the identity.
const IDENTITY_IN_USE_CONFLICT = 409;

const isIdentityInUseConflict = (error: unknown): boolean =>
    (error as { response?: { status?: number } }).response?.status ===
    IDENTITY_IN_USE_CONFLICT;

const getIdentityCollectionStatus = (
    status: string,
    selectedIdentity: IdentityResponse | undefined
): string => (selectedIdentity ? "" : status);

type GoogleIdentityResponse = IdentityResponse & {
    config: BigQueryIdentityConfig | GCSIdentityConfig;
};

type GoogleIdentityAuthentication =
    | BigQueryIdentityConfig["authentication"]
    | GCSIdentityConfig["authentication"];

const isGoogleIdentity = (
    identity: IdentityResponse | undefined
): identity is GoogleIdentityResponse =>
    identity?.config.type === "bigquery" || identity?.config.type === "gcs";

const isGoogleWorkloadAuthentication = (
    authentication: GoogleIdentityAuthentication
): authentication is Extract<
    GoogleIdentityAuthentication,
    { service_account_email: string }
> => "service_account_email" in authentication;

const isGoogleKeyAuthentication = (
    authentication: GoogleIdentityAuthentication
): authentication is GoogleServiceAccountKeyAuthentication =>
    "client_email" in authentication;

const isGoogleWorkloadIdentity = (
    identity: IdentityResponse | undefined
): boolean =>
    isGoogleIdentity(identity) &&
    isGoogleWorkloadAuthentication(identity.config.authentication);

const isS3AccessKeyIdentity = (
    identity: IdentityResponse | undefined
): boolean =>
    identity?.config.type === "s3" &&
    identity.config.authentication.mode === "access_key";

const isSnowflakeTokenFileIdentity = (
    identity: IdentityResponse | undefined
): boolean =>
    identity?.config.type === "snowflake" &&
    identity.config.authentication === "oauth_token_file";

const identitySaveLabel = (deploymentManaged: boolean): string =>
    deploymentManaged ? "Save group assignments" : "Save metadata and groups";

const identityAuthenticationDescription = (
    deploymentManaged: boolean,
    identity: IdentityResponse
): string =>
    isSnowflakeTokenFileIdentity(identity)
        ? "Kumo reads this deployment-managed OAuth token from the configured container file. No credential material is stored or rotated with this identity."
        : deploymentManaged
          ? "This deployment-managed ambient identity uses the approved AWS workload identity chain. Kumo stores no credential material for it."
          : isAmbientGoogleIdentity(identity)
            ? "Kumo resolves the Google credentials supplied by this deployment or devserver. No key is stored or rotated with this identity."
            : isGoogleWorkloadIdentity(identity)
              ? "Kumo uses its deployment workload identity to impersonate this service account. No customer private key is stored or rotated here."
              : isS3AccessKeyIdentity(identity)
                ? "Kumo uses an encrypted access key with this identity's S3-compatible endpoint settings. Rotation replaces the access key and session token; leave the CA certificate blank to keep the current certificate."
                : "Principal fields are immutable. A rotation updates every connector that references this identity.";

const identityHasRotatableMaterial = (
    identity: IdentityResponse | undefined
): boolean =>
    Boolean(identity) &&
    !isAmbientS3Identity(identity) &&
    !isAmbientGoogleIdentity(identity) &&
    !isGoogleWorkloadIdentity(identity) &&
    !isSnowflakeTokenFileIdentity(identity) &&
    identity?.config.type !== "adls";

const snowflakeRotationPayload = (
    authentication: "password" | "key_pair",
    credentials: Record<string, string>
): RotateIdentityCredentialsArgs =>
    authentication === "key_pair"
        ? {
              credentials: {
                  kind: "snowflake_key_pair",
                  private_key: credentials.private_key ?? "",
                  private_key_passphrase:
                      credentials.private_key_passphrase || null
              }
          }
        : {
              credentials: {
                  kind: "snowflake_password",
                  password: credentials.password ?? ""
              }
          };

const googleRotationPayload = (
    authentication: GoogleIdentityAuthentication,
    credentials: Record<string, string>
): RotateIdentityCredentialsArgs => {
    if (!isGoogleKeyAuthentication(authentication)) {
        throw new Error(
            "Workload identity credentials are deployment-managed."
        );
    }
    return {
        authentication: {
            mode: "service_account_key",
            project_id: credentials.project_id ?? "",
            client_email: credentials.client_email ?? "",
            client_id: credentials.client_id ?? "",
            auth_uri: credentials.auth_uri ?? "",
            token_uri: credentials.token_uri ?? "",
            universe_domain: credentials.universe_domain || "googleapis.com"
        },
        credentials: {
            kind: "google_service_account_key",
            private_key_id: credentials.private_key_id ?? "",
            private_key: credentials.private_key ?? ""
        }
    };
};

const s3RotationPayload = (
    identity: IdentityResponse,
    credentials: Record<string, string>
): RotateIdentityCredentialsArgs => {
    if (isS3AccessKeyIdentity(identity)) {
        const caCertificate = credentials.ca_certificate?.trim();
        const clearCaCertificate = credentials.clear_ca_certificate === "true";
        return {
            credentials: {
                kind: "s3_access_key",
                access_key_id: credentials.access_key_id ?? "",
                secret_access_key: credentials.secret_access_key ?? "",
                session_token: credentials.session_token || null,
                ...(clearCaCertificate
                    ? { ca_certificate: null }
                    : caCertificate
                      ? { ca_certificate: credentials.ca_certificate }
                      : {})
            }
        };
    }
    return {
        credentials: credentials.external_id
            ? {
                  kind: "s3_assume_role",
                  external_id: credentials.external_id
              }
            : null
    };
};

const buildRotationPayload = (
    identity: IdentityResponse,
    credentials: Record<string, string>
): RotateIdentityCredentialsArgs => {
    switch (identity.config.type) {
        case "databricks":
            return {
                credentials: {
                    kind: "databricks_oauth",
                    client_secret: credentials.client_secret ?? ""
                }
            };
        case "snowflake":
            if (identity.config.authentication === "oauth_token_file") {
                throw new Error(
                    "OAuth token-file credentials are deployment-managed."
                );
            }
            return snowflakeRotationPayload(
                identity.config.authentication,
                credentials
            );
        case "bigquery":
        case "gcs":
            return googleRotationPayload(
                identity.config.authentication,
                credentials
            );
        case "s3":
            return s3RotationPayload(identity, credentials);
        case "adls":
            return { credentials: null };
        default:
            throw new Error("Unsupported identity type");
    }
};

interface IdentityMetadataEditorProps {
    deploymentManaged: boolean;
    identity: IdentityResponse;
    draft: Draft;
    onDisplayNameChange: (value: string) => void;
    onDescriptionChange: (value: string) => void;
}

const IdentityMetadataEditor = ({
    deploymentManaged,
    identity,
    draft,
    onDisplayNameChange,
    onDescriptionChange
}: IdentityMetadataEditorProps) => {
    if (deploymentManaged) {
        return (
            <div
                className="mt-4 rounded-sm bg-surface-grey p-3 text-sm"
                aria-label="Deployment-managed identity details">
                <p className="font-medium text-title-primary">
                    {identity.display_name}
                </p>
                {identity.description && (
                    <p className="mt-1 text-title-secondary">
                        {identity.description}
                    </p>
                )}
                <p className="mt-2 text-title-secondary">
                    This ambient identity is managed by the Kumo deployment. Its
                    principal and metadata cannot be changed here.
                </p>
            </div>
        );
    }

    return (
        <div className="mt-4 grid grid-cols-2 gap-4">
            <TextInput
                id="identity-display-name"
                name="display_name"
                label="Identity name"
                value={draft.displayName}
                onChange={(event) => onDisplayNameChange(event.target.value)}
            />
            <TextInput
                id="identity-description"
                name="description"
                label="Description"
                value={draft.description}
                onChange={(event) => onDescriptionChange(event.target.value)}
            />
        </div>
    );
};

const GoogleAuthenticationDetails = ({
    identity
}: {
    identity: IdentityResponse;
}) => {
    if (identity.config.type !== "bigquery" && identity.config.type !== "gcs")
        return null;
    const authentication = identity.config.authentication;
    if (authentication.mode === "ambient") {
        return (
            <dl
                className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 rounded bg-surface-grey p-3 text-sm"
                aria-label="Google authentication details">
                <div>
                    <dt className="text-title-secondary">Mode</dt>
                    <dd>ambient</dd>
                </div>
                <div>
                    <dt className="text-title-secondary">Principal</dt>
                    <dd>Resolved by this deployment or devserver</dd>
                </div>
            </dl>
        );
    }
    const workload = isGoogleWorkloadAuthentication(authentication);
    const email = workload
        ? authentication.service_account_email
        : authentication.client_email;
    return (
        <dl
            className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 rounded bg-surface-grey p-3 text-sm"
            aria-label="Google authentication details">
            <div>
                <dt className="text-title-secondary">Mode</dt>
                <dd>{authentication.mode}</dd>
            </div>
            <div>
                <dt className="text-title-secondary">Service account</dt>
                <dd>{email}</dd>
            </div>
            {!workload && (
                <>
                    <div>
                        <dt className="text-title-secondary">Key project</dt>
                        <dd>{authentication.project_id}</dd>
                    </div>
                    <div>
                        <dt className="text-title-secondary">Client ID</dt>
                        <dd>{authentication.client_id}</dd>
                    </div>
                    <div>
                        <dt className="text-title-secondary">Universe</dt>
                        <dd>{authentication.universe_domain}</dd>
                    </div>
                    <div>
                        <dt className="text-title-secondary">
                            Credential status
                        </dt>
                        <dd>
                            {identity.credential_status.configured
                                ? "Configured"
                                : "Missing"}
                        </dd>
                    </div>
                </>
            )}
        </dl>
    );
};

const S3AuthenticationDetails = ({
    identity
}: {
    identity: IdentityResponse;
}) => {
    if (!isS3AccessKeyIdentity(identity) || identity.config.type !== "s3")
        return null;
    const connection = identity.config.connection;
    return (
        <dl
            className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 rounded bg-surface-grey p-3 text-sm"
            aria-label="S3 authentication details">
            <div>
                <dt className="text-title-secondary">Mode</dt>
                <dd>S3-compatible access key</dd>
            </div>
            <div>
                <dt className="text-title-secondary">Endpoint</dt>
                <dd>{connection?.endpoint_url || "AWS SDK default"}</dd>
            </div>
            <div>
                <dt className="text-title-secondary">Region</dt>
                <dd>{connection?.region || "AWS SDK default"}</dd>
            </div>
            <div>
                <dt className="text-title-secondary">Addressing</dt>
                <dd>{connection?.addressing_style || "auto"}</dd>
            </div>
        </dl>
    );
};

// eslint-disable-next-line complexity
const IdentitySettings = () => {
    const navigate = useNavigate();
    const [search, setSearch] = useState("");
    const [selectedIdentity, setSelectedIdentity] =
        useState<IdentityResponse>();
    const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
    const [blockingConnectors, setBlockingConnectors] = useState<
        IdentityConnectorUsage[]
    >([]);
    const [checkingBlockers, setCheckingBlockers] = useState(false);
    const [blockerCheckFailed, setBlockerCheckFailed] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [draft, setDraft] = useState<Draft>(emptyDraft);
    const [status, setStatus] = useState("");
    const [saving, setSaving] = useState(false);
    const { groups } = useKumoGroups();
    const { data, isLoading, refetch } = useGetIdentities();

    // Mirrors pendingDeleteId so an in-flight delete can tell whether the dialog
    // still targets the identity it started on before applying its result.
    const pendingDeleteIdRef = useRef<string | null>(null);
    useEffect(() => {
        pendingDeleteIdRef.current = pendingDeleteId;
    }, [pendingDeleteId]);

    // While a delete is pending, list the connectors that keep the identity in
    // use so the dialog can show them and refuse deletion until they are gone.
    useEffect(() => {
        if (!pendingDeleteId) {
            setBlockingConnectors([]);
            setCheckingBlockers(false);
            setBlockerCheckFailed(false);
            return;
        }
        let active = true;
        setCheckingBlockers(true);
        setBlockerCheckFailed(false);
        getIdentitiesConnectorsByIdentityId(pendingDeleteId)
            .then(({ data: usage }) => {
                if (active) setBlockingConnectors(usage.connectors ?? []);
            })
            .catch(() => {
                // Usage is unknown; the dialog keeps Delete disabled instead of
                // treating a failed check as "no connectors block deletion".
                if (active) setBlockerCheckFailed(true);
            })
            .finally(() => {
                if (active) setCheckingBlockers(false);
            });
        return () => {
            active = false;
        };
    }, [pendingDeleteId]);

    const identities = useMemo(() => {
        const term = search.trim().toLowerCase();
        const values = data?.data ?? [];
        if (!term) return values;
        return values.filter(
            (identity) =>
                identity.display_name.toLowerCase().includes(term) ||
                (identity.config.type ?? "").toLowerCase().includes(term)
        );
    }, [data?.data, search]);

    // Opening the confirmation eagerly marks the check in flight so Delete is
    // disabled on the first render, before the blocker fetch resolves.
    const requestDeleteIdentity = (id: string) => {
        setStatus("");
        setBlockingConnectors([]);
        setBlockerCheckFailed(false);
        setCheckingBlockers(true);
        setPendingDeleteId(id);
    };

    const deleteIdentity = async (id: string) => {
        const identity = (data?.data ?? []).find((item) => item.id === id);
        if (isAmbientS3Identity(identity) || deleting) return;
        setStatus("");
        setDeleting(true);
        try {
            await deleteIdentitiesByIdentityId(id);
            invalidateIdentityLists(queryClient);
            if (selectedIdentity?.id === id) setSelectedIdentity(undefined);
            // Only close the dialog if it still targets this identity.
            if (pendingDeleteIdRef.current === id) setPendingDeleteId(null);
            await refetch();
        } catch (error) {
            // The admin may have moved the dialog to another identity while this
            // delete was in flight; drop the result instead of overwriting it.
            if (pendingDeleteIdRef.current !== id) return;
            if (isIdentityInUseConflict(error)) {
                // A connector started using this identity between the check and
                // the delete. Re-list the blockers into the still-open dialog
                // instead of surfacing a bare conflict, without retrying delete.
                try {
                    const { data: usage } =
                        await getIdentitiesConnectorsByIdentityId(id);
                    if (pendingDeleteIdRef.current === id) {
                        setBlockingConnectors(usage.connectors ?? []);
                    }
                } catch (recheckError) {
                    if (pendingDeleteIdRef.current !== id) return;
                    setPendingDeleteId(null);
                    setStatus(
                        recheckError instanceof Error
                            ? recheckError.message
                            : String(recheckError)
                    );
                }
            } else {
                setPendingDeleteId(null);
                setStatus(
                    error instanceof Error ? error.message : String(error)
                );
            }
        } finally {
            setDeleting(false);
        }
    };

    const manageIdentity = async (identity: IdentityResponse) => {
        setStatus("");
        const response = await getIdentitiesGroupsByIdentityId(identity.id);
        setSelectedIdentity(identity);
        setDraft({
            displayName: identity.display_name,
            description: identity.description,
            groupIds: response.data.group_ids ?? [],
            credentials: {}
        });
    };

    const saveMetadataAndGroups = async () => {
        if (!selectedIdentity) return;
        setSaving(true);
        setStatus("");
        try {
            if (!isAmbientS3Identity(selectedIdentity)) {
                await updateIdentitiesByIdentityId(selectedIdentity.id, {
                    display_name: draft.displayName,
                    description: draft.description
                });
            }
            await updateIdentitiesGroupsByIdentityId(selectedIdentity.id, {
                group_ids: draft.groupIds
            });
            invalidateIdentityLists(queryClient);
            await refetch();
            setStatus(
                isAmbientS3Identity(selectedIdentity)
                    ? "Kumo groups updated."
                    : "Identity metadata and Kumo groups updated."
            );
        } catch (error) {
            setStatus(error instanceof Error ? error.message : String(error));
        } finally {
            setSaving(false);
        }
    };

    const rotationPayload = (): RotateIdentityCredentialsArgs => {
        if (!selectedIdentity) throw new Error("Select an identity first.");
        return buildRotationPayload(selectedIdentity, draft.credentials);
    };

    const rotateCredentials = async () => {
        if (!selectedIdentity || isAmbientS3Identity(selectedIdentity)) return;
        setSaving(true);
        setStatus("");
        try {
            await updateIdentitiesCredentialsByIdentityId(
                selectedIdentity.id,
                rotationPayload()
            );
            setDraft((current) => ({ ...current, credentials: {} }));
            await refetch();
            setStatus("Authentication material rotated and validated.");
        } catch (error) {
            setStatus(error instanceof Error ? error.message : String(error));
        } finally {
            setSaving(false);
        }
    };

    const parseGoogleRotationFile = async (file: File) => {
        if (!selectedIdentity || !isGoogleIdentity(selectedIdentity)) return;
        try {
            const parsed = parseGoogleServiceAccount(await file.text());
            const configured = selectedIdentity.config.authentication;
            if (
                !isGoogleKeyAuthentication(configured) ||
                parsed.client_email !== configured.client_email
            ) {
                throw new Error("service-account mismatch");
            }
            setDraft((current) => ({
                ...current,
                credentials: { ...parsed }
            }));
            setStatus(`Ready to rotate the key for ${parsed.client_email}.`);
        } catch {
            setDraft((current) => ({ ...current, credentials: {} }));
            setStatus(
                "Select a valid service-account JSON key for this exact configured account."
            );
        }
    };

    const testConnection = async () => {
        if (!selectedIdentity) return;
        setSaving(true);
        setStatus("");
        try {
            await createIdentitiesTestByIdentityId(selectedIdentity.id);
            setStatus("The configured external identity is valid.");
        } catch (error) {
            setStatus(error instanceof Error ? error.message : String(error));
        } finally {
            setSaving(false);
        }
    };

    const setCredential = (name: string, value: string) =>
        setDraft((current) => ({
            ...current,
            credentials: { ...current.credentials, [name]: value }
        }));

    const credentialInput = (name: string, label: string) => (
        <TextInput
            id={`rotate-${name}`}
            name={name}
            label={label}
            type="password"
            value={draft.credentials[name] ?? ""}
            onChange={(event) => setCredential(name, event.target.value)}
        />
    );

    const credentialTextArea = (
        name: string,
        label: string,
        disabled = false
    ) => (
        <div>
            <Label htmlFor={`rotate-${name}`}>{label}</Label>
            <TextArea
                id={`rotate-${name}`}
                name={name}
                rows={5}
                disabled={disabled}
                value={draft.credentials[name] ?? ""}
                onChange={(event) => setCredential(name, event.target.value)}
            />
        </div>
    );

    const deploymentManaged = isAmbientS3Identity(selectedIdentity);
    const hasRotatableMaterial = identityHasRotatableMaterial(selectedIdentity);
    const googleKeyReady =
        !isGoogleIdentity(selectedIdentity) ||
        Boolean(draft.credentials.private_key);
    const s3AccessKeyReady =
        !isS3AccessKeyIdentity(selectedIdentity) ||
        (Boolean(draft.credentials.access_key_id?.trim()) &&
            Boolean(draft.credentials.secret_access_key?.trim()));
    const clearS3CaCertificate =
        draft.credentials.clear_ca_certificate === "true";
    const collectionStatus = getIdentityCollectionStatus(
        status,
        selectedIdentity
    );
    const columns = useMemo(
        () =>
            identitySettingsColumns(
                (identity) => void manageIdentity(identity),
                (id) => requestDeleteIdentity(id)
            ),

        []
    );

    const pendingDeleteIdentity = pendingDeleteId
        ? (data?.data ?? []).find((i) => i.id === pendingDeleteId)
        : undefined;

    return (
        <>
            <AdminPage
                title="Identities"
                description="Manage reusable external principals and the groups whose projects may use them. Native IAM and warehouse grants enforce data access."
                descriptionClassName="max-w-5xl">
                <AdminCollection>
                    <AdminActionRow
                        search={search}
                        onSearchChange={setSearch}
                        searchPlaceholder="Search">
                        <Button
                            variant="primary"
                            onClick={() =>
                                navigate(navRoutes.AdminAddIdentity)
                            }>
                            <AddCircle /> Add identity
                        </Button>
                    </AdminActionRow>
                    {collectionStatus && (
                        <p className="text-sm">{collectionStatus}</p>
                    )}
                    <div>
                        <DataTable
                            id="identities-table"
                            columns={columns}
                            data={identities}
                            isLoading={isLoading}
                            enableGlobalFilter={false}
                            disableRowSelectionHover={true}
                        />
                    </div>
                </AdminCollection>

                {selectedIdentity && (
                    <AdminSection
                        title={`Manage ${selectedIdentity.display_name}`}
                        actions={
                            <Button
                                variant="outline"
                                onClick={() => {
                                    setSelectedIdentity(undefined);
                                    setDraft(emptyDraft);
                                    setStatus("");
                                }}>
                                Close
                            </Button>
                        }>
                        {status && <p className="my-3 text-sm">{status}</p>}
                        <ImmutableResourceSlug
                            id="identity-slug"
                            slug={selectedIdentity.slug}
                            nameEditable={!deploymentManaged}
                            className="mt-4 max-w-xl"
                        />
                        <IdentityMetadataEditor
                            deploymentManaged={deploymentManaged}
                            identity={selectedIdentity}
                            draft={draft}
                            onDisplayNameChange={(displayName) =>
                                setDraft((current) => ({
                                    ...current,
                                    displayName
                                }))
                            }
                            onDescriptionChange={(description) =>
                                setDraft((current) => ({
                                    ...current,
                                    description
                                }))
                            }
                        />
                        <GoogleAuthenticationDetails
                            identity={selectedIdentity}
                        />
                        <S3AuthenticationDetails identity={selectedIdentity} />
                        <Title
                            variant="section-header"
                            className="mt-5 font-bold">
                            Kumo groups
                        </Title>
                        <div className="mt-2 grid grid-cols-2 gap-2">
                            {groups.map((group) => (
                                <label
                                    key={group.id ?? group.display_name}
                                    className="flex items-center gap-2">
                                    <input
                                        type="checkbox"
                                        checked={draft.groupIds.includes(
                                            group.id ?? ""
                                        )}
                                        onChange={(event) =>
                                            group.id &&
                                            setDraft((current) => ({
                                                ...current,
                                                groupIds: event.target.checked
                                                    ? [
                                                          ...current.groupIds,
                                                          group.id as string
                                                      ]
                                                    : current.groupIds.filter(
                                                          (id) =>
                                                              id !== group.id
                                                      )
                                            }))
                                        }
                                    />
                                    {group.display_name}
                                </label>
                            ))}
                        </div>
                        <Button
                            className="mt-4"
                            disabled={saving}
                            onClick={() => void saveMetadataAndGroups()}>
                            {identitySaveLabel(deploymentManaged)}
                        </Button>

                        <Title
                            variant="section-header"
                            className="mt-6 font-bold">
                            Authentication
                        </Title>
                        <p className="my-2 text-sm text-title-secondary">
                            {identityAuthenticationDescription(
                                deploymentManaged,
                                selectedIdentity
                            )}
                        </p>
                        {hasRotatableMaterial && (
                            <div className="grid grid-cols-2 gap-4">
                                {selectedIdentity.config.type ===
                                    "databricks" &&
                                    credentialInput(
                                        "client_secret",
                                        "OAuth client secret"
                                    )}
                                {selectedIdentity.config.type === "snowflake" &&
                                    selectedIdentity.config.authentication ===
                                        "password" &&
                                    credentialInput("password", "Password")}
                                {selectedIdentity.config.type === "snowflake" &&
                                    selectedIdentity.config.authentication ===
                                        "key_pair" && (
                                        <>
                                            {credentialInput(
                                                "private_key",
                                                "Private key"
                                            )}
                                            {credentialInput(
                                                "private_key_passphrase",
                                                "Private-key passphrase"
                                            )}
                                        </>
                                    )}
                                {isGoogleIdentity(selectedIdentity) && (
                                    <div className="col-span-2">
                                        <label className="flex max-w-xl flex-col gap-2 text-sm font-medium">
                                            Replacement service-account JSON
                                            file
                                            <input
                                                aria-label="Replacement service-account JSON file"
                                                type="file"
                                                accept="application/json,.json"
                                                onChange={(event) => {
                                                    const file =
                                                        event.target.files?.[0];
                                                    if (file)
                                                        void parseGoogleRotationFile(
                                                            file
                                                        );
                                                    event.target.value = "";
                                                }}
                                            />
                                        </label>
                                        {draft.credentials.client_email && (
                                            <p className="mt-2 text-sm text-title-secondary">
                                                Parsed replacement for{" "}
                                                {draft.credentials.client_email}
                                            </p>
                                        )}
                                    </div>
                                )}
                                {selectedIdentity.config.type === "s3" &&
                                    isS3AccessKeyIdentity(selectedIdentity) && (
                                        <>
                                            {credentialInput(
                                                "access_key_id",
                                                "Access key ID"
                                            )}
                                            {credentialInput(
                                                "secret_access_key",
                                                "Secret access key"
                                            )}
                                            {credentialInput(
                                                "session_token",
                                                "Session token (optional)"
                                            )}
                                            {credentialTextArea(
                                                "ca_certificate",
                                                "New CA certificate (optional; leave blank to keep current)",
                                                clearS3CaCertificate
                                            )}
                                            <label className="flex items-center gap-2 text-sm">
                                                <input
                                                    type="checkbox"
                                                    checked={
                                                        clearS3CaCertificate
                                                    }
                                                    onChange={(event) =>
                                                        setDraft((current) => ({
                                                            ...current,
                                                            credentials: {
                                                                ...current.credentials,
                                                                ca_certificate:
                                                                    event.target
                                                                        .checked
                                                                        ? ""
                                                                        : current
                                                                              .credentials
                                                                              .ca_certificate,
                                                                clear_ca_certificate:
                                                                    event.target
                                                                        .checked
                                                                        ? "true"
                                                                        : ""
                                                            }
                                                        }))
                                                    }
                                                />
                                                Clear current CA certificate
                                            </label>
                                        </>
                                    )}
                                {selectedIdentity.config.type === "s3" &&
                                    !isS3AccessKeyIdentity(selectedIdentity) &&
                                    credentialInput(
                                        "external_id",
                                        "External ID (leave empty to clear)"
                                    )}
                            </div>
                        )}
                        {!hasRotatableMaterial &&
                            !deploymentManaged &&
                            !isSnowflakeTokenFileIdentity(selectedIdentity) && (
                                <p className="text-sm text-title-secondary">
                                    This identity uses deployment workload
                                    federation; Kumo stores no long-lived
                                    credential.
                                </p>
                            )}
                        <div className="mt-4 flex gap-2">
                            {hasRotatableMaterial && (
                                <Button
                                    disabled={
                                        saving ||
                                        !googleKeyReady ||
                                        !s3AccessKeyReady
                                    }
                                    onClick={() => void rotateCredentials()}>
                                    Rotate and validate
                                </Button>
                            )}
                            <Button
                                variant="outline"
                                disabled={saving}
                                onClick={() => void testConnection()}>
                                Test identity
                            </Button>
                        </div>
                    </AdminSection>
                )}
            </AdminPage>

            <DeleteIdentityDialog
                open={!!pendingDeleteId}
                onOpenChange={(open) => {
                    if (!open) setPendingDeleteId(null);
                }}
                identityName={pendingDeleteIdentity?.display_name ?? ""}
                blockingConnectors={blockingConnectors}
                loading={checkingBlockers || deleting}
                checkFailed={blockerCheckFailed}
                onConfirm={() => {
                    if (pendingDeleteId) void deleteIdentity(pendingDeleteId);
                }}
            />
        </>
    );
};

export default WithErrorBoundary(IdentitySettings);
