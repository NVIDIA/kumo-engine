// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
    createIdentities,
    getGetIdentitiesQueryKey
} from "gen/v0/api/identities/identities";
import { vi } from "vitest";

import AddIdentity from "../AddIdentity";

const navigate = vi.fn();
const featureFlags = vi.hoisted(() => ({
    workloadExchangeEnabled: false,
    ambientGcpEnabled: false,
    ambientGcpDeveloperCredentials: false
}));
const queryClientMock = vi.hoisted(() => ({ invalidateQueries: vi.fn() }));
const refetchGroups = vi.hoisted(() => vi.fn());

const privateKey =
    "-----BEGIN PRIVATE KEY-----\nprivate-material-must-not-render\n-----END PRIVATE KEY-----\n";
const googleKey = {
    type: "service_account",
    project_id: "identity-project",
    private_key_id: "key-id-1",
    private_key: privateKey,
    client_email: "kumo@identity-project.iam.gserviceaccount.com",
    client_id: "1234567890",
    auth_uri: "https://accounts.google.com/o/oauth2/auth",
    token_uri: "https://oauth2.googleapis.com/token",
    universe_domain: "googleapis.com"
};

vi.mock("react-router", async (importOriginal) => ({
    ...(await importOriginal<typeof import("react-router")>()),
    useNavigate: () => navigate
}));
vi.mock("@tanstack/react-query", async (importOriginal) => {
    const actual =
        await importOriginal<typeof import("@tanstack/react-query")>();
    return {
        ...actual,
        useQueryClient: () => queryClientMock
    };
});
// Preserve the real orval query-key helper so the invalidation assertions
// below stay in sync with the key the identity list actually subscribes to.
vi.mock("gen/v0/api/identities/identities", async (importOriginal) => ({
    ...(await importOriginal<
        typeof import("gen/v0/api/identities/identities")
    >()),
    createIdentities: vi.fn()
}));
vi.mock("hooks/utils/featureFlags", () => ({
    default: () => ({
        GCP_WORKLOAD_EXCHANGE_ENABLED: featureFlags.workloadExchangeEnabled,
        GCP_AMBIENT_ENABLED: featureFlags.ambientGcpEnabled,
        GCP_AMBIENT_DEVELOPER_CREDENTIALS:
            featureFlags.ambientGcpDeveloperCredentials
    })
}));
const useKumoGroupsMock = vi.hoisted(() => vi.fn());
vi.mock("hooks/groups/groups", () => ({
    useKumoGroups: useKumoGroupsMock
}));
const TWO_GROUPS = [
    { id: "group-sales", display_name: "Sales" },
    { id: "group-finance", display_name: "Finance" }
];

const providerLabels: Record<string, string> = {
    databricks: "Databricks",
    snowflake: "Snowflake",
    bigquery: "BigQuery",
    s3: "Amazon S3",
    gcs: "Google Cloud Storage",
    adls: "Azure Data Lake Storage"
};

const selectProvider = async (value: string) => {
    await userEvent.click(screen.getByLabelText("Provider"));
    await screen.findByRole("listbox");
    await userEvent.click(
        screen.getByRole("option", { name: providerLabels[value] })
    );
};

const selectAuthentication = async (value: string) => {
    const labels: Record<string, string> = {
        password: "Password",
        key_pair: "Key pair"
    };
    await userEvent.click(screen.getByLabelText("Authentication"));
    await screen.findByRole("listbox");
    await userEvent.click(screen.getByRole("option", { name: labels[value] }));
};

const selectS3Authentication = async (label: string) => {
    await userEvent.click(screen.getByLabelText("Authentication"));
    await screen.findByRole("listbox");
    await userEvent.click(screen.getByRole("option", { name: label }));
};

describe("Add identity", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        useKumoGroupsMock.mockReturnValue({
            groups: TWO_GROUPS,
            isLoading: false,
            isError: false,
            refetch: refetchGroups
        });
        featureFlags.workloadExchangeEnabled = false;
        featureFlags.ambientGcpEnabled = false;
        featureFlags.ambientGcpDeveloperCredentials = false;
        vi.mocked(createIdentities).mockResolvedValue({
            data: { id: "identity-1" }
        } as never);
    });

    const fillDatabricksIdentity = () => {
        for (const [label, value] of [
            ["Identity name", "Sales Databricks"],
            ["Workspace host", "https://workspace.example"],
            ["OAuth client ID", "service-principal"],
            ["OAuth client secret", "secret"]
        ]) {
            fireEvent.change(screen.getByLabelText(label), {
                target: { value }
            });
        }
    };

    it("stores credentials once and binds all selected Kumo groups", async () => {
        render(<AddIdentity />);

        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Sales Databricks" }
        });
        fireEvent.change(screen.getByLabelText("Workspace host"), {
            target: { value: "https://workspace.example" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client ID"), {
            target: { value: "service-principal" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client secret"), {
            target: { value: "secret" }
        });
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByLabelText("Finance"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createIdentities).toHaveBeenCalledWith({
                display_name: "Sales Databricks",
                description: undefined,
                group_ids: ["group-sales", "group-finance"],
                config: {
                    type: "databricks",
                    host: "https://workspace.example",
                    client_id: "service-principal"
                },
                credentials: {
                    kind: "databricks_oauth",
                    client_secret: "secret"
                }
            })
        );
        expect(navigate).toHaveBeenCalledWith("/admin/identity-settings");
    });

    it("defaults a new identity into the sole group of a single-tenant deploy", async () => {
        useKumoGroupsMock.mockReturnValue({
            groups: [{ id: "group-solo", display_name: "Default" }]
        });
        render(<AddIdentity />);

        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Solo Databricks" }
        });
        fireEvent.change(screen.getByLabelText("Workspace host"), {
            target: { value: "https://workspace.example" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client ID"), {
            target: { value: "sp" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client secret"), {
            target: { value: "secret" }
        });
        // The sole group is auto-selected; no checkbox is clicked here.
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createIdentities).toHaveBeenCalledWith(
                expect.objectContaining({ group_ids: ["group-solo"] })
            )
        );
    });

    it("keeps Save disabled until at least one group is selected", () => {
        render(<AddIdentity />);

        fillDatabricksIdentity();
        const saveButton = screen.getByRole("button", {
            name: "Add"
        });

        expect(saveButton).toBeDisabled();
        fireEvent.click(screen.getByLabelText("Sales"));
        expect(saveButton).toBeEnabled();
    });

    it("shows group loading state and keeps Save disabled", () => {
        useKumoGroupsMock.mockReturnValue({
            groups: [],
            isLoading: true,
            isError: false,
            refetch: refetchGroups
        });
        render(<AddIdentity />);

        fillDatabricksIdentity();

        expect(screen.getByText("Loading Kumo groups…")).toBeVisible();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("shows a group load error with a retry and keeps Save disabled", () => {
        useKumoGroupsMock.mockReturnValue({
            groups: [],
            isLoading: false,
            isError: true,
            refetch: refetchGroups
        });
        render(<AddIdentity />);

        fillDatabricksIdentity();
        expect(screen.getByRole("alert")).toHaveTextContent(
            "Kumo groups couldn’t be loaded"
        );
        fireEvent.click(screen.getByRole("button", { name: "Try again" }));

        expect(refetchGroups).toHaveBeenCalledOnce();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("shows an empty group state with a path to manage groups", () => {
        useKumoGroupsMock.mockReturnValue({
            groups: [],
            isLoading: false,
            isError: false,
            refetch: refetchGroups
        });
        render(<AddIdentity />);

        fillDatabricksIdentity();
        expect(screen.getByRole("status")).toHaveTextContent(
            "No Kumo groups are available"
        );
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
        fireEvent.click(screen.getByRole("button", { name: "Manage groups" }));

        expect(navigate).toHaveBeenCalledWith("/admin/groups");
    });

    it("disables Save when the auto-selected sole group is deselected", () => {
        useKumoGroupsMock.mockReturnValue({
            groups: [{ id: "group-solo", display_name: "Default" }]
        });
        render(<AddIdentity />);

        fillDatabricksIdentity();
        const saveButton = screen.getByRole("button", {
            name: "Add"
        });
        // The sole group is auto-selected; unchecking it must stick (the
        // one-time default guard does not re-select it).
        expect(screen.getByLabelText("Default")).toBeChecked();
        expect(saveButton).toBeEnabled();
        fireEvent.click(screen.getByLabelText("Default"));
        expect(saveButton).toBeDisabled();
        expect(createIdentities).not.toHaveBeenCalled();
    });

    it("invalidates the identity list cache after a successful create", async () => {
        render(<AddIdentity />);

        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Sales Databricks" }
        });
        fireEvent.change(screen.getByLabelText("Workspace host"), {
            target: { value: "https://workspace.example" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client ID"), {
            target: { value: "service-principal" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client secret"), {
            target: { value: "secret" }
        });
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        // The freshly created identity must be reflected in the list without a
        // manual page refresh, so the cached identity list is invalidated.
        await waitFor(() =>
            expect(queryClientMock.invalidateQueries).toHaveBeenCalledWith({
                queryKey: getGetIdentitiesQueryKey()
            })
        );
    });

    it("does not invalidate the identity list when create fails", async () => {
        vi.mocked(createIdentities).mockRejectedValueOnce({
            response: { status: 422, data: "bad identity" }
        });
        render(<AddIdentity />);

        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Sales Databricks" }
        });
        fireEvent.change(screen.getByLabelText("Workspace host"), {
            target: { value: "https://workspace.example" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client ID"), {
            target: { value: "service-principal" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client secret"), {
            target: { value: "secret" }
        });
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        expect(await screen.findByText("bad identity")).toBeInTheDocument();
        expect(queryClientMock.invalidateQueries).not.toHaveBeenCalled();
    });

    it("previews the derived ID and submits an explicit immutable ID", async () => {
        render(<AddIdentity />);

        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Sales Databricks" }
        });
        expect(screen.getByLabelText("ID (optional)")).toHaveAttribute(
            "placeholder",
            "sales-databricks"
        );
        fireEvent.change(screen.getByLabelText("ID (optional)"), {
            target: { value: "sales-warehouse" }
        });
        fireEvent.change(screen.getByLabelText("Workspace host"), {
            target: { value: "https://workspace.example" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client ID"), {
            target: { value: "service-principal" }
        });
        fireEvent.change(screen.getByLabelText("OAuth client secret"), {
            target: { value: "secret" }
        });
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createIdentities).toHaveBeenCalledWith(
                expect.objectContaining({
                    display_name: "Sales Databricks",
                    slug: "sales-warehouse"
                })
            )
        );
    });

    it("shows the server conflict response", async () => {
        vi.mocked(createIdentities).mockRejectedValue({
            response: {
                status: 409,
                data: "Identity slug 'sales-databricks' is already in use."
            }
        });
        render(<AddIdentity />);

        for (const [label, value] of [
            ["Identity name", "Sales Databricks"],
            ["Workspace host", "https://workspace.example"],
            ["OAuth client ID", "service-principal"],
            ["OAuth client secret", "secret"]
        ]) {
            fireEvent.change(screen.getByLabelText(label), {
                target: { value }
            });
        }
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        expect(
            await screen.findByText(
                "Identity slug 'sales-databricks' is already in use."
            )
        ).toBeVisible();
    });

    it("defaults to key mode and explains every unavailable option", async () => {
        render(<AddIdentity />);

        await selectProvider("bigquery");
        expect(
            screen.getByRole("radio", { name: /Workload identity/ })
        ).toBeDisabled();
        expect(
            screen.getByRole("radio", { name: /Use Kumo's GCP identity/ })
        ).toBeDisabled();
        expect(
            screen.getByRole("radio", { name: /Service account JSON key/ })
        ).toBeChecked();
        // Each disabled option names the setting that turns it on, rather than
        // one shared note below the group that explained only workload
        // exchange.
        expect(
            screen.getByText(/KUMO_GCP_WORKLOAD_EXCHANGE_ENABLED/)
        ).toBeVisible();
        expect(screen.getByText(/ambient_gcp/)).toBeVisible();
    });

    it("never recommends an option the deployment cannot offer", async () => {
        render(<AddIdentity />);

        await selectProvider("bigquery");
        expect(screen.queryByText(/\(Recommended\)/)).not.toBeInTheDocument();
    });

    it("recommends the best keyless option the deployment can offer", async () => {
        featureFlags.ambientGcpEnabled = true;
        render(<AddIdentity />);

        await selectProvider("bigquery");
        const recommended = screen.getByRole("radio", {
            name: /Use Kumo's GCP identity \(Recommended\)/
        });
        expect(recommended).toBeEnabled();
        expect(recommended).toBeChecked();
    });

    it("drops the parsed key when the chosen file is removed", async () => {
        render(<AddIdentity />);

        await selectProvider("gcs");
        fireEvent.change(screen.getByLabelText("Service-account JSON file"), {
            target: {
                files: [
                    new File([JSON.stringify(googleKey)], "identity.json", {
                        type: "application/json"
                    })
                ]
            }
        });
        await screen.findByLabelText("Parsed service-account details");

        await userEvent.click(
            screen.getByRole("button", { name: /Remove identity.json/ })
        );
        expect(
            screen.queryByLabelText("Parsed service-account details")
        ).not.toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
        // One click must clear the card itself, not just the parsed values.
        expect(screen.queryByText("identity.json")).not.toBeInTheDocument();
    });

    it.each(["bigquery", "gcs"])(
        "uploads one %s key without previewing the raw secret",
        async (provider) => {
            render(<AddIdentity />);

            fireEvent.change(screen.getByLabelText("Identity name"), {
                target: { value: `Analytics ${provider}` }
            });
            await selectProvider(provider);
            fireEvent.change(
                screen.getByLabelText("Service-account JSON file"),
                {
                    target: {
                        files: [
                            new File(
                                [JSON.stringify(googleKey)],
                                "identity.json",
                                { type: "application/json" }
                            )
                        ]
                    }
                }
            );

            expect(
                await screen.findByLabelText("Parsed service-account details")
            ).toHaveTextContent(googleKey.client_email);
            expect(screen.queryByText(privateKey)).not.toBeInTheDocument();
            fireEvent.click(screen.getByLabelText("Sales"));
            fireEvent.click(screen.getByRole("button", { name: "Add" }));

            await waitFor(() =>
                expect(createIdentities).toHaveBeenCalledWith({
                    display_name: `Analytics ${provider}`,
                    description: undefined,
                    group_ids: ["group-sales"],
                    config: {
                        type: provider,
                        authentication: {
                            mode: "service_account_key",
                            project_id: googleKey.project_id,
                            client_email: googleKey.client_email,
                            client_id: googleKey.client_id,
                            auth_uri: googleKey.auth_uri,
                            token_uri: googleKey.token_uri,
                            universe_domain: googleKey.universe_domain
                        }
                    },
                    credentials: {
                        kind: "google_service_account_key",
                        private_key_id: googleKey.private_key_id,
                        private_key: privateKey
                    }
                })
            );
        }
    );

    it.each(["bigquery", "gcs"])(
        "presents devserver %s user credentials as ambient and submits no secret",
        async (provider) => {
            featureFlags.ambientGcpEnabled = true;
            featureFlags.ambientGcpDeveloperCredentials = true;
            render(<AddIdentity />);

            await selectProvider(provider);
            expect(
                screen.getByRole("radio", {
                    name: /Ambient local Google credentials/
                })
            ).toBeChecked();
            expect(screen.getByText("gcloud auth login")).toBeVisible();
            expect(screen.queryByRole("status")).not.toBeInTheDocument();
            fireEvent.change(screen.getByLabelText("Identity name"), {
                target: { value: `Local ambient ${provider}` }
            });
            fireEvent.click(screen.getByLabelText("Sales"));
            fireEvent.click(screen.getByRole("button", { name: "Add" }));

            await waitFor(() =>
                expect(createIdentities).toHaveBeenCalledWith({
                    display_name: `Local ambient ${provider}`,
                    description: undefined,
                    group_ids: ["group-sales"],
                    config: {
                        type: provider,
                        authentication: { mode: "ambient" }
                    },
                    credentials: undefined
                })
            );
        }
    );

    it.each(["bigquery", "gcs"])(
        "defaults %s to workload identity and submits no secret when exchange is enabled",
        async (provider) => {
            featureFlags.workloadExchangeEnabled = true;
            render(<AddIdentity />);

            await selectProvider(provider);
            expect(
                screen.getByRole("radio", { name: /Workload identity/ })
            ).toBeChecked();
            expect(
                screen.getByRole("radio", { name: /Workload identity/ })
            ).toBeEnabled();
            expect(screen.queryByRole("status")).not.toBeInTheDocument();
            fireEvent.change(screen.getByLabelText("Identity name"), {
                target: { value: `Workload ${provider}` }
            });
            fireEvent.change(
                screen.getByLabelText("Target service-account email"),
                {
                    target: {
                        value: "storage@identity-project.iam.gserviceaccount.com"
                    }
                }
            );
            fireEvent.click(screen.getByLabelText("Sales"));
            fireEvent.click(screen.getByRole("button", { name: "Add" }));

            await waitFor(() =>
                expect(createIdentities).toHaveBeenCalledWith({
                    display_name: `Workload ${provider}`,
                    description: undefined,
                    group_ids: ["group-sales"],
                    config: {
                        type: provider,
                        authentication: {
                            mode: "workload_identity",
                            service_account_email:
                                "storage@identity-project.iam.gserviceaccount.com"
                        }
                    },
                    credentials: undefined
                })
            );
        }
    );

    it("sends a default-account key for the server to judge", async () => {
        const defaultAccountKey = {
            ...googleKey,
            client_email: "identity-project@appspot.gserviceaccount.com"
        };
        render(<AddIdentity />);

        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Analytics default account" }
        });
        await selectProvider("gcs");
        fireEvent.change(screen.getByLabelText("Service-account JSON file"), {
            target: {
                files: [
                    new File(
                        [JSON.stringify(defaultAccountKey)],
                        "identity.json",
                        { type: "application/json" }
                    )
                ]
            }
        });

        expect(
            await screen.findByLabelText("Parsed service-account details")
        ).toHaveTextContent(defaultAccountKey.client_email);
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createIdentities).toHaveBeenCalledWith(
                expect.objectContaining({
                    config: expect.objectContaining({
                        authentication: expect.objectContaining({
                            client_email: defaultAccountKey.client_email
                        })
                    })
                })
            )
        );
    });

    it("rejects an unreadable Google key without exposing its contents", async () => {
        render(<AddIdentity />);

        await selectProvider("gcs");
        fireEvent.change(screen.getByLabelText("Service-account JSON file"), {
            target: {
                files: [
                    new File(
                        [
                            JSON.stringify({
                                ...googleKey,
                                private_key: undefined
                            })
                        ],
                        "invalid.json",
                        { type: "application/json" }
                    )
                ]
            }
        });

        expect(
            await screen.findByText(
                "Select a complete Google service-account JSON key."
            )
        ).toBeVisible();
        expect(screen.queryByText(privateKey)).not.toBeInTheDocument();
        expect(createIdentities).not.toHaveBeenCalled();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("clears parsed Google secrets when the provider changes", async () => {
        render(<AddIdentity />);

        await selectProvider("gcs");
        fireEvent.change(screen.getByLabelText("Service-account JSON file"), {
            target: {
                files: [
                    new File([JSON.stringify(googleKey)], "identity.json", {
                        type: "application/json"
                    })
                ]
            }
        });
        await screen.findByLabelText("Parsed service-account details");

        await selectProvider("adls");
        expect(
            screen.queryByLabelText("Parsed service-account details")
        ).not.toBeInTheDocument();
        expect(screen.getByLabelText("Client ID")).toHaveValue("");
        expect(screen.queryByText(privateKey)).not.toBeInTheDocument();
    });

    it("clears secrets when Snowflake authentication mode changes", async () => {
        render(<AddIdentity />);

        await selectProvider("snowflake");
        fireEvent.change(screen.getByLabelText("Password"), {
            target: { value: "password-secret" }
        });
        await selectAuthentication("key_pair");
        expect(screen.getByLabelText("Private key")).toHaveValue("");
        fireEvent.change(screen.getByLabelText("Private key"), {
            target: { value: "private-key-secret" }
        });
        await selectAuthentication("password");
        expect(screen.getByLabelText("Password")).toHaveValue("");
    });

    it("preserves the existing AWS AssumeRole payload by default", async () => {
        render(<AddIdentity />);

        await selectProvider("s3");
        fireEvent.change(screen.getByLabelText("Identity name"), {
            target: { value: "Finance AWS role" }
        });
        fireEvent.change(screen.getByLabelText("Role ARN"), {
            target: { value: "arn:aws:iam::123456789012:role/kumo" }
        });
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createIdentities).toHaveBeenCalledWith({
                display_name: "Finance AWS role",
                description: undefined,
                group_ids: ["group-sales"],
                config: {
                    type: "s3",
                    authentication: {
                        mode: "assume_role",
                        partition: "aws",
                        role_arn: "arn:aws:iam::123456789012:role/kumo"
                    }
                },
                credentials: undefined
            })
        );
    });

    it("creates an independent S3-compatible access-key identity", async () => {
        render(<AddIdentity />);

        await selectProvider("s3");
        await selectS3Authentication("S3-compatible access key");
        for (const [label, value] of [
            ["Identity name", "Customer MinIO"],
            ["S3 API endpoint", "https://minio.example.com:9000"],
            ["Region", "us-east-1"],
            ["Access key ID", "minio-access"],
            ["Secret access key", "minio-secret"],
            ["Session token (optional)", "session-token"],
            [
                "CA certificate (optional)",
                "-----BEGIN CERTIFICATE-----\nca\n-----END CERTIFICATE-----"
            ]
        ]) {
            fireEvent.change(screen.getByLabelText(label), {
                target: { value }
            });
        }
        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createIdentities).toHaveBeenCalledWith({
                display_name: "Customer MinIO",
                description: undefined,
                group_ids: ["group-sales"],
                config: {
                    type: "s3",
                    authentication: { mode: "access_key" },
                    connection: {
                        endpoint_url: "https://minio.example.com:9000",
                        region: "us-east-1",
                        addressing_style: "path"
                    }
                },
                credentials: {
                    kind: "s3_access_key",
                    access_key_id: "minio-access",
                    secret_access_key: "minio-secret",
                    session_token: "session-token",
                    ca_certificate:
                        "-----BEGIN CERTIFICATE-----\nca\n-----END CERTIFICATE-----"
                }
            })
        );
    });

    it("clears S3 secrets when the authentication mode changes", async () => {
        render(<AddIdentity />);

        await selectProvider("s3");
        await selectS3Authentication("S3-compatible access key");
        fireEvent.change(screen.getByLabelText("Secret access key"), {
            target: { value: "minio-secret" }
        });

        await selectS3Authentication("AWS role");
        await selectS3Authentication("S3-compatible access key");

        expect(screen.getByLabelText("Secret access key")).toHaveValue("");
    });

    // Regression for kumo-ai/kumo#30041: the two-column field grid pairs each
    // label (col 1) with its control (col 2). Provider and authentication must
    // render as a separate label cell + KUI Select cell — not a single full-row
    // <label> — otherwise the odd cell shifts every following field's label into
    // the input column.
    it("renders provider and authentication as label-left / select-right cells", async () => {
        render(<AddIdentity />);

        const provider = screen.getByLabelText("Provider");
        expect(provider.closest("label")).toBeNull();
        expect(
            document.querySelector(`label[for="${provider.id}"]`)
        ).toBeInTheDocument();

        await selectProvider("snowflake");

        const authentication = screen.getByLabelText("Authentication");
        expect(authentication.closest("label")).toBeNull();
        expect(
            document.querySelector(`label[for="${authentication.id}"]`)
        ).toBeInTheDocument();
    });
});
