// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import {
    createIdentitiesTestByIdentityId,
    deleteIdentitiesByIdentityId,
    getGetIdentitiesQueryKey,
    getIdentitiesConnectorsByIdentityId,
    getIdentitiesGroupsByIdentityId,
    updateIdentitiesByIdentityId,
    updateIdentitiesCredentialsByIdentityId,
    updateIdentitiesGroupsByIdentityId
} from "gen/v0/api/identities/identities";
import { vi } from "vitest";

import { queryClient } from "utils/queryClient";

import IdentitySettings from "../IdentitySettings";

const replacementPrivateKey =
    "-----BEGIN PRIVATE KEY-----\nreplacement-private-material\n-----END PRIVATE KEY-----\n";

const replacementGoogleKey = (clientEmail: string, projectId: string) => ({
    type: "service_account",
    project_id: projectId,
    private_key_id: "replacement-key-id",
    private_key: replacementPrivateKey,
    client_email: clientEmail,
    client_id: "replacement-client-id",
    auth_uri: "https://accounts.google.com/o/oauth2/auth",
    token_uri: "https://oauth2.googleapis.com/token",
    universe_domain: "googleapis.com"
});

vi.mock("react-router", async (importOriginal) => ({
    ...(await importOriginal<typeof import("react-router")>()),
    useNavigate: () => vi.fn()
}));

vi.mock("gen/v0/api/identities/identities", async (importOriginal) => ({
    ...(await importOriginal<
        typeof import("gen/v0/api/identities/identities")
    >()),
    deleteIdentitiesByIdentityId: vi.fn(),
    createIdentitiesTestByIdentityId: vi.fn(),
    getIdentitiesConnectorsByIdentityId: vi.fn(),
    getIdentitiesGroupsByIdentityId: vi.fn(),
    updateIdentitiesByIdentityId: vi.fn(),
    updateIdentitiesCredentialsByIdentityId: vi.fn(),
    updateIdentitiesGroupsByIdentityId: vi.fn(),
    useGetIdentities: () => ({
        data: {
            data: [
                {
                    id: "identity-sales",
                    slug: "sales-service-account",
                    display_name: "Sales service account",
                    description: "Least privilege",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: {
                        configured: true,
                        last_rotated_at: "2026-01-01T00:00:00Z"
                    },
                    config: {
                        type: "databricks",
                        host: "https://workspace.example",
                        client_id: "service-principal"
                    }
                },
                {
                    id: "identity-ambient-s3",
                    slug: "ambient-s3",
                    display_name: "Deployment ambient AWS",
                    description: "Canonical workload identity",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: { configured: false },
                    config: {
                        type: "s3",
                        authentication: { mode: "ambient" }
                    }
                },
                {
                    id: "identity-assume-role-s3",
                    slug: "finance-aws-role",
                    display_name: "Finance AWS role",
                    description: "Explicit assume-role identity",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: {
                        configured: true,
                        last_rotated_at: "2026-01-01T00:00:00Z"
                    },
                    config: {
                        type: "s3",
                        authentication: {
                            mode: "assume_role",
                            partition: "aws",
                            role_arn:
                                "arn:aws:iam::123456789012:role/kumo-finance"
                        }
                    }
                },
                {
                    id: "identity-minio-s3",
                    slug: "customer-minio",
                    display_name: "Customer MinIO",
                    description: "Independent connector storage",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: {
                        configured: true,
                        last_rotated_at: "2026-01-01T00:00:00Z"
                    },
                    config: {
                        type: "s3",
                        authentication: { mode: "access_key" },
                        connection: {
                            endpoint_url: "https://minio.example.com:9000",
                            region: "us-east-1",
                            addressing_style: "path"
                        }
                    }
                },
                {
                    id: "identity-bigquery-key",
                    slug: "analytics-bigquery-key",
                    display_name: "Analytics BigQuery key",
                    description: "Google key identity",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: {
                        configured: true,
                        last_rotated_at: "2026-01-01T00:00:00Z"
                    },
                    config: {
                        type: "bigquery",
                        authentication: {
                            mode: "service_account_key",
                            project_id: "identity-project",
                            client_email:
                                "bigquery@identity-project.iam.gserviceaccount.com",
                            client_id: "111111",
                            auth_uri:
                                "https://accounts.google.com/o/oauth2/auth",
                            token_uri: "https://oauth2.googleapis.com/token",
                            universe_domain: "googleapis.com"
                        }
                    }
                },
                {
                    id: "identity-gcs-key",
                    slug: "storage-gcs-key",
                    display_name: "Storage GCS key",
                    description: "Google key identity",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: {
                        configured: true,
                        last_rotated_at: "2026-01-01T00:00:00Z"
                    },
                    config: {
                        type: "gcs",
                        authentication: {
                            mode: "service_account_key",
                            project_id: "storage-identity-project",
                            client_email:
                                "gcs@storage-identity-project.iam.gserviceaccount.com",
                            client_id: "222222",
                            auth_uri:
                                "https://accounts.google.com/o/oauth2/auth",
                            token_uri: "https://oauth2.googleapis.com/token",
                            universe_domain: "googleapis.com"
                        }
                    }
                },
                {
                    id: "identity-bigquery-workload",
                    slug: "analytics-bigquery-workload",
                    display_name: "Analytics BigQuery workload",
                    description: "Google workload identity",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: { configured: true },
                    config: {
                        type: "bigquery",
                        authentication: {
                            mode: "workload_identity",
                            service_account_email:
                                "bigquery-workload@identity-project.iam.gserviceaccount.com"
                        }
                    }
                },
                {
                    id: "identity-gcs-workload",
                    slug: "storage-gcs-workload",
                    display_name: "Storage GCS workload",
                    description: "Google workload identity",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: { configured: true },
                    config: {
                        type: "gcs",
                        authentication: {
                            mode: "workload_identity",
                            service_account_email:
                                "gcs-workload@identity-project.iam.gserviceaccount.com"
                        }
                    }
                },
                {
                    id: "identity-bigquery-ambient",
                    slug: "analytics-bigquery-ambient",
                    display_name: "Ambient BigQuery",
                    description: "Local or deployment Google credentials",
                    timestamps: {
                        created_at: "2026-01-01T00:00:00Z",
                        updated_at: "2026-01-01T00:00:00Z"
                    },
                    credential_status: { configured: false },
                    config: {
                        type: "bigquery",
                        authentication: { mode: "ambient" }
                    }
                }
            ]
        },
        isLoading: false,
        refetch: vi.fn()
    })
}));

vi.mock("hooks/groups/groups", () => ({
    useKumoGroups: () => ({
        groups: [
            { id: "group-sales", display_name: "Sales" },
            { id: "group-finance", display_name: "Finance" }
        ]
    })
}));

const projectIdentitiesKey = ["/v0/projects/project-1/identities/"];
const adminIdentitiesKey = getGetIdentitiesQueryKey();

describe("Identity administration", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
        vi.mocked(getIdentitiesGroupsByIdentityId).mockResolvedValue({
            data: {
                identity_id: "identity-sales",
                group_ids: ["group-sales"]
            }
        } as never);
        vi.mocked(updateIdentitiesByIdentityId).mockResolvedValue({} as never);
        vi.mocked(updateIdentitiesGroupsByIdentityId).mockResolvedValue(
            {} as never
        );
        vi.mocked(updateIdentitiesCredentialsByIdentityId).mockResolvedValue(
            {} as never
        );
        vi.mocked(createIdentitiesTestByIdentityId).mockResolvedValue(
            {} as never
        );
        vi.mocked(deleteIdentitiesByIdentityId).mockResolvedValue({} as never);
        // Default: nothing blocks deletion, so Delete is allowed once the check
        // resolves. Individual tests override this to exercise the blocked path.
        vi.mocked(getIdentitiesConnectorsByIdentityId).mockResolvedValue({
            data: { identity_id: "identity-assume-role-s3", connectors: [] }
        } as never);
    });

    it("edits assignments, rotates credentials, and tests access", async () => {
        queryClient.setQueryData(projectIdentitiesKey, { data: [] });
        queryClient.setQueryData(adminIdentitiesKey, { data: [] });
        render(<IdentitySettings />);

        expect(screen.getByTestId("admin-page")).toHaveAttribute(
            "data-admin-page-title",
            "Identities"
        );
        expect(
            screen.getByTestId("admin-search-container")
        ).toBeInTheDocument();
        expect(screen.getByText("Sales service account")).toBeInTheDocument();
        fireEvent.click(
            within(
                screen.getByText("Sales service account").closest("tr")!
            ).getByRole("button", { name: "Manage" })
        );

        await screen.findByText("Manage Sales service account");
        expect(screen.getByRole("textbox", { name: "ID" })).toHaveValue(
            "sales-service-account"
        );
        expect(screen.getByRole("textbox", { name: "ID" })).toHaveAttribute(
            "readonly"
        );
        fireEvent.click(screen.getByLabelText("Finance"));
        fireEvent.change(screen.getByLabelText("Description"), {
            target: { value: "Updated" }
        });
        fireEvent.click(
            screen.getByRole("button", {
                name: "Save metadata and groups"
            })
        );

        await waitFor(() =>
            expect(updateIdentitiesGroupsByIdentityId).toHaveBeenCalledWith(
                "identity-sales",
                {
                    group_ids: ["group-sales", "group-finance"]
                }
            )
        );
        expect(updateIdentitiesByIdentityId).toHaveBeenCalledWith(
            "identity-sales",
            {
                display_name: "Sales service account",
                description: "Updated"
            }
        );
        await waitFor(() =>
            expect(
                queryClient.getQueryState(projectIdentitiesKey)?.isInvalidated
            ).toBe(true)
        );
        expect(
            queryClient.getQueryState(adminIdentitiesKey)?.isInvalidated
        ).toBe(true);

        fireEvent.change(screen.getByLabelText("OAuth client secret"), {
            target: { value: "rotated-secret" }
        });
        fireEvent.click(
            screen.getByRole("button", { name: "Rotate and validate" })
        );
        await waitFor(() =>
            expect(
                updateIdentitiesCredentialsByIdentityId
            ).toHaveBeenCalledWith("identity-sales", {
                credentials: {
                    kind: "databricks_oauth",
                    client_secret: "rotated-secret"
                }
            })
        );

        fireEvent.click(screen.getByRole("button", { name: "Test identity" }));
        await waitFor(() =>
            expect(createIdentitiesTestByIdentityId).toHaveBeenCalledWith(
                "identity-sales"
            )
        );
    });

    it("binds and unbinds ambient S3 groups without mutable identity calls", async () => {
        render(<IdentitySettings />);

        const ambientRow = screen
            .getByText("Deployment ambient AWS")
            .closest("tr")!;
        expect(ambientRow).toHaveTextContent(
            "Deployment-managed ambient identity"
        );
        expect(
            within(ambientRow).queryByRole("button", { name: "Delete" })
        ).not.toBeInTheDocument();
        fireEvent.click(
            within(ambientRow).getByRole("button", { name: "Manage" })
        );

        await screen.findByText("Manage Deployment ambient AWS");
        expect(
            screen.getByLabelText("Deployment-managed identity details")
        ).toHaveTextContent("Canonical workload identity");
        expect(
            screen.queryByLabelText("Identity name")
        ).not.toBeInTheDocument();
        expect(screen.queryByLabelText("Description")).not.toBeInTheDocument();
        expect(
            screen.queryByLabelText("External ID (leave empty to clear)")
        ).not.toBeInTheDocument();
        expect(
            screen.queryByRole("button", { name: "Rotate and validate" })
        ).not.toBeInTheDocument();

        fireEvent.click(screen.getByLabelText("Sales"));
        fireEvent.click(screen.getByLabelText("Finance"));
        fireEvent.click(
            screen.getByRole("button", { name: "Save group assignments" })
        );

        await waitFor(() =>
            expect(updateIdentitiesGroupsByIdentityId).toHaveBeenCalledWith(
                "identity-ambient-s3",
                { group_ids: ["group-finance"] }
            )
        );
        expect(updateIdentitiesByIdentityId).not.toHaveBeenCalled();
        expect(updateIdentitiesCredentialsByIdentityId).not.toHaveBeenCalled();

        fireEvent.click(screen.getByRole("button", { name: "Test identity" }));
        await waitFor(() =>
            expect(createIdentitiesTestByIdentityId).toHaveBeenCalledWith(
                "identity-ambient-s3"
            )
        );
    });

    it("keeps metadata, rotation, and delete controls for assume-role S3", async () => {
        render(<IdentitySettings />);

        const explicitRow = screen.getByText("Finance AWS role").closest("tr")!;
        expect(
            within(explicitRow).getByRole("button", { name: "Delete" })
        ).toBeInTheDocument();
        fireEvent.click(
            within(explicitRow).getByRole("button", { name: "Manage" })
        );

        await screen.findByText("Manage Finance AWS role");
        fireEvent.change(screen.getByLabelText("Description"), {
            target: { value: "Updated explicit role" }
        });
        fireEvent.click(screen.getByLabelText("Finance"));
        fireEvent.click(
            screen.getByRole("button", { name: "Save metadata and groups" })
        );
        await waitFor(() =>
            expect(updateIdentitiesByIdentityId).toHaveBeenCalledWith(
                "identity-assume-role-s3",
                {
                    display_name: "Finance AWS role",
                    description: "Updated explicit role"
                }
            )
        );

        fireEvent.change(
            screen.getByLabelText("External ID (leave empty to clear)"),
            { target: { value: "finance-external-id" } }
        );
        fireEvent.click(
            screen.getByRole("button", { name: "Rotate and validate" })
        );
        await waitFor(() =>
            expect(
                updateIdentitiesCredentialsByIdentityId
            ).toHaveBeenCalledWith("identity-assume-role-s3", {
                credentials: {
                    kind: "s3_assume_role",
                    external_id: "finance-external-id"
                }
            })
        );

        queryClient.setQueryData(projectIdentitiesKey, { data: [] });
        fireEvent.click(
            within(explicitRow).getByRole("button", { name: "Delete" })
        );
        const dialog = await screen.findByTestId("delete-identity-dialog");
        // Confirm copy follows the house convention: Title-Case "Delete {Entity}"
        // and `delete identity "{name}"` in the body.
        expect(within(dialog).getByText("Delete Identity")).toBeInTheDocument();
        expect(dialog).toHaveTextContent(
            'Are you sure you want to delete identity "Finance AWS role"?'
        );
        const confirmDelete = within(dialog).getByRole("button", {
            name: "Delete"
        });
        // Delete stays disabled until the blocker check confirms none exist.
        await waitFor(() => expect(confirmDelete).toBeEnabled());
        fireEvent.click(confirmDelete);
        await waitFor(() =>
            expect(deleteIdentitiesByIdentityId).toHaveBeenCalledWith(
                "identity-assume-role-s3"
            )
        );
        await waitFor(() =>
            expect(
                queryClient.getQueryState(projectIdentitiesKey)?.isInvalidated
            ).toBe(true)
        );
    });

    it("shows MinIO connection details and rotates every access-key secret", async () => {
        render(<IdentitySettings />);

        const replacementCa = `-----BEGIN CERTIFICATE-----
MIIB
-----END CERTIFICATE-----`;

        const row = screen.getByText("Customer MinIO").closest("tr")!;
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));

        await screen.findByText("Manage Customer MinIO");
        const details = screen.getByLabelText("S3 authentication details");
        expect(details).toHaveTextContent("S3-compatible access key");
        expect(details).toHaveTextContent("https://minio.example.com:9000");
        expect(details).toHaveTextContent("us-east-1");
        expect(details).toHaveTextContent("path");

        const rotate = screen.getByRole("button", {
            name: "Rotate and validate"
        });
        expect(rotate).toBeDisabled();
        fireEvent.change(screen.getByLabelText("Access key ID"), {
            target: { value: "replacement-access" }
        });
        expect(rotate).toBeDisabled();

        for (const [label, value] of [
            ["Secret access key", "replacement-secret"],
            ["Session token (optional)", "replacement-token"],
            [
                "New CA certificate (optional; leave blank to keep current)",
                replacementCa
            ]
        ]) {
            fireEvent.change(screen.getByLabelText(label), {
                target: { value }
            });
        }
        expect(
            screen.getByLabelText(
                "New CA certificate (optional; leave blank to keep current)"
            ).tagName
        ).toBe("TEXTAREA");
        expect(rotate).toBeEnabled();
        fireEvent.click(rotate);

        await waitFor(() =>
            expect(
                updateIdentitiesCredentialsByIdentityId
            ).toHaveBeenCalledWith("identity-minio-s3", {
                credentials: {
                    kind: "s3_access_key",
                    access_key_id: "replacement-access",
                    secret_access_key: "replacement-secret",
                    session_token: "replacement-token",
                    ca_certificate: replacementCa
                }
            })
        );
    });

    it("keeps the stored CA certificate when rotating only MinIO access keys", async () => {
        render(<IdentitySettings />);

        const row = screen.getByText("Customer MinIO").closest("tr")!;
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));

        await screen.findByText("Manage Customer MinIO");
        expect(
            screen.getByText(/leave the CA certificate blank to keep/i)
        ).toBeInTheDocument();
        fireEvent.change(screen.getByLabelText("Access key ID"), {
            target: { value: "replacement-access" }
        });
        fireEvent.change(screen.getByLabelText("Secret access key"), {
            target: { value: "replacement-secret" }
        });
        fireEvent.click(
            screen.getByRole("button", { name: "Rotate and validate" })
        );

        await waitFor(() =>
            expect(
                updateIdentitiesCredentialsByIdentityId
            ).toHaveBeenCalledWith("identity-minio-s3", {
                credentials: {
                    kind: "s3_access_key",
                    access_key_id: "replacement-access",
                    secret_access_key: "replacement-secret",
                    session_token: null
                }
            })
        );
    });

    it("clears the stored MinIO CA certificate only when explicitly selected", async () => {
        render(<IdentitySettings />);

        const row = screen.getByText("Customer MinIO").closest("tr")!;
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));

        await screen.findByText("Manage Customer MinIO");
        fireEvent.change(screen.getByLabelText("Access key ID"), {
            target: { value: "replacement-access" }
        });
        fireEvent.change(screen.getByLabelText("Secret access key"), {
            target: { value: "replacement-secret" }
        });
        fireEvent.click(
            screen.getByRole("checkbox", {
                name: "Clear current CA certificate"
            })
        );
        expect(
            screen.getByLabelText(
                "New CA certificate (optional; leave blank to keep current)"
            )
        ).toBeDisabled();
        fireEvent.click(
            screen.getByRole("button", { name: "Rotate and validate" })
        );

        await waitFor(() =>
            expect(
                updateIdentitiesCredentialsByIdentityId
            ).toHaveBeenCalledWith("identity-minio-s3", {
                credentials: {
                    kind: "s3_access_key",
                    access_key_id: "replacement-access",
                    secret_access_key: "replacement-secret",
                    session_token: null,
                    ca_certificate: null
                }
            })
        );
    });

    it("retains failed MinIO rotation input and clears it when management closes", async () => {
        vi.mocked(
            updateIdentitiesCredentialsByIdentityId
        ).mockRejectedValueOnce(new Error("rotation failed"));
        render(<IdentitySettings />);

        const row = screen.getByText("Customer MinIO").closest("tr")!;
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));

        await screen.findByText("Manage Customer MinIO");
        fireEvent.change(screen.getByLabelText("Access key ID"), {
            target: { value: "replacement-access" }
        });
        fireEvent.change(screen.getByLabelText("Secret access key"), {
            target: { value: "replacement-secret" }
        });
        fireEvent.click(
            screen.getByRole("button", { name: "Rotate and validate" })
        );

        expect(await screen.findByText("rotation failed")).toBeInTheDocument();
        expect(screen.getByLabelText("Access key ID")).toHaveValue(
            "replacement-access"
        );
        expect(screen.getByLabelText("Secret access key")).toHaveValue(
            "replacement-secret"
        );

        fireEvent.click(screen.getByRole("button", { name: "Close" }));
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));
        await screen.findByText("Manage Customer MinIO");
        expect(screen.getByLabelText("Access key ID")).toHaveValue("");
        expect(screen.getByLabelText("Secret access key")).toHaveValue("");
    });

    it("lists the connectors blocking deletion and refuses to delete", async () => {
        vi.mocked(getIdentitiesConnectorsByIdentityId).mockResolvedValue({
            data: {
                identity_id: "identity-assume-role-s3",
                connectors: [
                    {
                        connector_id: "connector-prod",
                        connector_name: "Prod ingest",
                        project_id: "project-1",
                        project_display_name: "Production analytics"
                    },
                    {
                        connector_id: "connector-nightly",
                        connector_name: "Nightly sync",
                        project_id: "project-2"
                    }
                ]
            }
        } as never);
        render(<IdentitySettings />);

        const explicitRow = screen.getByText("Finance AWS role").closest("tr")!;
        fireEvent.click(
            within(explicitRow).getByRole("button", { name: "Delete" })
        );

        const dialog = await screen.findByTestId("delete-identity-dialog");
        // Falls back to project_id when a project has no display name.
        expect(
            await within(dialog).findByText(
                "Prod ingest (project Production analytics)"
            )
        ).toBeInTheDocument();
        expect(
            within(dialog).getByText("Nightly sync (project project-2)")
        ).toBeInTheDocument();

        // Blocked: the dialog matches the connector "Cannot Delete" alert —
        // failure-stating title, no Delete action at all, dismiss (Cancel) only.
        expect(
            within(dialog).getByText("Cannot Delete Identity")
        ).toBeInTheDocument();
        expect(
            within(dialog).queryByRole("button", { name: "Delete" })
        ).not.toBeInTheDocument();
        expect(
            within(dialog).getByRole("button", { name: "Cancel" })
        ).toBeInTheDocument();
        expect(deleteIdentitiesByIdentityId).not.toHaveBeenCalled();
    });

    it("re-lists blockers when the delete races a 409 conflict", async () => {
        // First check (on open) finds nothing; the delete then loses a race and
        // returns 409, so the backstop re-fetch surfaces the new blocker.
        vi.mocked(getIdentitiesConnectorsByIdentityId)
            .mockResolvedValueOnce({
                data: {
                    identity_id: "identity-assume-role-s3",
                    connectors: []
                }
            } as never)
            .mockResolvedValue({
                data: {
                    identity_id: "identity-assume-role-s3",
                    connectors: [
                        {
                            connector_id: "connector-race",
                            connector_name: "Race connector",
                            project_id: "project-9",
                            project_display_name: "Late project"
                        }
                    ]
                }
            } as never);
        vi.mocked(deleteIdentitiesByIdentityId).mockRejectedValue({
            response: { status: 409 }
        });
        render(<IdentitySettings />);

        const explicitRow = screen.getByText("Finance AWS role").closest("tr")!;
        fireEvent.click(
            within(explicitRow).getByRole("button", { name: "Delete" })
        );

        const dialog = await screen.findByTestId("delete-identity-dialog");
        const confirmDelete = within(dialog).getByRole("button", {
            name: "Delete"
        });
        await waitFor(() => expect(confirmDelete).toBeEnabled());
        fireEvent.click(confirmDelete);

        expect(
            await within(dialog).findByText(
                "Race connector (project Late project)"
            )
        ).toBeInTheDocument();
        // The backstop must not retry the delete or loop.
        expect(deleteIdentitiesByIdentityId).toHaveBeenCalledTimes(1);
        // Surfacing the blocker flips the dialog to its blocked state, which
        // drops the Delete action entirely (Cancel only).
        expect(
            within(dialog).queryByRole("button", { name: "Delete" })
        ).not.toBeInTheDocument();
    });

    it("keeps Delete disabled when the blocker check fails", async () => {
        vi.mocked(getIdentitiesConnectorsByIdentityId).mockRejectedValue(
            new Error("network down")
        );
        render(<IdentitySettings />);

        const explicitRow = screen.getByText("Finance AWS role").closest("tr")!;
        fireEvent.click(
            within(explicitRow).getByRole("button", { name: "Delete" })
        );

        const dialog = await screen.findByTestId("delete-identity-dialog");
        expect(
            await within(dialog).findByText(/couldn't check which connectors/i)
        ).toBeInTheDocument();
        const confirmDelete = within(dialog).getByRole("button", {
            name: "Delete"
        });
        expect(confirmDelete).toBeDisabled();
        fireEvent.click(confirmDelete);
        expect(deleteIdentitiesByIdentityId).not.toHaveBeenCalled();
    });

    it.each([
        [
            "Analytics BigQuery key",
            "identity-bigquery-key",
            "bigquery@identity-project.iam.gserviceaccount.com",
            "identity-project"
        ],
        [
            "Storage GCS key",
            "identity-gcs-key",
            "gcs@storage-identity-project.iam.gserviceaccount.com",
            "storage-identity-project"
        ]
    ])(
        "renders safe key details and rotates the exact account for %s",
        async (displayName, identityId, clientEmail, projectId) => {
            render(<IdentitySettings />);

            const row = screen.getByText(displayName).closest("tr")!;
            expect(row).toHaveTextContent("Service-account key configured");
            fireEvent.click(
                within(row).getByRole("button", { name: "Manage" })
            );

            await screen.findByText(`Manage ${displayName}`);
            const details = screen.getByLabelText(
                "Google authentication details"
            );
            expect(details).toHaveTextContent("service_account_key");
            expect(details).toHaveTextContent(clientEmail);
            expect(details).toHaveTextContent(projectId);
            expect(details).toHaveTextContent("Configured");
            expect(
                screen.queryByText(replacementPrivateKey)
            ).not.toBeInTheDocument();

            const key = replacementGoogleKey(clientEmail, projectId);
            fireEvent.change(
                screen.getByLabelText("Replacement service-account JSON file"),
                {
                    target: {
                        files: [
                            new File(
                                [JSON.stringify(key)],
                                "replacement.json",
                                { type: "application/json" }
                            )
                        ]
                    }
                }
            );
            expect(
                await screen.findByText(
                    `Ready to rotate the key for ${clientEmail}.`
                )
            ).toBeVisible();
            expect(
                screen.queryByText(replacementPrivateKey)
            ).not.toBeInTheDocument();
            fireEvent.click(
                screen.getByRole("button", { name: "Rotate and validate" })
            );

            await waitFor(() =>
                expect(
                    updateIdentitiesCredentialsByIdentityId
                ).toHaveBeenCalledWith(identityId, {
                    authentication: {
                        mode: "service_account_key",
                        project_id: projectId,
                        client_email: clientEmail,
                        client_id: key.client_id,
                        auth_uri: key.auth_uri,
                        token_uri: key.token_uri,
                        universe_domain: key.universe_domain
                    },
                    credentials: {
                        kind: "google_service_account_key",
                        private_key_id: key.private_key_id,
                        private_key: replacementPrivateKey
                    }
                })
            );
        }
    );

    it.each([
        [
            "Analytics BigQuery workload",
            "bigquery-workload@identity-project.iam.gserviceaccount.com"
        ],
        [
            "Storage GCS workload",
            "gcs-workload@identity-project.iam.gserviceaccount.com"
        ]
    ])(
        "renders safe workload details without rotation for %s",
        async (displayName, serviceAccountEmail) => {
            render(<IdentitySettings />);

            const row = screen.getByText(displayName).closest("tr")!;
            expect(row).toHaveTextContent("Workload identity");
            fireEvent.click(
                within(row).getByRole("button", { name: "Manage" })
            );

            await screen.findByText(`Manage ${displayName}`);
            const details = screen.getByLabelText(
                "Google authentication details"
            );
            expect(details).toHaveTextContent("workload_identity");
            expect(details).toHaveTextContent(serviceAccountEmail);
            expect(
                screen.queryByLabelText("Replacement service-account JSON file")
            ).not.toBeInTheDocument();
            expect(
                screen.queryByRole("button", { name: "Rotate and validate" })
            ).not.toBeInTheDocument();
            expect(screen.getByText(/no long-lived credential/i)).toBeVisible();
        }
    );

    it("renders ambient Google details without key-only fields", async () => {
        render(<IdentitySettings />);

        const row = screen.getByText("Ambient BigQuery").closest("tr")!;
        expect(row).toHaveTextContent("Ambient Google credentials");
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));

        await screen.findByText("Manage Ambient BigQuery");
        const details = screen.getByLabelText("Google authentication details");
        expect(details).toHaveTextContent("ambient");
        expect(details).toHaveTextContent(
            "Resolved by this deployment or devserver"
        );
        expect(details).not.toHaveTextContent("Key project");
        expect(
            screen.queryByLabelText("Replacement service-account JSON file")
        ).not.toBeInTheDocument();
    });

    it("rejects a rotation key for a different service account", async () => {
        render(<IdentitySettings />);

        const row = screen.getByText("Analytics BigQuery key").closest("tr")!;
        fireEvent.click(within(row).getByRole("button", { name: "Manage" }));
        await screen.findByText("Manage Analytics BigQuery key");
        fireEvent.change(
            screen.getByLabelText("Replacement service-account JSON file"),
            {
                target: {
                    files: [
                        new File(
                            [
                                JSON.stringify(
                                    replacementGoogleKey(
                                        "different@identity-project.iam.gserviceaccount.com",
                                        "identity-project"
                                    )
                                )
                            ],
                            "mismatch.json",
                            { type: "application/json" }
                        )
                    ]
                }
            }
        );

        expect(
            await screen.findByText(
                "Select a valid service-account JSON key for this exact configured account."
            )
        ).toBeVisible();
        expect(
            screen.getByRole("button", { name: "Rotate and validate" })
        ).toBeDisabled();
        expect(updateIdentitiesCredentialsByIdentityId).not.toHaveBeenCalled();
        expect(
            screen.queryByText(replacementPrivateKey)
        ).not.toBeInTheDocument();
    });
});
