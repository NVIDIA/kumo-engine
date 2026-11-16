// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createProjectsConnectorsByProjectId } from "gen/v0/api/connectors/connectors";
import {
    useGetIdentities,
    useGetProjectsIdentitiesByProjectId
} from "gen/v0/api/identities/identities";
import { IdentityResponse } from "gen/v0/types";
import { vi } from "vitest";

import { ReactQueryKeys } from "utils/reactQueryKeys";

import IdentityConnectorForm from "../IdentityConnectorForm";

const navigate = vi.fn();
const project = vi.hoisted(() => ({ id: "project-sales" }));
const queryClientMock = vi.hoisted(() => ({ invalidateQueries: vi.fn() }));
const admin = vi.hoisted(() => ({ isAdmin: false }));

vi.mock("@tanstack/react-query", async (importOriginal) => {
    const actual =
        await importOriginal<typeof import("@tanstack/react-query")>();
    return {
        ...actual,
        useQueryClient: () => queryClientMock
    };
});
vi.mock("gen/v0/api/connectors/connectors", () => ({
    createProjectsConnectorsByProjectId: vi.fn()
}));
vi.mock("gen/v0/api/identities/identities", () => ({
    useGetIdentities: vi.fn(),
    useGetProjectsIdentitiesByProjectId: vi.fn()
}));
vi.mock("hooks/auth/admin", () => ({
    useAdminInfo: () => ({ isAdmin: admin.isAdmin })
}));
vi.mock("hooks/project/projects", () => ({
    useProjectId: () => project.id
}));
vi.mock("utils/navigator", () => ({
    useCustomNavigation: () => navigate
}));

const timestamps = {
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z"
};

const identities: IdentityResponse[] = [
    {
        id: "identity-databricks",
        slug: "sales-databricks",
        display_name: "Sales Databricks",
        description: "",
        config: {
            type: "databricks",
            host: "https://workspace.example",
            client_id: "client-id"
        },
        credential_status: { configured: true },
        timestamps
    },
    {
        id: "identity-snowflake",
        slug: "finance-snowflake",
        display_name: "Finance Snowflake",
        description: "",
        config: {
            type: "snowflake",
            account: "acme-prod",
            username: "KUMO_FINANCE",
            authentication: "key_pair"
        },
        credential_status: { configured: true },
        timestamps
    },
    {
        id: "identity-bigquery",
        slug: "analytics-bigquery",
        display_name: "Analytics BigQuery",
        description: "",
        config: {
            type: "bigquery",
            authentication: {
                mode: "service_account_key",
                project_id: "identity-project",
                client_email: "kumo@identity-project.iam.gserviceaccount.com",
                client_id: "bigquery-client-id",
                auth_uri: "https://accounts.google.com/o/oauth2/auth",
                token_uri: "https://oauth2.googleapis.com/token",
                universe_domain: "googleapis.com"
            }
        },
        credential_status: { configured: true },
        timestamps
    },
    {
        id: "identity-s3",
        slug: "production-aws",
        display_name: "Production AWS",
        description: "",
        config: {
            type: "s3",
            authentication: {
                mode: "assume_role",
                partition: "aws",
                role_arn: "arn:aws:iam::123456789012:role/kumo-reader"
            }
        },
        credential_status: {
            configured: true,
            last_rotated_at: "2026-06-01T00:00:00Z"
        },
        timestamps
    },
    {
        id: "identity-gcs",
        slug: "data-lake-gcs",
        display_name: "Data Lake GCS",
        description: "",
        config: {
            type: "gcs",
            authentication: {
                mode: "workload_identity",
                service_account_email:
                    "kumo-storage@example.iam.gserviceaccount.com"
            }
        },
        credential_status: { configured: true },
        timestamps
    },
    {
        id: "identity-adls",
        slug: "production-azure",
        display_name: "Production Azure",
        description: "",
        config: {
            type: "adls",
            tenant_id: "tenant-id",
            client_id: "client-id",
            cloud: "public"
        },
        credential_status: { configured: true },
        timestamps
    }
];

const ambientS3Identity: IdentityResponse = {
    id: "identity-ambient-s3",
    slug: "ambient-s3",
    display_name: "Deployment ambient AWS",
    description: "Canonical workload identity",
    config: {
        type: "s3",
        authentication: { mode: "ambient" }
    },
    credential_status: { configured: false },
    timestamps
};

const minioS3Identity: IdentityResponse = {
    id: "identity-minio-s3",
    slug: "customer-minio",
    display_name: "Customer MinIO",
    description: "Independent connector storage",
    config: {
        type: "s3",
        authentication: { mode: "access_key" },
        connection: {
            endpoint_url: "https://minio.example.com:9000",
            region: "us-east-1",
            addressing_style: "path"
        }
    },
    credential_status: { configured: true },
    timestamps
};

const connectorTypeLabels: Record<string, string> = {
    databricks: "Databricks",
    snowflake: "Snowflake",
    bigquery: "BigQuery",
    s3: "Amazon S3",
    gcs: "Google Cloud Storage",
    adls: "Azure Data Lake Storage"
};

const selectType = (connectorType: string) =>
    fireEvent.click(
        screen.getByRole("radio", {
            name: new RegExp(`^${connectorTypeLabels[connectorType]}`)
        })
    );

// The catalog Select is a button plus a popover listbox, so a choice is made
// by opening it and clicking the option rather than by setting a value.
const selectIdentity = async (displayName: string) => {
    await userEvent.click(screen.getByLabelText("Identity"));
    await userEvent.click(
        await screen.findByRole("option", {
            name: new RegExp(`^${displayName}`)
        })
    );
};

// Details render as dt/dd pairs, so a row is found by its label and read from
// the description beside it.
const detailLabels = () =>
    Array.from(document.querySelectorAll("dl dt")).map(
        (node) => node.textContent
    );
const detailRow = (label: string) =>
    screen.getByText(label, { selector: "dt" }).nextElementSibling;

const fillDatabricksTarget = () => {
    fireEvent.change(screen.getByLabelText("Connector name"), {
        target: { value: "sales" }
    });
    fireEvent.change(screen.getByLabelText("SQL warehouse ID"), {
        target: { value: "warehouse-1" }
    });
    fireEvent.change(screen.getByLabelText("Catalog"), {
        target: { value: "sales" }
    });
    fireEvent.change(screen.getByLabelText("Schema"), {
        target: { value: "analytics" }
    });
};

const fillAdlsTarget = () => {
    fireEvent.change(screen.getByLabelText("Connector name"), {
        target: { value: "curated" }
    });
    fireEvent.change(screen.getByLabelText("ADLS base location (optional)"), {
        target: {
            value: "abfss://curated@companydata.dfs.core.windows.net"
        }
    });
};

describe("IdentityConnectorForm", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        project.id = "project-sales";
        admin.isAdmin = false;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: identities },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: undefined,
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetIdentities>);
        vi.mocked(createProjectsConnectorsByProjectId).mockResolvedValue({
            data: { id: "connector-sales" }
        } as never);
    });

    it("starts with six accessible connector cards and no identity controls", () => {
        render(<IdentityConnectorForm />);

        expect(
            screen.getByRole("group", { name: "Connector type" })
        ).toBeInTheDocument();
        Object.values(connectorTypeLabels).forEach((label) => {
            expect(
                screen.getByRole("radio", { name: label })
            ).toBeInTheDocument();
        });
        expect(screen.queryByLabelText("Identity")).not.toBeInTheDocument();
        expect(detailLabels()).toHaveLength(0);
        expect(
            screen.queryByLabelText("Connector name")
        ).not.toBeInTheDocument();
    });

    it("disables connector choices until a project is selected", () => {
        project.id = "";
        render(<IdentityConnectorForm />);

        screen.getAllByRole("radio").forEach((option) => {
            expect(option).toBeDisabled();
        });
        expect(
            screen.getByText("Select a project before creating a connector.")
        ).toBeInTheDocument();
        expect(screen.queryByLabelText("Identity")).not.toBeInTheDocument();
    });

    it("supports keyboard selection and exposes one checked connector", async () => {
        render(<IdentityConnectorForm />);

        const databricks = screen.getByRole("radio", { name: "Databricks" });
        databricks.focus();
        await userEvent.keyboard(" ");

        expect(databricks).toBeChecked();
        expect(
            screen
                .getAllByRole("radio")
                .filter((item) => (item as HTMLInputElement).checked)
        ).toHaveLength(1);
        expect(screen.getByLabelText("Identity")).toBeInTheDocument();
        expect(screen.getByText("Selected")).toBeInTheDocument();
    });

    it("starts with no identity chosen and blocks Add until one is picked", async () => {
        render(<IdentityConnectorForm />);

        selectType("databricks");
        fillDatabricksTarget();

        // Every connector stores exactly one identity, and the user now names
        // it rather than leaving the server to resolve one.
        expect(screen.getByText("Select an identity")).toBeInTheDocument();
        expect(detailLabels()).toHaveLength(0);
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();

        await selectIdentity("Sales Databricks");
        expect(screen.getByRole("button", { name: "Add" })).toBeEnabled();
    });

    it("sends the chosen identity with the created connector", async () => {
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        fillDatabricksTarget();
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-databricks",
                    name: "sales",
                    description: undefined,
                    config: {
                        type: "databricks",
                        warehouse_id: "warehouse-1",
                        catalog: "sales",
                        schema_name: "analytics"
                    }
                }
            )
        );
        expect(navigate).toHaveBeenCalledWith(
            "/project/project-sales/connectors/connector-sales"
        );
    });

    it("invalidates the connector list cache after a successful create", async () => {
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        fillDatabricksTarget();
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        // The freshly created connector must be reflected in the list without a
        // manual page refresh, so the cached connector list is invalidated.
        await waitFor(() =>
            expect(queryClientMock.invalidateQueries).toHaveBeenCalledWith({
                queryKey: [ReactQueryKeys.ConnectorList, "project-sales"]
            })
        );
    });

    it("shows the page loading overlay while the connector is being created", async () => {
        type CreateResult = Awaited<
            ReturnType<typeof createProjectsConnectorsByProjectId>
        >;
        let releaseCreate: (value: CreateResult) => void = () => {};
        vi.mocked(createProjectsConnectorsByProjectId).mockReturnValueOnce(
            new Promise<CreateResult>((resolve) => {
                releaseCreate = resolve;
            })
        );
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        fillDatabricksTarget();
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        // Same overlay Add table and Add graph show; without it a user cannot
        // tell the create registered and may submit twice.
        expect(await screen.findByTestId("animated-loading")).toBeVisible();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();

        releaseCreate({ data: { id: "connector-sales" } } as CreateResult);
        await waitFor(() => expect(navigate).toHaveBeenCalled());
    });

    it("does not invalidate the connector list when create fails", async () => {
        vi.mocked(createProjectsConnectorsByProjectId).mockRejectedValueOnce({
            response: {
                status: 409,
                data: {
                    detail: "Connector named sales already exists in this project"
                }
            }
        });
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        fillDatabricksTarget();
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        expect(
            await screen.findByText(/Connector named sales/)
        ).toBeInTheDocument();
        expect(queryClientMock.invalidateQueries).not.toHaveBeenCalled();
    });

    it("creates a project-scoped Snowflake connector with an explicit identity", async () => {
        render(<IdentityConnectorForm />);

        selectType("snowflake");
        await selectIdentity("Finance Snowflake");
        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "finance warehouse" }
        });
        fireEvent.change(screen.getByLabelText("Description (optional)"), {
            target: { value: "Finance reporting" }
        });
        fireEvent.change(screen.getByLabelText("Role (optional)"), {
            target: { value: "KUMO_READER" }
        });
        fireEvent.change(screen.getByLabelText("Warehouse"), {
            target: { value: "REPORTING_WH" }
        });
        fireEvent.change(screen.getByLabelText("Database"), {
            target: { value: "FINANCE" }
        });
        fireEvent.change(screen.getByLabelText("Schema"), {
            target: { value: "ANALYTICS" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-snowflake",
                    name: "finance warehouse",
                    description: "Finance reporting",
                    config: {
                        type: "snowflake",
                        role: "KUMO_READER",
                        warehouse: "REPORTING_WH",
                        database: "FINANCE",
                        schema_name: "ANALYTICS"
                    }
                }
            )
        );
    });

    it("resets identity when switching to BigQuery and creates the exact target", async () => {
        render(<IdentityConnectorForm />);

        selectType("snowflake");
        await selectIdentity("Finance Snowflake");
        selectType("bigquery");
        expect(screen.getByText("Select an identity")).toBeInTheDocument();
        await selectIdentity("Analytics BigQuery");
        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "product analytics" }
        });
        fireEvent.change(screen.getByLabelText("Data project ID"), {
            target: { value: "product-data" }
        });
        fireEvent.change(screen.getByLabelText("Dataset"), {
            target: { value: "events" }
        });
        fireEvent.change(
            screen.getByLabelText("Billing project ID (optional)"),
            {
                target: { value: "central-billing" }
            }
        );
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-bigquery",
                    name: "product analytics",
                    description: undefined,
                    config: {
                        type: "bigquery",
                        project_id: "product-data",
                        dataset_id: "events",
                        location: "US",
                        billing_project_id: "central-billing"
                    }
                }
            )
        );
    });

    it("creates a project-scoped GCS connector with full target configuration", async () => {
        render(<IdentityConnectorForm />);

        selectType("gcs");
        await selectIdentity("Data Lake GCS");
        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "training lake" }
        });
        fireEvent.change(
            screen.getByLabelText("GCS base location (optional)"),
            {
                target: { value: "gs://training-data/curated/2026" }
            }
        );
        fireEvent.change(
            screen.getByLabelText("Requester-pays project (optional)"),
            { target: { value: "requester-project" } }
        );
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-gcs",
                    name: "training lake",
                    description: undefined,
                    config: {
                        type: "gcs",
                        target: {
                            mode: "rooted",
                            bucket: "training-data",
                            prefix: "curated/2026"
                        },
                        requester_pays_project: "requester-project"
                    }
                }
            )
        );
    });

    it("offers only same-type identities and sends an explicit selection", async () => {
        render(<IdentityConnectorForm />);

        selectType("s3");
        await userEvent.click(screen.getByLabelText("Identity"));
        const options = await screen.findAllByRole("option");
        expect(options).toHaveLength(1);
        expect(options[0]).toHaveTextContent("Production AWS");
        // The principal rides along as the option's second line so the choice
        // is legible before it is made.
        expect(options[0]).toHaveTextContent(
            "arn:aws:iam::123456789012:role/kumo-reader"
        );
        expect(
            screen.queryByRole("option", { name: /Automatically select/ })
        ).not.toBeInTheDocument();

        await userEvent.click(options[0]);
        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "events" }
        });
        fireEvent.change(screen.getByLabelText("S3 base location (optional)"), {
            target: { value: "s3://company-events/raw/2026/events" }
        });
        fireEvent.change(screen.getByLabelText("Region (optional)"), {
            target: { value: "us-west-2" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-s3",
                    name: "events",
                    description: undefined,
                    config: {
                        type: "s3",
                        target: {
                            mode: "rooted",
                            bucket: "company-events",
                            prefix: "raw/2026/events",
                            region: "us-west-2"
                        }
                    }
                }
            )
        );
    });

    it("creates an any-URI S3 connector when the optional base is blank", async () => {
        render(<IdentityConnectorForm />);

        selectType("s3");
        await selectIdentity("Production AWS");
        expect(screen.getByLabelText("Connector name")).toHaveValue("s3");
        expect(
            screen.getByLabelText("S3 base location (optional)")
        ).toHaveValue("");
        expect(
            screen.queryByLabelText("Region (optional)")
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText(/rooted|rootless|any_uri/i)
        ).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-s3",
                    name: "s3",
                    description: undefined,
                    config: { type: "s3", target: { mode: "any_uri" } }
                }
            )
        );
    });

    it("uses an S3-compatible identity without duplicating its connection settings", async () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: [minioS3Identity, ...identities] },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("s3");
        await selectIdentity("Customer MinIO");

        expect(detailRow("Authentication")).toHaveTextContent(
            "S3-compatible access key"
        );
        expect(detailRow("S3 API endpoint")).toHaveTextContent(
            "https://minio.example.com:9000"
        );
        expect(detailRow("Region")).toHaveTextContent("us-east-1");

        fireEvent.click(screen.getByRole("button", { name: "Add" }));
        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-minio-s3",
                    name: "s3",
                    description: undefined,
                    config: { type: "s3", target: { mode: "any_uri" } }
                }
            )
        );
    });

    it("renders target fields before an identity is selected", () => {
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(screen.getByLabelText("Warehouse")).toBeInTheDocument();
        expect(screen.getByLabelText("Database")).toBeInTheDocument();
        expect(screen.getByLabelText("Schema")).toBeInTheDocument();
        expect(screen.getByText("Select an identity")).toBeInTheDocument();
    });

    it("explains what an identity is on hover and keyboard focus", async () => {
        const user = userEvent.setup();
        render(<IdentityConnectorForm />);
        selectType("s3");

        const help = screen.getByRole("button", {
            name: "What is an identity"
        });
        expect(
            screen.queryByText(/The credentials Kumo uses/)
        ).not.toBeInTheDocument();

        await user.hover(help);
        expect(await screen.findByRole("tooltip")).toHaveTextContent(
            "The credentials Kumo uses to reach your data source"
        );
        await user.unhover(help);
        help.focus();
        expect(await screen.findByRole("tooltip")).toHaveTextContent(
            "you can choose any identity assigned to this project's group"
        );
        expect(screen.getByRole("tooltip")).not.toHaveTextContent(
            /[Aa]utomatic/
        );
    });

    it("tells a non-admin to ask for access when no S3 identity is bound", () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: {
                data: identities.filter((item) => item.config.type !== "s3")
            },
            isLoading: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("s3");
        expect(screen.getByLabelText("Identity")).toBeDisabled();
        expect(
            screen.getByText(
                "No S3 identities are available for this project. Ask an administrator to add project access before creating this connector."
            )
        ).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("clears provider state, failed create errors, and identity when connector type changes", async () => {
        vi.mocked(createProjectsConnectorsByProjectId).mockRejectedValueOnce({
            response: {
                status: 409,
                data: {
                    detail: "Connector named s3 already exists in this project"
                }
            }
        });
        render(<IdentityConnectorForm />);

        selectType("s3");
        await selectIdentity("Production AWS");
        fireEvent.change(screen.getByLabelText("S3 base location (optional)"), {
            target: { value: "s3://company-events/deep/input" }
        });
        fireEvent.click(screen.getByRole("button", { name: "Add" }));
        expect(
            await screen.findByText(/Connector named s3/)
        ).toBeInTheDocument();

        selectType("databricks");

        expect(screen.getByText("Select an identity")).toBeInTheDocument();
        expect(detailLabels()).toHaveLength(0);
        expect(screen.getByLabelText("Connector name")).toHaveValue("");
        expect(
            screen.queryByLabelText("S3 base location (optional)")
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText(/Connector named s3/)
        ).not.toBeInTheDocument();

        await userEvent.click(screen.getByLabelText("Identity"));
        const options = await screen.findAllByRole("option");
        expect(options).toHaveLength(1);
        expect(options[0]).toHaveTextContent("Sales Databricks");
        expect(
            screen
                .getAllByRole("radio")
                .filter((item) => (item as HTMLInputElement).checked)
        ).toHaveLength(1);
    });

    it("validates the optional S3 base before enabling creation", () => {
        render(<IdentityConnectorForm />);
        selectType("s3");

        fireEvent.change(screen.getByLabelText("S3 base location (optional)"), {
            target: { value: "https://company-events/deep/input" }
        });
        expect(
            screen.getByText(
                "Enter a full S3 base location such as s3://bucket/prefix."
            )
        ).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    // Every label below is the one Add identity uses for the same field, so a
    // value is called the same thing on both pages. The tuple type is explicit
    // because inference widens the rows to string[][], which would let a
    // label-without-a-value through as a silent undefined.
    it.each<[string, string, [string, string][]]>([
        [
            "databricks",
            "Sales Databricks",
            [
                ["Workspace host", "https://workspace.example"],
                ["OAuth client ID", "client-id"]
            ]
        ],
        [
            "snowflake",
            "Finance Snowflake",
            [
                ["Account", "acme-prod"],
                ["Service-account username", "KUMO_FINANCE"],
                // Part of Snowflake's identity_key, so it is what separates
                // two identities sharing an account and username.
                ["Authentication", "Key pair"]
            ]
        ],
        [
            "s3",
            "Production AWS",
            [
                ["Role ARN", "arn:aws:iam::123456789012:role/kumo-reader"],
                ["AWS partition", "aws"]
            ]
        ],
        [
            "adls",
            "Production Azure",
            [
                ["Tenant ID", "tenant-id"],
                ["Client ID", "client-id"],
                // Add identity takes this as free text, so the raw stored
                // value is what gets shown back.
                ["Azure cloud", "public"]
            ]
        ]
    ])(
        "labels %s identity details with Add identity's wording",
        async (type, identityName, expected) => {
            render(<IdentityConnectorForm />);

            selectType(type);
            // Nothing is described until a choice is made, which is what keeps
            // Connector details above the fold.
            expect(detailLabels()).toHaveLength(0);

            await selectIdentity(identityName);

            expect(detailLabels()).toHaveLength(expected.length);
            expected.forEach(([label, value]) =>
                expect(detailRow(label)).toHaveTextContent(value)
            );
            // Credentials was dropped: it reports stored secret material, so
            // working ambient identities always read as "not configured".
            expect(screen.queryByText(/Credentials/)).not.toBeInTheDocument();
            expect(screen.queryByText(/Principal/)).not.toBeInTheDocument();
            expect(screen.queryByText(/Last rotated/)).not.toBeInTheDocument();
            expect(
                screen.queryByText(/secret|private key/i)
            ).not.toBeInTheDocument();
        }
    );

    it("names each Google authentication mode with the fields it returns", async () => {
        const googleIdentities: IdentityResponse[] = [
            identities[2],
            {
                ...identities[2],
                id: "identity-bigquery-workload",
                display_name: "Impersonated BigQuery",
                config: {
                    type: "bigquery",
                    authentication: {
                        mode: "workload_identity",
                        service_account_email:
                            "target@identity-project.iam.gserviceaccount.com"
                    }
                }
            },
            {
                ...identities[2],
                id: "identity-bigquery-ambient",
                display_name: "Ambient BigQuery",
                config: {
                    type: "bigquery",
                    authentication: { mode: "ambient" }
                }
            }
        ];
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: googleIdentities },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);
        selectType("bigquery");

        // The mode is a row of its own rather than something the admin has to
        // work into the identity's name.
        await selectIdentity("Analytics BigQuery");
        expect(detailRow("Authentication")).toHaveTextContent(
            "Service account JSON key"
        );
        expect(detailRow("Service account")).toHaveTextContent(
            "kumo@identity-project.iam.gserviceaccount.com"
        );
        expect(detailRow("Project")).toHaveTextContent("identity-project");

        await selectIdentity("Impersonated BigQuery");
        expect(detailRow("Authentication")).toHaveTextContent(
            "Workload identity (impersonation)"
        );
        expect(detailRow("Target service-account email")).toHaveTextContent(
            "target@identity-project.iam.gserviceaccount.com"
        );

        // Ambient returns no fields at all, so the mode is all there is.
        await selectIdentity("Ambient BigQuery");
        expect(detailLabels()).toEqual(["Authentication"]);
        expect(detailRow("Authentication")).toHaveTextContent(
            "Kumo's GCP identity"
        );
    });

    it("keeps the mode out of the dropdown line unless it is all there is", async () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: {
                data: [
                    identities[2],
                    {
                        ...identities[2],
                        id: "identity-bigquery-ambient",
                        display_name: "Ambient BigQuery",
                        config: {
                            type: "bigquery",
                            authentication: { mode: "ambient" }
                        }
                    }
                ]
            },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("bigquery");
        await userEvent.click(screen.getByLabelText("Identity"));
        const options = await screen.findAllByRole("option");

        // Two JSON-key identities would read identically if the option line
        // showed the mode, so it shows the account that separates them.
        expect(
            options.find((option) => option.textContent?.includes("Analytics"))
        ).toHaveTextContent("kumo@identity-project.iam.gserviceaccount.com");
        expect(
            options.find((option) => option.textContent?.includes("Ambient"))
        ).toHaveTextContent("Kumo's GCP identity");
    });

    it("describes an ambient S3 identity by its mode", async () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: [ambientS3Identity] },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("s3");
        await selectIdentity("Deployment ambient AWS");

        expect(detailRow("Authentication")).toHaveTextContent(
            "Verified ambient AWS workload identity"
        );
    });

    it("uses the public ADLS endpoint default for the chosen identity", async () => {
        render(<IdentityConnectorForm />);

        selectType("adls");
        await selectIdentity("Production Azure");
        fillAdlsTarget();
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                {
                    identity_id: "identity-adls",
                    name: "curated",
                    description: undefined,
                    config: {
                        type: "adls",
                        target: {
                            mode: "rooted",
                            storage_account: "companydata",
                            filesystem: "curated",
                            prefix: "",
                            endpoint_suffix: "dfs.core.windows.net"
                        }
                    }
                }
            )
        );
    });

    it.each([
        ["gcs", "gcs", "GCS base location (optional)", "Data Lake GCS"],
        ["adls", "adls", "ADLS base location (optional)", "Production Azure"]
    ])(
        "creates an any-URI %s connector when its optional base is blank",
        async (type, name, baseLabel, identityName) => {
            render(<IdentityConnectorForm />);
            selectType(type);
            await selectIdentity(identityName);

            expect(screen.getByLabelText("Connector name")).toHaveValue(name);
            expect(screen.getByLabelText(baseLabel)).toHaveValue("");
            expect(
                screen.getByText(
                    /source-only any-URI mode\. It cannot list tables or write outputs/i
                )
            ).toBeInTheDocument();
            fireEvent.click(screen.getByRole("button", { name: "Add" }));

            await waitFor(() =>
                expect(
                    createProjectsConnectorsByProjectId
                ).toHaveBeenCalledWith("project-sales", {
                    identity_id: `identity-${type}`,
                    name,
                    description: undefined,
                    config: {
                        type,
                        target: { mode: "any_uri" },
                        ...(type === "gcs"
                            ? { requester_pays_project: null }
                            : {})
                    }
                })
            );
        }
    );

    it("requires a SQL warehouse ID and offers no cluster ID field", async () => {
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        expect(screen.queryByLabelText("Cluster ID")).not.toBeInTheDocument();

        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "sales" }
        });
        fireEvent.change(screen.getByLabelText("Catalog"), {
            target: { value: "sales" }
        });
        fireEvent.change(screen.getByLabelText("Schema"), {
            target: { value: "analytics" }
        });
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();

        fireEvent.change(screen.getByLabelText("SQL warehouse ID"), {
            target: { value: "warehouse-1" }
        });
        expect(screen.getByRole("button", { name: "Add" })).toBeEnabled();
    });

    it.each([
        [
            409,
            "More than one identity can access this target. Choose an explicit identity."
        ],
        [422, "No available Databricks identity could access this target."]
    ])("shows the sanitized %s resolution error", async (status, detail) => {
        vi.mocked(createProjectsConnectorsByProjectId).mockRejectedValueOnce({
            response: { status, data: { detail } }
        });
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        fillDatabricksTarget();
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        expect(await screen.findByText(detail)).toBeInTheDocument();
    });

    const withoutSnowflake = identities.filter(
        (item) => item.config.type !== "snowflake"
    );

    it("tells an admin when an identity exists but is not assigned to the project's group", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: { data: identities },
            isLoading: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(vi.mocked(useGetIdentities)).toHaveBeenLastCalledWith({
            query: { enabled: true }
        });
        expect(screen.getByRole("status")).toHaveTextContent(
            "1 Snowflake identity exists but is not assigned to this project's group. Assign groups in Identity settings."
        );
        expect(
            screen.queryByText(/No explicit Snowflake identities available/)
        ).not.toBeInTheDocument();

        fireEvent.click(
            screen.getByRole("button", { name: "Identity settings" })
        );
        expect(navigate).toHaveBeenCalledWith("/admin/identity-settings");
    });

    it("waits for the admin identity lookup before showing an empty state", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: undefined,
            isLoading: true,
            isError: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(
            screen.queryByText(/No explicit Snowflake identities available/)
        ).not.toBeInTheDocument();
        expect(screen.queryByRole("status")).not.toBeInTheDocument();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("surfaces an admin identity lookup error instead of the empty state", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: undefined,
            isLoading: false,
            isError: true
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(screen.getByRole("alert")).toHaveTextContent(
            "Unassigned identities could not be loaded. Refresh the page, or contact an administrator if the problem persists."
        );
        expect(
            screen.queryByText(/No explicit Snowflake identities available/)
        ).not.toBeInTheDocument();
        expect(
            screen.queryByRole("button", { name: "Identity settings" })
        ).not.toBeInTheDocument();
    });

    it("counts only unassigned identities of the selected type for admins", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: {
                data: [
                    ...identities,
                    {
                        ...identities[1],
                        id: "identity-snowflake-2",
                        slug: "ops-snowflake",
                        display_name: "Ops Snowflake"
                    }
                ]
            },
            isLoading: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(screen.getByRole("status")).toHaveTextContent(
            "2 Snowflake identities exist but none is assigned to this project's group."
        );
    });

    it("does not report ambient S3 as an unassigned explicit identity", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: {
                data: identities.filter((item) => item.config.type !== "s3")
            },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: { data: [ambientS3Identity] },
            isLoading: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("s3");

        expect(screen.queryByRole("status")).not.toBeInTheDocument();
        expect(
            screen.getByText(
                "No S3 identities are available for this project. Ask an administrator to add project access before creating this connector."
            )
        ).toBeInTheDocument();
    });

    it("counts explicit S3 identities without counting ambient S3", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: {
                data: identities.filter((item) => item.config.type !== "s3")
            },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: {
                data: [
                    ambientS3Identity,
                    ...identities.filter((item) => item.config.type === "s3")
                ]
            },
            isLoading: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("s3");

        expect(screen.getByRole("status")).toHaveTextContent(
            "1 S3 identity exists but is not assigned to this project's group."
        );
    });

    it("ignores cached admin identity data for non-admins", () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: { data: identities },
            isLoading: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(vi.mocked(useGetIdentities)).toHaveBeenLastCalledWith({
            query: { enabled: false }
        });
        expect(screen.queryByRole("status")).not.toBeInTheDocument();
        expect(
            screen.queryByRole("button", { name: "Identity settings" })
        ).not.toBeInTheDocument();
        expect(
            screen.getByText(
                "No Snowflake identities are available for this project. Ask an administrator to add project access before creating this connector."
            )
        ).toBeInTheDocument();
    });

    it("falls back to the generic empty state when no identity of the type exists anywhere", () => {
        admin.isAdmin = true;
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        vi.mocked(useGetIdentities).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false
        } as ReturnType<typeof useGetIdentities>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(
            screen.getByText(
                "No Snowflake identities are available for this project. Ask an administrator to add project access before creating this connector."
            )
        ).toBeInTheDocument();
    });

    it("offers a working retry when the initial identities load fails", () => {
        const refetch = vi.fn();
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: undefined,
            isLoading: false,
            isError: true,
            refetch
        } as unknown as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");

        expect(screen.getByRole("alert")).toHaveTextContent(
            "Identities for this project could not be loaded. Try again, or contact an administrator if the problem persists."
        );
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();

        // Refreshing the page would discard everything already typed into the
        // form, so the recovery has to refetch in place.
        fireEvent.click(screen.getByRole("button", { name: "Try again" }));
        expect(refetch).toHaveBeenCalled();
    });

    it("stays usable when a refetch fails but cached identities remain", async () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: identities },
            isLoading: false,
            isError: true
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        // S3 needs no extra fields, so Add reflects the identity gate alone.
        selectType("s3");
        await selectIdentity("Production AWS");

        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Add" })).toBeEnabled();
    });

    const fillSnowflakeTarget = () => {
        fireEvent.change(screen.getByLabelText("Connector name"), {
            target: { value: "finance" }
        });
        fireEvent.change(screen.getByLabelText("Warehouse"), {
            target: { value: "COMPUTE_WH" }
        });
        fireEvent.change(screen.getByLabelText("Database"), {
            target: { value: "FINANCE" }
        });
        fireEvent.change(screen.getByLabelText("Schema"), {
            target: { value: "PUBLIC" }
        });
    };

    it("keeps Add disabled when no identity of the type is bound", () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: withoutSnowflake },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("snowflake");
        fillSnowflakeTarget();

        // There is nothing to choose, so the create could only fail
        // server-side.
        expect(screen.getByLabelText("Identity")).toBeDisabled();
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("keeps Add disabled while project identities are still loading", () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: undefined,
            isLoading: true,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("databricks");
        fillDatabricksTarget();

        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("keeps Add disabled when the project identities query fails", () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: undefined,
            isLoading: false,
            isError: true
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("databricks");
        fillDatabricksTarget();

        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("ignores a second Add click while the first create is in flight", async () => {
        let releaseCreate: () => void = () => {};
        vi.mocked(createProjectsConnectorsByProjectId).mockReturnValueOnce(
            new Promise((resolve) => {
                releaseCreate = () =>
                    resolve({ data: { id: "connector-sales" } } as Awaited<
                        ReturnType<typeof createProjectsConnectorsByProjectId>
                    >);
            })
        );
        render(<IdentityConnectorForm />);

        selectType("databricks");
        await selectIdentity("Sales Databricks");
        fillDatabricksTarget();
        const add = screen.getByRole("button", { name: "Add" });
        // One act block for both clicks: separately, the first would commit
        // saving and the disabled button alone would stop the second, so the
        // ref guard would never be exercised.
        act(() => {
            fireEvent.click(add);
            fireEvent.click(add);
        });

        releaseCreate();
        await waitFor(() => expect(navigate).toHaveBeenCalled());
        expect(createProjectsConnectorsByProjectId).toHaveBeenCalledTimes(1);
    });

    it("offers the ambient S3 identity as an ordinary option", async () => {
        vi.mocked(useGetProjectsIdentitiesByProjectId).mockReturnValue({
            data: { data: [ambientS3Identity, ...identities] },
            isLoading: false,
            isError: false
        } as ReturnType<typeof useGetProjectsIdentitiesByProjectId>);
        render(<IdentityConnectorForm />);

        selectType("s3");
        // A bound ambient S3 identity remains selectable.
        await selectIdentity("Deployment ambient AWS");
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(createProjectsConnectorsByProjectId).toHaveBeenCalledWith(
                "project-sales",
                expect.objectContaining({
                    identity_id: "identity-ambient-s3"
                })
            )
        );
    });
});
