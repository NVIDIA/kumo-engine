// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Row } from "@kumo-ai/kumo-ui-catalog";
import { render, screen } from "@testing-library/react";
import type { IdentityResponse } from "gen/v0/types";

import { identitySettingsColumns } from "../IdentitySettings";

const makeIdentity = (overrides: Partial<IdentityResponse>): IdentityResponse =>
    ({
        id: "identity-001",
        slug: "test-identity",
        display_name: "Test Identity",
        description: "",
        config: { type: "s3", authentication: { mode: "assume_role" } },
        credential_status: { configured: true },
        timestamps: {
            created_at: "2025-01-01T00:00:00Z",
            updated_at: "2025-01-01T00:00:00Z"
        },
        ...overrides
    }) as IdentityResponse;

const makeRow = (identity: IdentityResponse): Row<IdentityResponse> =>
    ({ original: identity }) as Row<IdentityResponse>;

const columns = identitySettingsColumns(vi.fn(), vi.fn());
const authColumn = columns.find((c) => c.id === "authentication")!;

const renderAuth = (identity: IdentityResponse) => {
    const cell = authColumn.cell as (ctx: {
        row: Row<IdentityResponse>;
    }) => React.ReactNode;
    render(<>{cell({ row: makeRow(identity) })}</>);
};

describe("identityAuthenticationLabel", () => {
    it("ambient S3 identity", () => {
        renderAuth(
            makeIdentity({
                config: { type: "s3", authentication: { mode: "ambient" } }
            })
        );
        expect(
            screen.getByText("Deployment-managed ambient identity")
        ).toBeInTheDocument();
    });

    it("Google workload identity (service_account_email present)", () => {
        renderAuth(
            makeIdentity({
                config: {
                    type: "bigquery",
                    authentication: {
                        service_account_email:
                            "sa@project.iam.gserviceaccount.com"
                    }
                }
            })
        );
        expect(screen.getByText("Workload identity")).toBeInTheDocument();
    });

    it("Google service-account key configured", () => {
        renderAuth(
            makeIdentity({
                config: {
                    type: "gcs",
                    authentication: {
                        mode: "service_account_key",
                        project_id: "my-project",
                        client_email: "sa@project.iam.gserviceaccount.com",
                        client_id: "123",
                        auth_uri: "https://accounts.google.com/o/oauth2/auth",
                        token_uri: "https://oauth2.googleapis.com/token",
                        universe_domain: "googleapis.com"
                    }
                },
                credential_status: { configured: true }
            })
        );
        expect(
            screen.getByText("Service-account key configured")
        ).toBeInTheDocument();
    });

    it("Google service-account key missing", () => {
        renderAuth(
            makeIdentity({
                config: {
                    type: "gcs",
                    authentication: {
                        mode: "service_account_key",
                        project_id: "my-project",
                        client_email: "sa@project.iam.gserviceaccount.com",
                        client_id: "123",
                        auth_uri: "https://accounts.google.com/o/oauth2/auth",
                        token_uri: "https://oauth2.googleapis.com/token",
                        universe_domain: "googleapis.com"
                    }
                },
                credential_status: { configured: false }
            })
        );
        expect(
            screen.getByText("Service-account key missing")
        ).toBeInTheDocument();
    });

    it("non-Google credential configured", () => {
        renderAuth(
            makeIdentity({
                config: {
                    type: "snowflake",
                    account: "xy12345",
                    username: "kumo",
                    authentication: "password"
                },
                credential_status: { configured: true }
            })
        );
        expect(
            screen.getByText("Stored material configured")
        ).toBeInTheDocument();
    });

    it("non-Google credential missing", () => {
        renderAuth(
            makeIdentity({
                config: {
                    type: "databricks",
                    host: "adb-123.azuredatabricks.net",
                    client_id: "abc"
                },
                credential_status: { configured: false }
            })
        );
        expect(screen.getByText("Workload identity")).toBeInTheDocument();
    });
});
