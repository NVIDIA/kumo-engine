// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen } from "@testing-library/react";

import DebugInfoKeyPairs, {
    DebugInfoKeyPairsProps
} from "../DebugInfoKeyPairs";

vi.mock("services/axios_client", () => ({
    getBaseURL: vi.fn(() => "https://api.example.com")
}));

const defaultProps: DebugInfoKeyPairsProps = {
    data: {
        release_version: "1.0.0",
        temporal_url: "https://temporal.example.com",
        job_queue_status: "active",
        db_path: "/path/to/db",
        feature_flags: {
            ENABLE_COLUMN_ANALYSIS: false,
            MAX_TABLES_IN_A_GRAPH: 0,
            IS_DATABRICKS_DEPLOY: false
        }
    },
    debugInfo: {
        release_version: "1.0.0",
        temporal_url: "https://temporal.example.com",
        job_queue_status: "active",
        db_path: "/path/to/db"
    },
    loading: false,
    hideINProd: false,
    userId: "user-456",
    accessToken: "token-789"
};

describe("DebugInfoKeyPairs", () => {
    const renderDebugInfoKeyPairs = async (props: DebugInfoKeyPairsProps) => {
        await act(async () => {
            render(<DebugInfoKeyPairs {...props} />);
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render all KeyPair labels", async () => {
        await renderDebugInfoKeyPairs(defaultProps);

        expect(screen.getByText(/Kumo Version :/i)).toBeInTheDocument();
        expect(screen.getByText(/User ID :/i)).toBeInTheDocument();
        expect(screen.getByText(/Access Token :/i)).toBeInTheDocument();
        expect(screen.getByText(/REST API URL :/i)).toBeInTheDocument();
        expect(screen.getByText(/Temporal URL :/i)).toBeInTheDocument();
        expect(screen.getByText(/Job Queue Info :/i)).toBeInTheDocument();
        expect(screen.getByText(/DB Path :/i)).toBeInTheDocument();
    });

    it("Should display debugInfo values when provided", async () => {
        await renderDebugInfoKeyPairs(defaultProps);

        expect(screen.getByText("1.0.0")).toBeInTheDocument();
        expect(
            screen.getByText("https://temporal.example.com")
        ).toBeInTheDocument();
        expect(screen.getByText("active")).toBeInTheDocument();
    });

    it("Should display userId and accessToken", async () => {
        await renderDebugInfoKeyPairs(defaultProps);

        expect(screen.getByText("user-456")).toBeInTheDocument();
        expect(screen.getByText("token-789")).toBeInTheDocument();
    });

    it("Should display REST API URL from getBaseURL", async () => {
        await renderDebugInfoKeyPairs(defaultProps);

        expect(screen.getByText("https://api.example.com")).toBeInTheDocument();
    });

    it("Should display 'Loading...' for debugInfo fields when loading is true", async () => {
        await renderDebugInfoKeyPairs({
            ...defaultProps,
            debugInfo: null,
            loading: true
        });

        const loadingTexts = screen.getAllByText(/Loading/i);
        expect(loadingTexts.length).toBeGreaterThan(0);
    });

    it("Should display 'N/A' for debugInfo fields when loading is false and debugInfo is null", async () => {
        await renderDebugInfoKeyPairs({
            ...defaultProps,
            debugInfo: null,
            loading: false
        });

        const naTexts = screen.getAllByText("N/A");
        expect(naTexts.length).toBeGreaterThan(0);
    });

    it("Should hide DB Path when hideINProd is true", async () => {
        await renderDebugInfoKeyPairs({
            ...defaultProps,
            hideINProd: true
        });

        expect(screen.queryByText("/path/to/db")).not.toBeInTheDocument();
    });

    it("Should show DB Path when hideINProd is false", async () => {
        await renderDebugInfoKeyPairs({
            ...defaultProps,
            hideINProd: false
        });

        expect(screen.getByText("/path/to/db")).toBeInTheDocument();
    });

    it("Should handle empty baseURL from getBaseURL", async () => {
        const { getBaseURL } = await import("services/axios_client");
        (getBaseURL as ReturnType<typeof vi.fn>).mockReturnValue("");

        await renderDebugInfoKeyPairs(defaultProps);

        expect(screen.getByText(/REST API URL/i)).toBeInTheDocument();
    });
});
