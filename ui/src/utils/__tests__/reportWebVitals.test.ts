// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from "@testing-library/dom";
import { describe, expect, it, vi } from "vitest";
import type { ReportHandler } from "web-vitals";

import reportWebVitals from "utils/reportWebVitals";

describe("reportWebVitals", () => {
    it("Should not throw when onPerfEntry is undefined", () => {
        expect(() => reportWebVitals()).not.toThrow();
    });

    it("Should not throw when onPerfEntry is not a function", () => {
        expect(() =>
            reportWebVitals("not a function" as unknown as ReportHandler)
        ).not.toThrow();
    });

    it("Should call the handler with metric data when onPerfEntry is a valid function", async () => {
        const mockHandler = vi.fn();

        reportWebVitals(mockHandler);

        await waitFor(() => {
            expect(mockHandler).toHaveBeenCalled();
        });

        expect(mockHandler.mock.calls.length).toBeGreaterThan(0);

        mockHandler.mock.calls.forEach((call) => {
            const metric = call[0];
            expect(metric).toHaveProperty("name");
            expect(metric).toHaveProperty("value");
            expect(metric).toHaveProperty("id");
        });
    });
});
