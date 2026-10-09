// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Row } from "@kumo-ai/kumo-ui-catalog";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ColumnArgs } from "types/tables";

import { SampleCell } from "../SampleCell";

import "@testing-library/jest-dom";

const baseRow = {
    id: "sample-1",
    original: {
        stype: "text",
        sample_rows: ["example"],
        isSampleAvailable: true
    }
};

describe("SampleCell", () => {
    it("Should render image when stype is image and sample_rows contains image URLs", () => {
        const row = {
            ...baseRow,
            original: {
                stype: "image",
                sample_rows: ["https://example.com/image1.png"]
            }
        };

        render(<SampleCell row={row as Row<ColumnArgs>} />);
        const image = screen.getByRole("img");
        expect(image).toBeInTheDocument();
        expect(image).toHaveAttribute("src", "https://example.com/image1.png");
    });

    it("Should render plain text if stype is not image and length is < 100 chars", () => {
        const row = {
            ...baseRow,
            original: {
                stype: "text",
                sample_rows: ["short text"],
                isSampleAvailable: true
            }
        };

        render(<SampleCell row={row as Row<ColumnArgs>} />);
        expect(screen.getByText("short text")).toBeInTheDocument();
    });

    it("Should render tooltip for long text values (> 100 chars)", async () => {
        const longText = "a".repeat(101);
        const row = {
            ...baseRow,
            original: {
                stype: "text",
                sample_rows: [longText],
                isSampleAvailable: true
            }
        };

        render(<SampleCell row={row as Row<ColumnArgs>} />);

        const trigger = screen.getByText(longText);
        expect(trigger).toBeInTheDocument();

        await userEvent.hover(trigger);

        const tooltip = await screen.findByRole("tooltip");
        expect(tooltip).toBeInTheDocument();
        expect(tooltip).toHaveTextContent(longText);
    });

    it("Should render Skeleton when no sample is available", () => {
        const row = {
            ...baseRow,
            original: {
                stype: "text",
                sample_rows: [],
                isSampleAvailable: false
            }
        };

        render(<SampleCell row={row as unknown as Row<ColumnArgs>} />);
        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should handle undefined or null sample_rows gracefully", () => {
        const rowUndefined = {
            ...baseRow,
            original: {
                stype: "text",
                sample_rows: undefined,
                isSampleAvailable: true
            }
        };

        const { unmount } = render(
            <SampleCell row={rowUndefined as Row<ColumnArgs>} />
        );
        expect(screen.queryByText(/.+/)).not.toBeInTheDocument();

        unmount();

        const rowNull = {
            ...baseRow,
            original: {
                stype: "text",
                sample_rows: null,
                isSampleAvailable: true
            }
        };

        render(<SampleCell row={rowNull as Row<ColumnArgs>} />);
        expect(screen.queryByText(/.+/)).not.toBeInTheDocument();
    });

    it("Should render image when value is a string (not array)", () => {
        const row = {
            ...baseRow,
            original: {
                stype: "image",
                sample_rows: "https://example.com/image.png"
            }
        };

        render(<SampleCell row={row as Row<ColumnArgs>} />);
        const image = screen.getByRole("img");
        expect(image).toBeInTheDocument();
        expect(image).toHaveAttribute("src", "https://example.com/image.png");
    });

    it("Should handle null textString with fallback to empty string", () => {
        const row = {
            ...baseRow,
            original: {
                stype: "text",
                sample_rows: null,
                isSampleAvailable: true
            }
        };

        render(<SampleCell row={row as Row<ColumnArgs>} />);
        const sampleCell = screen.queryByTestId("sample-cell");
        expect(sampleCell).toBeInTheDocument();
        expect(sampleCell).toHaveTextContent("");
    });
});
