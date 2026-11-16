// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErrWarnDetails } from "gen/v0/types";
import {
    mockLocationStateValue,
    mockNavigateCall,
    mockUseNavigate
} from "tests/mockRouter";
import {
    expectNoAlert,
    getAlert,
    verifyTextDoesNotExist,
    verifyTextExists
} from "tests/testHelpers";

import { ctaLevels } from "types/constants";
import { CTALevel } from "types/warnings";

import WarningErrorCTA, { WarningErrorCTAProps } from "../WarningErrorCTA";

const item = (overrides: Partial<ErrWarnDetails> = {}): ErrWarnDetails => ({
    id: "1",
    is_error: true,
    title: "Test Title",
    description: "Test description",
    is_dismissible: false,
    cta: null,
    ...overrides
});

const baseProps: WarningErrorCTAProps = {
    id: "cta-test",
    data: [item()],
    ctaLevel: { value: ctaLevels.Critical.value } as CTALevel,
    handleClose: vi.fn()
};

describe("WarningErrorCTA", () => {
    const renderCta = async (props: Partial<WarningErrorCTAProps> = {}) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <WarningErrorCTA {...baseProps} {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
        mockLocationStateValue(null, "/");
    });

    it.each([
        { description: "data is undefined", data: undefined },
        { description: "data is empty array", data: [] }
    ])("Should render nothing when $description", async ({ data }) => {
        await renderCta({ data: data as ErrWarnDetails[] });
        expectNoAlert();
    });

    it("Should render alert with item title and description", async () => {
        await renderCta({
            data: [item({ title: "Boom", description: "Bad happened" })],
            title: "Custom Title"
        });

        getAlert();
        await verifyTextExists("Custom Title");
        await verifyTextExists(/Boom/);
        await verifyTextExists("Bad happened");
    });

    it.each([
        {
            description:
                "Should map Info CTA level to catalog `info` (default title `Info`)",
            ctaLevel: ctaLevels.Info.value,
            data: [item()],
            expectedTitle: "Info"
        },
        {
            description:
                "Should map Warning CTA level to catalog `warning` (default title `Warnings` for multiple items)",
            ctaLevel: ctaLevels.Warning.value,
            data: [item({ id: "a" }), item({ id: "b" })],
            expectedTitle: "Warnings"
        },
        {
            description:
                "Should fall back to catalog `error` for any non-Info/Warning level (default title `Error`)",
            ctaLevel: ctaLevels.Critical.value,
            data: [item()],
            expectedTitle: "Error"
        }
    ])("$description", async ({ ctaLevel, data, expectedTitle }) => {
        await renderCta({
            data,
            ctaLevel: { value: ctaLevel } as CTALevel,
            title: ""
        });

        await verifyTextExists(expectedTitle);
    });

    it("Should toggle Show More / Show Less when data exceeds previewCount", async () => {
        const many = Array.from({ length: 5 }, (_, i) =>
            item({
                id: `${i}`,
                title: `Item ${i}`,
                description: `Desc ${i}`
            })
        );

        await renderCta({
            data: many,
            ctaLevel: { value: ctaLevels.Warning.value } as CTALevel
        });

        await verifyTextExists("Show More");

        await userEvent.click(screen.getByText(/Show More/i));

        await verifyTextExists("Show Less");
        await verifyTextExists(/Item 4/);
    });

    it("Should navigate via useCustomNavigation when a CTA is clicked", async () => {
        await renderCta({
            data: [
                item({
                    cta: { name: "Go to page", url: "/next-page" }
                })
            ]
        });

        await userEvent.click(
            screen.getByRole("button", { name: /Go to page/i })
        );

        expect(mockUseNavigate).toHaveBeenCalledWith("/next-page");
    });

    it("Should hide the CTA when its url matches the current pathname", async () => {
        mockLocationStateValue(null, "/current-page");

        await renderCta({
            data: [
                item({
                    cta: { name: "Stay here", url: "/current-page" }
                })
            ]
        });

        getAlert();
        verifyTextDoesNotExist(/Stay here/);
    });

    it("Should run descriptions through adjustError (empty description becomes fallback message)", async () => {
        await renderCta({
            data: [item({ title: "Boom", description: "" })]
        });

        await verifyTextExists(/We're sorry, an unexpected error occurred/);
    });
});
