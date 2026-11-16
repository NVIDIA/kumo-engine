// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
    ImmutableResourceSlug,
    ResourceSlugField
} from "components/ResourceSlug";

describe("ResourceSlugField", () => {
    it("previews the canonical derived ID while leaving the optional input empty", () => {
        render(
            <ResourceSlugField
                id="project-slug"
                displayName="  Fraud__Detection!  "
                value=""
                onChange={vi.fn()}
            />
        );

        expect(screen.getByLabelText(/ID/)).toHaveValue("");
        expect(screen.getByText("Preview: fraud-detection")).toBeVisible();
        expect(
            screen.getByText(
                "This ID cannot be changed later. The name can be changed."
            )
        ).toBeVisible();
    });

    it("hides the example placeholder and preview line when hidePreviewHint is set", () => {
        render(
            <ResourceSlugField
                id="identity-slug"
                displayName="Sales Databricks"
                value=""
                onChange={vi.fn()}
                hidePreviewHint
            />
        );

        // Derived ID surfaces only as the placeholder — no "fraud-detection"
        // example and no "Preview: <slug>" line.
        expect(screen.getByLabelText(/ID/)).toHaveAttribute(
            "placeholder",
            "sales-databricks"
        );
        expect(screen.queryByText(/^Preview:/)).not.toBeInTheDocument();
        expect(
            screen.getByText(
                /Enter a name to preview the automatically generated ID\./
            )
        ).toBeVisible();
    });

    it("reports invalid and reserved explicit IDs without rewriting them", () => {
        const { rerender } = render(
            <ResourceSlugField
                id="group-slug"
                displayName="Risk Team"
                value="Risk_Team"
                onChange={vi.fn()}
            />
        );

        expect(screen.getByLabelText(/ID/)).toHaveValue("Risk_Team");
        expect(screen.getByRole("alert")).toHaveTextContent(
            "contain only lowercase letters"
        );

        rerender(
            <ResourceSlugField
                id="group-slug"
                displayName="Risk Team"
                value="settings"
                onChange={vi.fn()}
            />
        );
        expect(screen.getByRole("alert")).toHaveTextContent(
            "ID “settings” is reserved"
        );
    });
});

describe("ImmutableResourceSlug", () => {
    it("renders an immutable ID and copies it accessibly", async () => {
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: { writeText }
        });

        render(<ImmutableResourceSlug id="identity-slug" slug="prod-aws" />);

        expect(screen.getByLabelText("ID")).toHaveValue("prod-aws");
        expect(screen.getByLabelText("ID")).toHaveAttribute("readonly");
        await userEvent.click(screen.getByRole("button", { name: "Copy ID" }));
        expect(writeText).toHaveBeenCalledWith("prod-aws");
        expect(screen.getByRole("status")).toHaveTextContent("ID copied.");
    });

    it("reports a clipboard failure without throwing", async () => {
        Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: { writeText: vi.fn().mockRejectedValue(new Error("denied")) }
        });
        render(<ImmutableResourceSlug id="identity-slug" slug="prod-aws" />);

        await userEvent.click(screen.getByRole("button", { name: "Copy ID" }));
        expect(screen.getByRole("status")).toHaveTextContent(
            "ID could not be copied."
        );
    });
});
