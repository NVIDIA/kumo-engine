// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
    expectNoDialog,
    getDialog,
    verifyButtonExists
} from "tests/testHelpers";

import {
    DeleteApiKeyDialog,
    DeleteApiKeyDialogProps
} from "../DeleteKeyDialog";

const onOpenChange = vi.fn();
const onConfirm = vi.fn();

const defaultProps: DeleteApiKeyDialogProps = {
    open: true,
    onOpenChange,
    onConfirm
};

describe("DeleteApiKeyDialog", () => {
    const renderDeleteApiKeyDialog = async (
        props: DeleteApiKeyDialogProps = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <DeleteApiKeyDialog {...props} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render the dialog with title, description, and action buttons", async () => {
        await renderDeleteApiKeyDialog();

        getDialog();
        expect(screen.getByText("Delete API Key")).toBeInTheDocument();
        expect(
            screen.getByText(/are you sure you want to delete this api key/i)
        ).toBeInTheDocument();
        verifyButtonExists(/cancel/i);
        verifyButtonExists(/^delete$/i);
    });

    it("Should not render the dialog when open is false", async () => {
        await renderDeleteApiKeyDialog({ ...defaultProps, open: false });

        expectNoDialog();
    });

    it("Should call onConfirm when Delete is clicked", async () => {
        await renderDeleteApiKeyDialog();

        await userEvent.click(
            screen.getByRole("button", { name: /^delete$/i })
        );

        expect(onConfirm).toHaveBeenCalledTimes(1);
    });

    it("Should call onOpenChange(false) and not call onConfirm when Cancel is clicked", async () => {
        await renderDeleteApiKeyDialog();

        await userEvent.click(screen.getByRole("button", { name: /cancel/i }));

        expect(onOpenChange).toHaveBeenCalledWith(false);
        expect(onConfirm).not.toHaveBeenCalled();
    });
});
