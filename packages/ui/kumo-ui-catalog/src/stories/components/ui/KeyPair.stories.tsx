// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle
} from "@/components/ui/dialog";
import { KeyPair } from "@/components/ui/key-pair";
import { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof KeyPair> = {
    title: "UI/KeyPair",
    component: KeyPair,
    tags: ["autodocs"],
    args: {
        keyLabel: "API Key",
        value: "6b920e47-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        full: false,
        loadingStatus: false,
        className: "",
        keyClassName: "",
        valueClassName: "",
        paddingY: 2,
        showCopy: true
    }
};

export default meta;

type Story = StoryObj<typeof KeyPair>;

export const Default: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        await expect(canvas.getByText(/API Key/)).toBeVisible();

        const copyButton = canvas.getByRole("button", {
            name: /Copy to clipboard/i
        });
        await expect(copyButton).toBeVisible();
        await userEvent.click(copyButton);
    }
};

export const FullValue: Story = {
    args: {
        full: true
    }
};

export const Loading: Story = {
    args: {
        loadingStatus: true
    }
};

export const showCopy: Story = {
    args: {
        showCopy: true
    }
};

export const tooltip: Story = {
    args: {
        value: "6b920e47-xxxxxxxxxxxxxxxxxxxxxxxxxxxx555555555555555555555tttttttttttttttttttyyyyyyyyyyyy"
    }
};

export const CustomStyles: Story = {
    args: {
        keyClassName: "text-red-600 font-bold",
        valueClassName: "text-green-700",
        paddingY: 4
    }
};

export const Dialogue: Story = {
    render: function DefaultFunc() {
        const [show, setShow] = useState(false);

        return (
            <>
                <Button id={"show"} onClick={() => setShow(true)}>
                    {"show"}
                </Button>
                <Dialog data-testid={"alert-dialog"} open={show}>
                    <DialogContent data-testid={"alert-dialog-content"}>
                        <DialogHeader className="mt-2">
                            <DialogTitle>{"Keypair test"}</DialogTitle>
                        </DialogHeader>
                        <DialogDescription
                            className="outline-none [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[var(--scrollbar-dark)] [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar]:h-2.5 [&::-webkit-scrollbar]:w-2.5"
                            style={{
                                maxHeight: 400,
                                overflow: "auto"
                            }}>
                            <KeyPair
                                keyLabel="Api Key"
                                value={
                                    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InN1YiI6Imdvb2dsZS1vYXV0aDJ8MTEwMDY1OTI0MTIxNDE2MTkwMDEzIiwiYXVkIjpbImt1bW8tcmVzdC1hcGkiLCJodHRwczovL2t1bW9haS51cy5hdXRoMC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzYwNjEyODYxLCJleHAiOjE3NjA2MjAwNjEsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgcmVhZDp0ZXN0IHdyaXRlOnRlc3Qgb2ZmbGluZV9hY2Nlc3MiLCJvcmdfaWQiOiJvcmdfcUNEVHV3MndaUnhJU0ZFSCIsImF6cCI6InFDVFJZVG5GME1VZHJzUUQwcnZUOGg2Ymt5dHI4OEFUIiwicGVybWlzc2lvbnMiOlsiYWRtaW46c3RhZ2luZzEiLCJhZG1pbjpzdGFnaW5nMiIsImFkbWluOnRlc3QiLCJyZWFkOmRlbW8iLCJyZWFkOmRvb3JkYXNoIiwicmVhZDpzdGFnaW5nMSIsInJlYWQ6c3RhZ2luZzIiLCJyZWFkOnRlc3QiLCJ3cml0ZTpkZW1vIiwid3JpdGU6ZG9vcmRhc2giLCJ3cml0ZTpzdGFnaW5nMSIsIndyaXRlOnN0YWdpbmcyIiwid3JpdGU6dGVzdCJdfQ.d-94Fsv3TLgDd0FUvPj9e3FSB4MDHjyJcgrmsNuMa86lH5izBsVMM_GRhuSQq-yR9lLgy9RNEZwOTdi_KJeMcoC5VLjGK_qBER30FQkn5hZbFP7ZeGI-7KQZfStb1VWt7K71lgzzOcMb46EOLIsnA-vspQ2nxb0pddriSZPD0iHmJwqNZnM2nNTOlIP1cG8dnOSZJmhMEZIAeTUki4Q60ArHPuoxKnriRq28c_0aS_IGT-PgWGrqWpqNK7Wadi0h__mAJxAnqI4wadS_6lAxgkhY9nUQy_dY2J_6iKBqf8oDvoY8RGitNi47WgNvSchjZO9CrvyvSGhjOTI23_efrw"
                                }
                                showCopy={true}
                            />
                        </DialogDescription>
                        <DialogFooter>
                            <Button id={"Close"} onClick={() => setShow(false)}>
                                {"Close"}
                            </Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const body = within(document.body);

        const showButton = canvas.getByRole("button", { name: /show/i });
        await userEvent.click(showButton);

        await waitFor(() => {
            expect(body.getByText("Keypair test")).toBeVisible();
        });
        await expect(body.getByText(/Api Key/)).toBeVisible();

        const closeButton = body.getByRole("button", { name: /Close/i });
        await userEvent.click(closeButton);
        await waitFor(() => {
            expect(body.queryByText("Keypair test")).not.toBeInTheDocument();
        });
    }
};
