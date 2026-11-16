// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Spacer from "react-spacer";

import {
    Button,
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle
} from "@kumo-ai/kumo-ui-catalog";

export interface DeleteApiKeyDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onConfirm: () => void;
}

export const DeleteApiKeyDialog = ({
    open,
    onOpenChange,
    onConfirm
}: DeleteApiKeyDialogProps) => (
    <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent data-testid={"alert-dialog-content"} className="top-1/2">
            <div className="flex flex-col justify-between px-7.5 py-6">
                <DialogTitle className="text-title-primary leading-7">
                    Delete API Key
                </DialogTitle>
                <Spacer height={16} />
                <DialogDescription className="flex-1 p-0 m-0 text-title-primary text-base font-medium leading-5 outline-none max-h-100 overflow-auto [&::-webkit-scrollbar]:h-2.5 [&::-webkit-scrollbar]:w-2.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[var(--scrollbar-dark)] [&::-webkit-scrollbar-track]:bg-transparent">
                    Are you sure you want to delete this API key? This action
                    cannot be undone.
                </DialogDescription>
                <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end mt-6">
                    <Button
                        id="cancel-delete"
                        variant="outline"
                        onClick={() => onOpenChange(false)}>
                        Cancel
                    </Button>

                    <Button
                        id="confirm-delete"
                        variant="destructive"
                        onClick={onConfirm}>
                        Delete
                    </Button>
                </div>
            </div>
        </DialogContent>
    </Dialog>
);
