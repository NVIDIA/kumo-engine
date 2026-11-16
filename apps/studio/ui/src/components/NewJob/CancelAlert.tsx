// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    Button,
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle
} from "@kumo-ai/kumo-ui-catalog";

/**
 * Confirmation dialog shown before discarding a job flow, whether from the
 * explicit Cancel button or from a blocked navigation away from a dirty
 * wizard (see pages/NewJob/Training/Training.tsx).
 *
 * @param title text
 * @param description text
 * @param show boolean
 * @param handleClose callback for close
 * @returns ```Custom Alert dialog```
 */
export interface CancelAlertProps {
    title: string;
    description: string;
    show: boolean;
    handleClose: () => void;
    handleConfirm: () => void;
}

const CancelAlert: React.FC<CancelAlertProps> = ({
    title = "Error",
    description = "Something went wrong!!",
    show,
    handleClose,
    handleConfirm = () => {}
}) => {
    return (
        <Dialog
            data-testid={"alert-dialog"}
            open={show}
            onOpenChange={(open) => {
                if (!open) handleClose();
            }}>
            <DialogContent
                data-testid={"alert-dialog-content"}
                className="top-1/2">
                <div className="flex flex-col gap-4 px-7.5 py-6">
                    <DialogTitle className="text-title-primary leading-7">
                        {title}
                    </DialogTitle>
                    <DialogDescription className="px-0 py-0 text-title-primary text-base font-medium leading-5">
                        {description}
                    </DialogDescription>
                    <div className="flex flex-col-reverse gap-2 pt-2 sm:flex-row sm:justify-end">
                        <Button
                            id="close"
                            variant="outline"
                            onClick={handleClose}>
                            Cancel
                        </Button>
                        <Button
                            id="discard"
                            variant="destructive"
                            onClick={handleConfirm}>
                            Discard
                        </Button>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default CancelAlert;
