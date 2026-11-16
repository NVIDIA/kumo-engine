// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle
} from "@/components/ui/dialog";
import * as React from "react";
import { useRef } from "react";

export interface CustomAlertProps {
    title: string;
    description: string;
    show: boolean;
    handleClose: () => void;
}

export const CustomAlert: React.FC<CustomAlertProps> = ({
    title = "Error",
    description = "Something went wrong!!",
    show,
    handleClose
}) => {
    // Preserve last valid values so the close animation doesn't flash defaults
    const lastTitle = useRef(title);
    const lastDescription = useRef(description);
    if (show) {
        lastTitle.current = title;
        lastDescription.current = description;
    }

    return (
        <Dialog data-testid={"alert-dialog"} open={show}>
            <DialogContent
                data-testid={"alert-dialog-content"}
                className="top-1/2">
                <div className="px-7.5 flex flex-col justify-between gap-4 py-6">
                    <DialogTitle className="text-title-primary leading-7">
                        {lastTitle.current}
                    </DialogTitle>
                    <DialogDescription
                        style={{ maxHeight: 400 }}
                        className="text-title-primary m-0 w-[450px] flex-1 overflow-y-auto break-words p-0 text-base font-medium leading-5 outline-none [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[var(--scrollbar-dark)] [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar]:h-2.5 [&::-webkit-scrollbar]:w-2.5">
                        {lastDescription.current}
                    </DialogDescription>
                    <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                        <Button
                            id="close-btn"
                            variant="outline"
                            onClick={handleClose}>
                            Cancel
                        </Button>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
};
