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
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import * as React from "react";
import {
    ForwardedRef,
    forwardRef,
    useCallback,
    useImperativeHandle,
    useState
} from "react";

export type ConfirmButtonVariant =
    | "outline"
    | "secondary"
    | "link"
    | "destructive";

export interface DialogRef<TObject extends object> {
    toggleDialog: (
        title?: string,
        description?: string | React.ReactNode,
        value?: TObject | null,
        enableSkipButton?: boolean,
        confirmButtonText?: string,
        confirmButtonVariant?: ConfirmButtonVariant,
        cancelButtonText?: string
    ) => void;
}

export interface ConfirmationDialogProps<TObject extends object> {
    cancelText?: string;
    confirmText?: string;
    confirmVariant?: ConfirmButtonVariant;
    handleClose: () => void;
    handleConfirm: (value?: TObject) => void;
    handleSkip?: (value?: TObject) => void;
}

export const ConfirmationDialog = forwardRef(
    <TObject extends object>(
        {
            cancelText = "Cancel",
            confirmText = "OK",
            confirmVariant,
            handleClose,
            handleConfirm,
            handleSkip
        }: ConfirmationDialogProps<TObject>,
        ref: ForwardedRef<DialogRef<TObject>>
    ) => {
        const [showDialog, setShowDialog] = useState(false);
        const [title, setTitle] = useState("");
        const [description, setDescription] = useState<
            string | React.ReactNode
        >("");
        const [value, setValue] = useState<TObject | null>(null);
        const [enableSkipButton, setSkipButton] = useState<boolean>(false);
        const [dynamicConfirmText, setDynamicConfirmText] = useState<
            string | undefined
        >(undefined);
        const [dynamicConfirmVariant, setDynamicConfirmVariant] = useState<
            ConfirmButtonVariant | undefined
        >(undefined);
        const [dynamicCancelText, setDynamicCancelText] = useState<
            string | undefined
        >(undefined);

        useImperativeHandle(
            ref,
            () => ({
                toggleDialog(
                    title = "",
                    description = "",
                    value = null,
                    enableSkipButton = false,
                    confirmButtonText = undefined,
                    confirmButtonVariant = undefined,
                    cancelButtonText = undefined
                ) {
                    if (title) {
                        setTitle(title);
                        setDescription(description);
                        setValue(value);
                        setSkipButton(enableSkipButton);
                        setDynamicConfirmText(confirmButtonText);
                        setDynamicConfirmVariant(confirmButtonVariant);
                        setDynamicCancelText(cancelButtonText);
                        setShowDialog(true);
                    } else {
                        setShowDialog(false);
                    }
                }
            }),
            []
        );

        const onConfirm = useCallback(() => {
            handleConfirm(value ?? undefined);
        }, [handleConfirm, value]);

        const onSkip = useCallback(() => {
            handleSkip?.(value ?? undefined);
        }, [handleSkip, value]);

        return (
            <Dialog
                data-testid={"dialog"}
                open={showDialog}
                onOpenChange={setShowDialog}>
                <DialogContent
                    data-testid={"dialog-content"}
                    className="top-1/2">
                    <div className="px-7.5 flex flex-col justify-between py-6">
                        <DialogTitle className="text-title-primary leading-7">
                            {title || (
                                <VisuallyHidden>
                                    Confirmation Dialog
                                </VisuallyHidden>
                            )}
                        </DialogTitle>
                        <div style={{ height: 16 }} />
                        <DialogDescription className="text-title-primary m-0 w-[450px] flex-1 overflow-y-auto break-words p-0 text-base font-medium leading-5">
                            {description}
                        </DialogDescription>

                        <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                            {enableSkipButton ? (
                                <>
                                    <Button
                                        id="skip-btn"
                                        variant="outline"
                                        onClick={onSkip}>
                                        {dynamicCancelText ?? cancelText}
                                    </Button>
                                    <Button
                                        id="confirm-btn"
                                        variant={
                                            dynamicConfirmVariant ??
                                            confirmVariant ??
                                            "outline"
                                        }
                                        onClick={onConfirm}>
                                        {dynamicConfirmText ?? confirmText}
                                    </Button>
                                </>
                            ) : (
                                <>
                                    <Button
                                        id="cancel-btn"
                                        variant="outline"
                                        onClick={handleClose}>
                                        {dynamicCancelText ?? cancelText}
                                    </Button>
                                    <Button
                                        id="confirm-btn"
                                        variant={
                                            dynamicConfirmVariant ??
                                            confirmVariant
                                        }
                                        onClick={onConfirm}>
                                        {dynamicConfirmText ?? confirmText}
                                    </Button>
                                </>
                            )}
                        </div>
                    </div>
                </DialogContent>
            </Dialog>
        );
    }
);
