// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    Button,
    Check,
    Close,
    cn,
    Description,
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle,
    ProgressActivity,
    ScrollArea
} from "@kumo-ai/kumo-ui-catalog";

import PQCodeEditor from "components/CodeEditor/PQCodeEditor";

import { useSDKCode } from "hooks/misc/sdk";

import { CustomError } from "types/common";
import { SDKObjectType } from "types/constants";

import { parseErrorMessage } from "utils/errorHandler";

export interface SDKCodeModalProps {
    id?: string;
    open: boolean;
    toggleModel: () => void;
    objectTypeToGenScript: SDKObjectType;
}

export const SDKCodeModal = ({
    id,
    open,
    toggleModel,
    objectTypeToGenScript
}: SDKCodeModalProps) => {
    const [isCopied, setIsCopied] = useState(false);
    const [isDownloaded, setIsDownloaded] = useState(false);
    const { data, isLoading, error, isError } = useSDKCode(
        id ?? "",
        objectTypeToGenScript
    );

    const handleDownload = () => {
        if (!data?.data) {
            return;
        }
        const fileName = `sdk-code-${id}.py`; // Name of the downloaded file
        // Create a Blob from the Python code
        const blob = new Blob([data?.data as string], {
            type: "text/x-python"
        });
        const url = URL.createObjectURL(blob);

        // Create a temporary <a> element to trigger download
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();

        // Cleanup
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        setIsDownloaded(true);
        setTimeout(() => {
            setIsDownloaded(false);
        }, 2000);
    };

    return (
        <Dialog
            data-testid={"sdk-dialog"}
            open={open}
            onOpenChange={() => toggleModel()}>
            <DialogContent
                data-testid={"sdk-content"}
                className="!max-w-186 !h-144.25 top-1/2 py-4 px-6 overflow-hidden">
                <div className="!mb-2 flex gap-1 text-center sm:text-left flex-row w-full justify-between items-center">
                    <DialogTitle className="text-title-primary text-xl font-base">
                        {"View SDK Code"}
                    </DialogTitle>
                    <Button
                        data-testid="close-icon"
                        variant="no-outline"
                        id={"Close"}
                        aria-label="Close"
                        className="p-2"
                        onClick={() => toggleModel()}>
                        <Close className="!w-5 !h-5" />
                    </Button>
                </div>
                <div className="flex flex-col gap-2">
                    <DialogDescription className="px-0 pt-0 m-0 text-title-secondary !font-medium">
                        Here is the generated code to reproduce this{" "}
                        {objectTypeToGenScript} in the SDK. Before running the
                        script, don't forget to add your API URL and API key.
                        Warehouse credentials are managed by the assigned
                        identity.
                    </DialogDescription>
                    <div className="flex w-172 h-91 bg-surface-secondary rounded-md">
                        {isLoading ? (
                            <>
                                <ProgressActivity className="size-4.5 animate-spin" />
                                <Description
                                    variant="secondary"
                                    size="sm"
                                    className="!mb-0 ps-2 pt-0">
                                    Loading
                                </Description>
                            </>
                        ) : (
                            <ScrollArea
                                type="always"
                                className="!w-full !h-full  rounded-md">
                                {isError ? (
                                    error ? (
                                        parseErrorMessage(error as CustomError)
                                    ) : (
                                        "Failed to load SDK code"
                                    )
                                ) : (
                                    <PQCodeEditor
                                        value={(data?.data ?? "").trim()}
                                        language={"PYTHON"}
                                        editable={false}
                                        isInvalid={false}
                                        errorMsg={""}
                                        onChange={() => {}}
                                        graphTables={[]}
                                        lineNumbers={true}
                                        highlightActiveLineGutter={false}
                                        highlightActiveLine={false}
                                        hideBorder
                                        height="auto"
                                        // eslint-disable-next-line no-restricted-syntax -- CodeMirror prop accepts hex, not a Tailwind class
                                        backgroundColor="#F4F6F8"
                                    />
                                )}
                            </ScrollArea>
                        )}
                    </div>
                </div>
                <div className="mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                    <Button
                        id={"Copy"}
                        variant="outline"
                        disabled={isLoading || isError}
                        className={cn(isCopied && "gap-1")}
                        onClick={() => {
                            navigator.clipboard.writeText(
                                (data?.data as string) || ""
                            );
                            setIsCopied(true);
                            setTimeout(() => {
                                setIsCopied(false);
                            }, 2000);
                        }}>
                        {isCopied ? (
                            <span className="flex items-center gap-1">
                                <Check className="!fill-emerald-500" />
                                {"Copied"}
                            </span>
                        ) : (
                            "Copy"
                        )}
                    </Button>
                    <Button
                        id={"Download"}
                        variant="outline"
                        disabled={isLoading || isError}
                        className={cn(isDownloaded && "gap-1")}
                        onClick={handleDownload}>
                        {isDownloaded ? (
                            <span className="flex items-center gap-1">
                                <Check className="!fill-emerald-500" />
                                {"Downloaded"}
                            </span>
                        ) : (
                            "Download"
                        )}
                    </Button>
                </div>
            </DialogContent>
        </Dialog>
    );
};
