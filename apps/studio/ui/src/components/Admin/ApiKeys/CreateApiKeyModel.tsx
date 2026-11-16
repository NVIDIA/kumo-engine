// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import Spacer from "react-spacer";

import {
    Button,
    Check,
    CopyToClipBoard,
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle,
    Input
} from "@kumo-ai/kumo-ui-catalog";
import { APIKeyType, CreateAPIKeyResponse } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { CreateKeysFunc } from "hooks/apiKeys/apiKeys";

export interface CreateApiKeyModelProps {
    open: boolean;
    toggleCreate: () => void;
    createKeys: CreateKeysFunc;
    keyType: APIKeyType;
    isAdmin: boolean;
}

const CreateAPIKeyComponent = ({
    open,
    toggleCreate,
    createKeys,
    keyType,
    isAdmin
}: CreateApiKeyModelProps) => {
    const [newKeyName, setNewKeyName] = useState("");
    const [keyCreated, setKeyCreated] = useState(false);
    const [keyCopied, setKeyCopied] = useState(false);
    const [newKey, setNewKey] = useState<CreateAPIKeyResponse | null>(null);
    const [errorMsg, setErrorMsg] = useState("");
    const copyTimerRef = useRef<ReturnType<typeof setTimeout>>();

    useEffect(() => () => clearTimeout(copyTimerRef.current), []);

    const onCreateClick = async () => {
        const config = isAdmin
            ? { name: newKeyName, type: keyType }
            : { name: newKeyName };
        const { error } = await createKeys(config, (data) => {
            setKeyCreated(true);
            setNewKey(data);
        });
        if (error) setErrorMsg(error);
    };

    const resetInputs = () => {
        setNewKeyName("");
        setKeyCreated(false);
        setKeyCopied(false);
        setErrorMsg("");
    };

    // An admin creates keys on either the Admin or the User tab, so the title
    // names which kind is being created; a non-admin only ever creates their
    // own key, so the title stays unqualified.
    const createTitle = isAdmin
        ? keyType === APIKeyType.ADMIN
            ? "Create Admin API Key"
            : "Create User API Key"
        : "Create API Key";

    return (
        <Dialog
            open={open}
            onOpenChange={(open) => {
                toggleCreate();
                if (!open) resetInputs();
            }}>
            <DialogContent
                data-testid="create-api-key-dialog"
                className="top-1/2 max-w-134.75"
                onInteractOutside={(e) => e.preventDefault()}>
                <div className="flex flex-col justify-between px-7.5 py-6">
                    <div className="h-9 items-center flex">
                        <DialogTitle className="text-title-primary leading-7">
                            {!keyCreated ? createTitle : "Save Your API Key"}
                        </DialogTitle>
                    </div>
                    <Spacer height={16} />
                    <DialogDescription
                        className="flex-1 p-0 m-0 text-title-primary text-base font-medium leading-5 outline-none"
                        asChild>
                        <div>
                            {!keyCreated ? (
                                <div className="flex flex-col gap-2">
                                    <label
                                        htmlFor="api-key-name-input"
                                        className="text-sm font-medium text-title-secondary">
                                        API key name
                                    </label>
                                    <Input
                                        id="api-key-name-input"
                                        placeholder="Enter name"
                                        value={newKeyName}
                                        onChange={(e) => {
                                            setNewKeyName(
                                                e.target.value.trim()
                                            );
                                            setErrorMsg("");
                                        }}
                                        className="h-9 !text-base font-medium py-2 px-4"
                                    />
                                    {errorMsg && (
                                        <p className="text-sm text-red-500">
                                            {errorMsg}
                                        </p>
                                    )}
                                </div>
                            ) : (
                                <div className="flex flex-col gap-4">
                                    <div className="rounded-md border border-emerald-500 bg-emerald-50 p-6 [&_svg]:fill-emerald-500">
                                        <div className="flex items-start gap-4">
                                            <Check />
                                            <div className="text-base font-medium text-title-primary leading-5">
                                                <p>
                                                    API key "{newKey?.name}"
                                                    created successfully!
                                                </p>
                                                <p className="mt-4">
                                                    Please copy and save your
                                                    API key for future use. For
                                                    security reasons, we will
                                                    not show you this key again.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex flex-col gap-2">
                                        <label
                                            htmlFor="api-key-value-input"
                                            className="text-sm font-medium text-title-secondary">
                                            API key
                                        </label>
                                        <div className="flex gap-2">
                                            <Input
                                                id="api-key-value-input"
                                                readOnly
                                                value={newKey?.value?.replaceAll(
                                                    ".",
                                                    "x"
                                                )}
                                                className="h-9 flex-1 rounded-sm text-title-primary !text-base !font-medium !py-2 !px-4"
                                            />
                                            <Button
                                                variant="light-outline"
                                                className="px-2 border h-9 w-9 border-border-secondary [&_span_svg]:fill-emerald-500">
                                                <CopyToClipBoard
                                                    copiedText={
                                                        newKey?.value ?? ""
                                                    }
                                                    iconSize={20}
                                                    tooltipText="Copied"
                                                />
                                            </Button>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </DialogDescription>

                    {keyCreated && errorMsg && (
                        <p className="text-sm text-red-500 mt-2">{errorMsg}</p>
                    )}
                    <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end mt-6">
                        {!keyCreated ? (
                            <>
                                <Button
                                    id="cancel-create"
                                    variant="outline"
                                    onClick={() => {
                                        resetInputs();
                                        toggleCreate();
                                    }}>
                                    Cancel
                                </Button>
                                <Button
                                    id="confirm-create"
                                    variant="primary"
                                    disabled={!newKeyName || !!errorMsg}
                                    onClick={onCreateClick}>
                                    Create
                                </Button>
                            </>
                        ) : (
                            <>
                                <Button
                                    id="cancel-view-key"
                                    variant="outline"
                                    onClick={() => {
                                        resetInputs();
                                        toggleCreate();
                                    }}>
                                    Cancel
                                </Button>
                                <Button
                                    id="copy-api-key"
                                    variant="outline"
                                    onClick={async () => {
                                        try {
                                            await navigator.clipboard.writeText(
                                                newKey?.value ?? ""
                                            );
                                            setKeyCopied(true);
                                            clearTimeout(copyTimerRef.current);
                                            copyTimerRef.current = setTimeout(
                                                () => setKeyCopied(false),
                                                2000
                                            );
                                        } catch {
                                            setErrorMsg(
                                                "Failed to copy to clipboard"
                                            );
                                        }
                                    }}>
                                    {keyCopied ? (
                                        <div className="flex items-center gap-1">
                                            <Check className="fill-emerald-500" />
                                            Copied
                                        </div>
                                    ) : (
                                        "Copy"
                                    )}
                                </Button>
                            </>
                        )}
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export const CreateAPIKey = WithErrorBoundary(CreateAPIKeyComponent);
