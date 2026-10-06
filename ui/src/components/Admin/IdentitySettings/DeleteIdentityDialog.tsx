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
import { IdentityConnectorUsage } from "gen/v0/types";

export interface DeleteIdentityDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onConfirm: () => void;
    identityName: string;
    /**
     * Connectors that currently keep this identity in use. When non-empty the
     * identity cannot be deleted, so the dialog lists them and drops the Delete
     * action entirely (Cancel only), mirroring the connector "Cannot Delete"
     * alert.
     */
    blockingConnectors: IdentityConnectorUsage[];
    /**
     * True while the blocker check (or the delete itself) is in flight. Delete
     * stays disabled so the admin cannot confirm before the check resolves.
     */
    loading: boolean;
    /**
     * True when the blocker check could not complete. The identity's usage is
     * unknown, so Delete stays disabled rather than risking a blind deletion.
     */
    checkFailed: boolean;
}

const blockerLabel = (connector: IdentityConnectorUsage): string =>
    `${connector.connector_name} (project ${
        connector.project_display_name ?? connector.project_id
    })`;

export const DeleteIdentityDialog = ({
    open,
    onOpenChange,
    onConfirm,
    identityName,
    blockingConnectors,
    loading,
    checkFailed
}: DeleteIdentityDialogProps) => {
    const isBlocked = blockingConnectors.length > 0;
    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent
                data-testid={"delete-identity-dialog"}
                className="top-1/2">
                <div className="flex flex-col justify-between px-7.5 py-6">
                    <DialogTitle className="text-title-primary leading-7">
                        {isBlocked
                            ? "Cannot Delete Identity"
                            : "Delete Identity"}
                    </DialogTitle>
                    <Spacer height={16} />
                    <DialogDescription className="flex-1 p-0 m-0 text-title-primary text-base font-medium leading-5 outline-none max-h-100 overflow-auto [&::-webkit-scrollbar]:h-2.5 [&::-webkit-scrollbar]:w-2.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[var(--scrollbar-dark)] [&::-webkit-scrollbar-track]:bg-transparent">
                        {isBlocked ? (
                            <>
                                You cannot delete this identity before removing
                                or repointing the connectors that use it:
                                <span className="mt-3 block font-normal">
                                    {blockingConnectors.map((connector) => (
                                        <span
                                            key={connector.connector_id}
                                            className="block">
                                            {blockerLabel(connector)}
                                        </span>
                                    ))}
                                </span>
                            </>
                        ) : checkFailed ? (
                            <>
                                We couldn&apos;t check which connectors use{" "}
                                <span className="font-semibold">
                                    {identityName}
                                </span>
                                . Close this dialog and try again.
                            </>
                        ) : (
                            <>
                                Are you sure you want to delete identity &quot;
                                <span className="font-semibold">
                                    {identityName}
                                </span>
                                &quot;? Any projects using this identity will
                                lose access to its configured credentials. This
                                action cannot be undone.
                            </>
                        )}
                    </DialogDescription>
                    <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end mt-6">
                        <Button
                            id="cancel-delete-identity"
                            variant="outline"
                            onClick={() => onOpenChange(false)}>
                            Cancel
                        </Button>
                        {!isBlocked && (
                            <Button
                                id="confirm-delete-identity"
                                variant="destructive"
                                disabled={loading || checkFailed}
                                onClick={onConfirm}>
                                Delete
                            </Button>
                        )}
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
};
