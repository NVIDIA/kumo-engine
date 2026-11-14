// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useRef, useState } from "react";

import {
    Button,
    Dialog,
    DialogContent,
    DialogDescription,
    DialogTitle,
    Input,
    Select
} from "@kumo-ai/kumo-ui-catalog";
import { ConnectorResponse, ProjectStatus } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useCopyConnector } from "hooks/connector/connector";
import { useProjectList } from "hooks/project/projects";

import { JsConnector } from "types/connectors";

import { getProjectPath } from "utils/Constants";
import { useCustomNavigation } from "utils/navigator";

export interface CopyConnectorDialogProps {
    open: boolean;
    onClose: () => void;
    connector: JsConnector | null;
    sourceProjectId: string;
}

const CopyConnectorDialogComponent = ({
    open,
    onClose,
    connector,
    sourceProjectId
}: CopyConnectorDialogProps) => {
    const navigate = useCustomNavigation();
    const { projectListResponse, isLoading: isProjectListLoading } =
        useProjectList();
    const { copyConnector, isCopying } = useCopyConnector();
    const [targetProjectId, setTargetProjectId] = useState("");
    const [name, setName] = useState(connector?.name ?? "");
    const [errorMsg, setErrorMsg] = useState("");
    const [copied, setCopied] = useState<ConnectorResponse | null>(null);
    const copyInFlightRef = useRef(false);

    const projectOptions = useMemo(
        () =>
            (projectListResponse?.data ?? [])
                .filter(
                    (p) =>
                        p.id !== sourceProjectId &&
                        p.status !== ProjectStatus.ARCHIVED
                )
                .map((p) => ({
                    value: p.id,
                    label: p.display_name || p.id
                })),
        [projectListResponse?.data, sourceProjectId]
    );

    const targetProjectLabel =
        projectOptions.find((p) => p.value === targetProjectId)?.label ?? "";

    const onCopyClick = async () => {
        if (!connector || copyInFlightRef.current) return;
        copyInFlightRef.current = true;
        try {
            const result = await copyConnector(sourceProjectId, connector.id, {
                target_project_id: targetProjectId,
                name: name.trim()
            });
            if (result.error !== null) {
                setErrorMsg(result.error);
            } else {
                setCopied(result.connector);
            }
        } finally {
            copyInFlightRef.current = false;
        }
    };

    const canCopy =
        !!connector && !!targetProjectId && !!name.trim() && !isCopying;

    return (
        <Dialog
            open={open}
            onOpenChange={(isOpen) => {
                if (!isOpen) onClose();
            }}>
            <DialogContent
                data-testid="copy-connector-dialog"
                className="top-1/2 max-w-134.75"
                onInteractOutside={(e) => e.preventDefault()}>
                <div className="flex flex-col px-7.5 py-6 gap-4">
                    <DialogTitle className="text-title-primary leading-7">
                        Copy connector to project
                    </DialogTitle>
                    <DialogDescription
                        className="p-0 m-0 text-title-primary text-base font-medium leading-5 outline-none"
                        asChild>
                        <div className="flex flex-col gap-4">
                            {copied ? (
                                <p>
                                    Connector "{copied.name}" was copied to "
                                    {targetProjectLabel}".
                                </p>
                            ) : (
                                <>
                                    <p className="text-sm text-title-secondary">
                                        The copy uses the same configuration and
                                        identity as "{connector?.name}". The
                                        identity must be assigned to the target
                                        project's group. Tables are not copied.
                                    </p>
                                    <div className="flex flex-col gap-2">
                                        <label
                                            htmlFor="copy-connector-target-project"
                                            className="text-sm font-medium text-title-secondary">
                                            Target project
                                        </label>
                                        <Select
                                            id="copy-connector-target-project"
                                            placeholder={
                                                isProjectListLoading
                                                    ? "Loading projects..."
                                                    : "Select project"
                                            }
                                            search={true}
                                            emptyText="No other projects found."
                                            mode="single"
                                            selectedValues={
                                                targetProjectId
                                                    ? [targetProjectId]
                                                    : undefined
                                            }
                                            options={projectOptions}
                                            onValuesChange={(values) => {
                                                setTargetProjectId(
                                                    values[0] ?? ""
                                                );
                                                setErrorMsg("");
                                            }}
                                        />
                                    </div>
                                    <div className="flex flex-col gap-2">
                                        <label
                                            htmlFor="copy-connector-name"
                                            className="text-sm font-medium text-title-secondary">
                                            Connector name
                                        </label>
                                        <Input
                                            id="copy-connector-name"
                                            value={name}
                                            onChange={(e) => {
                                                setName(e.target.value);
                                                setErrorMsg("");
                                            }}
                                            className="h-9 !text-base font-medium py-2 px-4"
                                        />
                                    </div>
                                    {errorMsg && (
                                        <p
                                            role="alert"
                                            className="text-sm text-red-500">
                                            {errorMsg}
                                        </p>
                                    )}
                                </>
                            )}
                        </div>
                    </DialogDescription>
                    <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end mt-2">
                        {copied ? (
                            <>
                                <Button
                                    id="close-copy-connector"
                                    variant="outline"
                                    onClick={onClose}>
                                    Close
                                </Button>
                                <Button
                                    id="open-copied-connector"
                                    variant="primary"
                                    onClick={() => {
                                        onClose();
                                        navigate(
                                            getProjectPath(copied.project_id)
                                                .Connectors
                                        );
                                    }}>
                                    Go to project
                                </Button>
                            </>
                        ) : (
                            <>
                                <Button
                                    id="cancel-copy-connector"
                                    variant="outline"
                                    onClick={onClose}>
                                    Cancel
                                </Button>
                                <Button
                                    id="confirm-copy-connector"
                                    variant="primary"
                                    disabled={!canCopy}
                                    onClick={onCopyClick}>
                                    {isCopying ? "Copying..." : "Copy"}
                                </Button>
                            </>
                        )}
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export const CopyConnectorDialog = WithErrorBoundary(
    CopyConnectorDialogComponent
);
