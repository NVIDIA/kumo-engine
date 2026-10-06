// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactNode, useRef, useState } from "react";

import {
    cn,
    Label,
    RadioGroup,
    RadioGroupItem
} from "@kumo-ai/kumo-ui-catalog";
import { S3SourceTable } from "gen/v0/types";
import { KumoapiTableTableDefinition } from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { EntityErrorMsg } from "types/common";
import { DropdownOption, InputState, TableDropdownOption } from "types/inputs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { formLabelClassName, formRowTopAlignedClassName } from "./formRow";
import LocalFileUpload from "./LocalFileUpload";
import SourceTable from "./SourceTable";

export interface LocalUploadFlowProps {
    tableNameField: ReactNode;
    sourceTableErrorLabel: ReactNode;
    showFileSelection: boolean;
    flowType: FlowType;
    nameInputState: InputState;
    onTableSelect: (e: TableDropdownOption) => void;
    handleError: HandleErrorFunc;
    clearErrors: () => void;
    selectedConnector: DropdownOption;
    connectorId: string;
    sourceTableName: string;
    resetTableSelection: () => void;
    errLabelMsg: EntityErrorMsg;
    clearErrLabel: () => void;
    setInProgress: (value: boolean) => void;
    setUploadSuccess: (value: boolean) => void;
    suggestTableName: (name: string) => void;
    setTableName: React.Dispatch<React.SetStateAction<string>>;
    uploadSuccess: boolean;
    tableArgs: KumoapiTableTableDefinition;
    setTableArgs: (args: KumoapiTableTableDefinition) => void;
    fileConnectors: string[];
    setColumnNamesAdjusted: React.Dispatch<React.SetStateAction<boolean>>;
    sourceTableReloadKey: number;
}

/** The two mutually exclusive ways to point the table at a file. */
const FileSource = {
    Existing: "existing",
    New: "new"
} as const;

type FileSourceMode = (typeof FileSource)[keyof typeof FileSource];

/**
 * Table-details layout for the local-upload flow. Both ways of choosing a file
 * write the same source path, so they are a radio choice and only the selected
 * one renders — offered together, a stale pick could survive under an upload.
 */
const LocalUploadFlow = ({
    tableNameField,
    sourceTableErrorLabel,
    showFileSelection,
    flowType,
    nameInputState,
    onTableSelect,
    handleError,
    clearErrors,
    selectedConnector,
    connectorId,
    sourceTableName,
    resetTableSelection,
    errLabelMsg,
    clearErrLabel,
    setInProgress,
    setUploadSuccess,
    suggestTableName,
    setTableName,
    uploadSuccess,
    tableArgs,
    setTableArgs,
    fileConnectors,
    setColumnNamesAdjusted,
    sourceTableReloadKey
}: LocalUploadFlowProps) => {
    const [fileSource, setFileSource] = useState<FileSourceMode>(
        FileSource.New
    );
    const [uploading, setUploading] = useState(false);

    /** Clears the selection only — the file itself stays on the server. */
    const clearSourceSelection = () => {
        const args = {
            ...tableArgs,
            source_table: { ...tableArgs.source_table }
        } as KumoapiTableTableDefinition;
        (args.source_table as S3SourceTable).source = {
            mode: "relative",
            path: ""
        };
        args.cols = [];
        args.pkey = "";
        args.end_time_col = "";
        args.time_col = "";
        setTableArgs(args);
    };

    /** Each side's last selection, so switching back restores it. */
    const parked = useRef<
        Record<
            FileSourceMode,
            { args: KumoapiTableTableDefinition; uploaded: boolean } | null
        >
    >({ existing: null, new: null });

    const onFileSourceChange = (value: string) => {
        const next = value as FileSourceMode;
        if (next === fileSource) {
            return;
        }
        if (errLabelMsg?.entity === "sourcetable") {
            clearErrLabel();
        }
        parked.current[fileSource] = sourceTableName
            ? { args: tableArgs, uploaded: uploadSuccess }
            : null;
        const restored = parked.current[next];
        if (restored) {
            setTableArgs(restored.args);
            setUploadSuccess(restored.uploaded);
        } else {
            clearSourceSelection();
            setUploadSuccess(false);
        }
        setFileSource(next);
    };

    const trackedSetInProgress = (status: boolean) => {
        if (errLabelMsg?.entity === "sourcetable") {
            clearErrLabel();
        }
        setUploading(status);
        setInProgress(status);
    };

    const fileUpload = (
        <LocalFileUpload
            setInProgress={trackedSetInProgress}
            setFileName={(name) => {
                if (errLabelMsg?.entity === "sourcetable") {
                    clearErrLabel();
                }
                suggestTableName(name);
            }}
            setUploadSuccess={(status) => {
                if (errLabelMsg?.entity === "sourcetable") {
                    clearErrLabel();
                }
                setUploadSuccess(status);
            }}
            setTableName={setTableName}
            uploadSuccess={uploadSuccess}
            tableArgs={tableArgs}
            setTableArgs={setTableArgs}
            fileConnectors={fileConnectors}
            flowType={flowType}
            connectorId={connectorId}
            selectedSourceTableName={sourceTableName}
            setColumnNamesAdjusted={setColumnNamesAdjusted}
        />
    );

    return (
        <div className="flex w-full flex-col gap-4">
            {showFileSelection ? (
                <div className={formRowTopAlignedClassName}>
                    <Label className={cn(formLabelClassName, "pt-2")}>
                        File
                    </Label>
                    <div className="flex w-full flex-col gap-3">
                        <RadioGroup
                            className="flex flex-row gap-6"
                            value={fileSource}
                            onValueChange={onFileSourceChange}
                            disabled={uploading}>
                            <RadioGroupItem
                                value={FileSource.New}
                                id="file-source-new"
                                title="Upload a new file"
                                active={fileSource === FileSource.New}
                            />
                            <RadioGroupItem
                                value={FileSource.Existing}
                                id="file-source-existing"
                                title="Choose an existing file"
                                active={fileSource === FileSource.Existing}
                                // A landed file holds the choice until removed.
                                disabled={uploadSuccess}
                            />
                        </RadioGroup>
                        {fileSource === FileSource.Existing ? (
                            <SourceTable
                                flowType={flowType}
                                nameValidationState={nameInputState}
                                onTableChange={onTableSelect}
                                handleError={handleError}
                                clearErrors={clearErrors}
                                selectedConnector={selectedConnector}
                                setLoading={() => {}}
                                connectorId={connectorId}
                                sourceTableName={sourceTableName}
                                resetTableSelection={resetTableSelection}
                                label=""
                                placeholder="Choose a file"
                                emptyText="No uploaded files found."
                                selectClassName="w-full"
                                reloadSignal={sourceTableReloadKey}
                            />
                        ) : (
                            fileUpload
                        )}
                        {sourceTableErrorLabel}
                    </div>
                </div>
            ) : (
                <div className="w-full">
                    {fileUpload}
                    {sourceTableErrorLabel}
                </div>
            )}
            {tableNameField}
        </div>
    );
};

export default WithErrorBoundary(LocalUploadFlow);
