// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

import {
    Button,
    Close,
    FilePresent,
    FileUploadItem,
    formatFileSize,
    Upload
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import { getProjectsSourceTablesByConnectorId } from "gen/v0/api/connectors/connectors";
import {
    S3SourceTable,
    StartFileUploadResponse,
    UploadFileType
} from "gen/v0/types";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useProjectId } from "hooks/project/projects";
import { useFileUpload } from "hooks/tables/tables";

import { CustomError } from "types/common";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { SUPPORT_CONTACT_MESSAGE } from "utils/Constants";
import { parseErrorMessage } from "utils/errorHandler";
import { getKeys } from "utils/Helpers/objectHelpers";
import { invalidateProjectSourceTables } from "utils/queryInvalidation";
import { generateUniqueTableName } from "utils/tables";

type PART_META = { part_number: number; etag: string };

const MAX_CONCURRENT_UPLOADS = 4;

export const runWithUploadConcurrency = async (
    taskCount: number,
    task: (index: number) => Promise<void>
) => {
    let nextTask = 0;
    let failed = false;
    const worker = async () => {
        while (!failed) {
            const index = nextTask++;
            if (index >= taskCount) return;
            try {
                await task(index);
            } catch (error) {
                failed = true;
                throw error;
            }
        }
    };
    await Promise.all(
        Array.from(
            { length: Math.min(MAX_CONCURRENT_UPLOADS, taskCount) },
            worker
        )
    );
};

export const calculateUploadProgress = (
    uploadedBytes: number,
    totalBytes: number
) => {
    const percentage = Math.min(100, (100 * uploadedBytes) / totalBytes);
    return percentage > 0 && percentage < 1 ? 1 : Math.round(percentage);
};

// The server caps a source-table name at 256 characters and rejects an empty
// one, so a file named ".csv" or one with a very long name would fail the
// upload rather than the file itself. The reserve leaves room for the
// uniqueness suffix, which is appended after the trim.
const MAX_SOURCE_NAME_LENGTH = 256;
const UNIQUE_SUFFIX_RESERVE = 8;

const deriveNameFromFile = (file: File) =>
    file?.name
        ?.split(".")[0]
        ?.replace(/[^a-zA-Z0-9]/g, "_")
        ?.slice(0, MAX_SOURCE_NAME_LENGTH - UNIQUE_SUFFIX_RESERVE) ||
    "uploaded_file";

type UploadStatus = {
    [key: string]: number | string | boolean | undefined;
    uploaded?: number;
    done?: boolean;
};

/**
 * File Upload components to upload file (csv, parquet) to
 * create Source Table for file connectors(csv, parquet)
 *
 */
export interface LocalFileUploadProps {
    setInProgress: (value: boolean) => void;
    setFileName: (name: string) => void;
    setUploadSuccess: (value: boolean) => void;
    uploadSuccess: boolean;
    tableArgs: KumoapiTableTableDefinition;
    setTableArgs: (args: KumoapiTableTableDefinition) => void;
    fileConnectors: string[];
    flowType: FlowType;
    setTableName: React.Dispatch<React.SetStateAction<string>>;
    connectorId: string;
    selectedSourceTableName: string;
    setColumnNamesAdjusted: React.Dispatch<React.SetStateAction<boolean>>;
}

const LocalFileUpload = ({
    setInProgress,
    setFileName,
    setUploadSuccess,
    uploadSuccess,
    tableArgs,
    setTableArgs,
    fileConnectors,
    flowType,
    setTableName,
    connectorId,
    selectedSourceTableName,
    setColumnNamesAdjusted
}: LocalFileUploadProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectId();
    const [selectedFiles, setSelectedFiles] = useState<File[] | null>(null);
    const [currentFile, setCurrentFile] = useState<File | null>(null);
    const [progress, setProgress] = useState(0);
    const [uploadError, setUploadError] = useState("");
    const [uploadedFileType, setUploadedFileType] =
        useState<UploadFileType | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const MAX_SIZE = 100 * 1024 * 1024 * 1024; // 100 GiB
    const MAX_SIZE_LABEL = "100 GiB";
    const MAX_SIZE_ERR =
        `File size is too large. Kumo supports a maximum` +
        ` file size of ${MAX_SIZE_LABEL}`;

    /** Method to update error */
    const trigerUploadError = useCallback(
        (err1 = "", prefix = "") => {
            setProgress(0);
            setInProgress(false);
            if (err1) {
                setUploadError(err1);
            } else {
                setUploadError(
                    prefix +
                        `Could not upload the file! ${SUPPORT_CONTACT_MESSAGE} for help.`
                );
            }
        },
        [setInProgress]
    );

    const handleUploadError = useCallback<HandleErrorFunc>(
        (error: CustomError) => {
            trigerUploadError(parseErrorMessage(error), "");
        },
        [trigerUploadError]
    );

    const { startFileUpload, completeFileUpload, deleteUploadedFile, upload } =
        useFileUpload(handleUploadError);

    useEffect(() => {
        resetAll();
    }, []);

    // Reset all the variables
    const resetAll = () => {
        setSelectedFiles(null);
        setCurrentFile(null);
        setUploadError("");
        setProgress(0);
    };

    /**
     * File Selection handler method for FilePicker.
     * Checks the file type and MAX allowed size
     *
     * @param files Selected files list
     */
    const onDrop = (files: Array<File>) => {
        abortRef.current?.abort();
        resetAll();
        setUploadedFileType(null);
        setColumnNamesAdjusted(false);
        if (files?.length > 0) {
            const fileType = files[0]?.name?.split(".")?.pop();
            // Kept in state so the card has something to show the error on.
            if (![".csv", ".parquet"].includes("." + fileType)) {
                setSelectedFiles(files);
                setUploadError(`Please select '.csv' or '.parquet' files only`);
                return;
            }
            if (files[0].size > MAX_SIZE) {
                setSelectedFiles(files);
                setUploadError(MAX_SIZE_ERR);
                return;
            }

            setSelectedFiles(files);
            void onUpload(files[0]);
        }
    };

    /**
     * Advisory only. The server copies a finished upload over an existing
     * path, so a repeat name destroys the first file; this narrows that
     * window but cannot close it — two uploads racing still collide. A real
     * guarantee needs the name reserved server-side (kumo#30771).
     */
    const resolveSourceTableName = async (file: File) => {
        const base = deriveNameFromFile(file);
        if (!projectId || !connectorId) {
            return base;
        }
        try {
            const response = await getProjectsSourceTablesByConnectorId(
                projectId,
                connectorId
            );
            return generateUniqueTableName(
                base,
                response?.data?.table_names ?? []
            );
        } catch {
            // Losing the lookup is not worth blocking an upload over.
            return base;
        }
    };

    /**
     * Transforms the content of the first chunk by modifying its first line
     * @param chunk The chunk content to transform
     * @returns Transformed chunk content
     */
    const transformFirstChunk = async (chunk: Blob): Promise<Blob> => {
        // Convert blob to text to modify the first line
        const text = await chunk.text();
        const lines = text.split("\n");

        const regex = /[\s]|[^\w,]/;
        if (lines.length > 0) {
            if (regex.test(lines[0].trim())) {
                setColumnNamesAdjusted(true);
            }

            const columns = lines[0].trim().split(",");
            let unnamedColumnCount = 0;

            const processedColumns = columns.map((column) => {
                const trimmedColumn = column.trim();

                // If column is empty or only contains whitespace
                if (!trimmedColumn) {
                    unnamedColumnCount++;
                    return `auto_named_${unnamedColumnCount - 1}`;
                }

                // Replace non-alphanumeric characters (except ,| and tabs) with underscores
                return trimmedColumn.replace(/[^a-zA-Z0-9,|\t]/g, "_");
            });

            // Join the processed columns back together
            lines[0] = processedColumns.join(",");
        }

        // Convert back to blob
        return new Blob([lines.join("\n")], { type: chunk.type });
    };

    /**
     * Uploading files to S3 bucket by getting pre-signed urls from server
     *
     * This method invokes the `start_upload_file` method to `get presigned_part_urls`
     * and uploades the files as chunk to S3 bucket using these URLs
     *
     * After upload, `complete_file_upload` API will be  invoked to complete the
     * upload process and refresh the UI source table list by calling
     * `get_source_table` API
     */
    const onUpload = async (file: File) => {
        const controller = new AbortController();
        abortRef.current = controller;
        setUploadError("");
        const fileToUpload = file;
        setCurrentFile(fileToUpload);
        setProgress(1);
        setInProgress(true);
        const fileType = fileToUpload?.name?.split(".").pop();

        if (!connectorId) {
            trigerUploadError("The project upload connector is not ready.", "");
            return;
        }
        const sourceTableName = await resolveSourceTableName(fileToUpload);
        // Suggest what it was actually stored as, so the two cannot disagree.
        setFileName(sourceTableName);
        const apiFileType =
            fileType === "parquet"
                ? UploadFileType.PARQUET
                : UploadFileType.CSV;
        setUploadedFileType(apiFileType);
        const args = {
            ...tableArgs,
            source_table: { ...tableArgs.source_table }
        } as KumoapiTableTableDefinition;
        (args.source_table as S3SourceTable).connector_id = connectorId;
        (args.source_table as S3SourceTable).source = {
            mode: "relative",
            path: sourceTableName
        };
        (args.source_table as S3SourceTable).file_type = apiFileType;
        //Creating num_parts
        let num_parts = 1;
        const CHUNK_SIZE = 1024 * 1024 * 100; //100MB
        if (fileToUpload && fileToUpload?.size > CHUNK_SIZE) {
            num_parts = Math.ceil(fileToUpload.size / CHUNK_SIZE);
        }

        const upload_config = {
            source_table_name: sourceTableName,
            file_type: apiFileType,
            num_parts: num_parts
        };
        let config: StartFileUploadResponse = {
            temp_upload_path: "",
            upload_id: "",
            presigned_part_urls: {}
        };
        try {
            const resp = await startFileUpload(upload_config);
            config = resp?.data as StartFileUploadResponse;
        } catch (error) {
            trigerUploadError(parseErrorMessage(error as CustomError), "");
            return;
        }
        const status: Array<UploadStatus> = [];
        const parts_metadata: Array<PART_META> = [];
        const uploadUrls: string[] = [];
        for (let i = 0; i < num_parts; i++) {
            const upload_url = config.presigned_part_urls?.[i + 1 + ""];
            if (!upload_url) {
                trigerUploadError("Upload part URL is missing.", "");
                return;
            }
            uploadUrls.push(upload_url);
            status[i] = { uploaded: 0, done: false } as UploadStatus;
        }

        const uploadPart = async (i: number) => {
            const start = CHUNK_SIZE * i;
            const end = Math.min(
                start + CHUNK_SIZE,
                (fileToUpload as File)?.size
            );
            const CHUNK = fileToUpload?.slice(start, end);
            const upload_url = uploadUrls[i];
            const isCsvFile = fileType === "csv";
            const transformedChunk =
                i === 0 && isCsvFile
                    ? await transformFirstChunk(CHUNK as Blob)
                    : (CHUNK as Blob);
            const response = await upload(
                upload_url,
                transformedChunk,
                (event: { loaded?: number }) => {
                    status[i].uploaded = event?.loaded;
                    const pKeys = getKeys(status);
                    let up = 0;
                    pKeys?.forEach((v) => {
                        up += status[parseInt(v)]?.uploaded as number;
                    });
                    setProgress(calculateUploadProgress(up, fileToUpload.size));
                },
                controller.signal
            );
            const etag = response?.headers?.etag || response?.data?.etag;
            if (!etag) {
                trigerUploadError("", "Error: ETag, ");
                throw new Error("Error: ETag missing");
            }
            parts_metadata.push({ part_number: i + 1, etag });
            status[i].done = true;
        };

        try {
            await runWithUploadConcurrency(num_parts, uploadPart);
            if (controller.signal.aborted) {
                return;
            }
            const p = getKeys(status);
            if (
                !p.map((p) => status[parseInt(p)].done === true).includes(false)
            ) {
                //meta data should be sorted by part_number
                parts_metadata.sort((a, b) => {
                    return a?.part_number - b?.part_number;
                });
                //Creating config to complete upload API call
                const config2 = {
                    source_table_name: sourceTableName,
                    file_type: apiFileType,
                    upload_id: config.upload_id,
                    temp_upload_path: config.temp_upload_path,
                    parts_metadata: parts_metadata
                };

                completeFileUpload(config2, () => {
                    // Completion can land after the upload was cancelled or
                    // replaced. The file exists on the server either way, so
                    // refresh the list to surface it rather than attaching a
                    // file the form has already moved on from.
                    if (controller.signal.aborted) {
                        if (projectId) {
                            invalidateProjectSourceTables(
                                queryClient,
                                projectId,
                                connectorId
                            );
                        }
                        return;
                    }
                    // The picked file is deliberately kept: it is what the
                    // success card renders its name and size from.
                    setUploadError("");
                    setProgress(0);
                    setInProgress(false);
                    setUploadSuccess(true);
                    (args.source_table as S3SourceTable).source = {
                        mode: "relative",
                        path: sourceTableName
                    };
                    (args.source_table as S3SourceTable).file_type =
                        apiFileType;
                    setTableArgs(args);
                    if (projectId) {
                        invalidateProjectSourceTables(
                            queryClient,
                            projectId,
                            connectorId
                        );
                    }
                });
            } else {
                trigerUploadError("Request failed. Please try again!", "");
            }
        } catch (error) {
            if (controller.signal.aborted) {
                return;
            }
            console.error(error);
            trigerUploadError();
        }
    };

    /** Abandons an upload in flight, leaving the picker ready for another. */
    const cancelUpload = () => {
        abortRef.current?.abort();
        resetAll();
        setUploadedFileType(null);
        setInProgress(false);
    };

    const removeTable = () => {
        const existingFileType = (tableArgs.source_table as S3SourceTable)
            .file_type;
        const fileType =
            uploadedFileType ??
            (existingFileType === UploadFileType.PARQUET
                ? UploadFileType.PARQUET
                : existingFileType === UploadFileType.CSV
                  ? UploadFileType.CSV
                  : null);
        if (!fileType) {
            return;
        }
        const config = {
            source_table_name: selectedSourceTableName ?? "",
            file_type: fileType
        };
        deleteUploadedFile(config, () => {
            if (projectId) {
                invalidateProjectSourceTables(
                    queryClient,
                    projectId,
                    connectorId
                );
            }
            setUploadSuccess(false);
        });

        setSelectedFiles(null);
        setCurrentFile(null);
        setUploadError("");
        setInProgress(false);
        setProgress(0);
        setTableName("");
        setUploadedFileType(null);
        const args = { ...tableArgs };
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

    const pickedFile = selectedFiles?.[0] ?? currentFile ?? null;

    // One file at a time, so a landed file replaces the picker. Edit and
    // duplicate open here too, minus the remove.
    const uploadedName = pickedFile?.name ?? selectedSourceTableName;
    if (
        fileConnectors.includes(connectorId as string) &&
        selectedSourceTableName !== "" &&
        (flowType !== FlowType.CREATE || uploadSuccess)
    ) {
        return (
            <UploadedFileCard
                name={uploadedName as string}
                size={pickedFile?.size}
                onRemove={
                    flowType === FlowType.CREATE && progress === 0
                        ? removeTable
                        : undefined
                }
            />
        );
    }

    /**
     * Rejected reads as error, landed as success, anything else in flight.
     * `null` rather than `undefined` when there is no file: `undefined` tells
     * Upload to manage its own state, which resurfaces a removed file.
     */
    const uploadItem: FileUploadItem | null = pickedFile
        ? {
              id: pickedFile.name,
              file: pickedFile,
              status: uploadError
                  ? "error"
                  : uploadSuccess
                    ? "success"
                    : "uploading",
              uploadedBytes: uploadSuccess
                  ? pickedFile.size
                  : Math.round((progress / 100) * pickedFile.size),
              errorMessage: uploadError || undefined,
              hidePreview: true
          }
        : null;

    return (
        <>
            {/* KUI's type scale runs small against this form, so its slots are
                stepped up from here rather than through className props. */}
            <div
                data-testid="local-upload-dropzone"
                className="[&_[data-slot=upload-item-content-heading]]:text-base [&_[data-slot=upload-item-content-info]]:text-sm">
                <Upload
                    accept=".csv,.parquet"
                    className="w-full font-medium [&_[data-slot=upload-description]]:text-sm"
                    kind="flat"
                    multiple={false}
                    showPreview={false}
                    status={
                        uploadError
                            ? "error"
                            : uploadSuccess
                              ? "success"
                              : undefined
                    }
                    value={uploadItem}
                    onValueChange={(item: FileUploadItem | undefined) => {
                        onDrop(item?.file ? [item.file] : []);
                    }}
                    onFileRemove={() => {
                        if (progress > 0) {
                            cancelUpload();
                            return;
                        }
                        resetAll();
                        setUploadedFileType(null);
                    }}>
                    {`Only '.CSV' or '.Parquet' files are ` +
                        `allowed with max size ${MAX_SIZE_LABEL}`}
                </Upload>
            </div>
        </>
    );
};

/**
 * Mirrors the card KUI draws while uploading, so the row keeps its shape when
 * the upload finishes. Size is only known while the browser holds the file.
 */
const UploadedFileCard = ({
    name,
    size,
    onRemove
}: {
    name: string;
    size?: number;
    onRemove?: () => void;
}) => (
    <div className="flex w-full items-center justify-between rounded-sm border border-border-secondary px-3 py-2.5">
        <div className="flex items-center gap-3">
            <FilePresent className="size-5 fill-title-secondary" />
            <div className="flex flex-col">
                <span className="text-base font-medium leading-5 text-title-primary">
                    {name}
                </span>
                <span className="text-sm leading-5 text-title-secondary">
                    {size ? `${formatFileSize(size)} · Uploaded` : "Uploaded"}
                </span>
            </div>
        </div>
        {onRemove && (
            <Button
                id={"remove-file"}
                aria-label="Remove"
                onClick={onRemove}
                variant="no-outline"
                size="xs">
                <Close className="size-4" />
            </Button>
        )}
    </div>
);

export default WithErrorBoundary(LocalFileUpload);
