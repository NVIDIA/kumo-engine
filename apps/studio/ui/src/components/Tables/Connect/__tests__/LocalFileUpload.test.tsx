// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { setDeploymentType } from "stores/__tests__/MockFunctions/DebugInfo.mock";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import { FlowType } from "types/tables";

import { getConnectorIDFromTableDefSource } from "utils/Converters/tablesConverter";
import { ReactQueryKeys } from "utils/reactQueryKeys";

import LocalFileUpload, {
    calculateUploadProgress,
    LocalFileUploadProps,
    runWithUploadConcurrency
} from "../LocalFileUpload";

import { setStartFileUploadTestType } from "./MockFunction/StartFileUpload.mock";

const projectId = "project_abc123def456";

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project_abc123def456"
}));

const tableData = {
    source_table: {
        connector_id: "project_upload_connector",
        file_type: "CSV",
        data_source_type: "S3",
        source: { mode: "relative", path: "test_table" }
    },
    cols: [],
    pkey: ""
} as KumoapiTableTableDefinition;
const connectorId = getConnectorIDFromTableDefSource(tableData?.source_table);

const setInProgress = vi.fn();
const setFileName = vi.fn();
const setUploadSuccess = vi.fn();
const setTableArgs = vi.fn();
const setTableName = vi.fn();

const defaultProps: LocalFileUploadProps = {
    setInProgress,
    setFileName,
    setUploadSuccess,
    uploadSuccess: false,
    tableArgs: tableData as KumoapiTableTableDefinition,
    setTableArgs,
    fileConnectors: [],
    flowType: FlowType.CREATE,
    setTableName,
    connectorId,
    selectedSourceTableName: "",
    setColumnNamesAdjusted: vi.fn()
};

describe("LocalFileUpload", () => {
    const queryClient = createQueryClient();
    const renderLocalFileUpload = async (props: LocalFileUploadProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <LocalFileUpload {...(props as LocalFileUploadProps)} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    /**
     * The KUI upload trigger is a label around a file input, so a selection is
     * simulated by putting the file on the input rather than firing a drop on
     * a dropzone wrapper.
     */
    const simulateFileDrop = async (file: File) => {
        const input = screen
            .getByTestId("local-upload-dropzone")
            .querySelector("input[type=file]") as HTMLInputElement;
        Object.defineProperty(input, "files", {
            value: [file],
            configurable: true
        });
        await act(() => fireEvent.change(input));
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        setDeploymentType(deploymentType.SAAS);
    });

    it("Should run at most four upload tasks concurrently", async () => {
        let active = 0;
        let maxActive = 0;

        await runWithUploadConcurrency(8, async () => {
            active++;
            maxActive = Math.max(maxActive, active);
            await new Promise((resolve) => setTimeout(resolve, 0));
            active--;
        });

        expect(maxActive).toBe(4);
    });

    it("Should include queued parts in upload progress", () => {
        expect(calculateUploadProgress(400 * 1024 ** 2, 500 * 1024 ** 2)).toBe(
            80
        );
    });

    it("Should render `LocalFileUpload`", async () => {
        await renderLocalFileUpload(defaultProps);

        expect(screen.getByText("Choose file")).toBeInTheDocument();
        expect(
            screen.getByText(/or drag and drop it here/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(
                "Only '.CSV' or '.Parquet' files are allowed with max size 100 GiB"
            )
        ).toBeInTheDocument();
    });

    it.each([
        { fileName: "sample.csv", fileType: "text/csv" },
        { fileName: "sample.parquet", fileType: "text/parquet" }
    ])(
        "Should handle drag and drop of $fileName file",
        async ({ fileName, fileType }) => {
            await renderLocalFileUpload(defaultProps);

            const file = new File(["sample"], fileName, { type: fileType });
            await simulateFileDrop(file);
            expect(await screen.findByText(fileName)).toBeInTheDocument();
            // No Upload button: choosing the file is what starts the upload.
            expect(
                screen.queryByRole("button", { name: "Upload" })
            ).not.toBeInTheDocument();
        }
    );

    it.each([
        {
            fileName: "sample.csv",
            fileType: "text/csv",
            uploadFileType: "CSV"
        },
        {
            fileName: "sample.parquet",
            fileType: "text/parquet",
            uploadFileType: "PARQUET"
        }
    ])(
        "Should show progress bar when `Upload` button is clicked for $fileName file",
        async ({ fileName, fileType, uploadFileType }) => {
            await renderLocalFileUpload({
                ...defaultProps,
                setInProgress,
                fileConnectors: [connectorId]
            });

            const file = new File(["sample"], fileName, { type: fileType });
            await simulateFileDrop(file);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    `v0/projects/${projectId}/connectors/uploads/start`,
                    {
                        file_type: uploadFileType,
                        num_parts: 1,
                        source_table_name: "sample"
                    },
                    {
                        "headers": {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });

            await waitFor(() =>
                expect(setInProgress).toHaveBeenCalledWith(true)
            );
        }
    );

    it("Should let an upload in flight be abandoned", async () => {
        const defaultPut = mockAxiosInstance.put.getMockImplementation();
        // Hold the part upload open so the cancel lands mid-flight.
        mockAxiosInstance.put.mockImplementation(
            () => new Promise(() => undefined)
        );
        try {
            await renderLocalFileUpload({
                ...defaultProps,
                setInProgress,
                fileConnectors: [connectorId]
            });
            await simulateFileDrop(
                new File(["sample"], "sample.csv", { type: "text/csv" })
            );
            await waitFor(() =>
                expect(setInProgress).toHaveBeenCalledWith(true)
            );

            await userEvent.click(
                await screen.findByRole("button", { name: "Remove sample.csv" })
            );

            await waitFor(() =>
                expect(setInProgress).toHaveBeenCalledWith(false)
            );
            // The picker is back and nothing was recorded as uploaded. The
            // dropzone renders alongside the file card, so assert the card
            // itself is gone: one click has to clear it, not hand state back
            // to the upload and resurface the file just removed.
            expect(
                screen.getByTestId("local-upload-dropzone")
            ).toBeInTheDocument();
            expect(screen.queryByText("sample.csv")).not.toBeInTheDocument();
            expect(setUploadSuccess).not.toHaveBeenCalledWith(true);
        } finally {
            if (defaultPut) {
                mockAxiosInstance.put.mockImplementation(defaultPut);
            }
        }
    });

    it("Should ignore a completion that lands after cancelling", async () => {
        const defaultPost = mockAxiosInstance.post.getMockImplementation();
        if (!defaultPost) throw new Error("Expected the default POST mock");
        let finishComplete: (() => void) | undefined;
        mockAxiosInstance.post.mockImplementation((url, ...args) => {
            if (
                typeof url === "string" &&
                url.includes("connectors/uploads/complete")
            ) {
                return new Promise<never>((resolve) => {
                    finishComplete = () =>
                        resolve({ data: {}, status: 200 } as never);
                });
            }
            return defaultPost(url, ...args);
        });
        try {
            await renderLocalFileUpload({
                ...defaultProps,
                setInProgress,
                fileConnectors: [connectorId]
            });
            await simulateFileDrop(
                new File(["sample"], "sample.csv", { type: "text/csv" })
            );
            await waitFor(() => expect(finishComplete).toBeDefined());

            await userEvent.click(
                await screen.findByRole("button", { name: "Remove sample.csv" })
            );
            // The request was already in flight, so it still resolves.
            await act(async () => finishComplete?.());

            // The abandoned file must not be adopted by the form.
            expect(setUploadSuccess).not.toHaveBeenCalledWith(true);
            expect(setTableArgs).not.toHaveBeenCalled();
        } finally {
            mockAxiosInstance.post.mockImplementation(defaultPost);
        }
    });

    it("Should create num_parts for files larger than 100MB", async () => {
        await renderLocalFileUpload({
            ...defaultProps,
            fileConnectors: [connectorId]
        });

        const file = new File(["sample"], "sample.csv", {
            type: "text/csv"
        });
        Object.defineProperty(file, "size", { value: 1024 * 1024 * 500 });
        await simulateFileDrop(file);

        expect(await screen.findByText("sample.csv")).toBeInTheDocument();

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/projects/${projectId}/connectors/uploads/start`,
                {
                    file_type: "CSV",
                    num_parts: 5,
                    source_table_name: "sample"
                },
                {
                    "headers": {
                        "Content-Type": "application/json"
                    }
                }
            );
        });
    });

    it("Should expose the uploaded source only after multipart completion", async () => {
        const defaultPost = mockAxiosInstance.post.getMockImplementation();
        if (!defaultPost) throw new Error("Expected the default POST mock");
        const pendingTableArgs = {
            ...tableData,
            source_table: {
                ...tableData.source_table,
                source: { mode: "relative", path: "" }
            }
        } as KumoapiTableTableDefinition;
        let completeUpload: (() => void) | undefined;
        mockAxiosInstance.post.mockImplementation((url, ...args) => {
            if (
                typeof url === "string" &&
                url.includes("connectors/uploads/complete")
            ) {
                return new Promise<never>((resolve) => {
                    completeUpload = () =>
                        resolve({ data: {}, status: 200 } as never);
                });
            }
            return defaultPost(url, ...args);
        });

        await renderLocalFileUpload({
            ...defaultProps,
            fileConnectors: [connectorId],
            tableArgs: pendingTableArgs
        });
        await simulateFileDrop(
            new File(["sample"], "sample.csv", { type: "text/csv" })
        );

        await waitFor(() => expect(completeUpload).toBeDefined());
        expect(setTableArgs).not.toHaveBeenCalled();
        expect(pendingTableArgs.source_table).toEqual(
            expect.objectContaining({
                source: { mode: "relative", path: "" }
            })
        );

        await act(async () => completeUpload?.());
        await waitFor(() =>
            expect(setTableArgs).toHaveBeenCalledWith(
                expect.objectContaining({
                    source_table: expect.objectContaining({
                        source: { mode: "relative", path: "sample" }
                    })
                })
            )
        );
    });

    it("Should show error message for unsupported file formats", async () => {
        await renderLocalFileUpload(defaultProps);

        const file = new File(["oops"], "sample.exe", {
            type: "application/x-msdownload"
        });
        await simulateFileDrop(file);
        expect(
            await screen.findByText(
                /Please select '.csv' or '.parquet' files only/i
            )
        ).toBeInTheDocument();
    });

    it.each([
        {
            fileName: "sample.parquet",
            fileType: "text/parquet",
            fileSize: 10 * 1000 * 1000,
            description: "less than 10MB"
        },
        {
            fileName: "sample.csv",
            fileType: "text/csv",
            fileSize: 10 * 1000 * 1000,
            description: "less than 10MB"
        },
        {
            fileName: "sample.parquet",
            fileType: "text/parquet",
            fileSize: 500 * 1000 * 1000,
            description: "between 100MB and 1GB"
        },
        {
            fileName: "sample.csv",
            fileType: "text/csv",
            fileSize: 500 * 1000 * 1000,
            description: "between 100MB and 1GB"
        },
        {
            fileName: "sample.parquet",
            fileType: "text/parquet",
            fileSize: 100 * 1024 * 1024 * 1024,
            description: "exactly 100GiB"
        },
        {
            fileName: "sample.csv",
            fileType: "text/csv",
            fileSize: 100 * 1024 * 1024 * 1024,
            description: "exactly 100GiB"
        }
    ])(
        "Should allow uploading $fileName file which is $description",
        async ({ fileName, fileType, fileSize }) => {
            await renderLocalFileUpload(defaultProps);

            const file = new File(["sample"], fileName, { type: fileType });
            Object.defineProperty(file, "size", { value: fileSize });
            await simulateFileDrop(file);

            expect(await screen.findByText(fileName)).toBeInTheDocument();
            // No Upload button: choosing the file is what starts the upload.
            expect(
                screen.queryByRole("button", { name: "Upload" })
            ).not.toBeInTheDocument();
        }
    );

    it.each([
        { fileName: "large.parquet", fileType: "text/parquet" },
        { fileName: "large.csv", fileType: "text/csv" }
    ])(
        "Should show an error message for $fileName files larger than 100GiB",
        async ({ fileName, fileType }) => {
            await renderLocalFileUpload(defaultProps);

            const file = new File(["sample"], fileName, { type: fileType });
            Object.defineProperty(file, "size", {
                value: 100 * 1024 * 1024 * 1024 + 1
            });
            await simulateFileDrop(file);
            await waitFor(() => {
                expect(
                    screen.getByText(
                        /File size is too large. Kumo supports a maximum file size of 100 GiB/i
                    )
                ).toBeInTheDocument();
            });
        }
    );

    it("Should show the uploaded file and Remove once the upload succeeded", async () => {
        await renderLocalFileUpload({
            ...defaultProps,
            fileConnectors: [connectorId],
            selectedSourceTableName: "test_table",
            uploadSuccess: true
        });

        expect(screen.getByText("test_table")).toBeInTheDocument();
        expect(
            await screen.findByRole("button", { name: "Remove" })
        ).toBeInTheDocument();
    });

    it.each([
        { flow: FlowType.EDIT, label: "edit" },
        { flow: FlowType.DUPLICATE, label: "duplicate" }
    ])(
        "Should show the existing file with no remove in $label",
        async ({ flow }) => {
            await renderLocalFileUpload({
                ...defaultProps,
                flowType: flow,
                fileConnectors: [connectorId],
                selectedSourceTableName: "existing_file",
                uploadSuccess: false
            });

            expect(screen.getByText("existing_file")).toBeInTheDocument();
            // The file backs a table that already exists, so it is shown but
            // not detachable from here.
            expect(
                screen.queryByRole("button", { name: "Remove" })
            ).not.toBeInTheDocument();
            expect(
                screen.queryByTestId("local-upload-dropzone")
            ).not.toBeInTheDocument();
        }
    );

    it("Should clear file on remove button click", async () => {
        const connectorId = "project_upload_connector";
        const tableData = {
            source_table: {
                connector_id: connectorId,
                file_type: "CSV",
                data_source_type: "S3",
                source: { mode: "relative", path: "test_table" }
            },
            cols: [],
            pkey: ""
        } as KumoapiTableTableDefinition;

        await renderLocalFileUpload({
            ...defaultProps,
            tableArgs: tableData,
            fileConnectors: [connectorId],
            connectorId,
            selectedSourceTableName: "test_table",
            uploadSuccess: true
        });

        const removeButton = await screen.findByRole("button", {
            name: "Remove"
        });
        await userEvent.click(removeButton);

        await waitFor(() => {
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                `v0/projects/${projectId}/connectors/uploads/delete`,
                {
                    file_type: "CSV",
                    source_table_name: "test_table"
                },
                {
                    "headers": {
                        "Content-Type": "application/json"
                    }
                }
            );
        });

        await waitFor(() => {
            expect(setUploadSuccess).toHaveBeenCalledWith(false);
            expect(setTableName).toHaveBeenCalledWith("");
            expect(setTableArgs).toHaveBeenCalledWith(
                expect.objectContaining({
                    cols: [],
                    pkey: "",
                    end_time_col: "",
                    time_col: "",
                    source_table: expect.objectContaining({
                        source: { mode: "relative", path: "" }
                    })
                })
            );
        });
    });

    it("Should drop the source-tables cache after the file is removed", async () => {
        const connectorId = "project_upload_connector";
        const cacheKey = [ReactQueryKeys.SourceTable, projectId, connectorId];
        // Seed the connector's cached source-table list, as a prior fetch would.
        queryClient.setQueryData(cacheKey, {
            data: {
                response_token: "",
                source_tables: [{ name: "test_table" }]
            }
        });

        await renderLocalFileUpload({
            ...defaultProps,
            tableArgs: {
                source_table: {
                    connector_id: connectorId,
                    file_type: "CSV",
                    data_source_type: "S3",
                    source: { mode: "relative", path: "test_table" }
                },
                cols: [],
                pkey: ""
            } as KumoapiTableTableDefinition,
            fileConnectors: [connectorId],
            connectorId,
            selectedSourceTableName: "test_table",
            uploadSuccess: true
        });

        await userEvent.click(
            await screen.findByRole("button", { name: "Remove" })
        );

        // Once the delete succeeds, the cache is invalidated so the dropdown
        // re-fetches without the removed file.
        await waitFor(() => {
            expect(queryClient.getQueryData(cacheKey)).toBeUndefined();
        });
    });

    it("Should display error message when `Upload` fails", async () => {
        const setInProgress = vi.fn();
        await renderLocalFileUpload({
            ...defaultProps,
            setInProgress,
            fileConnectors: [connectorId],
            connectorId: connectorId
        });

        const file = new File(["sample"], "sample.csv", { type: "text/csv" });
        // The failing PUT has to be in place before the file is chosen, since
        // that is now what kicks the upload off.
        const defaultPut = mockAxiosInstance.put.getMockImplementation();
        mockAxiosInstance.put.mockImplementation((url, payload, options) => {
            if (typeof url === "string" && url.startsWith("http")) {
                options?.onUploadProgress?.({ loaded: 1, total: 1 });
                return new Promise((_resolve, reject) => {
                    setTimeout(() => {
                        reject(new Error("Upload failed"));
                    }, 25);
                });
            }
            if (defaultPut) {
                return defaultPut(url, payload, options);
            }
            return Promise.reject(new Error(`UnSupported request: ${url}`));
        });

        try {
            await simulateFileDrop(file);

            await waitFor(() =>
                expect(setInProgress).toHaveBeenCalledWith(true)
            );

            await waitFor(
                () => {
                    expect(setInProgress).toHaveBeenCalledWith(false);
                },
                { timeout: 2000 }
            );
            await waitFor(() => {
                expect(
                    screen.getByText(
                        /Could not upload the file! Please contact Kumo support for help./i
                    )
                ).toBeInTheDocument();
            });
        } finally {
            if (defaultPut) {
                mockAxiosInstance.put.mockImplementation(defaultPut);
            }
        }
    });

    it("Should display an alert when `startFileUpload` API returns an error", async () => {
        setStartFileUploadTestType(testResponseType.ERROR);
        await renderLocalFileUpload({
            ...defaultProps,
            setInProgress,

            fileConnectors: [connectorId],
            connectorId
        });

        const file = new File(["sample"], "sample.csv", { type: "text/csv" });
        await simulateFileDrop(file);

        await waitFor(() => expect(setInProgress).toHaveBeenCalledWith(false));

        setStartFileUploadTestType(testResponseType.ALL);
    });

    describe("transformFirstChunk - column name handling", () => {
        it("Should detect and flag special characters in column names", async () => {
            const setColumnNamesAdjusted = vi.fn();
            await renderLocalFileUpload({
                ...defaultProps,
                fileConnectors: [connectorId],
                setColumnNamesAdjusted
            });

            const csvContent =
                "column with spaces,column@with#special,normal_column\nrow1,row2,row3";
            const file = new File([csvContent], "test.csv", {
                type: "text/csv"
            });
            await simulateFileDrop(file);

            await waitFor(() => {
                expect(setColumnNamesAdjusted).toHaveBeenCalledWith(true);
            });
        });

        it("Should handle empty column names in CSV headers", async () => {
            await renderLocalFileUpload({
                ...defaultProps,
                fileConnectors: [connectorId]
            });

            const csvContent =
                "column1,,column3,  ,column5\nrow1,row2,row3,row4,row5";
            const file = new File([csvContent], "test.csv", {
                type: "text/csv"
            });
            await simulateFileDrop(file);
        });
    });
});
