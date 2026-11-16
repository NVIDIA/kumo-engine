// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { createQueryClient } from "tests/testUtil";

import { DropdownOption, InputState } from "types/inputs";
import { FlowType } from "types/tables";

import LocalUploadFlow, { LocalUploadFlowProps } from "../LocalUploadFlow";

import "tests/mockAxios";

const connectorId = "connector-project-upload";

const tableArgs = {
    source_table: {
        connector_id: connectorId,
        file_type: "CSV",
        data_source_type: "S3",
        source: { mode: "relative", path: "" }
    },
    cols: [],
    pkey: ""
} as KumoapiTableTableDefinition;

const defaultProps: LocalUploadFlowProps = {
    tableNameField: <div>table-name-field</div>,
    sourceTableErrorLabel: <div>source-error-label</div>,
    showFileSelection: true,
    flowType: FlowType.CREATE,
    nameInputState: InputState.Valid,
    onTableSelect: vi.fn(),
    handleError: vi.fn(),
    clearErrors: vi.fn(),
    selectedConnector: { value: connectorId, label: "CSV" } as DropdownOption,
    connectorId,
    sourceTableName: "",
    resetTableSelection: vi.fn(),
    errLabelMsg: { entity: "", errorMsg: "" },
    clearErrLabel: vi.fn(),
    setInProgress: vi.fn(),
    setUploadSuccess: vi.fn(),
    suggestTableName: vi.fn(),
    setTableName: vi.fn(),
    uploadSuccess: false,
    tableArgs,
    setTableArgs: vi.fn(),
    fileConnectors: [connectorId, "connector-project-upload"],
    setColumnNamesAdjusted: vi.fn(),
    sourceTableReloadKey: 0
};

describe("LocalUploadFlow", () => {
    const queryClient = createQueryClient();
    const renderFlow = async (props: Partial<LocalUploadFlowProps> = {}) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <LocalUploadFlow {...defaultProps} {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    it("Should default to the upload drop zone with no file dropdown", async () => {
        await renderFlow({ sourceTableName: "", uploadSuccess: false });

        expect(
            screen.getByRole("radio", { name: "Upload a new file" })
        ).toBeChecked();
        expect(screen.getByTestId("local-upload-dropzone")).toBeInTheDocument();
        expect(
            screen.queryByTestId("testid-select-source-table")
        ).not.toBeInTheDocument();
    });

    it("Should swap the drop zone for the dropdown when choosing an existing file", async () => {
        await renderFlow({ sourceTableName: "", uploadSuccess: false });

        await userEvent.click(
            screen.getByRole("radio", { name: "Choose an existing file" })
        );

        // Exactly one of the two paths is live at a time, so a selection can
        // never sit underneath an upload.
        expect(
            await screen.findByTestId("testid-select-source-table")
        ).toBeInTheDocument();
        expect(
            screen.queryByTestId("local-upload-dropzone")
        ).not.toBeInTheDocument();
    });

    it("Should drop the previous selection when switching file source", async () => {
        const setTableArgs = vi.fn();
        const setUploadSuccess = vi.fn();
        await renderFlow({
            sourceTableName: "picked_file",
            uploadSuccess: false,
            setTableArgs,
            setUploadSuccess
        });

        await userEvent.click(
            screen.getByRole("radio", { name: "Choose an existing file" })
        );

        // The path is cleared locally but the file itself is left on the
        // server — switching modes must not delete anything.
        const nextArgs = setTableArgs.mock
            .calls[0][0] as KumoapiTableTableDefinition;
        expect(
            (nextArgs.source_table as { source: { path: string } }).source.path
        ).toBe("");
        expect(nextArgs.cols).toEqual([]);
        expect(setUploadSuccess).toHaveBeenCalledWith(false);
    });

    it("Should lock the existing-file option while a file is uploaded", async () => {
        await renderFlow({
            sourceTableName: "uploaded_file",
            uploadSuccess: true
        });

        // The uploaded file has to be removed before the other path opens.
        expect(
            screen.getByRole("radio", { name: "Choose an existing file" })
        ).toBeDisabled();
    });

    it("Should keep the uploaded file's chip and Remove visible after upload", async () => {
        await renderFlow({
            sourceTableName: "uploaded_file",
            uploadSuccess: true
        });

        expect(
            screen.getByRole("radio", { name: "Upload a new file" })
        ).toBeChecked();
        expect(
            await screen.findByRole("button", { name: "Remove" })
        ).toBeInTheDocument();
    });
});
