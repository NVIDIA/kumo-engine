// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    render,
    screen,
    waitFor,
    waitForElementToBeRemoved
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { SDKObjectType } from "types/constants";

import { SDKCodeModal, SDKCodeModalProps } from "../SDKCodeModal";

import { setSDKGraphResponseTestType } from "./MockFunctions/SDKGraphCodeModal.mock";
import { setSDKPredictionResponseTestType } from "./MockFunctions/SDKPredictionCodeModal.mock";
import { setSDKTableResponseTestType } from "./MockFunctions/SDKTableCodeModal.mock";
import { setSDKTrainingResponseTestType } from "./MockFunctions/SDKTrainingCodeModal.mock";

const defaultProps: SDKCodeModalProps = {
    id: "graph-21e1bb6de4aa29f819bc5ee63a9e805e",
    open: true,
    toggleModel: vi.fn(),
    objectTypeToGenScript: SDKObjectType.Graph
};

describe("SDKCodeModal", () => {
    const queryClient = createQueryClient();
    const renderSDKCodeModal = async (props: SDKCodeModalProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <SDKCodeModal {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const verifyCommonModalElements = async () => {
        const dialog = screen.getByTestId("sdk-content");
        expect(dialog).toBeInTheDocument();
        expect(dialog).toHaveTextContent("View SDK Code");

        expect(
            screen.getByRole("button", { name: "Copy" })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Download/i })
        ).toBeInTheDocument();
        expect(screen.getByTestId("close-icon")).toBeInTheDocument();
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it("Should display `Loading` when the id is set as empty", async () => {
        await renderSDKCodeModal({
            ...defaultProps,
            id: ""
        });

        expect(screen.getByText(/Loading/i)).toBeInTheDocument();
    });

    it.each([
        {
            type: SDKObjectType.Graph,
            id: "graph-21e1bb6de4aa29f819bc5ee63a9e805e",
            description: "graph",
            apiEndpoint: "v1/graphs/graph-21e1bb6de4aa29f819bc5ee63a9e805e/sdk",
            codeSnippet: /graph.validate()/i
        },
        {
            type: SDKObjectType.Table,
            id: "NA.NA-d410f0cbbeadba2b0b51e9f242cbf7ac",
            description: "table",
            apiEndpoint: "v1/tables/NA.NA-d410f0cbbeadba2b0b51e9f242cbf7ac/sdk",
            codeSnippet: /end_time_column=clip_clothing_embs_config\[/i
        },
        {
            type: SDKObjectType.TrainingJob,
            id: "trainingjob-439dc6723bf34ec6bac09b9bd5b0bd85",
            description: "training job workflow",
            apiEndpoint:
                "v0/training_jobs/trainingjob-439dc6723bf34ec6bac09b9bd5b0bd85/sdk",
            codeSnippet: /train_table=train_table,/i
        },
        {
            type: SDKObjectType.PredictionJob,
            id: "bp-job-c5ec67d2af234dd5859efc3f5314a03b",
            description: "prediction job workflow",
            apiEndpoint:
                "v1/batch_prediction_jobs/bp-job-c5ec67d2af234dd5859efc3f5314a03b/sdk",
            codeSnippet: /num_workers=batch_prediction_options\[/i
        }
    ])(
        "Should render the SDKCodeModal with SDKObjectType as $description",
        async ({ type, id, description, apiEndpoint }) => {
            await renderSDKCodeModal({
                ...defaultProps,
                id,
                objectTypeToGenScript: type
            });

            await verifyCommonModalElements();
            expect(
                screen.getByText(
                    new RegExp(
                        `Here is the generated code to reproduce this ${description} in the SDK.`,
                        "i"
                    )
                )
            ).toBeInTheDocument();

            await waitFor(() => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    apiEndpoint,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });

            expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
            const editor = screen.getByRole("textbox");
            expect(editor.textContent).toMatch(/from kumoai_studio/i);
            expect(editor.textContent).toContain(
                'project="project_0123456789abcdef0123456789abcdef"'
            );
            expect(editor.textContent).toMatch(/Connector\.get_by_name\(/);
            expect(editor.textContent).not.toContain("root_dir=");
            expect(editor.textContent).not.toBe("");
        }
    );

    it("Should show 'Copied' after clicking Copy button", async () => {
        await renderSDKCodeModal(defaultProps);

        const copyButton = await screen.findByRole("button", { name: /copy/i });
        await userEvent.click(copyButton);

        const copiedText = await screen.findByText(/copied/i);
        expect(copiedText).toBeInTheDocument();

        await waitForElementToBeRemoved(() => screen.queryByText(/copied/i), {
            timeout: 3000
        });
    });

    it("should show 'Downloaded' briefly after clicking Download button", async () => {
        await renderSDKCodeModal(defaultProps);

        const downloadButton = await screen.findByRole("button", {
            name: /download/i
        });
        await userEvent.click(downloadButton);

        const downloadedText = await screen.findByText(/downloaded/i);
        expect(downloadedText).toBeInTheDocument();

        await waitForElementToBeRemoved(
            () => screen.queryByText(/downloaded/i),
            {
                timeout: 3000
            }
        );
    });

    it("Should trigger toggleModel when Close button is clicked", async () => {
        const toggleModel = vi.fn();
        await renderSDKCodeModal({
            ...defaultProps,
            toggleModel
        });

        const closeButton = screen.getByTestId("close-icon");
        await userEvent.click(closeButton);

        expect(toggleModel).toHaveBeenCalled();
    });

    it("Should not render the SDKCodeModal when the prop `open` is set as false", async () => {
        await renderSDKCodeModal({
            ...defaultProps,
            open: false
        });

        const dialog = screen.queryByTestId("sdk-content");
        expect(dialog).not.toBeInTheDocument();
    });

    it.each([
        {
            type: SDKObjectType.Graph,
            id: "graph-21e1bb6de4aa29f819bc5ee63a9e805e",
            description: "graph",
            apiEndpoint: "v1/graphs/graph-21e1bb6de4aa29f819bc5ee63a9e805e/sdk",
            codeSnippet: /graph.validate()/i,
            mockSetter: setSDKGraphResponseTestType
        },
        {
            type: SDKObjectType.Table,
            id: "NA.NA-d410f0cbbeadba2b0b51e9f242cbf7ac",
            description: "table",
            apiEndpoint: "v1/tables/NA.NA-d410f0cbbeadba2b0b51e9f242cbf7ac/sdk",
            codeSnippet: /end_time_column=clip_clothing_embs_config\[/i,
            mockSetter: setSDKTableResponseTestType
        },
        {
            type: SDKObjectType.TrainingJob,
            id: "trainingjob-439dc6723bf34ec6bac09b9bd5b0bd85",
            description: "training job workflow",
            apiEndpoint:
                "v0/training_jobs/trainingjob-439dc6723bf34ec6bac09b9bd5b0bd85/sdk",
            codeSnippet: /train_table=train_table,/i,
            mockSetter: setSDKTrainingResponseTestType
        },
        {
            type: SDKObjectType.PredictionJob,
            id: "bp-job-c5ec67d2af234dd5859efc3f5314a03b",
            description: "prediction job workflow",
            apiEndpoint:
                "v1/batch_prediction_jobs/bp-job-c5ec67d2af234dd5859efc3f5314a03b/sdk",
            codeSnippet: /num_workers=batch_prediction_options\[/i,
            mockSetter: setSDKPredictionResponseTestType
        }
    ])(
        "Should display empty editor placeholder when the $description SDK code API response is empty",
        async ({
            type,
            id,
            description,
            apiEndpoint,
            codeSnippet,
            mockSetter
        }) => {
            mockSetter(testResponseType.EMPTY);
            await renderSDKCodeModal({
                ...defaultProps,
                id,
                objectTypeToGenScript: type
            });

            await waitFor(() => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    apiEndpoint,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });

            await verifyCommonModalElements();
            expect(
                screen.getByText(
                    new RegExp(
                        `Here is the generated code to reproduce this ${description} in the SDK.`,
                        "i"
                    )
                )
            ).toBeInTheDocument();

            expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
            expect(
                screen.queryByText(/from kumoai_studio.graph import Column/i)
            ).not.toBeInTheDocument();
            expect(screen.queryByText(codeSnippet)).not.toBeInTheDocument();

            const editor = screen.getByRole("textbox");
            expect(editor.textContent?.trim()).toBe(
                "Start typing your predictive query here..."
            );
        }
    );

    it("Should not trigger download when SDK code data is missing", async () => {
        setSDKGraphResponseTestType(testResponseType.EMPTY);
        await renderSDKCodeModal(defaultProps);

        const downloadSpy = vi.spyOn(URL, "createObjectURL");

        const downloadButton = screen.getByRole("button", {
            name: /Download/i
        });
        await userEvent.click(downloadButton);

        expect(downloadSpy).not.toHaveBeenCalled();
        expect(screen.queryByText("Downloaded")).not.toBeInTheDocument();

        downloadSpy.mockRestore();
    });
});
