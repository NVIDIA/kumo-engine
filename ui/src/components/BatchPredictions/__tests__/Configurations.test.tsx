// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";

import Configurations, {
    ConfigurationsProps
} from "../Configurations/Configurations";

const { mockUseBatchPredictionDownloadBp } = vi.hoisted(() => ({
    mockUseBatchPredictionDownloadBp: vi.fn()
}));

vi.mock("hooks/jobs/prediction/prediction", () => ({
    useBatchPredictionDownloadBp: mockUseBatchPredictionDownloadBp
}));

const mockPredictionInfo: UnlinkedBatchPredictionJobData = {
    id: "test-job-id",
    dateCreated: "2025-01-01T00:00:00Z",
    user: "test-user",
    taskType: "CLASSIFICATION",
    outputs: [
        {
            artifact_type: "PREDICTIONS",
            storage_type: "S3",
            file_path: "s3://bucket/path/predictions.csv",
            connector_id: "test-connector-123"
        }
    ],
    execution: {
        results: {
            output_artifact_stats: [
                {
                    artifact_type: "PREDICTIONS",
                    num_rows: 1000,
                    num_cols: 5,
                    size_bytes: 50000,
                    output_path: "s3://bucket/path/predictions.csv"
                }
            ]
        }
    },
    classes: 3,
    numParallelWorkersLimit: 4
} as unknown as UnlinkedBatchPredictionJobData;

const mockHandleError = vi.fn();

const defaultProps: ConfigurationsProps = {
    anchorTime: "2025-01-01T00:00:00Z",
    predictionInfo: mockPredictionInfo,
    handleError: mockHandleError
};

describe("Configurations", () => {
    const queryClient = createQueryClient();

    const renderConfigurations = async (props: ConfigurationsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <Configurations {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockUseBatchPredictionDownloadBp.mockReturnValue({
            s3PresignedUrl: "",
            isError: false
        });
    });

    it("renders local stats without requesting an unsupported download", async () => {
        const localPredictionInfo = {
            ...mockPredictionInfo,
            outputs: [],
            downloadSupported: false,
            execution: {
                status: "DONE",
                results: mockPredictionInfo.execution?.results
            }
        } as unknown as UnlinkedBatchPredictionJobData;

        await renderConfigurations({
            ...defaultProps,
            predictionInfo: localPredictionInfo
        });

        expect(screen.getByText("Local")).toBeInTheDocument();
        expect(screen.getByText("1.0K rows X 5 columns")).toBeInTheDocument();
        expect(screen.queryByTestId("download-bp")).not.toBeInTheDocument();
        expect(mockUseBatchPredictionDownloadBp).toHaveBeenCalledWith(
            "test-job-id",
            mockHandleError,
            "DONE",
            false
        );
    });

    it("enables managed downloads when the dataplane supports signed URLs", async () => {
        const downloadablePredictionInfo = {
            ...mockPredictionInfo,
            outputs: [],
            downloadSupported: true,
            execution: {
                status: "DONE",
                results: mockPredictionInfo.execution?.results
            }
        } as unknown as UnlinkedBatchPredictionJobData;

        await renderConfigurations({
            ...defaultProps,
            predictionInfo: downloadablePredictionInfo
        });

        expect(screen.getByTestId("download-bp")).toBeInTheDocument();
        expect(mockUseBatchPredictionDownloadBp).toHaveBeenCalledWith(
            "test-job-id",
            mockHandleError,
            "DONE",
            true
        );
    });

    it("Should render PropertiesOverviewCard with anchorTime and Data table", async () => {
        await renderConfigurations(defaultProps);

        expect(screen.getByText("Destination")).toBeInTheDocument();
        expect(screen.getByText("S3")).toBeInTheDocument();

        expect(screen.getByText("Anchor time")).toBeInTheDocument();
        expect(screen.getByText(defaultProps.anchorTime!)).toBeInTheDocument();

        expect(screen.getByText("Explanations")).toBeInTheDocument();
        expect(screen.getByText("Off")).toBeInTheDocument();

        expect(screen.getByText("Parallel workers")).toBeInTheDocument();
        expect(
            screen.getByText(
                `${mockPredictionInfo.numParallelWorkersLimit as number}`
            )
        ).toBeInTheDocument();

        expect(screen.getByText("Connector")).toBeInTheDocument();
        expect(screen.getByText("test-connector-123")).toBeInTheDocument();

        const rowsOriginal = screen.getAllByRole("row");
        expect(rowsOriginal.length).greaterThanOrEqual(1);
    });

    it("should not render DataTable when no outputs are available", async () => {
        const emptyPredictionInfo = {
            ...mockPredictionInfo,
            outputs: [],
            execution: {
                results: {
                    output_artifact_stats: []
                }
            }
        } as unknown as UnlinkedBatchPredictionJobData;

        await renderConfigurations({
            ...defaultProps,
            predictionInfo: emptyPredictionInfo
        });

        expect(screen.queryByRole("row")).not.toBeInTheDocument();
    });

    it("should handle null predictionInfo", async () => {
        await renderConfigurations({
            ...defaultProps,
            predictionInfo: null
        });

        expect(screen.queryByRole("row")).not.toBeInTheDocument();
    });

    it("should include max_categories for PREDICTIONS artifact type", async () => {
        const predictionInfoWithClasses = {
            ...mockPredictionInfo,
            classes: 5,
            outputs: [
                {
                    artifact_type: "PREDICTIONS",
                    storage_type: "S3",
                    file_path: "s3://bucket/predictions.csv"
                }
            ]
        } as unknown as UnlinkedBatchPredictionJobData;

        await renderConfigurations({
            ...defaultProps,
            predictionInfo: predictionInfoWithClasses
        });

        const rowsOriginal = screen.getAllByRole("row");
        expect(rowsOriginal.length).equal(2);
        const cells = rowsOriginal[1].querySelectorAll("td");
        const secondRowThirdCell = cells[2];
        expect(secondRowThirdCell.textContent).toContain("5");
    });

    it("should not include max_categories for non-PREDICTIONS artifact type", async () => {
        const predictionInfoWithEmbeddings = {
            ...mockPredictionInfo,
            classes: 5,
            outputs: [
                {
                    artifact_type: "EMBEDDINGS",
                    storage_type: "S3",
                    file_path: "s3://bucket/embeddings.csv"
                }
            ],
            execution: {
                results: {
                    output_artifact_stats: [
                        {
                            artifact_type: "EMBEDDINGS",
                            num_rows: 1000,
                            num_cols: 128,
                            size_bytes: 50000,
                            output_path: "s3://bucket/embeddings.csv"
                        }
                    ]
                }
            }
        } as unknown as UnlinkedBatchPredictionJobData;

        await renderConfigurations({
            ...defaultProps,
            predictionInfo: predictionInfoWithEmbeddings
        });

        const rowsOriginal = screen.getAllByRole("row");
        expect(rowsOriginal.length).equal(2);
        const cells = rowsOriginal[1].querySelectorAll("td");
        const secondRowThirdCell = cells[2];
        expect(secondRowThirdCell.textContent).toContain("");
    });

    it("should handle table_name for warehouse storage types", async () => {
        const tableName = "predictions_table";
        const predictionInfoWithTableName = {
            ...mockPredictionInfo,
            outputs: [
                {
                    artifact_type: "PREDICTIONS",
                    storage_type: "SNOWFLAKE",
                    table_name: tableName
                }
            ]
        } as unknown as UnlinkedBatchPredictionJobData;

        await renderConfigurations({
            ...defaultProps,
            predictionInfo: predictionInfoWithTableName
        });

        expect(screen.getByText("Table Name")).toBeInTheDocument();
        expect(screen.getByText(tableName)).toBeInTheDocument();
    });

    it.each(["GCS", "ADLS"] as const)(
        "should render %s file output path and type",
        async (storageType) => {
            const filePath = "exports/predictions.parquet";
            await renderConfigurations({
                ...defaultProps,
                predictionInfo: {
                    ...mockPredictionInfo,
                    outputs: [
                        {
                            artifact_type: "PREDICTIONS",
                            storage_type: storageType,
                            connector_id: "connector-1",
                            file_path: filePath
                        }
                    ]
                } as UnlinkedBatchPredictionJobData
            });

            expect(screen.getByText("Output Path")).toBeInTheDocument();
            expect(screen.getByText(filePath)).toBeInTheDocument();
            expect(screen.getByText("Parquet")).toBeInTheDocument();
        }
    );
});
