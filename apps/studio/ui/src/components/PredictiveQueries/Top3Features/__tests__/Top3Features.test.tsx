// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { setCategoryAggrFeatureResponseType } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/CategoryAggrFeature.mock";
import { setTopKFeatureResponseType } from "components/PredictiveQueries/Explanation/__tests__/MockFunctions/TopKFeature.mock";

import { Top3Features, Top3FeaturesProps } from "../Top3Features";

const entityId = "265";
const jobId = "trainingjob-00046000000000000000000000000000";
const category = "random";

const defaultProps: Top3FeaturesProps = {
    entityId,
    jobId,
    category
};

describe("Top3Features", () => {
    const queryClient = createQueryClient();

    const renderTop3Features = async (props: Top3FeaturesProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <Top3Features {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const verifyNoTopFeaturesMessage = () => {
        expect(screen.getByText("No top features found!")).toBeInTheDocument();
    };

    const verifyTopKFeatureApiCall = async (
        expectedJobId: string,
        expectedEntityId: string
    ) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${expectedJobId}/eval/topk_feature/${expectedEntityId}/3`
            );
        });
    };

    const verifyCategoryAggrFeatureApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                expect.stringContaining("category_aggr_feature")
            );
        });
    };

    const verifyTableHeaders = (headers: string[]) => {
        const tableHeader = screen.getByTestId("table-header");
        headers.forEach((header) => {
            expect(tableHeader).toHaveTextContent(header);
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    it("Should display message `No top features found!` when all props are empty", async () => {
        await renderTop3Features({
            entityId: "",
            jobId: "",
            category: ""
        });

        verifyNoTopFeaturesMessage();
    });

    it("Should call `getTopKFeatures` API when jobId and entityId are specified", async () => {
        await renderTop3Features({
            ...defaultProps,
            category: ""
        });

        await verifyTopKFeatureApiCall(jobId, entityId);

        verifyTableHeaders(["Table", "Column", "Value", "Mode"]);
    });

    it("Should show message `No top features found!` when `getTopKFeatures` API throws error", async () => {
        setTopKFeatureResponseType(testResponseType.ERROR);
        await renderTop3Features({
            ...defaultProps,
            category: ""
        });

        await verifyTopKFeatureApiCall(jobId, entityId);

        verifyNoTopFeaturesMessage();
        setTopKFeatureResponseType(testResponseType.ALL);
    });

    it("Should call `getCategoryAggrFeature` for each returned feature", async () => {
        await renderTop3Features(defaultProps);

        await verifyCategoryAggrFeatureApiCall();
    });

    it("Should handle error in `getCategoryAggrFeature` gracefully", async () => {
        setCategoryAggrFeatureResponseType(testResponseType.ERROR);
        await renderTop3Features(defaultProps);

        await verifyCategoryAggrFeatureApiCall();

        expect(screen.getByText("Column")).toBeInTheDocument();
        setCategoryAggrFeatureResponseType(testResponseType.ALL);
    });

    it("Should render with empty data array without crashing", async () => {
        setTopKFeatureResponseType(testResponseType.EMPTY);

        await renderTop3Features(defaultProps);

        await waitFor(() => {
            verifyNoTopFeaturesMessage();
        });

        setTopKFeatureResponseType(testResponseType.ALL);
    });

    it("Should not make API calls when jobId is empty", async () => {
        await renderTop3Features({
            ...defaultProps,
            jobId: ""
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
                expect.stringContaining("category_aggr_feature")
            );
        });
    });

    it("Should handle missing values in formattedData", async () => {
        setTopKFeatureResponseType(testResponseType.PARTIAL);

        await renderTop3Features(defaultProps);

        await waitFor(() => {
            const dashes = screen.getAllByText("—");
            expect(dashes.length).toBeGreaterThanOrEqual(2);
            expect(screen.getByText("33")).toBeInTheDocument();
        });

        setTopKFeatureResponseType(testResponseType.ALL);
    });
});
