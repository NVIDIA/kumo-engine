// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen, within } from "@testing-library/react";

import { binaryEntityOutput } from "../../__tests__/MockResponses/EntityOutputResponses";
import BinaryClassificationPrediction from "../BinaryClassificationPrediction";

const query = `PREDICT COUNT(FINANCE_TRANS.*, 0, 30,days)=0
FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID`;

const anchorTime = `Tue Dec 1, 1998`;

const defaultProps = {
    entityOutput: binaryEntityOutput,
    isDataLoading: false,
    query,
    anchorTime,
    showGroundTruth: true
};

describe("BinaryClassificationPrediction", () => {
    const renderBinaryClassificationPrediction = async (
        props = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <BinaryClassificationPrediction {...props} />
                </MemoryRouter>
            );
        });
    };

    const getSectionContent = (labelText: string) => {
        const label = screen.getByText(labelText);
        return within(label.closest("div")!);
    };

    it("Should render all sections with correct values", async () => {
        await renderBinaryClassificationPrediction();

        expect(
            getSectionContent("Prediction").getByText("0.00")
        ).toBeInTheDocument();
        expect(
            getSectionContent("Anchor Time").getByText("Tue Dec 1, 1998")
        ).toBeInTheDocument();
        expect(
            getSectionContent("Output").getByText(
                "COUNT(FINANCE_TRANS.*, 0, 30,days)=0"
            )
        ).toBeInTheDocument();
        expect(
            getSectionContent("Ground Truth").getByText("0.00")
        ).toBeInTheDocument();
    });

    it("Should render emdash in Output section when query has no match", async () => {
        const queryMissmatch = `PREDICT COUNT(FINANCE_TRANS.*, 0, 30,days)=0`;
        await renderBinaryClassificationPrediction({
            ...defaultProps,
            query: queryMissmatch
        });

        expect(getSectionContent("Output").getByText("—")).toBeInTheDocument();
    });

    it("Should render skeletons when isDataLoading is true", async () => {
        await renderBinaryClassificationPrediction({
            ...defaultProps,
            isDataLoading: true
        });

        const predictionSection = screen.getByText("Prediction").closest("div");
        expect(
            predictionSection?.querySelector('[data-slot="skeleton"]')
        ).toBeInTheDocument();

        const groundTruthSection = screen
            .getByText("Ground Truth")
            .closest("div");
        expect(
            groundTruthSection?.querySelector('[data-slot="skeleton"]')
        ).toBeInTheDocument();
    });

    it("Should not render Ground Truth section when showGroundTruth is false", async () => {
        await renderBinaryClassificationPrediction({
            ...defaultProps,
            showGroundTruth: false
        });

        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
    });

    it("Should correctly format prediction and target values to 2 decimals", async () => {
        await renderBinaryClassificationPrediction({
            ...defaultProps,
            entityOutput: {
                ...defaultProps.entityOutput,
                prediction: 0.00004,
                target: 0
            }
        });

        expect(
            getSectionContent("Prediction").getByText("0.00")
        ).toBeInTheDocument();
        expect(
            getSectionContent("Ground Truth").getByText("0.00")
        ).toBeInTheDocument();
    });
});
