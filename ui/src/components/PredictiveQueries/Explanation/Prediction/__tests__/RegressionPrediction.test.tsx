// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { MemoryRouter } from "react-router";

import { act, render, screen, within } from "@testing-library/react";

import { regressionEntityOutput } from "../../__tests__/MockResponses/EntityOutputResponses";
import RegressionPrediction from "../RegressionPrediction";

const query = `PREDICT COUNT(transaction.Quantity
WHERE transaction.Description STARTS WITH "CARD", 0, 90,days)
FOR EACH customer.CustomerID`;

const anchorTime = "Fri Sep 9, 2011, 1:46:00 PM";

const defaultProps = {
    entityOutput: regressionEntityOutput,
    isDataLoading: false,
    query,
    anchorTime,
    showGroundTruth: true
};

describe("RegressionPrediction", () => {
    const renderRegressionPrediction = async (props = defaultProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <RegressionPrediction {...props} />
                </MemoryRouter>
            );
        });
    };

    const getSectionByLabel = (labelText: string) => {
        const label = screen.getByText(labelText);
        return label.closest("div")!;
    };

    const expectSectionToContain = (
        labelText: string,
        expectedText: string
    ) => {
        const section = getSectionByLabel(labelText);
        expect(within(section).getByText(expectedText)).toBeInTheDocument();
    };

    const expectSectionToHaveSkeleton = (labelText: string) => {
        const section = getSectionByLabel(labelText);
        const skeleton = section.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    };

    it("Should render all sections with correct default values", async () => {
        await renderRegressionPrediction();

        expectSectionToContain("Output", "—");
        expectSectionToContain("Prediction", "137.75");
        expectSectionToContain("Anchor Time", anchorTime);
    });

    it("Should render Skeleton loaders when isDataLoading is true", async () => {
        await renderRegressionPrediction({
            ...defaultProps,
            isDataLoading: true,
            showGroundTruth: true
        });

        expectSectionToHaveSkeleton("Prediction");
        expectSectionToHaveSkeleton("Ground Truth");
    });

    it("Should not render Ground Truth section when showGroundTruth is false", async () => {
        await renderRegressionPrediction({
            ...defaultProps,
            showGroundTruth: false
        });

        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
    });

    it("Should render extracted output from query when there is a match", async () => {
        const matchingQuery = `PREDICT COUNT(transaction.Quantity) FOR EACH customer.CustomerID`;

        await renderRegressionPrediction({
            ...defaultProps,
            query: matchingQuery
        });

        expectSectionToContain("Output", "COUNT(transaction.Quantity)");
    });

    it("Should correctly format prediction and target values to 2 decimals", async () => {
        await renderRegressionPrediction({
            ...defaultProps,
            entityOutput: {
                ...defaultProps.entityOutput,
                prediction: 3.141592,
                target: 2
            },
            showGroundTruth: true
        });

        expect(screen.getByText("3.14")).toBeInTheDocument();
        expect(screen.getByText("2.00")).toBeInTheDocument();
    });
});
