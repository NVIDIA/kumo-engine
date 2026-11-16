// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { jobStatus } from "utils/Constants";

import RelatedJobsCard, { RelatedJobsCardProps } from "../RelatedJobsCard";

const defaultProps: RelatedJobsCardProps = {
    jobName: "Sample Job",
    jobId: "job-1234",
    loadingStatus: false,
    status: jobStatus.Running,
    children: <div data-testid="child-content">Related job content</div>
};

describe("RelatedJobsCard", () => {
    const renderRelatedJobsCard = async (props: RelatedJobsCardProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <RelatedJobsCard {...props} />
                </MemoryRouter>
            );
        });
    };

    it("Should render job name, job id, and status badge", async () => {
        await renderRelatedJobsCard(defaultProps);

        expect(screen.getByText("Sample Job")).toBeInTheDocument();
        expect(screen.getByText("job-1234")).toBeInTheDocument();
        expect(screen.getByTestId("header-status-chip-id")).toHaveTextContent(
            "Running"
        );
    });

    it("Should render skeletons when loading", async () => {
        await renderRelatedJobsCard({
            ...defaultProps,
            loadingStatus: true,
            jobName: ""
        });

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should expand and show children content on click", async () => {
        await renderRelatedJobsCard(defaultProps);

        const trigger = screen.getByTestId("acc-trig-related-jobs-job-1234");
        await userEvent.click(trigger);

        const content = await screen.findByTestId(
            "acc-content-related-jobs-job-1234"
        );
        expect(content).toBeVisible();

        expect(screen.getByTestId("child-content")).toHaveTextContent(
            "Related job content"
        );
    });

    it("Should not show badge if status is undefined", async () => {
        await renderRelatedJobsCard({ ...defaultProps, status: undefined });

        expect(
            screen.queryByTestId("header-status-chip-id")
        ).not.toBeInTheDocument();
    });
});
