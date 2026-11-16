// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { jobStatus } from "utils/Constants";

import DownloadPredictionIconButton, {
    DownloadPredictionProps
} from "../DownloadPredictionIconButton";

describe("DownloadPredictionIconButton", () => {
    const defaultProps: DownloadPredictionProps = {
        bpStatus: jobStatus.Done,
        jobCreatedTime: new Date().toISOString(),
        s3Url: "https://example.com/sample.parquet",
        urlFetchError: false
    };

    const renderDownloadPredictionIconButton = async (
        props: DownloadPredictionProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <DownloadPredictionIconButton {...props} />
                </MemoryRouter>
            );
        });
    };

    const getDownloadButton = () => screen.getByTestId("download-bp");

    const clickDownloadButtonAndVerifyModal = async () => {
        const button = getDownloadButton();
        await userEvent.click(button);
        expect(
            await screen.findByText("Download Information")
        ).toBeInTheDocument();
    };

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("Should render `DownloadPredictionIconButton` when all the props are set", async () => {
        await renderDownloadPredictionIconButton(defaultProps);

        const button = getDownloadButton();
        expect(button).toBeInTheDocument();
        expect(button).toBeEnabled();
    });

    it("Should open modal when bpStatus is 'Done' and props are valid", async () => {
        await renderDownloadPredictionIconButton(defaultProps);

        await clickDownloadButtonAndVerifyModal();
        expect(
            screen.getByText(
                /We currently support downloads in Parquet file format only/i
            )
        ).toBeInTheDocument();
        expect(screen.getByText("Dismiss")).toBeInTheDocument();
    });

    it.each([
        {
            scenario: "urlFetchError is true",
            props: { urlFetchError: true }
        },
        {
            scenario: "s3Url is empty",
            props: { s3Url: "" }
        },
        {
            scenario: "s3Url does not start with http",
            props: { s3Url: "invalid-url" }
        },
        {
            scenario: "job is older than 30 days",
            props: {
                jobCreatedTime: new Date(
                    Date.now() - 40 * 24 * 60 * 60 * 1000
                ).toISOString()
            }
        },
        {
            scenario: "bpStatus is Running",
            props: { bpStatus: jobStatus.Running }
        },
        {
            scenario: "bpStatus is Failed",
            props: { bpStatus: jobStatus.Failed }
        },
        {
            scenario: "bpStatus is Cancelled",
            props: { bpStatus: jobStatus.Cancelled }
        }
    ])("Should disable button when $scenario", async ({ props }) => {
        await renderDownloadPredictionIconButton({ ...defaultProps, ...props });

        const button = getDownloadButton();
        expect(button).toBeDisabled();
    });

    it.each([
        {
            scenario: "job is running",
            props: { bpStatus: jobStatus.Running },
            expectedTooltip: /The Batch Prediction job is still running/i
        },
        {
            scenario: "job failed",
            props: { bpStatus: jobStatus.Failed },
            expectedTooltip: /This Batch Prediction job has failed/i
        },
        {
            scenario: "job is cancelled",
            props: { bpStatus: jobStatus.Cancelled },
            expectedTooltip: /This Batch Prediction job was cancelled/i
        }
    ])(
        "Should show correct tooltip text when $scenario",
        async ({ props, expectedTooltip }) => {
            await renderDownloadPredictionIconButton({
                ...defaultProps,
                ...props
            });

            const button = getDownloadButton();
            await userEvent.hover(button);

            await waitFor(() => {
                expect(
                    screen.getAllByText(expectedTooltip).length
                ).toBeGreaterThan(0);
            });
        }
    );

    it("Should display an alert with TadViewer link when download button is clicked", async () => {
        await renderDownloadPredictionIconButton(defaultProps);

        await clickDownloadButtonAndVerifyModal();

        const alert = document.querySelector('[data-slot="alert-description"]');
        expect(alert).toBeInTheDocument();
        expect(alert).toHaveTextContent(
            "We currently support downloads in Parquet file format only. To view or transform your Parquet file into CSV, we recommend"
        );

        const link = screen.getByText("TadViewer.");
        expect(link).toBeInTheDocument();
        expect(link).toHaveAttribute("href", "https://www.tadviewer.com/");
    });

    it("Should redirect to s3Url when download button is clicked", async () => {
        const locationSpy = vi.spyOn(window.location, "href", "set");

        await renderDownloadPredictionIconButton(defaultProps);

        const button = getDownloadButton();
        await userEvent.click(button);

        expect(locationSpy).toHaveBeenCalledWith(defaultProps.s3Url);
        locationSpy.mockRestore();
    });

    it.each([
        {
            scenario: "predictions are older than 30 days",
            props: {
                jobCreatedTime: new Date(
                    Date.now() - 31 * 24 * 60 * 60 * 1000
                ).toISOString()
            },
            expectedTooltip:
                /These predictions are no longer available for download./i
        },
        {
            scenario: "URL is being fetched",
            props: {
                s3Url: "",
                jobCreatedTime: new Date("2023-01-01T00:00:00Z").toISOString()
            },
            expectedTooltip: /Fetching prediction url/i
        },
        {
            scenario: "urlFetchError is true",
            props: { s3Url: "", urlFetchError: true },
            expectedTooltip: /There was an error in generating sample/i
        }
    ])(
        "Should show correct tooltip text when $scenario",
        async ({ props, expectedTooltip }) => {
            await renderDownloadPredictionIconButton({
                ...defaultProps,
                ...props
            });

            const button = getDownloadButton();
            await userEvent.hover(button);

            await waitFor(() => {
                expect(
                    screen.getAllByText(expectedTooltip).length
                ).toBeGreaterThan(0);
            });
        }
    );

    it("Should close modal on dismiss click", async () => {
        await renderDownloadPredictionIconButton(defaultProps);

        await clickDownloadButtonAndVerifyModal();

        const dismiss = await screen.findByRole("button", { name: "Dismiss" });
        await userEvent.click(dismiss);

        expect(
            screen.queryByText("Download Information")
        ).not.toBeInTheDocument();
    });

    it("Should show expiration date in tooltip for valid download", async () => {
        await renderDownloadPredictionIconButton(defaultProps);

        const button = getDownloadButton();
        await userEvent.hover(button);

        await waitFor(() => {
            expect(
                screen.getAllByText(/These predictions are only available/i)
                    .length
            ).toBeGreaterThan(0);
        });
    });
});
