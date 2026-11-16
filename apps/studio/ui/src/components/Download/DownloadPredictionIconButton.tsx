// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";

import {
    Alert,
    AlertDescription,
    Button,
    Dialog,
    DialogContent,
    Download,
    TextLink,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { jobStatus, SUPPORT_CONTACT_MESSAGE } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";
import { isValidTimestamp, parseDate } from "utils/timeConverter";

/**
 * Customized component for Download Prediction Modal
 * @param bpStatus BP Job status
 * @param jobCreatedTime Job creation time.
 * @param s3Url s3 URL string
 * @param urlFetchError True/False - if there is a url fetch error
 * @returns `<DownloadPrediction/ >`
 * */
export interface DownloadPredictionProps {
    bpStatus?: string;
    jobCreatedTime?: string;
    s3Url: string;
    urlFetchError: boolean;
}

const DownloadPredictionIconButton: React.FC<DownloadPredictionProps> = ({
    bpStatus,
    jobCreatedTime,
    s3Url,
    urlFetchError
}) => {
    const [showModal, setShowModal] = useState<boolean>(false);
    const TADVIEWER_LINK = "https://www.tadviewer.com/";

    /**
     *  This method checks the time elapsed from the date when batch prediction
     *  is created. If created date is more than 30 days from the current date
     *  this method returns false otherwise returns true.
     */
    const isAvailableForDownload = () => {
        let retValue = true;
        if (isValidTimestamp(jobCreatedTime)) {
            const currentTime = new Date();
            const createdTime = new Date(jobCreatedTime as string);
            const dayFactor = 60 * 60 * 24 * 1000;

            const daysElapsed = Math.ceil(
                (currentTime?.getTime() - createdTime?.getTime()) / dayFactor
            );

            if (daysElapsed > 30) {
                retValue = false;
            }
        }
        return retValue;
    };

    const onDownloadClick = () => {
        if (
            isAvailableForDownload() &&
            isValid(s3Url) &&
            s3Url?.startsWith("http")
        ) {
            window.location.href = s3Url;
            setShowModal(true);
        }
    };

    let date;
    if (isValidTimestamp(jobCreatedTime)) {
        date = new Date(jobCreatedTime as string);
        date.setDate(date?.getDate() + 30);
        date = parseDate(date?.toISOString())?.split(" ")[0];
    }

    return (
        <>
            <Tooltip id={"download-bp"}>
                <TooltipTrigger className="flex items-center">
                    <span>
                        <Button
                            // id="download-bp"
                            data-testid="download-bp"
                            aria-label="Download predictions"
                            variant="link"
                            size="sm"
                            disabled={
                                bpStatus !== jobStatus.Done ||
                                isAvailableForDownload() === false ||
                                urlFetchError ||
                                !isValid(s3Url) ||
                                !s3Url?.startsWith("http")
                            }
                            onClick={onDownloadClick}>
                            <Download className="size-4.5" />
                        </Button>
                    </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" sideOffset={5} arrow={false}>
                    {bpStatus === jobStatus.Running
                        ? "The Batch Prediction job is still running. \n" +
                          "Downloading predictions is unavailable \n" +
                          "until the job is complete."
                        : bpStatus === jobStatus.Failed
                          ? "This Batch Prediction job has failed. \n" +
                            "Please try to run a new job to see predictions."
                          : bpStatus === jobStatus.Cancelled
                            ? "This Batch Prediction job was cancelled. \n" +
                              "Please try to run a new job to see predictions."
                            : urlFetchError
                              ? "There was an error in generating sample \n" +
                                "predictions." +
                                SUPPORT_CONTACT_MESSAGE +
                                " for assistance."
                              : bpStatus === jobStatus.Done &&
                                  isAvailableForDownload() &&
                                  s3Url?.startsWith("http")
                                ? "These predictions are only available \n" +
                                  "until " +
                                  date +
                                  "."
                                : !isValid(s3Url)
                                  ? "Fetching prediction url."
                                  : "These predictions are no longer available for download."}
                </TooltipContent>
            </Tooltip>

            <Dialog open={showModal}>
                <DialogContent>
                    <div className="mt-6 text-lg flex justify-center items-center break-words whitespace-pre-line modal-header border-0 mb-2">
                        <b>{"Download Information"}</b>
                    </div>
                    <div className="break-words whitespace-pre-line mx-4 mb-2 overflow-auto">
                        <Alert variant="default" enableClose={false}>
                            <AlertDescription className="text-base">
                                We currently support downloads in Parquet file
                                format only. To view or transform your Parquet
                                file into CSV, we recommend{" "}
                                <TextLink
                                    id={"tadviewer-link"}
                                    title={"TadViewer."}
                                    newTab={true}
                                    pageLink={TADVIEWER_LINK}
                                />
                            </AlertDescription>
                        </Alert>
                    </div>

                    <div className="pb-4 px-4 flex justify-end w-full">
                        <Button
                            id={"bp-download-dismiss"}
                            title={"Dismiss"}
                            onClick={() => setShowModal(false)}>
                            {"Dismiss"}
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>
        </>
    );
};

export default WithErrorBoundary(DownloadPredictionIconButton);
