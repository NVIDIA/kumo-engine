// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
    VisuallyHidden
} from "@kumo-ai/kumo-ui-catalog";
import { JobType } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import GraphSnapshotOverview from "components/Jobs/Graph/GraphSnapshotOverview";
import TrainingJob from "components/Jobs/Training/TrainingJob";
import TrainingTableGeneration from "components/Jobs/TrainingTable/TrainingTableGeneration";

import { JobOverviewType } from "types/jobs";

import PredictionJob from "../Prediction/PredictionJob";
import PredictionTableGeneration from "../PredictionTable/PredictionTableGeneration";

export interface JobOverviewProps {
    jobId: string | null;
    jobOverviewType: JobOverviewType | null;
    onClose: (id: string | null, type: JobType | null) => void;
    onArchive?: () => void;
    onRecover?: () => void;
}

const JobOverview = ({
    jobId,
    jobOverviewType,
    onClose,
    onArchive,
    onRecover
}: JobOverviewProps) => {
    const [show, setShow] = useState(jobId ? true : false);

    const onChange = (open: boolean) => {
        setShow(open);
        if (!open) {
            // Small delay to allow sheet animation to complete
            setTimeout(() => {
                onClose(null, null);
            }, 300);
        }
    };

    return (
        <div id="jobs-side-window-view">
            <Sheet open={show} onOpenChange={onChange}>
                <SheetContent className="bg-surface-grey">
                    <SheetHeader>
                        <SheetTitle>
                            <VisuallyHidden>Job Overview</VisuallyHidden>
                        </SheetTitle>
                        <SheetDescription>
                            <VisuallyHidden>
                                Detailed job information and status
                            </VisuallyHidden>
                        </SheetDescription>
                        {jobOverviewType === JobOverviewType.TRAINING_JOB &&
                            jobId && (
                                <TrainingJob
                                    id={jobId}
                                    onClose={() => onChange(false)}
                                    onArchive={onArchive}
                                    onRecover={onRecover}
                                />
                            )}
                        {jobOverviewType ===
                            JobOverviewType.BATCH_PREDICTION_JOB &&
                            jobId && (
                                <PredictionJob
                                    bpId={jobId}
                                    onClose={() => onChange(false)}
                                    onArchive={onArchive}
                                    onRecover={onRecover}
                                />
                            )}
                        {jobOverviewType === JobOverviewType.TRAIN_TABLE_JOB &&
                            jobId && (
                                <TrainingTableGeneration
                                    id={jobId}
                                    onClose={() => onChange(false)}
                                    onArchive={onArchive}
                                    onRecover={onRecover}
                                />
                            )}
                        {jobOverviewType ===
                            JobOverviewType.PREDICTION_TABLE_JOB &&
                            jobId && (
                                <PredictionTableGeneration
                                    id={jobId}
                                    onClose={() => onChange(false)}
                                    onArchive={onArchive}
                                    onRecover={onRecover}
                                />
                            )}
                        {jobOverviewType ===
                            JobOverviewType.GRAPH_SNAPSHOT_JOB &&
                            jobId && <GraphSnapshotOverview id={jobId} />}
                    </SheetHeader>
                </SheetContent>
            </Sheet>
        </div>
    );
};

export default WithErrorBoundary(JobOverview);
