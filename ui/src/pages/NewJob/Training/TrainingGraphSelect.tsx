// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";

import {
    Button,
    Description,
    GridCol,
    GridRow,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import GraphSelectList from "components/NewJob/Training/Select/GraphSelectList";

import { useProjectId } from "hooks/project/projects";

import { GraphListResponse } from "types/graphs";

import { getProjectPath } from "utils/Constants";
import { useCustomNavigation } from "utils/navigator";

const TrainingGraphSelect = () => {
    const navigate = useCustomNavigation();
    const projectId = useProjectId();

    const projectPaths = getProjectPath(projectId);
    const [isEmptyList, setIsEmptyList] = useState(false);

    const cancelUrl = `${projectPaths.ModelJobsList}?page=1&job_type=TRAINING_JOB`;

    const onCancelClick = useCallback(() => {
        navigate(cancelUrl, { replace: true });
    }, [cancelUrl, navigate]);

    const onGraphSelect = useCallback(
        (graph: GraphListResponse) => {
            const destination = projectPaths.NewTrainingJob;
            navigate(destination, {
                state: {
                    preselectedGraph: {
                        id: graph.id,
                        name: graph.name
                    }
                }
            });
        },
        [navigate, projectPaths.NewTrainingJob]
    );

    return (
        <>
            <GridRow className="flex justify-between items-center">
                <GridCol xs={12} sm={12} md={6}>
                    <Title
                        variant="page-heading"
                        className="!font-bold text-3xl">
                        New Model
                    </Title>
                    {!isEmptyList && (
                        <Description
                            size="md"
                            className="leading-5 mt-2 text-title-secondary">
                            Select a graph to train the new model
                        </Description>
                    )}
                </GridCol>

                {!isEmptyList && (
                    <GridCol xs={12} sm={12} md={6}>
                        <GridRow
                            xs={12}
                            sm={12}
                            md="auto"
                            className="flex flex-wrap justify-end items-center gap-1">
                            <GridCol xs={12} sm={12} md="auto">
                                <Button
                                    id="cancel-training"
                                    variant="outline"
                                    onClick={onCancelClick}>
                                    Cancel
                                </Button>
                            </GridCol>
                        </GridRow>
                    </GridCol>
                )}
            </GridRow>
            <div className="h-6" />
            <GraphSelectList
                setNext={onGraphSelect}
                projectId={projectId ?? undefined}
                onEmptyChange={setIsEmptyList}
            />
        </>
    );
};

export default WithErrorBoundary(TrainingGraphSelect);
