// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import { NavigateFunction, useNavigate } from "react-router";
import Spacer from "react-spacer";

import {
    AddCircle,
    Button,
    Card,
    CardContent,
    DeveloperBoard,
    EmptyEntity,
    EmptyEntityData,
    GridCol,
    GridRow,
    NetworkNode,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { JobType } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useConnectors } from "hooks/connector/connector";
import { useGraphs } from "hooks/graph/graphs";
import { useJobs } from "hooks/jobs/list/jobList";
import { useProjectId } from "hooks/project/projects";
import { useTables } from "hooks/tables/tables";

import { FlowType } from "types/tables";

import {
    docURLs,
    getProjectPath,
    jobStatus,
    newDocsURL,
    PageNames
} from "utils/Constants";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import { isPredictionsJobsPage } from "utils/jobs";

export interface EmptyEntitiesProps {
    currentPage: string;
    title?: string;
}

interface EmptyEntityOptions {
    navigate: NavigateFunction;
    projectId?: string | null;
    projectPaths: ReturnType<typeof getProjectPath>;
    connectorsPresent: boolean;
    isPredictionPage: boolean;
    jobsPresent: boolean;
}

const createEmptyEntities = ({
    navigate,
    projectId,
    projectPaths,
    connectorsPresent,
    isPredictionPage,
    jobsPresent
}: EmptyEntityOptions): Record<string, EmptyEntityData> => {
    const hasProject = Boolean(projectId);
    const docsText = hasProject ? "View docs" : "Quickstart Documentation";
    const emptyLabelClass =
        "flex flex-col items-center gap-2 text-center text-base font-medium text-title-secondary";

    return {
        Connectors: {
            project: hasProject,
            icon: undefined,
            label: <label>Add a data source to begin!</label>,
            renderButtonId: "create-connector",
            renderButtonText: "Connect",
            onRenderClick: () => {
                navigate(projectPaths.NewConnector, {
                    state: { connectorsPresent }
                });
            },
            docsId: "Data_source",
            docsText,
            onDocsClick: () => {
                window.open(newDocsURL + docURLs.QuickStart, "_blank");
            }
        },
        Tables: {
            project: hasProject,
            icon: hasProject ? undefined : <NetworkNode className="size-12" />,
            label: (
                <label>
                    {hasProject
                        ? "Add a table to create graph!"
                        : "Connect a table to create a Kumo Graph"}
                </label>
            ),
            renderButtonId: "create-table",
            renderButtonText: "Add table",
            onRenderClick: () => {
                navigate(projectPaths.NewTable);
            },
            docsId: "Tables_doc",
            docsText,
            onDocsClick: () => {
                window.open(newDocsURL + docURLs.QuickStart, "_blank");
            }
        },
        Graphs: {
            project: hasProject,
            icon: hasProject ? undefined : (
                <DeveloperBoard className="size-12" />
            ),
            label: <label>Connect a graph to create a Training Job</label>,
            renderButtonId: "create-graph",
            renderButtonText: "Add graph",
            onRenderClick: () => {
                navigate(projectPaths.NewGraph, {
                    state: {
                        flow_type: FlowType.CREATE,
                        name: ""
                    }
                });
            },
            docsId: "Graph_doc",
            docsText,
            onDocsClick: () => {
                window.open(newDocsURL + docURLs.QuickStart, "_blank");
            }
        },
        Training: {
            project: hasProject,
            icon: hasProject ? undefined : <NetworkNode className="size-12" />,
            label: hasProject ? (
                <label className={emptyLabelClass}>
                    <span>Welcome! 🎉</span>
                    <span>
                        {isPredictionPage
                            ? "Train your first model to start making predictions."
                            : "Create your first model to get started."}
                    </span>
                </label>
            ) : isPredictionPage && jobsPresent ? (
                <label className="text-center">
                    No predictions yet.
                    <br /> Let's start by making your first prediction!!
                </label>
            ) : (
                <label className="text-center">
                    No trained model yet.
                    <br /> Let's start by training your first model!!
                </label>
            ),
            renderButtonId: "create-training",
            renderButtonText:
                isPredictionPage && jobsPresent
                    ? "Make predictions"
                    : hasProject
                      ? "New model"
                      : "Train a model",
            onRenderClick: () => {
                if (isPredictionPage && jobsPresent) {
                    navigate(projectPaths.NewPredictionJob);
                } else {
                    navigate(projectPaths.NewTrainingGraphSelect);
                }
            },
            docsId: "Training_doc",
            docsText,
            onDocsClick: () => {
                if (isPredictionPage && jobsPresent) {
                    window.open(newDocsURL + docURLs.BatchPrediction, "_blank");
                } else {
                    window.open(newDocsURL + docURLs.QuickStart, "_blank");
                }
            }
        }
    };
};

const EmptyEntities = ({ currentPage, title }: EmptyEntitiesProps) => {
    const navigate = useNavigate();
    const projectId = useProjectId();
    const projectPaths = getProjectPath(projectId);

    const isPredictionPage = isPredictionsJobsPage(
        window?.location?.pathname ?? "",
        window?.location?.search ?? ""
    );

    const {
        isDataLoading: fetchingConnectors,
        isConnectorsPresent: connectorsPresent
    } = useConnectors(
        undefined,
        undefined,
        currentPage !== PageNames.Connector
    );

    const { isDataLoading: fetchingGraphs, isGraphsPresent: graphsPresent } =
        useGraphs({
            enabled:
                currentPage === PageNames.NewModel ||
                currentPage === PageNames.Jobs ||
                currentPage === PageNames.Graph,
            projectId
        });

    const { isDataLoading: fetchingTables, tablesPresent } = useTables(
        undefined,
        undefined,
        currentPage !== PageNames.Table,
        projectId as string
    );

    const [, , fetchingJobs, , jobsPresent] = useJobs(
        {
            "project_id": projectId ?? "",
            "limit": 13,
            "page": 0,
            "time_range_start": undefined,
            "time_range_end": undefined,
            // Only successfully trained jobs are retrieved because
            // predictions require completed training
            "status": [jobStatus.Done],
            "job_type": [JobType.TRAINING_JOB],
            "tags": {},
            "tags_filter": [],
            "search_keyword": "",
            "sort_desc": false
        },
        isPredictionPage
    );

    // Handle navigation logic
    useEffect(() => {
        if (!projectId) {
            return;
        }
        if (currentPage === PageNames.Table && connectorsPresent) {
            navigate(projectPaths.NewTable, {
                state: { tableNotExists: true }
            });
        } else if (currentPage === PageNames.Graph && tablesPresent) {
            navigate(projectPaths.NewGraph, {
                state: {
                    flow_type: FlowType.CREATE,
                    name: "",
                    graphNotExists: true
                }
            });
        } else if (currentPage === PageNames.NewModel && graphsPresent) {
            navigate(projectPaths.NewTrainingGraphSelect);
        }
    }, [
        connectorsPresent,
        tablesPresent,
        graphsPresent,
        currentPage,
        navigate,
        jobsPresent,
        projectPaths,
        projectId
    ]);
    const noEntities = createEmptyEntities({
        navigate,
        projectId,
        projectPaths,
        connectorsPresent,
        isPredictionPage,
        jobsPresent
    });

    const renderHeader = () => {
        if (!title) {
            return null;
        }
        return (
            <GridRow xs={12} sm={12} md={12}>
                <GridCol xs={12} sm={12} md="auto">
                    <Title
                        variant="primary-heading"
                        className="leading-none text-3xl">
                        {capitalizeFirstLetter(title)}
                    </Title>
                </GridCol>
            </GridRow>
        );
    };

    if (
        fetchingConnectors ||
        fetchingGraphs ||
        fetchingTables ||
        fetchingJobs
    ) {
        return <>{renderHeader()}</>;
    }

    const renderProjectEmptyPage = () => {
        const containerClassName =
            "inset-0 absolute flex flex-col items-center gap-2 p-3 text-muted-foreground justify-center font-medium";
        if (currentPage === PageNames.Connector) {
            if (!projectPaths) {
                return (
                    <div className={containerClassName}>
                        <p>No connectors available.</p>
                        <p>Select a project to continue.</p>
                    </div>
                );
            }
            return (
                <div className={containerClassName}>
                    <p>Get started by creating your first connector!</p>
                    <Button
                        id={"create-new-connector"}
                        size="sm"
                        onClick={() => navigate(projectPaths.NewConnector)}>
                        <AddCircle /> Add connector
                    </Button>
                </div>
            );
        }
        if (!projectId) {
            return (
                <div className={containerClassName}>
                    <p>{`No ${!connectorsPresent ? "connectors" : currentPage?.toLowerCase()} available.`}</p>
                    <p>Select a project to continue.</p>
                </div>
            );
        }
        if (!connectorsPresent)
            return (
                <div className={containerClassName}>
                    <p>Get started by creating your first connector!</p>
                    <Button
                        id={"create-new-connector"}
                        size="sm"
                        onClick={() => navigate(projectPaths!.NewConnector)}>
                        <AddCircle /> Add connector
                    </Button>
                </div>
            );

        switch (currentPage) {
            case PageNames.Jobs:
                if (graphsPresent)
                    return <EmptyEntity data={noEntities["Training"]} />;
                if (tablesPresent)
                    return <EmptyEntity data={noEntities["Graphs"]} />;
                if (connectorsPresent)
                    return <EmptyEntity data={noEntities["Tables"]} />;
                break;

            case PageNames.NewModel:
                if (tablesPresent)
                    return <EmptyEntity data={noEntities["Graphs"]} />;
                if (connectorsPresent)
                    return <EmptyEntity data={noEntities["Tables"]} />;
                break;

            case PageNames.Predictions:
                return <EmptyEntity data={noEntities["Training"]} />;

            case PageNames.Table:
                if (connectorsPresent)
                    return <EmptyEntity data={noEntities["Tables"]} />;
                break;

            case PageNames.Graph:
                if (connectorsPresent)
                    return <EmptyEntity data={noEntities["Tables"]} />;
                break;
        }
        return <EmptyEntity data={noEntities["Connectors"]} />;
    };

    return (
        <>
            {renderHeader()}
            {title && <Spacer height={10} />}
            <Card
                className={"w-full  border-none "}
                data-testid="project-empty-state">
                <CardContent className="p-2 flex items-center justify-center h-full">
                    {renderProjectEmptyPage()}
                </CardContent>
            </Card>
        </>
    );
};

export default WithErrorBoundary(EmptyEntities);
