// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useState } from "react";

import { AddCircle, Button } from "@kumo-ai/kumo-ui-catalog";

import EmptyEntities from "components/Empty/EmptyEntities";
import GraphOverview from "components/Graphs/Overview/GraphOverview";
import ListOverview from "components/ListOverview/ListOverview";
import Validation from "components/Validation/Validation";

import { useGraphs } from "hooks/graph/graphs";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { GraphListResponse, JSGraph } from "types/graphs";
import { FlowType } from "types/tables";

import { getProjectPath, PageNames } from "utils/Constants";
import { useCustomNavigation } from "utils/navigator";

/**
 * Graph Component
 *
 * Fetches list of graphs and render's graphs list in Table
 *
 */
const GraphsList: React.FC = () => {
    const navigate = useCustomNavigation();
    const projectId = useProjectId();
    const [selectedGraph, setSelectedGraph] =
        useState<GraphListResponse | null>(null);
    const [graphView, setGraphView] = useState<boolean>(true);
    const [loading, setLoading] = useState<boolean>(true);
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);
    const {
        isDataLoading,
        graphList: graphListOriginal,
        deleteGraph
    } = useGraphs({
        handleError,
        setPageLoading: setLoading,
        setSelectedGraph,
        projectId
    });

    const projectPaths = getProjectPath(projectId);

    const onCopyClick = (graphInfo: JSGraph) => {
        navigate(projectPaths.NewGraph, {
            state: {
                flow_type: FlowType.DUPLICATE,
                name: graphInfo?.name,
                jsGraph: graphInfo
            }
        });
        return;
    };

    const onEditClick = (graphInfo: JSGraph) => {
        navigate(projectPaths.EditGraph, {
            state: {
                flow_type: FlowType.EDIT,
                name: graphInfo?.name,
                jsGraph: graphInfo
            }
        });
    };

    const onConfirmDelete = useCallback(
        (value?: { data?: JSGraph }) => {
            dialogRef.current?.toggleDialog();
            if (value?.data) {
                deleteGraph(value.data);
            }
        },
        [deleteGraph, dialogRef]
    );

    const onDeleteClick = (graphInfo: JSGraph) => {
        dialogRef?.current?.toggleDialog(
            "Delete Graph",
            `Are you sure you want to delete graph "${graphInfo?.name}"? This action cannot be undone.`,
            { data: graphInfo },
            false,
            "Delete",
            "destructive"
        );
    };

    const buttons = [
        <Button
            id={"create-new-graph"}
            onClick={() => {
                navigate(projectPaths.NewGraph, {
                    state: {
                        flow_type: FlowType.CREATE,
                        name: ""
                    }
                });
            }}>
            <AddCircle /> Create graph
        </Button>
    ];
    const isLoading = isDataLoading || loading;
    if (!isLoading && graphListOriginal.length === 0) {
        return (
            <EmptyEntities
                currentPage={PageNames.Graph}
                title={PageNames.Graph}
            />
        );
    }

    return (
        <ListOverview
            title={"graphs"}
            data={graphListOriginal}
            loadingStatus={isDataLoading}
            entityPresent={
                graphListOriginal !== null && graphListOriginal?.length > 0
            }
            buttons={!projectId ? [] : buttons}
            selectedEntity={selectedGraph}
            setSelectedEntity={(graph) => {
                clearErrors();
                setSelectedGraph(graph);
            }}
            isGraphsPage={true}
            onEntityView={(visible) => setGraphView(visible)}
            overviewComponent={
                <>
                    <Validation
                        pageLoading={loading}
                        banner={banner}
                        alert={alert}
                        ctaMsgs={null}
                        dialogRef={dialogRef}
                        onDialogConfirm={onConfirmDelete}
                        hideWarnings={() => {}}
                        clearErrors={clearErrors}
                    />
                    {selectedGraph?.name && (
                        <GraphOverview
                            graphId={selectedGraph?.name}
                            onCopyClick={onCopyClick}
                            onDeleteClick={onDeleteClick}
                            onEditClick={onEditClick}
                            hideMenuOptions={!projectId}
                            graphView={graphView}
                            setGraphView={setGraphView}
                        />
                    )}
                </>
            }
        />
    );
};

export default GraphsList;
