// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import {
    Button,
    Checkbox,
    Description,
    GridCol,
    GridRow,
    Label,
    SectionHeader,
    Select,
    Skeleton,
    Spinner
} from "@kumo-ai/kumo-ui-catalog";

import { useGetGraphInfo, useGraphs } from "hooks/graph/graphs";

import { CustomError, OptionType } from "types/common";
import { JSGraph } from "types/graphs";
import { HandleErrorFunc } from "types/validation";

import { ADD_KUMO_ENTITY_VALUE } from "utils/Constants";
import { copyObject } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

import GraphNameInput from "../NameInput/GraphNameInput";

interface NewGraphFormProps {
    saveGraphToReuse: boolean;
    graphName?: string;
    setGraphName?: (name: string) => void;
    setNameError?: (error: string) => void;
    editableName: boolean;
    graphExist?: string;
    validatingGraph?: boolean;
    onResetGraph: () => void;
    onSaveGraphToggle: () => void;
}

const NewGraphForm = ({
    saveGraphToReuse,
    graphName,
    setGraphName,
    setNameError,
    editableName,
    graphExist,
    validatingGraph,
    onResetGraph,
    onSaveGraphToggle
}: NewGraphFormProps) => {
    return (
        <GridRow className="mb-2">
            <GridCol xs={12} sm={12} md={12}>
                {saveGraphToReuse && (
                    <GridRow>
                        <GridCol xs={9} sm={9} md={9}>
                            <GraphNameInput
                                name={graphName}
                                setName={setGraphName}
                                setError={setNameError}
                                readOnly={editableName}
                            />
                            {graphExist && (
                                <Description
                                    size="sm"
                                    className="pt-2"
                                    variant="destructive">{`Graph already exists with name ${graphExist}`}</Description>
                            )}
                            {!isValid(graphName) && (
                                <Description
                                    size="sm"
                                    variant="destructive"
                                    className="pt-2 !mb-0">
                                    Graph name is needed if saving for later use
                                </Description>
                            )}
                        </GridCol>
                        <GridCol
                            xs={3}
                            sm={3}
                            md={3}
                            className="flex items-start justify-end pt-5.5">
                            <Button
                                id="remove-graph"
                                onClick={onResetGraph}
                                variant="light-outline"
                                size="sm">
                                Remove graph
                            </Button>
                        </GridCol>
                    </GridRow>
                )}

                {validatingGraph ? (
                    <div className="flex mt-1">
                        <Spinner />
                    </div>
                ) : (
                    !saveGraphToReuse && (
                        <GridRow>
                            <GridCol
                                xs={9}
                                sm={9}
                                md={9}
                                className="flex items-center">
                                <Checkbox
                                    checked={saveGraphToReuse}
                                    onCheckedChange={onSaveGraphToggle}
                                    aria-label="checkbox-save-graph-reuse"
                                    className="cursor-pointer flex-shrink-0"
                                />
                                <Label className="mb-0 mt-0 ml-2 font-normal text-sm">
                                    Save graph to reuse later
                                </Label>
                            </GridCol>
                            <GridCol
                                xs={3}
                                sm={3}
                                md={3}
                                className="flex items-center justify-end">
                                <Button
                                    id="back-graph"
                                    onClick={onResetGraph}
                                    variant="light-outline"
                                    size="sm">
                                    Back
                                </Button>
                            </GridCol>
                        </GridRow>
                    )
                )}
            </GridCol>
        </GridRow>
    );
};

export interface GraphSelectionProps {
    graphName?: string;
    setGraphName?: (name: string) => void;
    setNameError?: (error: string) => void;
    editableName: boolean;
    graphExist?: string;
    validatingGraph?: boolean;
    setSaveGraph?: (value: boolean) => void;
    selectedGraph: OptionType;
    setSelectedGraph: (newValue: OptionType | null, value?: boolean) => void;
    previousSelected: string;
    handleError: HandleErrorFunc;
    fromNewGraphPage: boolean;
    onGraphSelect: (selectedGraph?: JSGraph) => void;
    projectId?: string;
}

export const GraphSelection = ({
    graphName,
    setGraphName,
    setNameError,
    editableName,
    graphExist,
    validatingGraph,
    setSaveGraph,
    selectedGraph,
    setSelectedGraph,
    previousSelected,
    handleError,
    fromNewGraphPage,
    onGraphSelect,
    projectId
}: GraphSelectionProps) => {
    const [saveGraphToReuse, setSaveGraphToReuse] = useState<boolean>(false);

    const [newGraph, setNewGraph] = useState<boolean>(false);

    const { isDataLoading: loadingGraphs, graphList: graphs } = useGraphs({
        enabled: fromNewGraphPage === false,
        projectId
    });

    const graphList = useMemo<OptionType[]>(
        () => [
            { value: ADD_KUMO_ENTITY_VALUE, label: "Create new graph" },
            ...(graphs?.map((e) => ({ value: e.id, label: e.name })) ?? [])
        ],
        [graphs]
    );

    // necessary and graphFlowInfo
    const { isLoading: fetchingGraph, selectedGraphInfo } = useGetGraphInfo(
        selectedGraph?.value ?? null,
        handleError,
        undefined,
        selectedGraph?.value !== ADD_KUMO_ENTITY_VALUE &&
            previousSelected !== selectedGraph?.value
    );

    const onSelectGraph = useCallback(
        (selected: OptionType | null) => {
            if (previousSelected === selected?.value) {
                return;
            }
            setSelectedGraph(selected);
            const isNewGraph = selected?.value === ADD_KUMO_ENTITY_VALUE;
            setNewGraph(isNewGraph);
            if (isNewGraph) {
                onGraphSelect();
            }
        },
        [onGraphSelect, previousSelected, setSelectedGraph]
    );

    const handleResetGraph = useCallback(() => {
        onSelectGraph(null);
        setSaveGraph?.(false);
        setGraphName?.("");
        setSaveGraphToReuse(false);
        onGraphSelect();
    }, [onSelectGraph, setSaveGraph, setGraphName, onGraphSelect]);

    const handleSaveGraphToggle = useCallback(() => {
        if (!graphExist || saveGraphToReuse) {
            setSaveGraphToReuse((prev) => {
                if (!prev) {
                    setGraphName?.(graphName ?? "");
                }
                setSaveGraph?.(!prev);
                return !prev;
            });
        } else {
            const err = `Graph already exists with name ${graphExist}`;
            const error: CustomError = {
                response: { data: err },
                name: "",
                message: err
            };
            handleError(error, "Graph already exists!");
        }
    }, [
        graphExist,
        saveGraphToReuse,
        setGraphName,
        graphName,
        setSaveGraph,
        handleError
    ]);

    useEffect(() => {
        if (
            fetchingGraph ||
            selectedGraphInfo === null ||
            (previousSelected?.trim() !== "" &&
                (previousSelected === selectedGraphInfo?.name ||
                    previousSelected === selectedGraphInfo?.id)) ||
            selectedGraph?.value === ADD_KUMO_ENTITY_VALUE
        ) {
            return;
        }
        onGraphSelect(copyObject(selectedGraphInfo));
    }, [
        fetchingGraph,
        onGraphSelect,
        previousSelected,
        selectedGraph?.value,
        selectedGraphInfo
    ]);

    // If selectedGraph has only an ID, find its name from the graph list and update selected graph info.
    useEffect(() => {
        if (!graphs?.length || !selectedGraph?.value || fromNewGraphPage)
            return;

        const matchedGraph = graphs?.find(
            (g) => g?.id === selectedGraph?.value
        );

        if (matchedGraph && selectedGraph?.label !== matchedGraph?.name) {
            setSelectedGraph(
                {
                    value: matchedGraph?.id,
                    label: matchedGraph?.name
                },
                false
            );
        }
    }, [
        fromNewGraphPage,
        graphs,
        selectedGraph?.value,
        selectedGraph?.label,
        setSelectedGraph
    ]);

    if (newGraph) {
        return (
            <NewGraphForm
                saveGraphToReuse={saveGraphToReuse}
                graphName={graphName}
                setGraphName={setGraphName}
                setNameError={setNameError}
                editableName={editableName}
                graphExist={graphExist}
                validatingGraph={validatingGraph}
                onResetGraph={handleResetGraph}
                onSaveGraphToggle={handleSaveGraphToggle}
            />
        );
    }

    return (
        fromNewGraphPage === false && (
            <>
                <SectionHeader
                    title={"Graph"}
                    titleSize="md"
                    titleClassName="pb-2 !pb-2 font-medium !font-medium"
                />
                <GridCol xs={12} sm={12} className="w-83.75">
                    {loadingGraphs ? (
                        <Skeleton className="h-8" />
                    ) : (
                        <Select
                            id={"select-graph"}
                            placeholder="Select graph"
                            search={true}
                            selectedValues={
                                selectedGraph && selectedGraph.value
                                    ? [selectedGraph.label]
                                    : []
                            }
                            emptyText="No graph found."
                            mode="single"
                            options={graphList.map((graph) => ({
                                value:
                                    graph.value === ADD_KUMO_ENTITY_VALUE
                                        ? graph.value
                                        : graph.label,
                                label: graph.label
                            }))}
                            onValuesChange={(newValue) => {
                                const currSel: OptionType = graphList.find(
                                    (item) =>
                                        newValue.includes(item.label) ||
                                        newValue[0] === ADD_KUMO_ENTITY_VALUE
                                ) as OptionType;
                                onSelectGraph(currSel);
                            }}
                        />
                    )}
                </GridCol>
            </>
        )
    );
};
