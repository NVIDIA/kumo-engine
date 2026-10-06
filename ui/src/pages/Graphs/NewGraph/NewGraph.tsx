// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router";
import Spacer from "react-spacer";

import {
    Button,
    Card,
    CardContent,
    Description,
    GridCol,
    GridRow,
    NetworkNode,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";

import GraphConfigPanel from "components/Graphs/GraphConfigPanel/GraphConfigPanel";
import GraphLinkSetup from "components/Graphs/LinkSetup/GraphLinkSetup";
import GraphNameInput from "components/Graphs/NameInput/GraphNameInput";
import Validation from "components/Validation/Validation";

import { useSuggestLinks } from "hooks/graph/graphs";
import { useCreateGraph, useGraphValidation } from "hooks/graph/newGraph";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { CustomError, EntityErrorMsg, OptionType } from "types/common";
import { ColumnGroup, GraphFlowInfo, GraphTables } from "types/graphs";
import { FlowType, TableListResponse } from "types/tables";

import { getProjectPath } from "utils/Constants";
import { prepareCustomError } from "utils/errorHandler";
import { getDefaultGraphGroups, getDefaultGraphInfo } from "utils/graph";
import { getKeys, isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import { invalidateGraphsList } from "utils/queryInvalidation";

/**
 * New Graph Component
 *
 * Component to show Create / Edit / Copy Graph workflow.
 *
 */
const NewGraph: React.FC = () => {
    const loc = useLocation();
    const navigate = useCustomNavigation();

    const graphNotExists = loc?.state?.graphNotExists ?? false;
    const selectedProject = useProjectId();
    const projectPaths = getProjectPath(selectedProject);

    const [loading, setLoading] = useState<boolean>(false);
    //Guards the Save submit (create or validate+edit) against re-entry
    //while a round-trip is in flight (e.g. a queued keyboard activation
    //firing before the disabled state of the Save button commits).
    const submitInFlightRef = useRef<boolean>(false);

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [isPkeyExists, setIsPkeyExists] = useState<boolean>(false);

    const [tableList, setTableList] = useState<Array<TableListResponse> | null>(
        null
    );

    const { validateGraph: validateGraphConfig } = useGraphValidation();
    const { createGraph } = useCreateGraph();

    //Two stages : Table selection stage ( first ) and link creation (next)
    const [tableStage, setTableStage] = useState<boolean>(true);
    const [groups, setGroups] = useState<Array<ColumnGroup>>(() =>
        getDefaultGraphGroups(loc?.state?.jsGraph)
    );
    const [selectedGraph, setSelectedGraph] = useState<OptionType | null>(null);
    // List of tables with individual table details
    const [allTables, setAllTables] = useState<GraphTables>({});

    const [tableIdMap, setTableIdMap] = useState<{ [key: string]: string }>({});
    const queryClient = useQueryClient();

    const [graphFlowInfo, setGraphFlowInfo] = useState<GraphFlowInfo>(() =>
        getDefaultGraphInfo(loc)
    );
    const [errLabelMsg, setErrLabelMsg] = useState<EntityErrorMsg>({
        entity: "",
        errorMsg: ""
    });

    /**** Name input validation  */
    const [name, setName] = useState<string>(graphFlowInfo?.name as string);
    //Server-reported name validation error, lifted from GraphNameInput so
    //the Next/Save gate reflects the same validity the field displays.
    //GraphNameInput resets it on every keystroke and sets it whenever the
    //server flags the name as invalid.
    const [nameServerError, setNameServerError] = useState<string>("");

    const { suggestLinks } = useSuggestLinks(
        graphFlowInfo,
        allTables,
        groups,
        setGroups,
        handleError,
        setLoading
    );

    const stageToggle = useCallback(() => {
        setTableStage(!tableStage);
    }, [tableStage]);

    const onCancelClick = useCallback(() => {
        if (graphFlowInfo?.flow_type === FlowType.CREATE) {
            navigate(projectPaths.Graphs, {
                replace: true
            });
        } else {
            navigate(-1);
        }
    }, [graphFlowInfo?.flow_type, navigate, projectPaths.Graphs]);

    const onNextClick = useCallback(() => {
        if (!isValid(name)) {
            setErrLabelMsg({
                entity: "name",
                errorMsg: "Please enter graph name."
            });
            return;
        }

        setGraphFlowInfo((graphFlowInfo) => ({
            ...graphFlowInfo,
            "name": name
        }));
        //Set the updated flag just so that the links are properly
        // shown in the next page.
        if (groups && groups.length > 0) {
            const grps: ColumnGroup[] = [...groups];
            grps.forEach((g) => {
                g.updated = true;
                g.state = "saved";
            });
            setGroups(grps);
        }

        stageToggle();
    }, [stageToggle, name, groups]);

    /**
     *  Function that makes an API call for Edit graph
     */
    const editGraph = useCallback(async () => {
        const gfc: GraphFlowInfo = { ...graphFlowInfo };
        if (gfc.flow_type === FlowType.DUPLICATE) {
            //Update the name if flow is Duplicate
            gfc.name = name;
            setGraphFlowInfo(gfc);
        }
        setLoading(true);
        try {
            const { graphId } = await createGraph(
                gfc,
                groups,
                handleError,
                true
            );
            if (graphId) {
                setLoading(false);

                if (isValid(graphId)) {
                    navigate(projectPaths.Graphs + gfc.name);
                }
            }
        } catch (error) {
            const title = "Graph Creation failed!";
            handleError(error as CustomError, title);
        }
    }, [
        graphFlowInfo,
        name,
        createGraph,
        groups,
        handleError,
        navigate,
        projectPaths.Graphs
    ]);

    /**
     *  This method invokes REST API call to  validate
     *  graph, and if successful, then calls Edit graph API.
     *  No-op while a previous submit is still in flight.
     */
    const validateAndEditGraph = useCallback(async () => {
        if (submitInFlightRef.current) {
            //A submit is already in flight; ignore re-entrant activations.
            return;
        }
        submitInFlightRef.current = true;
        setLoading(true);
        try {
            const { errors, statusCode } = await validateGraphConfig(
                graphFlowInfo,
                groups
            );
            if (errors?.length < 1) {
                invalidateGraphsList(queryClient);
                await editGraph();
            } else {
                const title = "Graph Validation failed!";
                const err = prepareCustomError(errors?.join(", "), statusCode);
                handleError(err, title);
            }
        } catch (error) {
            const title = "Graph Validation failed!";
            handleError(error as CustomError, title);
        } finally {
            submitInFlightRef.current = false;
        }
    }, [
        editGraph,
        graphFlowInfo,
        groups,
        handleError,
        queryClient,
        validateGraphConfig
    ]);

    /**
     *  Function that makes an API call for Create graph.
     *  No-op while a previous submit is still in flight.
     */
    const onCreateGraph = useCallback(async () => {
        if (submitInFlightRef.current) {
            //A submit is already in flight; ignore re-entrant activations.
            return;
        }
        submitInFlightRef.current = true;
        const gfc: GraphFlowInfo = { ...graphFlowInfo };
        //Update the name.
        gfc.name = name;
        setGraphFlowInfo(gfc);

        setLoading(true);
        try {
            const { errors: validationErrors, statusCode } =
                await validateGraphConfig(gfc, groups);
            if (validationErrors?.length > 0) {
                const title = "Graph Validation failed!";
                const err = prepareCustomError(
                    validationErrors?.join(", "),
                    statusCode
                );
                handleError(err, title);
                return;
            }

            const { graphId } = await createGraph(gfc, groups, handleError);
            if (graphId) {
                if (isValid(graphId)) {
                    navigate(projectPaths.Graphs + gfc.name);
                }
            }
        } catch (error) {
            const title = "Graph Creation failed!";
            handleError(error as CustomError, title);
        } finally {
            submitInFlightRef.current = false;
        }
    }, [
        graphFlowInfo,
        name,
        validateGraphConfig,
        groups,
        createGraph,
        handleError,
        navigate,
        projectPaths.Graphs
    ]);

    const onFinishClick = useCallback(async () => {
        //Check for error states of inputs
        const checkBeforeAdd = () => {
            if (!isValid(name)) {
                setErrLabelMsg({
                    entity: "name",
                    errorMsg: "Please enter graph name."
                });
                return true;
            } else if (getKeys(graphFlowInfo?.tables)?.length < 1) {
                setErrLabelMsg({
                    entity: "table",
                    errorMsg: "Please select table(s)."
                });
                return true;
            } else {
                return false; //No error
            }
        };
        if (checkBeforeAdd()) {
            return;
        }
        if (isObjectNotEmpty(graphFlowInfo) && groups) {
            if (graphFlowInfo.flow_type === FlowType.CREATE) {
                onCreateGraph();
            } else {
                validateAndEditGraph();
            }
        }
    }, [graphFlowInfo, groups, name, onCreateGraph, validateAndEditGraph]);

    const resetErrMsg = useCallback(() => {
        //Reset error-label for table entity if present
        if (errLabelMsg?.entity === "table") {
            //Reset
            setErrLabelMsg({
                entity: "",
                errorMsg: ""
            });
        }
    }, [errLabelMsg]);

    useEffect(() => {
        let pkeyExists = getKeys(graphFlowInfo?.tables)?.length < 1;
        getKeys(graphFlowInfo?.tables).forEach((table) => {
            if (isValid(graphFlowInfo?.tables[table]?.pkey)) {
                pkeyExists = true;
                return;
            }
        });
        setIsPkeyExists(pkeyExists);
    }, [graphFlowInfo?.tables, groups, clearErrors]); // groups dependency is required here, don't remove

    const isNextDisabled =
        (tableStage ? !isValid(name) : !isPkeyExists) || nameServerError !== "";

    return (
        <>
            <GridRow
                className="flex items-center justify-between"
                data-screenshot-ready={loading ? undefined : "new-graph-page"}>
                <GridCol>
                    <Title
                        data-testid="new-graph-page-heading"
                        variant="page-heading"
                        className="!font-bold text-3xl">
                        {graphFlowInfo?.flow_type === FlowType.EDIT
                            ? "Edit Graph"
                            : "Create New Graph"}
                    </Title>
                    <Description
                        size="md"
                        className="leading-5 mt-2 text-title-secondary">
                        Link tables to create graph schema
                    </Description>
                </GridCol>
                <GridCol>
                    <div className="flex gap-2 !gap-2">
                        {tableStage === false && (
                            <Button
                                id="back-graph-create"
                                variant="outline"
                                onClick={() => {
                                    if (errLabelMsg?.entity === "table") {
                                        //Reset
                                        setErrLabelMsg({
                                            entity: "",
                                            errorMsg: ""
                                        });
                                    }
                                    clearErrors();
                                    stageToggle();
                                }}>
                                Back
                            </Button>
                        )}
                        {!graphNotExists && (
                            <Button
                                id="cancel-graph-create"
                                variant="outline"
                                onClick={onCancelClick}>
                                Cancel
                            </Button>
                        )}
                        <Button
                            id="add-graph"
                            onClick={tableStage ? onNextClick : onFinishClick}
                            disabled={isNextDisabled || loading}>
                            {tableStage ? "Next" : "Save"}
                        </Button>
                    </div>
                </GridCol>
            </GridRow>
            {banner && <Spacer height={16} />}
            <Validation
                pageLoading={loading}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            <Spacer height={16} />
            <Card className="rounded-sm w-full shadow-none border-0">
                <CardContent className="px-0">
                    <div className="max-w-150">
                        <GraphNameInput
                            name={name}
                            placeholder="name"
                            setName={(e) => {
                                if (errLabelMsg.entity === "name") {
                                    setErrLabelMsg({
                                        entity: "",
                                        errorMsg: ""
                                    });
                                }
                                setName(e);
                            }}
                            readOnly={false}
                            setError={setNameServerError}
                            showState={true}
                            shouldValidate={graphFlowInfo.name !== name}
                        />
                    </div>
                    {errLabelMsg?.entity === "name" &&
                        errLabelMsg.errorMsg !== "" && (
                            <Description
                                id="errlabel-name"
                                variant="destructive"
                                size="sm"
                                className="leading-5">
                                {errLabelMsg.errorMsg}
                            </Description>
                        )}
                    <div className={"my-4"}>
                        {tableStage
                            ? "Select tables you want to use in this graph"
                            : "Configure the links between the tables to complete this graph"}
                    </div>
                    <GridRow xs={12} sm={12} md={12} className="g-0">
                        <GridCol xs={5} sm={5} className="overflow-auto">
                            <GraphLinkSetup
                                groups={groups}
                                setGroups={(g) => {
                                    clearErrors();
                                    setGroups(g);
                                }}
                                selectedGraph={selectedGraph as OptionType}
                                setSelectedGraph={setSelectedGraph}
                                graphFlowInfo={graphFlowInfo}
                                setGraphFlowInfo={setGraphFlowInfo}
                                allTables={allTables}
                                setAllTables={setAllTables}
                                tableList={tableList}
                                setTableList={setTableList}
                                isTableStage={tableStage}
                                fromNewGraphPage={true}
                                setTableIdMap={setTableIdMap}
                                handleError={handleError}
                                resetErrLabel={resetErrMsg}
                                suggestLinks={suggestLinks}
                                setLoading={setLoading}
                                projectId={selectedProject ?? undefined}
                            />
                            {errLabelMsg?.entity === "table" &&
                                errLabelMsg.errorMsg !== "" && (
                                    <Description
                                        id="errlabel-table"
                                        variant="destructive"
                                        size="sm"
                                        className="leading-5">
                                        {errLabelMsg.errorMsg}
                                    </Description>
                                )}
                        </GridCol>

                        {graphFlowInfo?.selected_tables?.length > 0 ? (
                            <GridCol
                                xs={7}
                                sm={7}
                                className="h-full overflow-auto">
                                <GraphConfigPanel
                                    gfc={graphFlowInfo}
                                    setGfc={setGraphFlowInfo}
                                    allTables={allTables}
                                    setAllTables={setAllTables}
                                    groups={groups}
                                    setGroups={setGroups}
                                    tableIdMap={tableIdMap}
                                    suggestLinks={suggestLinks}
                                />
                            </GridCol>
                        ) : (
                            <GridCol
                                xs={7}
                                sm={7}
                                className="bg-surface-grey h-160 flex justify-center items-center">
                                <div className="text-center">
                                    <NetworkNode className="size-16 fill-icon-secondary" />
                                    <div className="fs-5 font-inter text-title-secondary">
                                        {`No graph`}
                                    </div>
                                </div>
                            </GridCol>
                        )}
                    </GridRow>
                </CardContent>
            </Card>
        </>
    );
};

export default NewGraph;
