// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useState } from "react";
import Spacer from "react-spacer";

import {
    Button,
    CloseSmall,
    GridCol,
    GridRow,
    SectionHeader,
    Sheet,
    SheetContent,
    SheetHeader
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import WarningErrorCTA from "components/@ui/Warning/WarningErrorCTA";
import GraphLinkGroups from "components/Graphs/LinkGroups/GraphLinkGroups";
import ConnectTables from "components/Tables/Connect/ConnectTables";

import {
    useBackwardLinks,
    useTableSelectionForGraphs
} from "hooks/graph/newGraph";
import { useTables } from "hooks/tables/tables";

import { OptionType } from "types/common";
import { ctaLevels } from "types/constants";
import { ColumnGroup, GraphFlowInfo, GraphTables, JSGraph } from "types/graphs";
import { FlowType, TableListResponse } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { defFlowInfo, updateGraphsGroups } from "utils/graph";
import { getKeys, isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

import { GraphSelection } from "../GraphSelection/GraphSelection";
import GraphSelectTables from "../SelectTables/GraphSelectTables";
import { SelectTableDropdown } from "../SelectTables/SelectTableDropDown";

/**
 * LinkSetup Component to Setup Links Between Table columns
 * with different groups
 */

export interface GraphLinkSetupProps {
    groups: ColumnGroup[];
    setGroups: React.Dispatch<React.SetStateAction<ColumnGroup[]>>;
    selectedGraph: OptionType;
    setSelectedGraph: (newValue: OptionType | null, value?: boolean) => void;
    graphFlowInfo: GraphFlowInfo;
    setGraphFlowInfo: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    allTables: GraphTables;
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>;
    show?: boolean;
    setGraphName?: (name: string) => void;
    newGraphName?: string;
    setNameError?: (err: string) => void;
    tableList?: Array<TableListResponse> | null;
    setTableList?: React.Dispatch<
        React.SetStateAction<Array<TableListResponse> | null>
    >;
    isTableStage?: boolean; // Whether first table stage in Create Graph flow.
    fromNewGraphPage?: boolean;
    setTableIdMap: React.Dispatch<
        React.SetStateAction<{
            [key: string]: string;
        }>
    >;
    handleError: HandleErrorFunc;
    resetErrLabel?: () => void;
    editableName?: boolean;
    setSaveGraph?: (v: boolean) => void;
    graphExist?: string;
    validatingGraph?: boolean;
    suggestLinks: () => void;
    hideGraphSelection?: boolean;
    setLoading: (v: boolean) => void;
    projectId?: string;
}

const GraphLinkSetup = ({
    groups,
    setGroups,
    selectedGraph,
    setSelectedGraph,
    graphFlowInfo,
    setGraphFlowInfo,
    allTables,
    setAllTables,
    show = true,
    setGraphName,
    newGraphName,
    setNameError = () => {},
    tableList,
    setTableList = () => {},
    isTableStage = false,
    fromNewGraphPage = false,
    setTableIdMap,
    handleError,
    resetErrLabel,
    editableName = true,
    setSaveGraph,
    graphExist,
    validatingGraph,
    suggestLinks,
    hideGraphSelection = false,
    setLoading,
    projectId
}: GraphLinkSetupProps) => {
    // List of tables with individual table details
    const [allTableNames, setAllTableNames] = useState<string[]>([]);
    const [allTableIds, setAllTableIds] = useState<{ [key: string]: string }>(
        {}
    );

    const [showTableModel, setShowTableModel] = useState(false);
    const [newTable, setNewTable] = useState<string | null>(null);

    const onChange = (open: boolean) => {
        if (!open) {
            setTimeout(() => {
                setShowTableModel(false);
            }, 500);
        }
    };

    //Helper function to update allTables.
    const updateAllTables = useCallback(
        (t: GraphTables) => {
            if (isObjectNotEmpty(t)) {
                const allTbls = { ...allTables };
                for (const [key, value] of Object.entries(t)) {
                    allTbls[key] = value;
                }
                setAllTables(allTbls);
            }
        },
        [allTables, setAllTables]
    );

    const updateGroups = useCallback(
        (config: GraphFlowInfo) => {
            updateGraphsGroups(config, allTables, groups, setGroups);
        },
        [allTables, groups, setGroups]
    );

    // Gets graph details ( tables in the graph ) and updates allTables if
    // necessary and graphFlowInfo

    const onGraphSelect = useCallback(
        (selectedGraphInfo?: JSGraph) => {
            const info = { ...defFlowInfo };
            info.selected_tables = [];
            info.tables = {};
            if (selectedGraphInfo?.tables) {
                info.flow_type = FlowType.EDIT;
                updateAllTables(selectedGraphInfo?.tables as GraphTables);
                info.tables = selectedGraphInfo?.tables as GraphTables;
                for (const [key] of Object.entries(selectedGraphInfo?.tables)) {
                    info.selected_tables.push(key); //key= table name
                }
            }
            info.name = selectedGraphInfo?.name ?? "";
            info.id = selectedGraphInfo?.id ?? "";

            const grp: ColumnGroup[] = selectedGraphInfo?.groups ?? [];
            grp?.forEach((e) => {
                e.updated = true;
                e.state = "saved";
            });
            setGroups(grp);

            setGraphFlowInfo(info);
        },
        [setGraphFlowInfo, setGroups, updateAllTables]
    );

    const { tableList: tablesListResp, refetchTables } = useTables(handleError);

    useEffect(() => {
        if (tablesListResp && tablesListResp.length > 0) {
            const list = tablesListResp.map((i) => i.name);
            const ids: { [key: string]: string } = {};

            tablesListResp.forEach((t) => {
                ids[t?.name] = t.id;
            });
            if (setTableList) {
                setTableList(tablesListResp);
            }
            setAllTableIds(ids);
            setTableIdMap(ids);
            setAllTableNames(list);
        }
    }, [tablesListResp, setTableIdMap, setTableList]);

    useEffect(() => {
        // In Edit/Copy mode, if tables are present but no groups,
        // update the groups to create start links (if pkey tables present).
        if (
            graphFlowInfo?.flow_type !== FlowType.CREATE &&
            groups?.length < 1 &&
            graphFlowInfo?.selected_tables?.length > 0
        ) {
            try {
                const pkeyExists = getKeys(graphFlowInfo?.tables)
                    .map((t) => graphFlowInfo?.tables[t]?.pkey)
                    .filter((key) => isValid(key));
                if (pkeyExists.length > 0) {
                    updateGroups(graphFlowInfo);
                }
            } catch (error) {
                console.log("error", error);
            }
        }
    }, [
        graphFlowInfo,
        groups.length,
        graphFlowInfo?.selected_tables.length,
        updateGroups
    ]);

    const { backwardLinks, isPkeyExists } = useBackwardLinks(
        graphFlowInfo?.tables,
        groups
    );

    const { onSelectTables } = useTableSelectionForGraphs(
        graphFlowInfo,
        allTables,
        setAllTables,
        setGraphFlowInfo,
        updateGroups,
        handleError,
        allTableIds,
        suggestLinks,
        setLoading
    );

    const onRemoveTable = useCallback(
        (tableName: string) => onSelectTables([], [tableName]),
        [onSelectTables]
    );

    useEffect(() => {
        //Make sure allTables are up-to-date.
        const tables = graphFlowInfo?.tables;
        const allTbls = { ...allTables };
        let updateRequired: boolean = false;
        getKeys(tables)?.forEach((t) => {
            if (allTbls && !(t in allTables)) {
                allTbls[t] = tables[t];
                updateRequired = true;
            }
        });
        if (updateRequired) {
            setAllTables(allTbls);
        }
    }, [graphFlowInfo?.tables, allTables, setAllTables]);

    return (
        <div className={show ? "" : "hidden"}>
            {!isPkeyExists && !isTableStage && (
                <>
                    <WarningErrorCTA
                        id={"warning-pkey-required"}
                        ctaLevel={ctaLevels.Critical}
                        data={[
                            {
                                title: "",
                                description: `At least one of the tables need to
                                have a primary key to create a graph.
                                Please edit a table to add a Primary key`,
                                is_error: true,
                                is_dismissible: false
                            }
                        ]}
                        handleClose={() => {}}
                    />
                    <Spacer height={8} />
                </>
            )}
            {fromNewGraphPage && isTableStage && tableList && (
                <GraphSelectTables
                    tableList={tableList}
                    graphFlowInfo={graphFlowInfo}
                    setGraphFlowInfo={setGraphFlowInfo}
                    allTables={allTables}
                    setAllTables={setAllTables}
                    tableIdMap={allTableIds}
                    updateGroups={updateGroups}
                    suggestLinks={suggestLinks}
                    handleError={handleError}
                    setLoading={setLoading}
                    projectId={projectId}
                />
            )}
            {!hideGraphSelection && (
                <GraphSelection
                    editableName={
                        graphFlowInfo?.flow_type === FlowType.EDIT ||
                        !editableName
                    }
                    fromNewGraphPage={fromNewGraphPage}
                    graphExist={graphExist}
                    graphName={newGraphName}
                    handleError={handleError}
                    previousSelected={graphFlowInfo?.id ?? graphFlowInfo?.name}
                    selectedGraph={selectedGraph}
                    setSaveGraph={setSaveGraph}
                    setSelectedGraph={setSelectedGraph}
                    validatingGraph={validatingGraph}
                    setGraphName={setGraphName}
                    setNameError={setNameError}
                    onGraphSelect={onGraphSelect}
                    projectId={projectId}
                />
            )}

            {selectedGraph && (
                <>
                    <Spacer height={12} />
                    <SectionHeader
                        title={"Tables & links"}
                        titleSize="md"
                        titleClassName="font-medium !font-medium pb-1 !pb-1"
                        description={
                            "You can change the tables and update links between them, the changes will be saved as a new graph without altering the original graph. "
                        }
                        descriptionClassName="pb-4 !pb-4"
                    />
                </>
            )}
            {getKeys(graphFlowInfo?.tables)?.length > 0 && (
                <GraphLinkGroups
                    allTables={allTables}
                    graphFlowInfo={graphFlowInfo}
                    groups={groups}
                    setGroups={setGroups}
                    backwardLinks={backwardLinks}
                    handleError={handleError}
                    show={!isTableStage}
                    onRemoveTable={onRemoveTable}
                />
            )}

            <Spacer height={16} />
            {(selectedGraph?.label || (fromNewGraphPage && !isTableStage)) && (
                <SelectTableDropdown
                    tableNameList={allTableNames}
                    graphFlowInfo={graphFlowInfo}
                    setGraphFlowInfo={setGraphFlowInfo}
                    allTables={allTables}
                    setAllTables={setAllTables}
                    tableIdMap={allTableIds}
                    updateGroups={updateGroups}
                    suggestLinks={suggestLinks}
                    toggleTableModel={() => setShowTableModel(true)}
                    setNewTable={setNewTable}
                    newTable={newTable}
                    resetErrLabel={resetErrLabel}
                    handleError={handleError}
                    setLoading={setLoading}
                    projectId={projectId}
                />
            )}
            <div id="jobs-side-window-view">
                <Sheet open={showTableModel} onOpenChange={onChange}>
                    <SheetContent>
                        <SheetHeader>
                            <GridRow className="flex items-center justify-between">
                                <GridCol>
                                    <SectionHeader
                                        title={"Add table"}
                                        description={
                                            "Select the connector or another source to add a table"
                                        }
                                    />
                                </GridCol>
                                <GridCol>
                                    <Button
                                        id="side-table-view"
                                        aria-label="Close"
                                        className="px-2 [&_svg]:size-6"
                                        variant="outline"
                                        onClick={() => {
                                            onChange(false);
                                        }}>
                                        <CloseSmall />
                                    </Button>
                                </GridCol>
                            </GridRow>
                            <ConnectTables
                                hideTitle={true}
                                refetchTable={(id) => {
                                    refetchTables();
                                    setNewTable(id);
                                    onChange(false);
                                }}
                            />
                        </SheetHeader>
                    </SheetContent>
                </Sheet>
            </div>
        </div>
    );
};

export default WithErrorBoundary(GraphLinkSetup);
