// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState
} from "react";

import {
    Description,
    NetworkNode,
    Skeleton,
    TabCard,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import GraphInformationPanel from "components/Graphs/InformationPanel/GraphInformationPanel";
import GraphLinkSetup from "components/Graphs/LinkSetup/GraphLinkSetup";
import SubgraphsSampling from "components/NewJob/Training/SubgraphsSampling/SubgraphsSampling";
import ModelPlannerEditor from "components/PredictiveQueries/Write/ModelPlannerEditor";

import { useSuggestLinks } from "hooks/graph/graphs";

import { OptionType } from "types/common";
import { ColumnGroup, GraphFlowInfo, GraphTables, JSGraph } from "types/graphs";
import { DropdownOption } from "types/inputs";
import { CreateTrainingJobArgs } from "types/jobs";
import { FlowType, JSTable, TableListResponse } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { ADD_KUMO_ENTITY_VALUE } from "utils/Constants";
import { tablesFromGraphResponse } from "utils/Converters/graphsConverter";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

import ModelSettings from "./ModelSettings";

export interface TrainingSettingsProps {
    createJobArgs: CreateTrainingJobArgs;
    setField: (field: string, value: string | boolean | JSGraph | null) => void;
    changeRunMode: (value: DropdownOption) => void;
    modelPlanLoading: boolean;
    setNameError: (err: string) => void;
    showOverwrittenWarning: boolean;
    closeWarning: () => void;
    handleError: HandleErrorFunc;
    setNewGraph: (value: boolean) => void;
    setSaveGraph: (value: boolean) => void;
    setGroups: React.Dispatch<React.SetStateAction<ColumnGroup[]>>;
    groups: ColumnGroup[];
    graphFlowInfo: GraphFlowInfo;
    setGraphFlowInfo: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    graphExist: string;
    validatingGraph: boolean;
    hideGraphSelection?: boolean;
    setParentLoading: (value: boolean) => void;
    bottomPanelHeight?: number;
    projectId?: string;
    taskType?: string;
}

// Schema tab indices
const SCHEMA_TAB_GRAPHS = "0";
const SCHEMA_TAB_SUBGRAPHS = "1";

const TrainingSettings: React.FC<TrainingSettingsProps> = ({
    createJobArgs,
    setField,
    changeRunMode,
    modelPlanLoading,
    setNameError,
    showOverwrittenWarning,
    closeWarning,
    handleError,
    setNewGraph,
    setGroups,
    groups,
    graphFlowInfo,
    setGraphFlowInfo,
    setSaveGraph,
    graphExist,
    validatingGraph,
    hideGraphSelection = false,
    setParentLoading,
    bottomPanelHeight,
    projectId,
    taskType
}) => {
    const suggestLinksRef = useRef(false);

    const tabs = [
        { label: "Graph settings", value: 0 },
        { label: "Model settings", value: 1 },
        { label: "Model planner", value: 2 }
    ];

    // Height available for the Model Planner editor
    const MODEL_PLANNER_CHROME_HEIGHT = 120; // tab header + padding
    const editorHeight =
        bottomPanelHeight && bottomPanelHeight > 400
            ? `${bottomPanelHeight - MODEL_PLANNER_CHROME_HEIGHT}px`
            : undefined;
    const [activeTab, setActiveTab] = useState<number>(0);

    // ── Right panel: schema tabs ──────────────────────────────────────────────
    const [activeSchemaTab, setActiveSchemaTab] =
        useState<string>(SCHEMA_TAB_GRAPHS);

    const [allTables, setAllTables] = useState<GraphTables>({
        ...(graphFlowInfo?.tables ?? {})
    });
    const [tableList, setTableList] = useState<Array<TableListResponse> | null>(
        null
    );
    const [loadingGraph, setLoadingGraph] = useState<boolean>(false);
    const [tableIdMap, setTableIdMap] = useState<{ [key: string]: string }>({});
    /**
     * A copied job hydrates its saved groups after the form mounts. Preserve
     * the copied graph and model plan during that hydration; later group edits
     * invalidate both so validation can regenerate the plan.
     */
    const [preserveInitialGroupHydration, setPreserveInitialGroupHydration] =
        useState(
            () =>
                Boolean(
                    createJobArgs.graph && createJobArgs.model_planner_yaml
                ) &&
                groups.length === 0 &&
                graphFlowInfo.flow_type !== FlowType.CREATE &&
                graphFlowInfo.selected_tables.length > 0 &&
                getKeys(graphFlowInfo.tables).some((table) =>
                    isValid(graphFlowInfo.tables[table]?.pkey)
                )
        );
    const initialGroupHydrationStarted = useRef(false);

    const graphInfo: JSGraph = useMemo(
        () => ({
            id: "",
            name: "",
            tables: graphFlowInfo?.tables,
            groups: groups,
            totalLinks: 0,
            totalTables: getKeys(graphFlowInfo?.tables).length,
            created: "",
            updated: "",
            dateCreated: "",
            dateUpdated: "",
            isActive: true
        }),
        [groups, graphFlowInfo]
    );

    const [selectedGraph, setSelectedGraph] = useState<DropdownOption | null>(
        createJobArgs.graph_id
            ? ({
                  value: createJobArgs.graph_id,
                  label: createJobArgs.graph_name
              } as DropdownOption)
            : null
    );

    useEffect(() => {
        if (selectedGraph === null && createJobArgs?.graph_id) {
            setSelectedGraph({
                value: createJobArgs.graph_id,
                label: createJobArgs.graph_name
            } as DropdownOption);
        }
    }, [createJobArgs?.graph_id, createJobArgs?.graph_name, selectedGraph]);

    const onGroupsChange = useCallback(
        (value: React.SetStateAction<ColumnGroup[]>) => {
            const preserveModelPlan = preserveInitialGroupHydration;
            if (preserveModelPlan && Array.isArray(value) && value.length > 0) {
                if (
                    initialGroupHydrationStarted.current &&
                    value.every((group) => group.updated !== true)
                ) {
                    setPreserveInitialGroupHydration(false);
                } else {
                    initialGroupHydrationStarted.current = true;
                }
            }
            if (selectedGraph?.value === ADD_KUMO_ENTITY_VALUE) {
                setNewGraph(true);
            }
            if (!preserveModelPlan) {
                setField("graph", null);
                setField("model_planner_yaml", "");
            }
            setGroups(value);
        },
        [
            preserveInitialGroupHydration,
            selectedGraph?.value,
            setField,
            setGroups,
            setNewGraph
        ]
    );

    const { suggestLinks } = useSuggestLinks(
        graphFlowInfo,
        allTables,
        groups,
        onGroupsChange,
        handleError,
        (isLoading: boolean) => {
            setLoadingGraph(isLoading);
            setParentLoading?.(isLoading);
        }
    );

    const onGraphSelect = useCallback(
        (
            selected: OptionType | null | DropdownOption,
            resetModelPlanner: boolean = true
        ) => {
            if (selected?.value === ADD_KUMO_ENTITY_VALUE) {
                setField("graph_id", "");
                setField("graph_name", "");
                setNewGraph(true);
            } else if (selected !== null) {
                setField("graph_id", selected.value as string);
                setField("graph_name", selected.label as string);
                setNewGraph(false);
            } else {
                setField("graph_id", "");
                setField("graph_name", "");
                setNewGraph(false);
            }
            setSelectedGraph(selected as DropdownOption);
            if (resetModelPlanner) {
                setPreserveInitialGroupHydration(false);
                setField("graph", null);
                setField("model_planner_yaml", "");
            }
        },
        [setField, setNewGraph]
    );

    useEffect(() => {
        if (preserveInitialGroupHydration) {
            return;
        }
        if (graphInfo !== null) {
            setField("graph", graphInfo);
        }
    }, [graphInfo, preserveInitialGroupHydration, setField]);

    const tables: JSTable[] = useMemo(
        () => (graphInfo !== null ? tablesFromGraphResponse(graphInfo) : []),
        [graphInfo]
    );

    // Auto-advance config tab to "Model Planner" tab once a plan is generated
    useEffect(() => {
        if (createJobArgs?.model_planner_yaml !== "" && activeTab === 0) {
            setActiveTab(2);
        }
    }, [activeTab, createJobArgs?.model_planner_yaml]);

    const graphInfoPanel = useMemo(() => {
        if (!graphInfo || getKeys(graphFlowInfo?.tables).length === 0)
            return null;
        return (
            <GraphInformationPanel
                graphInfo={graphInfo}
                tables={tables}
                weakGraphEdges={[]}
                enableEditTable={true}
                gfc={graphFlowInfo}
                setGfc={(value) => {
                    if (selectedGraph?.value === ADD_KUMO_ENTITY_VALUE) {
                        setNewGraph(true);
                    }
                    setField("graph", null);
                    setField("model_planner_yaml", "");
                    setGraphFlowInfo(value);
                }}
                allTables={allTables}
                setAllTables={setAllTables}
                groups={groups}
                setGroups={setGroups}
                tableIds={tableIdMap}
                callCreateGraph={() => setNewGraph(true)}
                suggestLinks={
                    graphFlowInfo.flow_type !== FlowType.CREATE
                        ? () => {}
                        : () => {
                              suggestLinksRef.current = true;
                          }
                }
                columnsPerRow={4}
            />
        );
    }, [
        graphInfo,
        graphFlowInfo,
        tables,
        allTables,
        groups,
        setGroups,
        tableIdMap,
        selectedGraph,
        setField,
        setGraphFlowInfo,
        setNewGraph
    ]);

    useEffect(() => {
        if (
            graphFlowInfo?.flow_type === FlowType.CREATE &&
            suggestLinksRef.current
        ) {
            suggestLinksRef.current = false;
            suggestLinks();
        }
    }, [graphFlowInfo, suggestLinks]);

    return (
        <div className="flex h-full">
            {/* ── Left panel: Configurations ───────────────────────────────── */}
            <div className="border border-border-secondary p-4 flex flex-col gap-4 overflow-hidden w-1/2 border-r-0 rounded-l rounded-r-none">
                <h3 className="text-base font-bold text-title-primary leading-4">
                    Configurations
                </h3>
                <div id="config-tabs" className="flex-1 min-h-0 flex flex-col">
                    <style>{`
                        #config-tabs [data-slot="card"] { border: none; background: transparent; padding: 0; box-shadow: none; margin: 0; gap: 0; display: flex; flex-direction: column; flex: 1; min-height: 0; }
                        #config-tabs [data-slot="card"] > [data-slot="tabs"] { display: flex; flex-direction: column; flex: 1; min-height: 0; }
                        #config-tabs [data-slot="card-header"] { height: auto; border-bottom: none; padding: 0; min-height: 0; flex-shrink: 0; }
                        #config-tabs [data-slot="card-content"] { padding: 16px 0 0 0; flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; }
                        #config-tabs [data-slot="tabs-content"] { flex: 1; min-height: 0; display: flex; flex-direction: column; }
                        #config-tabs [data-slot="tabs-list"] { width: 100%; padding-bottom: 0; gap: 32px; }
                        #config-tabs [data-slot="tabs-trigger"] { padding: 4px 0 !important; line-height: 16px; }
                    `}</style>
                    <TabCard
                        title=""
                        tabs={tabs}
                        key={"training_settings"}
                        defaultSelected={activeTab}>
                        {(currentTab) => (
                            <>
                                {currentTab === 1 && (
                                    <ModelSettings
                                        createJobArgs={createJobArgs}
                                        changeRunMode={changeRunMode}
                                    />
                                )}
                                {currentTab === 2 &&
                                    (modelPlanLoading ? (
                                        <Skeleton className="h-12.5" />
                                    ) : (
                                        <ModelPlannerEditor
                                            title="Model planner"
                                            optionsYamlWarning=""
                                            value={
                                                createJobArgs?.model_planner_yaml
                                            }
                                            onChange={(newval: string) =>
                                                setField(
                                                    "model_planner_yaml",
                                                    newval
                                                )
                                            }
                                            editorHeight={editorHeight}
                                            taskType={taskType}
                                            graphTables={allTables}
                                            graphGroups={groups}
                                            showOverwrittenWarning={
                                                showOverwrittenWarning
                                            }
                                            closeWarning={closeWarning}
                                        />
                                    ))}
                                <GraphLinkSetup
                                    groups={groups}
                                    setGroups={onGroupsChange}
                                    selectedGraph={selectedGraph as OptionType}
                                    setSelectedGraph={onGraphSelect}
                                    graphFlowInfo={graphFlowInfo}
                                    setGraphFlowInfo={(value) => {
                                        if (
                                            selectedGraph?.value ===
                                            ADD_KUMO_ENTITY_VALUE
                                        ) {
                                            setNewGraph(true);
                                        }
                                        setField("graph", null);
                                        setField("model_planner_yaml", "");
                                        setGraphFlowInfo(value);
                                    }}
                                    allTables={allTables}
                                    setAllTables={setAllTables}
                                    tableList={tableList}
                                    setTableList={setTableList}
                                    show={currentTab === 0}
                                    newGraphName={
                                        createJobArgs?.graph_name ?? ""
                                    }
                                    setGraphName={(name) => {
                                        setField("graph_name", name);
                                    }}
                                    setNameError={setNameError}
                                    handleError={handleError}
                                    setTableIdMap={setTableIdMap}
                                    editableName={true}
                                    setSaveGraph={setSaveGraph}
                                    graphExist={graphExist}
                                    validatingGraph={validatingGraph}
                                    suggestLinks={suggestLinks}
                                    hideGraphSelection={hideGraphSelection}
                                    setLoading={setParentLoading}
                                    projectId={projectId}
                                />
                            </>
                        )}
                    </TabCard>
                </div>
            </div>

            {/* ── Right panel: Graph Preview tabs ──────────────────────────── */}
            <div className="border border-border-secondary p-4 flex flex-col gap-4 overflow-hidden w-1/2 rounded-r rounded-l-none">
                <h3 className="text-base font-bold text-title-primary leading-4">
                    Graph Preview
                </h3>
                <div id="schema-tabs" className="flex flex-col flex-1 min-h-0">
                    <style>{`
                        #schema-tabs [data-slot="tabs-list"] { width: 100%; padding-bottom: 0; gap: 32px; }
                        #schema-tabs [data-slot="tabs-trigger"] { padding: 4px 0 !important; line-height: 16px; }
                        #schema-tabs [data-slot="tabs-content"] { padding: 0; flex: 1; min-height: 0; }
                    `}</style>
                    <Tabs
                        value={activeSchemaTab}
                        onValueChange={setActiveSchemaTab}
                        className="flex flex-col flex-1 min-h-0 gap-4">
                        <TabsList>
                            <TabsTrigger value={SCHEMA_TAB_GRAPHS}>
                                Graphs
                            </TabsTrigger>
                            <TabsTrigger value={SCHEMA_TAB_SUBGRAPHS}>
                                Subgraphs
                            </TabsTrigger>
                        </TabsList>

                        {/* ── Graphs tab ──────────────────────────────────── */}
                        <TabsContent
                            value={SCHEMA_TAB_GRAPHS}
                            className="flex flex-col flex-1 min-h-0 gap-4">
                            <div className="flex-1 border border-border-secondary rounded overflow-auto min-h-0">
                                {loadingGraph ? (
                                    <div className="p-2">
                                        <Skeleton className="w-48 h-20" />
                                    </div>
                                ) : graphInfoPanel ? (
                                    graphInfoPanel
                                ) : (
                                    <div className="flex justify-center items-center h-full min-h-50">
                                        <div className="flex flex-col items-center gap-2">
                                            <NetworkNode className="size-8 fill-icon-secondary" />
                                            <Description
                                                size="sm"
                                                variant="secondary"
                                                className="!mb-0">
                                                {`Select a graph`}
                                            </Description>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        {/* ── Subgraphs tab ────────────────────────────────── */}
                        <TabsContent
                            value={SCHEMA_TAB_SUBGRAPHS}
                            className="flex-1 min-h-0 overflow-auto">
                            <SubgraphsSampling
                                graph={createJobArgs.graph ?? null}
                                predictiveQuery={createJobArgs.query_yaml ?? ""}
                                modelPlanYaml={
                                    createJobArgs.model_planner_yaml ?? ""
                                }
                                loading={modelPlanLoading}
                            />
                        </TabsContent>
                    </Tabs>
                </div>
            </div>
        </div>
    );
};

export default WithErrorBoundary(TrainingSettings);
