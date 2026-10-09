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
    addEdge,
    applyEdgeChanges,
    applyNodeChanges,
    Button,
    Connection,
    Edge,
    EdgeChange,
    Node,
    NodeChange,
    OnConnect,
    OnEdgesChange,
    OnNodesChange,
    Sheet,
    SheetContent,
    SheetFooter,
    TableGraph,
    TableGraphEdge,
    TableGraphNode,
    TableGraphNodeData
} from "@kumo-ai/kumo-ui-catalog";
import dagre from "dagre";
import { ColumnKey } from "gen/v1/types";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import TableSliderOverview from "components/Tables/Overview/TableSliderOverview";

import {
    ColumnGroup,
    EdgeStats,
    GraphFlowInfo,
    GraphTables,
    JSGraph
} from "types/graphs";
import { ColumnArgs, JSTable } from "types/tables";

import { updateGraphsGroups } from "utils/graph";
const MIN_NODE_WIDTH = 200;
const DAGRE_LAYOUT_NODE_THRESHOLD = 10;
type SliderInfo = {
    show: boolean;
    name: string;
};

const getStringsFromData = (data: TableNodeDataType) => {
    const foreignKeyCols = data?.cols?.filter(({ name }) =>
        data?.foreignKeys?.includes(name as string)
    );

    const pKey = data?.pkey ? data.pkey : null;
    const idCols = data?.cols?.filter(
        ({ name, stype }) =>
            stype === "ID" &&
            !data?.foreignKeys?.includes(name as string) &&
            pKey !== name
    );
    return {
        timeframe:
            (data?.earliestFact !== "-" &&
                `Timeframe: ${data?.earliestFact} - ${data?.latestFact}`) ||
            "",
        pkey: (pKey && `PK: ${pKey}`) || "",
        timeCol: (data?.timeCol && `T: ${data.timeCol}`) || "",
        foreignKeyCols: foreignKeyCols?.map(({ name }) => `FK: ${name}`),
        idCols: idCols?.map(({ name }) => `${name}`)
    };
};

type TableNodeDataType = {
    id: string;
    label: string;
    cols: Array<ColumnArgs>;
    position: { x: number; y: number };
    pkey: string;
    timeCol: string;
    earliestFact: string;
    latestFact: string;
    foreignKeys: Array<string>;
    setSelectedTable: (t: string) => void;
    pkeySelected?: boolean;
    highLightKeys?: Array<string>;
};

type TableNodeType = Node<TableNodeDataType>;

const getNodesFromTableData = (
    tableData: Array<JSTable>,
    allForeignKeys: Array<ColumnKey>,
    setSelectedTable: (t: string) => void,
    isSnapshot?: boolean
): Array<TableNodeType> => {
    const r = 400;
    const totalPoints = tableData?.length;

    const connectors = tableData?.map(({ id }) => id.split(".")[0]);
    const differentConnectors = !connectors.every((v) => v === connectors[0]);

    const tableNodes: Array<TableNodeType> = [];
    tableData?.forEach(
        (
            {
                id,
                name_alias,
                columns,
                earliestFact,
                latestFact,
                pkey,
                timeCol
            },
            idx
        ) => {
            const name = name_alias;
            // Expand if there are any tables with different connectors
            let tableName = differentConnectors
                ? id
                : id?.includes(".")
                  ? id?.split(".")[1]
                  : id;
            if (name && tableName !== name) {
                tableName += ` (${name})`;
            }

            // Calculate position
            const theta = (Math.PI * 2) / totalPoints;
            const angle = theta * idx;
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);

            const foreignKeys = allForeignKeys
                .filter(({ table_name }) => table_name === name)
                .filter(({ col_name }) => col_name !== pkey)
                .map(({ col_name }) => col_name);
            const tableNode = {
                id: isSnapshot ? name : id,
                // id: name,
                position: { x, y },
                data: {
                    id: id,
                    label: tableName,
                    cols: columns,
                    position: { x, y },
                    pkey: pkey,
                    timeCol: timeCol,
                    earliestFact: earliestFact,
                    latestFact: latestFact,
                    foreignKeys: foreignKeys,
                    setSelectedTable: setSelectedTable
                },
                type: "table",
                className: "light"
            };
            tableNodes.push(tableNode as TableNodeType);
        }
    );
    return tableNodes;
};

type TableEdgeDataType = {
    confirmRequired: boolean;
    isWeakEdge: boolean;
};

type TableEdgeType = Edge<TableEdgeDataType>;

const getEdgesFromGraphData = (
    graphData: JSGraph,
    tableIds: Array<string>,
    weakEdges: Array<EdgeStats>,
    primaryKeys: Array<string>
): Array<TableEdgeType> => {
    const tableEdges: Array<TableEdgeType> = [];

    // There must be at least one primary key in the graph
    if (primaryKeys?.length === 0) {
        return tableEdges;
    }
    graphData?.groups?.forEach?.((group) => {
        if ((group?.columns?.length ?? 0) > 1) {
            // The src node is the table with the primary key in the
            // column group
            const src = group.columns.filter(({ table_name, col_name }) =>
                primaryKeys?.includes(`${table_name}.${col_name}`)
            )[0];
            if (src === undefined) {
                return;
            }
            // All other columns in the column group become dst nodes
            const others = group.columns.filter((o) => o !== src);

            others?.forEach((dst) => {
                if (src !== dst) {
                    const edgeId = `${src?.table_name}.${src?.col_name} <-> ${dst?.table_name}.${dst?.col_name}`;

                    const weakEdge =
                        weakEdges?.filter(
                            (element) =>
                                src?.table_name === element?.col_a_table_id &&
                                dst?.table_name === element?.col_b_table_id &&
                                src?.col_name === element?.col_a_column_name &&
                                dst?.col_name === element?.col_b_column_name
                        ).length > 0
                            ? true
                            : false;

                    // TODO(frg100): Make spacing around text consistent
                    const tableEdge: TableEdgeType = {
                        id: edgeId,
                        source: src?.table_name,
                        sourceHandle: "source_PK: " + src?.col_name,
                        target: dst?.table_name,
                        targetHandle: "target_FK: " + dst?.col_name,
                        type: "customEdge",
                        data: {
                            confirmRequired: false,
                            isWeakEdge: weakEdge
                        }
                    };
                    if (
                        tableIds?.includes(src?.table_name) &&
                        tableIds?.includes(dst?.table_name)
                    ) {
                        tableEdges.push(tableEdge);
                    }
                }
            });
        }
    });
    return tableEdges;
};

const getForeignKeysFromGraphData = (graphData: JSGraph): Array<ColumnKey> => {
    const allForeignKeys: Array<ColumnKey> = [];

    graphData?.groups?.forEach?.((group) => {
        if (group?.columns?.length > 1) {
            group?.columns?.forEach((col) => {
                allForeignKeys.push(col);
            });
        }
    });
    return allForeignKeys;
};

const getApproximateDimensions = (node: TableNodeType) => {
    // Get all strings
    const { timeframe, pkey, timeCol, foreignKeyCols, idCols } =
        getStringsFromData(node?.data);

    // Calculate the length of the longest string
    const TABLE_WIDTH_FACTOR = 13; // Calculated empirically
    const COL_WIDTH_FACTOR = 9; // Calculated empirically
    let width = 0;
    const rows = [timeframe, pkey, timeCol, ...foreignKeyCols, ...idCols];
    rows.forEach((row) => {
        const colWidth = row?.length * COL_WIDTH_FACTOR;
        width = Math.max(width, colWidth);
    });
    // Check the table name width separately because it's in a bolder font
    const tableNameWidth = node?.data?.label?.length * TABLE_WIDTH_FACTOR;
    width = Math.max(width, tableNameWidth);
    width = Math.max(width, MIN_NODE_WIDTH);

    // Calculate height (NOTE: Depends on styling)
    const COL_HEIGHT = 28 + 16;
    let height = 24 + 12 * 2; // Title
    if (timeframe) {
        height += 24; // TimeFrame section
    }
    const bodyCols = [pkey, timeCol, ...foreignKeyCols, ...idCols].filter(
        (v) => v
    );
    height += bodyCols.length * COL_HEIGHT;
    height += 8 * 2; // 8px padding on each side

    // Return an approximate width and height
    return { width, height };
};

const applyDagreJSLayout = (
    nodes: Array<TableNodeType>,
    edges: Array<TableEdgeType>
): [TableNodeType[], TableEdgeType[]] => {
    const dagreGraph = new dagre.graphlib.Graph().setDefaultEdgeLabel(
        () => ({})
    );
    dagreGraph.setGraph({ rankdir: "LR", nodesep: 50, ranksep: 75 }); // TOP/BOTTOM (alternate is TB)

    nodes?.forEach((node) => {
        dagreGraph.setNode(node?.id, getApproximateDimensions(node));
    });

    edges?.forEach((edge) => {
        dagreGraph.setEdge(edge?.source, edge?.target);
    });

    dagre.layout(dagreGraph);

    nodes = nodes?.map((node) => {
        const nodeWithPosition = dagreGraph.node(node?.id);
        const { width, height } = getApproximateDimensions(node);

        const newNode = {
            ...node,
            // We are shifting the dagre node position (anchor=center center) to the top left
            // so it matches the React Flow node anchor point (top left).
            position: {
                x: nodeWithPosition.x - width / 2,
                y: nodeWithPosition.y - height / 2
            }
        };

        return newNode;
    });

    return [nodes, edges];
};

const applyResponsiveGridLayout = (
    nodes: Array<TableNodeType>,
    edges: Array<TableEdgeType>,
    viewportWidth: number,
    columnsPerRow?: number
): [TableNodeType[], TableEdgeType[]] => {
    if (nodes.length === 0) {
        return [nodes, edges];
    }

    const HORIZONTAL_GAP = 90;
    const VERTICAL_GAP = 40;
    const PADDING = 24;

    const dimensions = nodes.map((node) => getApproximateDimensions(node));

    const maxNodeWidth = Math.max(
        ...dimensions.map(({ width }) => width),
        MIN_NODE_WIDTH
    );

    const usableWidth = Math.max(viewportWidth - PADDING * 2, maxNodeWidth);

    const columns = columnsPerRow
        ? Math.max(1, columnsPerRow)
        : Math.max(
              1,
              Math.floor(
                  (usableWidth + HORIZONTAL_GAP) /
                      (maxNodeWidth + HORIZONTAL_GAP)
              )
          );

    const rowMaxHeights: number[] = [];
    dimensions.forEach(({ height }, i) => {
        const r = Math.floor(i / columns);
        rowMaxHeights[r] = Math.max(rowMaxHeights[r] ?? 0, height);
    });

    const rowOffsets = rowMaxHeights.reduce<number[]>((acc, _h, i) => {
        acc.push(
            i === 0 ? PADDING : acc[i - 1] + rowMaxHeights[i - 1] + VERTICAL_GAP
        );
        return acc;
    }, []);

    const laidOutNodes = nodes.map((node, index) => {
        const row = Math.floor(index / columns);
        const column = index % columns;

        const x = PADDING + column * (maxNodeWidth + HORIZONTAL_GAP);
        const y = rowOffsets[row];

        return {
            ...node,
            position: { x, y }
        };
    });

    return [laidOutNodes, edges];
};

const applySelectedStyle = (
    nodes: Array<TableNodeType>,
    edges: Array<TableEdgeType>,
    selectedNode: TableNodeType
): [Array<TableNodeType>, Array<TableEdgeType>] => {
    const selectedId = selectedNode?.id;

    const styledEdges = edges?.map((edge) => {
        const isSelected = edge?.source === selectedId;
        return {
            ...edge,
            labelBgStyle: {
                stroke: isSelected
                    ? "var(--kumo-grey-950)"
                    : "var(--color-neutral-400)",
                strokeWidth: isSelected ? 2 : 1,
                zIndex: isSelected ? 9999 : 1
            },
            style: {
                stroke: isSelected
                    ? "var(--kumo-grey-950)"
                    : edge?.data?.isWeakEdge
                      ? "var(--color-orange-400)"
                      : "var(--color-neutral-400)",
                strokeWidth: isSelected || edge?.data?.isWeakEdge ? 2 : 1,
                zIndex: isSelected ? 9999 : 1,
                strokeDasharray: edge?.data?.confirmRequired ? "8,4" : ""
            }
        };
    });

    const filteredEdges = selectedId
        ? styledEdges?.filter((edge) => edge.source === selectedId)
        : [];

    const styledNodes = nodes?.map((node) => {
        const filteredFkeyEdges = filteredEdges
            .filter((edge) => edge?.target === node?.id)
            .map((e) => e?.targetHandle);
        return {
            ...node,
            data: {
                ...node.data,
                pkeySelected: node?.id === selectedId,
                highLightKeys: filteredFkeyEdges as string[]
            }
        };
    });

    return [styledNodes, styledEdges];
};
export interface TableGraphViewProps {
    graphData: JSGraph;
    setSelectedTable: (table: string) => void;
    tableData: Array<JSTable>;
    weakEdges: Array<EdgeStats>;
    height: number;
    width: number;
    open: boolean;
    enableEditTable?: boolean;
    gfc: GraphFlowInfo;
    setGfc: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    allTables: GraphTables;
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>;
    groups: Array<ColumnGroup>;
    setGroups: React.Dispatch<React.SetStateAction<Array<ColumnGroup>>>;
    callCreateGraph?: () => void;
    suggestLinks: () => void;
    columnsPerRow?: number;
}

function TableGraphView({
    graphData,
    setSelectedTable,
    tableData,
    weakEdges,
    height,
    width,
    open,
    enableEditTable = false,
    gfc,
    setGfc,
    allTables,
    setAllTables,
    groups,
    setGroups,
    callCreateGraph,
    suggestLinks,
    columnsPerRow = 5
}: TableGraphViewProps) {
    const [slider, setSlider] = useState<SliderInfo>({
        show: false,
        name: ""
    } as SliderInfo);

    const initialNodes = useMemo(() => {
        const allForeignKeys = getForeignKeysFromGraphData(graphData);
        return getNodesFromTableData(
            tableData,
            allForeignKeys,
            setSelectedTable
        );
    }, [tableData, graphData, setSelectedTable]);

    const initialEdges = useMemo(() => {
        const tableIds = tableData?.map(({ id }) => id);
        // NOTE: pkey should be an empty string (nothing) if it doesn't exist; not null.
        const primaryKeys = tableData?.flatMap(({ id, pkey }) =>
            pkey !== null ? `${id}.${pkey}` : `${id}.`
        );

        return getEdgesFromGraphData(
            graphData,
            tableIds,
            weakEdges,
            primaryKeys
        );
    }, [tableData, graphData, weakEdges]);

    const [initialNodesWithLayout, initialEdgesWithLayout] = useMemo(() => {
        // DagreJS handles up to ~10 nodes well; above that, use responsive grid
        if (initialNodes.length <= DAGRE_LAYOUT_NODE_THRESHOLD) {
            return applyDagreJSLayout(initialNodes, initialEdges);
        }

        return applyResponsiveGridLayout(
            initialNodes,
            initialEdges,
            width,
            columnsPerRow
        );
    }, [initialNodes, initialEdges, width, columnsPerRow]);

    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    const [showModal, setShowModal] = useState<boolean>(false);

    const handleModalClose = () => setShowModal(false);
    const handleModalOpen = () => setShowModal(true);

    const [nodes, setNodes] = useState<Array<TableNodeType>>(
        initialNodesWithLayout
    );
    const [edges, setEdges] = useState<Array<TableEdgeType>>(
        initialEdgesWithLayout
    );

    // Sync state when graph data changes (e.g., tables added/removed)
    useEffect(() => {
        setNodes(initialNodesWithLayout);
        setEdges(initialEdgesWithLayout);
    }, [initialNodesWithLayout, initialEdgesWithLayout]);

    // When a table is saved via TableSliderOverview, suggestLinks is called
    // synchronously before React processes the state updates from updateTableInfo,
    // so the closure captures stale graphFlowInfo/groups. Instead, we set this ref
    // and call the real suggestLinks in a useEffect that fires after gfc updates —
    // by then suggestLinks has been recreated with fresh deps.
    const pendingSuggestLinksRef = useRef(false);
    const scheduleSuggestLinks = useCallback(() => {
        pendingSuggestLinksRef.current = true;
    }, []);
    useEffect(() => {
        if (pendingSuggestLinksRef.current) {
            pendingSuggestLinksRef.current = false;
            suggestLinks();
        }
    }, [gfc, suggestLinks]);

    // Apply selection styles for rendering
    const [styledNodes, styledEdges] = useMemo(
        () => applySelectedStyle(nodes, edges, selectedNode as TableNodeType),
        [nodes, edges, selectedNode]
    );

    const updateTableInfo = useCallback(
        (tName: string, tArgs: KumoapiTableTableDefinition) => {
            //Append timestamp_format to targs.cols, since it was missing.
            const flowConfig: GraphFlowInfo = { ...gfc };
            const allTbls: GraphTables = { ...allTables };
            if (tArgs && tArgs.cols?.length > 0) {
                tArgs.cols.forEach((c) => (c.timestamp_format = ""));
                //Next check if pkey / time_col is not present in cols.
                //First get the selected column names
                const colNames: Array<string> = tArgs?.cols?.map(
                    (c) => c.name as string
                );
                if (tArgs?.pkey && colNames.indexOf(tArgs?.pkey) > -1 === false)
                    //pkey not present in any cols
                    tArgs.pkey = "";
                if (
                    tArgs?.time_col &&
                    colNames.indexOf(tArgs?.time_col) > -1 === false
                )
                    //time_col not present in any cols
                    tArgs.time_col = "";
                if (
                    tArgs?.end_time_col &&
                    colNames.indexOf(tArgs?.end_time_col) > -1 === false
                )
                    //end_time_col not present in any cols
                    tArgs.end_time_col = "";

                //Update gfc and allTables
                if (flowConfig && allTbls) {
                    //Both should be valid
                    flowConfig.tables[tName] =
                        tArgs as KumoapiTableTableDefinition;
                    allTbls[tName] = tArgs as KumoapiTableTableDefinition;
                    setAllTables(allTbls);
                    setGfc(flowConfig);

                    updateGraphsGroups(flowConfig, allTbls, groups, setGroups);
                }
            } else {
                //No cols selected
                if (flowConfig && allTbls) {
                    flowConfig.tables[tName].cols = [];
                    flowConfig.tables[tName].pkey = "";
                    flowConfig.tables[tName].time_col = "";
                    flowConfig.tables[tName].end_time_col = "";
                    allTbls[tName].cols = [];
                    allTbls[tName].pkey = "";
                    allTbls[tName].time_col = "";
                    allTbls[tName].end_time_col = "";
                    setAllTables(allTbls);
                    setGfc(flowConfig);

                    updateGraphsGroups(flowConfig, allTbls, groups, setGroups);
                }
            }
        },
        [allTables, gfc, groups, setAllTables, setGfc, setGroups]
    );

    const onNodesChange: OnNodesChange = useCallback(
        (changes: Array<NodeChange>) => {
            setNodes((prevNodes) => {
                const updatedNodes = applyNodeChanges(changes, prevNodes);
                const updatedNodeIds = new Set(
                    updatedNodes?.map((n: { id: string }) => n?.id)
                );
                setEdges((prevEdges) =>
                    prevEdges?.filter(
                        (edge) =>
                            updatedNodeIds.has(edge?.source) &&
                            updatedNodeIds.has(edge?.target)
                    )
                );
                return updatedNodes as TableNodeType[];
            });
        },
        [setEdges]
    );
    const onEdgesChange: OnEdgesChange = useCallback(
        (changes: Array<EdgeChange>) =>
            setEdges(applyEdgeChanges(changes, edges) as TableEdgeType[]),
        [edges]
    );
    const onConnect: OnConnect = useCallback(
        (params: Connection) =>
            setEdges((eds: Array<TableEdgeType>) => addEdge(params, eds)),
        []
    );

    function onEditTableClick(id: string) {
        setSlider({ "show": true, "name": id });
    }

    const nodeTypes = useMemo(
        () => ({
            table: (props: { data: TableGraphNodeData }) => (
                <TableGraphNode
                    {...props}
                    enableEditTable={enableEditTable ?? false}
                    onEditTableClick={onEditTableClick}
                />
            )
        }),
        [enableEditTable]
    );

    const edgeTypes = useMemo(() => ({ customEdge: TableGraphEdge }), []);

    const hasData = nodes.length !== 0;
    const proOptions = { hideAttribution: true };

    if (!open) {
        return null;
    }

    const ReactFlowTableGraph = (
        <TableGraph
            proOptions={proOptions}
            nodes={styledNodes}
            edges={styledEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            height={height}
            width={width}
            selectedNode={selectedNode}
            setSelectedNode={setSelectedNode}
            hasData={hasData}
            handleModalOpen={handleModalOpen}
        />
    );
    return (
        <>
            <div
                style={{
                    height: height,
                    width: "100%"
                }}>
                {ReactFlowTableGraph}
            </div>
            <Sheet open={showModal}>
                <SheetContent className="h-[98vh] w-[99%] rounded" side="top">
                    <div className="h-[95%] w-full">{ReactFlowTableGraph}</div>
                    <SheetFooter className="p-2 px-3">
                        <div className="flex justify-end">
                            <Button
                                variant="light-outline"
                                onClick={handleModalClose}
                                className="w-35">
                                Close
                            </Button>
                        </div>
                    </SheetFooter>
                </SheetContent>
            </Sheet>
            {slider.show && (
                <TableSliderOverview
                    name={slider?.name}
                    tableId={slider?.name}
                    graphNameOrId={graphData.name ?? graphData.id!}
                    allTablesInGraph={graphData?.tables as GraphTables}
                    onClose={() => {
                        setSlider({ show: false, name: "" });
                    }}
                    updateTable={updateTableInfo}
                    callCreateGraph={callCreateGraph}
                    suggestLinks={scheduleSuggestLinks}
                    editable={true}
                />
            )}
        </>
    );
}

export default WithErrorBoundary(TableGraphView);
