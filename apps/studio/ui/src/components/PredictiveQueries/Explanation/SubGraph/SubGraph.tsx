// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import {
    Alert,
    Card,
    CardContent,
    CardHeader,
    ColumnDef,
    DataTable,
    EmptyTable,
    NodeGraph,
    Row,
    Skeleton,
    Symbol,
    Title,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { KumoapiExplainSubgraphSubgraph } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { SubgraphColumns } from "components/@ui/TableColumns/Subgraph";

import {
    GraphEdge,
    GraphNode,
    GraphNodeData,
    GraphXaiScores,
    XaiColumnArgs
} from "types/explorer";
import { ColumnArgs } from "types/tables";
import { SubgraphSummary } from "types/training_job";

import {
    createLinksFromEdges,
    extractNodesFromSubgraph,
    updateColumnsWithXaiData
} from "utils/Converters/explorerConverter";
import { getKeys } from "utils/Helpers/objectHelpers";
import { customSort } from "utils/Helpers/sortHelpers";
import { isValid } from "utils/Helpers/typeGuards";

export interface SubgraphProps {
    subgraph: KumoapiExplainSubgraphSubgraph | undefined;
    isDataLoading: boolean;
    xaiData?: GraphXaiScores;
    anchorTime: string;
    show: boolean;
    setActiveSubGraph?: (v: GraphNodeData | null) => void;
    entityId: string;
}

const SubGraph = ({
    subgraph,
    isDataLoading,
    xaiData,
    anchorTime,
    show,
    setActiveSubGraph,
    entityId
}: SubgraphProps) => {
    const [nodes, setNodesArr] = useState<GraphNode[]>([]);
    const [links, setLinksArr] = useState<GraphEdge[]>([]);

    const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(
        null
    );

    const setSelected = useCallback(
        (v: GraphNodeData | null) => {
            setSelectedNode(v);
            if (setActiveSubGraph && v) {
                setActiveSubGraph(v);
            }
        },
        [setActiveSubGraph]
    );

    useEffect(() => {
        if (subgraph) {
            const { nodeArr } = extractNodesFromSubgraph(subgraph);
            const links = createLinksFromEdges(
                subgraph?.subgraph_store?.edges,
                nodeArr
            );
            setNodesArr(nodeArr);
            setLinksArr(links);
        }
    }, [subgraph]);

    useEffect(() => {
        if (selectedNode?.cols && selectedNode?.cols?.length > 0) {
            if (!selectedNode?.cols[0]?.isXaiUpdated) {
                const tables = getKeys(xaiData);
                if (tables?.length > 0) {
                    const newCols = updateColumnsWithXaiData(
                        selectedNode?.cols,
                        selectedNode?.tableName,
                        xaiData as GraphXaiScores,
                        anchorTime
                    );

                    setSelected({ ...selectedNode, cols: newCols });
                }
            }
        }
    }, [anchorTime, selectedNode, xaiData, setSelected]);

    const tableName = useMemo(() => {
        if (selectedNode?.tableName?.includes(".")) {
            return selectedNode?.tableName?.split(".")?.pop();
        } else if (selectedNode?.tableName) {
            return selectedNode?.tableName;
        } else {
            return "";
        }
    }, [selectedNode?.tableName]);

    const handleSelect = useCallback(
        (node: GraphNodeData | null) => {
            setSelected(node);
        },
        [setSelected]
    );

    if (isDataLoading) {
        return <Skeleton className="w-160 h-20" />;
    }

    if (nodes?.length <= 1) {
        const isValidNode =
            nodes?.length === 1 ? isValid(nodes[0]?.data?.idName) : false;
        if (!isValidNode || nodes?.length < 1)
            return (
                <Alert variant="default" enableClose={false}>
                    Input graph data is not available for this entity.
                </Alert>
            );
    }

    return (
        <>
            <Card className="p-0 rounded-tl-none">
                <div className="flex flex-row">
                    <div className="p-0 m-0 overflow-auto max-w-[33vw] w-1/2">
                        <CardHeader
                            className="p-4 rounded-none h-12 bg-surface-grey"
                            bottomSeparator={true}>
                            <Title
                                variant="section-header"
                                children={"Subgraph"}
                                className="text-title-secondary"
                            />
                        </CardHeader>
                        <CardContent className="p-1">
                            <NodeGraph
                                key={entityId}
                                nodesArray={nodes}
                                linksArray={links}
                                onSelect={handleSelect}
                                show={show}
                            />
                        </CardContent>
                    </div>
                    <div className="flex-1 min-w-0 border-l border-[rgb(227,235,246)]">
                        {selectedNode !== null ? (
                            <>
                                <CardHeader className="px-6 py-3 h-12 rounded-none bg-surface-grey">
                                    <label className="text-title-primary font-medium text-sm leading-5">
                                        <div className="text-kumo-pink-800 flex justify-items-center justify-between">
                                            {tableName}
                                            <span className="text-title-secondary text-right">
                                                Selected table
                                            </span>
                                        </div>
                                    </label>
                                </CardHeader>
                                <CardContent className="p-0">
                                    <GraphTableView
                                        selectedNode={selectedNode}
                                    />
                                </CardContent>
                            </>
                        ) : (
                            <div className="flex items-center justify-center h-full text-title-secondary text-sm">
                                Select a node in the graph to view its details.
                            </div>
                        )}
                    </div>
                </div>
            </Card>
        </>
    );
};

export interface GraphTableViewProps {
    selectedNode: GraphNodeData | null;
}

export const GraphTableView = ({ selectedNode }: GraphTableViewProps) => {
    const scoreIncluded = useMemo(
        () =>
            selectedNode?.cols?.some(
                (col: ColumnArgs) => col?.grad_score !== null
            ),
        [selectedNode]
    );

    const columns = useMemo(
        () => [
            {
                id: "name",
                header: "Column",
                accessorKey: "name"
            },
            {
                id: "value",
                header: "Value",
                accessorKey: "value",
                cell: ({ getValue }: { getValue: () => string | number }) => {
                    const value = getValue();
                    return value?.toString()?.length > 15 ? (
                        <Tooltip id={`${value}`}>
                            <TooltipTrigger>
                                <span>
                                    {value.toString().substring(0, 15) + "..."}
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                arrow={false}
                                side={"bottom"}
                                sideOffset={5}>
                                {value?.toString()}
                            </TooltipContent>
                        </Tooltip>
                    ) : (
                        <span>
                            {(value ?? null) === null ? value : String(value)}
                        </span>
                    );
                }
            },
            ...(scoreIncluded
                ? [
                      {
                          id: "grad_score",
                          header: "Score",
                          accessorKey: "grad_score",
                          cell: ({ getValue }: { getValue: () => number }) => {
                              const value = getValue();
                              return isValid(value) ? (
                                  <>{value?.toFixed(2)}</>
                              ) : (
                                  <Symbol type="emdash" />
                              );
                          },
                          sortingFn: (
                              a: Row<ColumnArgs & XaiColumnArgs>,
                              b: Row<ColumnArgs & XaiColumnArgs>
                          ) => customSort(b, a, "grad_score")
                      }
                  ]
                : [])
        ],
        [scoreIncluded]
    );

    const data = useMemo(() => {
        if (selectedNode?.cols) {
            const dataRows = [...selectedNode.cols];
            if (selectedNode?.idName && selectedNode?.value) {
                dataRows?.unshift({
                    name: selectedNode?.idName,
                    value: selectedNode?.value,
                    grad_score: undefined
                });
            }
            // Filter data for extra secure
            return dataRows?.filter((d) => isValid(d));
        }
        return [];
    }, [selectedNode]);

    return (
        <>
            {data?.length > 0 ? (
                <DataTable
                    id="subgraph-node-table"
                    disableRowSelectionHover={true}
                    columns={
                        columns as ColumnDef<
                            ColumnArgs & XaiColumnArgs,
                            unknown
                        >[]
                    }
                    data={data}
                    enableGlobalFilter={false}
                    defaultPageSize={10}
                    pageCount={
                        data?.length > 0 ? Math.ceil(data?.length / 10) : 1
                    }
                    showTopRoundedBorder={false}
                />
            ) : (
                <EmptyTable id="no-data" label={"No data found!"} />
            )}
        </>
    );
};

export interface SubgraphSumaryTableProps {
    subgraphSummary: Array<SubgraphSummary>;
}

export const SubgraphSumaryTable = ({
    subgraphSummary
}: SubgraphSumaryTableProps) => {
    const columns: ColumnDef<SubgraphSummary>[] = SubgraphColumns();

    return (
        <>
            {subgraphSummary?.length > 0 ? (
                <DataTable
                    id="subgraph-node-table"
                    columns={columns}
                    disableRowSelectionHover={true}
                    data={subgraphSummary}
                    enableGlobalFilter={false}
                />
            ) : (
                <EmptyTable
                    id={"subgraph-node-empty-table"}
                    label="No data found!"
                />
            )}
        </>
    );
};

export default WithErrorBoundary(SubGraph);
