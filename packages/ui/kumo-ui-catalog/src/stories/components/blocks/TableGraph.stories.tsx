// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { useCallback, useMemo, useState } from "react";

import {
    addEdge,
    applyEdgeChanges,
    applyNodeChanges,
    Connection,
    Edge,
    EdgeChange,
    Node,
    NodeChange
} from "@xyflow/react";

import {
    TableGraph,
    TableGraphEdge,
    TableGraphNode
} from "@/components/blocks/visual-graph/table-graph";
import { TableGraphNodeData } from "@/types/visual-graph";

// ============================================================================
// TableGraph Stories
// ============================================================================

const tableGraphMeta: Meta<typeof TableGraph> = {
    title: "Blocks/TableGraph",
    component: TableGraph,
    tags: ["autodocs"],
    argTypes: {
        nodes: { control: "object" },
        edges: { control: "object" },
        hasData: { control: "boolean" }
    }
};

export default tableGraphMeta;

type TableGraphStory = StoryObj<typeof TableGraph>;

// Sample data for stories
const sampleNodes: Node[] = [
    {
        id: "transactions_1",
        position: { x: 309, y: 189 },
        data: {
            id: "transactions_1",
            label: "transactions_1",
            cols: [
                {
                    id: "t_dat",
                    name: "t_dat",
                    value: 1,
                    stype: "timestamp",
                    dtype: "date"
                },
                {
                    id: "customer_id",
                    name: "customer_id",
                    value: 1,
                    stype: "ID",
                    dtype: "string"
                },
                {
                    id: "article_id",
                    name: "article_id",
                    value: 1,
                    stype: "ID",
                    dtype: "int64"
                },
                {
                    id: "price",
                    name: "price",
                    value: 1,
                    stype: "numerical",
                    dtype: "float64"
                }
            ],
            position: { x: 400, y: 0 },
            pkey: null,
            timeCol: "t_dat",
            earliestFact: "-",
            latestFact: "-",
            foreignKeys: ["customer_id", "article_id"],
            pkeySelected: false,
            highLightKeys: [],
            setSelectedTable: (id: string) =>
                console.log("View table clicked:", id)
        },
        type: "table",
        className: "light",
        measured: { width: 252, height: 186 },
        selected: true
    },
    {
        id: "customers_1",
        position: { x: 17, y: 0 },
        data: {
            id: "customers_1",
            label: "customers_1",
            cols: [
                {
                    id: "customer_id",
                    name: "customer_id",
                    value: 1,
                    stype: "ID",
                    dtype: "string"
                },
                {
                    id: "FN",
                    name: "FN",
                    value: 1,
                    stype: "numerical",
                    dtype: "float64"
                },
                {
                    id: "Active",
                    name: "Active",
                    value: 1,
                    stype: "numerical",
                    dtype: "float64"
                },
                {
                    id: "club_member_status",
                    name: "club_member_status",
                    value: 1,
                    stype: "categorical",
                    dtype: "string"
                },
                {
                    id: "fashion_news_frequency",
                    name: "fashion_news_frequency",
                    value: 1,
                    stype: "categorical",
                    dtype: "string"
                },
                {
                    id: "age",
                    name: "age",
                    value: 1,
                    stype: "numerical",
                    dtype: "float64"
                },
                {
                    id: "postal_code",
                    name: "postal_code",
                    value: 1,
                    stype: "ID",
                    dtype: "string"
                }
            ],
            position: { x: -199.99999999999991, y: 346.4101615137755 },
            pkey: "customer_id",
            timeCol: null,
            earliestFact: "-",
            latestFact: "-",
            foreignKeys: [],
            pkeySelected: false,
            highLightKeys: [],
            setSelectedTable: (id: string) =>
                console.log("View table clicked:", id)
        },
        type: "table",
        className: "light",
        measured: { width: 252, height: 106 }
    },
    {
        id: "articles_1",
        position: { x: 0, y: 202 },
        data: {
            id: "articles_1",
            label: "articles_1",
            cols: [
                {
                    id: "article_id",
                    name: "article_id",
                    value: 1,
                    stype: "ID",
                    dtype: "int64"
                },
                {
                    id: "product_code",
                    name: "product_code",
                    value: 1,
                    stype: "ID",
                    dtype: "int64"
                },
                {
                    id: "prod_name",
                    name: "prod_name",
                    value: 1,
                    stype: "categorical",
                    dtype: "string"
                },
                {
                    id: "product_type_no",
                    name: "product_type_no",
                    value: 1,
                    stype: "ID",
                    dtype: "int64"
                },
                {
                    id: "product_type_name",
                    name: "product_type_name",
                    value: 1,
                    stype: "categorical",
                    dtype: "string"
                }
            ],
            position: { x: -200.00000000000017, y: -346.41016151377534 },
            pkey: "article_id",
            timeCol: null,
            earliestFact: "-",
            latestFact: "-",
            foreignKeys: [],
            pkeySelected: false,
            highLightKeys: [],
            setSelectedTable: (id: string) =>
                console.log("View table clicked:", id)
        },
        type: "table",
        className: "light",
        measured: { width: 252, height: 106 }
    }
];

const sampleEdges: Edge[] = [
    {
        id: "customers_1.customer_id <-> transactions_1.customer_id",
        source: "customers_1",
        sourceHandle: "source_PK: customer_id",
        target: "transactions_1",
        targetHandle: "target_FK: customer_id",
        type: "customEdge",
        data: { confirmRequired: false, isWeakEdge: false }
    },
    {
        id: "articles_1.article_id <-> transactions_1.article_id",
        source: "articles_1",
        sourceHandle: "source_PK: article_id",
        target: "transactions_1",
        targetHandle: "target_FK: article_id",
        type: "customEdge",
        data: { confirmRequired: false, isWeakEdge: false }
    }
];

// Helper component for TableGraph stories
const TableGraphWrapper = ({
    initialNodes,
    initialEdges,
    enableEditTable = false
}: {
    initialNodes: Node[];
    initialEdges: Edge[];
    enableEditTable?: boolean;
}) => {
    const [nodes, setNodes] = useState<Node[]>(initialNodes);
    const [edges, setEdges] = useState<Edge[]>(initialEdges);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    const onNodesChange = useCallback(
        (changes: NodeChange[]) =>
            setNodes((nds) => applyNodeChanges(changes, nds)),
        []
    );

    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) =>
            setEdges((eds) => applyEdgeChanges(changes, eds)),
        []
    );

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        []
    );

    const nodeTypes = useMemo(
        () => ({
            table: (props: { data: TableGraphNodeData }) => (
                <TableGraphNode
                    {...props}
                    enableEditTable={enableEditTable}
                    onEditTableClick={(id) =>
                        console.log("Edit table clicked:", id)
                    }
                />
            )
        }),
        [enableEditTable]
    );

    const edgeTypes = useMemo(() => ({ customEdge: TableGraphEdge }), []);

    return (
        <div className="h-[600px] w-full">
            <TableGraph
                proOptions={{ hideAttribution: true }}
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                height={600}
                width={800}
                selectedNode={selectedNode}
                setSelectedNode={setSelectedNode}
                hasData={nodes.length > 0}
                handleModalOpen={() => console.log("Modal opened")}
            />
            {selectedNode && (
                <div className="mt-4 rounded border p-4">
                    <h3 className="font-semibold">Selected Node:</h3>
                    <pre className="max-h-40 overflow-auto text-xs">
                        {JSON.stringify(selectedNode, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );
};

// Basic TableGraph story
export const Basic: TableGraphStory = {
    render: function DefaultFunc() {
        return (
            <TableGraphWrapper
                initialNodes={sampleNodes}
                initialEdges={sampleEdges}
            />
        );
    }
};

// TableGraph with edit enabled
export const WithEditEnabled: TableGraphStory = {
    render: function DefaultFunc() {
        return (
            <TableGraphWrapper
                initialNodes={sampleNodes}
                initialEdges={sampleEdges}
                enableEditTable={true}
            />
        );
    }
};

// TableGraph with weak edges
// The TableGraphEdge component automatically applies styling based on data.isWeakEdge:
// - Weak edge: #E1A42F (orange/amber), strokeWidth: 2
// - Healthy edge: ReactFlow default (gray)
export const WithWeakEdges: TableGraphStory = {
    render: function DefaultFunc() {
        const weakEdges: Edge[] = [
            {
                ...sampleEdges[0],
                data: { confirmRequired: false, isWeakEdge: true }
            },
            {
                ...sampleEdges[1],
                data: { confirmRequired: false, isWeakEdge: false }
            }
        ];

        return (
            <TableGraphWrapper
                initialNodes={sampleNodes}
                initialEdges={weakEdges}
            />
        );
    }
};

// Empty TableGraph
export const Empty: TableGraphStory = {
    render: function DefaultFunc() {
        return <TableGraphWrapper initialNodes={[]} initialEdges={[]} />;
    }
};

const TableGraphWithViewTrack = () => {
    const [viewedTable, setViewedTable] = useState<string | null>(null);

    const trackedNodes: Node[] = sampleNodes.map((n) => ({
        ...n,
        data: {
            ...(n.data as TableGraphNodeData),
            setSelectedTable: (id: string) => setViewedTable(id)
        }
    }));

    const [nodes, setNodes] = useState<Node[]>(trackedNodes);
    const [edges, setEdges] = useState<Edge[]>(sampleEdges);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    const onNodesChange = useCallback(
        (changes: NodeChange[]) =>
            setNodes((nds) => applyNodeChanges(changes, nds)),
        []
    );
    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) =>
            setEdges((eds) => applyEdgeChanges(changes, eds)),
        []
    );
    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        []
    );

    const nodeTypes = useMemo(
        () => ({
            table: (props: { data: TableGraphNodeData }) => (
                <TableGraphNode
                    {...props}
                    enableEditTable={false}
                    onEditTableClick={(id) =>
                        console.log("Edit table clicked:", id)
                    }
                />
            )
        }),
        []
    );
    const edgeTypes = useMemo(() => ({ customEdge: TableGraphEdge }), []);

    return (
        <div className="flex h-[600px] w-full flex-col gap-2">
            <TableGraph
                proOptions={{ hideAttribution: true }}
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                height={540}
                width={800}
                selectedNode={selectedNode}
                setSelectedNode={setSelectedNode}
                hasData={true}
                handleModalOpen={() => console.log("Modal opened")}
            />
            {viewedTable && (
                <div className="rounded border p-2 text-sm">
                    View table clicked: <strong>{viewedTable}</strong>
                </div>
            )}
        </div>
    );
};

export const WithViewTableTracking: TableGraphStory = {
    render: function DefaultFunc() {
        return <TableGraphWithViewTrack />;
    }
};

const TableGraphWithEditTrack = () => {
    const [editedTable, setEditedTable] = useState<string | null>(null);

    const [nodes, setNodes] = useState<Node[]>(sampleNodes);
    const [edges, setEdges] = useState<Edge[]>(sampleEdges);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    const onNodesChange = useCallback(
        (changes: NodeChange[]) =>
            setNodes((nds) => applyNodeChanges(changes, nds)),
        []
    );
    const onEdgesChange = useCallback(
        (changes: EdgeChange[]) =>
            setEdges((eds) => applyEdgeChanges(changes, eds)),
        []
    );
    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        []
    );

    const nodeTypes = useMemo(
        () => ({
            table: (props: { data: TableGraphNodeData }) => (
                <TableGraphNode
                    {...props}
                    enableEditTable={true}
                    onEditTableClick={(id) => setEditedTable(id)}
                />
            )
        }),
        []
    );
    const edgeTypes = useMemo(() => ({ customEdge: TableGraphEdge }), []);

    return (
        <div className="flex h-[600px] w-full flex-col gap-2">
            <TableGraph
                proOptions={{ hideAttribution: true }}
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                height={540}
                width={800}
                selectedNode={selectedNode}
                setSelectedNode={setSelectedNode}
                hasData={true}
                handleModalOpen={() => console.log("Modal opened")}
            />
            {editedTable && (
                <div className="rounded border p-2 text-sm">
                    Edit table clicked: <strong>{editedTable}</strong>
                </div>
            )}
        </div>
    );
};

export const WithEditTableTracking: TableGraphStory = {
    render: function DefaultFunc() {
        return <TableGraphWithEditTrack />;
    }
};
