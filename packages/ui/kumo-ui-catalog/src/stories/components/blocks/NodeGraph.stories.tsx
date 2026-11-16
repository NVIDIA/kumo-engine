// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";

import { NodeGraph } from "@/components/blocks/visual-graph/node-graph";
import { GraphEdge, GraphNode } from "@/types/visual-graph";

const meta: Meta<typeof NodeGraph> = {
    title: "Blocks/NodeGraph",
    component: NodeGraph,
    tags: ["autodocs"],
    argTypes: {
        nodesArray: { control: "object" },
        linksArray: { control: "object" },
        show: { control: "boolean" }
    }
};

export default meta;

type Story = StoryObj<typeof NodeGraph>;

// Basic NodeGraph
export const Basic: Story = {
    render: function DefaultFunc() {
        const [selectedNode, setSelectedNode] = useState<object | null>(null);

        return (
            <div className="h-[600px] w-full">
                <NodeGraph
                    nodesArray={nodesArray}
                    linksArray={edgesArray}
                    onSelect={setSelectedNode}
                    show={true}
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
    }
};

// Hidden NodeGraph
export const Hidden: Story = {
    render: function DefaultFunc() {
        const [, setSelectedNode] = useState<object | null>(null);

        return (
            <div className="h-[600px] w-full">
                <NodeGraph
                    nodesArray={nodesArray}
                    linksArray={edgesArray}
                    onSelect={setSelectedNode}
                    show={false}
                />
                <p className="text-slate-500">
                    NodeGraph is hidden (show=false)
                </p>
            </div>
        );
    }
};

// Minimal NodeGraph with fewer nodes
export const Minimal: Story = {
    render: function DefaultFunc() {
        const [, setSelectedNode] = useState<object | null>(null);

        const minimalNodes = nodesArray.slice(0, 5);
        const minimalEdges = edgesArray.filter(
            (edge) =>
                minimalNodes.some((n) => n.id === edge.source) &&
                minimalNodes.some((n) => n.id === edge.target)
        );

        return (
            <div className="h-[600px] w-full">
                <NodeGraph
                    nodesArray={minimalNodes}
                    linksArray={minimalEdges}
                    onSelect={setSelectedNode}
                    show={true}
                />
            </div>
        );
    }
};

const nodesArray: GraphNode[] = [
    {
        id: "2678",
        nodeId: "node_2678",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-05-06T00:00:00",
                    stype: "timestamp",
                    score: 0.0941471159,
                    grad_score: 0.0941471159
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.059305083,
                    stype: "numerical",
                    score: 0.0180056449,
                    grad_score: 0.0180056449
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0248939246,
                    grad_score: 0.0248939246
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2678,
            totalScore: 0.1370466854
        },
        position: {
            x: 0,
            y: 0
        }
    },
    {
        id: "2679",
        nodeId: "node_2679",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-05-06T00:00:00",
                    stype: "timestamp",
                    score: 0.145210579,
                    grad_score: 0.145210579
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0508305095,
                    stype: "numerical",
                    score: 0.0050607701,
                    grad_score: 0.0050607701
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0213462096,
                    grad_score: 0.0213462096
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2679,
            totalScore: 0.1716175587
        },
        position: {
            x: 100,
            y: 100
        }
    },
    {
        id: "2680",
        nodeId: "node_2680",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-05-06T00:00:00",
                    stype: "timestamp",
                    score: 0.1765893549,
                    grad_score: 0.1765893549,
                    isXaiUpdated: true,
                    xai_prediction: 0.067,
                    xai_global_avg: 0.094900039
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0254067797,
                    stype: "numerical",
                    score: 0.0287530348,
                    grad_score: 0.0287530348,
                    isXaiUpdated: true,
                    xai_prediction: 0.095,
                    xai_global_avg: 0.0941500185
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0230132136,
                    grad_score: 0.0230132136,
                    isXaiUpdated: true,
                    xai_prediction: 0.05,
                    xai_global_avg: 0.0932778
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2680,
            totalScore: 0.2283556033
        },
        position: {
            x: 200,
            y: 200
        }
    },
    {
        id: "2681",
        nodeId: "node_2681",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.2152332813,
                    grad_score: 0.2152332813
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0135423727,
                    stype: "numerical",
                    score: 0.0090716956,
                    grad_score: 0.0090716956
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0332928933,
                    grad_score: 0.0332928933
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2681,
            totalScore: 0.2575978702
        },
        position: {
            x: 300,
            y: 300
        }
    },
    {
        id: "2682",
        nodeId: "node_2682",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.3008559048,
                    grad_score: 0.3008559048
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0081186444,
                    stype: "numerical",
                    score: 0.0299340673,
                    grad_score: 0.0299340673
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0342074595,
                    grad_score: 0.0342074595
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2682,
            totalScore: 0.3649974316
        },
        position: {
            x: 400,
            y: 400
        }
    },
    {
        id: "2683",
        nodeId: "node_2683",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.0818053707,
                    grad_score: 0.0818053707
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0108305085,
                    stype: "numerical",
                    score: 0.0026875336,
                    grad_score: 0.0026875336
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0145299258,
                    grad_score: 0.0145299258
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2683,
            totalScore: 0.0990228301
        },
        position: {
            x: 500,
            y: 500
        }
    },
    {
        id: "2684",
        nodeId: "node_2684",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.1978737563,
                    grad_score: 0.1978737563
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0135423727,
                    stype: "numerical",
                    score: 0.0085703032,
                    grad_score: 0.0085703032
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0294533316,
                    grad_score: 0.0294533316
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2684,
            totalScore: 0.2358973911
        },
        position: {
            x: 600,
            y: 600
        }
    },
    {
        id: "2685",
        nodeId: "node_2685",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.4137719572,
                    grad_score: 0.4137719572
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0406610183,
                    stype: "numerical",
                    score: 0.0130351512,
                    grad_score: 0.0130351512
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0192264523,
                    grad_score: 0.0192264523
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2685,
            totalScore: 0.4460335607
        },
        position: {
            x: 700,
            y: 700
        }
    },
    {
        id: "2686",
        nodeId: "node_2686",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.1949949265,
                    grad_score: 0.1949949265
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0338813551,
                    stype: "numerical",
                    score: 0.0093514957,
                    grad_score: 0.0093514957
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0305218138,
                    grad_score: 0.0305218138
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2686,
            totalScore: 0.2348682359
        },
        position: {
            x: 800,
            y: 800
        }
    },
    {
        id: "2687",
        nodeId: "node_2687",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2019-09-28T00:00:00",
                    stype: "timestamp",
                    score: 0.1839134991,
                    grad_score: 0.1839134991
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0243898313,
                    stype: "numerical",
                    score: 0.0078068068,
                    grad_score: 0.0078068068
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0331572108,
                    grad_score: 0.0331572108
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2687,
            totalScore: 0.2248775167
        },
        position: {
            x: 900,
            y: 900
        }
    },
    {
        id: "2688",
        nodeId: "node_2688",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2020-01-12T00:00:00",
                    stype: "timestamp",
                    score: 0.2156628221,
                    grad_score: 0.2156628221
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0254067797,
                    stype: "numerical",
                    score: 0.0273663159,
                    grad_score: 0.0273663159
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0439669378,
                    grad_score: 0.0439669378
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2688,
            totalScore: 0.2869960759
        },
        position: {
            x: 1000,
            y: 1000
        }
    },
    {
        id: "2689",
        nodeId: "node_2689",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2020-01-12T00:00:00",
                    stype: "timestamp",
                    score: 0.1979788244,
                    grad_score: 0.1979788244
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0254067797,
                    stype: "numerical",
                    score: 0.0165259782,
                    grad_score: 0.0165259782
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.021379713,
                    grad_score: 0.021379713
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2689,
            totalScore: 0.2358845156
        },
        position: {
            x: 1100,
            y: 1100
        }
    },
    {
        id: "2690",
        nodeId: "node_2690",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2020-05-17T00:00:00",
                    stype: "timestamp",
                    score: 0.1063303128,
                    grad_score: 0.1063303128
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0215932205,
                    stype: "numerical",
                    score: 0.0068952385,
                    grad_score: 0.0068952385
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0273654424,
                    grad_score: 0.0273654424
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2690,
            totalScore: 0.1405909937
        },
        position: {
            x: 1200,
            y: 1200
        }
    },
    {
        id: "2691",
        nodeId: "node_2691",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2020-05-17T00:00:00",
                    stype: "timestamp",
                    score: 0.044095777,
                    grad_score: 0.044095777
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0071864408,
                    stype: "numerical",
                    score: 0.0107964016,
                    grad_score: 0.0107964016
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.0397221595,
                    grad_score: 0.0397221595
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2691,
            totalScore: 0.0946143381
        },
        position: {
            x: 1300,
            y: 1300
        }
    },
    {
        id: "2692",
        nodeId: "node_2692",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2020-05-17T00:00:00",
                    stype: "timestamp",
                    score: 0.0566728972,
                    grad_score: 0.0566728972
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0071864408,
                    stype: "numerical",
                    score: 0.0107826702,
                    grad_score: 0.0107826702
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.04657913,
                    grad_score: 0.04657913
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2692,
            totalScore: 0.1140346974
        },
        position: {
            x: 1400,
            y: 1400
        }
    },
    {
        id: "2693",
        nodeId: "node_2693",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: "2020-05-17T00:00:00",
                    stype: "timestamp",
                    score: 0.0566728972,
                    grad_score: 0.0566728972
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: 0.0071864408,
                    stype: "numerical",
                    score: 0.0107826702,
                    grad_score: 0.0107826702
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: 2,
                    stype: "categorical",
                    score: 0.04657913,
                    grad_score: 0.04657913
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: 2693,
            totalScore: 0.1140346974
        },
        position: {
            x: 1500,
            y: 1500
        }
    },
    {
        id: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        nodeId: "node_5682",
        data: {
            cols: [
                {
                    table_name: "customers_integ",
                    name: "FN",
                    value: null,
                    stype: "categorical",
                    score: 0.0881893709,
                    grad_score: 0.0881893709,
                    isXaiUpdated: true,
                    xai_prediction: 0.064,
                    xai_global_avg: 0.09867616
                },
                {
                    table_name: "customers_integ",
                    name: "Active",
                    value: null,
                    stype: "categorical",
                    score: 0.3260853589,
                    grad_score: 0.3260853589,
                    isXaiUpdated: true,
                    xai_prediction: 0.064,
                    xai_global_avg: 0.09831616
                },
                {
                    table_name: "customers_integ",
                    name: "club_member_status",
                    value: "ACTIVE",
                    stype: "categorical",
                    score: 0.4016885459,
                    grad_score: 0.4016885459,
                    isXaiUpdated: true,
                    xai_prediction: 0.086,
                    xai_global_avg: 0.09322052
                },
                {
                    table_name: "customers_integ",
                    name: "fashion_news_frequency",
                    value: "NONE",
                    stype: "categorical",
                    score: 0.0677976608,
                    grad_score: 0.0677976608,
                    isXaiUpdated: true,
                    xai_prediction: 0.065,
                    xai_global_avg: 0.095365
                },
                {
                    table_name: "customers_integ",
                    name: "age",
                    value: 35,
                    stype: "numerical",
                    score: 0.0352676436,
                    grad_score: 0.0352676436,
                    isXaiUpdated: true,
                    xai_prediction: 0.063,
                    xai_global_avg: 0.0979578796
                }
            ],
            tableName: "customers_integ",
            idName: "customer_id",
            value: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
            totalScore: 0.9190285802
        },
        position: {
            x: 1600,
            y: 1600
        }
    },
    {
        id: "684008001",
        nodeId: "node_8208",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 684008,
                    stype: "categorical",
                    score: 0.0027429378,
                    grad_score: 0.0027429378
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_no",
                    value: 57,
                    stype: "categorical",
                    score: 0.0069395211,
                    grad_score: 0.0069395211
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Swimsuit",
                    stype: "categorical",
                    score: 0.0089465762,
                    grad_score: 0.0089465762
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Swimwear",
                    stype: "categorical",
                    score: 0.0032772797,
                    grad_score: 0.0032772797
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "fully lined swimsuit in fast-drying functional fabric . narrow cut at the top with a zip at the front and mesh details at the sides .",
                    stype: "text",
                    score: 0.0234977286,
                    grad_score: 0.0234977286
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 684008001,
            totalScore: 0.1005926493
        },
        position: {
            x: 1700,
            y: 1700
        }
    },
    {
        id: "684824001",
        nodeId: "node_8209",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 684824,
                    stype: "categorical",
                    score: 0.0013551611,
                    grad_score: 0.0013551611
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Swimsuit",
                    stype: "categorical",
                    score: 0.0116261709,
                    grad_score: 0.0116261709
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Swimwear",
                    stype: "categorical",
                    score: 0.0015084075,
                    grad_score: 0.0015084075
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: null,
                    stype: "text",
                    score: 0.02765215,
                    grad_score: 0.02765215
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 684824001,
            totalScore: 0.1169056636
        },
        position: {
            x: 1800,
            y: 1800
        }
    },
    {
        id: "735402002",
        nodeId: "node_8210",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 735402,
                    stype: "categorical",
                    score: 0.0065284474,
                    grad_score: 0.0065284474
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Swimsuit",
                    stype: "categorical",
                    score: 0.0081733027,
                    grad_score: 0.0081733027
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "fully lined , patterned swimsuit with a deep neckline at the back .",
                    stype: "text",
                    score: 0.0278433058,
                    grad_score: 0.0278433058
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 735402002,
            totalScore: 0.0929837278
        },
        position: {
            x: 1900,
            y: 1900
        }
    },
    {
        id: "761204001",
        nodeId: "node_8211",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 761204,
                    stype: "categorical",
                    score: 0.0026161738,
                    grad_score: 0.0026161738
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Slippers",
                    stype: "categorical",
                    score: 0.0062957974,
                    grad_score: 0.0062957974
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Shoes",
                    stype: "categorical",
                    score: 0.0016808653,
                    grad_score: 0.0016808653
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 761204001,
            totalScore: 0.0855928365
        },
        position: {
            x: 2000,
            y: 2000
        }
    },
    {
        id: "833993001",
        nodeId: "node_8212",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 833993,
                    stype: "categorical",
                    score: 0.0047667995,
                    grad_score: 0.0047667995
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Earring",
                    stype: "categorical",
                    score: 0.0006111241,
                    grad_score: 0.0006111241
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Accessories",
                    stype: "categorical",
                    score: 0.0064837961,
                    grad_score: 0.0064837961
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "long metal hoop earrings with a rod-shaped pendant .",
                    stype: "text",
                    score: 0.0438879021,
                    grad_score: 0.0438879021
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 833993001,
            totalScore: 0.1451402502
        },
        position: { x: 2100, y: 2100 }
    },
    {
        id: "834084001",
        nodeId: "node_8213",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 834084,
                    stype: "categorical",
                    score: 0.0057180719,
                    grad_score: 0.0057180719
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Earring",
                    stype: "categorical",
                    score: 0.0018487526,
                    grad_score: 0.0018487526
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Accessories",
                    stype: "categorical",
                    score: 0.0040205088,
                    grad_score: 0.0040205088
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "earrings in metal with plastic beads in various sizes and designs . length 9 . 5 cm .",
                    stype: "text",
                    score: 0.04056729,
                    grad_score: 0.04056729
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 834084001,
            totalScore: 0.185455259
        },
        position: { x: 2200, y: 2200 }
    },
    {
        id: "828130001",
        nodeId: "node_8214",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 828130,
                    stype: "categorical",
                    score: 0.0087037664,
                    grad_score: 0.0087037664
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Hair clip",
                    stype: "categorical",
                    score: 0.0174519252,
                    grad_score: 0.0174519252
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Accessories",
                    stype: "categorical",
                    score: 0.0118083358,
                    grad_score: 0.0118083358
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "hair grips in metal decorated with sparkly stones in various colours . length 8 . 5 cm .",
                    stype: "text",
                    score: 0.0547854155,
                    grad_score: 0.0547854155
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 828130001,
            totalScore: 0.2151461747
        },
        position: { x: 2300, y: 2300 }
    },
    {
        id: "754323004",
        nodeId: "node_8215",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 754323,
                    stype: "categorical",
                    score: 0.0034055291,
                    grad_score: 0.0034055291
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Sweater",
                    stype: "categorical",
                    score: 0.0030158705,
                    grad_score: 0.0030158705
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Garment Upper body",
                    stype: "categorical",
                    score: 0.0008713625,
                    grad_score: 0.0008713625
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "jumper in a knit made partly from recycled polyester with a rib-knit polo neck .",
                    stype: "text",
                    score: 0.0515521951,
                    grad_score: 0.0515521951
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 754323004,
            totalScore: 0.1941692721
        },
        position: { x: 2400, y: 2400 }
    },
    {
        id: "730683003",
        nodeId: "node_8216",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 730683,
                    stype: "categorical",
                    score: 0.0128546655,
                    grad_score: 0.0128546655
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Leggings/Tights",
                    stype: "categorical",
                    score: 0.0033279005,
                    grad_score: 0.0033279005
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Garment Lower body",
                    stype: "categorical",
                    score: 0.0021238353,
                    grad_score: 0.0021238353
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "sports tights in fast-drying functional fabric with a high waist .",
                    stype: "text",
                    score: 0.049097307,
                    grad_score: 0.049097307
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 730683003,
            totalScore: 0.1879902572
        },
        position: { x: 2500, y: 2500 }
    },
    {
        id: "742079001",
        nodeId: "node_8217",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 742079,
                    stype: "categorical",
                    score: 0.0045514526,
                    grad_score: 0.0045514526
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Bra",
                    stype: "categorical",
                    score: 0.0047218669,
                    grad_score: 0.0047218669
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Underwear",
                    stype: "categorical",
                    score: 0.0029858379,
                    grad_score: 0.0029858379
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "sports bra in fast-drying functional fabric with a v-neck .",
                    stype: "text",
                    score: 0.0235783197,
                    grad_score: 0.0235783197
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 742079001,
            totalScore: 0.0885876443
        },
        position: { x: 2600, y: 2600 }
    },
    {
        id: "810825003",
        nodeId: "node_8218",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 810825,
                    stype: "categorical",
                    score: 0.0167981535,
                    grad_score: 0.0167981535
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Bra",
                    stype: "categorical",
                    score: 0.0122826491,
                    grad_score: 0.0122826491
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Underwear",
                    stype: "categorical",
                    score: 0.0086725485,
                    grad_score: 0.0086725485
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "lace bra with underwired , padded cups that shape the bust and provide good support .",
                    stype: "text",
                    score: 0.0651678815,
                    grad_score: 0.0651678815
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 810825003,
            totalScore: 0.3220826235
        },
        position: { x: 2700, y: 2700 }
    },
    {
        id: "810557001",
        nodeId: "node_8219",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 810557,
                    stype: "categorical",
                    score: 0.0045216982,
                    grad_score: 0.0045216982
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Bra",
                    stype: "categorical",
                    score: 0.0027494328,
                    grad_score: 0.0027494328
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Underwear",
                    stype: "categorical",
                    score: 0.0080748228,
                    grad_score: 0.0080748228
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "underwired bra in lace with removable inserts that shape the bust and provide good support .",
                    stype: "text",
                    score: 0.0521884933,
                    grad_score: 0.0521884933
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 810557001,
            totalScore: 0.1749314263
        },
        position: { x: 2800, y: 2800 }
    },
    {
        id: "880777002",
        nodeId: "node_8220",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 880777,
                    stype: "categorical",
                    score: 0.00158719,
                    grad_score: 0.00158719
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Swimwear bottom",
                    stype: "categorical",
                    score: 0.0033690832,
                    grad_score: 0.0033690832
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Swimwear",
                    stype: "categorical",
                    score: 0.0007669312,
                    grad_score: 0.0007669312
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "fully lined bikini bottoms with cutaway coverage at the back .",
                    stype: "text",
                    score: 0.0566180572,
                    grad_score: 0.0566180572
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 880777002,
            totalScore: 0.152465686
        },
        position: { x: 2900, y: 2900 }
    },
    {
        id: "620963002",
        nodeId: "node_8221",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 620963,
                    stype: "categorical",
                    score: 0.0091253091,
                    grad_score: 0.0091253091
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Bikini top",
                    stype: "categorical",
                    score: 0.0061816322,
                    grad_score: 0.0061816322
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Swimwear",
                    stype: "categorical",
                    score: 0.0012745905,
                    grad_score: 0.0012745905
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "lined triangle bikini top with removable inserts .",
                    stype: "text",
                    score: 0.038388066,
                    grad_score: 0.038388066
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 620963002,
            totalScore: 0.1438895271
        },
        position: { x: 3000, y: 3000 }
    },
    {
        id: "742912002",
        nodeId: "node_8222",
        data: {
            cols: [
                {
                    table_name: "articles_integ",
                    name: "product_code",
                    value: 742912,
                    stype: "categorical",
                    score: 0.0019087306,
                    grad_score: 0.0019087306
                },
                {
                    table_name: "articles_integ",
                    name: "product_type_name",
                    value: "Bikini top",
                    stype: "categorical",
                    score: 0.004256228,
                    grad_score: 0.004256228
                },
                {
                    table_name: "articles_integ",
                    name: "product_group_name",
                    value: "Swimwear",
                    stype: "categorical",
                    score: 0.0000961094,
                    grad_score: 0.0000961094
                },
                {
                    table_name: "articles_integ",
                    name: "detail_desc",
                    value: "lined triangle bikini top with removable inserts and ties at the back .",
                    stype: "text",
                    score: 0.0783174857,
                    grad_score: 0.0783174857
                }
            ],
            tableName: "articles_integ",
            idName: "article_id",
            value: 742912002,
            totalScore: 0.2747720079
        },
        position: { x: 3100, y: 3100 }
    },
    {
        id: "<AGGREGATED>_1",
        nodeId: "node_-1",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: null,
                    stype: "timestamp",
                    score: 1.8310275078,
                    grad_score: 1.8310275078,
                    isXaiUpdated: true,
                    xai_prediction: 0.058,
                    xai_global_avg: 0.094900039
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: null,
                    stype: "numerical",
                    score: 0.1754862964,
                    grad_score: 0.1754862964,
                    isXaiUpdated: true,
                    xai_prediction: 0.056,
                    xai_global_avg: 0.0941500185
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: null,
                    stype: "categorical",
                    score: 0.6127763391,
                    grad_score: 0.6127763391,
                    isXaiUpdated: true
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: "<AGGREGATED>",
            totalScore: 2.6192901433
        },
        position: { x: 3200, y: 3200 }
    },
    {
        id: "<AGGREGATED>_2",
        nodeId: "node_-2",
        data: {
            cols: [
                {
                    table_name: "transactions_integ",
                    name: "t_dat",
                    value: null,
                    stype: "timestamp",
                    score: 1.8310275078,
                    grad_score: 1.8310275078
                },
                {
                    table_name: "transactions_integ",
                    name: "price",
                    value: null,
                    stype: "numerical",
                    score: 0.1754862964,
                    grad_score: 0.1754862964
                },
                {
                    table_name: "transactions_integ",
                    name: "sales_channel_id",
                    value: null,
                    stype: "categorical",
                    score: 0.6127763391,
                    grad_score: 0.6127763391
                }
            ],
            tableName: "transactions_integ",
            idName: "",
            value: "<AGGREGATED>",
            totalScore: 2.6192901433
        },
        position: { x: 3300, y: 3300 }
    }
];

const edgesArray: GraphEdge[] = [
    { id: "e-1", source: "2678", target: "684008001" },
    { id: "e-2", source: "2679", target: "684824001" },
    { id: "e-3", source: "2680", target: "735402002" },
    { id: "e-4", source: "2681", target: "761204001" },
    { id: "e-5", source: "2682", target: "833993001" },
    { id: "e-6", source: "2683", target: "834084001" },
    { id: "e-7", source: "2684", target: "828130001" },
    { id: "e-8", source: "2685", target: "754323004" },
    { id: "e-9", source: "2686", target: "730683003" },
    { id: "e-10", source: "2687", target: "742079001" },
    { id: "e-11", source: "2688", target: "810825003" },
    { id: "e-12", source: "2689", target: "810557001" },
    { id: "e-13", source: "2690", target: "880777002" },
    { id: "e-14", source: "2691", target: "620963002" },
    { id: "e-15", source: "2693", target: "742912002" },
    {
        id: "e-16",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2678"
    },
    {
        id: "e-17",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2679"
    },
    {
        id: "e-18",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2680"
    },
    {
        id: "e-19",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2681"
    },
    {
        id: "e-20",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2682"
    },
    {
        id: "e-21",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2683"
    },
    {
        id: "e-22",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2684"
    },
    {
        id: "e-23",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2685"
    },
    {
        id: "e-24",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2686"
    },
    {
        id: "e-25",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2687"
    },
    {
        id: "e-26",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2688"
    },
    {
        id: "e-27",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2689"
    },
    {
        id: "e-28",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2690"
    },
    {
        id: "e-29",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2691"
    },
    {
        id: "e-30",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2692"
    },
    {
        id: "e-31",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "2693"
    },
    {
        id: "e-32",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "<AGGREGATED>_1"
    },
    {
        id: "e-33",
        source: "b11b5b39412eb2e9918b512d5d4db4522d04cba9479023cf1e7f92d5d9575a59",
        target: "<AGGREGATED>_2"
    }
];
