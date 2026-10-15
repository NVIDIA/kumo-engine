// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GeneratePredictionTableResponse,
    GenerateTrainTableResponse,
    JobSource,
    KumoCoreTypesEnumsTaskType
} from "gen/v0/types";
import {
    BatchPredictionJobResponse,
    GraphResourceResponse,
    KumoapiJobsPredictionArtifactType
} from "gen/v1/types";

import { GraphNode, SubgraphEdge } from "types/explorer";
import { JSGraph } from "types/graphs";
import { TrainingJobResults } from "types/predictiveQueries";
import { DataSourceType } from "types/tables";

import { jobStatus } from "utils/Constants";

export const bpWfResponse = {
    id: "bp-job-00054000000000000000000000000000",
    predictive_query_link: {
        id: "NA-0004c000000000000000000000000000",
        name: "NA-0004c000000000000000000000000000",
        training_job_id: null
    },
    timestamps: {
        created_at: "2025-06-05T04:36:30.038752+00:00",
        updated_at: "2025-06-05T04:52:37.719202+00:00"
    },
    jobs: [
        {
            id: "bp-job-0001e000000000000000000000000000",
            config: {
                predictive_query_id: "NA-00027000000000000000000000000000",
                training_job_id: "trainingjob-00046000000000000000000000000000",
                task_type: "binary_classification",
                binary_classification_threshold: 0.6,
                num_parallel_workers_limit: 2,
                outputs: [],
                generate_pred_table_job_id:
                    "gen-predtable-job-00060000000000000000000000000000",
                graph_id: "NA-0005b000000000000000000000000000",
                graph_data_version: "00033000000000000000000000000000",
                pquery_def_fpid: "00027000000000000000000000000000",
                explanations: false
            },
            timestamps: {
                created_at: "2025-04-09T05:19:10.268477+00:00",
                updated_at: "2025-04-09T05:27:46.421947+00:00"
            },
            predictive_query_link: {
                id: "NA-00027000000000000000000000000000",
                name: "NA-00027000000000000000000000000000",
                training_job_id: null
            },
            execution: {
                status: "DONE",
                log: {
                    stage_name: "Done",
                    last_updated_at: "2025-04-09T05:26:59.925502+00:00",
                    detail: "This job successfully completed!"
                },
                start_time: "2025-04-09T05:19:00.229711+00:00",
                last_updated_at: "2025-04-09T05:27:46.421947+00:00",
                workflow_end_state: 2,
                failure_info: null,
                progress: {
                    iterations_completed: 5,
                    iterations_remaining: 0,
                    start_time: "2025-04-09T05:26:07.466376+00:00",
                    elapsed_time: "PT1.427677S",
                    estimated_end_time: "2025-04-09T05:26:08.894053+00:00"
                },
                results: {
                    key_cols: ["COUNT(FINANCE_TRANS.*, 0, 30, days) = 0"],
                    output_artifact_stats: [],
                    num_rows: -1,
                    num_cols: -1,
                    size_bytes: -1,
                    output_paths: []
                }
            },
            graph_link: {
                name: "NA-0005b000000000000000000000000000",
                id: "NA-0005b000000000000000000000000000",
                snapshot_id: null
            },
            description: "",
            batch_prediction_workflow_id: null,
            tags: {
                pquery_id: "pquery-00027000000000000000000000000000",
                model_id: "trainingjob-00046000000000000000000000000000",
                new_prediction_job: "new_prediction_job"
            },
            job_source: "UI"
        }
    ]
};
export const mockEdgeHealthStats = {
    is_ready: true,
    link_stats: [
        {
            col_a: {
                table_id: "1",
                connector_id: "con1",
                table_name: "users",
                column_name: "id"
            },
            col_b: {
                table_id: "2",
                connector_id: "con1",
                table_name: "orders",
                column_name: "user_id"
            },
            link_type: "FK",
            matches: {
                a_in_b: 95,
                b_in_a: 90
            },
            percents: {
                a_in_b: 95.0,
                b_in_a: 90.0
            },
            edge_count: 1
        }
    ]
};

export const mockWeakEdgeHealthStats = {
    is_ready: true,
    link_stats: [
        {
            col_a: {
                table_id: "1",
                connector_id: "con1",
                table_name: "products",
                column_name: "product_id"
            },
            col_b: {
                table_id: "2",
                connector_id: "con1",
                table_name: "sales",
                column_name: "product_id"
            },
            link_type: "FK",
            matches: {
                a_in_b: 30,
                b_in_a: 20
            },
            percents: {
                a_in_b: 30.0,
                b_in_a: 20.0
            },
            edge_count: 1
        }
    ]
};

export const mockGraphResponse: GraphResourceResponse = {
    id: "graph-00037000000000000000000000000000",
    name_alias: "HNM_DOWNSAMPLED_DATASET_GRAPH",
    num_links: 2,
    num_tables: 3,
    timestamps: {
        created_at: "2025-05-21T05:02:45.616582+00:00",
        updated_at: "2025-06-04T20:20:48.294812+00:00"
    },
    graph: {
        tables: {
            // minimal table structure for test
            transactions_2020: {
                cols: [],
                source_table: {
                    connector_id: "",
                    source: { mode: "relative", path: "" },
                    file_type: "PARQUET",
                    data_source_type: DataSourceType.S3
                },
                pkey: undefined,
                time_col: "t_dat",
                end_time_col: undefined
            }
        },
        col_groups: [
            {
                columns: [
                    { table_name: "articles", col_name: "article_id" },
                    { table_name: "transactions_2020", col_name: "article_id" }
                ]
            }
        ]
    },
    table_name_to_fpid_mapping: {
        transactions_2020: "d5828d994e8679119b05ec8f07de920a"
    }
};
export const trainTableGenmockResponse: GenerateTrainTableResponse = {
    resource: {
        job_id: "gen-traintable-job-00068000000000000000000000000000",
        project_id: "project_00000000000000000000000000000001",
        job_status_report: {
            status: jobStatus.Done,
            tracking_url: "n/a (under development)",
            start_time: new Date(
                "2025-06-04T20:34:29.757558+00:00"
            ).toISOString(),
            end_time: new Date(
                "2025-06-04T20:46:54.026512+00:00"
            ).toISOString(),
            event_log: [
                {
                    stage_name: "Generating Training Data",
                    last_updated_at: new Date(
                        "2025-06-04T20:46:53.927149+00:00"
                    ).toISOString(),
                    detail: "Successfully completed generating train table."
                }
            ]
        },
        created_at: "2025-06-04T20:34:29.757558+00:00",
        tags: {
            hnm_integ_binclass_f1: "hnm_integ_binclass_f1",
            pquery_id: "pquery-0000f000000000000000000000000000",
            parent_job: "trainingjob-00068000000000000000000000000000"
        },
        config: {
            job_tags: {},
            pquery_id: "pquery-0000f000000000000000000000000000",
            plan: {
                split: "ApproxDateOffsetSplit(ratio=[0.8, 0.1, 0.1])",
                train_start_offset: "inferred",
                train_end_offset: "inferred",
                timeframe_step: "inferred"
            },
            graph_snapshot_id:
                "graph-00016000000000000000000000000000@00015000000000000000000000000000"
        },
        source: "UI"
    }
};

export const predictionTableGenmockResponse: GeneratePredictionTableResponse = {
    resource: {
        job_id: "gen-predtable-job-0003f000000000000000000000000000",
        project_id: "project_00000000000000000000000000000001",
        job_status_report: {
            status: jobStatus.Done,
            tracking_url: "n/a (under development)",
            start_time: "2025-06-05T04:36:27.551303+00:00",
            end_time: "2025-06-05T04:44:31.118493+00:00",
            event_log: [
                {
                    stage_name: "Generating Prediction Data",
                    last_updated_at: "2025-06-05T04:44:31.019879+00:00",
                    detail: "Successfully completed generating prediction table."
                }
            ],
            validation_response: undefined
        },
        created_at: "2025-06-05T04:36:27.551303+00:00",
        tags: {
            pquery_id: "pquery-0004c000000000000000000000000000"
        },
        config: {
            job_tags: {},
            pquery_id: "pquery-0004c000000000000000000000000000",
            plan: {
                anchor_time: "inferred",
                lag_timesteps: "???",
                forecast_length: undefined,
                year_over_year: false
            },
            graph_snapshot_id:
                "graph-00016000000000000000000000000000@00015000000000000000000000000000"
        },
        source: "UI"
    }
};

export const mockGetTablesDetails = {
    "connector": {
        "name": "ltv_test",
        "id": "ltv_test"
    },
    "id": "18e2b1a6e499fff45a6f249f3024662f",
    "name_alias": "transaction_ltv1",
    "stats": {
        "size_bytes": 3560810,
        "num_rows": 541909
    },
    "table": {
        "cols": [
            {
                "name": "InvoiceNo",
                "stype": "categorical",
                "dtype": "string",
                "timestamp_format": null
            },
            {
                "name": "StockCode",
                "stype": "ID",
                "dtype": "string",
                "timestamp_format": null
            },
            {
                "name": "Description",
                "stype": "categorical",
                "dtype": "string",
                "timestamp_format": null
            },
            {
                "name": "Quantity",
                "stype": "numerical",
                "dtype": "int64",
                "timestamp_format": null
            },
            {
                "name": "InvoiceDate",
                "stype": "timestamp",
                "dtype": "date",
                "timestamp_format": null
            },
            {
                "name": "UnitPrice",
                "stype": "numerical",
                "dtype": "float64",
                "timestamp_format": null
            },
            {
                "name": "CustomerID",
                "stype": "numerical",
                "dtype": "float64",
                "timestamp_format": null
            },
            {
                "name": "Country",
                "stype": "categorical",
                "dtype": "string",
                "timestamp_format": null
            }
        ],
        "pkey": null,
        "end_time_col": null,
        "source_table": {
            "connector_id": "ltv_test",
            "data_source_type": "S3",
            "file_type": "PARQUET",
            source: { mode: "relative", path: "transaction" }
        },
        "time_col": "InvoiceDate"
    },

    "time_range": {
        "min": "2010-12-01T08:26:00",
        "max": "2011-12-09T12:50:00",
        "approx": false
    },
    "timestamps": {
        "created_at": "2025-02-13T04:37:03.949636+00:00",
        "updated_at": "2025-02-13T04:38:11.683211+00:00"
    }
};
export const mockGraphInfo: JSGraph = {
    id: "graph_1",
    name: "Sample Graph",
    totalLinks: 0,
    totalTables: 1,
    tables: {
        artlcles: {
            cols: [
                {
                    "name": "article_id",
                    "stype": "ID",
                    "dtype": "int64",
                    "timestamp_format": undefined
                },

                {
                    "name": "prod_name",
                    "stype": "text",
                    "dtype": "string",
                    "timestamp_format": undefined
                }
            ],
            source_table: {
                connector_id: "hnm_integ",
                source: { mode: "relative", path: "articles" },
                file_type: "PARQUET",
                data_source_type: DataSourceType.S3
            },
            pkey: "article_id",
            time_col: "",
            end_time_col: ""
        }
    }
};

export const subgraphMockData = {
    feature_store: {
        features: {
            "0": {
                t_dat: {
                    table_name: "TRANSACTIONS",
                    name: "t_dat",
                    value: "2020-08-31T00:00:00",
                    stype: "timestamp",
                    score: 1
                },
                price: {
                    table_name: "TRANSACTIONS",
                    name: "price",
                    value: 0.03,
                    stype: "numerical",
                    score: 0.05
                }
            },
            "2": {
                FN: {
                    table_name: "CUSTOMERS",
                    name: "FN",
                    value: 1,
                    stype: "numerical",
                    score: 0.07
                }
            },
            "12": {
                prod_name: {
                    table_name: "ARTICLES",
                    name: "prod_name",
                    value: "ED Nora tee 2p",
                    stype: "categorical",
                    score: 0.1
                }
            },
            "-1": {
                FN: {
                    table_name: "CUSTOMERS",
                    name: "FN",
                    value: 1,
                    stype: "numerical",
                    score: null
                }
            }
        }
    },
    subgraph_store: {
        nodes: {
            "0": { id: 0, pkey_name: null, pkey: null },
            "2": { id: 2, pkey_name: "customer_id", pkey: "customer_pkey_2" },
            "12": { id: 12, pkey_name: "article_id", pkey: 900176001 },
            "-1": { id: -1, pkey_name: "customer_id", pkey: null }
        },
        tables: {
            TRANSACTIONS: [0],
            CUSTOMERS: [2, -1],
            ARTICLES: [12]
        },
        edges: {},
        root_id: 12,
        task_type: "temporal_link_prediction",
        seed_time: "2020-09-15T00:00:00"
    }
};
export const mockEdges: { [key: string]: SubgraphEdge } = {
    "0": {
        customer_id: [2]
    },
    "12": {
        article_id: [0]
    }
};

export const mockNodeArr: GraphNode[] = [
    {
        id: "0",
        nodeId: "node_0",
        data: {
            value: "0",
            tableName: "TRANSACTIONS",
            idName: "",
            cols: [],
            totalScore: 0
        },
        position: { x: 0, y: 0 }
    },
    {
        id: "2",
        nodeId: "node_2",
        data: {
            value: "customer_pkey_2",
            tableName: "CUSTOMERS",
            idName: "customer_id",
            cols: [],
            totalScore: 0.07
        },
        position: { x: 100, y: 100 }
    },
    {
        id: "12",
        nodeId: "node_12",
        data: {
            value: "900176001",
            tableName: "ARTICLES",
            idName: "article_id",
            cols: [],
            totalScore: 0.1
        },
        position: { x: 200, y: 200 }
    }
];

export const columns = [
    {
        "table_name": "Ratings",
        "name": "Rating",
        "value": 5,
        "stype": "numerical",
        "score": 0.008584852330386639,
        "grad_score": 0.008584852330386639,
        "isXaiUpdated": true,
        "xai_prediction": 4.849,
        "xai_global_avg": 4.829967470000001
    },
    {
        "table_name": "Ratings",
        "name": "timestamp",
        "value": "2013-05-28T00:00:00",
        "stype": "timestamp",
        "score": 0.412227988243103,
        "grad_score": 0.412227988243103,
        "isXaiUpdated": true
    }
];

export const xaiData = {
    "Ratings.timestamp_(1_hop)": {
        "globalAvgScore": 4.82972283,
        "labels": [
            "[0 months, 17 months)",
            "[17 months, 34 months)",
            "[34 months, 51 months)",
            "[51 months, 68 months)",
            "[68 months, 85 months)",
            "[85 months, 102 months)",
            "[119 months, 136 months)"
        ],
        "predictions": [
            "4.851",
            "4.851",
            "4.851",
            "4.844",
            "4.860",
            "4.862",
            "4.858"
        ]
    },
    "Ratings.Rating_(1_hop)": {
        "globalAvgScore": 4.829967470000001,
        "labels": [
            "[1 - 3]",
            "(3 - 3.75]",
            "(3.75 - 4]",
            "(4 - 4.333]",
            "(4.333 - 4.5]",
            "(4.5 - 4.766]",
            "(4.766 - 5]"
        ],
        "predictions": [
            "4.848",
            "4.859",
            "4.847",
            "4.853",
            "4.855",
            "4.858",
            "4.849"
        ]
    },
    "Ratings.COUNT(*)_(1_hop)": {
        "globalAvgScore": 4.7700679500000005,
        "labels": ["[0 - 0]", "(0 - 1]", "(1 - 2]", "(2 - 6]", "(6 - 44+]"],
        "predictions": ["4.810", "4.844", "4.859", "4.863", "4.845"]
    }
};
export const anchorTime = "Sat May 24, 2014";

export const mockOutputArtifactStatsFromResults: TrainingJobResults = {
    output_artifact_stats: [
        {
            output_path: "s3://file1",
            num_rows: 10,
            num_cols: 2,
            size_bytes: 1024,
            artifact_type: KumoapiJobsPredictionArtifactType.PREDICTIONS
        },
        {
            output_path: "s3://file2",
            num_rows: 20,
            num_cols: 3,
            size_bytes: 2048,
            artifact_type: KumoapiJobsPredictionArtifactType.EMBEDDINGS
        }
    ],

    output_paths: [],
    num_rows: 0,
    num_cols: 0,
    size_bytes: 0,
    key_cols: []
};

export const mockParseUnlinkedPredictionJobs: BatchPredictionJobResponse[] = [
    {
        id: "job-123",
        project_id: "project_00000000000000000000000000000001",
        batch_prediction_workflow_id: "workflow-1",
        description: "Sample Job",
        predictive_query_link: {
            id: "PQ123",
            name: "PredictiveQueryTest",
            training_job_id: "TJ456"
        },
        config: {
            predictive_query_id: "PQ123",
            training_job_id: "TJ456",
            outputs: [],
            task_type: KumoCoreTypesEnumsTaskType.REGRESSION,
            num_parallel_workers_limit: 1
        },
        execution: {
            status: jobStatus.Done,
            start_time: "2025-08-20T12:00:00Z",
            last_updated_at: "2025-08-20T12:30:00Z",
            results: {
                num_rows: 100,
                num_cols: 10,
                size_bytes: 2048,
                key_cols: ["id"],
                output_artifact_stats: [],
                output_paths: []
            },
            log: {
                stage_name: "",
                last_updated_at: "2025-09-20T11:30:00Z",
                detail: undefined
            }
        },
        timestamps: {
            created_at: "2025-08-20T11:00:00Z",
            updated_at: "2025-09-20T11:00:00Z"
        },
        graph_link: {
            id: "graph-001",
            name: "GraphTest"
        },
        tags: {},
        job_source: JobSource.UI
    }
];
