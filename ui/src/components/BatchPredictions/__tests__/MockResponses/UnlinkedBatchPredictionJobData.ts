// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KumoapiJobsPredictionArtifactType } from "gen/v1/types";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";
import { JobSource } from "types/jobs";
import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";

export const MockUnlinkedBatchPredictionJobData: UnlinkedBatchPredictionJobData =
    {
        "id": "bp-job-1937b6a194fd47cda79d490860314028",
        "workflow_id": undefined,
        "description": "",
        "predictiveQuery": "NA-ebc76ac857a419856408011218b34ab5",
        "predictiveQueryId": "NA-ebc76ac857a419856408011218b34ab5",
        "predictiveQuery_link": "/jobs/training/null",
        "status": jobStatus.Done,
        "created": "23 days ago",
        "dateCreated": "Thu Jul 3, 2025, 8:33:22 AM",
        "sortCreated": "2025-07-03, 08:33:22 UTC",
        "numRows": "4.5K",
        "numCols": "4",
        "num_rows_original": -1,
        "outputsList":
            "PREDICTIONS: s3://kumo-public-datasets/momentic_predictions/predictions/7e115_predictions.parquet",
        "outputs": [
            {
                "artifact_type": KumoapiJobsPredictionArtifactType.PREDICTIONS,
                "file_path":
                    "s3://kumo-public-datasets/momentic_predictions/predictions/7e115_predictions.parquet",
                "storage_type": "S3",
                "connector_id": "connector_test",
                "extra_fields": [],
                "file_type": undefined
            }
        ],
        "runtime": "07m",
        "id_link": "/jobs/",
        "log": {
            "stage_name": "Done",
            "last_updated_at": "2025-07-03T04:15:41.614130+00:00",
            "detail": "This job successfully completed!"
        },
        "size": "66.6 KB",
        "taskType": TaskType.BINARY_CLASSIFICATION,
        "keyColumns": "COUNT(FINANCE_TRANS.*, 0, 30, days) = 0",
        "outputPaths": [
            "s3://kumo-public-datasets/momentic_predictions/predictions/7e115_predictions.parquet/"
        ],
        "artifactStats": [
            {
                "artifact_type": KumoapiJobsPredictionArtifactType.PREDICTIONS,
                "num_rows": 4500,
                "num_cols": 4,
                "size_bytes": 66567,
                "output_path":
                    "s3://kumo-public-datasets/momentic_predictions/predictions/7e115_predictions.parquet/"
            }
        ],
        "graphId": "NA-44a59331b438072b533f52e33d01835d",
        "graphName": "NA-44a59331b438072b533f52e33d01835d",
        "explanations": true,
        "graphSnapshotId":
            "graph-44a59331b438072b533f52e33d01835d@3c7208307485c268cdc9ce1ddd9b151b",
        "classes": undefined,
        "isJobUI": false,
        "trainingJobID": "trainingjob-7e115bdc420046e388899b679d6c3a0b",
        "tags": {
            "ENABLED_EXPLANATION": "ENABLED_EXPLANATION",
            "pquery_id": "pquery-ebc76ac857a419856408011218b34ab5",
            "model_id": "trainingjob-7e115bdc420046e388899b679d6c3a0b"
        },
        "execution": {
            "status": jobStatus.Done,
            "log": {
                "stage_name": "Done",
                "last_updated_at": "2025-07-03T04:15:41.614130+00:00",
                "detail": "This job successfully completed!"
            },
            "start_time": "2025-07-03T08:33:22.602348+00:00",
            "last_updated_at": "2025-07-03T08:40:05.025898+00:00",
            "workflow_end_state": 2,
            "failure_info": undefined,
            "results": {
                "key_cols": ["COUNT(FINANCE_TRANS.*, 0, 30, days) = 0"],
                "output_artifact_stats": [
                    {
                        "artifact_type":
                            KumoapiJobsPredictionArtifactType.PREDICTIONS,
                        "num_rows": 4500,
                        "num_cols": 4,
                        "size_bytes": 66567,
                        "output_path":
                            "s3://kumo-public-datasets/momentic_predictions/predictions/7e115_predictions.parquet/"
                    }
                ],
                "num_rows": -1,
                "num_cols": -1,
                "size_bytes": -1,
                "output_paths": []
            }
        },
        "source": JobSource.WEB,
        "generatePredTableJobId":
            "gen-predtable-job-3fc98c362f5649bd992fff32929aec55",
        "threshold": 0.5,
        "threshold_lable": true,
        "failureInfo": ""
    };
