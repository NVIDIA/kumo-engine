// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryType } from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import OutputTable from "components/@ui/TablePlus/OutputTable";

import {
    usePredictionTableSchema,
    usePredictionTableSchemaSamples
} from "hooks/jobs/predictionTableGen/predictionTableGen";

import { TaskType } from "types/predictiveQueries";

export interface OutputTableSchemaProps {
    queryId: string;
    taskType?: TaskType;
    queryType?: QueryType;
    trainingTableID?: string;
}

const OutputTableSchema: React.FC<OutputTableSchemaProps> = ({
    queryId,
    taskType,
    queryType,
    trainingTableID
}) => {
    const [predTableSchema, loadingPredTableSchema] = usePredictionTableSchema(
        queryId,
        taskType,
        queryType,
        trainingTableID
    );

    const [predTableSchemaSamples] = usePredictionTableSchemaSamples(
        trainingTableID,
        taskType
    );

    if (predTableSchemaSamples.length !== 0) {
        const combinedOutput = predTableSchema.map((column) => {
            const sample = predTableSchemaSamples.find(
                (s) => s.name === column.name
            );

            return {
                name: column.name,
                colType: column.colType,
                samples: sample ? sample.sample_rows : [] // Default to empty array if no sample found
            };
        });

        return (
            <OutputTable
                title="Predictions table"
                cols={combinedOutput}
                loading={loadingPredTableSchema}
                areSamplesAvailable={true}
            />
        );
    } else {
        return (
            <OutputTable
                title="Predictions table"
                cols={predTableSchema}
                loading={loadingPredTableSchema}
                areSamplesAvailable={false}
            />
        );
    }
};

export default WithErrorBoundary(OutputTableSchema);
