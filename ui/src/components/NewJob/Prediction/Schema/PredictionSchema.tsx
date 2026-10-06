// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import {
    Card,
    CardContent,
    GridCol,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { QueryType } from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { TaskType } from "types/predictiveQueries";

import GraphSchema from "./GraphSchema";
import OutputTableSchema from "./OutputTableSchema";

export interface PredictionSchemaProps {
    queryId: string;
    graphSnapshotId: string;
    taskType?: TaskType;
    queryType?: QueryType;
    trainingTableId?: string;
}

const PredictionSchema: React.FC<PredictionSchemaProps> = ({
    queryId,
    graphSnapshotId,
    taskType,
    queryType,
    trainingTableId
}) => {
    const [tabValue, setTabValue] = useState("0");

    return (
        <GridCol
            data-testid="new-prediction-schema-panel"
            xs={12}
            sm={12}
            md={6}
            className="p-0 border-l border-border-secondary overflow-y-auto h-full">
            <Title variant="secondary-heading" className="py-4 px-4">
                Schema
            </Title>
            <Card className="p-0 gap-0 border-none bg-transparent mb-0 h-full">
                <Tabs value={tabValue} onValueChange={setTabValue}>
                    <div className="flex h-10 flex-row items-center px-4">
                        <TabsList className="w-full border-b border-border-secondary">
                            <TabsTrigger value="0" className="px-0.5 py-1.5">
                                Graph
                            </TabsTrigger>
                            <TabsTrigger value="1" className="px-0.5 py-1.5">
                                Output table
                            </TabsTrigger>
                        </TabsList>
                    </div>
                    <CardContent className="p-3 pt-4">
                        <TabsContent className="py-0" value="0">
                            <div className="ml-1">
                                <Card className="border-border-secondary">
                                    <GraphSchema
                                        graphSnapshotId={graphSnapshotId}
                                    />
                                </Card>
                            </div>
                        </TabsContent>
                        <TabsContent className="py-0" value="1">
                            <div className="ml-1">
                                <OutputTableSchema
                                    queryId={queryId}
                                    taskType={taskType}
                                    queryType={queryType}
                                    trainingTableID={trainingTableId}
                                />
                            </div>
                        </TabsContent>
                    </CardContent>
                </Tabs>
            </Card>
        </GridCol>
    );
};

export default WithErrorBoundary(PredictionSchema);
