// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useMemo, useState } from "react";

import {
    Card,
    CardContent,
    CardHeader,
    CopyToClipBoard,
    DataTable,
    Description,
    Info,
    ScrollArea,
    Skeleton,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
    Title,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { DocumentationLink } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import MPCodeEditor from "components/CodeEditor/MPCodeEditor";

import { useGetQueryEncoders } from "hooks/encoders/encoders";

import { TrainingJobID } from "types/training_job";
import { CTAInfo } from "types/warnings";

import { ctaInfoItems, docURLs } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";

import { CODE_EDITOR_SETUP } from "../Write/ModelPlannerEditor";

export interface TrainingSettingProps {
    queryId: string;
    trainingJobId: TrainingJobID;
    modelPlan: string;
    advancedOptions: string;
    generationPlan?: string;
    trainingPlan?: string;
    trainingstatus?: string;
    ctaMsgs: CTAInfo;
    loadingStatus?: boolean;
}

const TrainingSetting: React.FC<TrainingSettingProps> = ({
    queryId,
    trainingJobId,
    modelPlan,
    advancedOptions,
    generationPlan,
    trainingPlan,
    trainingstatus,
    ctaMsgs,
    loadingStatus = false
}) => {
    const encoders = useGetQueryEncoders(
        queryId,
        trainingJobId,
        trainingstatus
    );
    const tabs = useMemo(() => {
        const t = [];

        if ((modelPlan || advancedOptions) && !generationPlan) {
            t.push({
                label: modelPlan ? "Model plan" : "Advanced options",
                value: 0
            });
        } else {
            if (trainingPlan) {
                t.push({ label: "Model plan", value: 2 });
            }
            if (generationPlan) {
                t.push({ label: "Training table generation plan", value: 3 });
            }
        }
        if (getKeys(encoders).length > 0) {
            t.push({ label: "Column processing", value: 1 });
        }
        return t;
    }, [advancedOptions, encoders, generationPlan, modelPlan, trainingPlan]);
    const [activeTab, setActiveTab] = useState(
        tabs.length > 0 ? `${tabs[0].value}` : ""
    );

    const encoderText =
        "encoder_overrides:\n" +
        getKeys(encoders || {})
            .map((colname: string) => `    ${colname}: ${encoders[colname]}`)
            .join("\n");

    const info = useMemo(() => {
        const item = ctaMsgs?.info_items?.find(
            (d) => d?.id === ctaInfoItems.split_setup_validation
        );
        if (item?.description) {
            return item.description;
        }
        return "";
    }, [ctaMsgs]);

    const tableData = useMemo(
        () =>
            getKeys(encoders || {}).map((colname: string) => ({
                connectorId: colname?.split(".")[0],
                tableId: colname?.split(".")[1],
                columnName: colname?.split(".")[2],
                encoder: encoders[colname]
            })),
        [encoders]
    );

    return (
        <Card className="p-6 gap-6">
            {loadingStatus ? (
                <Skeleton className="h-8 w-250" />
            ) : (
                <CardHeader className="p-0">
                    <Title
                        variant="tertiary-heading"
                        children={"Training Settings"}
                    />
                    {info && (
                        <Tooltip>
                            <TooltipTrigger>
                                <span
                                    className="ml-1 flex items-center"
                                    data-testid="training-info-icon">
                                    <Info className="size-3.5" />
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                arrow={false}
                                side={"right"}
                                sideOffset={5}
                                className="max-w-100 justify-around">
                                {info}
                            </TooltipContent>
                        </Tooltip>
                    )}
                </CardHeader>
            )}
            <CardContent className="p-0">
                <Tabs
                    value={activeTab}
                    onValueChange={setActiveTab}
                    defaultValue={activeTab}>
                    <TabsList className="gap-8">
                        {tabs?.map(({ label, value }) => (
                            <TabsTrigger
                                key={`tab-${value}`}
                                value={`${value}`}
                                className={"p-1 pl-0 pr-0 leading-5"}>
                                {label}
                            </TabsTrigger>
                        ))}
                    </TabsList>
                    <hr />
                    {loadingStatus ? (
                        <Skeleton className="h-8 w-200" />
                    ) : (
                        <TabsContent className="p-0 pt-6" value={activeTab}>
                            {[`0`, `2`, `3`].includes(activeTab) && (
                                <ScrollArea
                                    id={
                                        activeTab === `0`
                                            ? "model_plan"
                                            : undefined
                                    }
                                    type="always"
                                    className="border rounded-sm w-full !h-125 bg-surface-grey">
                                    <MPCodeEditor
                                        language="YAML"
                                        value={
                                            activeTab === `0`
                                                ? modelPlan || advancedOptions
                                                : activeTab === `2`
                                                  ? (trainingPlan as string)
                                                  : (generationPlan as string)
                                        }
                                        basicSetup={CODE_EDITOR_SETUP}
                                        onChange={() => {}}
                                        editable={false}
                                        hideBorder
                                        height="100%"
                                        backgroundColor="var(--surface-grey)"
                                    />
                                </ScrollArea>
                            )}
                            {activeTab === `1` && (
                                <>
                                    <Description
                                        variant="secondary"
                                        size="md"
                                        className="mb-6 leading-5">
                                        This section shows the Encoders used to
                                        process your raw data (see{" "}
                                        <DocumentationLink
                                            id="query_suggestion_doc"
                                            url={docURLs.EncoderOverrides}
                                            linkText="docs"
                                            isNewDocs={true}
                                        />{" "}
                                        for more details on encoders). Kumo
                                        applies an encoder to each column of
                                        each table in your graph based on the
                                        inferred data type and statistics. These
                                        are the Encoders that were used in
                                        training this pQuery (including any that
                                        were overriden using the
                                        encoder_overrides field of the Model
                                        Plan). If you would like to exactly
                                        reproduce this pQuery, you can copy the
                                        encoders and add them to the Model Plan
                                        of a new pQuery.{" "}
                                        <CopyToClipBoard
                                            copiedText={encoderText as string}
                                            iconSize={16}
                                            tooltipText={"Copied"}
                                            className="align-middle"
                                        />
                                    </Description>
                                    <DataTable
                                        id={"info_card"}
                                        disableRowSelectionHover={true}
                                        columns={[
                                            {
                                                id: "connectorId",
                                                header: "Connector" as string,
                                                accessorKey: "connectorId"
                                            },
                                            {
                                                id: "tableId",
                                                header: "Table",
                                                accessorKey: "tableId"
                                            },
                                            {
                                                id: "columnName",
                                                header: "Column",
                                                accessorKey: "columnName"
                                            },
                                            {
                                                id: "encoder",
                                                header: "Encoder",
                                                accessorKey: "encoder"
                                            }
                                        ]}
                                        data={tableData}
                                        isLoading={false}
                                        pageCount={
                                            tableData?.length > 0
                                                ? Math.ceil(
                                                      tableData?.length / 6
                                                  )
                                                : 1
                                        }
                                        defaultPageSize={6}
                                        enableMultiSort={true}
                                        enableGlobalFilter={true}
                                    />
                                </>
                            )}
                        </TabsContent>
                    )}
                </Tabs>
            </CardContent>
        </Card>
    );
};

export default WithErrorBoundary(TrainingSetting);
