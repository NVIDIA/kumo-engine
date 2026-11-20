// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    Button,
    Description,
    DocumentationLink,
    ProgressSpinner
} from "@kumo-ai/kumo-ui-catalog";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import WarningErrorCTA from "components/@ui/Warning/WarningErrorCTA";

import { useTablePreprocessing } from "hooks/tables/preprocess";

import { ctaLevels } from "types/constants";
import { DropdownOption } from "types/inputs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { docURLs, warehouseSource } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";

import { ColumnSelectionTable } from "./ColumnSelectionTable";
import DataTypeSuggestions from "./DataTypeSuggestions";
import RecommendedTypes from "./RecomendedTypes";

export interface PreprocessingProps {
    tableArgs: KumoapiTableTableDefinition;
    flowType: FlowType;
    handleError: HandleErrorFunc;
    setLoading: (v: boolean) => void;
    isLoading?: boolean;
    setTableArgs: React.Dispatch<
        React.SetStateAction<KumoapiTableTableDefinition>
    >;
    selectedConnector: DropdownOption;
    cancelFetchSampleCal?: boolean;
    connectorId: string;
    selectedSourceTableName: string;
    selectFromTableArgs?: boolean;
    connectorNotExists?: boolean;
    setUnnamedConSourceTable: React.Dispatch<React.SetStateAction<string>>;
    sourcetableNotExists?: boolean;
    setColumnNamesAdjusted: React.Dispatch<React.SetStateAction<boolean>>;
    columnNamesAdjusted: boolean;
}

const Preprocessing = ({
    tableArgs,
    flowType,
    handleError,
    setLoading,
    isLoading,
    selectedConnector,
    setTableArgs,
    cancelFetchSampleCal = false,
    connectorId,
    selectedSourceTableName,
    selectFromTableArgs,
    connectorNotExists = false,
    setUnnamedConSourceTable,
    setColumnNamesAdjusted,
    columnNamesAdjusted
}: PreprocessingProps) => {
    const {
        columns,
        columnSTypes,
        columnDTypes,
        selected,
        errors,
        isLoading: fetchingData,
        isView,
        recommendedSTypes,
        onSetColumnDataType,
        onSetColumnSType,
        onSelectColumns,
        onSelectPrimaryKey,
        onCreateTimeColSelect,
        onEndTimeColSelect,
        applyRecommendedTypes,
        setInvalidError,
        setSampleRowError,
        skipSampleFetch
    } = useTablePreprocessing({
        connectorId,
        selectedSourceTableName,
        tableArgs,
        flowType,
        connectorNotExists,
        selectFromTableArgs,
        cancelFetchSampleCal,
        setTableArgs,
        setUnnamedConSourceTable,
        setLoading,
        handleError
    });

    if (fetchingData)
        return (
            <div className="flex flex-col justify-center items-center mt-8 gap-2">
                <ProgressSpinner />
                <Description
                    variant="secondary"
                    size="sm"
                    className="leading-5">
                    Inferring column types based on sample rows.
                    {selectedConnector?.type === warehouseSource.SnowFlake
                        ? " The duration depends on your warehouse size"
                        : ""}
                </Description>

                <Button
                    variant="light-outline"
                    size="sm"
                    onClick={skipSampleFetch}
                    id={"skip"}>
                    {"  Skip  "}
                </Button>
            </div>
        );

    let warnings: string = "";
    if (isValid(errors.columnWarnings)) {
        errors.columnWarnings?.warnings?.forEach((warning) => {
            warnings += warning.title + ": " + warning.message + "\n";
        });
    }

    return (
        <>
            <Description size="sm" className="m-0 p-0 leading-5">
                Select the columns & any preprocessing you want to apply to new
                table.
                {" See "}
                <DocumentationLink
                    id="column_preprocessing_doc"
                    url={docURLs.ColumnPreprocessing}
                    linkText="docs"
                    isNewDocs={true}
                />
                {" for more details."}
            </Description>

            {isValid(errors.invalidKeyError.description) && (
                <WarningErrorCTA
                    id={"error-invalid-keys"}
                    title={errors.invalidKeyError.title}
                    ctaLevel={ctaLevels.Critical}
                    data={[
                        {
                            title: "",
                            description: errors.invalidKeyError.description,
                            is_error: false,
                            is_dismissible: false,
                            cta: null
                        }
                    ]}
                    handleClose={() => {
                        setInvalidError({
                            title: "",
                            description: ""
                        });
                    }}
                />
            )}
            {isValid(errors.columnWarnings?.warnings) &&
                warnings.length > 0 && (
                    <WarningErrorCTA
                        id={"column-type-change-warning"}
                        title={"Column Type Change"}
                        ctaLevel={ctaLevels.Warning}
                        data={[
                            {
                                title: "",
                                description: warnings ?? "",
                                is_error: false,
                                is_dismissible: false,
                                cta: null
                            }
                        ]}
                        handleClose={() => {}}
                    />
                )}
            {isValid(errors.sampleRowError) && (
                <WarningErrorCTA
                    id={"error-sample-rows"}
                    title={"Request Failed!"}
                    ctaLevel={ctaLevels.Critical}
                    data={[
                        {
                            title: "",
                            description: errors.sampleRowError,
                            is_error: false,
                            is_dismissible: false,
                            cta: null
                        }
                    ]}
                    handleClose={() => {
                        setSampleRowError("");
                    }}
                />
            )}
            <DataTypeSuggestions columns={tableArgs?.cols} />

            <RecommendedTypes
                recommendedSTypes={recommendedSTypes}
                onApplyRecommendedTypes={applyRecommendedTypes}
            />
            {columnNamesAdjusted && (
                <WarningErrorCTA
                    id={"warning-column-header-adjuested"}
                    title="Warning"
                    ctaLevel={ctaLevels.Warning}
                    data={[
                        {
                            title: "Column names adjusted",
                            description:
                                ` Kumo supports only letters, numbers, and underscores in column names. ` +
                                ` Some column names were automatically adjusted, example: ` +
                                `"column header with spaces" → "column_header_with_spaces".`,
                            is_error: false,
                            is_dismissible: false
                        }
                    ]}
                    handleClose={() => setColumnNamesAdjusted(false)}
                />
            )}
            <div className="w-full overflow-auto">
                <ColumnSelectionTable
                    columns={columns}
                    selected={selected}
                    showPreview={!isView}
                    onSelectColumns={onSelectColumns}
                    columnDTypes={columnDTypes}
                    columnSTypes={columnSTypes}
                    onCreateTimeColSelect={onCreateTimeColSelect}
                    onEndTimeColSelect={onEndTimeColSelect}
                    onSelectPrimaryKey={onSelectPrimaryKey}
                    onSetColumnDataType={onSetColumnDataType}
                    onSetColumnSType={onSetColumnSType}
                    isLoading={isLoading as boolean}
                />
            </div>
        </>
    );
};
export default WithErrorBoundary(Preprocessing);
