// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { ReactElement } from "react";
import Spacer from "react-spacer";

import {
    Card,
    CardContent,
    Checkbox,
    cn,
    Description,
    GridCol,
    GridRow,
    Label,
    Select,
    TextInput,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { ErrorDetail } from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { BatchPredictionJobOutputJs } from "types/batchPredictions";
import {
    ConnectorDropdownOption,
    DropdownOption,
    SelectOption
} from "types/inputs";
import { TaskType } from "types/predictiveQueries";

import { artifactType, digitsRegex, outputStorage } from "utils/Constants";
import { copyObject, getKeys, getValues } from "utils/Helpers/objectHelpers";
import { isLinkPredictionType, isValid } from "utils/Helpers/typeGuards";
import { isAnyUriObjectStoreConnector } from "utils/objectStoreTablePath";

// Helper function to find error by title substring
const findErrorByTitle = (
    errors: ErrorDetail[] | undefined,
    titleSubstring: string
): ErrorDetail | undefined =>
    errors?.find((err) => err?.title?.includes(titleSubstring));

const isObjectStoreOutput = (destination: string): boolean =>
    [outputStorage.S3, outputStorage.GCS, outputStorage.ADLS].includes(
        destination as outputStorage
    );

export interface PredictionOutputProps {
    destination: string;
    predictionType: string;
    onSelect: (newVal: string) => void;
    onArtifactChange: (newVal: string) => void;
    selectedConnector: ConnectorDropdownOption;
    setSelectedConnector: (val: ConnectorDropdownOption | null) => void;
    connectors: { [key: string]: Array<ConnectorDropdownOption> };
    classes: number | string;
    setClasses: (val: number | string) => void;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
    error: ErrorDetail[];
}

const PredictionOutput: React.FC<PredictionOutputProps> = ({
    destination,
    predictionType,
    onSelect,
    onArtifactChange,
    selectedConnector,
    setSelectedConnector,
    connectors,
    classes,
    setClasses,
    outputs,
    setOutputs,
    error
}) => {
    const outputList = getKeys(outputs);
    const options: Array<DropdownOption> = getValues(outputStorage)
        .map((e) => {
            if (e === outputStorage.Web) return null;
            return { value: e, label: e };
        })
        .filter((val) => val !== null);

    let selected: SelectOption = {
        value: "",
        label: ""
    };
    if (destination) {
        selected = { value: destination, label: destination };
    }

    return (
        <>
            <GridRow xs={12} sm={12} md={12}>
                <GridCol xs={12} sm={12}>
                    <Label htmlFor="Output Destination">
                        Output Destination
                    </Label>
                    <Select
                        id={"output_destination"}
                        placeholder="Select Destination"
                        search={false}
                        mode="single"
                        sortOptions={false}
                        popoverClassName="w-200"
                        selectedValues={
                            isValid(selected?.value)
                                ? Array.of(selected?.value)
                                : undefined
                        }
                        options={options?.map((item) => {
                            return {
                                value: item?.value as string,
                                label: item?.label as string
                            };
                        })}
                        onValuesChange={(newValue) => {
                            const currSel = options?.find((item) =>
                                newValue.includes(item?.value as string)
                            );
                            if (currSel?.value) {
                                onSelect(currSel?.value as string);
                                setClasses("");
                            }
                        }}
                    />
                </GridCol>
            </GridRow>
            {destination !== outputStorage.LocalDownloadOnly && (
                <>
                    <Spacer height={24} />
                    <GridRow xs={12} sm={12} md={12}>
                        <GridCol xs={12} sm={12}>
                            <OutputStorage
                                classes={classes}
                                setClasses={setClasses}
                                selectedConnector={
                                    selectedConnector as DropdownOption
                                }
                                setSelectedConnector={setSelectedConnector}
                                connectors={connectors}
                                destination={destination}
                                outputs={outputs}
                                setOutputs={setOutputs}
                                error={error}
                                predictionType={predictionType as string}
                                onArtifactChange={(newVal) =>
                                    onArtifactChange(newVal)
                                }
                            />
                            {!outputList?.includes(artifactType.Predictions) &&
                                !outputList?.includes(
                                    artifactType.Embeddings
                                ) &&
                                findErrorByTitle(
                                    error,
                                    "Missing output artifact."
                                )?.description && (
                                    <Description
                                        size="sm"
                                        variant="destructive"
                                        className="!mb-0">
                                        {
                                            findErrorByTitle(
                                                error,
                                                "Missing output artifact."
                                            )?.description
                                        }
                                    </Description>
                                )}
                        </GridCol>
                    </GridRow>
                </>
            )}
            {destination === outputStorage.LocalDownloadOnly && (
                <>
                    <Spacer height={8} />
                    <Card className="border-0">
                        <CardContent>
                            <Description
                                size="sm"
                                className="!mb-0 leading-5"
                                variant="secondary">
                                With this option, we won't write the predictions
                                back to your data plane. You will be able to
                                download a sample of the outputs (up to 1GB).
                                Please note that you can download a sample even
                                if you choose to write predictions to another
                                data source.
                            </Description>
                        </CardContent>
                    </Card>
                    {(isLinkPredictionType(predictionType) ||
                        [
                            TaskType.MULTICLASS_CLASSIFICATION as string,
                            TaskType.MULTILABEL_CLASSIFICATION as string,
                            TaskType.MULTILABEL_RANKING as string
                        ].includes(predictionType as TaskType)) && (
                        <Card
                            id={"local-download-prediction-options"}
                            className="gap-0 text-base font-normal bg-surface-grey">
                            <CardContent>
                                <Label
                                    className="mb-0"
                                    htmlFor="prediction-options">
                                    {artifactType.Predictions}
                                </Label>
                                {(predictionType ===
                                    TaskType.MULTICLASS_CLASSIFICATION ||
                                    predictionType ===
                                        TaskType.MULTILABEL_CLASSIFICATION) && (
                                    <MaxCategories
                                        artifact={artifactType.Predictions}
                                        classes={classes}
                                        setClasses={setClasses}
                                        predictionType={
                                            predictionType as string
                                        }
                                        errorMsg={
                                            findErrorByTitle(
                                                error,
                                                "Predicted Links per Entity"
                                            )?.description
                                        }
                                    />
                                )}
                                {(predictionType ===
                                    TaskType.MULTILABEL_RANKING ||
                                    isLinkPredictionType(predictionType)) && (
                                    <TopK
                                        artifact={artifactType.Predictions}
                                        classes={classes}
                                        setClasses={setClasses}
                                        predictionType={
                                            predictionType as string
                                        }
                                    />
                                )}
                            </CardContent>
                        </Card>
                    )}
                </>
            )}
        </>
    );
};

export default WithErrorBoundary(PredictionOutput);

/**
 * Component for databricks Storage Output Artiacts
 * @param onArtifactChange callback to update select/unselect artifact
 * @param classes clsses input field value
 * @param setClasses callback to set classes
 * @param destination selected output storage type
 * @param outputs selected artifactType
 * @param setOutputs callback to update output artifact settings
 * @param predictionType predictive query type
 * @param validate callback to validate the config
 */

//Definition of DatabricksStorage
const OutputStorageDef = ({
    classes,
    setClasses,
    selectedConnector,
    setSelectedConnector,
    connectors,
    destination,
    outputs,
    setOutputs,
    predictionType,
    onArtifactChange,
    error
}: {
    classes: number | string;
    setClasses: (val: number | string) => void;
    selectedConnector: ConnectorDropdownOption;
    setSelectedConnector: (val: ConnectorDropdownOption) => void;
    connectors: { [key: string]: Array<ConnectorDropdownOption> };
    destination: string;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
    predictionType: string;
    onArtifactChange: (newVal: string) => void;
    error: ErrorDetail[];
}) => {
    const predictionErrors = error?.filter((err) =>
        err?.title?.includes(artifactType.Predictions)
    );
    const embeddingErrors = error?.filter((err) =>
        err?.title?.includes(artifactType.Embeddings)
    );

    const options =
        destination === outputStorage.S3
            ? connectors?.s3Connectors?.map((item) => {
                  return {
                      ...item,
                      value: item?.value as string,
                      label: item?.label as string
                  };
              })
            : destination === outputStorage.GCS
              ? connectors?.gcsConnectors?.map((item) => ({
                    ...item,
                    value: item?.value as string,
                    label: item?.label as string
                }))
              : destination === outputStorage.ADLS
                ? connectors?.adlsConnectors?.map((item) => ({
                      ...item,
                      value: item?.value as string,
                      label: item?.label as string
                  }))
                : destination === outputStorage.Bigquery
                  ? connectors?.bigQueryConnectors?.map((item) => {
                        return {
                            ...item,
                            value: item?.value as string,
                            label: item?.label as string
                        };
                    })
                  : destination === outputStorage.Databricks
                    ? connectors?.dataBricksConnectors?.map((item) => {
                          return {
                              ...item,
                              value: item?.value as string,
                              label: item?.label as string
                          };
                      })
                    : destination === outputStorage.Snowflake
                      ? connectors?.snowFlakeConnectors?.map((item) => {
                            return {
                                ...item,
                                value: item?.value as string,
                                label: item?.label as string
                            };
                        })
                      : [];
    const outputOptions = isObjectStoreOutput(destination)
        ? options?.map((connector) => {
              const sourceOnly = isAnyUriObjectStoreConnector(connector.config);
              return {
                  ...connector,
                  disabled: sourceOnly,
                  labelDescription: sourceOnly
                      ? "Source-only any-URI connectors cannot write outputs."
                      : undefined
              };
          })
        : options;
    return (
        <div id={"output-storage"}>
            <>
                {(destination === outputStorage.S3 ||
                    destination === outputStorage.GCS ||
                    destination === outputStorage.ADLS ||
                    destination === outputStorage.Bigquery ||
                    destination === outputStorage.Databricks ||
                    destination === outputStorage.Snowflake) && (
                    <GridRow>
                        <GridCol xs={12} sm={12} md={12}>
                            <Label htmlFor="Connector">Connector</Label>
                            <Select
                                id={"connectors"}
                                placeholder="Select Connector"
                                search={false}
                                mode="single"
                                sortOptions={false}
                                popoverClassName="w-200"
                                isInValid={!isValid(selectedConnector)}
                                selectedValues={
                                    isValid(selectedConnector?.value)
                                        ? Array.of(
                                              selectedConnector?.value as string
                                          )
                                        : undefined
                                }
                                options={outputOptions ?? []}
                                onValuesChange={(newValue) => {
                                    const currSel = outputOptions?.find(
                                        (item) =>
                                            newValue.includes(
                                                item?.value as string
                                            )
                                    );
                                    if (
                                        currSel?.value &&
                                        !isAnyUriObjectStoreConnector(
                                            currSel.config
                                        )
                                    ) {
                                        setSelectedConnector(
                                            currSel as ConnectorDropdownOption
                                        );
                                    }
                                }}
                            />
                            {findErrorByTitle(error, "connectors have been")
                                ?.description && (
                                <Description
                                    size="sm"
                                    variant="destructive"
                                    className="!mb-0 leading-5">
                                    {
                                        findErrorByTitle(
                                            error,
                                            "connectors have been"
                                        )?.description
                                    }
                                </Description>
                            )}
                            <Spacer height={16} />
                        </GridCol>
                    </GridRow>
                )}
            </>
            <OutputArtifacts
                onChange={onArtifactChange}
                artifact={artifactType.Predictions}
                classes={classes}
                setClasses={setClasses}
                destination={destination}
                outputs={outputs}
                setOutputs={setOutputs}
                predictionType={predictionType}
                error={predictionErrors}
            />
            <OutputArtifacts
                onChange={onArtifactChange}
                artifact={artifactType.Embeddings}
                classes={classes}
                setClasses={setClasses}
                destination={destination}
                outputs={outputs}
                setOutputs={setOutputs}
                predictionType={predictionType}
                error={embeddingErrors}
            />
        </div>
    );
};
const OutputStorage = WithErrorBoundary(OutputStorageDef);

const ErrorLabelInfo = ({
    color,
    message
}: {
    color: string;
    message: string | ReactElement;
}) => {
    // eslint-disable-next-line no-restricted-syntax -- runtime color prop
    return <span style={{ color: color }}>{message}</span>;
};

/**
 * Output Artiact component
 * @param artifact Artifact Type
 * @param destination selected output storage type
 * @param outputs selected artifactType
 * @param setOutputs callback to update output artifact settings
 * @param predictionType predictive query type
 * @param classes clsses input field value
 * @param setClasses callback to set classes
 * @param extOptions List of extensions
 * @param onChange callback to update select/unselect artifact
 * @param validate callback to validate the config
 */
const OutputArtifactsDef = ({
    artifact,
    destination,
    outputs,
    setOutputs,
    predictionType,
    classes,
    setClasses,
    onChange,
    error
}: {
    artifact: string;
    destination: string;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
    predictionType: string;
    classes: number | string;
    setClasses: (val: number | string) => void;
    onChange: (newVal: string) => void;
    error: ErrorDetail[];
}) => {
    const outputList = getKeys(outputs);
    const output = outputs[artifact];

    const extOptions = [
        { label: ".parquet", value: ".parquet" },
        { label: ".csv", value: ".csv" }
    ];

    return (
        <Card
            id={"prediction-output-artifacts"}
            className={cn(
                "mb-4 gap-0 text-base font-normal",
                outputList.includes(artifact) ? "bg-surface-grey" : "bg-white"
            )}>
            <CardContent>
                <div className="flex items-center">
                    <Checkbox
                        checked={outputList.includes(artifact)}
                        onCheckedChange={() => {
                            onChange(artifact);
                        }}
                        aria-label={"Checkbox-" + artifact}
                        data-testid={"testid-" + artifact}
                        className="cursor-pointer"
                    />
                    <Label className="!mb-0 pl-2 font-normal text-base">
                        {artifact}
                    </Label>
                </div>

                {outputList.includes(artifact) && (
                    <div>
                        {artifact === artifactType.Predictions &&
                            (predictionType ===
                                TaskType.MULTICLASS_CLASSIFICATION ||
                                predictionType ===
                                    TaskType.MULTILABEL_CLASSIFICATION) && (
                                <MaxCategories
                                    artifact={artifact}
                                    classes={classes}
                                    setClasses={setClasses}
                                    predictionType={predictionType}
                                    errorMsg={
                                        findErrorByTitle(
                                            error,
                                            "Predicted Links per Entity"
                                        )?.description
                                    }
                                />
                            )}
                        {artifact === artifactType.Predictions &&
                            (predictionType === TaskType.MULTILABEL_RANKING ||
                                isLinkPredictionType(predictionType)) && (
                                <TopK
                                    artifact={artifact}
                                    classes={classes}
                                    setClasses={setClasses}
                                    predictionType={predictionType}
                                    errorMsg={
                                        findErrorByTitle(
                                            error,
                                            "Predicted Links per Entity"
                                        )?.description
                                    }
                                />
                            )}
                        {destination !== outputStorage.Web && (
                            <>
                                {destination === outputStorage.Databricks && (
                                    <OutputTableSchema
                                        artifact={artifact}
                                        outputs={outputs}
                                        setOutputs={setOutputs}
                                        error={error}
                                    />
                                )}
                                <OutputTable
                                    artifact={artifact}
                                    destination={destination}
                                    outputs={outputs}
                                    setOutputs={setOutputs}
                                    error={error}
                                />

                                {isObjectStoreOutput(destination) && (
                                    <>
                                        <GridRow
                                            xs={12}
                                            sm={12}
                                            md={12}
                                            className="items-center">
                                            <GridCol xs={12} sm={12} md={9}>
                                                <ObjectStoreDirectory
                                                    outputs={outputs}
                                                    setOutputs={setOutputs}
                                                    artifact={artifact}
                                                    destination={destination}
                                                    isInValid={
                                                        !!findErrorByTitle(
                                                            error,
                                                            `${destination} Output directory`
                                                        )?.description
                                                    }
                                                    errorMsg={
                                                        findErrorByTitle(
                                                            error,
                                                            `${destination} Output directory`
                                                        )?.description
                                                    }
                                                />
                                            </GridCol>
                                            <GridCol
                                                xs={12}
                                                sm={12}
                                                md={3}
                                                className="mb-auto">
                                                <Tooltip
                                                    id={"filetype-tooltip"}>
                                                    <TooltipTrigger>
                                                        <span>
                                                            <Label htmlFor="File type">
                                                                File Type
                                                            </Label>
                                                        </span>
                                                    </TooltipTrigger>
                                                    <TooltipContent
                                                        arrow={false}
                                                        side={"right"}
                                                        sideOffset={5}>
                                                        {
                                                            "The file extension determines the data format."
                                                        }
                                                    </TooltipContent>
                                                </Tooltip>

                                                <Extensions
                                                    artifact={artifact}
                                                    outputs={outputs}
                                                    setOutputs={setOutputs}
                                                    extOptions={extOptions}
                                                />
                                            </GridCol>
                                        </GridRow>
                                    </>
                                )}
                                <GridRow xs={12} sm={12} md={12}>
                                    {output.errorInfo && (
                                        <>
                                            <Spacer height={5} />
                                            <ErrorLabelInfo
                                                color={output.errorInfo.color}
                                                message={output.errorInfo.info}
                                            />
                                        </>
                                    )}
                                </GridRow>
                            </>
                        )}
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

const OutputArtifacts = WithErrorBoundary(OutputArtifactsDef);

const MaxCategories = ({
    artifact,
    classes,
    setClasses,
    predictionType,
    errorMsg
}: {
    artifact: string;
    classes: number | string;
    setClasses: (val: number | string) => void;
    predictionType: string;
    errorMsg?: string;
}) => {
    return (
        <>
            <GridRow xs={12} sm={12} md={12} className="mt-2">
                <GridCol xs={12} sm={12}>
                    <TextInput
                        id={artifact + "_max_category"}
                        label={"Max Categories"}
                        labelToolTip={
                            "If provided, only return predictions for the top K " +
                            "classes with the highest prediction per entity. If not " +
                            "provided, predictions for all classes will be returned. " +
                            "This is useful for reducing the size of the output table, " +
                            "when making predictions for a large number of entities."
                        }
                        placeholder={"All"}
                        value={classes?.toString()}
                        isInvalid={
                            !isValidClassNumber(
                                predictionType,
                                classes?.toString()
                            )
                        }
                        onChange={(e) => {
                            const inputValue = e?.target?.value;
                            if (
                                inputValue === "" ||
                                digitsRegex.test(inputValue)
                            ) {
                                setClasses(
                                    inputValue !== ""
                                        ? parseInt(inputValue)
                                        : ""
                                );
                            }
                        }}
                        errorMsg={errorMsg}
                    />
                </GridCol>
            </GridRow>
            <Spacer height={16} />
        </>
    );
};

const isValidClassNumber = (predictionType: string, classes: string) => {
    if (predictionType === TaskType.BINARY_CLASSIFICATION) {
        return (
            (classes?.toString()?.trim() === "" || digitsRegex.test(classes)) &&
            (Number(classes) > 0 || classes.trim() === "")
        );
    } else if (
        isLinkPredictionType(predictionType) ||
        predictionType === TaskType.MULTILABEL_RANKING ||
        predictionType === TaskType.MULTILABEL_CLASSIFICATION ||
        predictionType === TaskType.MULTICLASS_CLASSIFICATION
    ) {
        return (
            classes?.toString()?.trim() !== "" &&
            digitsRegex.test(classes) &&
            Number(classes) > 0
        );
    } else {
        return true;
    }
};

const TopK = ({
    artifact,
    classes,
    setClasses,
    predictionType,
    errorMsg
}: {
    artifact: string;
    classes: number | string;
    setClasses: (val: number | string) => void;
    predictionType: string;
    errorMsg?: string;
}) => {
    const isInvalid = !isValidClassNumber(predictionType, classes?.toString());
    const isRequired =
        predictionType === TaskType.MULTILABEL_RANKING ||
        isLinkPredictionType(predictionType);

    return (
        <>
            <GridRow xs={12} sm={12} md={12} className="mt-2">
                <GridCol xs={12} sm={12}>
                    <TextInput
                        id={artifact + "_top_k"}
                        label={"Predicted Links per Entity"}
                        labelToolTip={
                            "Only returns the top K predictions for each entity. " +
                            "This is useful for only keeping the K most " +
                            "confident predictions for each entity. "
                        }
                        placeholder={
                            predictionType ===
                            TaskType.MULTILABEL_CLASSIFICATION
                                ? "All"
                                : "No predictions"
                        }
                        value={classes?.toString()}
                        isInvalid={isInvalid}
                        errorMsg={
                            errorMsg ||
                            (isInvalid && isRequired
                                ? "This field is required and must be greater than 0"
                                : undefined)
                        }
                        onChange={(e) => {
                            const inputValue = e?.target?.value;
                            if (
                                inputValue === "" ||
                                digitsRegex.test(inputValue)
                            ) {
                                setClasses(
                                    inputValue !== ""
                                        ? parseInt(inputValue)
                                        : ""
                                );
                            }
                        }}
                    />
                </GridCol>
            </GridRow>
            <Spacer height={16} />
        </>
    );
};

const ObjectStoreDirectory = ({
    artifact,
    destination,
    outputs,
    isInValid,
    errorMsg,
    setOutputs
}: {
    artifact: string;
    destination: string;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    isInValid: boolean;
    errorMsg?: string;
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
}) => {
    return (
        <>
            <TextInput
                id={"s3_directory"}
                label={"Output Directory"}
                labelToolTip={
                    "The directory where the output file will be stored."
                }
                placeholder={
                    destination === outputStorage.S3
                        ? "s3://<bucket>/<key prefix>"
                        : destination === outputStorage.GCS
                          ? "gs://<bucket>/<key prefix>"
                          : "abfss://<filesystem>@<account>.<endpoint>/<path>"
                }
                value={outputs[artifact]?.directory}
                isInvalid={isInValid}
                errorMsg={errorMsg}
                onChange={(e) => {
                    const val = e?.target?.value;
                    const tmpOutputs = outputs;
                    tmpOutputs[artifact].directory = val;
                    setOutputs(copyObject(tmpOutputs));
                }}
                onBlur={(e) => {
                    let val = e?.target?.value;
                    val = val?.trim();
                    if (isValid(val) && val.slice(-1) !== "/") {
                        const tmpOutputs = outputs;
                        tmpOutputs[artifact].directory = val + "/";
                        setOutputs(copyObject(tmpOutputs));
                    }
                }}
            />
        </>
    );
};

const OutputTableSchema = ({
    artifact,
    outputs,
    setOutputs,
    error
}: {
    artifact: string;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
    error: ErrorDetail[];
}) => {
    return (
        <>
            <GridRow xs={12} sm={12} md={12} className="mt-2">
                <GridCol xs={12} sm={12}>
                    <TextInput
                        id={"s3_directory"}
                        label={"Table Schema"}
                        labelToolTip={
                            "The name of the table schema where the data will be written."
                        }
                        placeholder={"Table Schema"}
                        value={outputs[artifact]?.table_schema}
                        isInvalid={!isValid(outputs[artifact].table_schema)}
                        errorMsg={
                            findErrorByTitle(error, "table schema is invalid")
                                ?.description
                        }
                        onChange={(e) => {
                            const tmpOutputs = outputs;
                            tmpOutputs[artifact].table_schema =
                                e?.target?.value;
                            setOutputs(copyObject(tmpOutputs));
                        }}
                        onBlur={() => {
                            const tmpOutputs = outputs;
                            tmpOutputs[artifact].table_schema =
                                tmpOutputs[artifact].table_schema.trim();
                            setOutputs(copyObject(tmpOutputs));
                        }}
                    />
                </GridCol>
            </GridRow>
            <Spacer height={16} />
        </>
    );
};

const OutputTable = ({
    artifact,
    destination,
    outputs,
    setOutputs,
    error
}: {
    artifact: string;
    destination: string;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
    error: ErrorDetail[];
}) => {
    let label = "File Name";
    if (
        destination === outputStorage.Snowflake ||
        destination === outputStorage.Databricks ||
        destination === outputStorage.Bigquery
    ) {
        label = "Table Name";
    }

    return (
        <>
            <GridRow xs={12} sm={12} md={12} className="mt-2">
                <GridCol xs={12} sm={12}>
                    <TextInput
                        id={"table_name"}
                        label={label}
                        labelToolTip={
                            destination === outputStorage.Databricks
                                ? "The name of the table where the data will be written."
                                : isObjectStoreOutput(destination)
                                  ? "The name of the output file, excluding the file extension."
                                  : destination === outputStorage.Snowflake
                                    ? "The name of the table in snowflake where the data will be written."
                                    : destination === outputStorage.Bigquery
                                      ? "The name of the table in BigQuery where the data will be written."
                                      : ""
                        }
                        placeholder={label}
                        value={outputs[artifact].table_name}
                        isInvalid={!isValid(outputs[artifact]?.table_name)}
                        errorMsg={
                            findErrorByTitle(error, "table name is invalid")
                                ?.description
                        }
                        onChange={(e) => {
                            const tmpOutputs = outputs;
                            tmpOutputs[artifact].table_name = e?.target?.value;
                            setOutputs(copyObject(tmpOutputs));
                        }}
                        onBlur={() => {
                            const tableName =
                                outputs[artifact].table_name.trim();
                            if (tableName === outputs[artifact].table_name) {
                                return;
                            }
                            const tmpOutputs = outputs;
                            tmpOutputs[artifact].table_name = tableName;
                            setOutputs(copyObject(tmpOutputs));
                        }}
                    />
                </GridCol>
            </GridRow>
            {label === "File Name" && <Spacer height={16} />}
        </>
    );
};

const Extensions = ({
    artifact,
    outputs,
    setOutputs,
    extOptions
}: {
    artifact: string;
    outputs: { [key: string]: BatchPredictionJobOutputJs };
    setOutputs: (val: { [key: string]: BatchPredictionJobOutputJs }) => void;
    extOptions: DropdownOption[];
}) => {
    const selected = outputs[artifact]?.extension;

    return (
        <Select
            id={"ext-select"}
            placeholder="File Type"
            search={false}
            mode="single"
            popoverClassName="w-fit"
            selectedValues={selected ? Array.of(selected) : undefined}
            options={extOptions?.map((item) => {
                return {
                    value: item?.value as string,
                    label: item?.label as string
                };
            })}
            onValuesChange={(newValue) => {
                const tmpOutputs = outputs;
                const currSel = extOptions?.find((item) =>
                    newValue.includes(item?.value as string)
                );
                if (currSel?.value) {
                    tmpOutputs[artifact].extension = currSel?.value as string;
                    setOutputs(copyObject(tmpOutputs));
                }
            }}
        />
    );
};
