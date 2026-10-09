// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";
import Spacer from "react-spacer";

import {
    CardContent,
    GridCol,
    Title,
    useWithDebounce
} from "@kumo-ai/kumo-ui-catalog";
import { GenericJob } from "gen/v0/types";
import { ErrorDetail, ErrorType } from "gen/v1/types";
import { getCurrentUserDisplayName } from "stores/userStore";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useConnectors } from "hooks/connector/connector";
import {
    useAllowedBpQueryOverrides,
    usePredictionFilterOverrides,
    usePredictionOutputs
} from "hooks/jobs/prediction/prediction";
import { useProjectId } from "hooks/project/projects";

import { BatchPredictionJobOutputJs } from "types/batchPredictions";
import { CustomError, HighLightText } from "types/common";
import { ConnectorDropdownOption, InputState } from "types/inputs";
import { CreatePredictionJobArgs } from "types/predictionJob";
import {
    PredictiveQueryInfo,
    QueryType,
    TaskType
} from "types/predictiveQueries";

import { initializeOutputs } from "utils/batchPredictions";
import {
    artifactType,
    floatRegex,
    outputStorage,
    warehouseSource
} from "utils/Constants";
import { copyObject, getKeys } from "utils/Helpers/objectHelpers";
import { isLinkPredictionType, isValid } from "utils/Helpers/typeGuards";
import {
    filePathForRootedObjectStoreConnector,
    isADLSConnectorConfig,
    isAnyUriObjectStoreConnector,
    isGCSConnectorConfig,
    isS3ConnectorConfig
} from "utils/objectStoreTablePath";

import PredictionAnchorTime from "./PredictionAnchorTime";
import PredictionExplanations from "./PredictionExplanations";
import PredictionFilter from "./PredictionFilter";
import PredictionOutput from "./PredictionOutput";
import PredictionTextInput from "./PredictionTextInput";

const CONNECTOR_LIST_KEY_BY_SOURCE: Partial<Record<warehouseSource, string>> = {
    [warehouseSource.S3]: "s3Connectors",
    [warehouseSource.GCS]: "gcsConnectors",
    [warehouseSource.ADLS]: "adlsConnectors",
    [warehouseSource.SnowFlake]: "snowFlakeConnectors",
    [warehouseSource.BigQuery]: "bigQueryConnectors",
    [warehouseSource.Databricks]: "dataBricksConnectors"
};

const CONNECTOR_LIST_KEY_BY_DESTINATION: Partial<
    Record<outputStorage, string>
> = {
    [outputStorage.S3]: "s3Connectors",
    [outputStorage.GCS]: "gcsConnectors",
    [outputStorage.ADLS]: "adlsConnectors",
    [outputStorage.Snowflake]: "snowFlakeConnectors",
    [outputStorage.Bigquery]: "bigQueryConnectors",
    [outputStorage.Databricks]: "dataBricksConnectors"
};

const isObjectStoreOutput = (destination: string): boolean =>
    [outputStorage.S3, outputStorage.GCS, outputStorage.ADLS].includes(
        destination as outputStorage
    );

export interface PredictionSettingsProps {
    selectedTrainingJob: (GenericJob & PredictiveQueryInfo) | null;
    setPredictionJobArgs: React.Dispatch<
        React.SetStateAction<CreatePredictionJobArgs | null>
    >;
    setError: React.Dispatch<React.SetStateAction<ErrorDetail[]>>;
    setFilterErrors: React.Dispatch<React.SetStateAction<ErrorDetail[]>>;
    setValidationState: React.Dispatch<React.SetStateAction<InputState>>;
    setFilterValidState: React.Dispatch<React.SetStateAction<InputState>>;
    setHighLightQuery: React.Dispatch<React.SetStateAction<HighLightText>>;
    error: ErrorDetail[];
}

const PredictionSettings: React.FC<PredictionSettingsProps> = ({
    selectedTrainingJob,
    setPredictionJobArgs,
    setError,
    setFilterErrors,
    setFilterValidState,
    setValidationState,
    setHighLightQuery,
    error
    // eslint-disable-next-line complexity -- coordinates the prediction form's dependent state and render branches
}) => {
    const [destination, setDestination] = useState<string>(outputStorage.S3);
    const [connectors, setConnectors] = useState<{
        [key: string]: Array<ConnectorDropdownOption>;
    }>({});
    const [selectedConnector, setSelectedConnector] =
        useState<ConnectorDropdownOption | null>(null);
    const [classes, setClasses] = useState<string | number>(
        isLinkPredictionType(selectedTrainingJob?.taskType as string) ? "" : ""
    );
    const [outputs, setOutputs] = useState<{
        [key: string]: BatchPredictionJobOutputJs;
    }>({});
    const outputRevision = useRef(0);

    const [numParallelWorkers, setNumParallelWorkers] = useState<string>("1");
    const [bcThreshold, setBcThreshold] = useState<string>("0.5");
    const [enableExplanations, setEnableExplanations] =
        useState<boolean>(false);

    const [entityFilter, setEntityFilter] = useState<string>("");
    const [targetFilter, setTargetFilter] = useState<string>("");

    const [predictionTime, setPredictionTime] = useState<string>("inferred");

    const projectId = useProjectId();
    const {
        isDataLoading: isConnectorDataLoading,
        connectorList: connectorListOriginal
    } = useConnectors(undefined, undefined, Boolean(projectId));

    useAllowedBpQueryOverrides({
        selectedJobId: selectedTrainingJob?.id || "",
        setEntityFilter,
        setTargetFilter
    });

    const { highLightQueryText } = usePredictionFilterOverrides({
        selectedTrainingJobId: selectedTrainingJob?.id || "",
        entityFilter,
        targetFilter,
        setFilterErrors,
        setFilterValidState,
        setHighLightQuery
    });

    useEffect(() => {
        if (selectedTrainingJob?.id) {
            const initialized = initializeOutputs(
                selectedTrainingJob.id,
                artifactType.Predictions,
                destination,
                null
            );
            setOutputs(copyObject(initialized));
        }
    }, [selectedTrainingJob?.id, destination]);

    const outputStore = usePredictionOutputs(
        destination,
        outputs,
        selectedConnector
    );

    const onDateTimeChange = (dateTime: Date) => {
        if (dateTime) {
            const year = dateTime.getFullYear();
            const month = (dateTime.getMonth() + 1)
                ?.toString()
                .padStart(2, "0"); // Month is 0-based
            const day = dateTime.getDate()?.toString()?.padStart(2, "0");

            const hour = dateTime.getHours()?.toString()?.padStart(2, "0");
            const minute = dateTime.getMinutes()?.toString()?.padStart(2, "0");
            const second = dateTime.getSeconds()?.toString()?.padStart(2, "0");

            const predictionTimeString =
                year +
                "-" +
                month +
                "-" +
                day +
                "T" +
                hour +
                ":" +
                minute +
                ":" +
                second;
            setPredictionTime(predictionTimeString);
        }
    };

    const onThresholdChange = (value: string) => {
        if (value === "" || floatRegex.test(value)) {
            if (value.split(".").length > 1) {
                if (value.split(".")[1].length > 2) {
                    value = value.substring(0, 4);
                }
            }
            setBcThreshold(value);
        }
    };

    const isValidBinaryThreshold = useCallback(() => {
        let retValue = false;
        if (bcThreshold && bcThreshold?.toString()?.trim()) {
            if (parseFloat(bcThreshold) >= 0 && parseFloat(bcThreshold) <= 1) {
                retValue = true;
            }
        }
        return retValue;
    }, [bcThreshold]);

    const onOutputStorageChange = (newVal: string) => {
        setDestination(newVal);

        if (newVal !== outputStorage.LocalDownloadOnly) {
            const newOutputs = initializeOutputs(
                selectedTrainingJob?.id as string,
                artifactType.Predictions,
                newVal,
                outputs as {
                    [key: string]: BatchPredictionJobOutputJs;
                }
            );
            setOutputs(copyObject(newOutputs));
            setClasses("");
        } else {
            const tmpOutputs: {
                [key: string]: BatchPredictionJobOutputJs;
            } = {};
            setOutputs(copyObject(tmpOutputs));
        }
        setSelectedConnector(null);
    };

    const onArtifactChange = (newVal: string) => {
        if (outputs[newVal]) {
            const tmpOutputs = outputs;
            delete tmpOutputs[newVal];
            setOutputs(copyObject(tmpOutputs));
        } else {
            const tmpOutputs = outputs;
            setOutputs({
                ...tmpOutputs,
                ...copyObject(
                    initializeOutputs(
                        selectedTrainingJob?.id as string,
                        newVal,
                        destination
                    )
                )
            });
            if (newVal === artifactType.Predictions) {
                setClasses("");
            }
        }
    };

    const updateConnectorsList = useCallback(async () => {
        const connectors: {
            [key: string]: Array<ConnectorDropdownOption>;
        } = {
            s3Connectors: [],
            gcsConnectors: [],
            adlsConnectors: [],
            snowFlakeConnectors: [],
            bigQueryConnectors: [],
            dataBricksConnectors: []
        };

        if (isConnectorDataLoading || connectorListOriginal?.length < 1) {
            return;
        }

        try {
            for (let i = 0; i < connectorListOriginal?.length; i++) {
                const data = connectorListOriginal[i];
                const connector: ConnectorDropdownOption = {
                    value: data?.id,
                    label: data?.name ?? data?.id,
                    config: data?.config
                };
                const connectorListKey =
                    CONNECTOR_LIST_KEY_BY_SOURCE[
                        data?.config?.type as warehouseSource
                    ];
                if (connectorListKey) {
                    connectors[connectorListKey].push(connector);
                }
            }
            setConnectors(connectors);
        } catch (error) {
            console.error(
                "Error in getConnectorsList",
                (error as CustomError).message
            );
        }
    }, [connectorListOriginal, isConnectorDataLoading]);

    const handleExplanationsChange = useCallback(() => {
        setEnableExplanations(!enableExplanations);
    }, [enableExplanations, setEnableExplanations]);

    const createPredictionJobArgs = useCallback(() => {
        const args: CreatePredictionJobArgs = {
            project_id: projectId ?? "",
            model_training_job_id: selectedTrainingJob?.trainingJobId as string,
            graph_snapshot_id: selectedTrainingJob?.graphSnapshotId,
            explanations: enableExplanations,
            filter_overrides: {
                entity_filter: entityFilter,
                target_filter: targetFilter
            },
            job_tags: {},
            predict_options: {
                num_workers: parseInt(numParallelWorkers)
            },
            predictive_query: null,
            user: getCurrentUserDisplayName()
        };

        const graph_snapshot_id = selectedTrainingJob?.graphSnapshotId;
        if (graph_snapshot_id != null) {
            args.graph_id = graph_snapshot_id.split("@")[0].split("graph-")[1];
        }

        if (selectedTrainingJob?.taskType === TaskType.BINARY_CLASSIFICATION) {
            args.predict_options.binary_classification_threshold =
                parseFloat(bcThreshold);
        }

        if (isValid(predictionTime)) {
            args.anchor_time = predictionTime;
        }

        if (isValid(classes)) {
            args.predict_options.num_classes_to_return = Number(classes);
        }

        if (outputStore?.length) {
            args.outputs = outputStore;
        }

        if (isValid(highLightQueryText)) {
            args.predictive_query = highLightQueryText;
        }

        return args;
    }, [
        bcThreshold,
        classes,
        enableExplanations,
        entityFilter,
        highLightQueryText,
        numParallelWorkers,
        outputStore,
        predictionTime,
        projectId,
        selectedTrainingJob?.graphSnapshotId,
        selectedTrainingJob?.taskType,
        selectedTrainingJob?.trainingJobId,
        targetFilter
    ]);
    const createPredictionJobArgsRef = useRef(createPredictionJobArgs);
    createPredictionJobArgsRef.current = createPredictionJobArgs;

    useEffect(() => {
        updateConnectorsList();
    }, [updateConnectorsList]);

    useEffect(() => {
        outputRevision.current += 1;
        setPredictionJobArgs(createPredictionJobArgsRef.current());
        setValidationState(InputState.Validating);
    }, [
        destination,
        outputs,
        selectedConnector,
        setPredictionJobArgs,
        setValidationState
    ]);

    const outputRevisionAtRender = outputRevision.current;

    // eslint-disable-next-line complexity -- reports every independent prediction validation error in one pass
    const validatePredictionJob = useCallback(async () => {
        if (outputRevisionAtRender !== outputRevision.current) return;
        const error: ErrorDetail[] = [];
        if (
            Boolean(
                parseInt(numParallelWorkers) && parseInt(numParallelWorkers) > 0
            ) == false
        ) {
            error.push({
                title: "Number of parallel workers is invalid.",
                description: `Please enter Number of parallel workers which must be greater than 0`,
                type: ErrorType.NUMBER_0,
                cta: null
            });
        }
        const value = Number(classes);
        if (
            getKeys(outputs).includes(artifactType.Predictions) &&
            (selectedTrainingJob?.taskType ===
                TaskType.MULTICLASS_CLASSIFICATION ||
                selectedTrainingJob?.taskType ===
                    TaskType.MULTILABEL_CLASSIFICATION ||
                selectedTrainingJob?.taskType === TaskType.MULTILABEL_RANKING ||
                (selectedTrainingJob?.taskType &&
                    isLinkPredictionType(selectedTrainingJob?.taskType))) &&
            (!isValid(classes) || !Number.isFinite(value) || value < 1)
        ) {
            error.push({
                title: "Predicted Links per Entity or Max Categories is invalid for PREDICTIONS.",
                description:
                    "Please enter Predicted Links per Entity or Max Categories which must be a positive number.",
                type: ErrorType.NUMBER_0,
                cta: null
            });
        }
        // Check if TopK is required but empty for multilabel ranking and link prediction types
        if (
            (selectedTrainingJob?.taskType === TaskType.MULTILABEL_RANKING ||
                isLinkPredictionType(
                    selectedTrainingJob?.taskType as string
                )) &&
            (!isValid(classes) ||
                classes?.toString()?.trim() === "" ||
                parseInt(classes as string) <= 0) &&
            getKeys(outputs).includes(artifactType.Predictions)
        ) {
            error.push({
                title: "Predicted Links per Entity is required.",
                description:
                    "Please enter Predicted Links per Entity which must be a positive number greater than 0.",
                type: ErrorType.NUMBER_0,
                cta: null
            });
        }

        if (selectedTrainingJob?.taskType === TaskType.BINARY_CLASSIFICATION) {
            if (!isValidBinaryThreshold()) {
                error.push({
                    title: "Threshold for binary classification is invalid.",
                    description:
                        "Please enter Threshold for binary classification which must be between 0 and 1.",
                    type: ErrorType.NUMBER_0,
                    cta: null
                });
            }
        }

        if (
            !getKeys(outputs).includes(artifactType.Predictions) &&
            !getKeys(outputs).includes(artifactType.Embeddings) &&
            destination !== outputStorage.LocalDownloadOnly
        ) {
            error.push({
                title: "Missing output artifact.",
                description: "Please select at least one output artifact.",
                type: ErrorType.NUMBER_0,
                cta: null
            });
        }
        if (destination !== outputStorage.LocalDownloadOnly) {
            // eslint-disable-next-line complexity -- validates storage-specific fields for each output artifact
            getKeys(outputs).forEach((output) => {
                const tmp = {
                    storage_type: destination,
                    artifact_type: ""
                };
                const outputItem = outputs[output];
                tmp.artifact_type = outputItem?.artifact_type;
                if (isObjectStoreOutput(tmp.storage_type)) {
                    const dir = outputItem?.directory;
                    let validDestination = false;
                    const config = selectedConnector?.config;
                    if (
                        isValid(dir) &&
                        isValid(outputItem?.table_name) &&
                        ((tmp.storage_type === outputStorage.S3 &&
                            isS3ConnectorConfig(config)) ||
                            (tmp.storage_type === outputStorage.GCS &&
                                isGCSConnectorConfig(config)) ||
                            (tmp.storage_type === outputStorage.ADLS &&
                                isADLSConnectorConfig(config)))
                    ) {
                        const separator = dir.endsWith("/") ? "" : "/";
                        const filePath = `${dir}${separator}${outputItem.table_name}${outputItem.extension}`;
                        try {
                            filePathForRootedObjectStoreConnector(
                                filePath,
                                config
                            );
                            validDestination = true;
                        } catch {
                            validDestination = false;
                        }
                    }
                    if (!validDestination) {
                        error.push({
                            title: `${destination} Output directory is invalid for ${tmp.artifact_type}.`,
                            description:
                                destination === outputStorage.S3
                                    ? "Enter a deep relative or matching full S3 path beneath the selected rooted connector."
                                    : `Enter a relative path or a matching full ${destination} URI beneath the selected rooted connector.`,
                            type: ErrorType.NUMBER_0,
                            cta: null
                        });
                    }

                    if (!isValid(outputItem?.table_name)) {
                        error.push({
                            title: `${destination} connectors table name is invalid for ${tmp.artifact_type}.`,
                            description: `Please enter ${destination} connector's table name for ${tmp.artifact_type}.`,
                            type: ErrorType.NUMBER_0,
                            cta: null
                        });
                    }
                }
                if (
                    CONNECTOR_LIST_KEY_BY_DESTINATION[
                        tmp.storage_type as outputStorage
                    ]
                ) {
                    const connectorListKey =
                        CONNECTOR_LIST_KEY_BY_DESTINATION[
                            destination as outputStorage
                        ];
                    const destinationConnectors = connectorListKey
                        ? connectors?.[connectorListKey]
                        : undefined;
                    const hasDestinationConnectors = Boolean(
                        isValid(destinationConnectors)
                    );
                    const connectorsFound = Boolean(
                        hasDestinationConnectors &&
                        destinationConnectors?.some(
                            (connector) =>
                                !isObjectStoreOutput(destination) ||
                                !isAnyUriObjectStoreConnector(connector.config)
                        )
                    );

                    if (!connectorsFound) {
                        const onlySourceConnectors =
                            isObjectStoreOutput(destination) &&
                            hasDestinationConnectors;
                        error.push({
                            title: onlySourceConnectors
                                ? `No writable/rooted ${destination} connectors have been added to Kumo.`
                                : `No ${destination} connectors have been added to Kumo.`,
                            description: onlySourceConnectors
                                ? `Only source-only any-URI ${destination} connectors are available. Please configure a rooted connector where the ${tmp.artifact_type} table will be saved.`
                                : `Please configure a new ${destination} connector where the ${tmp.artifact_type} table will be saved.`,
                            type: ErrorType.NUMBER_0,
                            cta: null
                        });
                    }

                    if (connectorsFound && !isValid(selectedConnector)) {
                        error.push({
                            title: `No ${destination} connectors have been selected.`,
                            description: `Please select the ${destination} connector where the ${tmp.artifact_type} table will be saved.`,
                            type: ErrorType.NUMBER_0,
                            cta: null
                        });
                    }

                    if (
                        !isObjectStoreOutput(tmp.storage_type) &&
                        !isValid(outputItem?.table_name)
                    ) {
                        error.push({
                            title: `${destination} connectors table name is invalid for ${tmp.artifact_type}.`,
                            description: `Please enter ${destination} connector's table name for ${tmp.artifact_type}.`,
                            type: ErrorType.NUMBER_0,
                            cta: null
                        });
                    }
                }

                if (tmp.storage_type === outputStorage.Databricks) {
                    if (!isValid(outputItem?.table_schema)) {
                        error.push({
                            title: `${destination} connectors table schema is invalid for ${tmp.artifact_type}.`,
                            description: `Please enter ${destination} connector's table schema for ${tmp.artifact_type}.`,
                            type: ErrorType.NUMBER_0,
                            cta: null
                        });
                    }
                }
            });
        }

        if (error.length === 0) {
            setValidationState(InputState.Valid);
            setError([]);
            setPredictionJobArgs(createPredictionJobArgs());
        } else {
            setValidationState(InputState.ValidationError);
            setError(error);
        }
    }, [
        classes,
        connectors,
        createPredictionJobArgs,
        destination,
        isValidBinaryThreshold,
        numParallelWorkers,
        outputs,
        outputRevisionAtRender,
        selectedConnector,
        selectedTrainingJob?.taskType,
        setError,
        setPredictionJobArgs,
        setValidationState
    ]);

    const restartPredictionJobValidation = useWithDebounce(
        validatePredictionJob,
        2
    );

    useEffect(() => {
        restartPredictionJobValidation();
    }, [restartPredictionJobValidation]);

    return (
        <GridCol
            data-testid="new-prediction-settings-panel"
            xs={12}
            sm={12}
            md={6}
            className="overflow-y-auto h-full">
            <Title variant="secondary-heading" className="py-4 px-4">
                Prediction Settings
            </Title>
            <CardContent className="py-2 pl-4 pr-4">
                <PredictionFilter
                    title="Prediction Entity Filter"
                    value={entityFilter}
                    onChange={(value: string) => {
                        setEntityFilter(value);
                    }}
                />
                {isLinkPredictionType(
                    selectedTrainingJob?.taskType as TaskType
                ) && (
                    <>
                        <Spacer height={16} />
                        <PredictionFilter
                            title="Prediction Target Filter"
                            value={targetFilter}
                            onChange={(value: string) => {
                                setTargetFilter(value);
                            }}
                        />
                    </>
                )}
                <Spacer height={16} />
                {selectedTrainingJob?.queryType !== QueryType.STATIC && (
                    <PredictionAnchorTime onChange={onDateTimeChange} />
                )}
                {selectedTrainingJob?.taskType ===
                    TaskType.BINARY_CLASSIFICATION && (
                    <>
                        <PredictionTextInput
                            id="binary-classification-threshold"
                            title="Threshold for Binary Classification"
                            placeholder="Threshold for Binary Classification"
                            value={bcThreshold}
                            errorMsg={
                                error?.find((err) =>
                                    err?.title?.includes(
                                        "binary classification"
                                    )
                                )?.description
                            }
                            isInvalid={!isValidBinaryThreshold()}
                            onChange={onThresholdChange}
                        />
                        <Spacer height={16} />
                    </>
                )}
                <PredictionTextInput
                    id={"parallel-workers-number"}
                    title="Number of parallel workers"
                    placeholder={"Number of parallel workers"}
                    isInvalid={
                        Boolean(
                            parseInt(numParallelWorkers) &&
                            parseInt(numParallelWorkers) > 0 &&
                            parseInt(numParallelWorkers) <= 4
                        ) == false
                    }
                    errorMsg={
                        error?.find((err) =>
                            err?.title?.includes(
                                "Number of parallel workers is invalid"
                            )
                        )?.description
                    }
                    value={numParallelWorkers}
                    onChange={(v) => {
                        setNumParallelWorkers(
                            v?.replace(/[^\w\s]/gi, "")?.slice(0, 3)
                        );
                    }}
                />
                <Spacer height={16} />
                <PredictionExplanations
                    isToggled={enableExplanations}
                    onChange={handleExplanationsChange}
                />
                <PredictionOutput
                    destination={destination}
                    onSelect={onOutputStorageChange}
                    onArtifactChange={onArtifactChange}
                    classes={classes}
                    setClasses={setClasses}
                    outputs={outputs}
                    setOutputs={setOutputs}
                    predictionType={selectedTrainingJob?.taskType as string}
                    selectedConnector={
                        selectedConnector as ConnectorDropdownOption
                    }
                    setSelectedConnector={setSelectedConnector}
                    connectors={connectors}
                    error={error}
                />
            </CardContent>
        </GridCol>
    );
};

export default WithErrorBoundary(PredictionSettings);
