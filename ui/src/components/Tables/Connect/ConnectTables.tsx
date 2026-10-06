// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState
} from "react";
import { useLocation } from "react-router";
import Spacer from "react-spacer";

import {
    Button,
    cn,
    Description,
    getFromSessionStorage,
    GridCol,
    GridRow,
    Label,
    Select,
    storeToSessionStorage,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { DocumentationLink } from "@kumo-ai/kumo-ui-catalog";
import { useGetProjectsConnectorByProjectId } from "gen/v0/api/connectors/connectors";
import {
    DatabricksSourceTable,
    KumoapiTableTableDefinition,
    KumoapiTableTableDefinitionSourceTable
} from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import Validation from "components/Validation/Validation";

import { useConnectorDropdownOptions } from "hooks/connector/connector";
import { useProjectId } from "hooks/project/projects";
import { useKeyColumnTypeValidation } from "hooks/tables/columnTypeValidation";
import { SubmitTableContext, useSubmitTable } from "hooks/tables/newTable";
import { useTables } from "hooks/tables/tables";
import useFeatureFlags from "hooks/utils/featureFlags";
import { useValidationState } from "hooks/validation/validation";

import { CustomError, EntityErrorMsg } from "types/common";
import { ConnectorConfig } from "types/connectors";
import { GraphTables } from "types/graphs";
import {
    ConnectorDropdownOption,
    DropdownOption,
    InputState,
    TableDropdownOption
} from "types/inputs";
import { DataSourceType, FlowType, TableListResponse } from "types/tables";

import {
    ADD_KUMO_ENTITY_VALUE,
    docURLs,
    errorStatusCode,
    getProjectPath,
    storageKeys,
    warehouseSource
} from "utils/Constants";
import {
    getConnectorIDFromTableDefSource,
    getSourceInfoFromConnectorConfig,
    getSourceNameFromTableDefSource
} from "utils/Converters/tablesConverter";
import { replaceSchemaTableNameSplitter } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import {
    defaultTableData,
    detectPrimaryKey,
    generateUniqueTableName,
    getInitialTableArgs,
    getMissingColumnError,
    getSourceTypeOptions,
    localUploadSource,
    removeUnderScore
} from "utils/tables";

import ConnectorSelection from "./Connector";
import CustomSqlTableSource from "./CustomSqlTableSource";
import { formLabelClassName, formRowClassName } from "./formRow";
import LocalUploadFlow from "./LocalUploadFlow";
import ValidatedNameInput from "./NameInput";
import Preprocessing from "./Preprocess";
import SourceTable from "./SourceTable";

interface ConnectTablesProps {
    hideTitle?: boolean;
    addTable?: (table: GraphTables) => void;
    refetchTable?: (id: string) => void;
}

// TextInput's validation-state icon sits in a fixed slot beside the field, so
// the box would stop 3rem short of the dropdowns above it. Hang that slot
// outside the field column instead, leaving every control the same width.
const stateIconOutdentClassName = "[&>.flex-row]:-mr-12";
const sectionClassName = "mt-8 flex flex-col gap-4";
// Form sections are held to a readable measure; Columns and Query keep the
// full width for the column grid and the SQL editor.
const formSectionClassName = cn(sectionClassName, "max-w-3xl");
const sectionTitleClassName = "font-bold text-title-primary";

const ErrorLabel = ({
    entity,
    errLabelMsg
}: {
    entity: string;
    errLabelMsg: EntityErrorMsg;
}) =>
    errLabelMsg?.entity === entity && errLabelMsg.errorMsg !== "" ? (
        <Description
            size="sm"
            variant="destructive"
            className="!mb-0 leading-5">
            {errLabelMsg.errorMsg}
        </Description>
    ) : null;

// Write the selected source-table name (and, for local file-upload, its
// originating connector) onto the table definition's source_table.
export const applySelectedSourceTable = (
    sourceTable: KumoapiTableTableDefinitionSourceTable,
    e: TableDropdownOption
) => {
    switch (sourceTable?.data_source_type) {
        case DataSourceType.BigQuery:
            sourceTable.table_name = e?.value;
            break;
        case DataSourceType.SnowFlake:
            sourceTable.table = e?.value as string;
            break;
        case DataSourceType.Databricks:
            sourceTable.table = e?.value as string;
            break;
        case DataSourceType.S3:
        case DataSourceType.GCS:
        case DataSourceType.ADLS:
            sourceTable.source = e.objectStoreSource ?? {
                mode: "relative",
                path: e.value
            };
            if (!e.fileType) {
                throw new Error(
                    `File type is required for ${sourceTable.data_source_type} table ${e.value}.`
                );
            }
            sourceTable.file_type = e.fileType;
            if (e?.connectorId) sourceTable.connector_id = e.connectorId;
            break;
        default:
            break;
    }
};

const ConnectTables: React.FC<ConnectTablesProps> = ({
    hideTitle = false,
    addTable,
    refetchTable
}) => {
    const featureFlags = useFeatureFlags();
    const SourceType = useMemo(
        () => getSourceTypeOptions(featureFlags),
        [featureFlags]
    );

    const projectId = useProjectId();
    const projectPaths = getProjectPath(projectId);

    const location = useLocation();
    const hasLocalUploadSource = SourceType.some(
        (source) => source.value === localUploadSource.value
    );
    // Persist state.islocal to sessionStorage on entry so subsequent renders
    // (after React Router clears history state) still see the flag.
    useEffect(() => {
        if (location?.state?.islocal) {
            storeToSessionStorage(storageKeys.isLocalUpload, "true");
        }
    }, [location?.state?.islocal]);

    const isLocalUploadFlow =
        hasLocalUploadSource &&
        (Boolean(location?.state?.islocal) ||
            getFromSessionStorage<string>(storageKeys.isLocalUpload) ===
                "true");
    const { data: uploadConnectorResponse } =
        useGetProjectsConnectorByProjectId(projectId ?? "", {
            query: {
                enabled: Boolean(projectId && hasLocalUploadSource)
            }
        });
    const uploadConnectorId = uploadConnectorResponse?.data?.connector_id ?? "";
    const fileConnectors = useMemo(
        () => (uploadConnectorId ? [uploadConnectorId] : []),
        [uploadConnectorId]
    );
    const initialArgs = getInitialTableArgs(
        isLocalUploadFlow,
        location?.state,
        uploadConnectorId
    );

    const flowType = location?.state?.flowtype ?? FlowType.CREATE;
    const tableNotExists = location?.state?.tableNotExists ?? false;

    // Validation, Error  and Loading state variable
    const [isLoading, setLoading] = useState(false);
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    // Local file Upload handler states variables
    const [uploadInProgress, setInProgress] = useState(false);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    // Bumped whenever the uploaded-file set changes (upload or remove) so the
    // uploaded-file dropdown re-fetches its source-table list.
    const [sourceTableReloadKey, setSourceTableReloadKey] = useState(0);
    const [nameInputState, setNameInputState] = useState(InputState.None);

    const handleUploadSuccess = useCallback((status: boolean) => {
        setUploadSuccess(status);
        setSourceTableReloadKey((k) => k + 1);
    }, []);

    const [alertPkey, setAlertPkey] = useState(false);

    // Name Input handler states variables
    const [nameValidationResponse, setNameValidationResponse] = useState({
        isValid: true,
        errMsg: ""
    });

    const navigate = useCustomNavigation();
    const nameInputRef = useRef<{
        updateName: (tableName: string) => void;
    }>(null);

    const [connectors, setConnectors] = useState<ConnectorDropdownOption[]>([]);
    const { connectors: projectConnectors } =
        useConnectorDropdownOptions(handleError);

    useEffect(() => {
        if (projectConnectors.length !== connectors.length) {
            setConnectors(projectConnectors);
        }
    }, [connectors.length, projectConnectors]);

    /**===== Local states */
    const [tableArgs, setTableArgs] = useState<KumoapiTableTableDefinition>(
        () => initialArgs
    );
    // Editing / duplicating a Custom SQL table lands here with a CUSTOM_SQL
    // source_table rather than a physical one, so open on the SQL surface (the
    // Source dropdown is read-only outside Create) instead of defaulting to the
    // connector flow, whose physical column fetch would render "No data found".
    const isSqlSourceTable =
        (initialArgs.source_table?.data_source_type as string) === "CUSTOM_SQL";
    const [selectedSource, setSelectedSource] = useState<DropdownOption>(() => {
        if (isLocalUploadFlow) {
            return localUploadSource;
        }
        if (isSqlSourceTable) {
            return (
                SourceType.find((source) => source.value === "sql") ??
                SourceType[0]
            );
        }
        return SourceType[0];
    });
    // Custom SQL draft, lifted so the Add flow can gate on validation. On
    // edit/duplicate it is seeded from the persisted defining SELECT so the
    // editor opens with the existing query.
    const [sqlText, setSqlText] = useState(
        () => (initialArgs.source_table as { sql?: string })?.sql ?? ""
    );
    const [sqlValidated, setSqlValidated] = useState(false);
    // Moving this from out of creatArgs is because every connector's
    // configuration is change and it is difficult to handle all changes
    // in all places.
    const connectID = getConnectorIDFromTableDefSource(tableArgs.source_table);
    const connectorNotExists = connectID === "" && flowType !== FlowType.CREATE;

    const sourceTableName = getSourceNameFromTableDefSource(
        tableArgs.source_table
    );
    const sourcetableNotExists =
        sourceTableName === "" && flowType !== FlowType.CREATE;
    const [unnamedConSourceTable, setUnnamedConSourceTable] = useState("");

    const [cancelFetchSampleCal, setCancelFetchSample] =
        useState<boolean>(false);

    useEffect(() => {
        if (
            isLocalUploadFlow &&
            flowType === FlowType.CREATE &&
            uploadConnectorId &&
            connectID !== uploadConnectorId
        ) {
            setTableArgs((current) => ({
                ...current,
                source_table: getSourceInfoFromConnectorConfig(
                    null,
                    uploadConnectorId
                ) as KumoapiTableTableDefinitionSourceTable
            }));
        }
    }, [connectID, flowType, isLocalUploadFlow, uploadConnectorId]);
    const selectedConnector: ConnectorDropdownOption = useMemo(
        () =>
            connectors.find(
                (c) =>
                    c?.value ===
                    getConnectorIDFromTableDefSource(tableArgs?.source_table)
            ),
        [connectors, tableArgs?.source_table]
    ) as ConnectorDropdownOption;

    // Custom SQL runs in a SQL warehouse, so restrict its connector list to
    // Snowflake / Databricks / BigQuery (no file/S3 connectors).
    const connectorsForSource = useMemo(() => {
        if (selectedSource?.value !== "sql") {
            return connectors;
        }
        const sqlWarehouses: string[] = [
            warehouseSource.SnowFlake,
            warehouseSource.Databricks,
            warehouseSource.BigQuery
        ];
        return connectors.filter((c) =>
            sqlWarehouses.includes(c?.config?.type as string)
        );
    }, [connectors, selectedSource?.value]);

    const [tableName, setTableName] = useState(
        location?.state?.tableName || ""
    );
    const [s3ConnctorConfig] = useState(
        location?.state?.s3ConnctorConfig || ""
    );
    const [errLabelMsg, setErrLabelMsg] = useState<EntityErrorMsg>({
        entity: "",
        errorMsg: ""
    });

    // Helper function to clear error label
    const clearErrLabel = useCallback(() => {
        setErrLabelMsg({ entity: "", errorMsg: "" });
    }, []);

    const [columnNamesAdjusted, setColumnNamesAdjusted] =
        useState<boolean>(false);

    const { tableList } = useTables(
        handleError,
        undefined,
        flowType !== FlowType.EDIT
    );

    const onSourceTypeChange = useCallback(
        (e: DropdownOption) => {
            setSelectedSource(e);
            setColumnNamesAdjusted(false); // reset if already set
            setSqlText("");
            setSqlValidated(false);
            let con = "";
            if (e?.value === "local") {
                con = uploadConnectorId;
            }
            setCancelFetchSample(false);
            const args = { ...defaultTableData };
            args.source_table = getSourceInfoFromConnectorConfig(
                null,
                con
            ) as KumoapiTableTableDefinitionSourceTable;
            setTableArgs(args);
            nameInputRef?.current?.updateName("");
            setNameValidationResponse({
                isValid: true,
                errMsg: ""
            });
            clearErrors();
            if (errLabelMsg?.entity === "sourcetable") {
                clearErrLabel();
            }
        },
        [clearErrors, errLabelMsg?.entity, uploadConnectorId, clearErrLabel]
    );

    const onConnectorChange = useCallback(
        (e: ConnectorDropdownOption) => {
            if (connectID === e?.value) {
                return;
            }
            if (e?.value === ADD_KUMO_ENTITY_VALUE) {
                return;
            }
            setCancelFetchSample(false);
            const args = { ...defaultTableData };
            args.source_table = getSourceInfoFromConnectorConfig(
                e?.config as ConnectorConfig,
                e?.value as string
            ) as KumoapiTableTableDefinitionSourceTable;
            setTableArgs(args);
            nameInputRef?.current?.updateName("");
            setNameValidationResponse({
                isValid: true,
                errMsg: ""
            });
            clearErrors();
        },
        [clearErrors, connectID]
    );

    const getUniqueTableName = useCallback(
        (name: string) => {
            const tableNames = (tableList as TableListResponse[])?.map(
                (table: TableListResponse) => table?.name
            );
            return generateUniqueTableName(name, tableNames ?? []);
        },
        [tableList]
    );

    const resetTableSelection = useCallback(() => {
        if ((tableArgs.source_table as DatabricksSourceTable).table !== "") {
            const tmpArgs = { ...tableArgs };
            (tmpArgs.source_table as DatabricksSourceTable).table = "";
            tmpArgs.cols = [];
            tmpArgs.pkey = "";
            tmpArgs.end_time_col = "";
            tmpArgs.time_col = "";
            setTableArgs(tmpArgs);
        }
        nameInputRef?.current?.updateName("");
    }, [tableArgs]);

    /**
     * The last name this page suggested. While the field still holds it, the
     * user has not named the table themselves and a new selection may replace
     * it. Both the upload and dropdown paths go through here so they cannot
     * drift apart.
     */
    const suggestedName = useRef("");

    // Read through a ref, not the render's `tableName`: the upload resolves its
    // stored name asynchronously, and a name typed while that request is in
    // flight would otherwise be judged against a stale value and overwritten.
    const currentName = useRef(tableName);
    currentName.current = tableName;

    const suggestTableName = useCallback((rawName: string) => {
        const typed = currentName.current?.trim();
        if (typed && typed !== suggestedName.current) {
            return;
        }
        const name = removeUnderScore(rawName) ?? "";
        suggestedName.current = name;
        setNameValidationResponse({ isValid: true, errMsg: "" });
        nameInputRef?.current?.updateName(name);
    }, []);

    const onTableSelect = useCallback(
        (e: TableDropdownOption) => {
            if (!e?.valid) {
                const error: CustomError = {
                    response: {
                        status: errorStatusCode.UnProcessableEntityError,
                        data: e.reason
                    },
                    name: "",
                    message: e.reason as string
                };
                handleError(error, `Can't select the Table '${e.value}'!`);
                return;
            }
            if (e?.value === sourceTableName) {
                return;
            }
            if (errLabelMsg?.entity === "sourcetable") {
                clearErrLabel();
            }
            clearErrors();
            setCancelFetchSample(false);
            // Picking from the dropdown is not an upload; clear the flag so the
            // uploaded-file chip stays hidden for dropdown selections.
            setUploadSuccess(false);

            const tmpArgs = { ...tableArgs };
            applySelectedSourceTable(tmpArgs.source_table, e);
            tmpArgs.cols = [];
            tmpArgs.pkey = "";
            tmpArgs.end_time_col = "";
            tmpArgs.time_col = "";
            setTableArgs(tmpArgs);
            const rawTableName = replaceSchemaTableNameSplitter(e?.value);
            suggestTableName(getUniqueTableName(rawTableName));
        },
        [
            suggestTableName,
            sourceTableName,
            errLabelMsg?.entity,
            clearErrors,
            tableArgs,
            handleError,
            getUniqueTableName,
            clearErrLabel
        ]
    );

    const updateLocalFileTableName = (name: string) => {
        setNameValidationResponse({
            isValid: true,
            errMsg: ""
        });
        const localFileName = removeUnderScore(name);
        nameInputRef?.current?.updateName(localFileName);
    };

    const { submitRequest, submitSqlTable } = useSubmitTable({
        handleError,
        addTable,
        refetchTable,
        setLoading,
        setAlertPkey,
        setCancelFetchSample
    });

    // Snapshot of the volatile form/flow state the submit flow reads. Built
    // fresh at each call site so `submitRequest` stays a stable reference.
    const buildSubmitContext = useCallback(
        (): SubmitTableContext => ({
            tableArgs,
            tableName,
            flowType,
            connectorNotExists,
            sourcetableNotExists,
            s3ConnctorConfig,
            unnamedConSourceTable
        }),
        [
            tableArgs,
            tableName,
            flowType,
            connectorNotExists,
            sourcetableNotExists,
            s3ConnctorConfig,
            unnamedConSourceTable
        ]
    );

    const isPkeyAvailable = useCallback(
        (args: KumoapiTableTableDefinition) => {
            // If Pkey is not set, check for key columns
            // if key column exists, Alert with warning!
            if (!isValid(args?.pkey)) {
                const tName = getSourceNameFromTableDefSource(
                    args.source_table
                );
                const key = detectPrimaryKey(tName, args.cols);
                if (key) {
                    setAlertPkey(true);
                    dialogRef.current?.toggleDialog(
                        `Add a Primary Key?`,
                        <>
                            {`Would you like to set column "${key}" as the primary key? You can update this in table settings at any time.`}
                            <br />
                            {`Learn more `}
                            <DocumentationLink
                                id="pkey_selection"
                                pageLink={docURLs.PkeySelection}
                                linkText="here."
                            />
                        </>,
                        { pkey: key },
                        true,
                        "Set as primary key",
                        undefined,
                        "Skip"
                    );
                    return true;
                }
            }

            return false;
        },
        [dialogRef]
    );

    const isTimeColAvailable = useCallback(
        (args: KumoapiTableTableDefinition) => {
            const cIndex = args?.cols?.findIndex(
                (e) => e?.name === args?.time_col
            );

            let timeColExists = false;
            // `time_col` not set, Check for `end_time_col`; if both not set, Alert
            // with warning!
            if (cIndex === -1) {
                let colName = "";
                args?.cols?.forEach((column) => {
                    if (
                        column?.stype?.includes("timestamp") &&
                        column?.name !== args?.end_time_col
                    ) {
                        colName = column?.name ?? "";
                        timeColExists = true;
                        return;
                    }
                });
                if (colName) {
                    dialogRef.current?.toggleDialog(
                        `Add a Create Date?`,
                        <>
                            {`Would you like to set column "${colName}" as the create date? You can update this in table settings at any time.`}
                            <br />
                            {`Learn more `}
                            <DocumentationLink
                                id="create_date_selection"
                                pageLink={docURLs.CreateDateSelection}
                                linkText="here."
                            />
                        </>,
                        { time_col: colName },
                        true,
                        "Set as create date",
                        undefined,
                        "Skip"
                    );
                }
            }
            return timeColExists;
        },
        [dialogRef]
    );

    /**
     * Validates inputs before adding a table.
     * Returns true if there are errors, false if validation passes.
     */
    const checkBeforeAdd = useCallback(() => {
        clearErrLabel();
        if (tableName === "") {
            setErrLabelMsg({
                entity: "name",
                errorMsg: "Please enter table name."
            });
            return true;
        } else if (
            selectedSource?.value === "connector" &&
            sourceTableName === ""
        ) {
            if (
                flowType === FlowType.EDIT &&
                tableArgs?.source_table?.data_source_type === DataSourceType.S3
            ) {
                return false;
            } else {
                setErrLabelMsg({
                    entity: "sourcetable",
                    errorMsg: "Please select a table."
                });
                return true;
            }
        } else if (
            selectedSource?.value === "local" &&
            sourceTableName === ""
        ) {
            setErrLabelMsg({
                entity: "sourcetable",
                errorMsg: "Please select / upload a table."
            });
            return true;
        } else if (
            !nameValidationResponse.isValid ||
            nameInputState === InputState.Validating
        ) {
            //No error-label. Validating or Error already being shown
            return true;
        } else {
            return false; //No error
        }
    }, [
        clearErrLabel,
        tableName,
        selectedSource?.value,
        sourceTableName,
        flowType,
        tableArgs?.source_table?.data_source_type,
        nameValidationResponse.isValid,
        nameInputState
    ]);

    /** Validate the table inputs and create the table when they are valid. */
    const onAddTableClick = useCallback(() => {
        clearErrors();
        setAlertPkey(false);
        // Custom SQL is authored + validated in the editor; Add persists that
        // already-validated definition. The SQL definition carries no physical
        // source table (so the raw create/validate path can't build its request),
        // and the create endpoint re-validates the SELECT server-side, so this
        // skips the front-end validate and calls the create mutation directly.
        if (selectedSource?.value === "sql") {
            if (checkBeforeAdd()) {
                return;
            }
            submitSqlTable({
                tableArgs,
                tableName,
                connectorId: connectID,
                sql: sqlText,
                flowType
            });
            return;
        }
        if (checkBeforeAdd()) {
            return;
        }

        const { err, colName } = getMissingColumnError(tableArgs);
        if (err !== "") {
            const error: CustomError = {
                response: {
                    status: errorStatusCode.UnProcessableEntityError,
                    data: err
                },
                name: "",
                message: err
            };
            handleError(error, `Please select the column ${colName}!`);
            return;
        }

        if (!isPkeyAvailable(tableArgs) && !isTimeColAvailable(tableArgs)) {
            submitRequest(buildSubmitContext());
        }
    }, [
        clearErrors,
        selectedSource?.value,
        tableArgs,
        tableName,
        connectID,
        selectedConnector,
        sqlText,
        flowType,
        isPkeyAvailable,
        isTimeColAvailable,
        handleError,
        submitRequest,
        submitSqlTable,
        buildSubmitContext,
        checkBeforeAdd
    ]);

    const onCancelClick = () => {
        sessionStorage.removeItem(storageKeys.isLocalUpload);
        if (flowType === FlowType.CREATE)
            navigate(projectPaths.Tables, {
                replace: true
            });
        else {
            navigate(-1);
        }
    };

    useKeyColumnTypeValidation({
        tableArgs,
        clearErrors,
        handleError
    });

    const onConfirmSave = useCallback(() => {
        dialogRef.current?.toggleDialog();
        if (!alertPkey) {
            submitRequest(buildSubmitContext());
        } else {
            setAlertPkey(false);
            setTimeout(() => {
                if (!isTimeColAvailable(tableArgs)) {
                    submitRequest(buildSubmitContext());
                }
            }, 500);
        }
    }, [
        alertPkey,
        dialogRef,
        isTimeColAvailable,
        submitRequest,
        buildSubmitContext,
        tableArgs
    ]);

    const onYesClick = useCallback(
        (value?: { pkey?: string; time_col?: string }) => {
            dialogRef.current?.toggleDialog();
            const tmpArgs = { ...tableArgs };
            if (value?.pkey) {
                setAlertPkey(false);
                tmpArgs.pkey = value?.pkey;
                const pKeyIndex = tmpArgs?.cols?.findIndex(
                    (c) => c.name === value?.pkey
                );
                if (pKeyIndex !== -1) {
                    tmpArgs.cols[pKeyIndex].stype = "ID";
                }
                setTableArgs(tmpArgs);
                setTimeout(() => {
                    if (!isTimeColAvailable(tmpArgs)) {
                        submitRequest(
                            buildSubmitContext(),
                            /*time_Col*/ undefined,
                            value?.pkey
                        );
                    }
                }, 500);
                return;
            }
            if (value?.time_col) {
                tmpArgs.time_col = value.time_col;
                setTableArgs(tmpArgs);
                setTimeout(() => {
                    submitRequest(buildSubmitContext(), value.time_col);
                }, 100);
            }
        },
        [
            dialogRef,
            isTimeColAvailable,
            submitRequest,
            buildSubmitContext,
            tableArgs
        ]
    );

    // Custom SQL Add persists an already-validated definition, so it enables
    // only once the editor has confirmed the SELECT (`sqlValidated`) and at
    // least one column is selected — editing the SQL clears both, re-disabling
    // Add until it is re-validated. Otherwise mirror the raw enable condition.
    const isAddDisabled =
        selectedSource?.value === "sql"
            ? !sqlValidated ||
              !tableName.trim() ||
              (tableArgs.cols?.length ?? 0) === 0
            : !tableName.trim() || uploadInProgress || sourceTableName === "";

    const AddButton = (
        <Button onClick={onAddTableClick} disabled={isAddDisabled}>
            {flowType === FlowType.EDIT ? "Save" : "Add"}
        </Button>
    );

    const isLocalFlow = selectedSource?.value === "local";
    const isLocalCreateFlow = isLocalFlow && flowType === FlowType.CREATE;
    const isSqlFlow = selectedSource?.value === "sql";

    // The column grid is populated from a sample-row fetch against the chosen
    // source table, so on create it has nothing to show until one is picked.
    const hasColumnSource =
        (flowType === FlowType.CREATE && Boolean(sourceTableName)) ||
        flowType === FlowType.EDIT ||
        flowType === FlowType.DUPLICATE;

    const tableNameField = (
        <div className={cn(formRowClassName, stateIconOutdentClassName)}>
            <ValidatedNameInput
                readOnly={flowType === FlowType.EDIT}
                value={tableName}
                setName={(value: string) => {
                    if (errLabelMsg?.entity === "name") {
                        clearErrLabel();
                    }
                    setTableName(value);
                }}
                nameValidationResponse={nameValidationResponse}
                nameInputState={nameInputState}
                setNameInputState={setNameInputState}
                setNameValidationResponse={setNameValidationResponse}
                connectorId={connectID}
                ref={nameInputRef}
                updateLocalFileTableName={updateLocalFileTableName}
                fileConnectors={fileConnectors}
                labelClassName={formLabelClassName}
            />
            <div className="col-start-2">
                <ErrorLabel entity="name" errLabelMsg={errLabelMsg} />
            </div>
        </div>
    );

    const sourceField = (
        <div className={formRowClassName} data-testid="table-source-selector">
            {/* The section is already headed "Source", so the field names the
                narrower thing it actually picks — connector, local upload or
                custom SQL — rather than repeating the heading. */}
            <Label htmlFor="select-source" className={formLabelClassName}>
                Source type
            </Label>
            <Select
                id="select-source"
                placeholder="Select Source"
                search={false}
                emptyText="No source found."
                mode="single"
                className="w-full"
                selectedValues={
                    selectedSource?.value
                        ? [selectedSource.value as string]
                        : undefined
                }
                options={SourceType}
                onValuesChange={(newValue) => {
                    const currSel = SourceType.find((item) =>
                        newValue.includes(item.value)
                    );
                    onSourceTypeChange(currSel as DropdownOption);
                }}
                disabled={
                    flowType !== FlowType.CREATE ||
                    uploadSuccess ||
                    uploadInProgress ||
                    nameInputState === InputState.Validating
                }
            />
        </div>
    );

    return (
        <>
            {!hideTitle && (
                <GridRow
                    className="flex justify-between items-center"
                    data-screenshot-ready={
                        isLoading ? undefined : "new-table-page"
                    }>
                    <GridCol>
                        <Title
                            data-testid="new-table-page-heading"
                            variant="page-heading"
                            className="!font-bold text-3xl">
                            {flowType === FlowType.EDIT
                                ? "Edit Table"
                                : "Add New Table"}
                        </Title>
                        <Description
                            size="md"
                            className="leading-5 mt-2 text-title-secondary">
                            {flowType === FlowType.EDIT
                                ? " Changes made here will be saved and will update the table schema."
                                : "Select and configure tables for your model"}
                        </Description>
                    </GridCol>
                    <GridCol className="gap-2 flex justify-between items-center">
                        {!tableNotExists && (
                            <Button
                                id="cancel-table-create"
                                variant="outline"
                                onClick={onCancelClick}>
                                Cancel
                            </Button>
                        )}
                        {AddButton}
                    </GridCol>
                </GridRow>
            )}

            {banner ? <Spacer height={16} /> : <></>}
            <Validation
                pageLoading={isLoading}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                onDialogConfirm={onYesClick}
                onDialogSkip={onConfirmSave}
                clearErrors={clearErrors}
            />
            <Spacer height={16} />
            <>
                {sourceField && (
                    <section className={cn(formSectionClassName, "mt-0")}>
                        <Title
                            variant="section-header"
                            className={sectionTitleClassName}>
                            Source
                        </Title>
                        {sourceField}
                    </section>
                )}
                <section className={formSectionClassName}>
                    <Title
                        variant="section-header"
                        className={sectionTitleClassName}>
                        Table details
                    </Title>
                    {isLocalFlow ? (
                        <LocalUploadFlow
                            tableNameField={tableNameField}
                            sourceTableErrorLabel={
                                <ErrorLabel
                                    entity="sourcetable"
                                    errLabelMsg={errLabelMsg}
                                />
                            }
                            showFileSelection={isLocalCreateFlow}
                            flowType={flowType}
                            nameInputState={nameInputState}
                            onTableSelect={onTableSelect}
                            handleError={handleError}
                            clearErrors={clearErrors}
                            selectedConnector={selectedConnector}
                            connectorId={connectID}
                            sourceTableName={sourceTableName}
                            resetTableSelection={resetTableSelection}
                            errLabelMsg={errLabelMsg}
                            clearErrLabel={clearErrLabel}
                            setInProgress={setInProgress}
                            setUploadSuccess={handleUploadSuccess}
                            suggestTableName={suggestTableName}
                            setTableName={setTableName}
                            uploadSuccess={uploadSuccess}
                            tableArgs={tableArgs}
                            setTableArgs={setTableArgs}
                            fileConnectors={fileConnectors}
                            setColumnNamesAdjusted={setColumnNamesAdjusted}
                            sourceTableReloadKey={sourceTableReloadKey}
                        />
                    ) : (
                        <>
                            {!connectorNotExists && (
                                <ConnectorSelection
                                    connectors={connectorsForSource}
                                    fileConnectors={fileConnectors}
                                    flowType={flowType}
                                    nameValidationState={nameInputState}
                                    onConnectorChange={onConnectorChange}
                                    connectorId={connectID}
                                    horizontal
                                />
                            )}
                            {selectedSource?.value === "connector" &&
                                !sourcetableNotExists && (
                                    <SourceTable
                                        flowType={flowType}
                                        nameValidationState={nameInputState}
                                        onTableChange={onTableSelect}
                                        handleError={handleError}
                                        clearErrors={clearErrors}
                                        selectedConnector={selectedConnector}
                                        setLoading={
                                            flowType !== FlowType.CREATE
                                                ? setLoading
                                                : () => {}
                                        }
                                        connectorId={connectID}
                                        sourceTableName={sourceTableName}
                                        resetTableSelection={
                                            resetTableSelection
                                        }
                                        errorLabel={
                                            <ErrorLabel
                                                entity="sourcetable"
                                                errLabelMsg={errLabelMsg}
                                            />
                                        }
                                        horizontal
                                    />
                                )}
                            {tableNameField}
                        </>
                    )}
                </section>
                {/* Custom SQL discovers its columns from the query schema and
                    selects them in its own editor, so it replaces the Columns
                    section rather than adding a second column list. */}
                {isSqlFlow ? (
                    <section className={sectionClassName}>
                        <Title
                            variant="section-header"
                            className={sectionTitleClassName}>
                            Query
                        </Title>
                        <CustomSqlTableSource
                            connectorId={connectID}
                            tableDefinition={tableArgs}
                            setTableArgs={setTableArgs}
                            sqlText={sqlText}
                            setSqlText={setSqlText}
                            validated={sqlValidated}
                            setValidated={setSqlValidated}
                        />
                    </section>
                ) : (
                    <section className={sectionClassName}>
                        <Title
                            variant="section-header"
                            className={sectionTitleClassName}>
                            Columns
                        </Title>
                        {hasColumnSource ? (
                            <Preprocessing
                                setLoading={setLoading}
                                isLoading={isLoading}
                                flowType={flowType}
                                tableArgs={tableArgs}
                                handleError={handleError}
                                setTableArgs={setTableArgs}
                                selectedConnector={selectedConnector}
                                cancelFetchSampleCal={cancelFetchSampleCal}
                                connectorId={connectID}
                                selectedSourceTableName={sourceTableName}
                                connectorNotExists={connectorNotExists}
                                setUnnamedConSourceTable={
                                    setUnnamedConSourceTable
                                }
                                sourcetableNotExists={sourcetableNotExists}
                                columnNamesAdjusted={columnNamesAdjusted}
                                setColumnNamesAdjusted={setColumnNamesAdjusted}
                            />
                        ) : (
                            <Description
                                size="sm"
                                className="leading-5 text-title-secondary">
                                Select a source table to see its columns
                            </Description>
                        )}
                    </section>
                )}
                {hideTitle && (
                    <>
                        <Spacer height={16} />
                        <GridRow>
                            <GridCol>{AddButton}</GridCol>
                        </GridRow>
                    </>
                )}
            </>
        </>
    );
};

export default WithErrorBoundary(ConnectTables);
