// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useEffect, useMemo, useRef, useState } from "react";

import {
    Button,
    cn,
    Form,
    FormField,
    FormWrapper,
    GridCol,
    Label,
    Select,
    Skeleton,
    TextInput
} from "@kumo-ai/kumo-ui-catalog";
import { SourceTableConfigResponseSourceTable } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useProjectId } from "hooks/project/projects";
import {
    getObjectStoreSourceTableConfig,
    useSourceTables
} from "hooks/tables/sourceTables";

import {
    ConnectorDropdownOption,
    InputState,
    TableDropdownOption
} from "types/inputs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import {
    isAnyUriObjectStoreConnector,
    isObjectStoreConnectorConfig,
    normalizeObjectStoreTablePath,
    sourcePathFromObjectStoreSource
} from "utils/objectStoreTablePath";

import { formLabelClassName, formRowTopAlignedClassName } from "./formRow";

export interface SourceTableProps {
    onTableChange: (e: TableDropdownOption) => void;
    nameValidationState: InputState;
    selectedConnector: ConnectorDropdownOption;
    flowType: FlowType;
    handleError: HandleErrorFunc;
    setLoading: (v: boolean) => void;
    connectorId?: string;
    sourceTableName?: string;
    resetTableSelection: () => void;
    errorLabel?: ReactNode;
    clearErrors?: () => void;
    label?: string;
    placeholder?: string;
    emptyText?: string;
    selectClassName?: string;
    reloadSignal?: number;
    /** Force-disable the dropdown regardless of flow/validation state. */
    disabled?: boolean;
    /** Renders the label beside the control instead of above it. */
    horizontal?: boolean;
}

// Resolved once so the component body itself stays free of layout branching.
const layoutClassNames = (horizontal: boolean, selectClassName: string) => ({
    field: horizontal ? "w-full" : selectClassName,
    label: horizontal ? cn(formLabelClassName, "pt-2") : "text-title-secondary"
});

const FieldLayout = ({
    horizontal,
    label,
    children
}: {
    horizontal: boolean;
    label: ReactNode;
    children: ReactNode;
}) =>
    horizontal ? (
        <div className={formRowTopAlignedClassName}>
            {label}
            <div className="flex flex-col">{children}</div>
        </div>
    ) : (
        <GridCol>
            <Form>
                <FormWrapper>
                    <FormField className="gap-0">
                        {label}
                        {children}
                    </FormField>
                </FormWrapper>
            </Form>
        </GridCol>
    );

const objectStoreSourceFromResponse = (
    sourceTable: SourceTableConfigResponseSourceTable
) => ("source" in sourceTable ? sourceTable.source : undefined);

const manualPathPlaceholder = (
    config: ConnectorDropdownOption["config"]
): string => {
    if (!isAnyUriObjectStoreConnector(config)) {
        return "relative/path/table.parquet or matching full URI";
    }
    switch (config?.type) {
        case "gcs":
            return "gs://bucket/deep/path/table.parquet";
        case "adls":
            return "abfss://filesystem@account.dfs.core.windows.net/deep/path/table.parquet";
        default:
            return "s3://bucket/deep/path/table.parquet";
    }
};

const SourceTable = ({
    onTableChange,
    nameValidationState,
    selectedConnector,
    flowType,
    handleError,
    setLoading,
    connectorId,
    sourceTableName,
    resetTableSelection,
    errorLabel,
    clearErrors,
    label = "Source table",
    placeholder = "Select Table",
    emptyText = "No tables found.",
    selectClassName = "w-72.5",
    reloadSignal,
    disabled = false,
    horizontal
}: SourceTableProps) => {
    const isHorizontal = Boolean(horizontal);
    const layout = layoutClassNames(isHorizontal, selectClassName);
    const projectId = useProjectId();
    const [manualPath, setManualPath] = useState("");
    const [manualPathError, setManualPathError] = useState("");
    const [validatingManualPath, setValidatingManualPath] = useState(false);
    const exactSelectionSequence = useRef(0);
    const { sourceTables, fetchSourceTables } = useSourceTables({
        selectedConnector,
        flowType,
        handleError,
        setLoading,
        connectorId,
        sourceTableName,
        resetTableSelection,
        clearErrors,
        reloadSignal
    });
    const objectStoreConfig = useMemo(
        () =>
            isObjectStoreConnectorConfig(selectedConnector?.config)
                ? selectedConnector.config
                : undefined,
        [selectedConnector?.config]
    );
    const sourceTableOptions = useMemo(() => {
        const options = (sourceTables ?? []).map((item) => ({
            value: item.value as string,
            label: item.label as string
        }));
        if (
            sourceTableName &&
            !options.some((item) => item.value === sourceTableName)
        ) {
            options.push({ value: sourceTableName, label: sourceTableName });
        }
        return options;
    }, [sourceTableName, sourceTables]);
    const selectionDisabled =
        disabled ||
        flowType !== FlowType.CREATE ||
        nameValidationState === InputState.Validating ||
        validatingManualPath;
    const discoveryDisabled =
        selectionDisabled ||
        isAnyUriObjectStoreConnector(selectedConnector?.config);

    useEffect(() => {
        exactSelectionSequence.current += 1;
        setManualPath("");
        setManualPathError("");
        setValidatingManualPath(false);
        return () => {
            exactSelectionSequence.current += 1;
        };
    }, [selectedConnector?.value]);

    const selectExactObjectStoreTable = async (normalized: string) => {
        if (!projectId || !connectorId) {
            setManualPathError("Select a connector before choosing a path.");
            return;
        }
        const sequence = ++exactSelectionSequence.current;
        try {
            setValidatingManualPath(true);
            const response = await getObjectStoreSourceTableConfig(
                projectId,
                connectorId,
                normalized
            );
            if (sequence !== exactSelectionSequence.current) {
                return;
            }
            const sourceTable = response.data.source_table;
            const objectStoreSource =
                objectStoreSourceFromResponse(sourceTable);
            const resolvedValue = objectStoreSource
                ? sourcePathFromObjectStoreSource(objectStoreSource)
                : normalized;
            setManualPathError("");
            clearErrors?.();
            onTableChange({
                value: resolvedValue,
                label: resolvedValue,
                valid: true,
                connectorId,
                fileType:
                    "file_type" in sourceTable
                        ? sourceTable.file_type
                        : undefined,
                objectStoreSource
            });
        } catch (error) {
            if (sequence !== exactSelectionSequence.current) {
                return;
            }
            setManualPathError("The table path could not be validated.");
            handleError(error as Parameters<HandleErrorFunc>[0]);
        } finally {
            if (sequence === exactSelectionSequence.current) {
                setValidatingManualPath(false);
            }
        }
    };

    const chooseManualPath = async () => {
        if (!objectStoreConfig) {
            return;
        }
        try {
            const normalized = normalizeObjectStoreTablePath(
                manualPath,
                objectStoreConfig
            );
            setManualPath(normalized);
            setManualPathError("");
            await selectExactObjectStoreTable(normalized);
        } catch (error) {
            setManualPathError(
                error instanceof Error ? error.message : "Invalid table path."
            );
        }
    };

    const chooseDiscoveredTable = async (table: TableDropdownOption) => {
        setManualPath("");
        setManualPathError("");
        if (objectStoreConfig) {
            await selectExactObjectStoreTable(table.value);
        } else {
            onTableChange(table);
        }
    };

    const labelNode = (
        <Label htmlFor="Source Table" className={layout.label}>
            {label}
        </Label>
    );

    const body = (
        <>
            <div className={layout.field}>
                {fetchSourceTables ? (
                    <Skeleton className="h-8" />
                ) : (
                    <Select
                        id={"select-source-table"}
                        placeholder={placeholder}
                        search={true}
                        emptyText={emptyText}
                        mode="single"
                        selectedValues={
                            sourceTableName === ""
                                ? []
                                : Array.of(sourceTableName as string)
                        }
                        options={sourceTableOptions}
                        onValuesChange={(newValue) => {
                            const currSel = sourceTables?.find((item) =>
                                newValue.includes(item?.value as string)
                            );
                            if (currSel) {
                                void chooseDiscoveredTable(currSel);
                            }
                        }}
                        disabled={discoveryDisabled}
                    />
                )}
            </div>
            {objectStoreConfig && flowType === FlowType.CREATE && (
                <div className="mt-3 flex items-end gap-2">
                    <div className={layout.field}>
                        <TextInput
                            id="object-store-table-path"
                            label="Or enter a table path"
                            labelVariant="secondary"
                            value={manualPath}
                            placeholder={manualPathPlaceholder(
                                selectedConnector.config
                            )}
                            onChange={(event) => {
                                setManualPath(event.target.value);
                                setManualPathError("");
                            }}
                            onSubmit={() => void chooseManualPath()}
                            isInvalid={Boolean(manualPathError)}
                            errorMsg={manualPathError}
                            readOnly={selectionDisabled}
                        />
                    </div>
                    <Button
                        id="use-object-store-table-path"
                        type="button"
                        variant="outline"
                        onClick={() => void chooseManualPath()}
                        disabled={
                            selectionDisabled ||
                            validatingManualPath ||
                            manualPath.length === 0
                        }>
                        Use path
                    </Button>
                </div>
            )}
            {errorLabel}
        </>
    );

    return (
        <FieldLayout horizontal={isHorizontal} label={labelNode}>
            {body}
        </FieldLayout>
    );
};

export default WithErrorBoundary(SourceTable);
