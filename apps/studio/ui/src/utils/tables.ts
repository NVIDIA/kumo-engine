// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SourceColumn } from "gen/v0/types";
import {
    CustomSQLSourceTable,
    KumoapiSourceTableFileType,
    KumoapiTableTableDefinitionSourceTable,
    TableMetadataRequest,
    ValidationResponse
} from "gen/v1/types";
import { Column } from "gen/v1/types/column";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { Stype as KumoapiTypingStype } from "gen/v1/types/stype";

import { CustomError } from "types/common";
import { DropdownOption } from "types/inputs";
import {
    ColumnArgs,
    ConnectTablesLocationState,
    DataSourceType,
    FlowType,
    GetColumnsResponse,
    SourceTypeDeploymentFlags
} from "types/tables";

import { errorStatusCode } from "./Constants/enums";
import {
    getConnectorIDFromTableDefSource,
    getSourceInfoFromConnectorConfig
} from "./Converters/tablesConverter";
import { parseNumbers } from "./Helpers/numberHelpers";
import { capitalizeFirstLetter } from "./Helpers/stringHelpers";
import { isValid } from "./Helpers/typeGuards";
import {
    ENABLE_CUSTOM_SQL_TABLES,
    tableDataTypesToSTypesMapping
} from "./Constants";

/** Preserve durable column semantics when rebuilding an API column. */
type ApiColumnInput = Pick<ColumnArgs, "name" | "dtype" | "stype"> & {
    timestamp_format?: Column["timestamp_format"];
    multicategorical_delimiter?: Column["multicategorical_delimiter"];
};

export function toApiColumn(column: ApiColumnInput): Column {
    return {
        name: column.name,
        dtype: column.dtype,
        stype: column.stype,
        timestamp_format: column.timestamp_format ?? undefined,
        multicategorical_delimiter:
            column.multicategorical_delimiter ?? undefined
    } as Column;
}

type MulticategoricalDelimiter = "," | "|" | ";" | "\t" | null;

export interface RecommendedStype {
    name: string;
    newStype: string;
    multicategoricalDelimiter?: MulticategoricalDelimiter;
}

/**
 * Whether the table-create wizard can ask the backend for stats-based
 * metadata inference (`POST /v1/tables/infer_metadata`) for this source.
 *
 * Physical warehouse/object-store tables qualify once their backing
 * connector is known. Custom SQL tables resolve their schema through the
 * dedicated `resolve_sql` path, and UNAVAILABLE sources have no backend
 * source to scan, so both are excluded.
 */
export function canInferStatsBasedMetadata(
    sourceTable: KumoapiTableTableDefinitionSourceTable | undefined
): sourceTable is TableMetadataRequest["source_table"] {
    const dataSourceType = sourceTable?.data_source_type as string | undefined;
    if (!isValid(dataSourceType)) return false;
    if (dataSourceType === "CUSTOM_SQL" || dataSourceType === "UNAVAILABLE") {
        return false;
    }
    return isValid(getConnectorIDFromTableDefSource(sourceTable));
}

/**
 * Reconcile the wizard's semantic-type recommendations across its two
 * inference signals.
 *
 * The connector `columns`/`sample` previews the wizard renders first are
 * stats-FREE (name/dtype heuristics only), while `infer_metadata` performs
 * the same stats-BASED inference the SDK uses for its defaults. Where both
 * signals exist for a column, the stats-based one wins, so a table authored
 * through the wizard converges on the same default stypes as an SDK-authored
 * table (kumo-ai/kumo#30131). Columns the stats-based response does not
 * cover keep their sample-derived recommendation.
 *
 * A recommendation is only emitted when the target differs from the
 * column's current stype/delimiter and stays valid for the column's dtype.
 */
export function reconcileStypeRecommendations(
    statsBasedCols: Column[] | null,
    sampleRecommendations: RecommendedStype[],
    sourceColumns: ColumnArgs[]
): RecommendedStype[] {
    const statsByName = new Map(
        (statsBasedCols ?? []).map((col) => [col.name, col])
    );
    const sampleByName = new Map(
        sampleRecommendations.map((rec) => [rec.name, rec])
    );
    const recommendations: RecommendedStype[] = [];
    for (const column of sourceColumns) {
        if (!column?.name) continue;
        if (column.stype === "unsupported" || column.dtype === "unsupported") {
            continue;
        }
        let targetStype: string;
        let targetDelimiter: MulticategoricalDelimiter;
        const statsCol = statsByName.get(column.name);
        if (statsCol) {
            targetStype = statsCol.stype as string;
            targetDelimiter = (statsCol.multicategorical_delimiter ??
                null) as MulticategoricalDelimiter;
        } else {
            const sampleRec = sampleByName.get(column.name);
            if (!sampleRec) continue;
            targetStype = sampleRec.newStype;
            targetDelimiter = sampleRec.multicategoricalDelimiter ?? null;
        }
        if (
            !tableDataTypesToSTypesMapping[column.dtype as string]?.includes(
                targetStype
            )
        ) {
            continue;
        }
        if (
            column.stype !== targetStype ||
            (column.multicategorical_delimiter ?? null) !== targetDelimiter
        ) {
            recommendations.push({
                name: column.name,
                newStype: targetStype,
                multicategoricalDelimiter: targetDelimiter
            });
        }
    }
    return recommendations;
}

function resolveColumnTypes(
    sourceColumn: SourceColumn,
    existingColumn?: Column
): {
    stype: string;
    dtype: string;
    multicategoricalDelimiter?: MulticategoricalDelimiter;
} {
    return {
        stype: existingColumn?.stype ?? sourceColumn.stype ?? "unsupported",
        dtype: existingColumn?.dtype ?? sourceColumn.dtype,
        multicategoricalDelimiter:
            existingColumn?.multicategorical_delimiter ??
            sourceColumn.multicategorical_delimiter
    };
}

function shouldSelectSourceColumn(
    flowType: FlowType,
    hasExistingColumn: boolean,
    disabled: boolean
): boolean {
    if (disabled) return false;
    if (flowType === FlowType.CREATE) return true;
    return (
        hasExistingColumn &&
        [FlowType.EDIT, FlowType.AS_TEMPLATE, FlowType.DUPLICATE].includes(
            flowType
        )
    );
}

function applySourceColumnMetadata(
    sourceColumn: SourceColumn,
    existingColumn: Column | undefined,
    stype: string,
    dtype: string,
    multicategoricalDelimiter: MulticategoricalDelimiter | undefined,
    pkey: string,
    timeCol: string,
    endTimeCol?: string
): void {
    const disabled = stype === "unsupported" || dtype === "unsupported";
    Object.assign(sourceColumn, {
        stype,
        dtype,
        multicategorical_delimiter: multicategoricalDelimiter,
        disableSelection: disabled,
        tooltip: disabled
            ? "The data type of this column is currently not supported by Kumo." +
              " If you wish to include this column, please change the data type of this column in your data " +
              "warehouse to a primitive, such as 'int', 'float', 'string', or 'timestamp'. Nested data types" +
              " (such as 'array' or 'map') are not supported."
            : "",
        sTypeLabel: capitalizeFirstLetter(stype),
        dTypeLabel: capitalizeFirstLetter(dtype),
        isPrimary: pkey === sourceColumn.name,
        isTimeColumn: timeCol === sourceColumn.name,
        isEndTimeColumn: endTimeCol === sourceColumn.name,
        userUpdatedType: Boolean(existingColumn)
    });
}

/**
 * Creating json Sample Rows from 2 dimension array
 * to show it in a UI
 * @returns
 */
function createSampleRowsJson(
    rows: Array<Array<string>>,
    i: number,
    sTypeLabel: string
) {
    //Creating sample rows object
    const sampleRows = [];
    let nonEmpty = null; // To store first non-empty sample value.
    for (let j = 0; j < rows?.length; j++) {
        let value = rows[j][i];
        if (sTypeLabel?.toLowerCase() === "numerical" && value !== null) {
            value = parseNumbers(rows[j][i]) as string;
        }

        if (value !== null) {
            sampleRows.push(value);
            if (nonEmpty === null) {
                nonEmpty = value;
            }
        }
        if (!isValid(value)) {
            sampleRows.push("N/A");
        }
    }
    return { sampleRows, nonEmpty };
}

/**
 * Restructurising the REST API response, assigning sample rows
 * to specific columns and updating global config classes
 *
 * @param data response of `connectorService.getSampleRows`
 * @param tableIndex Selected table list index
 * @param connectorID Active connector id
 */
export function parseColsAndSmplRows(
    data: SourceColumn[] | GetColumnsResponse,
    columns: Column[],
    flowType: FlowType,
    pkey: string,
    timeCol: string,
    endTimeCol?: string
) {
    const finalColumns = data as SourceColumn[];
    const cols = [],
        sourceColumns: Array<ColumnArgs> = [],
        selectedColumns: Array<string> = [];
    let primaryKeyCol = "";
    let timeColumn = "";
    let endTimeColumn = "";
    for (let i = 0; i < finalColumns.length; i++) {
        const sourceColumn = finalColumns[i];
        const existingColumn = columns.find(
            (column) => column.name === sourceColumn.name
        );
        const { stype, dtype, multicategoricalDelimiter } = resolveColumnTypes(
            sourceColumn,
            existingColumn
        );
        const disabledColumn =
            stype === "unsupported" || dtype === "unsupported";

        if (pkey === sourceColumn.name) {
            primaryKeyCol = sourceColumn.name;
        }
        if (timeCol === sourceColumn.name) {
            timeColumn = sourceColumn.name;
        }
        if (endTimeCol === sourceColumn.name) {
            endTimeColumn = sourceColumn.name;
        }

        applySourceColumnMetadata(
            sourceColumn,
            existingColumn,
            stype,
            dtype,
            multicategoricalDelimiter,
            pkey,
            timeCol,
            endTimeCol
        );
        const normalizedColumn = sourceColumn as unknown as ColumnArgs;
        sourceColumns.push(normalizedColumn);
        if (
            shouldSelectSourceColumn(
                flowType,
                Boolean(existingColumn),
                disabledColumn
            )
        ) {
            cols.push(toApiColumn(normalizedColumn));
            selectedColumns[i] = "true";
        }
    }

    primaryKeyCol = primaryKeyCol ?? "";
    timeColumn = timeColumn ?? "";
    endTimeColumn = endTimeColumn ?? "";
    return {
        sourceColumns,
        cols,
        selectedColumns,
        primaryKeyCol,
        timeColumn,
        endTimeColumn
    };
}

/**
 * Restructurising the REST API response, assigning sample rows
 * to specific columns and return columns with sample rows
 */
export function assignSampleRows(
    columns: Array<SourceColumn | ColumnArgs>,
    sourceColumns: ColumnArgs[],
    rows: Array<Array<string>>
): [
    ColumnArgs[],
    {
        name: string;
        newStype: string;
        multicategoricalDelimiter?: "," | "|" | ";" | "\t" | null;
    }[]
] {
    const columnsWithNewSTypes = [];
    const copySource = [...sourceColumns];
    for (let i = 0; i < columns?.length; i++) {
        const stype = columns[i].stype;

        const idx = sourceColumns?.findIndex(
            (e) => e?.name === columns[i]?.name
        );

        if (idx > -1) {
            // Checking for user selected stype
            const col = sourceColumns?.find(
                (e) => e?.name === columns[i]?.name
            );

            if (col) {
                if (
                    col.stype !== stype ||
                    col.multicategorical_delimiter !==
                        columns[i].multicategorical_delimiter
                ) {
                    const ColWithNewStype = {
                        name: col.name,
                        newStype: stype,
                        multicategoricalDelimiter:
                            columns[i].multicategorical_delimiter
                    };
                    columnsWithNewSTypes.push(ColWithNewStype);
                }
            }

            //Capitalizing first letter of stype
            const sTypeLabel = col?.sTypeLabel;
            //Creating sample rows object
            const { sampleRows, nonEmpty } = createSampleRowsJson(
                rows,
                i,
                sTypeLabel as string
            );

            // Adding created post processing, Capitalized stype, sample rows, isPrimary
            // and isTimeColumn objects to each columns
            const disabledColumn =
                col?.stype === "unsupported" || col?.dtype === "unsupported";
            const new_val = Object.assign(sourceColumns[idx], {
                sample_rows: nonEmpty === null ? nonEmpty : sampleRows,
                stype: col?.stype,
                multicategorical_delimiter:
                    columns[i].multicategorical_delimiter,
                sTypeLabel: sTypeLabel,
                disableSelection: disabledColumn,
                noOfRows: rows?.length,
                nonEmpty: nonEmpty,
                isSampleAvailable: true
            });
            copySource[idx] = new_val;
        }
    }
    return [
        copySource,
        columnsWithNewSTypes as {
            name: string;
            newStype: string;
            multicategoricalDelimiter?: "," | "|" | ";" | "\t" | null;
        }[]
    ];
}

/**
 * Applies stype recommendations to the wizard's column state.
 *
 * `markUserUpdated` distinguishes a user-initiated apply (the banner's
 * buttons), which pins the columns as user-updated, from the wizard's
 * automatic applies, which must stay overridable by a later stats-based
 * inference result (kumo-ai/kumo#30131). `checkUserUpdated` skips columns
 * the user already retyped in either case.
 */
export function updateWithRecommendeStype(
    columnsList: ColumnArgs[],
    selectedColumnSType: DropdownOption[],
    tableArgs: KumoapiTableTableDefinition,
    newStypes: RecommendedStype[],
    checkUserUpdated: boolean,
    markUserUpdated: boolean = true
) {
    const updatedColumn = [...columnsList];
    const updatedSelectStype = selectedColumnSType;
    const newArgs = { ...tableArgs };
    updatedColumn.forEach((column, index) =>
        applyRecommendedStype(
            column,
            index,
            newStypes,
            checkUserUpdated,
            markUserUpdated,
            updatedColumn,
            updatedSelectStype,
            newArgs
        )
    );
    return {
        newArgs,
        updatedColumn,
        updatedSelectStype
    };
}

function delimiterForRecommendedStype(
    stype: string,
    dtype: string | undefined,
    recommendation: RecommendedStype
): MulticategoricalDelimiter {
    return stype === "multicategorical" && dtype === "string"
        ? (recommendation.multicategoricalDelimiter ?? null)
        : null;
}

function applyRecommendedStype(
    column: ColumnArgs,
    index: number,
    recommendations: RecommendedStype[],
    checkUserUpdated: boolean,
    markUserUpdated: boolean,
    updatedColumns: ColumnArgs[],
    updatedStypes: DropdownOption[],
    tableDefinition: KumoapiTableTableDefinition
): void {
    const recommendation = recommendations.find(
        (candidate) => candidate.name === column.name
    );
    if (!recommendation || (checkUserUpdated && column.userUpdatedType)) return;

    const stype = recommendation.newStype;
    const label = capitalizeFirstLetter(stype);
    const updatedColumn = updatedColumns[index];
    updatedColumn.stype = stype;
    updatedColumn.multicategorical_delimiter = delimiterForRecommendedStype(
        stype,
        updatedColumn.dtype,
        recommendation
    );
    updatedColumn.sTypeLabel = label === "Embedding" ? "Sequence" : label;
    if (markUserUpdated) {
        updatedColumn.userUpdatedType = true;
    }
    updatedStypes[index] = { value: stype, label };

    const configuredColumn = tableDefinition.cols.find(
        (candidate) => candidate.name === column.name
    );
    if (configuredColumn) {
        configuredColumn.stype = stype as KumoapiTypingStype;
        configuredColumn.multicategorical_delimiter =
            delimiterForRecommendedStype(
                stype,
                configuredColumn.dtype,
                recommendation
            );
    }
    if (updatedColumn.isTimeColumn) {
        updatedColumn.isTimeColumn = false;
        tableDefinition.time_col = "";
    }
    if (updatedColumn.isEndTimeColumn) {
        updatedColumn.isEndTimeColumn = false;
        tableDefinition.end_time_col = "";
    }
}

function primaryKeyCandidateScore(
    column: Column,
    cleaned: string,
    normalized: string,
    singular: string,
    normalizedSingular: string
): number {
    const name = column.name?.toLowerCase() ?? "";
    const normalizedName = name.replace(/[_-]/g, "");
    const exactCandidates: Array<[string, number]> = [
        ["id", 100],
        ["pk", 95],
        ["primary_key", 95],
        ["rowid", 90],
        [`${singular}_uuid`, 92],
        [`${singular}_id`, 90],
        [`${cleaned}_uuid`, 89],
        [`${singular}id`, 88],
        [`${singular}key`, 88],
        [`${cleaned}_id`, 82],
        [`${cleaned}id`, 80]
    ];
    const exact = exactCandidates.find(([candidate]) => candidate === name);
    if (exact) return exact[1];
    if (normalizedName === `${normalizedSingular}id`) return 85;
    if (normalizedName === `${normalized}id`) return 75;
    if (/uuid|guid/.test(name)) return 70;
    const sharesPrefix = name.startsWith(normalized.slice(0, 3));
    if (
        sharesPrefix &&
        ["id", "_key", "_identifier"].some((suffix) => name.endsWith(suffix))
    )
        return 68;
    if (sharesPrefix && name.includes("id") && column.stype === "ID") return 67;
    if (/^id[A-Z]/.test(column.name ?? "")) return 65;
    return sharesPrefix && name.endsWith("_id") ? 55 : 0;
}

/**
 * Heuristically determines the primary key column for a given table based on its name and columns.
 * Applies advanced singularization with industry-relevant irregular mappings, strips common suffixes (versions, dates, backups, snapshots),
 * supports snake_case, camelCase, acronyms, and normalized names,
 * recognizes UUID/GUID, rowid, identifier, key, and other conventions,
 * and returns null when ambiguous or no clear PK.
 * @param {string} tableName - Name of the table (e.g., "users", "sales_2021_q1").
 * @param {string[]} columns - Array of column names in the table.
 * @returns {string|null} The column name most likely to be the primary key, or null if none or ambiguous.
 * @reference https://github.com/kumo-ai/kumo/issues/19818
 *
 */
export function detectPrimaryKey(
    tableName: string,
    columns: Column[]
): string | null {
    if (!tableName || !Array.isArray(columns) || columns.length === 0)
        return null;

    const tl = tableName.toLowerCase();
    // strip version/date suffixes
    const cleaned = tl.replace(
        /(_v?\d+|_\d{4}(?:_\d{2}(?:_\d{2})?)?|_backup|_snapshot)$/i,
        ""
    );
    const norm = cleaned.replace(/[_-]/g, "");

    const singularize = (name: string) =>
        name
            .replace(/ies$/, "y")
            .replace(/(ses|xes|ches|shes|oes)$/, (s) => s.replace(/es$/, ""))
            .replace(/s$/, (s) => (s.endsWith("ss") ? s : s.slice(0, -1)));

    const sing = singularize(cleaned);
    const singNorm = sing.replace(/[_-]/g, "");
    let best = 0,
        cands: Column[] = [];

    columns.forEach((col) => {
        const score = primaryKeyCandidateScore(
            col,
            cleaned,
            norm,
            sing,
            singNorm
        );
        if (score > best) {
            best = score;
            cands = [col];
        } else if (score === best && score > 0) cands.push(col);
    });
    // Extra findings if there is only one column and stype is ID.
    if (cands.length < 1 && columns.length === 1) {
        const col = columns[0];
        const low = col?.name?.toLowerCase() || "";
        if (low?.startsWith(norm?.slice(0, 3)) && col?.stype === "ID") {
            best = 75;
            cands = [col];
        }
    }

    return best < 60 || cands.length !== 1 ? null : (cands[0]?.name ?? null);
}

/**
 * Applies the optionally-confirmed create-date / primary-key selections onto a
 * (shallow-copied) table definition, mirroring the submit flow.
 */
export function applyKeyOverrides(
    tmpArgs: KumoapiTableTableDefinition,
    timeCol?: string,
    pkey?: string
): void {
    if (timeCol) {
        tmpArgs.time_col = timeCol;
    }
    if (pkey) {
        tmpArgs.pkey = pkey;
        const pKeyIndex = tmpArgs?.cols?.findIndex((c) => c.name === pkey);
        if (pKeyIndex !== -1) {
            tmpArgs.cols[pKeyIndex].stype = "ID";
        }
    }
}

/**
 * Build the create-request definition for a Custom SQL table: swap the source
 * table for a `CustomSQLSourceTable` carrying the connector + defining SELECT,
 * and strip the previewed columns down to the API `Column` shape (mirroring the
 * raw-table submit path). The pkey / time-col roles already live on the incoming
 * definition (set from the column selection), so they pass through unchanged.
 *
 * `connector_type` and `referenced_tables` are server-owned (the server derives
 * them from the connector and the SQL), so the client omits them. The backend
 * also recomputes the synthetic `table` name from the normalized SQL on persist,
 * so a provisional name (the display name) is sufficient here.
 */
export function buildCustomSqlTableArgs(
    tableArgs: KumoapiTableTableDefinition,
    connectorId: string,
    sql: string,
    syntheticName: string
): KumoapiTableTableDefinition {
    const sourceTable: CustomSQLSourceTable = {
        connector_id: connectorId,
        sql,
        table: syntheticName,
        data_source_type: "CUSTOM_SQL"
    };
    return {
        ...tableArgs,
        // The authoring surface leaves UI-only fields (labels, selection flags,
        // preview rows) on each column; strip them to the API `Column` shape,
        // as the raw submit path does.
        cols: (tableArgs.cols ?? []).map((col) =>
            toApiColumn(col as ColumnArgs)
        ),
        source_table:
            sourceTable as unknown as KumoapiTableTableDefinitionSourceTable
    };
}

/**
 * Fields required to normalize an S3 table's source metadata on submit.
 */
export interface S3SourceOverrideContext {
    connectorNotExists: boolean;
    sourcetableNotExists: boolean;
    flowType: FlowType;
    s3ConnctorConfig?: string | { id?: string };
    unnamedConSourceTable: string;
}

/**
 * Restores project-scoped S3 connector/source metadata for an edited table
 * whose connector or source was dropped from the form state.
 */
export function applyS3Overrides(
    tmpArgs: KumoapiTableTableDefinition,
    ctx: S3SourceOverrideContext
): void {
    if (
        ctx.connectorNotExists &&
        ctx.sourcetableNotExists &&
        ctx.flowType === FlowType.EDIT &&
        tmpArgs?.source_table?.data_source_type === DataSourceType.S3
    ) {
        tmpArgs.source_table.connector_id =
            (ctx.s3ConnctorConfig as { id?: string })?.id ?? "";
        tmpArgs.source_table.source = ctx.unnamedConSourceTable.startsWith(
            "s3://"
        )
            ? { mode: "uri", uri: ctx.unnamedConSourceTable }
            : { mode: "relative", path: ctx.unnamedConSourceTable };
    }
}

/** Default (empty) S3 table definition used to seed the create/edit form. */
export const defaultTableData: KumoapiTableTableDefinition = {
    cols: [],
    source_table: {
        connector_id: "",
        source: { mode: "relative", path: "" },
        file_type: KumoapiSourceTableFileType.PARQUET,
        data_source_type: DataSourceType.S3
    },
    pkey: "",
    time_col: "",
    end_time_col: ""
};

/** Source dropdown option representing the local-upload flow. */
export const localUploadSource: DropdownOption = {
    value: "local",
    label: "Local Upload"
};

/**
 * Builds the "Source" dropdown options. Local Upload is only offered when the
 * deployment is not Databricks.
 */
export function getSourceTypeOptions(
    flags: SourceTypeDeploymentFlags
): Array<{ label: string; value: string }> {
    const sourceType = [{ label: "Connector", value: "connector" }];
    if (!flags.IS_DATABRICKS_DEPLOY) {
        sourceType.push({ label: "Local Upload", value: "local" });
    }
    // Custom SQL tables are authored against a warehouse connector; gated behind
    // a dark-launch toggle.
    if (ENABLE_CUSTOM_SQL_TABLES) {
        sourceType.push({ label: "Custom SQL", value: "sql" });
    }
    return sourceType;
}

/**
 * Seeds the initial table definition for the create/edit form, applying the
 * local-upload connector and any create/copy args carried via route state.
 */
export function getInitialTableArgs(
    isLocalUploadFlow: boolean,
    locationState?: ConnectTablesLocationState,
    uploadConnectorId = ""
): KumoapiTableTableDefinition {
    return {
        ...defaultTableData,
        ...(isLocalUploadFlow
            ? {
                  source_table: getSourceInfoFromConnectorConfig(
                      null,
                      uploadConnectorId
                  ) as KumoapiTableTableDefinitionSourceTable
              }
            : {}),
        ...(locationState?.createtableargs ?? {}),
        ...locationState?.copytableargs
    };
}

/**
 * Returns a table name that is unique against `existingNames` by appending an
 * incrementing `_n` suffix.
 */
export function generateUniqueTableName(
    name: string,
    existingNames: Array<string | undefined>
): string {
    let uniqueName = name;
    let counter = 1;
    while (existingNames?.includes(uniqueName)) {
        uniqueName = `${name}_${counter}`;
        counter++;
    }
    return uniqueName;
}

/** Collapses runs of underscores in a (table) name to a single underscore. */
export function removeUnderScore(name: string): string {
    return name?.replace(/_+/g, "_");
}

/**
 * Validates that the columns referenced as pkey / create-date / end-date are
 * still selected, and that at least one column is selected. Returns the
 * user-facing error message and offending column name (both empty when valid).
 */
export function getMissingColumnError(tableArgs: KumoapiTableTableDefinition): {
    err: string;
    colName: string;
} {
    const cols = tableArgs?.cols ?? [];
    const colNames = cols.map((c) => c?.name);
    const keyRoles: Array<{ col?: string | null; role: string }> = [
        { col: tableArgs?.pkey, role: "Primary Key" },
        { col: tableArgs?.time_col, role: "Create Date" },
        { col: tableArgs?.end_time_col, role: "End Date" }
    ];
    for (const { col, role } of keyRoles) {
        if (col && !colNames.includes(col)) {
            return {
                colName: col,
                err: `Please select the column "${col}" that is selected as ${role}`
            };
        }
    }
    if (cols.length < 1) {
        return { err: "Please select one or more column(s).", colName: "" };
    }
    return { err: "", colName: "" };
}

/**
 * Flattens a failed table-validation response into a user-facing `CustomError`.
 */
export function buildValidationError(
    response: ValidationResponse
): CustomError {
    const errors: string[] | undefined = response?.errors
        ?.map((e) => e.message)
        .filter((e): e is string => e !== undefined);
    const msg: string | undefined =
        errors && errors?.length > 1 ? errors?.join(", ") : errors?.[0];
    return {
        response: {
            status: errorStatusCode.UnProcessableEntityError,
            data: msg
        },
        name: "",
        message: msg as string
    };
}
