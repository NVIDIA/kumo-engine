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
import Spacer from "react-spacer";

import {
    Alert,
    AlertDescription,
    AlertTitle,
    Badge,
    Button,
    Description,
    SectionHeader,
    useDebounce
} from "@kumo-ai/kumo-ui-catalog";
import { useCreateV1TablesResolveSql } from "gen/v1/api/tables/tables";
import {
    Column,
    Dtype as KumoapiTypingDtype,
    KumoapiTableTableDefinition,
    Stype as KumoapiTypingStype
} from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import SQLCodeEditor from "components/CodeEditor/SQLCodeEditor";

import { useGetColumns } from "hooks/connector/connector";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { DropdownOption } from "types/inputs";
import { ColumnArgs } from "types/tables";

import { parseErrorMessage } from "utils/errorHandler";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import {
    assignSampleRows,
    buildCustomSqlTableArgs,
    toApiColumn
} from "utils/tables";

import { ColumnSelectionTable } from "./ColumnSelectionTable";
import { catalogToSchema, mockSqlCatalog } from "./customSqlCatalog";

// Discovery runs a warehouse describe on every settled edit; debounce it so
// typing never fires a per-keystroke call.
const SQL_DISCOVERY_DEBOUNCE_MS = 500;

// The scoped (RBAC group) connector rejection is surfaced as a friendly
// "unavailable" notice rather than a hard error.
const SCOPED_CONNECTOR_PATTERN = /scoped|rbac group/i;

export interface CustomSqlTableSourceProps {
    connectorId: string;
    // Current draft definition, used to build the `resolve_sql` body (source
    // connector + declared columns).
    tableDefinition: KumoapiTableTableDefinition;
    setTableArgs: React.Dispatch<
        React.SetStateAction<KumoapiTableTableDefinition>
    >;
    // SQL draft lifted to the parent so the Add flow can gate on validation.
    sqlText: string;
    setSqlText: (value: string) => void;
    validated: boolean;
    setValidated: (value: boolean) => void;
}

// Preview row = the API `Column` contract plus the display labels and the
// pkey/time selection flags the shared `ColumnSelectionTable` reads. Being a
// `Column` superset it flows into `tableArgs.cols` without a cast.
type PreviewColumn = Column &
    Pick<
        ColumnArgs,
        | "dTypeLabel"
        | "sTypeLabel"
        | "isPrimary"
        | "isTimeColumn"
        | "isEndTimeColumn"
        | "sample_rows"
        | "isSampleAvailable"
        | "nonEmpty"
    >;

interface AuthoringError {
    message: string;
    scoped: boolean;
}

interface EditSeed {
    sql: string;
    pkey: string;
    timeCol: string;
    endTimeCol: string;
}

/** Edit seed: when the surface opens on an existing SQL table (a non-empty
 * draft that already carries persisted columns), the first discovery describes
 * the same, already-valid query. Capturing the persisted role assignments
 * (primary key / create-date / end-date) lets that initial describe restore
 * them instead of silently dropping them, and lets the surface open already
 * "validated" so Save is enabled without a redundant re-validate. A create-flow
 * draft — empty SQL or no persisted columns — has no seed. */
const initialEditSeed = (
    sqlText: string,
    tableDefinition: KumoapiTableTableDefinition
): EditSeed | null => {
    const sql = sqlText.trim();
    if (sql === "" || (tableDefinition.cols?.length ?? 0) === 0) {
        return null;
    }
    return {
        sql,
        pkey: (tableDefinition.pkey as string) ?? "",
        timeCol: (tableDefinition.time_col as string) ?? "",
        endTimeCol: (tableDefinition.end_time_col as string) ?? ""
    };
};

const toTypeOption = (value?: string, label?: string): DropdownOption => ({
    value,
    label,
    origValue: value
});

const capitalize = (value?: string | null) =>
    value ? capitalizeFirstLetter(value) : "";

/** Map a discovered/validated column onto a preview row. */
const toPreviewColumn = (col: {
    name?: string;
    dtype?: KumoapiTypingDtype | null;
    stype?: KumoapiTypingStype | null;
    timestamp_format?: Column["timestamp_format"];
    multicategorical_delimiter?: Column["multicategorical_delimiter"];
}): PreviewColumn => ({
    name: col.name ?? "",
    dtype: (col.dtype ?? "") as KumoapiTypingDtype,
    stype: (col.stype ?? "") as KumoapiTypingStype,
    timestamp_format: col.timestamp_format,
    multicategorical_delimiter: col.multicategorical_delimiter,
    dTypeLabel: capitalize(col.dtype),
    sTypeLabel: capitalize(col.stype),
    isPrimary: false,
    isTimeColumn: false,
    isEndTimeColumn: false
});

/** Turn any discovery/validation failure into a surfaceable notice, tagging the
 * scoped-connector rejection so it reads as "unavailable" rather than an error. */
const toAuthoringError = (err: Error): AuthoringError => {
    const message = parseErrorMessage(err as CustomError);
    return { message, scoped: SCOPED_CONNECTOR_PATTERN.test(message) };
};

/** Identity of a validation request: connector + query. A response is stale
 * (and discarded) once this no longer matches the surface's current request —
 * covering both an SQL edit and a connector switch that keeps the same SQL. */
const sqlRequestId = (connectorId: string, sql: string) =>
    `${connectorId}\n${sql.trim()}`;

/** Post-Validate feedback: empty-query hint, connector/validation errors. */
const SqlValidationNotices: React.FC<{
    noColumnsDiscovered: boolean;
    authoringError: AuthoringError | null;
    errorMessages: string[];
}> = ({ noColumnsDiscovered, authoringError, errorMessages }) => (
    <>
        {noColumnsDiscovered && (
            <Alert variant="warn" className="mb-3">
                <AlertTitle>This query returns no columns</AlertTitle>
                <AlertDescription>
                    Write a SELECT that outputs at least one column to continue.
                </AlertDescription>
            </Alert>
        )}
        {authoringError && (
            <Alert
                variant={authoringError.scoped ? "warn" : "destructive"}
                className="mb-3">
                <AlertTitle>
                    {authoringError.scoped
                        ? "Custom SQL isn't available on this connector"
                        : "Couldn't validate the SQL"}
                </AlertTitle>
                <AlertDescription>{authoringError.message}</AlertDescription>
            </Alert>
        )}
        {errorMessages.length > 0 && (
            <Alert variant="destructive" className="mb-3">
                <AlertTitle>Fix the SQL definition</AlertTitle>
                <AlertDescription>{errorMessages.join(" ")}</AlertDescription>
            </Alert>
        )}
    </>
);

interface SqlColumnsPreviewProps {
    validated: boolean;
    discovering: boolean;
    sampleLoaded: boolean;
    previewNote: string | null;
    columns: PreviewColumn[];
    selected: string[];
    columnDTypes: DropdownOption[];
    columnSTypes: DropdownOption[];
    onSelectColumns: (rows: ColumnArgs[]) => void;
    onSetColumnDataType: (e: DropdownOption, rowIndex: number) => void;
    onSetColumnSType: (e: DropdownOption, rowIndex: number) => void;
    onSelectPrimaryKey: (index: number) => void;
    onCreateTimeColSelect: (rowIndex: number) => void;
    onEndTimeColSelect: (rowIndex: number) => void;
}

/** Discovered / confirmed column list. Validation returns a bounded preview
 * with the schema, so the rows render inline once confirmed; a `previewNote`
 * explains a query whose preview the cost gate skipped. */
const SqlColumnsPreview: React.FC<SqlColumnsPreviewProps> = ({
    validated,
    discovering,
    sampleLoaded,
    previewNote,
    columns,
    selected,
    columnDTypes,
    columnSTypes,
    onSelectColumns,
    onSetColumnDataType,
    onSetColumnSType,
    onSelectPrimaryKey,
    onCreateTimeColSelect,
    onEndTimeColSelect
}) => (
    <>
        <Spacer height={16} />
        <SectionHeader
            title={validated ? "Columns" : "Discovered columns"}
            titleSize="md"
            description={
                validated
                    ? "Confirmed schema with warehouse-authoritative types."
                    : "Validate to confirm types and referenced tables."
            }
        />
        {previewNote && (
            <>
                <Spacer height={8} />
                <Description size="sm" className="m-0">
                    {previewNote}
                </Description>
            </>
        )}
        <Spacer height={8} />
        <div className="w-full overflow-auto">
            <ColumnSelectionTable
                columns={columns as ColumnArgs[]}
                selected={selected}
                showPreview={sampleLoaded}
                onSelectColumns={onSelectColumns}
                columnDTypes={columnDTypes}
                columnSTypes={columnSTypes}
                onCreateTimeColSelect={onCreateTimeColSelect}
                onEndTimeColSelect={onEndTimeColSelect}
                onSelectPrimaryKey={onSelectPrimaryKey}
                onSetColumnDataType={onSetColumnDataType}
                onSetColumnSType={onSetColumnSType}
                isLoading={discovering}
            />
        </div>
    </>
);

/**
 * Custom SQL authoring surface for the "Add table" flow.
 *
 * Discover-then-validate: as the SQL settles (debounced), `useGetColumns` with a
 * `query` describes the output columns and populates the list; the explicit
 * "Validate" action then confirms the contract via `resolve_sql`, applying
 * authoritative stypes and surfacing referenced tables + errors. Validation also
 * returns a bounded preview (Q2), so the sample rows render inline once
 * confirmed — there is no separate sample fetch.
 */
const CustomSqlTableSource: React.FC<CustomSqlTableSourceProps> = ({
    connectorId,
    tableDefinition,
    setTableArgs,
    sqlText,
    setSqlText,
    validated,
    setValidated
}) => {
    const [validation, setValidation] = useState<{
        referenced_tables: string[];
        error_messages: string[];
    } | null>(null);

    // Column-preview state, mirroring the parallel-array contract that
    // `ColumnSelectionTable` expects from the raw-table preprocess hook.
    const [columns, setColumns] = useState<PreviewColumn[]>([]);
    const [columnDTypes, setColumnDTypes] = useState<DropdownOption[]>([]);
    const [columnSTypes, setColumnSTypes] = useState<DropdownOption[]>([]);
    const [selected, setSelected] = useState<string[]>([]);

    const [authoringError, setAuthoringError] = useState<AuthoringError | null>(
        null
    );
    // Validation returns a bounded preview alongside the schema (Q2), so the
    // rows render inline once confirmed — no separate sample fetch. `sampleLoaded`
    // gates the preview column in the table; `previewNote` explains a skip.
    const [sampleLoaded, setSampleLoaded] = useState(false);
    const [previewNote, setPreviewNote] = useState<string | null>(null);

    const catalog = useMemo(() => catalogToSchema(mockSqlCatalog), []);

    const debouncedSql = useDebounce(sqlText.trim(), SQL_DISCOVERY_DEBOUNCE_MS);
    const discoverEnabled = connectorId !== "" && debouncedSql !== "";

    // Identity of the request the surface currently reflects, so an in-flight
    // validation that resolves after the draft or connector moved on can be
    // discarded. Synced in an effect for prop-driven changes and updated
    // synchronously in `onSqlChange` to close the pre-render window.
    const latestRequestRef = useRef(sqlRequestId(connectorId, sqlText));
    useEffect(() => {
        latestRequestRef.current = sqlRequestId(connectorId, sqlText);
    }, [connectorId, sqlText]);

    // Seeded (see `initialEditSeed`) when the surface opens on an existing SQL
    // table, then cleared the moment the user edits the SQL — after which
    // discovery behaves exactly as in the create flow.
    const editSeedRef = useRef<EditSeed | null>(
        initialEditSeed(sqlText, tableDefinition)
    );

    const noop = useCallback(() => {}, []);

    const clearColumns = useCallback(() => {
        setColumns([]);
        setColumnDTypes([]);
        setColumnSTypes([]);
        setSelected([]);
    }, []);

    // Discovery: describe the query's output columns (no data scan). A failure
    // clears the (now unreliable) column list so it can't be validated.
    const handleDiscoverError = useCallback(
        (err: Error) => {
            clearColumns();
            setValidation(null);
            setValidated(false);
            setAuthoringError(toAuthoringError(err));
        },
        [clearColumns, setValidated]
    );

    const { columns: discovered, isLoading: discovering } = useGetColumns(
        {
            connectorId,
            params: { source_table_name: "", query: debouncedSql }
        },
        handleDiscoverError,
        noop,
        discoverEnabled
    );

    // A connector switch changes the schema entirely — reset the surface.
    useEffect(() => {
        setValidation(null);
        setValidated(false);
        setAuthoringError(null);
        setSampleLoaded(false);
        setPreviewNote(null);
        setColumns([]);
        setColumnDTypes([]);
        setColumnSTypes([]);
        setSelected([]);
    }, [connectorId, setValidated]);

    // Seed the column list from discovery. A new result means a new schema, so
    // any prior validation/sample is stale. An empty result (query returns no
    // columns) resets the list so nothing stale can be validated.
    useEffect(() => {
        // `discovered === undefined` means discovery is disabled or still
        // loading — keep the current list; Validate is gated on `discovering`.
        // Skip results that don't match the editor's current text: an in-flight
        // fetch for a since-edited query must not repopulate stale columns.
        if (
            !discoverEnabled ||
            !discovered ||
            debouncedSql !== sqlText.trim()
        ) {
            return;
        }
        // On the initial (unedited) describe of an edited SQL table, carry the
        // persisted role assignments onto the freshly described schema.
        const seed =
            editSeedRef.current && debouncedSql === editSeedRef.current.sql
                ? editSeedRef.current
                : null;
        const previewCols = discovered.map((col) => {
            const preview = toPreviewColumn(col);
            if (!seed || preview.name === "") {
                return preview;
            }
            // Restore the role a persisted column held (matched by name). A
            // primary key is always an ID, mirroring the create-flow rule.
            const isPrimary = preview.name === seed.pkey;
            return {
                ...preview,
                isPrimary,
                isTimeColumn: preview.name === seed.timeCol,
                isEndTimeColumn: preview.name === seed.endTimeCol,
                ...(isPrimary
                    ? {
                          stype: "ID" as KumoapiTypingStype,
                          sTypeLabel: "ID"
                      }
                    : {})
            };
        });
        setColumns(previewCols);
        setColumnDTypes(
            previewCols.map((c) => toTypeOption(c.dtype, c.dTypeLabel))
        );
        setColumnSTypes(
            previewCols.map((c) => toTypeOption(c.stype, c.sTypeLabel))
        );
        setSelected(previewCols.map(() => "true"));
        setValidation(null);
        // The persisted definition was already valid, so an unedited edit seed
        // starts validated; a create-flow discovery still requires Validate.
        setValidated(Boolean(seed));
        setAuthoringError(null);
        setSampleLoaded(false);
        setPreviewNote(null);
    }, [discovered, discoverEnabled, debouncedSql, sqlText, setValidated]);

    // Single source of truth for the definition: derive cols and the pkey /
    // time / end-time roles from the previewed selection + exclusive row flags
    // (mirrors raw-table preprocessing).
    useEffect(() => {
        const selectedCols = columns.filter((_, i) => selected[i] === "true");
        setTableArgs((args) => ({
            ...args,
            cols: selectedCols,
            pkey: selectedCols.find((c) => c.isPrimary)?.name ?? "",
            time_col: selectedCols.find((c) => c.isTimeColumn)?.name ?? "",
            end_time_col:
                selectedCols.find((c) => c.isEndTimeColumn)?.name ?? ""
        }));
    }, [columns, selected, setTableArgs]);

    const onSqlChange = useCallback(
        (value: string) => {
            // CodeMirror may report its initial document after mounting. A
            // no-op notification must not invalidate an existing table whose
            // persisted SQL and schema are already confirmed.
            if (value === sqlText) {
                return;
            }
            // Update synchronously (ahead of the passive effect) so an in-flight
            // validation that resolves before the next render is recognized as
            // stale and discarded.
            latestRequestRef.current = sqlRequestId(connectorId, value);
            // The user has taken over the draft: drop the edit seed so the next
            // discovery repopulates from the query schema like the create flow
            // (fresh output schema, requires re-validation).
            editSeedRef.current = null;
            setSqlText(value);
            // The SQL changed: the prior discovery/validation no longer
            // describes this query. Clear the list so nothing stale is shown or
            // validated; debounced discovery repopulates once typing settles.
            clearColumns();
            setValidation(null);
            setValidated(false);
            setAuthoringError(null);
            setSampleLoaded(false);
            setPreviewNote(null);
        },
        [connectorId, sqlText, setSqlText, setValidated, clearColumns]
    );

    const projectId = useProjectId() ?? "";
    const { mutate: validateSql, isPending: validating } =
        useCreateV1TablesResolveSql();

    // Apply the validated result onto the discovered columns: warehouse-
    // authoritative stypes plus the bounded preview rows that `resolve_sql`
    // returns alongside the schema (Q2). No separate sample fetch.
    const applyValidatedColumns = useCallback(
        (authoritative: Column[], previewRows: (string | null)[][]) => {
            const byName = new Map(authoritative.map((c) => [c.name, c]));
            setColumns((prev) => {
                let next = prev.map((col) => {
                    const match = col.name ? byName.get(col.name) : undefined;
                    return match
                        ? {
                              ...col,
                              dtype: match.dtype as KumoapiTypingDtype,
                              stype: match.stype as KumoapiTypingStype,
                              timestamp_format: match.timestamp_format,
                              multicategorical_delimiter:
                                  match.multicategorical_delimiter,
                              dTypeLabel: capitalize(match.dtype),
                              sTypeLabel: capitalize(match.stype)
                          }
                        : col;
                });
                // Fold the preview rows (row-major, aligned to `authoritative`)
                // onto the columns so the table renders them inline.
                if (previewRows.length > 0) {
                    const [withSamples] = assignSampleRows(
                        authoritative as unknown as ColumnArgs[],
                        next as ColumnArgs[],
                        previewRows as string[][]
                    );
                    next = withSamples as PreviewColumn[];
                }
                setColumnDTypes(
                    next.map((c) => toTypeOption(c.dtype, c.dTypeLabel))
                );
                setColumnSTypes(
                    next.map((c) => toTypeOption(c.stype, c.sTypeLabel))
                );
                return next;
            });
        },
        []
    );

    const onValidate = useCallback(() => {
        const submittedQuery = sqlText.trim();
        const submittedRequest = sqlRequestId(connectorId, submittedQuery);
        const declaredCols: Column[] = columns
            .filter((_, index) => selected[index] === "true")
            .map(toApiColumn);
        // Build the same Custom SQL `source_table` the create path submits
        // (`buildCustomSqlTableArgs`): the SELECT rides on the source table so
        // the backend recognizes the SQL table (`is_sql`) and derives the
        // server-owned `connector_type` from the connector it resolves. The
        // synthetic `table` name is provisional here — the server recomputes it
        // from the normalized SQL.
        const { cols: columnHints, ...request } = buildCustomSqlTableArgs(
            { ...tableDefinition, cols: declaredCols },
            connectorId,
            submittedQuery,
            "custom_sql"
        );
        if (!("sql" in request.source_table)) {
            throw new Error("Custom SQL resolution requires a SQL source.");
        }
        validateSql(
            {
                data: {
                    ...request,
                    source_table: request.source_table,
                    column_hints:
                        columnHints.length > 0 ? columnHints : undefined
                },
                params: { project_id: projectId }
            },
            {
                onSuccess: (resp) => {
                    // Discard if the connector or draft moved on while the
                    // validation was pending.
                    if (latestRequestRef.current !== submittedRequest) {
                        return;
                    }
                    const result = resp.data;
                    const definition = result.table_definition;
                    const resolvedColumns = definition?.cols ?? [];
                    const errorMessages = (result.errors ?? []).map(
                        (error) => error.message
                    );
                    const referencedTables =
                        definition &&
                        "referenced_tables" in definition.source_table
                            ? (definition.source_table.referenced_tables ?? [])
                            : [];
                    setAuthoringError(null);
                    // A valid definition must return at least one column; fall
                    // back to an explicit message if the backend sent none.
                    const confirmed =
                        errorMessages.length === 0 &&
                        resolvedColumns.length > 0;
                    setValidation({
                        referenced_tables: referencedTables,
                        error_messages:
                            confirmed || errorMessages.length > 0
                                ? errorMessages
                                : [
                                      "The query returned no columns. Adjust " +
                                          "the SELECT to return at least one " +
                                          "column."
                                  ]
                    });
                    if (confirmed) {
                        // Apply authoritative stypes + the bounded preview rows
                        // the response carries; an oversized/cost-gated query
                        // returns no rows and a note explaining the skip.
                        const previewRows = result.preview_rows ?? [];
                        applyValidatedColumns(resolvedColumns, previewRows);
                        setSampleLoaded(previewRows.length > 0);
                        setPreviewNote(result.preview_note ?? null);
                        setValidated(true);
                    } else {
                        setSampleLoaded(false);
                        setPreviewNote(null);
                        setValidated(false);
                    }
                },
                onError: (err) => {
                    if (latestRequestRef.current !== submittedRequest) {
                        return;
                    }
                    setValidation(null);
                    setValidated(false);
                    setSampleLoaded(false);
                    setPreviewNote(null);
                    setAuthoringError(toAuthoringError(err as Error));
                }
            }
        );
    }, [
        columns,
        connectorId,
        projectId,
        selected,
        sqlText,
        tableDefinition,
        validateSql,
        setValidated,
        applyValidatedColumns
    ]);

    const onSelectColumns = useCallback(
        (selectedRows: ColumnArgs[]) => {
            const selectedNames = new Set(selectedRows.map((c) => c.name));
            const next = columns.map((c) => String(selectedNames.has(c.name)));
            const changed =
                next.length !== selected.length ||
                next.some((value, index) => value !== selected[index]);
            if (changed) {
                setSelected(next);
                setValidated(false);
            }
        },
        [columns, selected, setValidated]
    );

    const onSetColumnDataType = useCallback(
        (e: DropdownOption, rowIndex: number) => {
            setColumnDTypes((prev) => {
                const next = [...prev];
                next[rowIndex] = { value: e?.value, label: e?.label };
                return next;
            });
            setColumns((prev) => {
                const next = [...prev];
                next[rowIndex] = {
                    ...next[rowIndex],
                    dtype: e?.value as KumoapiTypingDtype,
                    dTypeLabel: e?.label
                };
                return next;
            });
            setValidated(false);
        },
        [setValidated]
    );

    const onSetColumnSType = useCallback(
        (e: DropdownOption, rowIndex: number) => {
            setColumnSTypes((prev) => {
                const next = [...prev];
                next[rowIndex] = { value: e?.value, label: e?.label };
                return next;
            });
            setColumns((prev) => {
                const next = [...prev];
                next[rowIndex] = {
                    ...next[rowIndex],
                    stype: e?.value as KumoapiTypingStype,
                    sTypeLabel: e?.label
                };
                return next;
            });
            setValidated(false);
        },
        [setValidated]
    );

    // Each property (primary key / create date / end date) is exclusive across
    // rows, and a single row holds at most one — mirroring raw preprocessing.
    const onSelectPrimaryKey = useCallback(
        (index: number) => {
            setColumns((prev) =>
                prev.map((col, i) =>
                    i === index
                        ? {
                              ...col,
                              isPrimary: true,
                              isTimeColumn: false,
                              isEndTimeColumn: false,
                              // Primary keys are IDs.
                              stype: "ID" as KumoapiTypingStype,
                              sTypeLabel: "ID"
                          }
                        : { ...col, isPrimary: false }
                )
            );
            setColumnSTypes((prev) => {
                const next = [...prev];
                next[index] = { value: "ID", label: "ID" };
                return next;
            });
            setValidated(false);
        },
        [setValidated]
    );

    const onCreateTimeColSelect = useCallback(
        (rowIndex: number) => {
            setColumns((prev) =>
                prev.map((col, i) =>
                    i === rowIndex
                        ? {
                              ...col,
                              isTimeColumn: true,
                              isPrimary: false,
                              isEndTimeColumn: false
                          }
                        : { ...col, isTimeColumn: false }
                )
            );
            setValidated(false);
        },
        [setValidated]
    );

    const onEndTimeColSelect = useCallback(
        (rowIndex: number) => {
            setColumns((prev) =>
                prev.map((col, i) =>
                    i === rowIndex
                        ? {
                              ...col,
                              isEndTimeColumn: true,
                              isPrimary: false,
                              isTimeColumn: false
                          }
                        : { ...col, isEndTimeColumn: false }
                )
            );
            setValidated(false);
        },
        [setValidated]
    );

    const hasColumns = columns.length > 0;
    const errorMessages = validation?.error_messages ?? [];
    const referencedTables = validation?.referenced_tables ?? [];
    const noColumnsDiscovered =
        discoverEnabled && !discovering && !authoringError && !hasColumns;
    const validateDisabled =
        sqlText.trim() === "" ||
        connectorId === "" ||
        !hasColumns ||
        // Guard the stale window: don't validate until discovery has settled on
        // the columns for the query currently in the editor.
        discovering ||
        debouncedSql !== sqlText.trim() ||
        validating;

    return (
        <div className="w-full">
            <Description size="sm" className="leading-5 m-0">
                Define this table with a read-only SELECT. The query runs in
                your warehouse and only the results are ingested.
            </Description>
            <Spacer height={8} />
            <SqlValidationNotices
                noColumnsDiscovered={noColumnsDiscovered}
                authoringError={authoringError}
                errorMessages={errorMessages}
            />
            <SQLCodeEditor
                value={sqlText}
                onChange={onSqlChange}
                catalog={catalog}
                height="220px"
            />
            <Spacer height={12} />
            <Button
                id="validate-custom-sql"
                onClick={onValidate}
                disabled={validateDisabled}>
                Validate
            </Button>

            {referencedTables.length > 0 && (
                <>
                    <Spacer height={16} />
                    <SectionHeader
                        title="Referenced tables"
                        titleSize="md"
                        description="External tables this query reads from."
                    />
                    <div className="flex flex-wrap gap-2 mt-1">
                        {referencedTables.map((table) => (
                            <Badge key={table} size="sm" variant="default">
                                {table}
                            </Badge>
                        ))}
                    </div>
                </>
            )}

            {hasColumns && (
                <SqlColumnsPreview
                    validated={validated}
                    discovering={discovering}
                    sampleLoaded={sampleLoaded}
                    previewNote={previewNote}
                    columns={columns}
                    selected={selected}
                    columnDTypes={columnDTypes}
                    columnSTypes={columnSTypes}
                    onSelectColumns={onSelectColumns}
                    onSetColumnDataType={onSetColumnDataType}
                    onSetColumnSType={onSetColumnSType}
                    onSelectPrimaryKey={onSelectPrimaryKey}
                    onCreateTimeColSelect={onCreateTimeColSelect}
                    onEndTimeColSelect={onEndTimeColSelect}
                />
            )}
        </div>
    );
};

export default WithErrorBoundary(CustomSqlTableSource);
