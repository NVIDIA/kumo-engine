// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Dispatch,
    SetStateAction,
    useCallback,
    useEffect,
    useMemo,
    useState
} from "react";
import Spacer from "react-spacer";

import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
    Button,
    Cable,
    cn,
    DeleteIcon,
    GridCol,
    GridRow,
    Key,
    Label,
    Select,
    Skeleton,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    useWindowDimensions
} from "@kumo-ai/kumo-ui-catalog";
import { ColumnKey } from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useValidationState } from "hooks/validation/validation";

import { CustomError } from "types/common";
import {
    ColumnGroup,
    GraphFlowInfo,
    GraphLinkColumnOption,
    GraphLinkGroup,
    GraphTables
} from "types/graphs";
import { DropdownOption } from "types/inputs";
import { ColumnArgs } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import { getKeys, isObjectEmpty } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

const createEmptyLinkGroup = (): GraphLinkGroup => ({
    columnOptions: [],
    selectedTable: { label: "", value: "" },
    selectedColumn: { label: "", value: "" },
    invalid: false
});

export interface GraphLinkGroupsProps {
    allTables: GraphTables;
    graphFlowInfo: GraphFlowInfo;
    groups: Array<ColumnGroup>;
    setGroups: Dispatch<SetStateAction<Array<ColumnGroup>>>;
    backwardLinks: {
        [key: string]: {
            tableName: string;
            columnName: string;
            pkeyName: string;
        }[];
    };
    handleError: HandleErrorFunc;
    show?: boolean;
    onRemoveTable?: (tableName: string) => void;
}

export const GraphLinkGroups = ({
    allTables,
    graphFlowInfo,
    groups,
    setGroups,
    backwardLinks,
    handleError,
    show = true,
    onRemoveTable
}: GraphLinkGroupsProps) => {
    const [activeKeys, setActiveKey] = useState<string[]>([]); // Track the active key

    if (!show) return null;

    return (
        <Accordion
            {...{ type: "multiple" }}
            value={activeKeys}
            onValueChange={setActiveKey}
            id={"graph-link"}>
            {groups.length > 0 &&
                groups?.map((data: ColumnGroup, index: number) => (
                    <LinkGroup
                        key={`group-con-${index}`}
                        id={index}
                        allTables={allTables}
                        graphFlowInfo={graphFlowInfo}
                        groups={groups}
                        setGroups={setGroups}
                        activeKeys={activeKeys}
                        handleErrorParent={handleError}
                        onRemoveTable={onRemoveTable}
                    />
                ))}
            {getKeys(backwardLinks)?.map((link, i) => (
                <AccordionItem
                    key={"bklinks" + i}
                    value={"bklinks-" + i}
                    className={"p-0 m-0"}>
                    <div className="flex items-center gap-2">
                        <div className="flex-1 min-w-0">
                            <AccordionTrigger
                                data-testid={"bklinks-" + i}
                                showArrow={false}>
                                <GroupHeader
                                    id={groups.length + i}
                                    tableName={link}
                                    pkeyName={null}
                                    links={backwardLinks[link].length}
                                />
                            </AccordionTrigger>
                        </div>
                        {onRemoveTable &&
                            !activeKeys.includes("bklinks-" + i) && (
                                <Button
                                    id={"remove-table-bk-" + i}
                                    aria-label={`Remove table ${link}`}
                                    variant="light-outline"
                                    size="sm"
                                    className="!border-0 shrink-0"
                                    onClick={() => onRemoveTable(link)}>
                                    <DeleteIcon />
                                </Button>
                            )}
                    </div>
                    <AccordionContent className="bg-surface-grey p-2">
                        <BackLinks backlinks={backwardLinks[link]} />
                    </AccordionContent>
                </AccordionItem>
            ))}
        </Accordion>
    );
};

const getUpdatedGroups = (
    defaultGroups: ColumnGroup[],
    allTablesObj: GraphTables,
    gfcTables: GraphTables,
    id: number
): [GraphLinkGroup[], boolean] => {
    let invalidGroup: boolean = false;
    const tmpSelected: GraphLinkGroup[] = [];
    if (defaultGroups.length < 1) {
        return [tmpSelected, invalidGroup];
    }
    if (defaultGroups[id]?.columns.length < 0) {
        return [tmpSelected, invalidGroup];
    }

    const colGroups: ColumnKey[] = defaultGroups[id]?.columns ?? [];
    for (let i = 0; i < colGroups?.length; i++) {
        // group table name
        const gtName = colGroups[i]?.table_name;
        //allTable name and temp=table itself
        const temp = allTablesObj[gtName];
        if (temp === undefined) {
            continue; // Skip below logic and goto next index
        }
        const selectedTable: DropdownOption = {
            value: gtName,
            label: gtName
        };

        const tables: GraphTables = gfcTables;
        const tableFound = Boolean(tables && gtName in tables);

        if (tableFound === false) {
            invalidGroup = true;
        }

        const clm: ColumnArgs = temp?.cols?.find(
            (c) => c?.name === colGroups[i]?.col_name
        ) as ColumnArgs;
        let selectedColumn: DropdownOption = {
            label: "",
            value: ""
        };
        if (clm !== undefined) {
            selectedColumn = {
                value: clm?.name,
                label: clm?.name + (temp?.pkey === clm?.name ? " (PK)" : "")
            };
        }

        let options: GraphLinkColumnOption[] = [];
        if (temp?.cols !== undefined) {
            options = temp.cols.map((column) => ({
                value: column.name,
                label: column.name + (temp.pkey === column.name ? " (PK)" : ""),
                sortIndex:
                    temp.pkey === column.name
                        ? 0
                        : column.stype === "ID"
                          ? 10
                          : 20
            })) as GraphLinkColumnOption[];
            options.sort((a, b) => a.sortIndex - b.sortIndex);
        }

        const selectedValue: GraphLinkGroup = {
            columnOptions: options,
            selectedTable: selectedTable,
            selectedColumn: selectedColumn,
            invalid: !tableFound
        };
        tmpSelected[i] = selectedValue;
    }
    tmpSelected.push(createEmptyLinkGroup());
    return [tmpSelected, invalidGroup];
};

const LinkGroup = ({
    id,
    allTables,
    graphFlowInfo,
    groups,
    setGroups,
    activeKeys,
    handleErrorParent,
    onRemoveTable
}: {
    id: number;
    allTables: GraphTables;
    graphFlowInfo: GraphFlowInfo;
    groups: Array<ColumnGroup>;
    setGroups: Dispatch<SetStateAction<Array<ColumnGroup>>>;
    activeKeys: string[];
    handleErrorParent: HandleErrorFunc;
    onRemoveTable?: (tableName: string) => void;
}) => {
    const [isLoading, setLoading] = useState<boolean>(false);
    const [invalid, setInvalid] = useState<boolean>(false);
    const [selected, setSelected] = useState<Array<GraphLinkGroup>>(
        getUpdatedGroups(groups, allTables, graphFlowInfo?.tables, id)[0]
    );

    const [, , , clearErrors] = useValidationState(setLoading);

    const tables: Array<DropdownOption> = getKeys(graphFlowInfo?.tables)?.map(
        (tableName) => ({
            value: tableName,
            label: tableName
        })
    );

    const pkeyEntry: Array<GraphLinkGroup> = selected.filter((e) =>
        e?.selectedColumn?.label?.endsWith("(PK)")
    );

    const updateLinkGroups = useCallback(
        (
            graphFlowInfo: GraphFlowInfo,
            groups: Array<ColumnGroup>,
            allTables: GraphTables,
            id: number
        ) => {
            if (isObjectEmpty(graphFlowInfo)) return;

            if (groups[id]) {
                groups[id].updated = false;
            }
            setGroups(groups);

            if (groups.length > 0) {
                if (groups[id]?.columns.length > 0) {
                    const [tmpSelected, invalidGroup] = getUpdatedGroups(
                        groups,
                        allTables,
                        graphFlowInfo?.tables,
                        id
                    );

                    setTimeout(() => {
                        setSelected(tmpSelected);
                        setInvalid(invalidGroup);
                    }, 1);
                }
            }
            setLoading(false);
        },
        [setGroups]
    );

    useEffect(() => {
        const graphConfig: GraphFlowInfo = { ...graphFlowInfo };
        const grps: Array<ColumnGroup> = [...groups];
        const alltbls: GraphTables = { ...allTables };
        if (groups?.length > 0 && groups[id]?.updated === true) {
            updateLinkGroups(graphConfig, grps, alltbls, id);
        }
    }, [allTables, graphFlowInfo, groups, id, updateLinkGroups]);

    const getGroupConfig = useCallback((tmpSelected: Array<GraphLinkGroup>) => {
        const tmpArray: Array<ColumnKey> = [];
        for (let i = 0; i < tmpSelected.length; i++) {
            const table: string = tmpSelected[i]?.selectedTable
                ?.value as string;
            const column: string = tmpSelected[i]?.selectedColumn
                ?.value as string;
            if (isValid(table) && isValid(column)) {
                tmpArray.push({ table_name: table, col_name: column });
            }
        }
        return tmpArray;
    }, []);

    /** Method to handle Table change/selection event*/
    const onTableChange = useCallback(
        (selectedOption: DropdownOption, index: number) => {
            const selectedTable: string = selectedOption?.value as string;
            const grps: Array<ColumnGroup> = [...groups];

            const tmp = graphFlowInfo?.tables?.[selectedTable];

            let options: Array<GraphLinkColumnOption> = [];
            if (tmp?.cols !== undefined) {
                options = tmp.cols.map((column) => ({
                    value: column.name,
                    label:
                        column.name + (tmp.pkey === column.name ? " (PK)" : ""),
                    sortIndex:
                        tmp.pkey === column.name
                            ? 0
                            : column.stype === "ID"
                              ? 10
                              : 20
                })) as GraphLinkColumnOption[];
            }

            options.sort((a, b) => a?.sortIndex - b?.sortIndex);
            const selectedValue: GraphLinkGroup = {
                columnOptions: options,
                selectedTable: selectedOption,
                selectedColumn: {
                    label: "",
                    value: ""
                },
                invalid: false
            };

            const tmpSelected: Array<GraphLinkGroup> = [...selected];
            tmpSelected[index] = selectedValue;
            if (grps[id]) grps[id].columns = getGroupConfig(tmpSelected);
            setGroups(grps);
            setSelected(tmpSelected);
            if (invalid) {
                clearErrors();
                setInvalid(false);
            }
        },
        [
            clearErrors,
            getGroupConfig,
            graphFlowInfo.tables,
            groups,
            id,
            invalid,
            selected,
            setGroups
        ]
    );

    const triggerError = useCallback(
        (errorInfo: { title: string; message: string }) => {
            setLoading(false);
            setInvalid(true);
            const err: CustomError = {
                response: {
                    status: errorStatusCode.UnProcessableEntityError,
                    data: errorInfo.message
                },
                name: "",
                message: errorInfo.message
            };
            handleErrorParent(err, errorInfo.title);
        },
        [handleErrorParent]
    );

    // Transforms the selected array into group and calls setGroups
    const setGroup = useCallback(
        (selectedArray: GraphLinkGroup[]) => {
            const tmpArray: Array<ColumnKey> = [];
            let errorInfo: { title: string; message: string } = {
                title: "",
                message: ""
            };
            const grps: Array<ColumnGroup> = [...groups];
            //Save group
            for (let i = 0; i < selectedArray.length; i++) {
                const table: string = selectedArray[i]?.selectedTable
                    ?.value as string;
                const column: string = selectedArray[i]?.selectedColumn
                    ?.value as string;
                if (isValid(table) && isValid(column)) {
                    tmpArray.push({ table_name: table, col_name: column });
                } else if (isValid(table) && !isValid(column)) {
                    errorInfo = {
                        "title": "Column group creation failed!",
                        "message": "Column should not be empty"
                    };
                    triggerError(errorInfo);
                    return;
                }
            }

            grps[id] = {
                columns: tmpArray,
                state: "saved",
                updated: true
            };
            setGroups(grps);
        },
        [groups, id, setGroups, triggerError]
    );

    /** Method to handle Column change/selection event*/
    const onColumnChange = useCallback(
        (selectedOption: DropdownOption, index: number) => {
            let errorInfo: { title: string; message: string } = {
                title: "",
                message: ""
            };

            if (selectedOption?.label?.endsWith("(PK)")) {
                errorInfo = {
                    "title": "Column group creation failed!",
                    "message":
                        "No more than one of the selected columns can be a Primary Key. If you want to connect two Primary Keys to each other, you likely want to connect two tables that both refer to the same type of data (eg Users and UserLocations, or Transactions and Payments). In order to represent this relationship in Kumo, you must either combine the tables or remove the Primary Key from one of the tables to make it a Foreign Key."
                };
                triggerError(errorInfo);
                return;
            }
            const tmpSelected: Array<GraphLinkGroup> = [...selected];
            tmpSelected[index].selectedColumn = selectedOption;

            if (tmpSelected.length - 1 === index) {
                tmpSelected.push(createEmptyLinkGroup()); //Adding empty data to show new Table & Column selection section
            }

            setGroup(tmpSelected);
            setSelected(tmpSelected);
            if (invalid) {
                clearErrors();
                setInvalid(false);
            }
        },
        [clearErrors, invalid, selected, setGroup, triggerError]
    );

    /**
     * This method will removes values from *selected* state variable
     * from the given index
     * @param index
     */
    const onDeleteClick = useCallback(
        (index: number, totalCount: number) => {
            const tmpSelected: Array<GraphLinkGroup> = [...selected];
            if (index !== totalCount - 1) {
                tmpSelected?.splice(index, 1);
                const invalidGroup: GraphLinkGroup = tmpSelected?.find(
                    (group: { invalid: boolean }) => group?.invalid === true
                ) as GraphLinkGroup;
                if (invalidGroup === undefined) {
                    setSelected(tmpSelected);
                    clearErrors();
                    setInvalid(false);
                } else {
                    setSelected(tmpSelected);
                }
            } else {
                tmpSelected[index] = createEmptyLinkGroup();
                setSelected(tmpSelected);
                if (invalid) {
                    clearErrors();
                    setInvalid(false);
                }
            }
            setGroup(tmpSelected);
        },
        [clearErrors, invalid, selected, setGroup]
    );
    //Find foward links for group[id]
    const forwardLinks = useMemo(() => {
        let fLinks: number = 0; //Forward links
        const tables = graphFlowInfo?.tables;
        if (tables) {
            groups[id]?.columns?.forEach((c) => {
                if (
                    c?.table_name &&
                    tables[c?.table_name]?.pkey === c?.col_name
                ) {
                    //This is a pkey. So forward links = number of cols
                    // or group length - 1
                    fLinks = groups[id]?.columns?.length - 1;
                }
            });
        }
        return fLinks;
    }, [graphFlowInfo?.tables, groups, id]);

    return (
        <>
            {isLoading && <Skeleton className="w-150 h-10" />}
            <ColumnLinks
                id={id}
                selected={selected}
                tables={tables}
                pkeyEntry={pkeyEntry}
                invalid={invalid}
                onColumnChange={onColumnChange}
                onTableChange={onTableChange}
                onDeleteClick={onDeleteClick}
                fLinks={forwardLinks}
                activeKeys={activeKeys}
                onRemoveTable={onRemoveTable}
            />
        </>
    );
};

export default WithErrorBoundary(GraphLinkGroups);

/**
 * Column Links component
 *
 * @param id
 * @param pkeyEntry
 * @param selected
 * @param tables
 * @param invalid,
 * @param onTableChange,
 * @param onColumnChange,
 * @param onDeleteClick
 * @returns ``` ```
 */
const ColumnLinks = ({
    id,
    pkeyEntry,
    selected,
    tables,
    invalid,
    onColumnChange,
    onTableChange,
    onDeleteClick,
    fLinks,
    activeKeys,
    onRemoveTable
}: {
    id: number;
    pkeyEntry: Array<GraphLinkGroup>;
    selected: Array<GraphLinkGroup>;
    tables: Array<DropdownOption>;
    invalid: boolean;
    onColumnChange: (selectedOption: DropdownOption, index: number) => void;
    onTableChange: (selectedOption: DropdownOption, index: number) => void;
    onDeleteClick: (index: number, totalCount: number) => void;
    fLinks: number;
    activeKeys: string[];
    onRemoveTable?: (tableName: string) => void;
}) => {
    const tableName = pkeyEntry[0]?.selectedTable?.label || "";

    return (
        <AccordionItem value={"fwdlinks-" + id} className={"p-0 m-0"}>
            <div className="flex items-center gap-2">
                <div className="flex-1 min-w-0">
                    <AccordionTrigger
                        data-testid={"fwdlinks-" + id}
                        showArrow={false}
                        className={
                            activeKeys.includes("fwdlinks-" + id)
                                ? "bg-surface-grey"
                                : ""
                        }>
                        <GroupHeader
                            id={id}
                            tableName={tableName}
                            pkeyName={
                                pkeyEntry[0]?.selectedColumn?.label?.slice(
                                    0,
                                    -4
                                ) || null
                            }
                            links={fLinks}
                        />
                    </AccordionTrigger>
                </div>
                {onRemoveTable &&
                    tableName &&
                    !activeKeys.includes("fwdlinks-" + id) && (
                        <Button
                            id={"remove-table-" + id}
                            aria-label={`Remove table ${tableName}`}
                            variant="light-outline"
                            size="sm"
                            className="!border-0 shrink-0"
                            onClick={() => onRemoveTable(tableName)}>
                            <DeleteIcon />
                        </Button>
                    )}
            </div>

            <AccordionContent className="bg-surface-grey p-4">
                {selected?.map((element: GraphLinkGroup, index: number) => {
                    if (element?.selectedColumn?.label?.endsWith("(PK)")) {
                        return null;
                    }
                    return (
                        <div key={`table-column-${index}`}>
                            {!element?.selectedTable?.value ? (
                                <Label className="text-title-secondary !px-1">
                                    {tables?.length > 1
                                        ? "Add more links"
                                        : "Add another table to create a link"}
                                </Label>
                            ) : !element?.selectedColumn?.value ? (
                                <Label className="text-title-secondary !px-1">
                                    Select a reference column which links the
                                    two tables
                                </Label>
                            ) : null}
                            <GridRow xs={12} sm={12} md={12}>
                                <GridCol xs={6}>
                                    <TableSelection
                                        id={id}
                                        index={index}
                                        options={tables}
                                        linkGroup={element}
                                        onTableChange={onTableChange}
                                    />
                                </GridCol>
                                <GridCol xs={5}>
                                    <ColumnSelection
                                        id={id}
                                        index={index}
                                        linkGroup={element}
                                        onColumnChange={onColumnChange}
                                        invalid={invalid}
                                    />
                                </GridCol>
                                <GridCol xs={1}>
                                    {element?.selectedTable?.value !== "" ? (
                                        <Button
                                            id={"delete-column"}
                                            aria-label="Delete column"
                                            onClick={() =>
                                                onDeleteClick(
                                                    index,
                                                    selected?.length
                                                )
                                            }
                                            variant="light-outline"
                                            size="sm"
                                            className="!border-0 shrink-0">
                                            <DeleteIcon />
                                        </Button>
                                    ) : (
                                        <Spacer width={36} />
                                    )}
                                </GridCol>
                            </GridRow>
                            <Spacer height={10} />
                        </div>
                    );
                })}
            </AccordionContent>
        </AccordionItem>
    );
};

const TableSelection = ({
    id,
    index,
    linkGroup,
    onTableChange,
    options
}: {
    id: number;
    index: number;
    linkGroup: GraphLinkGroup;
    onTableChange: (selectedOption: DropdownOption, index: number) => void;
    options: DropdownOption[];
}) => {
    const selectedTable = linkGroup?.selectedTable?.value
        ? linkGroup?.selectedTable
        : null;
    return (
        <Select
            id={"group-" + id + "-table-" + index}
            placeholder="Table"
            mode="single"
            search={true}
            options={options?.map((item) => {
                return {
                    value: item?.value as string,
                    label: item?.label as string
                };
            })}
            selectedValues={
                selectedTable ? Array.of(selectedTable?.value as string) : []
            }
            isInValid={linkGroup?.invalid}
            onValuesChange={(newValue) => {
                const currSel = options?.find((item) =>
                    newValue.includes(item?.value as string)
                );
                if (currSel?.value) {
                    onTableChange(currSel, index);
                }
            }}
        />
    );
};

const ColumnSelection = ({
    id,
    index,
    linkGroup,
    onColumnChange,
    invalid
}: {
    id: number;
    index: number;
    linkGroup: GraphLinkGroup;
    onColumnChange: (selectedOption: DropdownOption, index: number) => void;
    invalid: boolean;
}) => {
    const selectedColumn = linkGroup?.selectedColumn?.value
        ? linkGroup?.selectedColumn
        : null;
    return (
        <Select
            id={"group-" + id + "-column-" + index}
            placeholder="Column"
            mode="single"
            search={true}
            options={linkGroup?.columnOptions?.map((item) => {
                return {
                    value: item?.value as string,
                    label: item?.label as string
                };
            })}
            selectedValues={
                selectedColumn ? Array.of(selectedColumn?.value as string) : []
            }
            isInValid={
                invalid &&
                isValid(linkGroup?.selectedTable?.value) &&
                !isValid(selectedColumn?.value)
            }
            onValuesChange={(newValue) => {
                const currSel = linkGroup?.columnOptions?.find((item) =>
                    newValue.includes(item?.value as string)
                );
                if (currSel?.value) {
                    onColumnChange(currSel, index);
                }
            }}
        />
    );
};

const GroupHeader = ({
    id,
    tableName,
    pkeyName,
    links
}: {
    id: number;
    tableName: string;
    pkeyName: string | null;
    links: number;
}) => {
    const textSize = 30;
    const { width } = useWindowDimensions();
    const otherSpace = width > 1160 && width < 1330 ? 400 : 300;
    const availWidth = (width - otherSpace) / 2;
    const tableWidth =
        tableName?.length * textSize < availWidth
            ? tableName?.length * textSize
            : availWidth;
    const pkeyWidth =
        (pkeyName || "")?.length * textSize < availWidth
            ? (pkeyName || "")?.length * textSize
            : availWidth;
    return (
        <GridRow
            id={"grp-hdr" + id}
            className="flex justify-between items-center">
            <GridCol className="ms-2 flex items-center">
                <TruncatedLabel
                    value={tableName}
                    className="font-medium text-base break-all"
                    availWidth={tableWidth}
                />
                {pkeyName && (
                    <>
                        {" "}
                        <Key className="fill-pink-600 size-5.5 ms-3 me-1" />
                        <TruncatedLabel
                            value={pkeyName}
                            className="text-base break-all"
                            availWidth={pkeyWidth}
                        />
                    </>
                )}
            </GridCol>
            <GridCol className="me-2 flex items-center">
                <Cable
                    className={cn(
                        "size-4.5",
                        pkeyName ? "fill-sky-600" : "fill-icon-secondary"
                    )}
                />
                {` ${links} ${pkeyName ? " Link" : " Backlink"}${links > 1 ? "s" : ""}`}
            </GridCol>
        </GridRow>
    );
};

const BackLinks = ({
    backlinks
}: {
    backlinks: {
        tableName: string;
        columnName: string;
        pkeyName: string;
    }[];
}) => {
    return (
        <div className="p-2 ">
            {backlinks?.map((backlink, i) => (
                <div
                    className="grid grid-cols-2 gap-4 p-1 "
                    key={`backlinks-div-${i}`}>
                    <Select
                        id={"backlinks-table-" + i}
                        options={Array.of({
                            value: backlink?.tableName,
                            label: backlink?.tableName
                        })}
                        placeholder="Column"
                        mode="single"
                        selectedValues={
                            backlink?.tableName
                                ? Array.of(backlink?.tableName)
                                : undefined
                        }
                        onValuesChange={() => {}}
                        disabled={true}
                    />

                    <Select
                        id={"backlinks-columns-" + i}
                        options={Array.of({
                            value: backlink?.pkeyName,
                            label: backlink?.pkeyName
                        })}
                        placeholder="Column"
                        mode="single"
                        selectedValues={
                            backlink?.pkeyName
                                ? Array.of(backlink?.pkeyName)
                                : undefined
                        }
                        onValuesChange={() => {}}
                        disabled={true}
                    />
                </div>
            ))}
        </div>
    );
};

const TruncatedLabel = ({
    value,
    className = "",
    availWidth
}: {
    value: string;
    className?: string;
    availWidth: number;
}) => {
    const textSize = 30;

    if (value?.length * textSize > availWidth)
        return (
            <div className="w-fit">
                <Tooltip>
                    <TooltipTrigger>
                        <span className={className}>
                            {value?.substring(
                                0,
                                Math.round(availWidth / textSize)
                            ) + "..."}
                        </span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        side={"right"}
                        sideOffset={5}
                        className="justify-around">
                        {value}
                    </TooltipContent>
                </Tooltip>
            </div>
        );

    return <span className={className}>{value}</span>;
};
