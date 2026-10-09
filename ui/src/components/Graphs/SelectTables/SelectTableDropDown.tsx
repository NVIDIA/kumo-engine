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

import { GridCol, Select } from "@kumo-ai/kumo-ui-catalog";

import { useTableSelectionForGraphs } from "hooks/graph/newGraph";

import { OptionType } from "types/common";
import { GraphFlowInfo, GraphTables } from "types/graphs";
import { HandleErrorFunc } from "types/validation";

import { ADD_KUMO_ENTITY_VALUE } from "utils/Constants";
import { getKeys } from "utils/Helpers/objectHelpers";

export interface SelectTableDropdownProps {
    tableNameList: string[];
    graphFlowInfo: GraphFlowInfo;
    setGraphFlowInfo: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    allTables: GraphTables;
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>;
    tableIdMap: { [key: string]: string };
    updateGroups: (config: GraphFlowInfo) => void;
    suggestLinks: () => void;
    resetErrLabel?: () => void;
    toggleTableModel: () => void;
    newTable: string | null;
    setNewTable: (d: string | null) => void;
    handleError: HandleErrorFunc;
    setLoading: (value: boolean) => void;
    projectId?: string | null;
}

export const SelectTableDropdown = ({
    tableNameList,
    graphFlowInfo,
    setGraphFlowInfo,
    allTables,
    setAllTables,
    tableIdMap,
    updateGroups,
    suggestLinks,
    resetErrLabel,
    toggleTableModel,
    newTable,
    setNewTable,
    handleError,
    setLoading,
    projectId
}: SelectTableDropdownProps) => {
    const [prevSel, setPrevSel] = useState<string[]>([]);
    const [selectedTable, setSelectedTable] = useState<OptionType[]>(
        [] as OptionType[]
    );
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const spacerRef = useRef<HTMLDivElement>(null);

    const { onSelectTables } = useTableSelectionForGraphs(
        graphFlowInfo,
        allTables,
        setAllTables,
        setGraphFlowInfo,
        updateGroups,
        handleError,
        tableIdMap,
        suggestLinks,
        setLoading,
        projectId ?? undefined
    );

    useEffect(() => {
        //Updated selectedTable if gfc already has tables ( Edit mode )
        const names: Array<string> = getKeys(graphFlowInfo?.tables);
        if (names && names?.length > 0) {
            const selected = names?.map((n) => {
                return { "value": n, "label": n } as OptionType;
            });
            if (selected) {
                setSelectedTable(selected);
                const sel: string[] = selected?.map((t) => t?.value);
                setPrevSel(sel);
            }
        } else {
            setSelectedTable([]);
            setPrevSel([]);
        }
    }, [graphFlowInfo]);

    const onSelectTable = useCallback(
        (tbl: OptionType[]) => {
            //Get current dropdown selection in string array format.
            const currSel: string[] = tbl?.map((t) => t.value);

            //Compare it against the prev selection to get a list of added or
            // removed table names.
            const added: string[] = currSel.filter(
                (item) => !prevSel.includes(item)
            );
            const removed: string[] = prevSel.filter(
                (item) => !currSel.includes(item)
            );
            //Since we are selecting a table, reset error-label if any
            resetErrLabel?.();

            onSelectTables(added, removed);
            setPrevSel(currSel); //Update curr selected table list.
        },
        [onSelectTables, prevSel, resetErrLabel]
    );

    const tableOptionList = useMemo(() => {
        const tmpList: Array<OptionType> = [];
        tmpList.push({ value: ADD_KUMO_ENTITY_VALUE, label: "Add new table" });
        tableNameList?.forEach((e) => {
            tmpList.push({ value: e, label: e });
        });
        return tmpList;
    }, [tableNameList]);

    // Scroll container to bottom when dropdown opens (after spacer renders)
    useEffect(() => {
        if (isDropdownOpen) {
            spacerRef.current?.scrollIntoView({
                block: "end",
                behavior: "smooth"
            });
        }
    }, [isDropdownOpen]);

    // Clip the Radix portal popover to the scroll container bounds on scroll
    useEffect(() => {
        if (!isDropdownOpen) return;
        const wrapper = wrapperRef.current;
        if (!wrapper) return;

        let scrollContainer: Element | null = wrapper.parentElement;
        while (scrollContainer) {
            const { overflow, overflowY } =
                window.getComputedStyle(scrollContainer);
            if (/scroll|auto/.test(overflow) || /scroll|auto/.test(overflowY))
                break;
            scrollContainer = scrollContainer.parentElement;
        }

        if (!scrollContainer) return;

        const trigger = wrapper.querySelector("button");
        const popoverId = trigger?.getAttribute("aria-controls");
        const getPopover = (): HTMLElement | null =>
            popoverId ? document.getElementById(popoverId) : null;

        const updateClip = () => {
            const popover = getPopover();
            if (!popover) return;

            const containerRect = scrollContainer!.getBoundingClientRect();
            const popoverRect = popover.getBoundingClientRect();

            const clipTop = Math.max(0, containerRect.top - popoverRect.top);
            const clipBottom = Math.max(
                0,
                popoverRect.bottom - containerRect.bottom
            );

            popover.style.clipPath =
                clipTop > 0 || clipBottom > 0
                    ? `inset(${clipTop}px 0px ${clipBottom}px 0px)`
                    : "";
        };

        scrollContainer.addEventListener("scroll", updateClip, {
            passive: true
        });
        const timer = setTimeout(updateClip, 0);

        return () => {
            clearTimeout(timer);
            scrollContainer!.removeEventListener("scroll", updateClip);
            const popover = getPopover();
            if (popover) popover.style.clipPath = "";
        };
    }, [isDropdownOpen]);

    useEffect(() => {
        if (newTable) {
            const find: OptionType | undefined = tableOptionList.find(
                (e) => e.value === newTable
            );
            if (find) {
                setNewTable(null);
                onSelectTable([...selectedTable, find]);
            }
        }
    }, [newTable, onSelectTable, tableOptionList, selectedTable, setNewTable]);

    return (
        <GridCol xs={12} sm={12} className="w-83.75">
            <div ref={wrapperRef}>
                <Select
                    placeholder="Select tables"
                    search={tableOptionList?.length ? true : false}
                    emptyText="No table found."
                    multiModeText="Tables"
                    selectedValues={
                        selectedTable?.length > 0
                            ? selectedTable?.map((table) => table?.value)
                            : []
                    }
                    options={tableOptionList}
                    onValuesChange={(newValue) => {
                        const currSel: OptionType[] = tableOptionList?.filter(
                            (item) => newValue.includes(item.value)
                        );
                        if (newValue.includes(ADD_KUMO_ENTITY_VALUE)) {
                            toggleTableModel();
                            return;
                        }
                        setSelectedTable(currSel);
                        onSelectTable(currSel);
                    }}
                    showCheckbox={true}
                    maxPopoverHeight={150}
                    popoverSide="bottom"
                    onOpenChange={setIsDropdownOpen}
                />
            </div>
            {isDropdownOpen && (
                <div ref={spacerRef}>
                    <Spacer height={300} />
                </div>
            )}
        </GridCol>
    );
};
