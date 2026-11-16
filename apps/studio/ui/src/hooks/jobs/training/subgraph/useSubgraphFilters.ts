// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { KumoapiExplainSubgraphSubgraph } from "gen/v0/types";

interface UseSubgraphFiltersParams {
    allTableNames: string[];
    subgraph: KumoapiExplainSubgraphSubgraph | undefined;
}

export function useSubgraphFilters({
    allTableNames,
    subgraph
}: UseSubgraphFiltersParams) {
    const [selectedTables, setSelectedTables] = useState<string[]>([]);
    const [selectedSizes, setSelectedSizes] = useState<string[]>([]);
    const [isWarningDismissed, setIsWarningDismissed] = useState(false);
    const [expandedChipParents, setExpandedChipParents] = useState<Set<string>>(
        new Set()
    );
    const [isFullExpand, setIsFullExpand] = useState(false);
    const isUserCollapsingRef = useRef(false);

    useEffect(() => {
        setSelectedTables(allTableNames);
    }, [allTableNames]);

    useEffect(() => {
        isUserCollapsingRef.current = false;
        setIsFullExpand(false);
        setExpandedChipParents(new Set());
    }, [subgraph]);

    const isTableFilterActive =
        selectedTables.length > 0 &&
        selectedTables.length < allTableNames.length;
    const isSizeFilterActive = selectedSizes.length > 0;
    const isAnyFilterActive = isTableFilterActive || isSizeFilterActive;

    const onTableValuesChange = useCallback((vals: string[]) => {
        setSelectedTables(vals);
        setIsWarningDismissed(false);
    }, []);

    const onSizeValuesChange = useCallback((vals: string[]) => {
        if (vals.includes("__clear__")) {
            setSelectedSizes([]);
        } else {
            setSelectedSizes(vals);
        }
        setIsWarningDismissed(false);
    }, []);

    const tableOptions = useMemo(
        () => allTableNames.map((t) => ({ value: t, label: t })),
        [allTableNames]
    );

    const selectedTableCount =
        selectedTables.length > 0 &&
        selectedTables.length < allTableNames.length
            ? selectedTables.length
            : undefined;

    const selectedSizeCount =
        selectedSizes.length > 0 ? selectedSizes.length : undefined;

    return {
        selectedTables,
        selectedSizes,
        expandedChipParents,
        setExpandedChipParents,
        isWarningDismissed,
        setIsWarningDismissed,
        isTableFilterActive,
        isSizeFilterActive,
        isAnyFilterActive,
        isFullExpand,
        setIsFullExpand,
        isUserCollapsingRef,
        onTableValuesChange,
        onSizeValuesChange,
        tableOptions,
        selectedTableCount,
        selectedSizeCount
    };
}
