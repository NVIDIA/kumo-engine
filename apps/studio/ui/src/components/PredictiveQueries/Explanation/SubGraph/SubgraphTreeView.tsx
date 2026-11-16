// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Alert,
    AlertDescription,
    Card,
    CardContent,
    CardHeader,
    cn,
    Select,
    Sheet,
    SheetContent,
    SheetTitle,
    Skeleton,
    Switch,
    WarningIcon,
    XaiSubgraph
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useSplitPane } from "hooks/jobs/training/subgraph/useSplitPane";
import {
    type SubgraphTreeViewProps,
    useSubgraphTreeView
} from "hooks/jobs/training/subgraph/useSubgraphTreeView";

import { SIZE_OPTIONS, TIER_LABELS } from "utils/subgraphTreeViewUtils";

import { ColumnContributions } from "./ColumnContributions";

export type { SubgraphTreeViewProps };

function SubgraphTreeViewDef(props: SubgraphTreeViewProps) {
    const { containerRef, leftCardRef, isDragging, splitPercent } =
        useSplitPane();
    const {
        displayNodes,
        displayEdges,
        legendItems,
        isLegendOpen,
        setIsLegendOpen,
        isFullscreen,
        setIsFullscreen,
        isFullExpand,
        setIsFullExpand,
        isUserCollapsingRef,
        setExpandedChipParents,
        allChipBoundaryParents,
        selectedNodeData,
        tableColorMap,
        selectedTables,
        selectedSizes,
        tableOptions,
        selectedTableCount,
        selectedSizeCount,
        onTableValuesChange,
        onSizeValuesChange,
        onNodeMouseEnter,
        onNodeMouseLeave,
        onNodeClick,
        onPaneClick,
        visibleCount,
        totalNodes,
        isAnyFilterActive,
        showInfoBanner,
        showWarningBanner,
        showWarningChip,
        setIsWarningDismissed,
        warningLine1,
        pathOnlyCount,
        predictions
    } = useSubgraphTreeView(props);

    return (
        <>
            <div
                ref={containerRef}
                className="flex min-h-120 flex-row items-stretch">
                <Card
                    ref={leftCardRef}
                    className="p-6 gap-6 shadow-none rounded-sm flex min-w-0 flex-col overflow-hidden"
                    style={
                        props.isLoading
                            ? { flex: 1 }
                            : { width: `calc(${splitPercent}% - 8px)` }
                    }>
                    <CardHeader className="flex flex-row items-center justify-between p-0">
                        <div className="flex flex-col gap-2">
                            <span className="text-base font-bold leading-4 text-title-primary">
                                Subgraph
                            </span>
                            {!props.isLoading && (
                                <div className="flex items-center gap-1.5 text-sm font-medium leading-4 text-title-secondary">
                                    {isAnyFilterActive ? (
                                        <>
                                            <span>{visibleCount} visible</span>
                                            <span>·</span>
                                            <span>{totalNodes} total</span>
                                        </>
                                    ) : (
                                        <span>{totalNodes} total</span>
                                    )}
                                    {showWarningChip && (
                                        <button
                                            onClick={() =>
                                                setIsWarningDismissed(false)
                                            }
                                            className="ml-1"
                                            aria-label="View warning">
                                            <div className="inline-flex size-5 flex-col items-center justify-center gap-2.5 rounded-sm bg-amber-50 px-px pt-px pb-0.5 outline outline-1 outline-offset-[-1px] outline-amber-300 hover:bg-amber-100">
                                                <WarningIcon className="h-3 w-3.5 fill-amber-500" />
                                            </div>
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                        {!props.isLoading && (
                            <div className="flex items-center gap-2">
                                <label className="flex items-center gap-2 cursor-pointer">
                                    <span className="text-sm font-medium leading-5 text-title-secondary">
                                        Full
                                    </span>
                                    <Switch
                                        checked={isFullExpand}
                                        onCheckedChange={(checked) => {
                                            isUserCollapsingRef.current =
                                                !checked;
                                            setIsFullExpand(checked);
                                            setExpandedChipParents(
                                                checked
                                                    ? new Set(
                                                          allChipBoundaryParents
                                                      )
                                                    : new Set()
                                            );
                                        }}
                                    />
                                </label>
                                <Select
                                    options={tableOptions}
                                    selectedValues={selectedTables}
                                    onValuesChange={onTableValuesChange}
                                    placeholder="Table"
                                    collisionBoundary={leftCardRef.current}
                                    popoverTriggerClassName="w-32"
                                    multiModeText={
                                        <span className="flex items-center gap-2">
                                            <span className="text-sm font-medium leading-5 text-title-secondary">
                                                Table
                                            </span>
                                            {selectedTableCount !==
                                                undefined && (
                                                <span className="inline-flex size-6 items-center justify-center rounded bg-(--color-surface-grey) text-sm font-medium leading-5 text-title-secondary">
                                                    {selectedTableCount}
                                                </span>
                                            )}
                                        </span>
                                    }
                                    showSelectedCount={false}
                                    mode="multi"
                                    showCheckbox
                                    size="sm"
                                    variant="secondary"
                                />
                                <Select
                                    options={SIZE_OPTIONS}
                                    selectedValues={selectedSizes}
                                    onValuesChange={onSizeValuesChange}
                                    placeholder="Size"
                                    popoverClassName="w-35"
                                    popoverTriggerClassName="w-32"
                                    search={false}
                                    sortOptions={false}
                                    multiModeText={
                                        <span className="flex items-center gap-2">
                                            <span className="text-sm font-medium leading-5 text-title-secondary">
                                                Size
                                            </span>
                                            {selectedSizeCount !==
                                                undefined && (
                                                <span className="inline-flex size-6 items-center justify-center rounded bg-(--color-surface-grey) text-sm font-medium leading-5 text-title-secondary">
                                                    {selectedSizeCount}
                                                </span>
                                            )}
                                        </span>
                                    }
                                    showSelectedCount={false}
                                    mode="multi"
                                    showCheckbox
                                    size="sm"
                                    variant="secondary"
                                />
                            </div>
                        )}
                    </CardHeader>

                    <CardContent className="flex shrink-0 flex-col gap-2 p-0">
                        {showInfoBanner && (
                            <Alert variant="info" enableClose={false}>
                                <div className="flex flex-col gap-2">
                                    <p className="text-sm font-semibold leading-5">
                                        {pathOnlyCount}{" "}
                                        {pathOnlyCount !== 1
                                            ? "nodes are"
                                            : "node is"}{" "}
                                        dimmed for path context
                                    </p>
                                    <p className="text-sm font-medium leading-5">
                                        They don&apos;t match your filter but
                                        lie between the entity and a deeper
                                        match.
                                    </p>
                                </div>
                            </Alert>
                        )}

                        {showWarningBanner && (
                            <Alert
                                variant="warn"
                                className="[&_svg]:size-4"
                                onClose={() => setIsWarningDismissed(true)}>
                                <AlertDescription>
                                    <div className="flex flex-col gap-4">
                                        <p className="font-medium">
                                            {warningLine1}
                                        </p>
                                        {pathOnlyCount > 0 && (
                                            <p className="font-medium">
                                                {pathOnlyCount} other
                                                {pathOnlyCount !== 1
                                                    ? "s"
                                                    : ""}{" "}
                                                dimmed to show the path to
                                                deeper matches.
                                            </p>
                                        )}
                                    </div>
                                </AlertDescription>
                            </Alert>
                        )}

                        <Card
                            className={cn(
                                "p-0 gap-0 shadow-none rounded-sm h-113 overflow-hidden",
                                isFullscreen && "hidden"
                            )}>
                            {props.isLoading ? (
                                <Skeleton className="h-full w-full" />
                            ) : (
                                <XaiSubgraph
                                    key={props.entityId}
                                    displayNodes={displayNodes}
                                    displayEdges={displayEdges}
                                    isLegendOpen={isLegendOpen}
                                    legendItems={legendItems}
                                    tierLabels={TIER_LABELS}
                                    onNodeMouseEnter={onNodeMouseEnter}
                                    onNodeMouseLeave={onNodeMouseLeave}
                                    onNodeClick={onNodeClick}
                                    onPaneClick={onPaneClick}
                                    onLegendOpenChange={setIsLegendOpen}
                                    onOpenFullscreen={() =>
                                        setIsFullscreen(true)
                                    }
                                    predictions={predictions}
                                />
                            )}
                        </Card>
                    </CardContent>
                </Card>

                {!props.isLoading && (
                    <>
                        <div
                            className="flex w-4 shrink-0 cursor-col-resize items-center justify-center"
                            onMouseDown={(e) => {
                                e.preventDefault();
                                isDragging.current = true;
                            }}>
                            <div className="h-6 w-1 rounded-full pointer-events-none bg-(--color-title-secondary)" />
                        </div>

                        <Card className="p-6 gap-6 shadow-none rounded-sm flex min-w-0 flex-1 flex-col">
                            <CardHeader className="p-0">
                                <span className="text-base font-bold leading-4 text-title-primary">
                                    Column Contributions
                                </span>
                            </CardHeader>
                            <ColumnContributions
                                selectedNode={selectedNodeData}
                                tableColor={
                                    selectedNodeData
                                        ? (tableColorMap[
                                              selectedNodeData.tableName
                                          ] ?? "var(--color-kumo-grey-400)")
                                        : "var(--color-kumo-grey-400)"
                                }
                            />
                        </Card>
                    </>
                )}
            </div>

            <Sheet open={isFullscreen} onOpenChange={setIsFullscreen}>
                <SheetContent
                    side="top"
                    aria-describedby={undefined}
                    className="h-[98vh] w-[99%] rounded p-0 [&>button]:hidden">
                    <SheetTitle className="sr-only">
                        Subgraph fullscreen view
                    </SheetTitle>
                    <XaiSubgraph
                        key={`${props.entityId}-fs`}
                        displayNodes={displayNodes}
                        displayEdges={displayEdges}
                        isLegendOpen={isLegendOpen}
                        legendItems={legendItems}
                        tierLabels={TIER_LABELS}
                        onNodeMouseEnter={onNodeMouseEnter}
                        onNodeMouseLeave={onNodeMouseLeave}
                        onNodeClick={onNodeClick}
                        onPaneClick={onPaneClick}
                        onLegendOpenChange={setIsLegendOpen}
                        onOpenFullscreen={() => {}}
                        onCloseFullscreen={() => setIsFullscreen(false)}
                        predictions={predictions}
                    />
                </SheetContent>
            </Sheet>
        </>
    );
}

export const SubgraphTreeView = WithErrorBoundary(SubgraphTreeViewDef);
