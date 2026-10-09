// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useMemo, useRef, useState } from "react";

import {
    Card,
    cn,
    CodeIcon,
    DeleteIcon,
    Duplicate,
    Edit,
    GridCol,
    GridRow,
    OverviewDropdown,
    OverviewProperties,
    PropertiesOverviewCard,
    Replay,
    SectionHeader,
    Skeleton,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { useCreateV1GraphsRefreshByGraphId } from "gen/v1/api/graphs/graphs";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import GraphInformationPanel from "components/Graphs/InformationPanel/GraphInformationPanel";
import { SDKCodeModal } from "components/SDKCodeModal/SDKCodeModal";
import Validation from "components/Validation/Validation";

import { useGetGraphInfo } from "hooks/graph/graphs";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { MenuItemType } from "types/common";
import { SDKObjectType } from "types/constants";
import { GraphFlowInfo, JSGraph } from "types/graphs";
import { JSTable } from "types/tables";

import { tablesFromGraphResponse } from "utils/Converters/graphsConverter";
import { defFlowInfo } from "utils/graph";
import { copyObject } from "utils/Helpers/objectHelpers";
import { checkDash } from "utils/Helpers/stringHelpers";
import { formatToNumericLocalDateString } from "utils/timeConverter";

const CreatedAtTooltip = ({
    displayValue,
    rawValue
}: {
    displayValue: string;
    rawValue?: string;
}) => (
    <Tooltip id="createdAt">
        <TooltipTrigger>
            <span>{checkDash(displayValue)}</span>
        </TooltipTrigger>
        <TooltipContent arrow={false} side="bottom" sideOffset={5}>
            {rawValue ?? ""}
        </TooltipContent>
    </Tooltip>
);

const createMenuItem = (
    icon: React.ReactNode,
    label: string,
    onSelect: () => void
): MenuItemType => ({
    content: (
        <>
            {icon}
            {` ${label} `}
        </>
    ),
    disabled: false,
    onSelect,
    tooltipText: ""
});

/**
 * Graph Overview component
 * This component contains Graph Details and GraphInformationPanel along with
 * Buttons and functionality to Create/Delete/Edit a Graph
 * Used to show graph details in the based graph list page (GraphsList.tsx)
 * and Related job section of jobs(only if job contains graphID)
 */

export interface GraphOverviewProps {
    graphId: string;
    onEditClick: (g: JSGraph) => void;
    onDeleteClick: (g: JSGraph) => void;
    onCopyClick: (g: JSGraph) => void;
    hideMenuOptions?: boolean;

    graphView?: boolean;
    setGraphView?: React.Dispatch<React.SetStateAction<boolean>>;
    isRelatedJob?: boolean;
}

const GraphOverview = ({
    graphId,
    onEditClick,
    onDeleteClick,
    onCopyClick,
    hideMenuOptions = true,
    graphView,
    setGraphView,
    isRelatedJob = false
}: GraphOverviewProps) => {
    const [loading, setLoading] = useState<boolean>(true);

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [tables, setTables] = useState<Array<JSTable>>([]);
    const { isLoading, selectedGraphInfo } = useGetGraphInfo(
        graphId ?? null,
        handleError,
        setLoading
    );

    // In the overview (list) page the canvas wrapper is a flex child that fills
    // whatever space remains in the fixed-height detail panel. Measure that
    // wrapper directly and size the graph canvas to it so the panel never
    // overflows (no inner scrollbar) at any screen resolution. The legacy
    // viewport-based estimate is used as the first-paint fallback and for the
    // related-job context, where the panel is not height-constrained.
    const canvasWrapperRef = useRef<HTMLDivElement>(null);
    const [measuredCanvasHeight, setMeasuredCanvasHeight] = useState<
        number | undefined
    >(undefined);

    useEffect(() => {
        const el = canvasWrapperRef.current;
        if (isRelatedJob || !el) return;

        const measure = () => setMeasuredCanvasHeight(el.clientHeight);
        measure();
        window.addEventListener("resize", measure);

        let observer: ResizeObserver | undefined;
        if (typeof ResizeObserver !== "undefined") {
            observer = new ResizeObserver(measure);
            observer.observe(el);
        }

        return () => {
            window.removeEventListener("resize", measure);
            observer?.disconnect();
        };
    }, [isRelatedJob, isLoading, selectedGraphInfo?.id]);

    const graphCanvasHeight =
        (!isRelatedJob && measuredCanvasHeight) ||
        window.innerHeight - (window.innerHeight < 850 ? 363 : 333);
    const [graphFlowInfo, setGraphFlowInfo] = useState<GraphFlowInfo>(
        copyObject(defFlowInfo)
    );
    const { mutate: refreshGraph } = useCreateV1GraphsRefreshByGraphId();
    const projectId = useProjectId();

    const [isModalOpen, setIsModalOpen] = useState(false);

    // Create tables from graphInfo
    useEffect(() => {
        setTables(tablesFromGraphResponse(selectedGraphInfo));
    }, [selectedGraphInfo]);

    const menuItems: MenuItemType[] = useMemo(() => {
        if (!selectedGraphInfo || hideMenuOptions) return [];

        return [
            createMenuItem(<Edit className="size-4" />, "Edit", () =>
                onEditClick(selectedGraphInfo)
            ),
            createMenuItem(<DeleteIcon className="size-4" />, "Delete", () =>
                onDeleteClick(selectedGraphInfo)
            ),
            createMenuItem(
                <Duplicate className="size-4 fill-icon-secondary" />,
                "Duplicate",
                () => onCopyClick(selectedGraphInfo)
            ),
            createMenuItem(<Replay className="size-4" />, "Re-ingest", () => {
                if (selectedGraphInfo?.id && projectId) {
                    refreshGraph({
                        graphId: selectedGraphInfo.id,
                        params: { project_id: projectId }
                    });
                }
            }),
            createMenuItem(<CodeIcon className="size-4" />, "SDK Code", () =>
                setIsModalOpen(true)
            )
        ];
    }, [
        selectedGraphInfo,
        hideMenuOptions,
        onEditClick,
        onDeleteClick,
        onCopyClick,
        projectId,
        refreshGraph
    ]);

    const { noOfTables, noOfLinks } = useMemo(() => {
        let noOfTables = 0;
        let noOfLinks = 0;
        try {
            if (selectedGraphInfo) {
                noOfTables = Number(selectedGraphInfo.totalTables) || 0;
                noOfLinks = Number(selectedGraphInfo.totalLinks) || 0;
            }
        } catch (e) {
            console.warn(e);
        }
        return { noOfTables, noOfLinks };
    }, [selectedGraphInfo]);

    const graphInfoPanel = useMemo(() => {
        if (!selectedGraphInfo) return null;

        return (
            <GraphInformationPanel
                graphInfo={selectedGraphInfo}
                tables={tables}
                weakGraphEdges={[]}
                gfc={graphFlowInfo}
                setGfc={setGraphFlowInfo}
                allTables={{}}
                setAllTables={() => {}}
                groups={[]}
                setGroups={() => {}}
                tableIds={selectedGraphInfo.table_name_to_fpid_mapping}
                suggestLinks={() => {}}
                customHeight={graphCanvasHeight}
                graphView={graphView}
                setGraphView={setGraphView}
            />
        );
    }, [
        graphFlowInfo,
        graphView,
        selectedGraphInfo,
        setGraphView,
        tables,
        graphCanvasHeight
    ]);

    const properties = useMemo(() => {
        if (!graphId) {
            return [];
        }
        const tmpList: OverviewProperties[] = [
            {
                name: "Created",
                value: (
                    <CreatedAtTooltip
                        displayValue={formatToNumericLocalDateString(
                            selectedGraphInfo?.timestamps?.created_at ?? ""
                        )}
                        rawValue={selectedGraphInfo?.dateCreated}
                    />
                )
            }
        ];

        if (selectedGraphInfo && !isLoading) {
            tmpList.push({
                name: "Configs",
                value:
                    noOfTables && noOfLinks ? (
                        `${noOfTables} table${noOfTables > 1 ? "s" : ""} x ${noOfLinks} link${noOfLinks > 1 ? "s" : ""}`
                    ) : (
                        <Symbol type="emdash" />
                    )
            });
        }
        return tmpList;
    }, [graphId, isLoading, noOfLinks, noOfTables, selectedGraphInfo]);

    const isAnyLoading = isLoading || loading;

    const panelClassName = cn("flex flex-col gap-4", !isRelatedJob && "h-full");

    if (isAnyLoading) {
        return (
            <div className={panelClassName}>
                <div>
                    <GridRow className="flex justify-between items-center">
                        <Skeleton className="w-60 h-7" />
                    </GridRow>
                    <Validation
                        pageLoading={false}
                        banner={banner}
                        alert={alert}
                        ctaMsgs={null}
                        dialogRef={dialogRef}
                        hideWarnings={() => {}}
                        clearErrors={clearErrors}
                    />
                </div>
                <Skeleton className="w-full h-7" />
            </div>
        );
    }

    return (
        <div className={panelClassName}>
            <div className="flex flex-col gap-4">
                <div>
                    <GridRow className="flex justify-between items-center">
                        <GridCol>
                            <SectionHeader
                                title={graphId}
                                titleSize="xl"
                                titleLetterSpacing={0.1}
                                className={cn(isRelatedJob && "px-6")}
                            />
                        </GridCol>
                        {hideMenuOptions ? null : (
                            <GridCol>
                                <OverviewDropdown
                                    id="graph-overview-dropdown"
                                    actionItems={menuItems}
                                />
                            </GridCol>
                        )}
                    </GridRow>
                    <Validation
                        pageLoading={false}
                        banner={banner}
                        alert={alert}
                        ctaMsgs={null}
                        dialogRef={dialogRef}
                        hideWarnings={() => {}}
                        clearErrors={clearErrors}
                    />
                </div>
                <PropertiesOverviewCard
                    properties={properties}
                    cardClassName="p-4"
                    hideTitle={!isRelatedJob}
                />
            </div>
            {selectedGraphInfo && (
                <div
                    ref={canvasWrapperRef}
                    className={cn(
                        isRelatedJob ? "px-6 pb-6" : "min-h-0 flex-1"
                    )}>
                    <Card className="border-none rounded-xl overflow-hidden h-full">
                        {graphInfoPanel}
                    </Card>
                </div>
            )}
            {isModalOpen && (
                <SDKCodeModal
                    id={selectedGraphInfo?.id ?? ""}
                    open={isModalOpen}
                    toggleModel={() => setIsModalOpen(!isModalOpen)}
                    objectTypeToGenScript={SDKObjectType.Graph}
                />
            )}
        </div>
    );
};

export default WithErrorBoundary(GraphOverview);
