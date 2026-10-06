// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { InkSelection } from "@/icons";
import { cn } from "@/lib/utils";
import { Skeleton } from "./skeleton";
import { TruncatedTooltip } from "./tooltip";

interface ListViewProps<T> {
    className?: string;
    id?: string;
    searchFiltered?: T[];
    loadingStatus?: boolean;
    overviewComponent?: React.ReactNode;
    selectedEntity: T | null;
    handleCellClick: (item: T) => void;
    emptyTableComponent?: React.ReactNode;
    selectedVisible: boolean;
    selectString?: string;
    pageHeight?: number;
}

export function ListView<
    T extends {
        id: string;
        name: string;
        dateCreated?: string;
    }
>({
    className,
    id,
    searchFiltered,
    loadingStatus = false,
    overviewComponent = null,
    selectedEntity,
    handleCellClick,
    emptyTableComponent,
    selectedVisible,
    selectString,
    pageHeight,
    ...props
}: ListViewProps<T>) {
    return (
        <Card
            id={id}
            style={{ maxHeight: pageHeight }}
            className={cn(
                "inline-flex w-full flex-row items-start justify-start self-stretch overflow-hidden rounded-sm bg-white px-2 py-2 shadow-none outline-1 outline-slate-100",
                className
            )}
            {...props}>
            {/* Left Panel */}
            <ScrollArea
                style={{ height: pageHeight }}
                className="scrollbar-always-visible pb-4"
                scrollHideDelay={0}
                type="always">
                <div
                    className={
                        "max-w-250 flex w-60 flex-col items-start justify-start gap-1 rounded-none border-none bg-white px-0 py-0"
                    }>
                    {loadingStatus
                        ? Array.from({ length: 5 }).map((_, i) => (
                              <div
                                  key={`skeleton-${i}`}
                                  className="flex flex-col gap-1 self-stretch px-4 py-2">
                                  <Skeleton className="w-45 h-5 rounded-sm" />
                                  <Skeleton className="w-38 h-4 rounded-sm" />
                              </div>
                          ))
                        : searchFiltered && searchFiltered.length < 1
                          ? emptyTableComponent
                          : searchFiltered?.map((item, i) => {
                                const isSelected =
                                    item?.id === selectedEntity?.id;
                                return (
                                    <button
                                        type="button"
                                        key={item.id}
                                        data-state={
                                            isSelected ? "Active" : "Resting"
                                        }
                                        aria-pressed={isSelected}
                                        onClick={() => handleCellClick(item)}
                                        className={cn(
                                            // KUI Vertical Nav selected item (Figma light theme, node 4010:51081):
                                            // the indicator is a 2px-wide pill rail INSET 4px from the left and
                                            // 4px from top/bottom (size/050 width, radius/round) — NOT a flush
                                            // full-height border. Rendered via ::before so it floats inside.
                                            // Corners = KUI radius/sm (2px); selected bg = kumo-grey-50
                                            // (subtle off-white) so visual weight comes from the green bar only.
                                            "relative flex w-full cursor-pointer flex-col items-start justify-center gap-1 self-stretch overflow-hidden rounded-[2px] px-4 py-2 text-left",
                                            isSelected
                                                ? "text-title-primary before:bg-brand-green bg-kumo-grey-50 before:absolute before:bottom-1 before:left-1 before:top-1 before:w-0.5 before:rounded-full before:content-['']"
                                                : "text-title-secondary hover:bg-kumo-grey-50 bg-white"
                                        )}>
                                        <TruncatedTooltip
                                            tooltipContent={item?.name}
                                            testId={`entity-name-${i}`}
                                            // KUI Vertical Nav: selected = Bold (700),
                                            // unselected = Regular (400). Size unchanged.
                                            className={cn(
                                                "self-stretch truncate text-base",
                                                isSelected
                                                    ? "font-bold"
                                                    : "font-normal"
                                            )}>
                                            {item?.name}
                                        </TruncatedTooltip>
                                        <div
                                            className={cn(
                                                // Date line: bold when selected (matches the
                                                // name), regular otherwise. Size unchanged (14px).
                                                "self-stretch text-sm leading-tight",
                                                isSelected
                                                    ? "font-bold"
                                                    : "font-normal"
                                            )}>
                                            {item?.dateCreated
                                                ? item.dateCreated
                                                : ""}
                                        </div>
                                    </button>
                                );
                            })}
                </div>
            </ScrollArea>

            <Card className="justify-centre items-centre inline-flex w-full self-stretch overflow-y-auto rounded bg-slate-100 px-4 py-4 shadow-none">
                {selectedEntity && selectedVisible ? (
                    overviewComponent
                ) : (
                    <div
                        test-id="empty-view"
                        className="bg-slate-100 pt-2"
                        style={{
                            height: (pageHeight as number) + 5,
                            display: "flex",
                            justifyContent: "center",
                            alignItems: "center",
                            borderRadius: "4px"
                        }}>
                        <div
                            style={{
                                textAlign: "center"
                            }}>
                            <InkSelection className="mx-auto size-8" />
                            <div
                                data-testid="entity-item-not-selected"
                                className="fs-5">
                                {`Select a ${selectString} from the left panel to view its details here.`}
                            </div>
                        </div>
                    </div>
                )}
            </Card>
        </Card>
    );
}
