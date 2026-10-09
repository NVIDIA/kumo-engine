// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    cn,
    ColumnDef,
    Info,
    Select,
    SelectOption,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { HoldoutDownload } from "components/Download/HoldoutDownload";

export const EvaluationMetricsColumns = ({
    tuneMetric,
    enableK = false,
    enableMetrics = false,
    menuOptions,
    onSelect,
    placeHolder,
    downloadClicked,
    onDownloadClick,
    windowWidth
}: {
    tuneMetric: string;
    enableK?: boolean;
    enableMetrics?: boolean;
    menuOptions?: SelectOption[];
    onSelect?: (value: SelectOption) => void;
    placeHolder?: string;
    downloadClicked?: boolean;
    onDownloadClick?: () => void;
    windowWidth: number;
}) => {
    let columnFixedWidth = 250;
    const header: ColumnDef<{ [key: string]: string | number }>[] = [
        {
            header: () => <Header>{"Metric"}</Header>,
            accessorKey: "metric_name",
            size: 250,
            enableSorting: false,
            id: "metric_name",
            cell: ({ row, getValue }) => {
                const value = getValue() as string;

                if (tuneMetric === row?.original?.compare_key)
                    return (
                        <div className={"text-title-primary text-base ms-3"}>
                            {value}
                        </div>
                    );
                return (
                    <div className={"text-title-primary text-base"}>
                        {value}
                    </div>
                );
            }
        }
    ];

    if (enableK) {
        columnFixedWidth += 200;
        header.push({
            accessorKey: "k",
            enableSorting: false,
            size: 200,
            header: () => <Header>K</Header>,
            cell: ({ row, getValue }) => {
                const value = getValue() as string;

                if (tuneMetric === row?.original?.compare_key)
                    return <HighlightCell value={value} />;
                return (
                    <div className={"text-title-primary text-base"}>
                        {value}
                    </div>
                );
            }
        });
    }
    if (enableMetrics) {
        columnFixedWidth += 200;
        header.push({
            accessorKey: "metrics",
            enableSorting: false,
            size: 200,
            header: () => <Header>Metrics</Header>,
            cell: ({ row, getValue }) => {
                const value = getValue() as string;
                if (tuneMetric === row?.original?.compare_key)
                    return (
                        <div className={"text-title-primary text-base ms-3"}>
                            {value?.replaceAll("PER_", "")}
                        </div>
                    );
                return (
                    <div className={"text-title-primary text-base"}>
                        {value?.replaceAll("PER_", "")}
                    </div>
                );
            }
        });
    }

    columnFixedWidth =
        columnFixedWidth +
        (menuOptions && menuOptions?.length > 0 && onSelect ? 120 : 0);

    const INNER_WINDOW = windowWidth * 0.8;
    const columnWidth = (INNER_WINDOW - columnFixedWidth - 100) / 3;
    const splits = ["Holdout", "Validation", "Training"];

    splits.forEach((split) => {
        header.push({
            accessorKey: split.toLowerCase(),
            enableSorting: false,
            size: columnWidth,
            header: () => (
                <div className="flex items-center gap-2">
                    <Header
                        className={
                            split === "Holdout"
                                ? "text-pink-600 text-base"
                                : "text-title-secondary text-base"
                        }>
                        {split}
                    </Header>

                    {split === "Holdout" && (
                        <HoldoutDownload
                            downloadClicked={downloadClicked}
                            onDownloadClick={onDownloadClick}
                        />
                    )}
                    {split === "Training" && (
                        <Tooltip id={"training-metrics-tooltip"}>
                            <TooltipTrigger>
                                <span>
                                    <Info className="size-3.5" />
                                </span>
                            </TooltipTrigger>
                            <TooltipContent
                                arrow={false}
                                className="max-w-100 text-justify"
                                side={"bottom"}
                                sideOffset={5}>
                                {
                                    "Shows the metric at the last training epoch. Kumo uses the best checkpoint for predictions."
                                }
                            </TooltipContent>
                        </Tooltip>
                    )}
                </div>
            ),
            cell: ({ row, getValue }) => {
                const value = getValue() as string;

                if (tuneMetric === row?.original?.compare_key)
                    return (
                        <HighlightCell
                            value={value}
                            style={
                                split === "Holdout"
                                    ? {
                                          borderTopLeftRadius: 4,
                                          borderBottomLeftRadius: 4
                                      }
                                    : split === "Training"
                                      ? {
                                            borderTopRightRadius: 4,
                                            borderBottomRightRadius: 4,
                                            width: 100
                                        }
                                      : {}
                            }
                        />
                    );
                return (
                    <div className={"text-title-primary text-base"}>
                        {value}
                    </div>
                );
            }
        });
    });
    if (menuOptions && menuOptions?.length > 0 && onSelect) {
        header.push({
            header: () => (
                <Select
                    id="ranked-metrics-top"
                    placeholder={
                        placeHolder ??
                        (menuOptions?.length > 0 ? menuOptions[0]?.label : "")
                    }
                    variant="default"
                    popoverClassName="w-50"
                    size="sm"
                    search={false}
                    options={menuOptions}
                    onValuesChange={(values: string[]) => {
                        const option = menuOptions.find(
                            (opt) => opt.value === values[0]
                        );
                        if (option) onSelect(option);
                    }}
                    mode="single"
                />
            ),
            accessorKey: "selection",
            enableSorting: false,
            id: "selection",
            size: 120
        });
    }
    return header;
};

const HighlightCell = ({
    isBold,
    value,
    style = {},
    className
}: {
    isBold?: boolean;
    value: number | string;
    style?: React.CSSProperties;
    className?: string;
}) => {
    return (
        <div
            className={cn(
                "ps-3 text-pink-600 bg-pink-50 text-base items-center flex",
                isBold && "font-bold",
                className
            )}
            style={{ ...style, height: 26 }}>
            {value}
        </div>
    );
};

const Header = ({
    children,
    className = "text-title-secondary text-base"
}: {
    children: string | React.ReactNode;
    className?: string;
}) => {
    return <div className={`${className}`}>{children}</div>;
};
