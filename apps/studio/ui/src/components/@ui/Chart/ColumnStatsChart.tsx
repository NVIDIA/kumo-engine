// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";
import Spacer from "react-spacer";

import {
    AreaChart,
    BarChart,
    Card,
    CardContent,
    Description,
    GridCol,
    GridRow,
    Symbol,
    useWindowDimensions
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { ColumnArgs } from "types/tables";

import { sTypes } from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";
import { isValid } from "utils/Helpers/typeGuards";

const ChartType = {
    None: "",
    AreaChart: "AreaChart",
    BarChart: "BarChart"
};

export interface ColumnStatsChartProps {
    data: ColumnArgs;
    isSnapShot?: boolean;
}

interface ColumnChartData {
    xAxisLabels: Array<string | number>;
    dataSet: Array<string | number>;
    chartType: string;
    trimXAxisLabel: boolean;
    toolTipText: string;
}

/**
 * Derives the chart series + display options for a column from its statistics,
 * dispatching on the column's semantic type. Returns empty series (chartType
 * None) when there is no distribution to plot.
 */
const getColumnChartData = (
    data: ColumnArgs,
    stype?: string
): ColumnChartData => {
    const statistics = data?.statistics;

    if (stype === sTypes.Numerical && statistics?.distribution_points) {
        const stats = statistics.distribution_points;
        const labels: Array<string | number> = stats
            .slice(0, 20)
            .map((val: Array<number | string>) => val[0]);
        if (typeof stats[0]?.[0] === "number")
            labels[labels.length - 1] = labels[labels.length - 1] + "+";
        return {
            xAxisLabels: labels,
            dataSet: stats
                .slice(0, 20)
                .map((val: Array<number | string>) => val[1]),
            chartType:
                Number(statistics.counts?.num_unique) > 20
                    ? ChartType.AreaChart
                    : ChartType.BarChart,
            trimXAxisLabel: true,
            toolTipText: "Percent Share"
        };
    }
    if (
        (stype === sTypes.Categorical || stype === sTypes.ID) &&
        statistics?.distribution_category
    ) {
        const stats = statistics.distribution_category;
        return {
            xAxisLabels: stats
                .slice(0, 10)
                .map((value: Array<number | string>) => value[0]),
            dataSet: stats
                .slice(0, 10)
                .map((val: Array<number | string>) => val[1]),
            chartType: ChartType.BarChart,
            trimXAxisLabel: true,
            toolTipText: "Percent Share"
        };
    }
    if (stype === sTypes.Timestamp && statistics?.distribution_date) {
        const stats = statistics.distribution_date;
        return {
            xAxisLabels: stats.map(
                (value: Array<number | string>) => value[0] + " - " + value[1]
            ),
            dataSet: stats.map((val: Array<number | string>) => val[2]),
            chartType: ChartType.BarChart,
            trimXAxisLabel: false,
            toolTipText: "Count"
        };
    }
    return {
        xAxisLabels: [],
        dataSet: [],
        chartType: ChartType.None,
        trimXAxisLabel: true,
        toolTipText: "Percent Share"
    };
};

/**
 * Column Statistics component
 * @param data Statistics
 * @returns ```<Chart>```
 */
const ColumnStatsChart = ({ data, isSnapShot }: ColumnStatsChartProps) => {
    const statistics = data.statistics;
    const counts = statistics?.counts;
    const sampleRows = data.sample_rows ?? [];
    const summaryStats = statistics?.summary_stats ?? [];
    const stype = data.stype;

    const { numUnique, missingPercentage, countTotal } = useMemo(() => {
        if (!counts) {
            return {
                numUnique: null,
                missingPercentage: null,
                countTotal: null
            };
        }
        return {
            numUnique: parseNumbers(counts.num_unique as number),
            missingPercentage: parseNumbers(
                counts.missing_percentage as number
            ),
            countTotal: parseNumbers(counts.total as number, { decimals: 2 })
        };
    }, [counts]);

    const { xAxisLabels, dataSet, chartType, trimXAxisLabel, toolTipText } =
        useMemo(() => getColumnChartData(data, stype), [data, stype]);

    const { width } = useWindowDimensions();

    if (!isValid(statistics) && sampleRows.length < 1) {
        return (
            <div>We are unable to display column statistics at this time</div>
        );
    }

    const showChart =
        chartType === ChartType.AreaChart || chartType === ChartType.BarChart;
    const cardStyle: React.CSSProperties = {
        backgroundColor: "var(--color-slate-50)",
        overflowY: "auto"
    };
    const chartViewEnabled = showChart && dataSet.length > 0;

    return (
        <GridRow className="flex flex-wrap">
            <GridCol
                xs={12}
                sm={12}
                md={3}
                className="flex flex-col"
                data-testid="stats">
                <Description
                    size="md"
                    variant="secondary"
                    className="font-medium pb-2">
                    {"Stats"}
                </Description>
                <Card className="flex-grow" style={cardStyle}>
                    <CardContent>
                        <GridRow>
                            {isValid(countTotal) && (
                                <GridCol
                                    xs={12}
                                    sm={12}
                                    md={6}
                                    data-testid="count">
                                    <StatsField
                                        label={"Count"}
                                        value={countTotal}
                                    />
                                </GridCol>
                            )}
                            {isValid(numUnique) && (
                                <GridCol
                                    xs={12}
                                    sm={12}
                                    md={6}
                                    data-testid="unique">
                                    <StatsField
                                        label={"Unique"}
                                        value={
                                            stype === sTypes.Numerical ? (
                                                <Symbol type="emdash" />
                                            ) : (
                                                numUnique
                                            )
                                        }
                                    />
                                </GridCol>
                            )}
                            {isValid(missingPercentage) && (
                                <GridCol
                                    xs={12}
                                    sm={12}
                                    md={6}
                                    data-testid="missing">
                                    <StatsField
                                        label={"Missing"}
                                        value={missingPercentage + "%"}
                                    />
                                </GridCol>
                            )}
                            {summaryStats.map(
                                (
                                    value: Array<number | string>,
                                    index: number
                                ) =>
                                    (stype === sTypes.ID &&
                                        value[0] === "Mode") ||
                                    !isValid(value[1]) ? null : (
                                        <GridCol
                                            xs={12}
                                            sm={12}
                                            md={6}
                                            key={"stats_" + index}>
                                            <StatsField
                                                label={value[0] as string}
                                                value={parseNumbers(value[1], {
                                                    decimals: 2
                                                })}
                                            />
                                        </GridCol>
                                    )
                            )}
                        </GridRow>
                    </CardContent>
                </Card>
            </GridCol>

            {dataSet.length > 0 && (
                <GridCol
                    xs={12}
                    sm={12}
                    md={6}
                    className="flex flex-col"
                    data-testid="distribution">
                    <Description
                        size="md"
                        variant="secondary"
                        className="font-medium pb-2">
                        {"Distribution"}
                    </Description>
                    <Card style={cardStyle}>
                        <CardContent>
                            {showChart && (
                                <div>
                                    {chartType === ChartType.AreaChart ? (
                                        <AreaChart
                                            id="area-chart"
                                            xAxisLabel={""}
                                            yAxisLabel={""}
                                            legend={{
                                                display: false
                                            }}
                                            xAxisTickLabels={xAxisLabels}
                                            dataSet={dataSet}
                                            customToolTip={true}
                                            toolTipText={toolTipText}
                                            height={380}
                                            width={width / 3.25}
                                        />
                                    ) : chartType === ChartType.BarChart ? (
                                        <BarChart
                                            id={"bar-chart"}
                                            xAxisLabel={""}
                                            dataSetLabel={""}
                                            yAxisLabel={""}
                                            legend={{
                                                display: false
                                            }}
                                            xAxisTickLabels={xAxisLabels}
                                            dataSet={dataSet}
                                            trimXAxisLabel={trimXAxisLabel}
                                            customToolTip={true}
                                            customLabel={false}
                                            toolTipText={toolTipText}
                                            height={380}
                                            width={width / 3.25}
                                        />
                                    ) : null}

                                    <Spacer height={20} />
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </GridCol>
            )}
            {!isSnapShot && (
                <GridCol
                    xs={12}
                    sm={12}
                    md={chartViewEnabled ? 3 : 8}
                    className="flex flex-col min-w-30 max-w-fit"
                    // eslint-disable-next-line no-restricted-syntax -- runtime width
                    style={{
                        width: chartViewEnabled
                            ? width / (width > 1600 ? 6.5 : 7)
                            : width / 2.8
                    }}
                    data-testid="sample">
                    <Description
                        size="md"
                        variant="secondary"
                        className="font-medium pb-2">
                        {"Sample"}
                    </Description>
                    <Card
                        className="flex-grow"
                        style={cardStyle}
                        data-testid="sample-value">
                        <CardContent>
                            <SampleRows rows={sampleRows} />
                        </CardContent>
                    </Card>
                </GridCol>
            )}
        </GridRow>
    );
};

export default WithErrorBoundary(ColumnStatsChart);

const StatsField = ({
    label,
    value
}: {
    label: string;
    value: string | number | React.ReactNode | null;
}) => {
    return (
        <>
            <span
                className="text-sm leading-5 text-left text-title-secondary [text-underline-position:from-font] [text-decoration-skip-ink:none]
">
                {label}
            </span>
            <Spacer height={8} />
            <span
                className="text-title-primary text-base font-semibold leading-5 text-left text-wrap [text-underline-position:from-font] [text-decoration-skip-ink:none]
">
                {value}
            </span>
        </>
    );
};

const SampleRows = ({
    rows
}: {
    rows: string | (string | null)[] | null | undefined;
}) => {
    const data =
        !rows || (Array.isArray(rows) && rows.length === 0)
            ? ["N/A"]
            : Array.isArray(rows)
              ? [...rows]
              : [rows.toString()];

    return (
        <div className="whitespace-nowrap overflow-auto max-h-100 break-words">
            {data.map((row, i) => (
                <span key={`sample-${i}`} className="output-stats-label ">
                    <SampleData value={row} />
                    <Spacer height={8} />
                </span>
            ))}
        </div>
    );
};

export const SampleData = ({ value }: { value?: string | number | null }) => {
    if (!isValid(value)) {
        return <>{"N/A"}</>;
    }
    const isImage = /\.(jpeg|jpg|gif|png)$/.test(value as string);
    if (isImage) {
        return (
            <img
                src={value?.toString()?.trim()}
                className="w-12.5 m-1.25"
                alt="sample"
            />
        );
    }
    return <>{value}</>;
};
