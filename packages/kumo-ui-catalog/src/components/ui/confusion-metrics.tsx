// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { Table, TableBody, TableCell, TableRow } from "./table";
import { ConfusionMatrixCell } from "@/types/threshold-metrics";

interface ConfusionMetricTableProps {
    classes: Array<React.JSX.Element> | string[];
    values: Array<Array<ConfusionMatrixCell>>;
    sumsRight: Array<number>;
    sumsBottom: Array<number>;
    title: string;
}

function getBgColor(percentage: number) {
    // Starting color (white) and ending color (rgb(116, 171, 179))
    const startColor = { r: 255, g: 255, b: 255 }; // white
    const endColor = { r: 116, g: 171, b: 179 }; // rgb(116, 171, 179)

    // Calculate the color based on the slider percentage
    const r = Math.round(
        startColor.r + (endColor.r - startColor.r) * (percentage / 100)
    );
    const g = Math.round(
        startColor.g + (endColor.g - startColor.g) * (percentage / 100)
    );
    const b = Math.round(
        startColor.b + (endColor.b - startColor.b) * (percentage / 100)
    );

    // Create the dynamic background color based on the RGB values
    const backgroundColor = `rgb(${r}, ${g}, ${b})`;
    return backgroundColor;
}

export const ConfusionMetricTable: React.FC<ConfusionMetricTableProps> = ({
    classes,
    values,
    sumsRight,
    sumsBottom,
    title
}) => {
    return (
        <div className="inline-flex flex-col items-start justify-end gap-4">
            <div className="justify-start text-sm font-bold leading-none text-slate-500">
                {title?.toUpperCase()}
            </div>
            <Table className="bg-transparent">
                <TableBody>
                    <TableRow className="!border-b-0 bg-transparent hover:bg-transparent group-hover:bg-transparent">
                        <TableCell
                            colSpan={classes.length + 2}
                            className="!h-2 bg-transparent pl-32 text-center text-sm font-medium leading-none text-gray-800 hover:bg-transparent group-hover:bg-transparent">
                            Predicted
                        </TableCell>
                    </TableRow>
                    <TableRow className="!border-b-0 !border-slate-400 bg-transparent hover:bg-transparent group-hover:bg-transparent">
                        <TableCell
                            rowSpan={classes.length + 3}
                            sticky={{ left: 0, zIndex: 10 }}
                            className="w-[30px] bg-transparent text-center align-middle hover:bg-transparent group-hover:bg-transparent">
                            <div className="rotate-[-90deg] text-sm font-medium leading-none text-gray-800">
                                Actual
                            </div>
                        </TableCell>
                    </TableRow>

                    {/* Column headers */}
                    <TableRow className="!border-b-0 !border-slate-400 bg-transparent hover:bg-transparent group-hover:bg-transparent">
                        {classes.map((label, index) => (
                            <React.Fragment key={`header-${index}`}>
                                {index === 0 && (
                                    <TableCell className="bg-transparent hover:bg-transparent group-hover:bg-transparent" />
                                )}
                                <TableCell className="justify-start bg-transparent text-center text-sm font-medium leading-none text-slate-500 hover:bg-transparent group-hover:bg-transparent">
                                    {label}
                                </TableCell>
                            </React.Fragment>
                        ))}
                    </TableRow>

                    {/* Matrix values */}
                    {values.map((row, rowIndex) => (
                        <TableRow
                            className="!border-0 bg-transparent hover:bg-transparent group-hover:bg-transparent"
                            key={`row-${rowIndex}`}>
                            {row.map((cell, colIndex) => (
                                <React.Fragment
                                    key={`cell-${rowIndex}-${colIndex}`}>
                                    {colIndex === 0 && (
                                        <TableCell
                                            sticky={{ left: 0 }}
                                            className="!m-0 h-[60px] w-[30px] justify-start !border-0 bg-transparent !p-0 text-center text-sm font-medium leading-none text-slate-500 hover:bg-transparent group-hover:bg-transparent">
                                            <div className="origin-center rotate-[-90deg] whitespace-nowrap text-center font-normal text-gray-500">
                                                {classes[rowIndex]}
                                            </div>
                                        </TableCell>
                                    )}

                                    <TableCell
                                        className="!border-1.1 text-md h-14 w-44 justify-start !border-slate-300 bg-transparent text-center !font-['Source_Code_Pro'] font-semibold leading-none text-gray-800 hover:bg-transparent group-hover:bg-transparent"
                                        style={{
                                            backgroundColor: getBgColor(
                                                cell.percentage
                                            ),
                                            border: "0.14em solid rgba(204,204,204,1)"
                                        }}>
                                        {cell.count} (
                                        {cell.percentage.toFixed(1)}%)
                                    </TableCell>
                                    {colIndex === row.length - 1 && (
                                        <TableCell className="justify-start bg-transparent text-sm font-medium leading-none text-slate-500 hover:bg-transparent group-hover:bg-transparent">
                                            {sumsRight[rowIndex]}
                                        </TableCell>
                                    )}
                                </React.Fragment>
                            ))}
                        </TableRow>
                    ))}

                    {/* Bottom sum row */}
                    <TableRow className="!border-b-0 !border-slate-400 bg-transparent hover:bg-transparent group-hover:bg-transparent">
                        <TableCell className="text-slate-500 hover:bg-transparent group-hover:bg-transparent" />
                        {sumsBottom.map((sum, index) => (
                            <TableCell
                                key={`sum-${index}`}
                                className="bg-transparent text-center text-sm font-medium text-slate-500 hover:bg-transparent group-hover:bg-transparent">
                                {sum}
                            </TableCell>
                        ))}
                    </TableRow>
                </TableBody>
            </Table>
        </div>
    );
};
