// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Row,
    Skeleton,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { ColumnArgs } from "types/tables";

import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";

const MAX_SAMPLE_VALUES_CHAR_LEN = 100;

export const SampleCell = ({ row }: { row: Row<ColumnArgs> }) => {
    if (
        (row?.original && row?.original?.isSampleAvailable) ||
        (row?.original?.sample_rows && row?.original?.sample_rows?.length > 0)
    ) {
        const value = row?.original?.sample_rows;
        if (value === undefined) {
            // Extra secure
            return <></>;
        }
        try {
            if (
                capitalizeFirstLetter(row?.original?.stype) === "Image" &&
                value !== null
            ) {
                const urls: string[] =
                    typeof value === "string"
                        ? [value as string]
                        : (value as string[])?.map((url: string) =>
                              url?.trim()
                          );

                const imageComponents = urls
                    ?.filter((url: string) => /\.(jpeg|jpg|gif|png)$/.test(url))
                    ?.map((url: string, index: number) => (
                        <img
                            key={index}
                            src={url}
                            alt="Sample"
                            className="w-12.5 m-1.25"
                        />
                    ));
                return (
                    <div className="w-[inherit] overflow-hidden">
                        {imageComponents}
                    </div>
                );
            }
            const textString =
                value && Array.isArray(value)
                    ? value?.join(", ")
                    : (value as string);

            return (
                // If not image, for other stypes
                <div
                    data-testid="sample-cell"
                    className="w-[inherit] text-justify whitespace-nowrap overflow-hidden text-ellipsis">
                    {textString?.length <= MAX_SAMPLE_VALUES_CHAR_LEN ? (
                        (textString ?? "")
                    ) : (
                        <Tooltip id={row?.id as string}>
                            <TooltipTrigger>
                                <span>{textString ?? ""}</span>
                            </TooltipTrigger>
                            <TooltipContent
                                className="max-w-112.5 max-h-100 overflow-y-auto text-justify"
                                arrow={false}
                                side={"left"}
                                sideOffset={5}>
                                {textString ?? ""}
                            </TooltipContent>
                        </Tooltip>
                    )}
                </div>
            );
        } catch {
            return <></>;
        }
    } else {
        return <Skeleton className="h-4 w-40" />;
    }
};
