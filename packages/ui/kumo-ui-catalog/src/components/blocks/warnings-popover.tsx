// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import {
    Popover,
    PopoverContent,
    PopoverTrigger
} from "@/components/ui/popover";
import { Title } from "@/components/ui/title";
import { WarningIcon } from "@/icons";
import { cn } from "@/lib/utils";

export type WarningItem = {
    description: string;
};

export interface WarningsPopoverProps {
    warnings: WarningItem[];
}

function WarningsPopover({ warnings = [] }: WarningsPopoverProps) {
    return (
        <Popover>
            <PopoverTrigger asChild>
                <Button
                    id="warnings-button"
                    variant="outline"
                    className="hover:bg-kui-bg-hover px-2 [&_svg]:fill-amber-500">
                    <WarningIcon className="fill-amber-500" />
                    Warnings
                </Button>
            </PopoverTrigger>
            <PopoverContent
                align="end"
                side="bottom"
                aria-label="Warnings"
                className="text-title-primary w-150 rounded-md border-amber-500 bg-amber-50 p-6">
                <div className="flex items-start gap-4">
                    <span>
                        <WarningIcon className="size-5.5 fill-amber-500 stroke-amber-500" />
                    </span>
                    <div>
                        <Title
                            variant="section-header"
                            className="text-title-primary mt-1 font-semibold">
                            {`${warnings.length} warnings detected`}
                        </Title>
                        <ul className="max-h-75 list-disc overflow-auto ps-4">
                            {warnings?.map((w, i) => (
                                <li
                                    className={cn(
                                        "text-title-primary my-4 whitespace-pre-line text-sm",
                                        i === warnings.length - 1 && "mb-0"
                                    )}
                                    key={`warning-${i}`}>
                                    {w.description}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </PopoverContent>
        </Popover>
    );
}

export { WarningsPopover };
