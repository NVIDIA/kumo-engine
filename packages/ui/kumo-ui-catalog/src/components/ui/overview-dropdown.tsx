// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as React from "react";

import { Button } from "@/components/ui/button";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { MoreVerticle } from "@/icons/MaterialIcons/MoreVerticle";

interface OverviewDropdownItem {
    content: React.ReactNode;
    disabled: boolean;
    onSelect: () => void;
    tooltipText: string;
}

interface OverviewDropdownProps {
    id: string;
    actionItems: OverviewDropdownItem[];
    buttonClassName?: string;
}

export const OverviewDropdown: React.FC<OverviewDropdownProps> = ({
    id,
    actionItems,
    buttonClassName
}) => {
    return (
        <DropdownMenu key={"key-" + id} modal={false}>
            <DropdownMenuTrigger
                asChild
                id={id}
                data-testid="more-options-dropdown">
                <Button
                    variant="outline"
                    aria-label="More options"
                    className={
                        buttonClassName ??
                        "hover:bg-kui-bg-hover border-border-primary w-9 px-1.5"
                    }>
                    <MoreVerticle />
                </Button>
            </DropdownMenuTrigger>
            {actionItems && (
                <DropdownMenuContent
                    align="end"
                    className="text-title-primary text-sm">
                    {actionItems?.map((item, i) => (
                        <DropdownMenuItem
                            key={"dropDownItem_" + i}
                            id={"dropDownItem_button_" + i}
                            tooltipText={item?.tooltipText ?? ""}
                            onSelect={item?.onSelect}
                            className={
                                item.disabled
                                    ? "text-title-secondary pointer-events-none cursor-default"
                                    : "text-title-secondary hover:bg-kumo-grey-50 hover:text-title-primary hover:[&_svg]:fill-title-primary"
                            }>
                            <span className="flex items-center gap-2">
                                {item?.content}
                            </span>
                        </DropdownMenuItem>
                    ))}
                </DropdownMenuContent>
            )}
        </DropdownMenu>
    );
};
