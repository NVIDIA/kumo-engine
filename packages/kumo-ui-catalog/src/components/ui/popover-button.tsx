// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";
import { Button } from "./button";
import { GridRow } from "../blocks/layout/GridRow";
import { GridCol } from "../blocks/layout/GridCol";
import { CloseSmall } from "@/icons";
import { cn } from "@/lib/utils";
import * as PopoverPrimitive from "@radix-ui/react-popover";

interface PopoverButtonProps {
    content?: React.ReactNode;
    contentClassName?: string;
    children?: string | React.ReactNode;
}

export function PopoverButton({
    content = null,
    contentClassName = "",
    children = "Show",
    ...props
}: PopoverButtonProps & React.ComponentProps<typeof PopoverPrimitive.Root>) {
    const [openPopover, setOpenPopover] = useState(false);

    const handleClose = useCallback(() => {
        setOpenPopover(!openPopover);
    }, [openPopover]);

    return (
        <Popover
            open={openPopover}
            onOpenChange={handleClose}
            modal={true}
            {...props}>
            <PopoverTrigger asChild>
                {
                    <Button
                        data-testid="popup-trigger"
                        variant="light-outline"
                        size="sm"
                        className="border-slate-500 hover:bg-slate-200"
                        onClick={handleClose}>
                        {children}
                    </Button>
                }
            </PopoverTrigger>
            <PopoverContent
                className={cn("p-3", contentClassName)}
                align="end"
                side="bottom">
                <GridRow className="flex justify-between gap-2" gutter="sm">
                    <GridCol>{content}</GridCol>
                    <GridCol>
                        <div className="cursor-pointer" onClick={handleClose}>
                            <CloseSmall
                                className="fill-text-title-primary p-0"
                                viewBox="100 -800 800 800"
                            />
                        </div>
                    </GridCol>
                </GridRow>
            </PopoverContent>
        </Popover>
    );
}
