// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@/lib/utils";
import { ScrollArea } from "./scroll-area";
import { Skeleton } from "./skeleton";

function TextScrollArea({
    text,
    loading = false,
    width, // new prop
    height, // new prop
    className,
    ...props
}: {
    text: string;
    loading?: boolean;
    width?: number; // optional, in pixels
    height?: number; // optional, in pixels
    className?: string;
} & React.ComponentProps<typeof ScrollArea>) {
    const containerStyle = {
        width: width ? `${width}px` : undefined,
        height: height ? `${height}px` : undefined
    };
    return (
        <ScrollArea
            className={cn(
                "relative rounded-sm border text-sm",
                !width && "w-full",
                !height && "h-64",
                loading
                    ? ""
                    : "bg-slate-200 p-2 [&_[data-slot=scroll-area-thumb]]:bg-[var(--scrollbar-dark)]",
                className
            )}
            type="always"
            style={containerStyle} // apply dynamic width/height here
            {...props}>
            {loading ? (
                <Skeleton
                    style={containerStyle} // match width/height on Skeleton too
                    className={cn(
                        !width && "w-full",
                        !height && "h-64",
                        "rounded-sm"
                    )}
                />
            ) : (
                <div className="overflow-auto whitespace-pre-wrap break-all !font-mono">
                    {text}
                </div>
            )}
        </ScrollArea>
    );
}

export { TextScrollArea };
