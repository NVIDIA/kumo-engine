// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EmptyEntityData } from "@/types/empty-entity";
import * as React from "react";

import { AddCircle } from "@/icons/MaterialIcons/AddCircle";
import { cn } from "@/lib/utils";
import { Button } from "./button";
import { ImportContacts } from "@/icons";

function EmptyEntity({
    data,
    className,
    children,
    ...props
}: React.ComponentProps<"div"> & {
    data: EmptyEntityData;
}) {
    return data ? (
        <div
            className={cn(
                "text-muted-foreground flex flex-col items-center gap-4 p-3",
                data.project
                    ? "absolute inset-0 justify-center"
                    : "h-full justify-between",
                className
            )}
            {...props}>
            {data.icon && data.icon}
            {data.label}
            <div
                className={cn(
                    "flex gap-3",
                    data.project ? "flex-row" : "flex-col items-center"
                )}>
                {data.project ? (
                    <>
                        {/* Docs button first when project */}
                        <Button
                            id={data.docsId}
                            variant="secondary"
                            className="text-sm"
                            onClick={data.onDocsClick}>
                            {data.docsIcon ?? (
                                <ImportContacts className="size-5" />
                            )}
                            {data.docsText}
                        </Button>

                        {/* Render button second when project  */}
                        <Button
                            id={data.renderButtonId}
                            className="w-fit text-sm font-bold"
                            onClick={data.onRenderClick}>
                            <>
                                {data.renderButtonIcon ?? (
                                    <AddCircle className="size-4.5 fill-white" />
                                )}
                                {data.renderButtonText}
                            </>
                        </Button>
                    </>
                ) : (
                    <>
                        {/* Render button */}
                        <Button
                            id={data.renderButtonId}
                            className="w-fit text-sm font-bold"
                            onClick={data.onRenderClick}>
                            <>
                                {data.renderButtonIcon ?? (
                                    <AddCircle className="size-4.5 fill-white" />
                                )}
                                {data.renderButtonText}
                            </>
                        </Button>

                        {/* Docs button */}
                        <Button
                            id={data.docsId}
                            className="w-fit text-sm font-bold"
                            onClick={data.onDocsClick}>
                            {data.docsIcon}
                            {data.docsText}
                        </Button>
                    </>
                )}
            </div>
            {children}
        </div>
    ) : (
        <></>
    );
}

export { EmptyEntity };
