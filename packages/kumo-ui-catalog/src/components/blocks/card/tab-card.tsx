// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface TabCardProps {
    title?: string | React.ReactNode;
    loadingStatus?: boolean;
    children: (currentTab: number) => React.ReactNode;
    tabs: Array<{
        value: number;
        label: string;
        customLabel?: React.ReactNode;
    }>;
    wide?: boolean;
    defaultSelected?: number;
    disableSelection?: boolean;
    className?: string;
    contentClassName?: string;
}

const getTabTestId = (label: unknown) => {
    if (typeof label !== "string") {
        return undefined;
    }

    const normalizedLabel = label
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "");

    return normalizedLabel ? `tab-${normalizedLabel}` : undefined;
};

function TabCard({
    title,
    children,
    tabs,
    wide = false,
    defaultSelected,
    disableSelection = false,
    loadingStatus = false,
    className = "",
    contentClassName = ""
}: TabCardProps) {
    const defaultTab: number = defaultSelected || tabs?.[0].value || 0;
    const [tabValue, setTabValue] = useState(defaultTab.toString());

    useEffect(() => {
        if (defaultSelected !== undefined) {
            setTabValue(defaultSelected.toString());
        }
    }, [defaultSelected]);

    return (
        <Card className={cn("gap-0 p-0", className)}>
            <Tabs
                defaultValue={tabValue}
                value={tabValue}
                onValueChange={(newValue: string) => {
                    if (!disableSelection) {
                        setTabValue(newValue);
                    }
                }}>
                {loadingStatus ? (
                    <Skeleton className="h-8 w-[250px]" />
                ) : (
                    title && <h3 className="mb-0">{title}</h3>
                )}
                <TabsList className="gap-8">
                    {tabs?.map(({ value, label, customLabel }, i) => (
                        <TabsTrigger
                            key={"tabs-trigger-" + i}
                            data-testid={getTabTestId(label)}
                            value={value.toString()}
                            className={"p-1 pl-0 pr-0"}>
                            {customLabel ?? label}
                        </TabsTrigger>
                    ))}
                </TabsList>
                <hr />
                <CardContent
                    className={cn(wide ? "p-2" : "p-3", contentClassName)}>
                    <TabsContent className="py-0" value={tabValue}>
                        {loadingStatus ? (
                            <Skeleton className="h-8 w-[200px]" />
                        ) : (
                            children(parseInt(tabValue))
                        )}
                    </TabsContent>
                </CardContent>
            </Tabs>
        </Card>
    );
}

export { TabCard };
