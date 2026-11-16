// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";

import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { SectionHeader } from "@/components/ui/section-header";
import { GridCol } from "@/components/blocks/layout/GridCol";
import { GridRow } from "@/components/blocks/layout/GridRow";
import { cn } from "@/lib/utils";
import type { SelectOption } from "@/types/select";

export interface DropdownCardProps {
    title: string;
    children: (selected: SelectOption) => React.ReactNode;
    options: SelectOption[];
    placeholder: string;
    className?: string;
}

function DropdownCard({
    title,
    children,
    options,
    placeholder,
    className
}: DropdownCardProps) {
    const [selectedOption, setSelectedOption] = useState<SelectOption>(
        options?.[0]
    );

    return (
        <Card className={cn("mb-0", className)}>
            <CardHeader className="px-6">
                <GridRow
                    xs={12}
                    sm={12}
                    md={12}
                    className="mb-3 flex items-center justify-between">
                    <GridCol>
                        {title && (
                            <SectionHeader
                                title={title}
                                titleSize="md"
                                titleClassName="leading-none"
                            />
                        )}
                    </GridCol>
                    <GridCol className="min-w-37.5">
                        <Select
                            id={title + "dropdown"}
                            selectedValues={[selectedOption?.value ?? ""]}
                            options={options}
                            onValuesChange={(newValue) => {
                                const next = options?.find((item) =>
                                    newValue.includes(item.value)
                                );
                                if (next) setSelectedOption(next);
                            }}
                            search={false}
                            mode="single"
                            placeholder={placeholder}
                            variant="secondary"
                            size="sm"
                            isInValid={false}
                        />
                    </GridCol>
                </GridRow>
            </CardHeader>
            <CardContent>{children(selectedOption)}</CardContent>
        </Card>
    );
}

export { DropdownCard };
