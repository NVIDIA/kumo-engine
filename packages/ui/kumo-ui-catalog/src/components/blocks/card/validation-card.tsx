// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, CardContent } from "@/components/ui/card";
import { Error } from "@/icons";
import { cn } from "@/lib/utils";

export type ValidationItem = {
    title?: string | null;
    description: string;
};

export type ValidationCardProps = {
    data: ValidationItem[];
    variant: "error" | "warning";
};

const VARIANT_STYLES = {
    error: {
        borderColor: "border-red-500",
        bgColor: "bg-red-50",
        iconColor: "fill-red-500"
    },
    warning: {
        borderColor: "border-amber-500",
        bgColor: "bg-amber-50",
        iconColor: "fill-amber-500"
    }
} as const;

function ValidationCard({ data, variant }: ValidationCardProps) {
    const style = VARIANT_STYLES[variant];

    return (
        <Card
            data-testid="validation-card"
            data-variant={variant}
            className={cn(
                "border-1 w-100 max-h-125 overflow-y-auto",
                style.borderColor,
                style.bgColor
            )}>
            <CardContent>
                {data?.map((d, i) => (
                    <div key={`tip-${variant}-${i}`}>
                        {d?.title && (
                            <div className="text-title-primary mb-2 flex items-center font-bold">
                                <span className="pr-2">
                                    <Error
                                        className={cn(
                                            "size-4",
                                            style.iconColor
                                        )}
                                    />
                                </span>
                                {d.title}
                            </div>
                        )}
                        <div className="text-title-primary font-normal">
                            {d.description
                                ?.split("\n")
                                .map((line, index) => (
                                    <div key={index}>{line}</div>
                                ))}
                        </div>
                    </div>
                ))}
            </CardContent>
        </Card>
    );
}

export { ValidationCard };
