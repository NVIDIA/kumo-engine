// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator
} from "@/components/ui/breadcrumb";
import * as React from "react";

export interface PathBreadcrumbProps {
    paths: string[];
}

export const PathBreadcrumb: React.FC<PathBreadcrumbProps> = ({ paths }) => {
    if (!paths || paths.length === 0) {
        return null;
    }

    return (
        <Breadcrumb>
            <BreadcrumbList>
                {paths.map((path, index) => {
                    const isLast = index === paths.length - 1;

                    return (
                        <React.Fragment key={`pbc-${index}`}>
                            <BreadcrumbItem>
                                {isLast ? (
                                    <BreadcrumbPage>{path}</BreadcrumbPage>
                                ) : (
                                    <span>{path}</span>
                                )}
                            </BreadcrumbItem>
                            {!isLast && <BreadcrumbSeparator />}
                        </React.Fragment>
                    );
                })}
            </BreadcrumbList>
        </Breadcrumb>
    );
};
