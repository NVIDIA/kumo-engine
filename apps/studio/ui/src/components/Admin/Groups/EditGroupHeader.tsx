// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
    Button,
    ChevronRight,
    Title
} from "@kumo-ai/kumo-ui-catalog";

interface EditGroupHeaderProps {
    displayName: string;
    onBack: () => void;
    onSave: () => void;
    saveDisabled: boolean;
}

const EditGroupHeader: React.FC<EditGroupHeaderProps> = ({
    displayName,
    onBack,
    onSave,
    saveDisabled
}) => (
    <>
        <Breadcrumb className="mb-4">
            <BreadcrumbList>
                <BreadcrumbItem>
                    <BreadcrumbLink asChild>
                        <button
                            type="button"
                            className="cursor-pointer text-title-secondary hover:text-title-primary"
                            onClick={onBack}>
                            Group
                        </button>
                    </BreadcrumbLink>
                </BreadcrumbItem>
                <BreadcrumbSeparator>
                    <ChevronRight />
                </BreadcrumbSeparator>
                <BreadcrumbItem>
                    <BreadcrumbPage className="!text-title-primary">
                        {displayName}
                    </BreadcrumbPage>
                </BreadcrumbItem>
            </BreadcrumbList>
        </Breadcrumb>

        <div className="flex items-center justify-between">
            <Title
                data-testid="edit-group-page-heading"
                variant="primary-heading"
                className="font-bold text-3xl text-title-primary">
                Edit Group
            </Title>
            <div className="flex gap-2">
                <Button
                    onClick={onBack}
                    id="cancel"
                    variant="outline"
                    className="rounded-sm px-4 py-2 text-base h-9 leading-5"
                    size="sm">
                    Cancel
                </Button>
                <Button
                    onClick={onSave}
                    id="save"
                    className="rounded-sm py-2 px-4 text-base h-9 leading-5"
                    variant="primary"
                    size="sm"
                    disabled={saveDisabled}>
                    Save
                </Button>
            </div>
        </div>
    </>
);

export default EditGroupHeader;
