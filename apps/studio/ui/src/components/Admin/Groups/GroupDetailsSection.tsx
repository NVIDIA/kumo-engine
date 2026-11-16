// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import {
    cn,
    CopyToClipBoard,
    Input,
    Label,
    TextArea,
    Title
} from "@kumo-ai/kumo-ui-catalog";

// Font styling shared by the Group name input and the ID value so they match
// the Description field (NVIDIA Sans, primary text, medium weight, base size).
// text-base must be important: the catalog Input/TextArea end with md:text-sm,
// and a plain text-base is a separate utility rather than a conflicting one, so
// the breakpoint rule would win above 768px and shrink the field to 14px.
const fieldTextClassName =
    "font-medium !text-base leading-5 text-title-primary";
// The KUI Label already carries an 8px bottom margin (mb-2), matching the
// label-to-input spacing used on the Project Settings page.
const labelClassName = "text-sm font-medium text-title-secondary leading-5";

interface GroupDetailsSectionProps {
    groupName: string;
    onGroupNameChange: (value: string) => void;
    slug: string;
    description: string;
    onDescriptionChange: (value: string) => void;
}

const GroupDetailsSection: React.FC<GroupDetailsSectionProps> = ({
    groupName,
    onGroupNameChange,
    slug,
    description,
    onDescriptionChange
}) => (
    <section className="flex flex-col gap-6">
        <Title
            variant="section-header"
            className="font-bold text-title-primary">
            Group details
        </Title>
        <div className="max-w-102 w-full">
            <Label className={labelClassName}>Group name</Label>
            <Input
                value={groupName}
                placeholder="Enter name"
                className={fieldTextClassName}
                onChange={(event) => onGroupNameChange(event.target.value)}
            />
        </div>
        <div>
            <Label className={labelClassName}>ID</Label>
            <div className="flex items-center gap-2">
                <span className={cn(fieldTextClassName, "break-all")}>
                    {slug || "—"}
                </span>
                {slug ? (
                    <CopyToClipBoard
                        copiedText={slug}
                        iconSize={16}
                        tooltipText="Copied"
                    />
                ) : null}
            </div>
        </div>
        <div>
            <Label className={labelClassName}>Description</Label>
            <TextArea
                value={description}
                placeholder="Description"
                className="px-4 py-2 min-h-9 leading-5 font-medium !text-base placeholder:text-base resize-none max-w-300 text-title-primary"
                onChange={(event) => onDescriptionChange(event.target.value)}
            />
        </div>
    </section>
);

export default GroupDetailsSection;
