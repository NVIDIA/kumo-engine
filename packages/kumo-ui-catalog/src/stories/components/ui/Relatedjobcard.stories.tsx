// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge } from "@/components/ui/badge";
import RelatedJobCard from "@/components/ui/related-job-card";
import type { Meta, StoryFn } from "@storybook/react-vite";
import { useState } from "react";

const meta: Meta<typeof RelatedJobCard> = {
    title: "UI/RelatedJobCard",
    component: RelatedJobCard,
    tags: ["autodocs"]
};

export default meta;

export const Default: StoryFn<typeof RelatedJobCard> = () => {
    const [expanded, setExpanded] = useState(false);

    const handleToggle = () => {
        setExpanded(!expanded);
    };
    return (
        <RelatedJobCard
            title="GRAPH SNAPSHOT"
            id="31as4"
            className={"flex-row"}
            badge={
                <Badge size="sm" variant="done">
                    Done
                </Badge>
            }
            expanded={expanded}
            onClick={handleToggle}>
            <div className="text-sm text-gray-600">
                Expanded content goes here...
            </div>
        </RelatedJobCard>
    );
};
