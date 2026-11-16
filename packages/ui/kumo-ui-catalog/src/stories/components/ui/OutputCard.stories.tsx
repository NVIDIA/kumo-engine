// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { OutputCard } from "@/components/ui/output-card";
import { FolderOpen } from "@/icons";
import { ModelTraining } from "@/icons/MaterialIcons/ModelTraining";
import { OutputCardContent } from "@/types/output-card";

import { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof OutputCard> = {
    title: "UI/OutputCard",
    component: OutputCard,

    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof OutputCard>;

const items: OutputCardContent[] = [
    {
        icon: <ModelTraining className="size-4.5 fill-black" />,
        text: "S3 PARQUET"
    },
    {
        icon: <FolderOpen className="size-4.5 fill-black" />,
        text: "s3://example-bucket/folder"
    }
];
export const Output_Card_Default: Story = {
    render: function DefaultFunc() {
        return (
            <div className="flex gap-4">
                <OutputCard
                    id={"test-id"}
                    headerLeft={["Predictions" /* , "Embeddings" */]}
                    headerRight={["1.33 KB" /* , "Optional" */]}
                    contentItems={items}
                />
                <OutputCard
                    id={"test-id1"}
                    headerLeft={["Embeddings"]}
                    headerRight={["159.7 KB" /* , "Optional" */]}
                    contentItems={items}
                />
            </div>
        );
    }
};
