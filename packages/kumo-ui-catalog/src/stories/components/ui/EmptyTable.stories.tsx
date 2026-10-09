// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from "@/components/ui/button";
import { EmptyTable } from "@/components/ui/empty-table";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof EmptyTable> = {
    title: "UI/EmptyTable",
    component: EmptyTable,

    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof EmptyTable>;

export const EmptyTable_Default: Story = {
    render: () => (
        <EmptyTable id={"Empty-table"} label={"This is an EmptyTable label."} />
    )
};

export const EmptyTable_Loading: Story = {
    render: () => <EmptyTable id={"Empty-table"} showLoading={true} />
};

export const EmptyTable_Disable_Card: Story = {
    render: () => (
        <EmptyTable
            id={"Empty-table"}
            label={"This is an EmptyTable with 'enableCard' = false."}
            enableCard={false}
        />
    )
};

export const EmptyTable_With_Children: Story = {
    render: () => (
        <EmptyTable id={"Empty-table"} label={"Label."}>
            <Button> Child Button </Button>
        </EmptyTable>
    )
};
