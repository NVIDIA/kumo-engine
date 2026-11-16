// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PropertyCard } from "@/components/ui/property-card";
import { BookSpark, CalendarBasic, Cube } from "@/icons";
import { Meta, StoryObj } from "@storybook/react-vite";

const meta: Meta<typeof PropertyCard> = {
    title: "UI/PropertyCard",
    component: PropertyCard,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof PropertyCard>;

const propertyList = [
    {
        icon: <CalendarBasic className="size-4" />,
        label: "Now"
    },
    {
        icon: <Cube className="size-4" />,
        label: "s3//kumo-public-datasets/hnm_integ_test/parquet"
    },
    {
        icon: <BookSpark className="size-4" />,
        label: "test"
    }
];

export const PropertyCarddefault: Story = {
    render: () => (
        <PropertyCard propertyList={propertyList} id={"Propery-card"} />
    )
};
