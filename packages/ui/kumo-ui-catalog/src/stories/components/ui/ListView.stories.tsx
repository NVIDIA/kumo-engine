// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ListView } from "@/components/ui/list-view";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useCallback, useState } from "react";
import { expect, userEvent, waitFor, within } from "storybook/test";

const meta: Meta<typeof ListView> = {
    title: "UI/ListOverview",
    component: ListView,
    tags: ["autodocs"]
};

export default meta;

type Story = StoryObj<typeof ListView>;

const mockItems = [
    {
        id: "1",
        name: "Customers",
        createdAt: "Wed, Sep 25, 2024"
    },
    {
        id: "2",
        name: "Transactions",
        createdAt: "Sat, Apr 15, 2024"
    },
    {
        id: "3",
        name: "Products",
        createdAt: "Mon, Jan 1, 2024"
    },
    {
        id: "4",
        name: "Customers1",
        createdAt: "Wed, Sep 25, 2024"
    },
    {
        id: "5",
        name: "Transactions1",
        createdAt: "Sat, Apr 15, 2024"
    },
    {
        id: "6",
        name: "Products1",
        createdAt: "Mon, Jan 1, 2024"
    },
    {
        id: "7",
        name: "Customers2",
        createdAt: "Wed, Sep 25, 2024"
    },
    {
        id: "8",
        name: "Transactions2",
        createdAt: "Sat, Apr 15, 2024"
    },
    {
        id: "9",
        name: "Products2",
        createdAt: "Mon, Jan 1, 2024"
    },
    {
        id: "10",
        name: "Customers3",
        createdAt: "Wed, Sep 25, 2024"
    },
    {
        id: "11",
        name: "Transactions3",
        createdAt: "Sat, Apr 15, 2024"
    },
    {
        id: "12",
        name: "Products3",
        createdAt: "Mon, Jan 1, 2024"
    },
    {
        id: "13",
        name: "e2e_obt_user_tracking_vite",
        createdAt: "Fri, Jun 27, 2025"
    },
    {
        id: "14",
        name: "AMAZON_10K_SNOWFLAKE_TEST",
        createdAt: "Sun, Ar 10, 2022"
    },
    {
        id: "15",
        name: "swedish_cities_environment_yyyy",
        createdAt: "Sun, Ar 10, 2022"
    },
    {
        id: "16",
        name: "transactions_no_timesadassas",
        createdAt: "Sun, Ar 11, 2023"
    },
    {
        id: "18",
        name: "Test20738_llllllllllllllllllllllllllWWWWWW",
        createdAt: "Sun, Ar 11, 2023"
    }
];

const searchFiltered = [
    {
        id: "1",
        name: "Customers",
        createdAt: "Wed, Sep 25, 2024",
        dateCreated: "Wed, Sep 25, 2024"
    },
    {
        id: "2",
        name: "Transactions",
        createdAt: "Sat, Apr 15, 2024",
        dateCreated: "Sat, Apr 15, 2024"
    },
    {
        id: "3",
        name: "Products",
        createdAt: "Mon, Jan 1, 2024",
        dateCreated: "Mon, Jan 1, 2024"
    },
    {
        id: "4",
        name: "Customers1",
        createdAt: "Wed, Sep 25, 2024",
        dateCreated: "Sat, Apr 15, 2024"
    },
    {
        id: "5",
        name: "Transactions1",
        createdAt: "Sat, Apr 15, 2024",
        dateCreated: "Sun, Apr 16, 2024"
    },
    {
        id: "6",
        name: "Products1",
        createdAt: "Mon, Jan 1, 2024",
        dateCreated: "Wed, Apr 2, 2025"
    },
    {
        id: "7",
        name: "Customers2",
        createdAt: "Wed, Sep 25, 2024",
        dateCreated: "Wed, Sep 25, 2024"
    },
    {
        id: "8",
        name: "Transactions2",
        createdAt: "Sat, Apr 15, 2024",
        dateCreated: "Sat, Apr 15, 2024"
    },
    {
        id: "9",
        name: "Products2",
        createdAt: "Mon, Jan 1, 2024",
        dateCreated: "Mon, Jan 1, 2024"
    },
    {
        id: "10",
        name: "Customers3",
        createdAt: "Wed, Sep 25, 2024",
        dateCreated: "Wed, Sep 25, 2024"
    },
    {
        id: "11",
        name: "Transactions3",
        createdAt: "Sat, Apr 15, 2024",
        dateCreated: "Sat, Apr 15, 2024"
    },
    {
        id: "12",
        name: "Products3",
        createdAt: "Mon, Jan 1, 2024",
        dateCreated: "Mon, Jan 1, 2024"
    },
    {
        id: "13",
        name: "e2e_obt_user_tracking_vite",
        createdAt: "Fri, Jun 27, 2025",
        dateCreated: "Fri, Jun 27, 2025"
    },
    {
        id: "14",
        name: "AMAZON_10K_SNOWFLAKE_TEST",
        createdAt: "Sun, Ar 10, 2022",
        dateCreated: "Sun, Ar 10, 2022"
    },
    {
        id: "15",
        name: "swedish_cities_environment_yyyy",
        createdAt: "Sun, Ar 10, 2022",
        dateCreated: "Sun, Ar 10, 2022"
    },
    {
        id: "16",
        name: "Test20738_lllllllllllllllllllllllllllW",
        createdAt: "Sun, Ar 11, 2023",
        dateCreated: "Sun, Ar 6, 2024"
    },
    {
        id: "17",
        name: "transactions_no_timesadassas",
        createdAt: "Sun, Ar 11, 2023",
        dateCreated: "Sun, Ar 6, 2024"
    },
    {
        id: "18",
        name: "Test20738_llllllllllllllllllllllllllWWWWWW",
        createdAt: "Sun, Ar 11, 2023",
        dateCreated: "Sun, Ar 6, 2024"
    }
];

const ListViewExample = () => {
    const [selectedEntity, setSelectedEntity] = useState<
        (typeof mockItems)[0] | null
    >(mockItems[0]);
    const [selectedVisible, setSelectedVisible] = useState<boolean>(true);

    const handleCellClick = useCallback(
        (item: (typeof mockItems)[0]) => {
            if (selectedEntity?.id === item?.id) {
                setSelectedEntity(null);
                setSelectedVisible(false);
            } else {
                setTimeout(() => {
                    if (item) {
                        if (selectedEntity?.id !== item?.id) {
                            setSelectedEntity(item);
                            setSelectedVisible(true);
                        }
                    }
                }, 1);
            }
        },
        [selectedEntity?.id]
    );

    return (
        <ListView
            id="entity-list-overview"
            selectedEntity={selectedEntity}
            handleCellClick={handleCellClick}
            selectedVisible={selectedVisible}
            searchFiltered={searchFiltered}
            overviewComponent={
                selectedEntity ? (
                    <div className="text-base font-medium text-gray-800">
                        <h2 className="text-lg font-semibold">
                            {selectedEntity.name}
                        </h2>
                        <p className="text-sm text-gray-500">
                            Created on: {selectedEntity.createdAt}
                        </p>
                    </div>
                ) : (
                    <div>No item selected</div>
                )
            }
            emptyTableComponent={<div>No data available</div>}
            selectString="entity"
            pageHeight={500}
        />
    );
};

export const Default: Story = {
    render: () => <ListViewExample />,
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const buttons = canvas.getAllByRole("button");
        await expect(buttons.length).toBe(searchFiltered.length);

        const firstButton = canvas
            .getByTestId("entity-name-0")
            .closest("button")!;
        await expect(firstButton).toHaveAttribute("aria-pressed", "true");

        const secondButton = canvas
            .getByTestId("entity-name-1")
            .closest("button")!;
        await userEvent.click(secondButton);
        await waitFor(() => {
            expect(secondButton).toHaveAttribute("aria-pressed", "true");
            expect(firstButton).toHaveAttribute("aria-pressed", "false");
        });

        await userEvent.click(secondButton);
        await waitFor(() => {
            expect(secondButton).toHaveAttribute("aria-pressed", "false");
            expect(
                canvas.getByTestId("entity-item-not-selected")
            ).toBeVisible();
        });

        const thirdButton = canvas
            .getByTestId("entity-name-2")
            .closest("button")!;
        await userEvent.click(thirdButton);
        await waitFor(() => {
            expect(thirdButton).toHaveAttribute("aria-pressed", "true");
        });
    }
};
