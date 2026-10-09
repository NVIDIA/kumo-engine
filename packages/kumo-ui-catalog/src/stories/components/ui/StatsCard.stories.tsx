// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatsCard } from "@/components/ui/stats-card";

import { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { expect, screen, userEvent, waitFor, within } from "storybook/test";

type SplitStats = {
    [key: string]: {
        "max_date": string;
        "min_date": string;
        "num_train_rows": number;
    };
};

const meta: Meta<typeof StatsCard> = {
    title: "UI/StatsCard",
    component: StatsCard,

    tags: ["autodocs"],
    argTypes: {}
};

export default meta;
type Story = StoryObj<typeof StatsCard>;

function StatsComponent({ clickable }: { clickable: boolean }) {
    const [downloading, setDownloading] = useState<boolean>(false);
    const [selected, setSelected] = useState<string>("");

    const splitKeys = [
        {
            "key": "train_table_stats",
            title: "Training",
            value_key: "train"
        },
        {
            "key": "val_table_stats",
            title: "Validation",
            value_key: "val"
        },
        {
            "key": "test_table_stats",
            title: "Holdout",
            value_key: "test"
        }
    ];
    let total = 0;

    splitKeys.forEach((item) => {
        if (splitStats) total += splitStats[item.key].num_train_rows;
    });
    return (
        <div className="flex gap-4">
            {splitKeys.map((split) => {
                if (splitStats && splitStats[split.key]) {
                    const stats = splitStats[split.key];
                    const numTrainRows = stats?.num_train_rows ?? 0;
                    const percentage = (numTrainRows / total) * 100;

                    const minDate = stats?.min_date
                        ? stats?.min_date?.split?.("T")?.[0]
                        : "";
                    const maxDate = stats?.max_date
                        ? stats?.max_date?.split?.("T")?.[0]
                        : "";
                    return (
                        <StatsCard
                            id={"query-id"}
                            onClick={() => {
                                setSelected(split.value_key);
                            }}
                            clickable={clickable}
                            isActive={selected === split.value_key}
                            title={split.title}
                            enableDownload={split.value_key === "test"}
                            downloading={downloading}
                            onDownloadClick={() => {
                                setDownloading(!downloading);
                            }}
                            percent={
                                percentage ? percentage?.toFixed(2) : undefined
                            }
                            numTrainRows={numTrainRows ?? undefined}
                            dateRange={
                                minDate && maxDate
                                    ? `${minDate} ~ ${maxDate}`
                                    : undefined
                            }
                        />
                    );
                }
            })}
        </div>
    );
}

export const Stats_Card_Default: Story = {
    render: function DefaultFunc() {
        return <StatsComponent clickable={false} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Training")).toBeVisible();
        await expect(canvas.getByText("Validation")).toBeVisible();
        await expect(canvas.getByText("Holdout")).toBeVisible();
    }
};

export const Stats_Card_Clickable: Story = {
    render: function DefaultFunc() {
        return <StatsComponent clickable={true} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const getCard = (name: string) =>
            canvas.getByText(name).closest("[id]")!;

        const cardNames = ["Training", "Validation", "Holdout"];

        for (let i = 0; i < cardNames.length; i++) {
            const card = getCard(cardNames[i]);
            await userEvent.click(card);
            await expect(card).toHaveClass("bg-pink-50");
            if (i > 0) {
                await expect(getCard(cardNames[i - 1])).not.toHaveClass(
                    "bg-pink-50"
                );
            }
        }

        const holdoutCard = getCard("Holdout");

        const downloadIcon = holdoutCard.querySelector(".cursor-pointer")!;
        await userEvent.click(downloadIcon);

        await userEvent.hover(
            holdoutCard.querySelector(
                "[data-slot='stats-card-header'] div div"
            )!
        );
        await waitFor(() => {
            expect(screen.getByRole("tooltip")).toBeVisible();
        });
        expect(screen.getByRole("tooltip")).toHaveTextContent(
            "Downloading ..."
        );
    }
};

//Sample data
const splitStats: SplitStats = {
    "train_table_stats": {
        "max_date": "2020-07-24T00:00:00",
        "min_date": "2020-02-25T00:00:00",
        "num_train_rows": 500
    },
    "val_table_stats": {
        "max_date": "2020-08-23T00:00:00",
        "min_date": "2020-07-24T00:00:00",
        "num_train_rows": 100
    },
    "test_table_stats": {
        "max_date": "2020-09-22T00:00:00",
        "min_date": "2020-08-23T00:00:00",
        "num_train_rows": 200
    }
};
