// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import Spinner from "../../animations/spinner";
import AnimatedDone from "../../animations/animated-done";

type SizeType = "xs" | "sm" | "md" | "lg" | "xl" | number;

interface AnimationCardsProps {
    defaultSpinnerSize: SizeType;
    defaultDoneSize: SizeType;
    spinnerClassName?: string;
    doneClassName?: string;
}

const AnimationCards = ({
    defaultSpinnerSize = "sm",
    defaultDoneSize = "sm",
    spinnerClassName = "",
    doneClassName = ""
}: AnimationCardsProps) => {
    return (
        <div className="rounded-lg bg-white p-8">
            <h1 className="mb-2 text-3xl font-bold">Animations</h1>
            <p className="mb-8 text-gray-600">
                Animations enhance user experience by offering intuitive
                feedback, guiding user attention, and adding a sense of
                dynamism.
            </p>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <div className="flex flex-col items-center rounded-lg border border-gray-200 p-4 transition-colors hover:border-blue-200 hover:bg-blue-50">
                    <div className="mb-2 flex h-12 items-center justify-center">
                        <Spinner
                            size={defaultSpinnerSize}
                            className={spinnerClassName}
                        />
                    </div>
                    <span className="text-center text-base font-medium">
                        Loading Spinner
                    </span>
                </div>

                <div className="flex flex-col items-center rounded-lg border border-gray-200 p-4 transition-colors hover:border-blue-200 hover:bg-blue-50">
                    <div className="mb-2 flex h-12 items-center justify-center">
                        <AnimatedDone
                            size={defaultDoneSize}
                            className={doneClassName}
                        />
                    </div>
                    <span className="text-center text-base font-medium">
                        Check
                    </span>
                </div>
            </div>
        </div>
    );
};

// Storybook 8.x meta configuration
const meta: Meta<typeof AnimationCards> = {
    title: "Design System/Design Tokens",
    component: AnimationCards,
    parameters: {
        layout: "fullscreen"
    },
    // Storybook 8.x controls configuration
    argTypes: {
        defaultSpinnerSize: {
            control: { type: "select" },
            options: ["xs", "sm", "md", "lg", "xl", 16, 24, 32, 48, 64],
            description: "Default size of the spinner",
            table: {
                category: "Spinner"
            }
        },
        spinnerClassName: {
            control: { type: "text" },
            description:
                "CSS classes for the spinner (e.g., 'border-t-blue-600')",
            table: {
                category: "Spinner"
            }
        },
        defaultDoneSize: {
            control: { type: "select" },
            options: ["xs", "sm", "md", "lg", "xl", 16, 24, 32, 48, 64],
            description: "Default size of the checkmark",
            table: {
                category: "Checkmark"
            }
        },
        doneClassName: {
            control: { type: "text" },
            description:
                "CSS classes for the checkmark (e.g., 'text-green-600')",
            table: {
                category: "Checkmark"
            }
        }
    }
};

export default meta;
type Story = StoryObj<typeof AnimationCards>;

// Storybook 8.x story definition
export const Animations: Story = {
    args: {
        defaultSpinnerSize: "md",
        defaultDoneSize: "md",
        spinnerClassName: "",
        doneClassName: ""
    }
};
