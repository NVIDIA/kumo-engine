// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export default {
    title: "Design System/Design Tokens",
    parameters: {
        layout: "fullscreen",
        docs: {
            description: {
                component:
                    "A comprehensive visual guide to the design tokens used in our Tailwind CSS system."
            }
        }
    }
};

export const Shadow = () => {
    const shadows = [
        {
            name: "shadow-sm",
            depth: "Extra Small",
            description: "Subtle shadow for low-prominence elements",
            usage: "Input fields, toggles, subtle elevation"
        },
        {
            name: "shadow",
            depth: "Small",
            description: "Default shadow for general UI elements",
            usage: "Cards, popovers, dropdown menus"
        },
        {
            name: "shadow-md",
            depth: "Medium",
            description: "Noticeable shadow for emphasized UI elements",
            usage: "Highlighted cards, active elements"
        },
        {
            name: "shadow-lg",
            depth: "Large",
            description: "Pronounced shadow for high-emphasis elements",
            usage: "Modals, interactive cards"
        },
        {
            name: "shadow-xl",
            depth: "Extra Large",
            description: "Heavy shadow for floating elements",
            usage: "Dialogs, important notifications"
        },
        {
            name: "shadow-2xl",
            depth: "XXL",
            description: "Maximum shadow depth for critical elements",
            usage: "Key UI elements, critical notifications"
        },
        {
            name: "shadow-inner",
            depth: "Inset",
            description: "Shadow applied inside an element",
            usage: "Inset form fields, pressed buttons"
        }
    ];

    return (
        <div className="rounded-lg bg-white p-8">
            <h1 className="mb-2 text-3xl font-bold">Shadow Tokens</h1>
            <p className="mb-8 text-gray-600">
                Shadows establish depth and hierarchy, helping users understand
                the relative importance of UI elements.
            </p>

            <div className="mb-12">
                <h2 className="mb-4 text-xl font-semibold">Shadow Scale</h2>
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                    {shadows.map((shadow) => (
                        <div
                            key={shadow.name}
                            className="overflow-hidden rounded-lg border border-gray-200 transition-shadow hover:shadow-md">
                            <div className="border-b border-gray-200 bg-gray-50 p-4">
                                <div className="font-medium">{shadow.name}</div>
                                <div className="text-sm text-gray-500">
                                    {shadow.depth}
                                </div>
                            </div>
                            <div className="flex items-center justify-center p-6">
                                <div
                                    className={`h-32 w-40 rounded-md bg-white ${shadow.name} flex items-center justify-center`}>
                                    <span className="text-sm text-gray-400">
                                        {shadow.name}
                                    </span>
                                </div>
                            </div>
                            <div className="bg-white p-4">
                                <p className="mb-1 text-sm text-gray-700">
                                    {shadow.description}
                                </p>
                                <p className="text-sm font-medium">
                                    Usage: {shadow.usage}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
