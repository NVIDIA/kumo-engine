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

export const Spacing = () => {
    const spacingGroups = [
        {
            title: "Extra Small (0-2)",
            description: "Tiny gaps, internal padding, and tight spaces",
            values: [
                { name: "0", size: "0", px: "0px", usage: "No spacing" },
                {
                    name: "px",
                    size: "px",
                    px: "1px",
                    usage: "Borders, hair lines"
                },
                {
                    name: "0.5",
                    size: "0.5",
                    px: "2px",
                    usage: "Very tight spacing"
                },
                { name: "1", size: "1", px: "4px", usage: "Minimal spacing" },
                {
                    name: "1.5",
                    size: "1.5",
                    px: "6px",
                    usage: "Tight spacing between related items"
                },
                {
                    name: "2",
                    size: "2",
                    px: "8px",
                    usage: "Default spacing between related items"
                }
            ]
        },
        {
            title: "Small (2.5-4)",
            description: "Standard spacing between related elements",
            values: [
                {
                    name: "2.5",
                    size: "2.5",
                    px: "10px",
                    usage: "Standard spacing"
                },
                {
                    name: "3",
                    size: "3",
                    px: "12px",
                    usage: "Standard padding/margin"
                },
                {
                    name: "3.5",
                    size: "3.5",
                    px: "14px",
                    usage: "Default spacing between related elements"
                },
                {
                    name: "4",
                    size: "4",
                    px: "16px",
                    usage: "Standard spacing for related content"
                }
            ]
        },
        {
            title: "Medium (5-8)",
            description: "Separation between distinct sections",
            values: [
                {
                    name: "5",
                    size: "5",
                    px: "20px",
                    usage: "Standard padding around content"
                },
                {
                    name: "6",
                    size: "6",
                    px: "24px",
                    usage: "Padding in cards, spacing between components"
                },
                {
                    name: "7",
                    size: "7",
                    px: "28px",
                    usage: "Section separation"
                },
                {
                    name: "8",
                    size: "8",
                    px: "32px",
                    usage: "Large component separation"
                }
            ]
        },
        {
            title: "Large (10-16)",
            description: "Major section separation",
            values: [
                {
                    name: "10",
                    size: "10",
                    px: "40px",
                    usage: "Generous spacing between sections"
                },
                {
                    name: "12",
                    size: "12",
                    px: "48px",
                    usage: "Significant section spacing"
                },
                {
                    name: "16",
                    size: "16",
                    px: "64px",
                    usage: "Major section spacing"
                }
            ]
        }
    ];

    return (
        <div className="rounded-lg bg-white p-8">
            <h1 className="mb-2 text-3xl font-bold">Spacing Tokens</h1>
            <p className="mb-8 text-gray-600">
                Consistent spacing creates rhythm and hierarchy while improving
                readability and focus.
            </p>

            {spacingGroups.map((group, index) => (
                <div key={index} className="mb-12">
                    <div className="mb-4">
                        <h2 className="text-xl font-semibold">{group.title}</h2>
                        <p className="text-gray-600">{group.description}</p>
                    </div>

                    <div className="space-y-6">
                        {group.values.map((value) => (
                            <div
                                key={value.name}
                                className="flex flex-col overflow-hidden rounded-lg border border-gray-200 md:flex-row md:items-center">
                                <div className="bg-gray-100 p-4 md:w-48">
                                    <div className="text-sm font-medium">
                                        Space {value.name}
                                    </div>
                                    <div className="text-xs text-gray-500">
                                        {value.px}
                                    </div>
                                </div>

                                <div className="flex-grow p-4">
                                    <div className="mb-3 flex items-center">
                                        <div className="h-6 bg-pink-500"></div>
                                        <div
                                            className="h-6 bg-pink-700"
                                            style={{
                                                width: parseInt(value.px) || 0
                                            }}></div>
                                        <div className="h-6 flex-grow bg-pink-200"></div>
                                    </div>
                                    <p className="text-sm text-gray-700">
                                        <span className="font-medium">
                                            Usage:{" "}
                                        </span>
                                        {value.usage}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
};
