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

export const BorderRadius = () => {
    const borderRadii = [
        {
            name: "rounded-none",
            value: "0px",
            usage: "Square corners, full-width elements"
        },
        {
            name: "rounded-sm",
            value: "0.125rem (2px)",
            usage: "Subtle rounding, small elements"
        },
        {
            name: "rounded",
            value: "0.25rem (4px)",
            usage: "Default rounding for most elements"
        },
        {
            name: "rounded-md",
            value: "0.375rem (6px)",
            usage: "Medium rounding for cards, buttons"
        },
        {
            name: "rounded-lg",
            value: "0.5rem (8px)",
            usage: "Large rounding for prominent elements"
        },
        {
            name: "rounded-xl",
            value: "0.75rem (12px)",
            usage: "Extra large rounding for featured elements"
        },
        {
            name: "rounded-2xl",
            value: "1rem (16px)",
            usage: "Very large rounding for special components"
        },
        {
            name: "rounded-3xl",
            value: "1.5rem (24px)",
            usage: "Extra large rounding for hero elements"
        },
        {
            name: "rounded-full",
            value: "9999px",
            usage: "Fully rounded, circles, pills"
        }
    ];

    return (
        <div className="rounded-lg bg-white p-8">
            <h1 className="mb-2 text-3xl font-bold">Border Radius Tokens</h1>
            <p className="mb-8 text-gray-600">
                Border radius defines the roundness of corners, contributing to
                the overall look and feel of the interface.
            </p>

            <div className="mb-12">
                <h2 className="mb-4 text-xl font-semibold">
                    Border Radius Scale
                </h2>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    {borderRadii.map((radius) => (
                        <div
                            key={radius.name}
                            className="overflow-hidden rounded-lg border border-gray-200 transition-shadow hover:shadow-sm">
                            <div className="border-b border-gray-200 bg-gray-50 p-3">
                                <div className="font-medium">{radius.name}</div>
                                <div className="text-xs text-gray-500">
                                    {radius.value}
                                </div>
                            </div>
                            <div className="flex justify-center p-4">
                                <div
                                    className={`h-24 w-24 border border-pink-200 bg-pink-100 ${radius.name} flex items-center justify-center`}>
                                    <span className="text-xs text-pink-800">
                                        {radius.name}
                                    </span>
                                </div>
                            </div>
                            <div className="p-3">
                                <p className="text-sm text-gray-700">
                                    <span className="font-medium">Usage: </span>
                                    {radius.usage}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
