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

export const Color = () => {
    // Pre-defined color blocks with examples
    const colorSections = [
        {
            name: "Primary Colors",
            description:
                "Our main brand colors used for primary actions, key UI elements and emphasis.",
            colors: [
                /*{
                    name: "pink-50",
                    class: "bg-pink-50",
                    textClass: "text-pink-900",
                    usage: "Backgrounds, hover states"
                },
                {
                    name: "pink-100",
                    class: "bg-pink-100",
                    textClass: "text-pink-900",
                    usage: "Subtle backgrounds, selected states"
                },
                {
                    name: "pink-200",
                    class: "bg-pink-200",
                    textClass: "text-pink-900",
                    usage: "Backgrounds, borders"
                },
                {
                    name: "pink-300",
                    class: "bg-pink-300",
                    textClass: "text-pink-900",
                    usage: "Borders, dividers"
                },
                {
                    name: "pink-400",
                    class: "bg-pink-400",
                    textClass: "text-pink-900",
                    usage: "Icons, highlights"
                },
                {
                    name: "pink-500",
                    class: "bg-pink-500",
                    textClass: "text-white",
                    usage: "Primary buttons, accents"
                },
                {
                    name: "pink-600",
                    class: "bg-pink-600",
                    textClass: "text-white",
                    usage: "Hover states, focus rings"
                },
                {
                    name: "pink-700",
                    class: "bg-pink-700",
                    textClass: "text-white",
                    usage: "Primary buttons, accents, focus rings"
                },*/
                {
                    name: "pink-700",
                    class: "bg-primary",
                    textClass: "text-white",
                    usage: "Primary buttons, accents, focus rings"
                },
                {
                    name: "white",
                    class: "bg-white",
                    textClass: "text-black",
                    usage: "Text on primary backgrounds, Sidebar, Card, Input and Dropdown backgrounds"
                },
                {
                    name: "pink-800",
                    class: "bg-pink-800",
                    textClass: "text-white",
                    usage: "Hover states"
                } /*,
                {
                    name: "pink-900",
                    class: "bg-pink-900",
                    textClass: "text-white",
                    usage: "Dark text on light backgrounds"
                }*/
            ]
        },
        {
            name: "Neutral Colors",
            description: "Used for text, backgrounds, lines, and borders.",
            colors: [
                {
                    name: "gray-50",
                    class: "bg-gray-50",
                    textClass: "text-gray-900",
                    usage: "Page backgrounds"
                },
                {
                    name: "gray-100",
                    class: "bg-gray-100",
                    textClass: "text-gray-900",
                    usage: "Card backgrounds, subtle UI"
                },
                {
                    name: "gray-200",
                    class: "bg-gray-200",
                    textClass: "text-gray-900",
                    usage: "Input backgrounds, borders"
                },
                {
                    name: "gray-300",
                    class: "bg-gray-300",
                    textClass: "text-gray-900",
                    usage: "Borders, dividers"
                },
                {
                    name: "gray-400",
                    class: "bg-gray-400",
                    textClass: "text-white",
                    usage: "Disabled text, icons"
                },
                {
                    name: "gray-500",
                    class: "bg-gray-500",
                    textClass: "text-white",
                    usage: "Placeholder text"
                },
                {
                    name: "gray-600",
                    class: "bg-gray-600",
                    textClass: "text-white",
                    usage: "Secondary text"
                },
                {
                    name: "gray-700",
                    class: "bg-gray-700",
                    textClass: "text-white",
                    usage: "Primary text on light bg"
                },
                {
                    name: "gray-800",
                    class: "bg-gray-800",
                    textClass: "text-white",
                    usage: "Headings"
                },
                {
                    name: "gray-900",
                    class: "bg-gray-900",
                    textClass: "text-white",
                    usage: "High contrast text"
                }
            ]
        },
        {
            name: "Semantic Colors",
            description: "Communicate feedback and system states.",
            colors: [
                {
                    name: "green-200",
                    class: "bg-green-200",
                    textClass: "text-white",
                    usage: "Success, done background"
                },
                {
                    name: "amber-200",
                    class: "bg-amber-200",
                    textClass: "text-black",
                    usage: "Warning, queued background"
                },
                {
                    name: "red-200",
                    class: "bg-red-200",
                    textClass: "text-white",
                    usage: "Error, destructive actions background"
                },
                {
                    name: "gray-200",
                    class: "bg-gray-200",
                    textClass: "text-white",
                    usage: "Cancelled actions background"
                },
                {
                    name: "purple-200",
                    class: "bg-purple-200",
                    textClass: "text-white",
                    usage: "Untrained background"
                },
                {
                    name: "green-500",
                    class: "bg-green-500",
                    textClass: "text-white",
                    usage: "Success, done foreground"
                },
                {
                    name: "amber-500",
                    class: "bg-amber-500",
                    textClass: "text-black",
                    usage: "Warning, queued foreground"
                },
                {
                    name: "red-500",
                    class: "bg-red-500",
                    textClass: "text-white",
                    usage: "Error, destructive actions foreground"
                },
                {
                    name: "gray-500",
                    class: "bg-gray-500",
                    textClass: "text-white",
                    usage: "Cancelled actions foreground"
                },
                {
                    name: "purple-500",
                    class: "bg-purple-500",
                    textClass: "text-white",
                    usage: "Untrained foreground"
                }
            ]
        }
    ];

    return (
        <div className="rounded-lg bg-white p-8">
            <h1 className="mb-2 text-3xl font-bold">Color Tokens</h1>
            <p className="mb-8 text-gray-600">
                Colors establish the visual tone of our product and help create
                emphasis and hierarchy.
            </p>

            {colorSections.map((section) => (
                <div key={section.name} className="mb-12">
                    <div className="mb-4">
                        <h2 className="text-xl font-semibold">
                            {section.name}
                        </h2>
                        <p className="text-gray-600">{section.description}</p>
                    </div>

                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-5">
                        {section.colors.map((color) => (
                            <div
                                key={color.name}
                                className="flex flex-col overflow-hidden rounded-md border border-gray-200 shadow-sm transition-shadow hover:shadow-md">
                                <div
                                    className={`${color.class} flex h-24 items-center justify-center`}>
                                    <span
                                        className={`${color.textClass} font-medium`}>
                                        {color.name}
                                    </span>
                                </div>
                                <div className="bg-white p-3">
                                    <div className="mb-1 text-sm text-gray-500">
                                        Usage:
                                    </div>
                                    <div className="text-sm font-medium">
                                        {color.usage}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
};
