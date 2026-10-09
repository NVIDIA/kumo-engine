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

export const Typography = () => {
    const fontSizes = [
        {
            name: "xs",
            size: "xs",
            remValue: "0.75rem",
            pxValue: "12px",
            usage: "Fine print, captions, legal text"
        },
        {
            name: "sm",
            size: "sm",
            remValue: "0.875rem",
            pxValue: "14px",
            usage: "Secondary text, help text, metadata"
        },
        {
            name: "base",
            size: "base",
            remValue: "1rem",
            pxValue: "16px",
            usage: "Body text, default paragraph size"
        },
        {
            name: "lg",
            size: "lg",
            remValue: "1.125rem",
            pxValue: "18px",
            usage: "Enhanced body text, important info"
        },
        {
            name: "xl",
            size: "xl",
            remValue: "1.25rem",
            pxValue: "20px",
            usage: "Small headings, subheadings"
        },
        {
            name: "2xl",
            size: "2xl",
            remValue: "1.5rem",
            pxValue: "24px",
            usage: "Section headers, card titles"
        },
        {
            name: "3xl",
            size: "3xl",
            remValue: "1.875rem",
            pxValue: "30px",
            usage: "Major section headers"
        },
        {
            name: "4xl",
            size: "4xl",
            remValue: "2.25rem",
            pxValue: "36px",
            usage: "Page titles, hero headings"
        }
    ];

    const fontWeights = [
        {
            name: "light",
            weight: "300",
            usage: "Large headlines, display text"
        },
        {
            name: "normal",
            weight: "400",
            usage: "Body text, default paragraphs"
        },
        {
            name: "medium",
            weight: "500",
            usage: "Emphasis, important information"
        },
        {
            name: "semibold",
            weight: "600",
            usage: "Section headers, strong emphasis"
        },
        { name: "bold", weight: "700", usage: "Headlines, titles, buttons" }
    ];

    return (
        <div className="rounded-lg bg-white p-8">
            <h1 className="mb-2 text-3xl font-bold">Typography Tokens</h1>
            <p className="mb-8 text-gray-600">
                Typography establishes hierarchy, improves readability, and
                enhances the user experience.
            </p>

            <div className="mb-12">
                <h2 className="mb-4 text-xl font-semibold">Font Sizes</h2>
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                    {fontSizes.map((size) => (
                        <div
                            key={size.name}
                            className="flex overflow-hidden rounded-lg border border-gray-200 transition-shadow hover:shadow-md">
                            <div className="w-32 flex-shrink-0 bg-gray-100 p-4">
                                <div className="text-sm font-medium">
                                    text-{size.name}
                                </div>
                                <div className="text-xs text-gray-500">
                                    {size.remValue} / {size.pxValue}
                                </div>
                            </div>
                            <div className="flex flex-grow flex-col justify-center p-4">
                                <div className={`text-${size.size} mb-1`}>
                                    The quick brown fox
                                </div>
                                <p className="text-sm text-gray-600">
                                    {size.usage}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="mb-12">
                <h2 className="mb-4 text-xl font-semibold">Font Weights</h2>
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                    {fontWeights.map((weight) => (
                        <div
                            key={weight.name}
                            className="flex overflow-hidden rounded-lg border border-gray-200 transition-shadow hover:shadow-md">
                            <div className="w-32 flex-shrink-0 bg-gray-100 p-4">
                                <div className="text-sm font-medium">
                                    font-{weight.name}
                                </div>
                                <div className="text-xs text-gray-500">
                                    {weight.weight}
                                </div>
                            </div>
                            <div className="flex flex-grow flex-col justify-center p-4">
                                <div
                                    className={`font-${weight.name} mb-1 text-lg`}>
                                    The quick brown fox
                                </div>
                                <p className="text-sm text-gray-600">
                                    {weight.usage}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
