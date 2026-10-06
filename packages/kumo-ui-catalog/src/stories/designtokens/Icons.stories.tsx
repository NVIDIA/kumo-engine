// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from "react";

export default {
    title: "Design System/Design Tokens",
    argTypes: {
        className: {
            control: { type: "text" },
            description:
                'The tailwind classes to be applied to the component ( Ex : "fill-red-700" )',
            defaultValue: "fill-slate-500"
        }
    }
};

interface IconProps {
    className?: string;
}

interface IconItem {
    name: string;
    Component: React.ComponentType<IconProps>;
}

interface IconModule {
    [key: string]: React.ComponentType<IconProps>;
}

// Function to load all icon components using Vite's import.meta.glob
const loadIcons = async (): Promise<IconItem[]> => {
    const icons: IconItem[] = [];

    try {
        // Use Vite's import.meta.glob without trying to specify the return type
        const iconModules = import.meta.glob("../../icons/MaterialIcons/*.tsx");

        // Load all the modules and extract their components
        const modulePromises = Object.entries(iconModules).map(
            async ([path, importModule]) => {
                try {
                    // Extract component name from the path
                    const fileName = path.split("/").pop() || "";
                    const componentName = fileName.replace(/\.tsx$/, "");

                    // Import the module and use type assertion to tell TypeScript what we expect
                    const module: IconModule =
                        (await importModule()) as IconModule;

                    // Access the named export
                    const Component = module[componentName];

                    if (Component) {
                        // Add to our array
                        icons.push({
                            name: componentName,
                            Component
                        });
                    } else {
                        console.warn(
                            `No export named ${componentName} found in ${path}`
                        );
                        // Debug which exports are available
                        console.log("Available exports:", Object.keys(module));
                    }
                } catch (err) {
                    console.error(`Error loading icon from ${path}:`, err);
                }
            }
        );

        // Wait for all imports to complete
        await Promise.all(modulePromises);

        // Sort alphabetically by name
        return icons.sort((a, b) => a.name.localeCompare(b.name));
    } catch (error) {
        console.error("Error loading icons:", error);
        return [];
    }
};

interface IconGalleryProps {
    className?: string;
}

export const Icons: React.FC<IconGalleryProps> = ({ className }) => {
    const [icons, setIcons] = useState<IconItem[]>([]);
    const [searchTerm, setSearchTerm] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    // Load icons on component mount
    useEffect(() => {
        const fetchIcons = async () => {
            setIsLoading(true);
            try {
                const loadedIcons = await loadIcons();
                console.log("loadedIcons", loadedIcons);
                setIcons(loadedIcons);
            } catch (err) {
                console.error("Failed to load icons:", err);
                setError("Failed to load icons. Check console for details.");
            } finally {
                setIsLoading(false);
            }
        };

        fetchIcons();
    }, []);

    // Filter icons based on search term
    const filteredIcons = searchTerm
        ? icons.filter((icon) =>
              icon.name.toLowerCase().includes(searchTerm.toLowerCase())
          )
        : icons;

    return (
        <div className="rounded-lg bg-white p-4">
            <h1 className="mb-2 text-3xl font-bold">Icons</h1>
            <p className="mb-8 text-gray-600">
                Icons provide immediate visual cues that improve usability while
                enhancing design aesthetics.
            </p>
            <div className="mb-6">
                {/* Search input */}
                <div className="relative mb-4">
                    <input
                        type="text"
                        className="w-full rounded-lg border border-gray-300 py-2 pl-10 pr-4"
                        placeholder="Search icons..."
                        value={searchTerm}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setSearchTerm(e.target.value)
                        }
                    />
                    <div className="absolute left-3 top-1/2 -translate-y-1/2 transform text-gray-400">
                        <svg
                            width="16"
                            height="16"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                            />
                        </svg>
                    </div>
                </div>

                <div className="text-sm text-gray-500">
                    Showing {filteredIcons.length} of {icons.length} icons
                </div>
            </div>

            {isLoading ? (
                <div className="rounded-lg bg-gray-50 py-12 text-center">
                    <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-blue-500 border-t-transparent"></div>
                    <p>Loading icons...</p>
                </div>
            ) : error ? (
                <div className="rounded-lg bg-red-50 py-12 text-center text-red-700">
                    <p>{error}</p>
                </div>
            ) : icons.length === 0 ? (
                <div className="rounded-lg bg-yellow-50 py-12 text-center text-yellow-800">
                    <p>No icons found.</p>
                </div>
            ) : filteredIcons.length === 0 ? (
                <div className="rounded-lg bg-gray-50 py-12 text-center">
                    <p>No icons match your search.</p>
                </div>
            ) : (
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
                    {filteredIcons.map(({ name, Component }) => (
                        <div
                            key={name}
                            className="flex flex-col items-center rounded-lg border border-gray-200 p-4 transition-colors hover:border-blue-200 hover:bg-blue-50">
                            <div className="mb-2 flex h-12 items-center justify-center">
                                <Component className={className} />
                            </div>
                            <span className="text-center text-sm font-medium">
                                {name}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
