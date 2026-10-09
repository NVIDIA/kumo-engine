// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";

import { ThemeProvider, useTheme } from "@/components/theme-provider";
import { Button } from "@/components/ui/button";

const meta: Meta<typeof ThemeProvider> = {
    title: "UI/ThemeProvider",
    component: ThemeProvider,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof ThemeProvider>;

function ThemeDemo() {
    const { theme, setTheme } = useTheme();

    return (
        <div className="bg-background text-foreground flex flex-col gap-4 rounded-lg border p-6">
            <div className="text-sm">
                Current theme: <span className="font-semibold">{theme}</span>
            </div>
            <div className="flex gap-2">
                <Button
                    variant={theme === "light" ? "primary" : "light-outline"}
                    size="sm"
                    onClick={() => setTheme("light")}>
                    Light
                </Button>
                <Button
                    variant={theme === "dark" ? "primary" : "light-outline"}
                    size="sm"
                    onClick={() => setTheme("dark")}>
                    Dark
                </Button>
                <Button
                    variant={theme === "system" ? "primary" : "light-outline"}
                    size="sm"
                    onClick={() => setTheme("system")}>
                    System
                </Button>
            </div>
        </div>
    );
}

export const Default: Story = {
    render: () => (
        <ThemeProvider>
            <ThemeDemo />
        </ThemeProvider>
    )
};

export const LightTheme: Story = {
    render: () => (
        <ThemeProvider defaultTheme="light" storageKey="storybook-light">
            <ThemeDemo />
        </ThemeProvider>
    )
};

export const DarkTheme: Story = {
    render: () => (
        <ThemeProvider defaultTheme="dark" storageKey="storybook-dark">
            <ThemeDemo />
        </ThemeProvider>
    )
};

export const SystemTheme: Story = {
    render: () => (
        <ThemeProvider defaultTheme="system" storageKey="storybook-system">
            <ThemeDemo />
        </ThemeProvider>
    )
};
