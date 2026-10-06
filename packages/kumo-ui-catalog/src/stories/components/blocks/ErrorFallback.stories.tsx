// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { ErrorFallback } from "@/components/blocks/error-fallback";

const meta = {
    title: "Components/Blocks/ErrorFallback",
    component: ErrorFallback,
    tags: ["autodocs"],
    parameters: { layout: "centered" },
    argTypes: {
        showGoBack: { control: "boolean" },
        centered: { control: "boolean" },
        showStackTrace: { control: "boolean" },
        statusCode: { control: "number" }
    },
    args: {
        message: "An unexpected error occurred.",
        showGoBack: false,
        centered: false,
        showStackTrace: false
    }
} satisfies Meta<typeof ErrorFallback>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const WithComponentName: Story = {
    args: {
        message: "Failed to load data.",
        componentName: "DataTable"
    }
};

export const WithStatusCode: Story = {
    args: {
        message: "The requested resource could not be found.",
        statusCode: 404
    }
};

export const WithGoBack: Story = {
    args: {
        message: "You do not have permission to view this page.",
        statusCode: 403,
        showGoBack: true
    }
};

export const WithStackTrace: Story = {
    args: {
        message: "An unexpected error occurred.",
        componentName: "PredictionChart",
        showStackTrace: true,
        stack: `Error: Cannot read properties of undefined (reading 'map')
    at PredictionChart (PredictionChart.tsx:42:18)
    at renderWithHooks (react-dom.development.js:14985:18)
    at mountIndeterminateComponent (react-dom.development.js:17811:13)`
    }
};

export const Centered: Story = {
    parameters: { layout: "fullscreen" },
    args: {
        message: "Something went wrong loading this page.",
        statusCode: 500,
        centered: true,
        showGoBack: true
    }
};

export const FullDetails: Story = {
    args: {
        message: "Failed to fetch predictions.",
        componentName: "ModelDashboard",
        statusCode: 500,
        showGoBack: true,
        showStackTrace: true,
        stack: `Error: Network request failed
    at fetchPredictions (api.ts:88:11)
    at ModelDashboard (ModelDashboard.tsx:24:5)`
    }
};
