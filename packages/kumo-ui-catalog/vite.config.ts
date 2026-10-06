// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";
import dts from "vite-plugin-dts";

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        react(),
        tailwindcss(),
        cssInjectedByJsPlugin(),
        dts({
            outDir: "dist",
            insertTypesEntry: true
        })
    ],
    //cssCodeSplit: true,
    build: {
        sourcemap: process.env.LEGAL_NOTICE_SOURCEMAP === "1",
        lib: {
            entry: "./src/index.ts",
            name: "kumo-ui-catalog",
            fileName: (format) => `index.${format}.js`
        },
        rollupOptions: {
            external: ["react", "react-dom"],
            output: {
                globals: {
                    react: "React",
                    "react-dom": "ReactDOM"
                }
            }
        }
    },
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
            "reactflow": "@xyflow/react"
        }
    },
    server: {
        host: true
    }
});
