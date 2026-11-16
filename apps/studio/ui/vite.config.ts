// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import browserslistToEsbuild from "browserslist-to-esbuild";
import path from "path";
import process from "process";
import { defineConfig, PluginOption, UserConfig } from "vite";
import checker from "vite-plugin-checker";
import viteTsconfigPaths from "vite-tsconfig-paths";

import { createLocalApiProxy } from "./viteApiProxy";

const esbuildBrowserTargets = browserslistToEsbuild();
const localApiProxy = createLocalApiProxy();

// Enable hot-reload development with local kumo-ui-catalog
const uiCatalogPath = process.env.UI_CATALOG_PATH;

const uiCatalogAlias = uiCatalogPath
    ? {
          "@kumo-ai/kumo-ui-catalog/kumo-ui-catalog.css": path.resolve(
              uiCatalogPath,
              "src/kumo-ui-catalog.css"
          ),
          "@kumo-ai/kumo-ui-catalog": path.resolve(
              uiCatalogPath,
              "src/index.ts"
          ),
          // Resolve catalog's internal @/ path aliases
          "@/": path.resolve(uiCatalogPath, "src") + "/",
          // Catalog uses @xyflow/react but some deps expect reactflow
          reactflow: "@xyflow/react"
      }
    : undefined;

// Force React to resolve from this project's node_modules to prevent duplicates
const reactAlias = {
    react: path.resolve(__dirname, "node_modules/react"),
    "react-dom": path.resolve(__dirname, "node_modules/react-dom")
};

// When developing with catalog source, resolve dependencies from catalog's node_modules too
const uiCatalogResolve = uiCatalogPath
    ? {
          alias: { ...reactAlias, ...uiCatalogAlias },
          modules: [
              path.resolve(uiCatalogPath, "node_modules"),
              "node_modules"
          ],
          dedupe: ["react", "react-dom"]
      }
    : {
          alias: reactAlias,
          dedupe: ["react", "react-dom"]
      };

// Tailwind plugin is always required — processes @import "tailwindcss" from kumo-ui-catalog.css
const uiCatalogPlugins: PluginOption[] = [tailwindcss()];

export default defineConfig({
    plugins: [
        react(),
        viteTsconfigPaths(),
        ...(process.env.VITEST ? [] : [checker({ typescript: true })]),
        ...uiCatalogPlugins
    ],
    resolve: uiCatalogResolve,
    optimizeDeps: {
        include: ["react", "react-dom"],
        exclude: ["@kumo-ai/kumo-ui-catalog"],
        esbuildOptions: {
            supported: { destructuring: true }
        }
    },
    esbuild: {
        // esbuild >=0.28 refuses to emit destructuring for the oldest
        // browserslist targets (e.g. iOS Safari 11) and cannot lower it.
        // Mark it supported so it is emitted as-is, matching prior behavior.
        supported: { destructuring: true }
    },
    build: {
        target: esbuildBrowserTargets,
        sourcemap: true,
        outDir: "build"
    },
    define: {
        process: process
    },
    preview: {
        port: (process.env.REACT_APP_PORT || 3000) as number,
        allowedHosts: ["localhost"],
        proxy: localApiProxy
    },
    server: {
        host: true,
        allowedHosts: ["localhost"],
        port: (process.env.REACT_APP_PORT || 3000) as number,
        fs: uiCatalogPath ? { allow: [".", uiCatalogPath] } : undefined,
        watch: process.env.USE_POLLING
            ? { usePolling: true, interval: 1000 }
            : undefined,
        proxy: localApiProxy
    },
    test: {
        pool: "threads",
        maxWorkers: 4,
        minWorkers: 1,
        globals: true,
        exclude: ["playwright-tests/**", "**/node_modules/**"],
        environment: "happy-dom",
        setupFiles: "./src/tests/setup.ts",
        css: false,
        testTimeout: process.env.CI ? 90000 : 40000,
        hookTimeout: process.env.CI ? 90000 : 40000,
        teardownTimeout: process.env.CI ? 30000 : 5000,
        retries: 0,
        isolate: true,
        coverage: {
            provider: "istanbul",
            reportsDirectory: "./ui-coverage",
            reporter: ["text", "lcov", "html", "json"],
            include: ["src/**/*.{ts,tsx}"],
            exclude: [
                "node_modules",
                "tests",
                "**/__tests__**",
                "**/utils/PQLParsing/grammar/**",
                "**/src/gen/**",
                "**/src/components/Project/**"
            ]
        }
    }
} as UserConfig);
