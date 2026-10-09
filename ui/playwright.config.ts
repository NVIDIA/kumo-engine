// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, devices } from "@playwright/test";
import dotenv from "dotenv";
import { existsSync } from "fs";
import { join } from "path";

dotenv.config();

function edgeIsInstalled(): boolean {
    const candidates =
        process.platform === "darwin"
            ? ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
            : process.platform === "win32"
              ? [
                    process.env.LOCALAPPDATA
                        ? join(
                              process.env.LOCALAPPDATA,
                              "Microsoft/Edge/Application/msedge.exe"
                          )
                        : "",
                    process.env.ProgramFiles
                        ? join(
                              process.env.ProgramFiles,
                              "Microsoft/Edge/Application/msedge.exe"
                          )
                        : "",
                    process.env["ProgramFiles(x86)"]
                        ? join(
                              process.env["ProgramFiles(x86)"],
                              "Microsoft/Edge/Application/msedge.exe"
                          )
                        : ""
                ]
              : [
                    "/usr/bin/microsoft-edge",
                    "/usr/bin/microsoft-edge-stable",
                    "/opt/microsoft/msedge/msedge"
                ];

    return candidates.some(
        (candidate) => candidate !== "" && existsSync(candidate)
    );
}

const includeEdge =
    process.env.PLAYWRIGHT_ENABLE_EDGE === "1" || edgeIsInstalled();

const mutatingSpecs = [
    /.*table-refresh-readback\.spec\.ts/,
    /.*jobs-list-controls\.spec\.ts/
];

const browserProjects = [
    {
        name: "chromium",
        use: {
            ...devices["Desktop Chrome"]
        }
    },

    {
        name: "firefox",
        use: {
            ...devices["Desktop Firefox"]
        }
    },

    {
        name: "webkit",
        use: {
            ...devices["Desktop Safari"]
        }
    },

    ...(includeEdge
        ? [
              {
                  name: "edge",
                  use: {
                      ...devices["Desktop Edge"],
                      channel: "msedge"
                  }
              }
          ]
        : [])
];

const nonMutatingProjects = browserProjects.map((project) => ({
    ...project,
    testIgnore: mutatingSpecs
}));

const configuredWorkers = Number(process.env.PLAYWRIGHT_WORKERS ?? 4);
const workers =
    Number.isFinite(configuredWorkers) && configuredWorkers > 0
        ? configuredWorkers
        : 4;

export default defineConfig({
    testDir: "./playwright-tests",
    fullyParallel: true,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 1 : 0,
    workers,
    // 200s timeout: localhost startup before tests begin in CI, plus intentional waitForTimeout delays for UI stability
    timeout: process.env.CI ? 200000 : 200000,
    expect: { timeout: 15000 },
    reporter: "html",
    use: {
        baseURL: process.env.BASE_URL,
        trace: "retain-on-failure",
        screenshot: "only-on-failure",
        video: "retain-on-failure"
    },
    projects: [
        ...nonMutatingProjects,
        {
            name: "chromium-mutating",
            dependencies: nonMutatingProjects.map((project) => project.name),
            testMatch: mutatingSpecs,
            workers: 1,
            use: {
                ...devices["Desktop Chrome"]
            }
        }
    ]
});
