// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";
import { useNavigate } from "react-router";

import { Button, FolderOpen } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useProjectId } from "hooks/project/projects";

import { defaultModelsPageURL, getProjectPath } from "utils/Constants";

const Home = () => {
    const navigate = useNavigate();
    const selectedProject = useProjectId();

    // Returning users (or 404 redirects) land here with a stored project —
    // send them straight to their project instead of showing an empty state.
    useEffect(() => {
        if (selectedProject) {
            navigate(
                getProjectPath(selectedProject).ModelJobsList +
                    "?" +
                    defaultModelsPageURL.split("?")[1],
                { replace: true }
            );
        }
    }, [selectedProject, navigate]);

    if (selectedProject) return null;

    return (
        <div className="flex flex-col items-center justify-center h-full gap-4">
            <FolderOpen className="w-10 h-10 fill-text-title-secondary" />
            <div className="text-center">
                <p className="text-base font-medium text-title-primary">
                    No project selected
                </p>
                <p className="text-sm text-title-secondary mt-1">
                    Select or create a project to get started.
                </p>
            </div>
            <Button
                variant="secondary"
                className="rounded-sm"
                onClick={() => window.reopenProjectList?.()}>
                Select a project
            </Button>
        </div>
    );
};

export default WithErrorBoundary(Home);
