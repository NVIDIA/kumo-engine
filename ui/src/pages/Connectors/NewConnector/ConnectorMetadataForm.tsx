// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useLocation } from "react-router";

import { Button, TextInput, Title } from "@kumo-ai/kumo-ui-catalog";
import { updateProjectsConnectorsByConnectorId } from "gen/v0/api/connectors/connectors";

import { useProjectId } from "hooks/project/projects";

import { getProjectPath } from "utils/Constants";
import { useCustomNavigation } from "utils/navigator";

type EditConnectorState = {
    connectorID?: string;
    configs?: { name?: string };
    name?: string;
    description?: string;
};

const ConnectorMetadataForm = () => {
    const navigate = useCustomNavigation();
    const { state } = useLocation() as { state: EditConnectorState | null };
    const connectorId = state?.connectorID ?? "";
    const projectId = useProjectId() ?? "";
    const projectRoutes = getProjectPath(projectId);
    const [name, setName] = useState(state?.name ?? state?.configs?.name ?? "");
    const [description, setDescription] = useState(state?.description ?? "");
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState("");

    const submit = async () => {
        if (!connectorId || !projectId || !name.trim() || !projectRoutes)
            return;
        setSaving(true);
        setError("");
        try {
            await updateProjectsConnectorsByConnectorId(
                projectId,
                connectorId,
                { name: name.trim(), description }
            );
            navigate(projectRoutes.Connectors + connectorId, { replace: true });
        } catch (caught) {
            setError(caught instanceof Error ? caught.message : String(caught));
            setSaving(false);
        }
    };

    return (
        <div className="max-w-3xl">
            <div className="flex items-center justify-between">
                <Title variant="page-heading" className="text-3xl font-bold">
                    Edit connector
                </Title>
                <div className="flex gap-2">
                    <Button
                        variant="outline"
                        onClick={() =>
                            projectRoutes && navigate(projectRoutes.Connectors)
                        }>
                        Cancel
                    </Button>
                    <Button
                        disabled={
                            saving || !connectorId || !projectId || !name.trim()
                        }
                        onClick={() => void submit()}>
                        Save
                    </Button>
                </div>
            </div>
            <p className="mt-2 text-title-secondary">
                The referenced identity and target configuration are immutable.
                Rename the project-owned connector or update its description
                here.
            </p>
            {error && <p className="mt-3 text-destructive">{error}</p>}
            <div className="mt-6 grid max-w-md gap-4">
                <TextInput
                    id="connector-name"
                    name="connector-name"
                    label="Connector name"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                />
                <TextInput
                    id="description"
                    name="description"
                    label="Description"
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                />
            </div>
        </div>
    );
};

export default ConnectorMetadataForm;
