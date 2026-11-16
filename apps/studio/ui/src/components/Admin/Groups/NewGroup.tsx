// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from "react";
import Spacer from "react-spacer";

import {
    Button,
    Close,
    Dialog,
    DialogContent,
    DialogTitle,
    Input,
    Label,
    TextArea,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { ResourceSlugField } from "components/ResourceSlug";
import Validation from "components/Validation/Validation";

import { useCreateKumoGroup } from "hooks/groups/groups";
import { useValidationState } from "hooks/validation/validation";

import { resourceSlugError } from "utils/resourceSlug";

interface NewGroupProps {
    open: boolean;
    onClose: () => void;
}

const NewGroup: React.FC<NewGroupProps> = ({ open, onClose }) => {
    const [loading, setLoading] = useState(false);
    const [groupName, setGroupName] = useState("");
    const [slug, setSlug] = useState("");
    const [description, setDescription] = useState("");
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);
    const { createGroup } = useCreateKumoGroup(handleError, setLoading);

    const handleClose = () => {
        setGroupName("");
        setSlug("");
        setDescription("");
        clearErrors();
        onClose();
    };

    const handleCreate = () => {
        const displayName = groupName.trim();
        if (!displayName) return;
        void createGroup(
            {
                display_name: displayName,
                ...(slug ? { slug } : {}),
                description: description.trim() || undefined
            },
            handleClose
        );
    };

    return (
        <Dialog open={open} onOpenChange={handleClose}>
            <DialogContent className="gap-0 !max-w-175 px-7.5 !pt-5 !pb-6.25 top-1/2 flex flex-col">
                <DialogTitle>
                    <div className="flex items-center justify-between">
                        <Title
                            variant="primary-heading"
                            className="text-xl font-semibold text-title-primary">
                            New Group
                        </Title>
                        <Button
                            variant="outline"
                            className="rounded-sm p-1.5 border-none shadow-none h-auto"
                            size="sm"
                            aria-label="Close"
                            onClick={handleClose}>
                            <Close className="!fill-text-title-secondary !w-6 !h-6" />
                        </Button>
                    </div>
                </DialogTitle>
                <Spacer height={16} />
                <Validation
                    pageLoading={loading}
                    banner={banner}
                    alert={alert}
                    ctaMsgs={null}
                    dialogRef={dialogRef}
                    hideWarnings={() => {}}
                    clearErrors={clearErrors}
                />
                <div className="flex flex-col gap-2">
                    <Label className="text-sm font-medium text-title-secondary">
                        Group name
                    </Label>
                    <Input
                        value={groupName}
                        placeholder="Enter name"
                        onChange={(event) => setGroupName(event.target.value)}
                    />
                </div>
                <Spacer height={24} />
                <ResourceSlugField
                    id="new-group-slug"
                    displayName={groupName}
                    value={slug}
                    onChange={setSlug}
                />
                <Spacer height={24} />
                <div className="flex flex-col gap-2">
                    <Label className="text-sm font-medium text-title-secondary">
                        Description{" "}
                        <span className="font-normal">(optional)</span>
                    </Label>
                    <TextArea
                        value={description}
                        placeholder="Enter description"
                        onChange={(event) => setDescription(event.target.value)}
                    />
                </div>
                <div className="flex justify-end gap-2 pt-6">
                    <Button variant="outline" onClick={handleClose}>
                        Cancel
                    </Button>
                    <Button
                        disabled={
                            !groupName.trim() ||
                            Boolean(resourceSlugError(slug, groupName)) ||
                            loading
                        }
                        onClick={handleCreate}>
                        Create
                    </Button>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default WithErrorBoundary(NewGroup);
