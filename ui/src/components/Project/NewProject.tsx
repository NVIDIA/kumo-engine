// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import Spacer from "react-spacer";

import {
    Button,
    Dialog,
    DialogContent,
    DialogTitle,
    Info,
    Label,
    RadioGroup,
    RadioGroupItem,
    SectionHeader,
    storeToLocalStorage,
    TextArea,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { ResourceSlugField } from "components/ResourceSlug";
import Validation from "components/Validation/Validation";

import { useCreateProject } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import {
    defaultModelsPageURL,
    getProjectPath,
    storageKeys
} from "utils/Constants";
import { useCustomNavigation } from "utils/navigator";
import { resourceSlugError } from "utils/resourceSlug";

import ProjectNameInput from "./NameInput/ProjectNameInput";

export interface NewProjectProps {
    groupId: string | null;
    groupName: string | null;
    open: boolean;
    toggleDialog: (value: boolean) => void;
}

const NewProject = ({
    groupId,
    groupName,
    open,
    toggleDialog
}: NewProjectProps) => {
    const [newProjectName, setNewProjectName] = useState("");
    const [newProjectSlug, setNewProjectSlug] = useState("");
    const [newProjectDescription, setNewProjectDescription] = useState("");
    const [newProjectRadio, setNewProjectRadio] = useState("Private");
    const [newProjectError, setNewProjectError] = useState("");
    const [loading, setLoading] = useState<boolean>(false);

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const { createProject } = useCreateProject(handleError, setLoading);
    const navigate = useCustomNavigation();

    const onNewProjectCancel = () => {
        toggleDialog(false);
        resetNewProjectForm();
        window.reopenProjectList?.();
    };

    const onNewProjectSuccess = (projectId: string) => {
        toggleDialog(false);
        resetNewProjectForm();
        storeToLocalStorage(storageKeys.selectedProject, projectId);
        navigate(
            getProjectPath(projectId).ModelJobsList +
                "?" +
                defaultModelsPageURL.split("?")[1]
        );
    };

    const resetNewProjectForm = () => {
        setNewProjectName("");
        setNewProjectSlug("");
        setNewProjectDescription("");
        setNewProjectRadio("Private");
        setNewProjectError("");
        clearErrors();
    };

    return (
        <Dialog open={open} onOpenChange={toggleDialog}>
            <DialogContent className="!max-w-186 !h-auto !max-h-[90vh] overflow-y-auto !pt-5 px-7.5 !pb-6.25 top-1/2 flex flex-col">
                <DialogTitle data-testid="new-project-page-heading">
                    <SectionHeader
                        titleSize={"xl"}
                        title={"Create New Project"}
                        titleClassName="leading-7"
                    />
                </DialogTitle>
                {banner && <Spacer height={16} />}
                <Validation
                    pageLoading={loading}
                    banner={banner}
                    alert={alert}
                    ctaMsgs={null}
                    dialogRef={dialogRef}
                    hideWarnings={() => {}}
                    clearErrors={clearErrors}
                    className="overflow-y-hidden"
                />
                <Spacer height={16} />
                <div>
                    <Label className="font-medium text-sm text-title-secondary leading-5">
                        {"Group"}
                    </Label>
                    <p className="text-base font-medium text-title-primary leading-5 mt-1">
                        {groupName ?? "—"}
                    </p>
                </div>
                <Spacer height={24} />
                <div className="max-w-101.25 w-full">
                    <ProjectNameInput
                        name={newProjectName}
                        setName={setNewProjectName}
                        setError={setNewProjectError}
                        placeholder={"Enter name"}
                        showState={true}
                        shouldValidate={true}
                    />
                </div>
                <Spacer height={24} />
                <ResourceSlugField
                    id="new-project-slug"
                    displayName={newProjectName}
                    value={newProjectSlug}
                    onChange={setNewProjectSlug}
                    className="max-w-101.25 w-full"
                />
                <Spacer height={24} />
                <Label className="font-medium text-title-secondary leading-5">
                    {"Project description (optional)"}
                </Label>
                <TextArea
                    id={"new-project-description"}
                    name={"new-project-description"}
                    rows={1}
                    value={newProjectDescription}
                    onChange={(e) => setNewProjectDescription(e.target.value)}
                    placeholder={"Enter description here"}
                    className="px-4 py-2 min-h-9 leading-5 break-all font-medium !text-md placeholder:text-md placeholder:leading-5 !text-base resize-none"
                />
                <Spacer height={24} />
                <Label className="font-medium text-sm text-title-secondary leading-5 mb-0">
                    {"Access"}
                </Label>
                <Spacer height={8} />
                <RadioGroup
                    className="gap-0"
                    value={newProjectRadio}
                    onValueChange={setNewProjectRadio}>
                    <div className="flex items-center gap-1">
                        <RadioGroupItem
                            value="Private"
                            id="np-r1"
                            title={"Private"}
                            active={newProjectRadio === "Private"}
                        />
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Info className="size-4 text-title-secondary cursor-pointer" />
                            </TooltipTrigger>
                            <TooltipContent
                                side="right"
                                sideOffset={5}
                                arrow={false}
                                className="z-[999999]">
                                Only visible to you and workspace admins
                            </TooltipContent>
                        </Tooltip>
                    </div>
                    <Spacer height={16} />
                    <div className="flex items-center gap-1">
                        <RadioGroupItem
                            value="Share"
                            id="np-r2"
                            title={"Share with group"}
                            active={newProjectRadio === "Share"}
                        />
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Info className="size-4 text-title-secondary cursor-pointer" />
                            </TooltipTrigger>
                            <TooltipContent
                                side="right"
                                sideOffset={5}
                                arrow={false}
                                className="z-[999999]">
                                All members in your group can view and access
                                this project
                            </TooltipContent>
                        </Tooltip>
                    </div>
                </RadioGroup>
                <Spacer height={24} />
                <div className="flex justify-end gap-2 mt-auto">
                    <Button
                        onClick={onNewProjectCancel}
                        id="new-project-cancel"
                        variant="outline"
                        className="rounded-sm px-4 py-2 border-border-primary !text-base text-title-secondary w-21.25 h-9 leading-5"
                        size="sm">
                        Back
                    </Button>
                    <Button
                        onClick={() => {
                            createProject(
                                groupId ?? "",
                                {
                                    name: newProjectName.trim(),
                                    description:
                                        newProjectDescription.trim() ||
                                        undefined,
                                    is_private: newProjectRadio === "Private",
                                    ...(newProjectSlug
                                        ? { slug: newProjectSlug }
                                        : {})
                                },
                                onNewProjectSuccess
                            );
                        }}
                        id="new-project-create"
                        variant="primary"
                        className="rounded-sm py-2 px-4 font-medium h-9 w-21"
                        size="md"
                        disabled={
                            !!newProjectError ||
                            !!resourceSlugError(
                                newProjectSlug,
                                newProjectName
                            ) ||
                            newProjectName.trim() === "" ||
                            loading
                        }>
                        Create
                    </Button>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default WithErrorBoundary(NewProject);
