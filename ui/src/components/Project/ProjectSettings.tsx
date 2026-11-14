// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import Spacer from "react-spacer";

import {
    Button,
    cn,
    CopyToClipBoard,
    Info,
    Label,
    PropertiesOverviewCard,
    RadioGroup,
    RadioGroupItem,
    TextArea,
    Title,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";
import { ProjectDBEntry, ProjectStatus } from "gen/v0/types";
import { useUserDetails } from "stores/userStore";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import Validation from "components/Validation/Validation";

import { useAdminInfo } from "hooks/auth/admin";
import { useKumoGroups } from "hooks/groups/groups";
import {
    useArchiveProject,
    useProject,
    useProjectId,
    useUpdateProject
} from "hooks/project/projects";
import { useAutoResizeTextarea } from "hooks/utils/useAutoResizeTextarea";
import { useValidationState } from "hooks/validation/validation";

import { DEFAULT_GROUP_ID, DEFAULT_PROJECT_ID } from "utils/Constants";
import { openProject } from "utils/navigator";
import { formatToShortLocalDateString } from "utils/timeConverter";

import ProjectNameInput from "./NameInput/ProjectNameInput";

// The shared formatter keeps this page consistent with the rest of the app
// (local time, zone labeled); the em dash is this page's placeholder for
// absent values.
const formatDate = (dateStr?: string) => {
    if (!dateStr) return "—";
    return formatToShortLocalDateString(dateStr);
};

const canArchiveProject = (
    project: ProjectDBEntry | null,
    userId: string | undefined,
    isAdmin: boolean
): boolean =>
    !!project &&
    project.id !== DEFAULT_PROJECT_ID &&
    project.status !== ProjectStatus.ARCHIVED &&
    (isAdmin || project.created_by === userId);

const ProjectSettings = () => {
    const projectId = useProjectId();
    const [loading, setLoading] = useState(false);
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);
    const { projectInfo: project, isError: projectError } = useProject(
        projectId,
        handleError
    );
    const { updateProject } = useUpdateProject(handleError, setLoading);
    const { archiveProject } = useArchiveProject(handleError, setLoading);
    const userDetails = useUserDetails();
    const { isAdmin } = useAdminInfo();
    const { groups } = useKumoGroups();
    const groupName =
        groups.find((g) => g.id === project?.group_id)?.display_name ?? "—";

    const [projectName, setProjectName] = useState("");
    const [projectNameError, setProjectNameError] = useState("");
    const [userHasEditedName, setUserHasEditedName] = useState(false);
    const [description, setDescription] = useState("");
    const navigate = useNavigate();
    const [radioValue, setRadioValue] = useState("Share");

    const { textareaRef } = useAutoResizeTextarea(description, 1232);

    useEffect(() => {
        if (project) {
            setProjectName(project.display_name ?? "");
            setDescription(project.description ?? "");
            setRadioValue(project.is_private ? "Private" : "Share");
        }
    }, [project]);

    const properties = [
        {
            name: "Group",
            value: groupName,
            fullWidth: true
        },
        {
            name: "Created",
            value: formatDate(project?.timestamps?.created_at)
        },
        {
            name: "Last updated",
            value: formatDate(project?.timestamps?.updated_at)
        }
    ];
    const onCancelClick = () => {
        navigate(-1);
    };

    const isSaveActive = useMemo(() => {
        return (
            !projectNameError &&
            (projectName !== project?.display_name ||
                description !== project?.description ||
                radioValue !== (project?.is_private ? "Private" : "Share"))
        );
    }, [
        description,
        project?.description,
        project?.display_name,
        projectName,
        projectNameError,
        radioValue,
        project?.is_private
    ]);

    const onSaveClick = () => {
        if (!projectId) return;
        updateProject(projectId, {
            name: projectName.trim(),
            description,
            is_private: radioValue === "Private"
        });
    };

    const canArchive = canArchiveProject(project, userDetails.user_id, isAdmin);

    const onArchiveClick = () => {
        clearErrors();
        dialogRef.current?.toggleDialog(
            "Archive project",
            `Archive "${project?.display_name}"? New resources and jobs can't be created in an archived project. This can't be undone.`,
            {},
            false,
            "Archive",
            "destructive"
        );
    };

    const onArchiveConfirm = async () => {
        dialogRef.current?.toggleDialog();
        if (!projectId) return;
        const archived = await archiveProject(projectId);
        if (archived) {
            openProject(navigate, DEFAULT_PROJECT_ID, DEFAULT_GROUP_ID);
        }
    };

    return (
        <>
            <div className="flex items-center justify-between">
                <Title
                    data-testid="project-settings-page-heading"
                    variant="primary-heading"
                    className="font-bold text-3xl">
                    Project Settings
                </Title>
                <div className="flex gap-2">
                    <Button
                        onClick={onCancelClick}
                        id="cancel"
                        variant="outline"
                        className="rounded-sm px-4 py-2 border-border-primary text-base text-title-secondary w-21.25 h-9 leading-5"
                        size="sm">
                        Cancel
                    </Button>
                    <Button
                        onClick={onSaveClick}
                        id="save"
                        className="rounded-sm py-2 px-4 text-base h-9 w-17.5 leading-5"
                        variant="primary"
                        size="sm"
                        disabled={loading || !isSaveActive}>
                        Save
                    </Button>
                </div>
            </div>
            <Spacer height={24} />
            <Validation
                pageLoading={loading}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                onDialogConfirm={onArchiveConfirm}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            {(projectError || banner) && <Spacer height={16} />}
            <div className="max-w-102 w-full">
                <ProjectNameInput
                    name={projectName}
                    setName={(name) => {
                        setProjectName(name);
                        setUserHasEditedName(project.display_name !== name);
                    }}
                    setError={setProjectNameError}
                    placeholder={"Enter name"}
                    showState={true}
                    shouldValidate={userHasEditedName}
                />
            </div>
            <Spacer height={24} />
            <div className="max-w-102 w-full">
                <Label className="text-sm font-medium text-title-secondary leading-5">
                    ID
                </Label>
                <div className="flex items-center gap-2">
                    <span
                        className={cn(
                            "font-medium text-base leading-5 text-title-primary",
                            "break-all"
                        )}>
                        {project?.slug || "—"}
                    </span>
                    {project?.slug ? (
                        <CopyToClipBoard
                            copiedText={project.slug}
                            iconSize={16}
                            tooltipText="Copied"
                        />
                    ) : null}
                </div>
            </div>
            <Spacer height={24} />
            <Label className="leading-5 font-medium text-title-secondary">
                {"Project description "}
            </Label>

            <TextArea
                ref={textareaRef}
                id={"description"}
                name={"description"}
                rows={1}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder={"Description"}
                className="px-4 py-2 min-h-9 leading-5 break-all font-medium !text-base placeholder:text-base placeholder:leading-5 resize-none max-w-300"
            />
            <Spacer height={24} />
            <div>
                <Label className="font-medium text-sm text-title-secondary leading-5">
                    {"Access"}
                </Label>
                <RadioGroup
                    className="gap-0 mt-2"
                    value={radioValue}
                    onValueChange={setRadioValue}>
                    <div className="flex items-center gap-1">
                        <RadioGroupItem
                            value="Private"
                            id="r1"
                            title={"Private"}
                            active={radioValue === "Private"}
                        />
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Info className="size-4 text-title-secondary cursor-pointer" />
                            </TooltipTrigger>
                            <TooltipContent
                                side="right"
                                sideOffset={5}
                                arrow={false}>
                                Only visible to you and workspace admins
                            </TooltipContent>
                        </Tooltip>
                    </div>
                    <Spacer height={16} />
                    <div className="flex items-center gap-1">
                        <RadioGroupItem
                            value="Share"
                            id="r2"
                            title={"Share with group"}
                            active={radioValue === "Share"}
                        />
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Info className="size-4 text-title-secondary cursor-pointer" />
                            </TooltipTrigger>
                            <TooltipContent
                                side="right"
                                sideOffset={5}
                                arrow={false}>
                                All members in your group can view and access
                                this project
                            </TooltipContent>
                        </Tooltip>
                    </div>
                </RadioGroup>
            </div>
            <Spacer height={24} />
            <PropertiesOverviewCard
                variant="project"
                properties={properties}
                hideTitle
                cardClassName={"px-0 pb-0"}
                titleSize="md"
            />
            {canArchive && (
                <>
                    <Spacer height={32} />
                    <div
                        data-testid="archive-project-section"
                        className="flex items-center justify-between gap-4 max-w-300 border-t border-border-primary pt-6">
                        <div>
                            <Label className="font-medium text-base text-title-primary leading-5">
                                Archive project
                            </Label>
                            <p className="text-sm text-title-secondary leading-5 mt-1">
                                Archived projects are read-only and hidden from
                                the project picker.
                            </p>
                        </div>
                        <Button
                            id="archive-project"
                            variant="destructive"
                            size="sm"
                            className="rounded-sm px-4 py-2 h-9 text-base leading-5"
                            disabled={loading}
                            onClick={onArchiveClick}>
                            Archive project
                        </Button>
                    </div>
                </>
            )}
        </>
    );
};

export default WithErrorBoundary(ProjectSettings);
