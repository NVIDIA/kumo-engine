// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
import Spacer from "react-spacer";

import {
    Button,
    ChevronDown,
    cn,
    DataTable,
    Dialog,
    DialogContent,
    DialogTitle,
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuPortal,
    DropdownMenuTrigger,
    FolderOpen,
    GridCol,
    GridRow,
    Group,
    removeFromLocalStorage,
    SearchBar,
    SectionHeader,
    storeToLocalStorage,
    Tabs,
    TabsList,
    TabsTrigger,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    useLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { ProjectDBEntry } from "gen/v0/types";
import { useUserDetails } from "stores/userStore";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { ProjectListTableColumns } from "components/@ui/TableColumns/ProjectListTable";
import Validation from "components/Validation/Validation";

import { useUserGroups } from "hooks/groups/groups";
import { useProjectId, useProjectListByGroup } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import {
    defaultModelsPageURL,
    getProjectPath,
    storageKeys
} from "utils/Constants";
import { getSelectedGroup } from "utils/storage";
import { formatToShortLocalDateString } from "utils/timeConverter";

import NewProject from "./NewProject";

export interface ProjectListProps {
    open: boolean;
    toggleModal: () => void;
    /**
     * Shows the Cancel button. The dialog is controlled and has no
     * `onOpenChange`, so Cancel is its only dismiss affordance; passing
     * `false` renders the picker unclosable by the user.
     */
    enableClose: boolean;
}

const getEmptyStateMessage = (
    activeTab: "all" | "my-projects",
    selectedGroupId: string | null
) =>
    activeTab === "my-projects"
        ? "Create your first project!"
        : selectedGroupId
          ? "No projects in this group yet. Create one to get started."
          : "Select a group to see your projects";

const ProjectList = ({ open, toggleModal, enableClose }: ProjectListProps) => {
    const navigate = useNavigate();
    const userDetails = useUserDetails();
    const selectedProject = useProjectId();
    const [activeTab, setActiveTab] = useState<"all" | "my-projects">("all");
    const [selectedGroupId, setSelectedGroupId] = useState<string | null>(
        getSelectedGroup()
    );
    const [, set] = useLocalStorage(
        storageKeys.selectedGroup,
        getSelectedGroup()
    );

    const setLoading = useState<boolean>(true)[1];

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [searchStr, setSearchStr] = useState("");
    const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);
    const { groups: userGroups = {}, isLoading: isGroupsLoading } =
        useUserGroups();
    const groupIds = useMemo(() => Object.keys(userGroups), [userGroups]);

    useEffect(() => {
        if (isGroupsLoading) return;
        const storedGroupId = getSelectedGroup();

        if (storedGroupId && !userGroups[storedGroupId]) {
            removeFromLocalStorage(storageKeys.selectedProject);
            removeFromLocalStorage(storageKeys.selectedGroup);
            setSelectedGroupId(null);
        } else if (groupIds.length === 1 && !storedGroupId) {
            const onlyGroupId = groupIds[0];
            storeToLocalStorage(storageKeys.selectedGroup, onlyGroupId);
            setSelectedGroupId(onlyGroupId);
            set(onlyGroupId);
        }
    }, [groupIds, isGroupsLoading, set, userGroups]);

    const {
        data: projects,
        isLoading,
        isError
    } = useProjectListByGroup(selectedGroupId ?? "", handleError);

    const tableData = useMemo<ProjectDBEntry[]>(() => {
        const entries = projects ?? [];
        const tabFiltered =
            activeTab === "my-projects"
                ? entries.filter((p) => p?.created_by === userDetails?.user_id)
                : entries;
        const search = searchStr?.toLowerCase();
        const searchFiltered = tabFiltered?.filter(
            (p) =>
                !search ||
                p.display_name?.toLowerCase().includes(search) ||
                p.created_by_name?.toLowerCase().includes(search) ||
                p.created_by?.toLowerCase().includes(search) ||
                formatToShortLocalDateString(p.timestamps.updated_at)
                    .toLowerCase()
                    .includes(search)
        );
        const rows = searchFiltered
            .map((p) => ({
                id: p.id,
                slug: p.slug,
                display_name: p.display_name,
                created_by: p.created_by,
                created_by_name: p.created_by_name,
                description: p.description,
                is_private: p.is_private,
                status: p.status,
                updated_at: formatToShortLocalDateString(
                    p.timestamps.updated_at
                ),
                timestamps: p.timestamps,
                group_id: p.group_id
            }))
            .sort((a, b) => {
                const aName = a.display_name || "";
                const bName = b.display_name || "";
                return aName?.localeCompare(bName);
            });
        return rows;
    }, [projects, activeTab, userDetails.user_id, searchStr]);

    //Guards selection against re-entry: toggleModal flips the dialog open
    //state, so a second click (a habitual double click) would reopen it.
    //The dialog stays mounted between openings, hence the reset below.
    const selectionInFlightRef = useRef<boolean>(false);

    useEffect(() => {
        if (open) selectionInFlightRef.current = false;
    }, [open]);

    const onRowClick = useCallback(
        (rowData: object) => {
            if (rowData && !selectionInFlightRef.current) {
                selectionInFlightRef.current = true;
                toggleModal();
                const projectID = (rowData as { id: string }).id;
                storeToLocalStorage(storageKeys.selectedProject, projectID);
                storeToLocalStorage(storageKeys.selectedGroup, selectedGroupId);
                navigate(
                    getProjectPath(projectID).ModelJobsList +
                        "?" +
                        defaultModelsPageURL.split("?")[1]
                );
            }
        },
        [navigate, selectedGroupId, toggleModal]
    );

    const onGroupSelect = (groupId: string) => {
        setSelectedGroupId(groupId);
        set(groupId);
    };
    const columns = ProjectListTableColumns(
        activeTab,
        selectedProject as string
    );
    const emptyStateMessage = getEmptyStateMessage(activeTab, selectedGroupId);
    return (
        <>
            <Dialog open={open}>
                <DialogContent
                    className="gap-0 !max-w-186 !h-144.25 !pt-5 !px-7.5 !pb-6.25 top-1/2 flex flex-col"
                    data-testid={"project-details"}>
                    <DialogTitle
                        className="!font-bold"
                        data-testid="project-list-heading">
                        <GridRow className="flex justify-between items-center h-9">
                            <GridCol>
                                <SectionHeader
                                    titleSize={"xl"}
                                    title={"Select Project"}
                                    titleClassName="leading-7"
                                />
                            </GridCol>
                            <GridCol>
                                <Tooltip id={"new-proj-tooltip"}>
                                    <TooltipTrigger>
                                        <span>
                                            <Button
                                                data-testid="project-create-button"
                                                id="create-project"
                                                variant="secondary"
                                                className="rounded-sm px-4 leading-6 py-2 w-38.25 h-10 text-base hover:bg-kumo-grey-50 "
                                                onClick={() => {
                                                    setShowNewProjectDialog(
                                                        true
                                                    );
                                                    toggleModal();
                                                }}
                                                disabled={!selectedGroupId}>
                                                <FolderOpen className="!w-5 !h-5" />
                                                New project
                                            </Button>
                                        </span>
                                    </TooltipTrigger>
                                    <TooltipContent
                                        arrow={false}
                                        side={"bottom"}
                                        className="z-[9999999999999]"
                                        sideOffset={5}>
                                        {!selectedGroupId
                                            ? "Please select group"
                                            : ""}
                                    </TooltipContent>
                                </Tooltip>
                            </GridCol>
                        </GridRow>
                    </DialogTitle>
                    {isError && <Spacer height={16} />}
                    <Validation
                        pageLoading={isLoading}
                        banner={banner}
                        alert={alert}
                        ctaMsgs={null}
                        dialogRef={dialogRef}
                        hideWarnings={() => {}}
                        clearErrors={clearErrors}
                        className="overflow-y-hidden"
                    />
                    <Spacer height={16} />
                    {groupIds.length > 1 && (
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button
                                    data-testid="project-group-selector"
                                    id="project"
                                    variant="light-outline"
                                    className="justify-start text-title-primary shadow-none [&_svg]:fill-text-title-primary w-50 min-h-11">
                                    <Group
                                        className={cn(
                                            "!w-5 !h-5 shrink-0",
                                            selectedGroupId &&
                                                userGroups[selectedGroupId] &&
                                                "fill-text-title-primary"
                                        )}
                                    />
                                    <span
                                        className={cn(
                                            "truncate flex-1 text-left",
                                            (!selectedGroupId ||
                                                !userGroups[selectedGroupId]) &&
                                                "text-title-secondary"
                                        )}>
                                        {selectedGroupId
                                            ? (userGroups[selectedGroupId] ??
                                              "Select Group")
                                            : "Select Group"}
                                    </span>
                                    <ChevronDown className="!fill-text-title-secondary shrink-0 ml-1" />
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuPortal>
                                <DropdownMenuContent
                                    align="start"
                                    className="!border-border-secondary rounded-sm px-2 py-1 z-[999999]">
                                    {Object.entries(userGroups).map(
                                        ([groupId, alias]) => (
                                            <DropdownMenuItem
                                                data-testid="project-group-option"
                                                className="text-title-secondary text-base font-medium py-2"
                                                key={groupId}
                                                onClick={() =>
                                                    onGroupSelect(groupId)
                                                }>
                                                {alias}
                                            </DropdownMenuItem>
                                        )
                                    )}
                                    {Object.entries(userGroups)?.length < 1 && (
                                        <DropdownMenuItem
                                            data-testid="project-group-empty-option"
                                            className="text-title-secondary text-base font-medium py-2"
                                            key={"empty-list"}
                                            onClick={() => {}}>
                                            {"No groups"}
                                        </DropdownMenuItem>
                                    )}
                                </DropdownMenuContent>
                            </DropdownMenuPortal>
                        </DropdownMenu>
                    )}
                    <Spacer height={16} />
                    <GridRow align="center">
                        <GridCol xs={6} sm={6} md={6} lg={6}>
                            <SearchBar
                                className="input-group-merge px-4 input-group-reverse rounded-sm w-74 h-9 [&_svg]:!size-4"
                                placeholder="Search"
                                value={searchStr}
                                variant="secondary"
                                inputClassName="placeholder:text-title-secondary text-sm font-medium"
                                onChange={(
                                    e: React.ChangeEvent<HTMLInputElement>
                                ) => {
                                    setSearchStr(e?.target?.value || "");
                                }}></SearchBar>
                        </GridCol>
                    </GridRow>
                    <Spacer height={16} />
                    <Tabs
                        className=""
                        defaultValue="all"
                        onValueChange={(v) =>
                            setActiveTab(v as "all" | "my-projects")
                        }>
                        <TabsList className="gap-8 w-full">
                            <TabsTrigger
                                className="leading-4 hover:text-title-primary"
                                value="my-projects">
                                My projects
                            </TabsTrigger>
                            <TabsTrigger
                                className="leading-4 hover:text-title-primary"
                                value="all">
                                All projects
                            </TabsTrigger>
                        </TabsList>
                    </Tabs>
                    <Spacer height={16} />
                    <DataTable
                        id="projects-table"
                        columns={columns}
                        data={tableData}
                        isLoading={isLoading}
                        enableGlobalFilter={false}
                        onRowClick={onRowClick}
                        hideBodyWhenEmpty={tableData?.length === 0}
                        scrollHeight={250}
                    />
                    {tableData.length === 0 && !isLoading && (
                        <>
                            <Spacer height={64} />
                            <p className="text-center text-title-secondary text-base font-medium leading-6">
                                {emptyStateMessage}
                            </p>
                        </>
                    )}
                    <Spacer height={24} />
                    {enableClose && (
                        <div className="flex justify-end mt-auto">
                            <Button
                                onClick={() => toggleModal()}
                                id="cancel"
                                variant="outline">
                                Cancel
                            </Button>
                        </div>
                    )}
                </DialogContent>
            </Dialog>
            <NewProject
                groupId={selectedGroupId}
                groupName={selectedGroupId ? userGroups[selectedGroupId] : ""}
                open={showNewProjectDialog}
                toggleDialog={(v) => {
                    setShowNewProjectDialog(v);
                    if (v) toggleModal();
                }}
            />
        </>
    );
};

export default WithErrorBoundary(ProjectList);
