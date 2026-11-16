// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router";
import Spacer from "react-spacer";

import { useGetGroupsByGroupId } from "gen/v0/api/groups/groups";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import Validation from "components/Validation/Validation";

import { useUpdateKumoGroup } from "hooks/groups/groups";
import { useValidationState } from "hooks/validation/validation";

import { CustomError } from "types/common";

import { navRoutes } from "utils/Constants";

import EditGroupHeader from "./EditGroupHeader";
import EditGroupSections from "./EditGroupSections";
import { SectionSaveHandle } from "./LdapSsoGroupsSection";

const EditGroup: React.FC = () => {
    const navigate = useNavigate();
    const { groupId = "" } = useParams<{ groupId: string }>();

    const [loading, setLoading] = useState(false);
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const [groupName, setGroupName] = useState("");
    const [description, setDescription] = useState("");
    const [seededGroupId, setSeededGroupId] = useState<string | null>(null);
    const [mappingsDirty, setMappingsDirty] = useState(false);
    const [membersDirty, setMembersDirty] = useState(false);

    const mappingsRef = useRef<SectionSaveHandle>(null);
    const membersRef = useRef<SectionSaveHandle>(null);

    const { updateGroup, loading: metadataLoading } =
        useUpdateKumoGroup(handleError);
    const groupDetail = useGetGroupsByGroupId(groupId, {
        query: { enabled: Boolean(groupId) }
    });

    const group = groupDetail.data?.data;
    const detailsLoaded = groupDetail.isSuccess && Boolean(group);

    // Seed the form once per group; subsection saves refetch the group detail
    // and must not overwrite unsaved name/description edits.
    useEffect(() => {
        if (group && seededGroupId !== group.id) {
            setGroupName(group.display_name);
            setDescription(group.description ?? "");
            setSeededGroupId(group.id);
        }
    }, [group, seededGroupId]);

    const goBack = () => navigate(navRoutes.AdminGroups);

    const detailsDirty =
        !!group &&
        (groupName.trim() !== group.display_name ||
            description.trim() !== (group.description ?? ""));

    // The page-level Save persists every unsaved change on the page: group
    // details, staged membership edits, and staged LDAP/SSO mappings.
    const saveAll = async () => {
        if (!group) return;
        setLoading(true);
        try {
            clearErrors();
            if (detailsDirty && groupName.trim()) {
                await updateGroup(
                    group.id,
                    {
                        display_name: groupName.trim(),
                        description: description.trim()
                    },
                    () => {}
                );
            }
            await membersRef.current?.save();
            await mappingsRef.current?.save();
            await groupDetail.refetch();
        } catch (error) {
            handleError(error as CustomError);
        } finally {
            setLoading(false);
        }
    };

    const pageDirty = detailsDirty || mappingsDirty || membersDirty;
    const busy = loading || metadataLoading;

    return (
        <div className="font-sans flex h-[calc(100dvh-64px)] min-h-0 flex-col">
            <div className="shrink-0 px-1">
                <EditGroupHeader
                    displayName={group?.display_name ?? "—"}
                    onBack={goBack}
                    onSave={() => void saveAll()}
                    saveDisabled={busy || !groupName.trim() || !pageDirty}
                />
                <Spacer height={24} />
                <Validation
                    pageLoading={busy}
                    banner={banner}
                    alert={alert}
                    ctaMsgs={null}
                    dialogRef={dialogRef}
                    hideWarnings={() => {}}
                    clearErrors={clearErrors}
                />
                {banner && <Spacer height={16} />}
            </div>

            <EditGroupSections
                group={group}
                groupId={groupId}
                detailsLoaded={detailsLoaded}
                groupName={groupName}
                onGroupNameChange={setGroupName}
                description={description}
                onDescriptionChange={setDescription}
                busy={busy}
                handleError={handleError}
                clearErrors={clearErrors}
                onRefetch={() => void groupDetail.refetch()}
                mappingsRef={mappingsRef}
                membersRef={membersRef}
                onMappingsDirtyChange={setMappingsDirty}
                onMembersDirtyChange={setMembersDirty}
            />
        </div>
    );
};

export default WithErrorBoundary(EditGroup);
