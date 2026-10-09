// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { AdminGroupDetailResponse } from "gen/v0/types";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import GroupDetailsSection from "./GroupDetailsSection";
import GroupUsersSection from "./GroupUsersSection";
import LdapSsoGroupsSection, {
    SectionSaveHandle
} from "./LdapSsoGroupsSection";

interface EditGroupSectionsProps {
    group?: AdminGroupDetailResponse;
    groupId: string;
    detailsLoaded: boolean;
    groupName: string;
    onGroupNameChange: (value: string) => void;
    description: string;
    onDescriptionChange: (value: string) => void;
    busy: boolean;
    handleError: HandleErrorFunc;
    clearErrors: () => void;
    onRefetch: () => void;
    mappingsRef: React.Ref<SectionSaveHandle>;
    membersRef: React.Ref<SectionSaveHandle>;
    onMappingsDirtyChange: (dirty: boolean) => void;
    onMembersDirtyChange: (dirty: boolean) => void;
}

const EditGroupSections: React.FC<EditGroupSectionsProps> = ({
    group,
    groupId,
    detailsLoaded,
    groupName,
    onGroupNameChange,
    description,
    onDescriptionChange,
    busy,
    handleError,
    clearErrors,
    onRefetch,
    mappingsRef,
    membersRef,
    onMappingsDirtyChange,
    onMembersDirtyChange
}) => (
    <div className="flex-1 min-h-0 overflow-auto">
        <div className="flex flex-col gap-10 pb-6 px-1 max-w-300">
            <GroupDetailsSection
                groupName={groupName}
                onGroupNameChange={onGroupNameChange}
                slug={group?.slug ?? ""}
                description={description}
                onDescriptionChange={onDescriptionChange}
            />

            <LdapSsoGroupsSection
                ref={mappingsRef}
                onDirtyChange={onMappingsDirtyChange}
                groupId={groupId}
                initialMappings={group?.external_group_mappings ?? []}
                detailsLoaded={detailsLoaded}
                disabled={busy}
                onError={handleError as (error: CustomError) => void}
                clearErrors={clearErrors}
                onSaved={onRefetch}
            />

            <GroupUsersSection
                ref={membersRef}
                onDirtyChange={onMembersDirtyChange}
                groupId={groupId}
                members={group?.members ?? []}
                detailsLoaded={detailsLoaded}
                handleError={handleError}
            />
        </div>
    </div>
);

export default EditGroupSections;
