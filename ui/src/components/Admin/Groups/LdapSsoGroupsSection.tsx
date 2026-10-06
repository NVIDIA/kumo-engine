// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    forwardRef,
    useEffect,
    useImperativeHandle,
    useMemo,
    useState
} from "react";

import {
    Button,
    ColumnDef,
    DataTable,
    Input,
    Label,
    Row,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { AdminExternalGroupMappingResponse } from "gen/v0/types";
import { useUpdateV1GroupsExternalMappingsByGroupId } from "gen/v1/api/groups/groups";

import { CustomError } from "types/common";

interface DraftExternalMapping {
    key: string;
    identity_provider_id: string;
    external_group_id: string;
}

interface LdapSsoGroupsSectionProps {
    groupId: string;
    initialMappings: AdminExternalGroupMappingResponse[];
    detailsLoaded: boolean;
    disabled: boolean;
    onError: (error: CustomError) => void;
    clearErrors: () => void;
    onSaved: () => void;
    onDirtyChange: (dirty: boolean) => void;
}

export interface SectionSaveHandle {
    save: () => Promise<void>;
}

// Same label treatment as the Group name field; the KUI Label already
// carries the 8px bottom margin.
const labelClassName = "text-sm font-medium text-title-secondary leading-5";
// Matches the Group name and Description fields above so every input on the
// page reads at the same size and weight (NVIDIA Sans, primary text, medium,
// base size) rather than falling back to the catalog's smaller default.
// text-base must be important to beat the catalog Input's trailing md:text-sm,
// which otherwise wins above 768px and leaves these two fields at 14px.
const fieldTextClassName =
    "font-medium !text-base leading-5 text-title-primary";

const leftAlignedHeader = (label: string) => (
    <span className="block text-left">{label}</span>
);

let nextDraftMappingId = 0;

// Original two-step behavior: "Add mapping" stages a row locally and
// "Save LDAP/SSO groups" persists the whole list (PUT replace). Only the
// presentation changed — staged and saved rows render in the same table
// styling as the Users section.
const LdapSsoGroupsSection = forwardRef<
    SectionSaveHandle,
    LdapSsoGroupsSectionProps
>(function LdapSsoGroupsSection(
    {
        groupId,
        initialMappings,
        detailsLoaded,
        disabled,
        onError,
        clearErrors,
        onSaved,
        onDirtyChange
    },
    ref
) {
    const [identityProviderId, setIdentityProviderId] = useState("");
    const [externalGroupId, setExternalGroupId] = useState("");
    const [draftMappings, setDraftMappings] = useState<DraftExternalMapping[]>(
        []
    );
    const [dirty, setDirty] = useState(false);
    const [inputError, setInputError] = useState("");

    const replaceMappings = useUpdateV1GroupsExternalMappingsByGroupId();

    // Hydrate only from a loaded group response, and keep mirroring the
    // server value until the user edits the draft — a first render with an
    // empty placeholder must never freeze the section on stale data.
    useEffect(() => {
        if (detailsLoaded && !dirty) {
            setDraftMappings(
                initialMappings.map((mapping) => ({
                    key: mapping.id,
                    identity_provider_id: mapping.identity_provider_id,
                    external_group_id: mapping.external_group_id
                }))
            );
        }
    }, [detailsLoaded, dirty, initialMappings]);

    // Reset local edits when navigating to a different group.
    useEffect(() => {
        setDirty(false);
        setInputError("");
        setIdentityProviderId("");
        setExternalGroupId("");
    }, [groupId]);

    // Let the page-level Save reflect unsaved mapping changes.
    useEffect(() => {
        onDirtyChange(dirty);
    }, [dirty, onDirtyChange]);

    const busy = disabled || !detailsLoaded || replaceMappings.isPending;

    const onAddMapping = () => {
        const provider = identityProviderId.trim();
        const external = externalGroupId.trim();
        if (!provider || !external) return;
        if (
            draftMappings.some(
                (mapping) =>
                    mapping.identity_provider_id === provider &&
                    mapping.external_group_id === external
            )
        ) {
            setInputError("That LDAP/SSO group is already listed.");
            return;
        }
        nextDraftMappingId += 1;
        setDraftMappings((current) => [
            ...current,
            {
                key: `draft-external-mapping-${nextDraftMappingId}`,
                identity_provider_id: provider,
                external_group_id: external
            }
        ]);
        setIdentityProviderId("");
        setExternalGroupId("");
        setDirty(true);
        setInputError("");
    };

    const onRemoveMapping = (key: string) => {
        setDraftMappings((current) =>
            current.filter((candidate) => candidate.key !== key)
        );
        setDirty(true);
        setInputError("");
    };

    const onSaveMappings = async () => {
        try {
            clearErrors();
            const response = await replaceMappings.mutateAsync({
                groupId,
                data: {
                    mappings: draftMappings.map((mapping) => ({
                        identity_provider_id: mapping.identity_provider_id,
                        external_group_id: mapping.external_group_id
                    }))
                }
            });
            setDraftMappings(
                (response.data.mappings ?? []).map((mapping) => ({
                    key: mapping.id,
                    identity_provider_id: mapping.identity_provider_id,
                    external_group_id: mapping.external_group_id
                }))
            );
            setDirty(false);
            setInputError("");
            onSaved();
        } catch (error) {
            onError(error as CustomError);
        }
    };

    // Same columns/tokens as the Users table: header and cell text inherit
    // the DataTable defaults (text-title-secondary, text-base, font-medium).
    const columns = useMemo<ColumnDef<DraftExternalMapping>[]>(
        () => [
            {
                header: () => leftAlignedHeader("Identity Provider ID"),
                accessorKey: "identity_provider_id",
                size: 320,
                cell: ({ row }: { row: Row<DraftExternalMapping> }) => (
                    <span className="block text-left">
                        {row.original.identity_provider_id}
                    </span>
                )
            },
            {
                header: () => leftAlignedHeader("External Group ID"),
                accessorKey: "external_group_id",
                size: 320,
                cell: ({ row }: { row: Row<DraftExternalMapping> }) => (
                    <span className="block text-left">
                        {row.original.external_group_id}
                    </span>
                )
            },
            {
                header: () => leftAlignedHeader(""),
                accessorKey: "key",
                size: 120,
                enableSorting: false,
                cell: ({ row }: { row: Row<DraftExternalMapping> }) => (
                    <div className="flex justify-start">
                        <Button
                            size="sm"
                            variant="outline"
                            disabled={busy}
                            onClick={() => onRemoveMapping(row.original.key)}>
                            Remove
                        </Button>
                    </div>
                )
            }
        ],
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [busy, groupId]
    );

    useImperativeHandle(ref, () => ({
        save: async () => {
            if (dirty) await onSaveMappings();
        }
    }));

    return (
        <section className="flex flex-col gap-6">
            <div className="flex items-center justify-between gap-3">
                <Title
                    variant="section-header"
                    className="font-bold text-title-primary">
                    LDAP/SSO groups
                </Title>
                <Button
                    size="sm"
                    variant="primary"
                    className="h-9 px-4 py-2 text-base"
                    disabled={!dirty || busy}
                    onClick={() => void onSaveMappings()}>
                    Save LDAP/SSO groups
                </Button>
            </div>
            <div className="flex items-end gap-2">
                <div className="max-w-100 w-full">
                    <Label className={labelClassName}>
                        Identity provider ID
                    </Label>
                    <Input
                        value={identityProviderId}
                        placeholder="Enter identity provider ID"
                        className={fieldTextClassName}
                        onChange={(event) =>
                            setIdentityProviderId(event.target.value)
                        }
                    />
                </div>
                <div className="max-w-100 w-full">
                    <Label className={labelClassName}>External group ID</Label>
                    <Input
                        value={externalGroupId}
                        placeholder="Enter external group ID"
                        className={fieldTextClassName}
                        onChange={(event) =>
                            setExternalGroupId(event.target.value)
                        }
                    />
                </div>
                <Button
                    size="sm"
                    variant="secondary"
                    className="h-9 px-4 py-2 text-base"
                    disabled={
                        !identityProviderId.trim() ||
                        !externalGroupId.trim() ||
                        busy
                    }
                    onClick={onAddMapping}>
                    Add mapping
                </Button>
            </div>
            {inputError ? (
                <p role="alert" className="text-sm text-destructive">
                    {inputError}
                </p>
            ) : null}
            {draftMappings.length === 0 ? (
                // Same empty-state treatment as the Users table.
                <div className="relative">
                    <DataTable
                        id="group-mappings-table"
                        columns={columns}
                        data={[]}
                        enableGlobalFilter={false}
                    />
                    <div className="text-muted-foreground pointer-events-none absolute inset-x-px bottom-px top-10 flex items-center justify-center bg-white text-base font-medium leading-5">
                        Add an LDAP/SSO group mapping
                    </div>
                </div>
            ) : (
                <DataTable
                    id="group-mappings-table"
                    columns={columns}
                    data={draftMappings}
                    enableGlobalFilter={false}
                />
            )}
        </section>
    );
});

export default LdapSsoGroupsSection;
