// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router";

import { Button, DataTable, Group } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { adminGroupsColumns } from "components/@ui/TableColumns/AdminGroups";
import {
    AdminActionRow,
    AdminCollection,
    AdminPage
} from "components/Admin/AdminPage";
import Validation from "components/Validation/Validation";

import { useKumoGroups } from "hooks/groups/groups";
import { useValidationState } from "hooks/validation/validation";

import { navRoutes } from "utils/Constants";

import NewGroup from "./NewGroup";

const Groups: React.FC = () => {
    const setLoading = useState<boolean>(true)[1];

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const navigate = useNavigate();
    const { groups, isLoading } = useKumoGroups(handleError);
    const columns = useMemo(() => adminGroupsColumns(), []);
    const [search, setSearch] = useState("");
    const [showNewGroup, setShowNewGroup] = useState(false);

    const filtered = useMemo(() => {
        if (!search) return groups ?? [];
        const lower = search.toLowerCase();
        return (groups ?? []).filter(
            (g) =>
                g.display_name?.toLowerCase().includes(lower) ||
                g.description?.toLowerCase().includes(lower)
        );
    }, [groups, search]);

    return (
        <AdminPage
            title="Groups"
            description="Manage access groups and the projects assigned to them.">
            <Validation
                pageLoading={false}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            <AdminCollection>
                <AdminActionRow search={search} onSearchChange={setSearch}>
                    <Button
                        variant="primary"
                        className="rounded-sm px-4 leading-6 py-2 h-9 text-base w-37.5"
                        onClick={() => setShowNewGroup(true)}>
                        <Group className="!size-5" />
                        New Group
                    </Button>
                </AdminActionRow>
                <DataTable
                    id="admin-groups-table"
                    columns={columns}
                    data={filtered}
                    isLoading={isLoading}
                    enableGlobalFilter={false}
                    defaultPageSize={15}
                    pageCount={Math.ceil(filtered.length / 15)}
                    onRowClick={(rowData) =>
                        navigate(`${navRoutes.AdminGroups}/${rowData.id}`)
                    }
                />
            </AdminCollection>

            <NewGroup
                open={showNewGroup}
                onClose={() => setShowNewGroup(false)}
            />
        </AdminPage>
    );
};

export default WithErrorBoundary(Groups);
