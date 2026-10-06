// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import {
    Button,
    EnhancedEncryption,
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
    useDebounce
} from "@kumo-ai/kumo-ui-catalog";
import { APIKeyType } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import {
    AdminActionRow,
    AdminCollection,
    AdminPage
} from "components/Admin/AdminPage";
import Validation from "components/Validation/Validation";

import { useApiKeys } from "hooks/apiKeys/apiKeys";
import { useAdminInfo } from "hooks/auth/admin";
import { useValidationState } from "hooks/validation/validation";

import { ApiKeysTable } from "./ApiKeysTable";
import { CreateAPIKey } from "./CreateApiKeyModel";
import { DeleteApiKeyDialog } from "./DeleteKeyDialog";

const tabList = [
    {
        name: "Admin API keys",
        value: APIKeyType.ADMIN
    },
    {
        name: "User API keys",
        value: APIKeyType.USER
    }
];

const ApiKeys = () => {
    const [loading, setLoading] = useState(false);
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    // A workspace admin manages both admin and user keys from this single
    // module (Admin/User toggle, owner column, admin endpoints); a non-admin
    // sees only their own keys. There is no separate admin screen.
    const { isAdmin: useAdminApiKeyManagement } = useAdminInfo(handleError);
    const [keyType, setKeyType] = useState<APIKeyType>(
        useAdminApiKeyManagement ? APIKeyType.ADMIN : APIKeyType.USER
    );
    const [searchStr, setSearchStr] = useState<string>("");
    const [keyToDelete, setKeyToDelete] = useState<string | null>(null);
    const [createOpen, setCreateOpen] = useState(false);

    const debouncedSearchStr = useDebounce(
        searchStr,
        useAdminApiKeyManagement ? 1000 : 500
    );

    useEffect(() => {
        setKeyType(
            useAdminApiKeyManagement ? APIKeyType.ADMIN : APIKeyType.USER
        );
    }, [useAdminApiKeyManagement]);

    const { apiKeys, isLoading, createKeys, deleteKeys } = useApiKeys(
        keyType,
        debouncedSearchStr,
        useAdminApiKeyManagement,
        handleError,
        setLoading
    );

    const onDeleteClick = (id: string) => setKeyToDelete(id);
    const onDeleteConfirm = () => {
        if (keyToDelete)
            deleteKeys(keyToDelete, () => {
                setKeyToDelete(null);
            });
    };

    const content = (
        <>
            <Validation
                pageLoading={loading}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            <AdminCollection>
                <AdminActionRow
                    search={searchStr}
                    onSearchChange={setSearchStr}>
                    <Button
                        id="create-api-key"
                        size="sm"
                        className="h-9 w-43.25 px-4 py-2 text-base"
                        onClick={() => setCreateOpen((prev) => !prev)}>
                        <EnhancedEncryption /> Create API key
                    </Button>
                </AdminActionRow>
                <Tabs
                    value={keyType}
                    onValueChange={(newValue: string) => {
                        setKeyType(newValue as APIKeyType);
                    }}>
                    {useAdminApiKeyManagement && (
                        <>
                            <TabsList>
                                {tabList.map((tab) => (
                                    <TabsTrigger
                                        key={`tab-${tab.value}`}
                                        className="leading-4"
                                        value={tab.value}>
                                        {tab.name}
                                    </TabsTrigger>
                                ))}
                            </TabsList>
                            <hr />
                        </>
                    )}
                    <TabsContent value={keyType}>
                        <ApiKeysTable
                            data={apiKeys}
                            searchStr={debouncedSearchStr}
                            onDeleteClick={onDeleteClick}
                            isLoading={isLoading}
                            showUserColumn={useAdminApiKeyManagement}
                        />
                    </TabsContent>
                </Tabs>
            </AdminCollection>
            <DeleteApiKeyDialog
                open={keyToDelete !== null}
                onOpenChange={() => {
                    setKeyToDelete(null);
                }}
                onConfirm={onDeleteConfirm}
            />
            <CreateAPIKey
                open={createOpen}
                toggleCreate={() => setCreateOpen((prev) => !prev)}
                createKeys={createKeys}
                isAdmin={useAdminApiKeyManagement}
                keyType={keyType}
            />
        </>
    );

    return (
        <AdminPage
            title="API Keys"
            description="Review and manage API keys for this workspace.">
            {content}
        </AdminPage>
    );
};

export default WithErrorBoundary(ApiKeys);
