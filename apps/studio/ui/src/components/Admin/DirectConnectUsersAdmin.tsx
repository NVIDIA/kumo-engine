// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import { AddCircle, Button, DataTable } from "@kumo-ai/kumo-ui-catalog";
import type { DirectConnectUserResponse } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { useDirectConnectUserColumns } from "components/@ui/TableColumns/DirectConnectUsers";
import { AdminActionRow, AdminCollection } from "components/Admin/AdminPage";
import {
    CreateDirectConnectUserDialog,
    DirectConnectPasswordDialog,
    DirectConnectUserDraft,
    emptyDirectConnectUserDraft
} from "components/Admin/DirectConnectUserForms";

import { useDirectConnectUsers } from "hooks/auth/directConnectAdmin";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { validateDirectConnectPassword } from "utils/directConnectAuth";

interface DirectConnectUsersAdminProps {
    isPasswordLogin: boolean;
    handleError: HandleErrorFunc;
    setLoading: (isLoading: boolean) => void;
}

const DirectConnectUsersAdmin = ({
    isPasswordLogin,
    handleError,
    setLoading
}: DirectConnectUsersAdminProps) => {
    const [pendingUsers, setPendingUsers] = useState<
        Record<
            string,
            Partial<Pick<DirectConnectUserResponse, "role" | "disabled">>
        >
    >({});
    const [passwordUser, setPasswordUser] =
        useState<DirectConnectUserResponse | null>(null);
    const [passwordDraft, setPasswordDraft] = useState("");
    const [isCreateUserOpen, setIsCreateUserOpen] = useState(false);
    const [search, setSearch] = useState("");
    const [newUserDraft, setNewUserDraft] = useState<DirectConnectUserDraft>(
        emptyDirectConnectUserDraft
    );
    const {
        users: directConnectUsers,
        isLoading,
        createUser,
        updateUsers,
        setUserPassword,
        isUpdating
    } = useDirectConnectUsers(true, handleError);

    useEffect(() => {
        setLoading(isLoading || isUpdating);
    }, [isLoading, isUpdating, setLoading]);

    const directConnectRows = useMemo(
        () =>
            directConnectUsers.map((user) => ({
                ...user,
                ...pendingUsers[user.user_id]
            })),
        [directConnectUsers, pendingUsers]
    );

    const filteredDirectConnectRows = useMemo(() => {
        const normalizedSearch = search.trim().toLowerCase();
        if (!normalizedSearch) return directConnectRows;

        return directConnectRows.filter((user) =>
            [
                user.display_name,
                user.email,
                user.role,
                user.disabled ? "disabled" : "enabled"
            ].some((value) => value?.toLowerCase().includes(normalizedSearch))
        );
    }, [directConnectRows, search]);

    const updatePendingUser = useCallback(
        (
            userId: string,
            change: Partial<
                Pick<DirectConnectUserResponse, "role" | "disabled">
            >
        ) => {
            setPendingUsers((current) => ({
                ...current,
                [userId]: {
                    ...current[userId],
                    ...change
                }
            }));
        },
        []
    );

    const onSetPasswordClick = useCallback(
        (user: DirectConnectUserResponse) => {
            setIsCreateUserOpen(false);
            setPasswordUser(user);
            setPasswordDraft("");
        },
        []
    );
    const directConnectColumns = useDirectConnectUserColumns({
        isPasswordLogin,
        onUserChange: updatePendingUser,
        onSetPasswordClick
    });

    const pendingUpdates = useMemo(
        () =>
            Object.entries(pendingUsers).map(([user_id, update]) => ({
                user_id,
                ...update
            })),
        [pendingUsers]
    );

    const newUserPasswordPolicyError = isPasswordLogin
        ? validateDirectConnectPassword(
              newUserDraft.password,
              newUserDraft.email
          )
        : null;
    const passwordDraftPolicyError = passwordUser
        ? validateDirectConnectPassword(passwordDraft, passwordUser.email)
        : null;
    const visibleNewUserPasswordError =
        isPasswordLogin && newUserDraft.password
            ? newUserPasswordPolicyError
            : null;
    const visiblePasswordDraftError = passwordDraft
        ? passwordDraftPolicyError
        : null;
    const createUserDisabled =
        !newUserDraft.email.trim() ||
        Boolean(isPasswordLogin && newUserPasswordPolicyError);

    const resetNewUserDraft = () => {
        setNewUserDraft(emptyDirectConnectUserDraft);
    };

    const updateNewUserDraft = (change: Partial<DirectConnectUserDraft>) =>
        setNewUserDraft((current) => ({ ...current, ...change }));

    const createDirectConnectUser = async () => {
        if (createUserDisabled) return;
        setLoading(true);
        try {
            await createUser({
                email: newUserDraft.email,
                display_name: newUserDraft.displayName,
                role: newUserDraft.role,
                disabled: newUserDraft.disabled,
                password: isPasswordLogin ? newUserDraft.password : undefined
            });
            setIsCreateUserOpen(false);
            resetNewUserDraft();
        } catch (error) {
            handleError(error as CustomError);
        } finally {
            setLoading(false);
        }
    };

    const saveDirectConnectUserPassword = async () => {
        if (!passwordUser || passwordDraftPolicyError) return;
        setLoading(true);
        try {
            await setUserPassword(passwordUser.user_id, passwordDraft);
            setPasswordUser(null);
            setPasswordDraft("");
        } catch (error) {
            handleError(error as CustomError);
        } finally {
            setLoading(false);
        }
    };

    const saveDirectConnectUsers = async () => {
        if (!pendingUpdates.length) return;
        setLoading(true);
        try {
            await updateUsers(pendingUpdates);
            setPendingUsers({});
        } catch (error) {
            handleError(error as CustomError);
        } finally {
            setLoading(false);
        }
    };

    return (
        <AdminCollection>
            <AdminActionRow search={search} onSearchChange={setSearch}>
                <Button
                    size="sm"
                    variant="secondary"
                    className="h-9 px-4 py-2 text-base"
                    onClick={() => {
                        setPasswordUser(null);
                        setIsCreateUserOpen(true);
                    }}>
                    <AddCircle className="size-4" />
                    Add user
                </Button>
                <Button
                    size="sm"
                    variant="primary"
                    className="h-9 px-4 py-2 text-base"
                    disabled={pendingUpdates.length === 0 || isUpdating}
                    onClick={saveDirectConnectUsers}>
                    Save
                </Button>
            </AdminActionRow>
            <DataTable
                id="direct-connect-users"
                columns={directConnectColumns}
                data={filteredDirectConnectRows}
                enableGlobalFilter={false}
                isLoading={isLoading}
                disableRowSelectionHover={true}
            />
            <CreateDirectConnectUserDialog
                open={isCreateUserOpen}
                draft={newUserDraft}
                isPasswordLogin={isPasswordLogin}
                visiblePasswordError={visibleNewUserPasswordError}
                createDisabled={createUserDisabled}
                onDraftChange={updateNewUserDraft}
                onCreate={createDirectConnectUser}
                onCancel={() => {
                    setIsCreateUserOpen(false);
                    resetNewUserDraft();
                }}
            />
            <DirectConnectPasswordDialog
                open={passwordUser !== null}
                password={passwordDraft}
                visibleError={visiblePasswordDraftError}
                saveDisabled={Boolean(passwordDraftPolicyError)}
                onPasswordChange={setPasswordDraft}
                onSave={saveDirectConnectUserPassword}
                onCancel={() => {
                    setPasswordUser(null);
                    setPasswordDraft("");
                }}
            />
        </AdminCollection>
    );
};

export default WithErrorBoundary(DirectConnectUsersAdmin);
