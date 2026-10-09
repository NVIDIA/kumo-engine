// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    Dialog,
    DialogContent,
    DialogTitle,
    FormError,
    FormField,
    Label,
    Select,
    Switch,
    TextInput
} from "@kumo-ai/kumo-ui-catalog";

export interface DirectConnectUserDraft {
    email: string;
    displayName: string;
    role: "admin" | "user";
    disabled: boolean;
    password: string;
}

export const emptyDirectConnectUserDraft: DirectConnectUserDraft = {
    email: "",
    displayName: "",
    role: "user",
    disabled: false,
    password: ""
};

const roleOptions = [
    { label: "User", value: "user" },
    { label: "Admin", value: "admin" }
];

interface CreateDirectConnectUserDialogProps {
    open: boolean;
    draft: DirectConnectUserDraft;
    isPasswordLogin: boolean;
    visiblePasswordError: string | null;
    createDisabled: boolean;
    onDraftChange: (change: Partial<DirectConnectUserDraft>) => void;
    onCreate: () => void;
    onCancel: () => void;
}

export const CreateDirectConnectUserDialog = ({
    open,
    draft,
    isPasswordLogin,
    visiblePasswordError,
    createDisabled,
    onDraftChange,
    onCreate,
    onCancel
}: CreateDirectConnectUserDialogProps) => (
    <Dialog
        open={open}
        onOpenChange={(nextOpen) => {
            if (!nextOpen) onCancel();
        }}>
        <DialogContent
            data-testid="create-direct-connect-user-dialog"
            className="top-1/2 max-w-lg"
            onInteractOutside={(event) => event.preventDefault()}>
            <form
                className="flex flex-col justify-between px-7.5 py-6"
                onSubmit={(event) => {
                    event.preventDefault();
                    onCreate();
                }}>
                <DialogTitle className="text-title-primary leading-7">
                    Add User
                </DialogTitle>
                <div className="mt-4 flex flex-col gap-4 p-0 text-title-primary outline-none">
                    <FormField className="flex flex-col gap-2">
                        <Label
                            htmlFor="direct-connect-user-email"
                            className="text-sm font-medium text-title-secondary">
                            Email
                        </Label>
                        <TextInput
                            id="direct-connect-user-email"
                            type="email"
                            value={draft.email}
                            onChange={(event) =>
                                onDraftChange({ email: event.target.value })
                            }
                            placeholder="name@example.com"
                            className="h-9 !text-base font-medium py-2 px-4"
                        />
                    </FormField>
                    <FormField className="flex flex-col gap-2">
                        <Label
                            htmlFor="direct-connect-user-display-name"
                            className="text-sm font-medium text-title-secondary">
                            Display name
                        </Label>
                        <TextInput
                            id="direct-connect-user-display-name"
                            value={draft.displayName}
                            onChange={(event) =>
                                onDraftChange({
                                    displayName: event.target.value
                                })
                            }
                            placeholder="Full name"
                            className="h-9 !text-base font-medium py-2 px-4"
                        />
                    </FormField>
                    <FormField className="flex flex-col gap-2">
                        <Label
                            htmlFor="direct-connect-user-role"
                            className="text-sm font-medium text-title-secondary">
                            Role
                        </Label>
                        <Select
                            id="direct-connect-user-role"
                            mode="single"
                            options={roleOptions}
                            search={false}
                            selectedValues={[draft.role]}
                            onValuesChange={(values) =>
                                onDraftChange({
                                    role: values[0] as "admin" | "user"
                                })
                            }
                            size="sm"
                            variant="secondary"
                        />
                    </FormField>
                    {isPasswordLogin && (
                        <FormField className="flex flex-col gap-2">
                            <Label
                                htmlFor="direct-connect-user-password"
                                className="text-sm font-medium text-title-secondary">
                                Temporary password
                            </Label>
                            <TextInput
                                id="direct-connect-user-password"
                                type="password"
                                value={draft.password}
                                onChange={(event) =>
                                    onDraftChange({
                                        password: event.target.value
                                    })
                                }
                                className="h-9 !text-base font-medium py-2 px-4"
                            />
                            {visiblePasswordError && (
                                <FormError>{visiblePasswordError}</FormError>
                            )}
                        </FormField>
                    )}
                    <div className="flex items-center gap-3">
                        <Label
                            htmlFor="new-user-enabled"
                            className="mb-0 text-sm font-medium text-title-secondary">
                            Enabled
                        </Label>
                        <Switch
                            id="new-user-enabled"
                            checked={!draft.disabled}
                            onCheckedChange={(checked) =>
                                onDraftChange({ disabled: !checked })
                            }
                        />
                    </div>
                </div>
                <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                    <Button type="button" variant="outline" onClick={onCancel}>
                        Cancel
                    </Button>
                    <Button
                        type="submit"
                        variant="primary"
                        disabled={createDisabled}>
                        Create user
                    </Button>
                </div>
            </form>
        </DialogContent>
    </Dialog>
);

interface DirectConnectPasswordDialogProps {
    open: boolean;
    password: string;
    visibleError: string | null;
    saveDisabled: boolean;
    onPasswordChange: (password: string) => void;
    onSave: () => void;
    onCancel: () => void;
}

export const DirectConnectPasswordDialog = ({
    open,
    password,
    visibleError,
    saveDisabled,
    onPasswordChange,
    onSave,
    onCancel
}: DirectConnectPasswordDialogProps) => (
    <Dialog
        open={open}
        onOpenChange={(nextOpen) => {
            if (!nextOpen) onCancel();
        }}>
        <DialogContent
            data-testid="direct-connect-password-dialog"
            className="top-1/2 max-w-lg"
            onInteractOutside={(event) => event.preventDefault()}>
            <form
                className="flex flex-col justify-between px-7.5 py-6"
                onSubmit={(event) => {
                    event.preventDefault();
                    onSave();
                }}>
                <DialogTitle className="text-title-primary leading-7">
                    Set Password
                </DialogTitle>
                <div className="mt-4 flex flex-col gap-2 p-0 text-title-primary outline-none">
                    <FormField className="flex flex-col gap-2">
                        <Label
                            htmlFor="direct-connect-password"
                            className="text-sm font-medium text-title-secondary">
                            Temporary password
                        </Label>
                        <TextInput
                            id="direct-connect-password"
                            type="password"
                            value={password}
                            onChange={(event) =>
                                onPasswordChange(event.target.value)
                            }
                            className="h-9 !text-base font-medium py-2 px-4"
                        />
                        {visibleError && <FormError>{visibleError}</FormError>}
                    </FormField>
                </div>
                <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                    <Button type="button" variant="outline" onClick={onCancel}>
                        Cancel
                    </Button>
                    <Button
                        type="submit"
                        variant="primary"
                        disabled={saveDisabled}>
                        Save password
                    </Button>
                </div>
            </form>
        </DialogContent>
    </Dialog>
);
