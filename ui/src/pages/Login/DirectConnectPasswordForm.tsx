// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type FormEvent, useState } from "react";

import {
    Alert,
    AlertDescription,
    Button,
    Description,
    Input,
    Label
} from "@kumo-ai/kumo-ui-catalog";

import {
    completeDirectConnectPasswordReset,
    loginDirectConnectWithPassword
} from "hooks/auth/directConnectAuth";

import { directConnectErrorMessage } from "pages/Login/directConnectLoginErrors";

import { validateDirectConnectPassword } from "utils/directConnectAuth";

interface DirectConnectPasswordFormProps {
    checkingSession: boolean;
    onLoginComplete: () => Promise<void>;
}

const DirectConnectPasswordForm = ({
    checkingSession,
    onLoginComplete
}: DirectConnectPasswordFormProps) => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [resetToken, setResetToken] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [submitting, setSubmitting] = useState(false);
    // Independent visibility per field so revealing the temporary password
    // never leaves the new permanent password exposed after the mode switch.
    const [showLoginPassword, setShowLoginPassword] = useState(false);
    const [showNewPassword, setShowNewPassword] = useState(false);

    const newPasswordPolicyError = resetToken
        ? validateDirectConnectPassword(newPassword, email)
        : null;
    const visibleError =
        error || (resetToken && newPassword ? newPasswordPolicyError : null);
    const resetMode = Boolean(resetToken);
    const loginFieldsDisabled = submitting || checkingSession;
    const submitDisabled =
        submitting ||
        checkingSession ||
        Boolean(resetToken && newPasswordPolicyError);

    const submitPasswordLogin = async () => {
        setError(null);
        setSubmitting(true);
        try {
            const response = await loginDirectConnectWithPassword(
                email,
                password
            );
            if (response.status === "password_reset_required") {
                setResetToken(response.reset_token || null);
                setPassword("");
                return;
            }
            await onLoginComplete();
        } catch (signInError) {
            setError(directConnectErrorMessage(signInError, "Sign in failed"));
        } finally {
            setSubmitting(false);
        }
    };

    const submitPasswordReset = async () => {
        if (!resetToken) return;
        if (newPasswordPolicyError) {
            setError(newPasswordPolicyError);
            return;
        }
        setError(null);
        setSubmitting(true);
        try {
            await completeDirectConnectPasswordReset(resetToken, newPassword);
            await onLoginComplete();
        } catch (resetError) {
            setError(
                directConnectErrorMessage(resetError, "Password reset failed")
            );
        } finally {
            setSubmitting(false);
        }
    };

    const onPasswordFormSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!event.currentTarget.checkValidity()) {
            event.currentTarget.reportValidity();
            return;
        }
        if (resetMode) {
            void submitPasswordReset();
            return;
        }
        void submitPasswordLogin();
    };

    const passwordToggle = (shown: boolean, onToggle: () => void) => (
        <button
            type="button"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-title-secondary hover:text-title-primary"
            onClick={onToggle}>
            {shown ? "Hide" : "Show"}
        </button>
    );

    if (resetMode) {
        return (
            <form
                className="mx-auto flex w-full max-w-sm flex-col gap-5"
                onSubmit={onPasswordFormSubmit}>
                <div className="flex flex-col gap-1">
                    <h1 className="text-xl font-semibold text-title-primary">
                        Create your password
                    </h1>
                    <p className="text-sm text-title-secondary">
                        Your temporary password has been verified. Create a
                        permanent password to access your account.
                    </p>
                </div>
                {email && (
                    <p className="text-sm text-title-secondary">
                        Signing in as{" "}
                        <span className="font-medium text-title-primary">
                            {email}
                        </span>
                    </p>
                )}
                <Alert variant="warn" enableClose={false}>
                    <AlertDescription>
                        Store your password securely. Password resets require
                        administrator assistance.
                    </AlertDescription>
                </Alert>
                <div className="flex flex-col gap-1.5">
                    <Label
                        htmlFor="direct-connect-new-password"
                        className="text-sm font-semibold text-title-primary">
                        New password
                    </Label>
                    <div className="relative">
                        <Input
                            id="direct-connect-new-password"
                            name="direct-connect-new-password"
                            type={showNewPassword ? "text" : "password"}
                            autoComplete="new-password"
                            className="pr-14"
                            disabled={submitting}
                            minLength={12}
                            required
                            value={newPassword}
                            aria-invalid={Boolean(visibleError)}
                            aria-describedby={
                                visibleError ? "reset-error" : undefined
                            }
                            onChange={(event) => {
                                setError(null);
                                setNewPassword(event.target.value);
                            }}
                        />
                        {passwordToggle(showNewPassword, () =>
                            setShowNewPassword((v) => !v)
                        )}
                    </div>
                    {visibleError && (
                        <Description
                            id="reset-error"
                            role="alert"
                            variant="destructive"
                            size="sm">
                            {visibleError}
                        </Description>
                    )}
                </div>
                <Button size="sm" type="submit" disabled={submitDisabled}>
                    Set password
                </Button>
            </form>
        );
    }

    return (
        <form
            className="mx-auto flex w-full max-w-sm flex-col gap-5"
            onSubmit={onPasswordFormSubmit}>
            <h1 className="text-xl font-semibold text-title-primary">
                Sign in
            </h1>
            <div className="flex flex-col gap-1.5">
                <Label
                    htmlFor="direct-connect-email"
                    className="text-sm font-semibold text-title-primary">
                    Email Address
                </Label>
                <Input
                    id="direct-connect-email"
                    name="direct-connect-email"
                    type="email"
                    autoFocus
                    autoComplete="email"
                    disabled={loginFieldsDisabled}
                    pattern="^[^\s@]+@[^\s@]+\.[^\s@]+$"
                    placeholder="you@example.com"
                    required
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                />
            </div>
            <div className="flex flex-col gap-1.5">
                <Label
                    htmlFor="direct-connect-password"
                    className="text-sm font-semibold text-title-primary">
                    Password
                </Label>
                <div className="relative">
                    <Input
                        id="direct-connect-password"
                        name="direct-connect-password"
                        type={showLoginPassword ? "text" : "password"}
                        autoComplete="current-password"
                        className="pr-14"
                        disabled={loginFieldsDisabled}
                        placeholder="Enter your password"
                        required
                        value={password}
                        aria-invalid={Boolean(visibleError)}
                        aria-describedby={
                            visibleError ? "signin-error" : undefined
                        }
                        onChange={(event) => setPassword(event.target.value)}
                    />
                    {passwordToggle(showLoginPassword, () =>
                        setShowLoginPassword((v) => !v)
                    )}
                </div>
                {visibleError && (
                    <Description
                        id="signin-error"
                        role="alert"
                        variant="destructive"
                        size="sm">
                        {visibleError}
                    </Description>
                )}
            </div>
            <Button size="sm" type="submit" disabled={submitDisabled}>
                Sign in
            </Button>
        </form>
    );
};

export default DirectConnectPasswordForm;
