// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { default as React, useEffect, useState } from "react";

import { AdminPage } from "components/Admin/AdminPage";
import DirectConnectUsersAdmin from "components/Admin/DirectConnectUsersAdmin";
import Validation from "components/Validation/Validation";

import { useAdminInfo } from "hooks/auth/admin";
import { useValidationState } from "hooks/validation/validation";

import { isDirectConnectPasswordLogin } from "utils/directConnectAuth";

const Admin: React.FC = () => {
    const [loading, setLoading] = useState<boolean>(true);
    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setLoading);

    const isPasswordLogin = isDirectConnectPasswordLogin();

    const { isLoading: isAdminInfoLoading } = useAdminInfo(handleError);

    useEffect(() => {
        if (loading && !isAdminInfoLoading) {
            setLoading(false);
        }
    }, [isAdminInfoLoading, loading]);

    return (
        <AdminPage
            title="Users"
            description="Manage workspace users and roles.">
            <Validation
                pageLoading={loading}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            <DirectConnectUsersAdmin
                isPasswordLogin={isPasswordLogin}
                handleError={handleError}
                setLoading={setLoading}
            />
        </AdminPage>
    );
};

export default Admin;
