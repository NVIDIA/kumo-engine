// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useLocation } from "react-router";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { FlowType } from "types/tables";

import ConnectorMetadataForm from "./ConnectorMetadataForm";
import IdentityConnectorForm from "./IdentityConnectorForm";

/**
 * Project-owned connectors always reference a reusable identity. Credentials
 * are administered separately and never collected on the connector form.
 */
const NewConnector = () => {
    const { state } = useLocation();
    return state?.flowType === FlowType.EDIT ? (
        <ConnectorMetadataForm />
    ) : (
        <IdentityConnectorForm />
    );
};

export default WithErrorBoundary(NewConnector);
