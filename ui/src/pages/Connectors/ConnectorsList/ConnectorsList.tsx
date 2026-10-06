// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo, useState } from "react";

import { AddCircle, Button, DeleteIcon, Edit } from "@kumo-ai/kumo-ui-catalog";

import Overview from "components/Connectors/Overview/Overview";
import EmptyEntities from "components/Empty/EmptyEntities";
import ListOverview from "components/ListOverview/ListOverview";
import Validation from "components/Validation/Validation";

import { useConnectors } from "hooks/connector/connector";
import { useProjectId } from "hooks/project/projects";
import { useValidationState } from "hooks/validation/validation";

import { CustomError, MenuItemType } from "types/common";
import { JsConnector } from "types/connectors";
import { FlowType, JSTable } from "types/tables";

import { getProjectPath, PageNames } from "utils/Constants";
import { useCustomNavigation } from "utils/navigator";

const connectorEditMenuItems = (onEditClick: () => void): MenuItemType[] => {
    return [
        {
            content: (
                <>
                    <Edit className="size-4" /> {" Edit "}
                </>
            ),
            disabled: false,
            onSelect: onEditClick,
            tooltipText: ""
        }
    ];
};

/**
 * Connectors List Component
 *
 * Fetches list of connectors and render's it in Table
 *
 */
const ConnectorsList = () => {
    const navigate = useCustomNavigation();
    const projectId = useProjectId();
    const [isPageLoading, setPageLoading] = useState(false);
    const [selectedConnector, setSelectedConnector] =
        useState<JsConnector | null>(null);
    const [connectedTables, setConnectedTables] = useState<JSTable[]>([]);

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setPageLoading);
    const projectPaths = getProjectPath(projectId);

    const {
        isDataLoading: isSystemDataLoading,
        isConnectorsPresent,
        connectorList: visibleConnectors,
        deleteConnector
    } = useConnectors(handleError, setPageLoading, true);
    const isDataLoading = isSystemDataLoading;

    const connectorInfo = useMemo(() => {
        const info = visibleConnectors.find(
            (c) => c.id === selectedConnector?.id
        );
        if (info) return info;
        else return null;
    }, [visibleConnectors, selectedConnector?.id]);

    const onDeleteClick = useCallback(() => {
        if (connectedTables && connectedTables.length > 0) {
            const tableNames = connectedTables
                .map((t) => t.name_alias)
                .join(", ");
            handleError(
                {
                    message: `You cannot delete this connector before deleting all associated tables: ${tableNames}`
                } as CustomError,
                "Cannot Delete Connector"
            );
            return;
        }
        dialogRef.current?.toggleDialog(
            "Delete Connector",
            `Are you sure you want to delete connector "${connectorInfo?.name}"? This action cannot be undone.`,
            {},
            false,
            "Delete",
            "destructive"
        );
    }, [connectedTables, dialogRef, connectorInfo?.name, handleError]);

    const onDeleteConfirm = useCallback(() => {
        dialogRef.current?.toggleDialog();
        clearErrors();
        deleteConnector(
            connectorInfo?.id as string,
            connectorInfo?.name as string,
            setSelectedConnector
        );
    }, [
        clearErrors,
        connectorInfo?.id,
        connectorInfo?.name,
        deleteConnector,
        dialogRef
    ]);

    const onEditClick = () => {
        if (!projectPaths) return;
        navigate(projectPaths.EditConnector, {
            state: {
                configs: connectorInfo?.config,
                name: connectorInfo?.name,
                flowType: FlowType.EDIT,
                connectorID: connectorInfo?.id,
                description: connectorInfo?.description,
                connectorsPresent: isConnectorsPresent
            }
        });
    };

    const editMenuItems = connectorEditMenuItems(onEditClick);
    const menuItems: MenuItemType[] = [
        ...editMenuItems,
        {
            content: (
                <>
                    <DeleteIcon className="size-4" /> {" Delete "}
                </>
            ),
            disabled: false,
            onSelect: () => {
                onDeleteClick();
            },
            tooltipText: ""
        }
    ];

    // Combine all loading state
    const isLoading = isDataLoading || isPageLoading;

    const ValidationComponent = (
        <Validation
            pageLoading={selectedConnector !== null && isLoading}
            banner={banner}
            alert={alert}
            ctaMsgs={null}
            dialogRef={dialogRef}
            onDialogConfirm={onDeleteConfirm}
            hideWarnings={() => {}}
            clearErrors={clearErrors}
        />
    );
    if (!isLoading && visibleConnectors.length === 0) {
        return (
            <EmptyEntities
                currentPage={PageNames.Connector}
                title={PageNames.Connector}
            />
        );
    }
    const shouldHideAddButton = !projectPaths;
    return (
        <>
            {selectedConnector === null && ValidationComponent}
            <ListOverview
                title={"Connectors"}
                data={visibleConnectors}
                loadingStatus={isDataLoading}
                entityPresent={true}
                buttons={
                    shouldHideAddButton
                        ? []
                        : [
                              <Button
                                  id={"create-new-connector"}
                                  onClick={() => {
                                      navigate(projectPaths.NewConnector, {
                                          state: {
                                              connectorsPresent:
                                                  isConnectorsPresent
                                          }
                                      });
                                  }}>
                                  <AddCircle /> Add connector
                              </Button>
                          ]
                }
                selectedEntity={selectedConnector}
                setSelectedEntity={(connector) => {
                    clearErrors();
                    setSelectedConnector(connector as JsConnector);
                }}
                validationComponent={ValidationComponent}
                overviewComponent={
                    <Overview
                        connectorInfo={connectorInfo}
                        actionItems={menuItems}
                        loadingStatus={false}
                        onFetchConnectedTables={setConnectedTables}
                        isProjectView={true}
                    />
                }
            />
        </>
    );
};

export default ConnectorsList;
