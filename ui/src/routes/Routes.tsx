// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { RouteObject } from "react-router";

import { PageNotFound } from "@kumo-ai/kumo-ui-catalog";

import RouteErrorBoundary from "components/@ui/RouteErrorBoundary/RouteErrorBoundary";
import ClearState from "components/@ui/State/ClearState";

import AdminRouteGuard from "routes/guards/AdminRouteGuard";
import PrivateRoute from "routes/guards/PrivateRoute";

import {
    MODEL_JOB_TYPE_OPTIONS,
    ModelTitle,
    navRoutes,
    PREDICTION_JOB_TYPE_OPTIONS,
    PredictionTitle
} from "utils/Constants";

import {
    connectorsLoader,
    graphsLoader,
    modelsLoader,
    newPredictionLoader,
    predictionsLoader,
    tablesLoader
} from "./Loaders";

type LazyModule = {
    lazy: () => Promise<{ Component: React.ComponentType }>;
    preload: () => Promise<{ default: React.ComponentType }>;
};

const createLazyModule = (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    importFn: () => Promise<{ default: React.ComponentType<any> }>
): LazyModule => {
    let modulePromise: Promise<{ default: React.ComponentType }> | null = null;

    return {
        lazy: async () => {
            modulePromise = modulePromise || importFn();
            const module = await modulePromise;
            return { Component: module.default };
        },
        preload: () => {
            modulePromise = modulePromise || importFn();
            return modulePromise;
        }
    };
};

export const Login = createLazyModule(
    () => import("pages/Login/DirectConnectLogin")
);
export const Logout = createLazyModule(
    () => import("pages/Login/DirectConnectLogout")
);
export const Admin = createLazyModule(() => import("pages/Admin/Admin"));
export const ConnectorsList = createLazyModule(
    () => import("pages/Connectors/ConnectorsList/ConnectorsList")
);
export const NewConnector = createLazyModule(
    () => import("pages/Connectors/NewConnector/NewConnector")
);

export const TablesList = createLazyModule(
    () => import("pages/Tables/TablesList/TablesList")
);
export const ConnectTables = createLazyModule(
    () => import("components/Tables/Connect/ConnectTables")
);
export const GraphsList = createLazyModule(
    () => import("pages/Graphs/GraphsList/GraphsList")
);
export const NewGraph = createLazyModule(
    () => import("pages/Graphs/NewGraph/NewGraph")
);
export const JobsList = createLazyModule(() => import("pages/Jobs/JobsList"));
export const NewTrainingJob = createLazyModule(
    () => import("pages/NewJob/Training/Training")
);
export const NewTrainingGraphSelect = createLazyModule(
    () => import("pages/NewJob/Training/TrainingGraphSelect")
);
export const NewPredictionJob = createLazyModule(
    () => import("pages/NewJob/Prediction/Prediction")
);
export const Home = createLazyModule(() => import("pages/Home/Home"));
export const ProjectSettings = createLazyModule(
    () => import("components/Project/ProjectSettings")
);
export const AdminLayout = createLazyModule(
    () => import("pages/Admin/AdminLayout")
);
export const AdminGroups = createLazyModule(
    () => import("components/Admin/Groups/Groups")
);
export const AdminGroupEdit = createLazyModule(
    () => import("components/Admin/Groups/EditGroup")
);
export const AdminIdentitySettings = createLazyModule(
    () => import("components/Admin/IdentitySettings/IdentitySettings")
);
export const AdminAddIdentity = createLazyModule(
    () => import("components/Admin/IdentitySettings/AddIdentity")
);

export const ApiKeys = createLazyModule(
    () => import("components/Admin/ApiKeys/ApiKeys")
);

export const ModelsJobsList = createLazyModule(async () => {
    const JobsListComponent = (await import("pages/Jobs/JobsList")).default;
    const ModelsComponent = () => (
        <JobsListComponent
            title={ModelTitle}
            jobTypeOptions={MODEL_JOB_TYPE_OPTIONS}
        />
    );
    return { default: ModelsComponent };
});

export const PredictionsJobsList = createLazyModule(async () => {
    const JobsListComponent = (await import("pages/Jobs/JobsList")).default;
    const PredictionsComponent = () => (
        <JobsListComponent
            title={PredictionTitle}
            jobTypeOptions={PREDICTION_JOB_TYPE_OPTIONS}
        />
    );
    return { default: PredictionsComponent };
});

export const routes: RouteObject[] = [
    // Public routes
    {
        path: "/",
        lazy: Login.lazy
    },
    {
        path: "/login",
        lazy: Login.lazy
    },
    {
        path: "/clear",
        element: <ClearState />
    },
    {
        path: "/logout",
        lazy: Logout.lazy
    },
    // Authenticated routes
    {
        element: <PrivateRoute />,
        errorElement: <RouteErrorBoundary />,
        HydrateFallback: () => null,
        children: [
            {
                path: navRoutes.Admin,
                element: <AdminRouteGuard />,
                children: [
                    {
                        lazy: AdminLayout.lazy,
                        children: [
                            {
                                path: "members",
                                lazy: Admin.lazy
                            },
                            {
                                path: "groups",
                                lazy: AdminGroups.lazy
                            },
                            {
                                path: "groups/:groupId",
                                lazy: AdminGroupEdit.lazy
                            },
                            {
                                path: "identity-settings",
                                lazy: AdminIdentitySettings.lazy
                            },
                            {
                                path: "identity-settings/add",
                                lazy: AdminAddIdentity.lazy
                            }
                        ]
                    }
                ]
            },
            {
                path: navRoutes.Home,
                lazy: Home.lazy
            },
            {
                path: navRoutes.ApiKeys,
                lazy: ApiKeys.lazy
            },
            // Project-scoped routes — all resources under a specific project
            {
                path: navRoutes.ProjectBase,
                children: [
                    {
                        path: "settings",
                        lazy: ProjectSettings.lazy
                    },
                    {
                        path: "connectors/:id?",
                        loader: connectorsLoader,
                        lazy: ConnectorsList.lazy
                    },
                    {
                        path: "connectors/new-connector",
                        lazy: NewConnector.lazy
                    },
                    {
                        path: "connectors/edit-connector",
                        lazy: NewConnector.lazy
                    },
                    {
                        path: "tables/:id?",
                        loader: tablesLoader,
                        lazy: TablesList.lazy
                    },
                    {
                        path: "tables/new-table",
                        lazy: ConnectTables.lazy
                    },
                    {
                        path: "tables/edit-table",
                        lazy: ConnectTables.lazy
                    },
                    {
                        path: "graphs/:id?",
                        loader: graphsLoader,
                        lazy: GraphsList.lazy
                    },
                    {
                        path: "graphs/new-graph",
                        lazy: NewGraph.lazy
                    },
                    {
                        path: "graphs/edit-graph",
                        lazy: NewGraph.lazy
                    },
                    {
                        path: "models/",
                        loader: modelsLoader,
                        lazy: ModelsJobsList.lazy
                    },
                    {
                        path: "predictions/",
                        loader: predictionsLoader,
                        lazy: PredictionsJobsList.lazy
                    },
                    {
                        path: "models/training/:id/:queryTab?",
                        loader: modelsLoader,
                        lazy: ModelsJobsList.lazy
                    },
                    {
                        path: "predictions/prediction/:id",
                        loader: predictionsLoader,
                        lazy: PredictionsJobsList.lazy
                    },
                    {
                        path: "models/trainingtable/:id/:queryTab?",
                        loader: modelsLoader,
                        lazy: ModelsJobsList.lazy
                    },
                    {
                        path: "predictions/predictiontable/:id",
                        loader: predictionsLoader,
                        lazy: PredictionsJobsList.lazy
                    },
                    {
                        path: "newjob/training/graph-select/",
                        lazy: NewTrainingGraphSelect.lazy
                    },
                    {
                        path: "newjob/training/",
                        lazy: NewTrainingJob.lazy
                    },
                    {
                        path: "newjob/prediction/",
                        loader: newPredictionLoader,
                        lazy: NewPredictionJob.lazy
                    }
                ]
            },
            {
                path: "*",
                element: <PageNotFound redirectLink={navRoutes.Home} />
            }
        ]
    }
];
