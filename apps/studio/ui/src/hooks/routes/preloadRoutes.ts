// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from "react";

import {
    Admin,
    ConnectorsList,
    ConnectTables,
    GraphsList,
    JobsList,
    NewConnector,
    NewGraph,
    NewPredictionJob,
    NewTrainingJob,
    TablesList
} from "../../routes/Routes";

const scheduleIdle =
    typeof requestIdleCallback === "function"
        ? requestIdleCallback
        : (...[cb]: Parameters<typeof requestIdleCallback>) =>
              window.setTimeout(cb, 1);

const preloadTier1Routes = () => {
    return Promise.all([
        ConnectorsList.preload(),
        TablesList.preload(),
        GraphsList.preload(),
        JobsList.preload()
    ]);
};

const preloadTier2Routes = () => {
    return Promise.all([
        NewConnector.preload(),
        ConnectTables.preload(),
        NewGraph.preload(),
        NewTrainingJob.preload(),
        NewPredictionJob.preload()
    ]);
};

const preloadTier3Routes = () => {
    return Promise.all([Admin.preload()]);
};

export const usePreloadRoutes = () => {
    const hasPreloaded = useRef(false);

    useEffect(() => {
        if (hasPreloaded.current) return;
        hasPreloaded.current = true;

        scheduleIdle(
            () => {
                preloadTier1Routes();

                setTimeout(() => {
                    preloadTier2Routes();

                    setTimeout(() => {
                        preloadTier3Routes();
                    }, 1000);
                }, 500);
            },
            { timeout: 2000 }
        );
    }, []);
};
