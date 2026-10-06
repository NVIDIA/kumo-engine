// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen } from "@testing-library/react";

import ModelSettings, { ModelSettingsProps } from "../Settings/ModelSettings";

const changeRunMode = vi.fn();
const defaultProps: ModelSettingsProps = {
    createJobArgs: {
        name: "",
        graph_id: "",
        query_yaml: "",
        query_dialect: "",
        model_planner_yaml: "",
        description: "",
        run_mode: null,
        graph_name: null
    },
    changeRunMode
};

describe("ModelSettings", () => {
    const renderModelSettings = async (props: ModelSettingsProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ModelSettings {...props} />
                </MemoryRouter>
            );
        });
    };

    it("does not offer unsupported baseline models", async () => {
        await renderModelSettings(defaultProps);

        expect(
            screen.queryByText("Baseline Comparison")
        ).not.toBeInTheDocument();
        expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    });
});
