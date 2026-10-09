// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const repoRoot = resolve(import.meta.dirname, "../../..");
const helperPath = resolve(
    repoRoot,
    ".agents/skills/build-ui/scripts/ensure-screenshot-plan.py"
);

const runHelper = ({ changedFiles, existingPlan, screenshotOverride }) => {
    const dir = mkdtempSync(resolve(tmpdir(), "build-ui-plan-"));
    const planPath = resolve(dir, "screenshot-plan.json");
    const changedFilesPath = resolve(dir, "changed-files.txt");
    const reportPath = resolve(dir, "screenshot-plan-resolution.json");
    const overridePath = resolve(dir, "screenshot-overrides.json");

    writeFileSync(changedFilesPath, `${changedFiles.join("\n")}\n`);
    if (existingPlan) {
        writeFileSync(planPath, `${JSON.stringify(existingPlan, null, 2)}\n`);
    }
    if (screenshotOverride) {
        writeFileSync(
            overridePath,
            `${JSON.stringify(screenshotOverride, null, 2)}\n`
        );
    }

    const args = [helperPath, planPath, changedFilesPath, reportPath];
    if (screenshotOverride) {
        args.push(overridePath);
    }

    const output = execFileSync("python3", args).toString();

    return {
        output,
        plan: JSON.parse(readFileSync(planPath, "utf8")),
        report: JSON.parse(readFileSync(reportPath, "utf8"))
    };
};

describe("ensure-screenshot-plan.py", () => {
    it.each([
        [
            "new_prediction",
            "ui/src/pages/NewJob/Prediction/Prediction.tsx",
            "new-prediction-config-layout"
        ],
        [
            "new_connector",
            "ui/src/pages/Connectors/NewConnector/NewConnector.tsx",
            "new-connector-page"
        ],
        [
            "new_table",
            "ui/src/components/Tables/Connect/ConnectTables.tsx",
            "new-table-page"
        ],
        [
            "new_graph",
            "ui/src/pages/Graphs/NewGraph/NewGraph.tsx",
            "new-graph-page"
        ],
        [
            "new_graph_select_tables",
            "ui/src/components/Graphs/SelectTables/GraphSelectTables.tsx",
            "new-graph-select-tables-panel"
        ],
        [
            "entity_lists",
            "ui/src/components/ListOverview/ListOverview.tsx",
            "shared-list-overview-connectors-page"
        ],
        [
            "models_list",
            "ui/src/pages/Jobs/__tests__/ModelList.test.tsx",
            "models-list-page"
        ],
        [
            "jobs_lists",
            "ui/src/pages/Jobs/JobsList.tsx",
            "shared-jobs-list-models-page"
        ],
        [
            "model_details",
            "ui/src/components/Jobs/Training/TrainingJobOverview.tsx",
            "model-detail-page"
        ],
        [
            "predictions_list",
            "ui/src/pages/Jobs/__tests__/PredictionList.test.tsx",
            "predictions-list-page"
        ],
        [
            "training_table_details",
            "ui/src/components/Jobs/TrainingTable/TrainingTableGeneration.tsx",
            "training-table-detail-page"
        ],
        [
            "prediction_details",
            "ui/src/components/Jobs/Prediction/PredictionJobOverview.tsx",
            "prediction-detail-page"
        ],
        [
            "prediction_table_details",
            "ui/src/components/Jobs/PredictionTable/PredictionTableGeneration.tsx",
            "prediction-table-detail-page"
        ],
        [
            "prediction_outputs",
            "ui/src/components/BatchPredictions/PredictionOutputsOverview/PredictionOutputsOverview.tsx",
            "prediction-outputs-panel"
        ],
        [
            "project_settings",
            "ui/src/components/Project/ProjectSettings.tsx",
            "project-settings-page"
        ],
        [
            "project_list",
            "ui/src/components/Project/ProjectList.tsx",
            "project-switcher-dialog"
        ],
        [
            "new_project",
            "ui/src/components/Project/NewProject.tsx",
            "new-project-dialog"
        ],
        [
            "project_forms",
            "ui/src/components/Project/NameInput/ProjectNameInput.tsx",
            "shared-project-form-settings-page"
        ]
    ])(
        "selects the %s recipe for changed file %s",
        (recipeKey, changedFile, scenarioName) => {
            const { output, plan, report } = runHelper({
                changedFiles: [changedFile]
            });

            expect(output).toContain(recipeKey);
            expect(report.selectedRecipe).toBe(recipeKey);
            expect(report.action).toBe("wrote");
            expect(report.verification.status).toBe("passed");
            expect(plan.scenarios[0].name).toBe(scenarioName);
            expect(plan._recipeKey).toBe(recipeKey);
            expect(plan._visualCoverage.length).toBeGreaterThan(0);
        }
    );

    it("rewrites weak New Training plans to the catalog recipe", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/NewJob/Training/Settings/TrainingSetting.tsx"
            ],
            existingPlan: {
                scenarios: [
                    {
                        name: "weak-tabs-plan",
                        route: "/newjob/training/",
                        readySelector: '#config-tabs [data-slot="tabs-list"]',
                        assertions: [
                            {
                                type: "text",
                                selector:
                                    '#config-tabs [data-slot="tabs-list"]',
                                matches: "Model planner"
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("rewrote-recipe");
        expect(report.selectedRecipe).toBe("new_training");
        expect(report.action).toBe("rewrote-recipe");
        expect(report.verification.status).toBe("passed");
        expect(report.matchedRecipes[0].visualCoverage).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: "new_training_task_type_visible"
                })
            ])
        );
        expect(plan._recipeKey).toBe("new_training");
        expect(plan.scenarios[0].name).toBe(
            "new-training-job-task-type-and-model-planner"
        );
        expect(plan.scenarios[0].assertions).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    selector: '[data-testid="infer-task-type"]'
                }),
                expect.objectContaining({
                    selector: "text=Model planner",
                    equals: "Model planner"
                })
            ])
        );
    });

    it("selects the graph table-view recipe for graph information panel changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/Graphs/InformationPanel/TableView.tsx"
            ]
        });

        expect(output).toContain("new_training_graph_table_view");
        expect(report.selectedRecipe).toBe("new_training_graph_table_view");
        expect(report.action).toBe("wrote");
        expect(report.verification.status).toBe("passed");
        expect(plan._recipeKey).toBe("new_training_graph_table_view");
        expect(plan._visualCoverage).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: "graph_table_view_opened"
                })
            ])
        );
        expect(plan.scenarios[0].name).toBe(
            "new-training-job-table-view-from-graph-preview"
        );
        expect(plan.scenarios[0].setup).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    action: "click",
                    selector: '[data-testid="transaction_show_icon"]'
                })
            ])
        );
    });

    it("uses route-map ownership when catalog patterns are not specific enough", () => {
        const { output, plan, report } = runHelper({
            changedFiles: ["ui/src/components/Graphs/SelectTables/Widget.tsx"]
        });

        expect(output).toContain("new_graph_select_tables");
        expect(report.selectedRecipe).toBe("new_graph_select_tables");
        expect(report.matchedRecipes[0].matchSources).toContain(
            "graph_select_tables_workflow"
        );
        expect(report.matchedRecipes[0].matchedRouteMap).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    key: "graph_select_tables_workflow"
                })
            ])
        );
        expect(plan._recipeKey).toBe("new_graph_select_tables");
        expect(plan.scenarios[0].assertions).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    selector: '[data-testid="graph-select-tables-panel"]'
                })
            ])
        );
    });

    it("captures all page variants for shared entity list changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: ["ui/src/components/ListOverview/ListOverview.tsx"]
        });

        expect(output).toContain("entity_lists");
        expect(report.selectedRecipe).toBe("entity_lists");
        expect(plan._recipeKey).toBe("entity_lists");
        expect(plan.scenarios.map((scenario) => scenario.route)).toEqual([
            "/connectors/",
            "/tables/",
            "/graphs/"
        ]);
        expect(
            plan.scenarios.map((scenario) => scenario.readySelector)
        ).toEqual([
            '[data-screenshot-ready="connector-list-page"]',
            '[data-screenshot-ready="table-list-page"]',
            '[data-screenshot-ready="graph-list-page"]'
        ]);
    });

    it("captures both models and predictions for shared jobs list changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: ["ui/src/pages/Jobs/JobsList.tsx"]
        });

        expect(output).toContain("jobs_lists");
        expect(report.selectedRecipe).toBe("jobs_lists");
        expect(plan._recipeKey).toBe("jobs_lists");
        expect(plan.scenarios.map((scenario) => scenario.route)).toEqual([
            "/models/?page=1&job_type=TRAINING_JOB",
            "/predictions/?page=1&job_type=BATCH_PREDICTION_JOB"
        ]);
        expect(
            plan.scenarios.map((scenario) => scenario.readySelector)
        ).toEqual([
            '[data-testid="models-list-page-heading"]',
            '[data-testid="predictions-list-page-heading"]'
        ]);
        expect(plan.scenarios[0].visualCoverage).toEqual([
            expect.objectContaining({
                id: "jobs_lists_models_visible"
            })
        ]);
        expect(plan.scenarios[1].visualCoverage).toEqual([
            expect.objectContaining({
                id: "jobs_lists_predictions_visible"
            })
        ]);
    });

    it("merges independent matched recipes for broad multi-surface changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/pages/Connectors/NewConnector/NewConnector.tsx",
                "ui/src/components/Project/ProjectSettings.tsx"
            ]
        });

        expect(output).toContain("new_connector,project_settings");
        expect(report.selectedRecipe).toBe("multi");
        expect(report.selectedRecipes).toEqual([
            "new_connector",
            "project_settings"
        ]);
        expect(report.action).toBe("wrote");
        expect(report.verification.status).toBe("passed");
        expect(plan._recipeKey).toBe("multi");
        expect(plan._recipeKeys).toEqual(["new_connector", "project_settings"]);
        expect(plan.scenarios.map((scenario) => scenario.name)).toEqual([
            "new-connector-page",
            "project-settings-page"
        ]);
        expect(plan._visualCoverage).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    id: "new_connector_route_loaded"
                }),
                expect.objectContaining({
                    id: "project_settings_route_loaded"
                })
            ])
        );
    });

    it("preserves compatible agent-authored scenarios as primary evidence in broad multi-surface changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/pages/NewJob/Prediction/Prediction.tsx",
                "ui/src/pages/NewJob/Training/Training.tsx",
                "ui/src/components/Project/ProjectSettings.tsx"
            ],
            existingPlan: {
                scenarios: [
                    {
                        name: "ticket-new-prediction-layout",
                        route: "/newjob/prediction/",
                        readySelector:
                            '[data-screenshot-ready="new-prediction-config-layout"]',
                        setup: [
                            {
                                action: "waitFor",
                                selector:
                                    '[data-testid="new-prediction-training-job-select-list"]'
                            },
                            {
                                action: "click",
                                selector: ':nth-match([id^="trainingjob-"], 1)',
                                covers: ["new_prediction_training_job_selected"]
                            }
                        ],
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-page-heading"]',
                                covers: [
                                    "new_prediction_route_loaded",
                                    "new_prediction_heading_visible"
                                ]
                            },
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-config-layout"]',
                                covers: ["new_prediction_config_layout_visible"]
                            },
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-settings-panel"]',
                                covers: [
                                    "new_prediction_settings_panel_visible"
                                ]
                            },
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-schema-panel"]',
                                covers: ["new_prediction_schema_panel_visible"]
                            }
                        ]
                    },
                    {
                        name: "ticket-new-model-header",
                        route: "/newjob/training/",
                        readySelector:
                            '[data-screenshot-ready="new-training-page"]',
                        assertions: [
                            {
                                type: "visible",
                                selector: '[data-testid="infer-task-type"]',
                                covers: ["new_training_task_type_visible"]
                            },
                            {
                                type: "text",
                                selector: '[data-testid="infer-task-type"]',
                                matches: "^Task type: [a-z]+(?: [a-z]+)*$",
                                covers: ["new_training_task_type_sentence_case"]
                            },
                            {
                                type: "text",
                                selector: '[data-testid="tab-model-planner"]',
                                equals: "Model planner",
                                covers: [
                                    "new_training_model_planner_tab_sentence_case"
                                ]
                            }
                        ]
                    },
                    {
                        name: "project-settings-page",
                        route: "/project/settings/",
                        setup: [],
                        readySelector:
                            '[data-testid="project-settings-page-heading"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="project-settings-page-heading"]',
                                covers: [
                                    "project_settings_route_loaded",
                                    "project_settings_heading_visible"
                                ]
                            },
                            {
                                type: "visible",
                                selector: "#project-name",
                                covers: ["project_settings_name_input_visible"]
                            }
                        ]
                    },
                    {
                        name: "stale-connectors-list-page",
                        route: "/connectors/",
                        readySelector:
                            '[data-screenshot-ready="connector-list-page"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="connector-list-page-heading"]',
                                covers: [
                                    "connectors_list_route_loaded",
                                    "connectors_list_heading_visible"
                                ]
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("merged-primary");
        expect(report.action).toBe("merged-primary");
        expect(report.selectedRecipe).toBe("multi");
        expect(report.verification.status).toBe("passed");
        expect(
            plan.scenarios
                .filter((scenario) => scenario.presentation === "primary")
                .map((scenario) => scenario.name)
        ).toEqual(["ticket-new-prediction-layout", "ticket-new-model-header"]);
        expect(plan.scenarios.slice(0, 2)).toEqual([
            expect.objectContaining({
                name: "ticket-new-prediction-layout",
                presentation: "primary",
                visualCoverage: expect.arrayContaining([
                    expect.objectContaining({
                        id: "new_prediction_config_layout_visible"
                    })
                ])
            }),
            expect.objectContaining({
                name: "ticket-new-model-header",
                presentation: "primary",
                visualCoverage: expect.arrayContaining([
                    expect.objectContaining({
                        id: "new_training_task_type_visible"
                    })
                ])
            })
        ]);
        expect(JSON.stringify(plan)).not.toContain("connectors_list");
        expect(
            plan.scenarios.some(
                (scenario) =>
                    scenario.name === "project-settings-page" &&
                    scenario.presentation === "regression"
            )
        ).toBe(true);
    });

    it("does not preserve weak primary scenarios that miss required surface coverage", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/pages/NewJob/Prediction/Prediction.tsx",
                "ui/src/components/Project/ProjectSettings.tsx"
            ],
            existingPlan: {
                scenarios: [
                    {
                        name: "weak-new-prediction-heading-only",
                        route: "/newjob/prediction/",
                        readySelector:
                            '[data-screenshot-ready="new-prediction-page"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-page-heading"]',
                                covers: [
                                    "new_prediction_route_loaded",
                                    "new_prediction_heading_visible"
                                ]
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("merged-primary");
        expect(report.action).toBe("merged-primary");
        expect(report.verification.status).toBe("passed");
        expect(
            plan.scenarios.some(
                (scenario) =>
                    scenario.name === "weak-new-prediction-heading-only"
            )
        ).toBe(false);
        expect(plan.scenarios[0]).toEqual(
            expect.objectContaining({
                name: "new-prediction-config-layout",
                presentation: "primary",
                readySelector: '[data-testid="new-prediction-config-layout"]',
                setup: expect.arrayContaining([
                    expect.objectContaining({
                        selector: ':nth-match([id^="trainingjob-"], 1)'
                    })
                ])
            })
        );
    });

    it("preserves focused catalog-equivalent plans as primary evidence", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/pages/NewJob/Prediction/Prediction.tsx",
                "ui/src/components/Project/ProjectSettings.tsx"
            ],
            existingPlan: {
                scenarios: [
                    {
                        name: "new-prediction-config-layout",
                        route: "/newjob/prediction/",
                        setup: [
                            {
                                action: "waitFor",
                                selector:
                                    '[data-testid="new-prediction-training-job-select-list"]'
                            },
                            {
                                action: "click",
                                selector: ':nth-match([id^="trainingjob-"], 1)',
                                covers: ["new_prediction_training_job_selected"]
                            }
                        ],
                        readySelector:
                            '[data-screenshot-ready="new-prediction-config-layout"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-page-heading"]',
                                covers: [
                                    "new_prediction_route_loaded",
                                    "new_prediction_heading_visible"
                                ]
                            },
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-config-layout"]',
                                covers: ["new_prediction_config_layout_visible"]
                            },
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-settings-panel"]',
                                covers: [
                                    "new_prediction_settings_panel_visible"
                                ]
                            },
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-prediction-schema-panel"]',
                                covers: ["new_prediction_schema_panel_visible"]
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("merged-primary");
        expect(report.action).toBe("merged-primary");
        expect(report.verification.status).toBe("passed");
        expect(plan.scenarios[0]).toEqual(
            expect.objectContaining({
                name: "new-prediction-config-layout",
                presentation: "primary"
            })
        );
    });

    it("keeps exactly one recipe when surfaces mutually subsume each other", () => {
        const script = `
import importlib.util
import pathlib

module_path = pathlib.Path(${JSON.stringify(helperPath)})
spec = importlib.util.spec_from_file_location("ensure_screenshot_plan", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

surfaces = [
    {
        "key": "alpha",
        "matched_files": ["ui/src/pages/Shared.tsx"],
        "canonical_routes": {"/shared/"},
    },
    {
        "key": "beta",
        "matched_files": ["ui/src/pages/Shared.tsx"],
        "canonical_routes": {"/shared/"},
    },
]

pruned = module.prune_subsumed_surfaces(surfaces)
assert [surface["key"] for surface in pruned] == ["alpha"], pruned
`;

        execFileSync("python3", ["-c", script]);
    });

    it("captures both consumers for shared project form changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/Project/NameInput/ProjectNameInput.tsx"
            ]
        });

        expect(output).toContain("project_forms");
        expect(report.selectedRecipe).toBe("project_forms");
        expect(plan._recipeKey).toBe("project_forms");
        expect(plan.scenarios.map((scenario) => scenario.route)).toEqual([
            "/project/settings/",
            "/connectors/"
        ]);
        expect(plan.scenarios[0].assertions).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    selector: "#project-name"
                })
            ])
        );
        expect(plan.scenarios[1].assertions).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    selector: "#project-name"
                })
            ])
        );
    });

    it("does not let table list changes keep table creation plans", () => {
        const { output, plan, report } = runHelper({
            changedFiles: ["ui/src/pages/Tables/TablesList/TablesList.tsx"],
            existingPlan: {
                scenarios: [
                    {
                        name: "wrong-new-table-page",
                        route: "/tables/new-table",
                        readySelector:
                            '[data-screenshot-ready="new-table-page"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-table-page-heading"]',
                                covers: "new_table_heading_visible"
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("tables_list");
        expect(report.selectedRecipe).toBe("tables_list");
        expect(report.action).toBe("rewrote-mismatch");
        expect(plan._recipeKey).toBe("tables_list");
        expect(plan.scenarios[0].route).toBe("/tables/");
    });

    it("does not let graph overview changes keep graph creation plans", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/Graphs/Overview/GraphOverview.tsx"
            ],
            existingPlan: {
                scenarios: [
                    {
                        name: "wrong-new-graph-page",
                        route: "/graphs/new-graph",
                        readySelector:
                            '[data-screenshot-ready="new-graph-page"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="new-graph-page-heading"]',
                                covers: "new_graph_route_loaded"
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("graphs_list");
        expect(report.selectedRecipe).toBe("graphs_list");
        expect(report.action).toBe("rewrote-mismatch");
        expect(plan._recipeKey).toBe("graphs_list");
        expect(plan.scenarios[0].route).toBe("/graphs/");
    });

    it("rewrites unrelated existing plans for project settings changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: ["ui/src/components/Project/ProjectSettings.tsx"],
            existingPlan: {
                scenarios: [
                    {
                        name: "wrong-table-page",
                        route: "/tables/",
                        readySelector:
                            '[data-screenshot-ready="table-list-page"]',
                        assertions: [
                            {
                                type: "visible",
                                selector:
                                    '[data-testid="table-list-page-heading"]',
                                covers: "tables_list_heading_visible"
                            }
                        ]
                    }
                ]
            }
        });

        expect(output).toContain("rewrote-mismatch");
        expect(report.selectedRecipe).toBe("project_settings");
        expect(report.action).toBe("rewrote-mismatch");
        expect(report.verification.status).toBe("passed");
        expect(plan._recipeKey).toBe("project_settings");
        expect(plan.scenarios[0].route).toBe("/project/settings/");
    });

    it("does not use list screenshots for job detail component changes", () => {
        const { output, plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/Jobs/Training/TrainingJobOverview.tsx"
            ]
        });

        expect(output).toContain("model_details");
        expect(report.selectedRecipe).toBe("model_details");
        expect(plan.scenarios[0]).toEqual(
            expect.objectContaining({
                name: "model-detail-page",
                readySelector: '[data-testid="model-detail-page"]'
            })
        );
        expect(plan.scenarios[0].setup).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    action: "click",
                    selector: ':nth-match([id^="trainingjob-"], 1)'
                })
            ])
        );
    });

    it("uses table detail screenshots for table generation component changes", () => {
        const { plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/Jobs/TrainingTable/TrainingTableGenOverview.tsx",
                "ui/src/components/Jobs/PredictionTable/PredictionTableGenOverview.tsx"
            ]
        });

        expect(report.selectedRecipe).toBe("multi");
        expect(report.selectedRecipes).toEqual([
            "prediction_table_details",
            "training_table_details"
        ]);
        expect(plan.scenarios.map((scenario) => scenario.name)).toEqual([
            "prediction-table-detail-page",
            "training-table-detail-page"
        ]);
        expect(plan.scenarios[0].route).toContain(
            "/predictions/predictiontable/"
        );
        expect(plan.scenarios[1].route).toContain("/models/trainingtable/");
    });

    it("uses the prediction output panel recipe instead of the predictions list for output changes", () => {
        const { plan, report } = runHelper({
            changedFiles: [
                "ui/src/components/BatchPredictions/PredictionOutputsOverview/PredictionOutputsOverview.tsx"
            ]
        });

        expect(report.selectedRecipe).toBe("prediction_outputs");
        expect(plan.scenarios[0]).toEqual(
            expect.objectContaining({
                name: "prediction-outputs-panel",
                readySelector: '[data-testid="prediction-outputs-overview"]'
            })
        );
        expect(plan.scenarios[0].assertions).toEqual(
            expect.arrayContaining([
                expect.objectContaining({
                    selector: '[data-testid="prediction-outputs-overview"]'
                })
            ])
        );
    });

    it("applies explicit screenshot override flags to select a catalog recipe", () => {
        const { output, plan, report } = runHelper({
            changedFiles: ["ui/src/components/Uncataloged/Widget.tsx"],
            screenshotOverride: {
                route: "/newjob/training/",
                coverageIds: ["new_training_task_type_visible"],
                fixtures: ["seeded_query_editor"]
            }
        });

        expect(output).toContain("wrote-override");
        expect(report.selectedRecipe).toBe("new_training");
        expect(report.selectedRecipes).toEqual(["new_training"]);
        expect(report.action).toBe("wrote-override");
        expect(report.screenshotOverride).toEqual(
            expect.objectContaining({
                route: "/newjob/training/"
            })
        );
        expect(plan._recipeKey).toBe("new_training");
        expect(plan._screenshotOverride).toEqual({
            route: "/newjob/training/",
            coverageIds: ["new_training_task_type_visible"],
            fixtures: ["seeded_query_editor"]
        });
        expect(plan.scenarios[0].fixtures).toContain("seeded_query_editor");
    });

    it("does not invent a plan for uncataloged surfaces", () => {
        const dir = mkdtempSync(resolve(tmpdir(), "build-ui-plan-"));
        const planPath = resolve(dir, "screenshot-plan.json");
        const changedFilesPath = resolve(dir, "changed-files.txt");
        const reportPath = resolve(dir, "screenshot-plan-resolution.json");

        writeFileSync(
            changedFilesPath,
            "ui/src/components/Uncataloged/Widget.tsx\n"
        );

        const output = execFileSync("python3", [
            helperPath,
            planPath,
            changedFilesPath,
            reportPath
        ]).toString();
        const report = JSON.parse(readFileSync(reportPath, "utf8"));

        expect(output).toContain("skipped");
        expect(report.action).toBe("skipped");
        expect(report.selectedRecipes).toEqual([]);
        expect(report.verification.status).toBe("skipped");
        expect(report.matchedRecipes).toEqual([]);
    });
});
