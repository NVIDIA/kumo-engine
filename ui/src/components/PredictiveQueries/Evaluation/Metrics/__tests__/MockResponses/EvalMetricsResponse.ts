// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const eval_metrics_res_reg = {
    "metrics": {
        "global_average": {
            "val_metrics": [
                {
                    "name": "mae",
                    "val": 0.069
                },
                {
                    "name": "mse",
                    "val": 0.119
                },
                {
                    "name": "rmse",
                    "val": 0.345
                }
            ],
            "test_metrics": [
                {
                    "name": "mae",
                    "val": 0.197
                },
                {
                    "name": "mse",
                    "val": 0.279
                },
                {
                    "name": "rmse",
                    "val": 0.529
                }
            ]
        },
        "global_median": {
            "val_metrics": [
                {
                    "name": "mae",
                    "val": 0.06
                },
                {
                    "name": "mse",
                    "val": 0.12
                },
                {
                    "name": "rmse",
                    "val": 0.346
                }
            ],
            "test_metrics": [
                {
                    "name": "mae",
                    "val": 0.17
                },
                {
                    "name": "mse",
                    "val": 0.29
                },
                {
                    "name": "rmse",
                    "val": 0.539
                }
            ]
        },
        "global_mode": {
            "val_metrics": [
                {
                    "name": "mae",
                    "val": 0.06
                },
                {
                    "name": "mse",
                    "val": 0.12
                },
                {
                    "name": "rmse",
                    "val": 0.346
                }
            ],
            "test_metrics": [
                {
                    "name": "mae",
                    "val": 0.17
                },
                {
                    "name": "mse",
                    "val": 0.29
                },
                {
                    "name": "rmse",
                    "val": 0.539
                }
            ]
        },
        "per_entity_average": {
            "val_metrics": [
                {
                    "name": "mae",
                    "val": 0.06
                },
                {
                    "name": "mse",
                    "val": 0.08
                },
                {
                    "name": "rmse",
                    "val": 0.283
                }
            ],
            "test_metrics": [
                {
                    "name": "mae",
                    "val": 0.177
                },
                {
                    "name": "mse",
                    "val": 0.292
                },
                {
                    "name": "rmse",
                    "val": 0.541
                }
            ]
        },
        "per_entity_last_1_label": {
            "val_metrics": [
                {
                    "name": "mae",
                    "val": 0.05
                },
                {
                    "name": "mse",
                    "val": 0.09
                },
                {
                    "name": "rmse",
                    "val": 0.3
                }
            ],
            "test_metrics": [
                {
                    "name": "mae",
                    "val": 0.21
                },
                {
                    "name": "mse",
                    "val": 0.37
                },
                {
                    "name": "rmse",
                    "val": 0.608
                }
            ]
        }
    }
};

export const eval_metrics_res_classification = {
    "metrics": {
        "global_frequency": {
            "val_metrics": [
                {
                    "name": "ap_macro",
                    "val": 0.122
                },
                {
                    "name": "ap_micro",
                    "val": 0.121
                },
                {
                    "name": "ap_per_label_2",
                    "val": 0.155
                },
                {
                    "name": "ap_per_label_1",
                    "val": 0.09
                },
                {
                    "name": "auprc_macro",
                    "val": 0.118
                },
                {
                    "name": "auprc_micro",
                    "val": 0.136
                },
                {
                    "name": "auprc_per_label_2",
                    "val": 0.191
                },
                {
                    "name": "auprc_per_label_1",
                    "val": 0.045
                },
                {
                    "name": "auroc_macro",
                    "val": 0.462
                },
                {
                    "name": "auroc_micro",
                    "val": 0.471
                },
                {
                    "name": "auroc_per_label_2",
                    "val": 0.473
                },
                {
                    "name": "auroc_per_label_1",
                    "val": 0.451
                }
            ],
            "test_metrics": [
                {
                    "name": "ap_macro",
                    "val": 0.122
                },
                {
                    "name": "ap_micro",
                    "val": 0.117
                },
                {
                    "name": "ap_per_label_2",
                    "val": 0.134
                },
                {
                    "name": "ap_per_label_1",
                    "val": 0.11
                },
                {
                    "name": "auprc_macro",
                    "val": 0.162
                },
                {
                    "name": "auprc_micro",
                    "val": 0.14
                },
                {
                    "name": "auprc_per_label_2",
                    "val": 0.128
                },
                {
                    "name": "auprc_per_label_1",
                    "val": 0.195
                },
                {
                    "name": "auroc_macro",
                    "val": 0.482
                },
                {
                    "name": "auroc_micro",
                    "val": 0.482
                },
                {
                    "name": "auroc_per_label_2",
                    "val": 0.437
                },
                {
                    "name": "auroc_per_label_1",
                    "val": 0.528
                }
            ]
        },
        "per_entity_frequency": {
            "val_metrics": [
                {
                    "name": "ap_macro",
                    "val": 0.49
                },
                {
                    "name": "ap_micro",
                    "val": 0.502
                },
                {
                    "name": "ap_per_label_2",
                    "val": 0.547
                },
                {
                    "name": "ap_per_label_1",
                    "val": 0.433
                },
                {
                    "name": "auprc_macro",
                    "val": 0.561
                },
                {
                    "name": "auprc_micro",
                    "val": 0.577
                },
                {
                    "name": "auprc_per_label_2",
                    "val": 0.617
                },
                {
                    "name": "auprc_per_label_1",
                    "val": 0.504
                },
                {
                    "name": "auroc_macro",
                    "val": 0.781
                },
                {
                    "name": "auroc_micro",
                    "val": 0.782
                },
                {
                    "name": "auroc_per_label_2",
                    "val": 0.773
                },
                {
                    "name": "auroc_per_label_1",
                    "val": 0.789
                }
            ],
            "test_metrics": [
                {
                    "name": "ap_macro",
                    "val": 0.442
                },
                {
                    "name": "ap_micro",
                    "val": 0.474
                },
                {
                    "name": "ap_per_label_2",
                    "val": 0.568
                },
                {
                    "name": "ap_per_label_1",
                    "val": 0.316
                },
                {
                    "name": "auprc_macro",
                    "val": 0.489
                },
                {
                    "name": "auprc_micro",
                    "val": 0.523
                },
                {
                    "name": "auprc_per_label_2",
                    "val": 0.611
                },
                {
                    "name": "auprc_per_label_1",
                    "val": 0.367
                },
                {
                    "name": "auroc_macro",
                    "val": 0.811
                },
                {
                    "name": "auroc_micro",
                    "val": 0.813
                },
                {
                    "name": "auroc_per_label_2",
                    "val": 0.821
                },
                {
                    "name": "auroc_per_label_1",
                    "val": 0.8
                }
            ]
        },
        "per_entity_last_1_label": {
            "val_metrics": [
                {
                    "name": "ap_macro",
                    "val": 0.292
                },
                {
                    "name": "ap_micro",
                    "val": 0.287
                },
                {
                    "name": "ap_per_label_2",
                    "val": 0.345
                },
                {
                    "name": "ap_per_label_1",
                    "val": 0.238
                },
                {
                    "name": "auprc_macro",
                    "val": 0.516
                },
                {
                    "name": "auprc_micro",
                    "val": 0.503
                },
                {
                    "name": "auprc_per_label_2",
                    "val": 0.555
                },
                {
                    "name": "auprc_per_label_1",
                    "val": 0.476
                },
                {
                    "name": "auroc_macro",
                    "val": 0.709
                },
                {
                    "name": "auroc_micro",
                    "val": 0.7
                },
                {
                    "name": "auroc_per_label_2",
                    "val": 0.689
                },
                {
                    "name": "auroc_per_label_1",
                    "val": 0.728
                }
            ],
            "test_metrics": [
                {
                    "name": "ap_macro",
                    "val": 0.263
                },
                {
                    "name": "ap_micro",
                    "val": 0.267
                },
                {
                    "name": "ap_per_label_2",
                    "val": 0.289
                },
                {
                    "name": "ap_per_label_1",
                    "val": 0.238
                },
                {
                    "name": "auprc_macro",
                    "val": 0.478
                },
                {
                    "name": "auprc_micro",
                    "val": 0.482
                },
                {
                    "name": "auprc_per_label_2",
                    "val": 0.504
                },
                {
                    "name": "auprc_per_label_1",
                    "val": 0.452
                },
                {
                    "name": "auroc_macro",
                    "val": 0.685
                },
                {
                    "name": "auroc_micro",
                    "val": 0.689
                },
                {
                    "name": "auroc_per_label_2",
                    "val": 0.698
                },
                {
                    "name": "auroc_per_label_1",
                    "val": 0.672
                }
            ]
        }
    }
};

export const eval_metrics_res_link_pred = {
    "metrics": {
        "global_frequency": {
            "val_metrics": [
                { "name": "f1@10", "val": 0.091 },
                { "name": "f1@100", "val": 0.02 },
                { "name": "map@1", "val": 0.0 },
                { "name": "recall@1", "val": 0.0 },
                { "name": "recall@100", "val": 1.0 },
                { "name": "map@100", "val": 0.064 },
                { "name": "precision@1", "val": 0.0 },
                { "name": "precision@10", "val": 0.05 },
                { "name": "precision@100", "val": 0.01 },
                { "name": "f1@1", "val": 0.0 },
                { "name": "map@10", "val": 0.05 },
                { "name": "recall@10", "val": 0.5 }
            ],
            "test_metrics": [
                { "name": "f1@10", "val": 0.0 },
                { "name": "f1@100", "val": 0.0 },
                { "name": "map@1", "val": 0.0 },
                { "name": "recall@1", "val": 0.0 },
                { "name": "recall@100", "val": 0.0 },
                { "name": "map@100", "val": 0.0 },
                { "name": "precision@1", "val": 0.0 },
                { "name": "precision@10", "val": 0.0 },
                { "name": "precision@100", "val": 0.0 },
                { "name": "f1@1", "val": 0.0 },
                { "name": "map@10", "val": 0.0 },
                { "name": "recall@10", "val": 0.0 }
            ]
        },
        "per_entity_frequency": {
            "val_metrics": [
                { "name": "f1@10", "val": 0.167 },
                { "name": "f1@100", "val": 0.02 },
                { "name": "map@1", "val": 0.143 },
                { "name": "recall@1", "val": 0.071 },
                { "name": "recall@100", "val": 0.5 },
                { "name": "map@100", "val": 0.204 },
                { "name": "precision@1", "val": 0.143 },
                { "name": "precision@10", "val": 0.1 },
                { "name": "precision@100", "val": 0.01 },
                { "name": "f1@1", "val": 0.095 },
                { "name": "map@10", "val": 0.204 },
                { "name": "recall@10", "val": 0.5 }
            ],
            "test_metrics": [
                { "name": "f1@10", "val": 0.182 },
                { "name": "f1@100", "val": 0.02 },
                { "name": "map@1", "val": 0.188 },
                { "name": "recall@1", "val": 0.188 },
                { "name": "recall@100", "val": 1.0 },
                { "name": "map@100", "val": 0.449 },
                { "name": "precision@1", "val": 0.188 },
                { "name": "precision@10", "val": 0.1 },
                { "name": "precision@100", "val": 0.01 },
                { "name": "f1@1", "val": 0.188 },
                { "name": "map@10", "val": 0.449 },
                { "name": "recall@10", "val": 1.0 }
            ]
        }
    }
};
