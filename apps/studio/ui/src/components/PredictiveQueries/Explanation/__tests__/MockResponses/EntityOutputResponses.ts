// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stype as KumoapiTypingStype } from "gen/v1/types";

import { RegressionEntityOutput } from "types/explorer";
import { TaskType } from "types/predictiveQueries";

export const regressionEntityOutput = {
    "task_type": TaskType.REGRESSION,
    "prediction": 137.74754,
    "target": 148.94376
};

export const nullRegressionEntityOutput = {
    task_type: TaskType.REGRESSION,
    prediction: null,
    target: null
} as unknown as RegressionEntityOutput;

export const binaryEntityOutput = {
    "task_type": TaskType.BINARY_CLASSIFICATION,
    "prediction": 0.00004,
    "target": 0
};

export const multilabelEntityOutput = {
    "task_type": "multilabel_classification",
    "prediction": {
        "1": 0.03229,
        "2": 0.19225
    },
    "target": {
        "1": 0,
        "2": 0
    }
};

export const multiclassEntityOutput = {
    "task_type": TaskType.MULTICLASS_CLASSIFICATION,
    "prediction": {
        "SIPO": 0.9905222058296204,
        "UVER": 0.007395103108137846,
        "POJISTNE": 0.0011245975038036704,
        "LEASING": 0.0009581056656315923
    },
    "target": {
        "SIPO": 1,
        "UVER": 0,
        "POJISTNE": 0,
        "LEASING": 0
    }
};

export const staticLpEntityOutput = [
    {
        "id": "bb14799cc0b3f273b92ad49fa646764327adb6982d9d9384d339278132b9d95e",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 34,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "beff9532810b991926eff38dcd86c55dade0c4896ba9cc7fa4b2500e79fe03c8",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "2eae3c43acf8c43d25dd97991e35cd84fde7b5065dd8eee961f9f001820af981",
        "cols": [
            {
                "name": "FN",
                "value": 1,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": 1,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 54,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "Regularly",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "59f9548a4962d2509e6df75236786b38d9b8c7896bc926afedb0049a539dfde7",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "75a8ecf178e0c0298f9d8b65e312115a21860d28dfeb27f88c0469114708b6bb",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 52,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "3fdc869cab236ca2725d0eee08c53dbc665e6b4dd279d9faa29af73707557dde",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "9dff69cea43e5172e3820dee629e3546c8d233b5be2671e643c63b869b8684b1",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 26,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "72f3fb8aa672573df3888bd24698f6b861278a05548d3afc269e2f72b0e546ed",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "2bed6fe69ca034e8e4d551ac40882bf17657efb6076dc4109a1ce43b0cbf8b3b",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 26,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "fdf83e5ffb42c9f652575a1fcc1411b669df80769d8939e4791ac97025847225",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "94ab162d374ce5696f70c52572f00ddc231ae9dabf1307744ec5c56678e84de5",
        "cols": [
            {
                "name": "FN",
                "value": 1,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": 1,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 51,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "Regularly",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "065cec628d19a5a89c6578f569f1ff5270abc6076c456d03d2a14b003e1e9da1",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "6e6a58b6d9431f4a65c4886f6ea16941d696271bdfe77f840ea6b5b6205bc581",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 43,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "6e7b4c017831a0425bedc6fc03b6a39d10e875e1299aadbe3395a57bccbbb83d",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "5644572e0155ee92e9066f741602b846580baf5e413b223eb9d284b0c75fbc27",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 24,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "a34cbd1dbb6ebcfeda39ca8c7b9c9a85be17d737ea4225938bbd4e9113564000",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "cbb052df7b4f0432d54dfc9c75fcf39dff9b98f04561d54edbce3cd3157ffa51",
        "cols": [
            {
                "name": "FN",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": undefined,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 27,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "NONE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "8658e2ccb1e8b57115803d58fc46197f25b0c3db3ee3c6b8f9b04d4a2eaefe32",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    },
    {
        "id": "1ad647cad02c69d73e593d3844ddc1a6bcb009f3e93961882bfec1869c1d55b7",
        "cols": [
            {
                "name": "FN",
                "value": 1,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "Active",
                "value": 1,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "age",
                "value": 54,
                "stype": KumoapiTypingStype.NUMERICAL,
                "grad_score": undefined
            },
            {
                "name": "club_member_status",
                "value": "ACTIVE",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "fashion_news_frequency",
                "value": "Regularly",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            },
            {
                "name": "postal_code",
                "value":
                    "61551f368e6f0e6660d03d8b01364d3e9d64437bbf70a99e60ccc260bf505851",
                "stype": KumoapiTypingStype.CATEGORICAL,
                "grad_score": undefined
            }
        ],
        "id_name": "customer_id"
    }
];
